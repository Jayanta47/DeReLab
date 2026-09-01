import json
from typing import Callable, Dict, List, Optional

import networkx as nx

from data_handler.kb_manager import KnowledgeBaseManager
from path_relation_module.graph_resolutor import GraphResolutor

from graph_populator.graph_reader import GraphReader


class NaturalLanguageGenerator:
    """
    Converts populated graph edges into surface text and builds staged
    conversations by re-querying PMPM after each defeasible update.
    """

    def __init__(
        self,
        kb_manager: KnowledgeBaseManager,
        llm_sentence_generator: Optional[Callable[[str, Dict[str, str]], str]] = None,
    ):
        self.reader = GraphReader()
        self.kb = kb_manager
        self.llm_sentence_generator = llm_sentence_generator
        self.resolutor = GraphResolutor()

    def generate_edge_sentences(self, graph: nx.DiGraph, include_defeasible: bool = True) -> Dict[str, str]:
        self.graph = graph
        self.reader.set_graph(graph)

        generated_text = {}

        for u, v, edge_data in self.graph.edges(data=True):
            edge_type = edge_data.get("type")
            is_defeasible = edge_type in {
                "defeasible_has_property",
                "defeasible_lacks_property",
            }
            if is_defeasible and not include_defeasible:
                continue

            edge_id = edge_data.get("id")
            payload = self._build_payload(u, v)
            sentence = self._generate_sentence(edge_type, payload)

            if sentence:
                sentence = sentence[0].upper() + sentence[1:]
                edge_data["nl_sentence"] = sentence
                if edge_id:
                    generated_text[edge_id] = sentence

        return generated_text

    def build_conversation_dataset(self, graph: nx.DiGraph) -> List[Dict[str, object]]:
        """
        Builds multi-turn conversation examples. The backbone text is revealed first.
        Each defeasible edge is then revealed in serial order, and the hypothesis is
        re-evaluated through PMPM after every release.
        """
        self.graph = graph
        self.reader.set_graph(graph)
        self.generate_edge_sentences(graph=graph, include_defeasible=True)

        topology = self.graph.graph.get("topology", "default")
        if topology == "default":
            return self._build_default_conversations()
        elif topology == "linear_inheritance":
            return self._build_inheritance_conversations()
        return self._build_tree_inheritance_conversations()

    def save_conversation_dataset(self, output_path: str) -> Dict[str, object]:
        payload = {
            "topology": self.graph.graph.get("topology", "default"),
            "conversations": self.build_conversation_dataset(),
            "edge_sentences": self._collect_edge_sentences(),
        }
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return payload

    # ---------- Type-3 unknown-exception helpers ----------

    def _collect_exception_groups(self) -> Dict[str, List[tuple]]:
        """
        Detect Type-3 unknown-exception groups: defeasible edges that share
        a target node and contain a mix of has/lacks types. The disjunctive
        sentence "At least one of {subjects} is not <target>" comes from
        such a group.

        Returns: {target_node_id: [(subject, target, edge_data), ...]}
        """
        by_target: Dict[str, List[tuple]] = {}
        for u, v, data in self.graph.edges(data=True):
            if data.get("type") in {"defeasible_has_property", "defeasible_lacks_property"}:
                by_target.setdefault(v, []).append((u, v, data))

        groups: Dict[str, List[tuple]] = {}
        for target, members in by_target.items():
            types = {d.get("type") for _, _, d in members}
            if len(members) >= 2 and len(types) == 2:
                # mixed has + lacks → genuine disjunctive exception
                groups[target] = members
        return groups

    def _build_reveal_schedule(
        self,
        defeasible_edges: List[tuple],
        groups: Dict[str, List[tuple]],
    ) -> List[tuple]:
        """
        Walk defeasible edges in serial order. The first time a member of
        an unfired group is encountered, emit a ('group', target, members)
        event before that edge event. Each group fires exactly once.

        Event shapes:
          ('group', target, members)
          ('edge', u, v, data)

        Example: Before "a is on table"  or "b is not on table", schedule "At least one of a and b not on table"
        """
        schedule: List[tuple] = []
        fired_groups: set = set()
        for u, v, d in defeasible_edges:
            if v in groups and v not in fired_groups:
                schedule.append(("group", v, groups[v]))
                fired_groups.add(v)
            schedule.append(("edge", u, v, d))
        return schedule

    def _generate_disjunctive_sentence(self, members: List[tuple]) -> str:
        """
        Members all share the same target. Emit:
        'At least one of {subjects} is not <object_predicate_tail>.'
        """
        subjects = [self.graph.nodes[u].get("label", u) for u, _, _ in members]
        if len(subjects) == 2:
            joined = f"{subjects[0]} or {subjects[1]}"
        else:
            joined = ", ".join(subjects[:-1]) + f", or {subjects[-1]}"

        # Use any member to fetch the object predicate (target is shared).
        payload = self._build_payload(members[0][0], members[0][1])
        predicate = payload["object_predicate"]

        if predicate.startswith("is "):
            return f"At least one of {joined} is not {predicate[3:]}."
        if predicate.startswith("has "):
            return f"At least one of {joined} does not have {predicate[4:]}."
        if predicate.startswith("can "):
            return f"At least one of {joined} cannot {predicate[4:]}."
        if predicate.startswith("serves as "):
            return f"At least one of {joined} does not serve as {predicate[10:]}."
        if predicate.startswith("appears "):
            return f"At least one of {joined} does not appear {predicate[8:]}."
        if predicate.startswith("produces "):
            return f"At least one of {joined} does not produce {predicate[9:]}."
        if predicate.startswith("tastes or smells "):
            return f"At least one of {joined} does not taste or smell {predicate[17:]}."
        return f"At least one of {joined} does not {predicate}."

    def _build_revealed_graph_for_step(
        self,
        schedule: List[tuple],
        step_idx: int,
        groups: Dict[str, List[tuple]],
    ) -> nx.DiGraph:
        """
        Build the graph state after the first (step_idx + 1) reveal events
        have fired.

        Pass step_idx = -1 for the pristine pre-reveal state (backbone only).

        Semantics:
          - Group events add NO edges to the graph; their effect on the
            yes/no answer is captured only when an individual edge later
            resolves the disjunction.
          - Individual edge events add the named edge.
          - Once a 2-member group has exactly one member fired, the
            complement member's edge is inferred and added (resolution).
        """
        revealed = nx.DiGraph()
        revealed.add_nodes_from(self.graph.nodes(data=True))
        revealed.graph.update(self.graph.graph)

        # Backbone: everything that is not a defeasible update, not a Type-2
        # irrelevant update, and not a hypothesis edge.
        for u, v, data in self.graph.edges(data=True):
            et = data.get("type")
            target_type = self.graph.nodes[v].get("type")
            if et == "hypothesis_edge":
                continue
            if et in {"defeasible_has_property", "defeasible_lacks_property"}:
                continue
            if et == "has_attribute" and target_type == "irrelevant_property":
                continue
            revealed.add_edge(u, v, **data)

        if step_idx < 0:
            return revealed

        # Walk schedule events up to and including step_idx.
        fired_members: Dict[str, set] = {}  # target -> set of subjects whose edge fired
        for event in schedule[: step_idx + 1]:
            if event[0] == "group":
                _, target, _ = event
                fired_members.setdefault(target, set())
                # Group fire adds no graph edges.
            else:
                _, u, v, data = event
                if v in fired_members:
                    fired_members[v].add(u)
                revealed.add_edge(u, v, **data)

        # Resolution: for any 2-member group with exactly one member fired,
        # the unresolved member's edge is implied and gets added.
        for target, fired in fired_members.items():
            members = groups.get(target, [])
            if len(members) == 2 and len(fired) == 1:
                for u, v, data in members:
                    if u not in fired:
                        revealed.add_edge(u, v, **data)
                        break

        return revealed

    # -----------------------------------------------------

    def _build_default_conversations(self) -> List[Dict[str, object]]:
        """
        Delegates to DefaultReasoningQuestionGenerator + DefaultReasoningNLGenerator.

        Pipeline:
          1. Instantiate Type4Handler (submodule) with sampled source names.
          2. Inject it into DefaultReasoningQuestionGenerator.
          3. Generate QuestionRecord objects (pure logic, no NL).
          4. Render into a single merged multi-turn conversation.

        Pass type4_handler=None to DefaultReasoningQuestionGenerator to
        disable Type 4 generation.
        """
        import random as _random
        from question_generation.default_question_generator import (
            DefaultReasoningQuestionGenerator,
            Type4ConversationPlanner,
        )
        from question_generation.default_nl_generator import DefaultReasoningNLGenerator

        lo, hi = self.graph.graph.get("num_sources_range", (2, 2))
        n_sources = lo if lo == hi else _random.randint(lo, hi)
        source_names = self.kb.get_source_names(n_sources)

        cfg = self.graph.graph.get("default_config") or {}

        def _cfg(attr, default):
            if cfg and not isinstance(cfg, dict):
                return getattr(cfg, attr, default)
            return cfg.get(attr, default) if isinstance(cfg, dict) else default

        t4_scenarios    = _cfg("type4_scenarios", None)
        t4_b_first_prob = _cfg("type4_b_first_prob", 0.7)
        t4_enabled      = _cfg("type4_enabled", True)
        t4_fraction     = _cfg("type4_object_fraction", 0.5)
        max_sections    = _cfg("max_hypothesis_sections", None)
        max_updates     = _cfg("max_updates_per_section", None)
        max_turns       = _cfg("max_total_turns", None)
        reveal_prob     = _cfg("reveal_order_random_prob", 0.5)

        t4_planner = Type4ConversationPlanner(
            source_names=source_names,
            scenarios=t4_scenarios,
            b_first_prob=t4_b_first_prob,
        )

        qgen = DefaultReasoningQuestionGenerator(
            type4_planner=t4_planner,
            type4_enabled=t4_enabled,
            type4_object_fraction=t4_fraction,
            max_hypothesis_sections=max_sections,
            max_updates_per_section=max_updates,
        )
        nl_gen = DefaultReasoningNLGenerator(kb_manager=self.kb)

        records, plans, report = qgen.generate_planned(self.graph)
        self._last_default_report = report

        for warning in report.get("warnings", []):
            import logging
            logging.getLogger(__name__).warning("DefaultReasoningQuestionGenerator: %s", warning)

        return nl_gen.build_conversations(
            self.graph, records, type4_plans=plans,
            source_names=source_names, max_total_turns=max_turns,
            reveal_order_random_prob=reveal_prob,
        )

    def get_default_generation_report(self) -> dict:
        """
        Return the generation report from the most recent _build_default_conversations()
        call.  Stores the report as self._last_default_report.
        """
        report = getattr(self, "_last_default_report", {})
        self._last_default_report = report
        return report

    def _build_inheritance_conversations(self) -> List[Dict[str, object]]:
        hypothesis_edges = sorted(
            self.reader.get_edges_by_type("hypothesis_edge"),
            key=lambda item: item[0],
        )
        if not hypothesis_edges:
            return []

        defeasible_edges = sorted(
            [
                (u, v, data)
                for u, v, data in self.graph.edges(data=True)
                if data.get("type") in {"defeasible_has_property", "defeasible_lacks_property"}
            ],
            key=lambda item: item[2].get("serial", 0),
        )

        backbone_text = self._compose_pretext(include_defeasible=False)
        conversations = []

        _entailment_q = (
            "Given all the information so far, does the information support the hypothesis?"
        )

        _undirected = self.graph.to_undirected()

        for subject, hypothesis_target, _ in hypothesis_edges:
            subject_label = self.graph.nodes[subject].get("label", subject)
            hypothesis_statement = self._make_property_statement(subject, hypothesis_target)

            initial_answer = self._normalize_answer(
                self.resolutor.resolve(self.graph, subject, hypothesis_target, serial_cutoff=0)
            )

            # Use the node's stored layer as depth — avoids the 2-hop shortcut
            # root → property_node → subject that nx.shortest_path_length would find
            # on the undirected graph.
            hyp_dist_from_root = self.graph.nodes[subject].get("layer", -1)

            turns = [
                {
                    "role": "user",
                    "content": (
                        f"Pretext: {backbone_text} "
                        f"Hypothesis: {hypothesis_statement} "
                        f"Question: {_entailment_q}"
                    ),
                },
                {
                    "role": "assistant",
                    "content": "",
                    "ground_truth": initial_answer,
                    "task_type": "initial_answer",
                },
            ]

            previous_answer = initial_answer

            for upd_source, upd_target, edge_data in defeasible_edges:
                serial = edge_data.get("serial", 0)
                current_answer = self._normalize_answer(
                    self.resolutor.resolve(self.graph, subject, hypothesis_target, serial_cutoff=serial)
                )
                effect = self._classify_effect(
                    upd_target,
                    hypothesis_target,
                    edge_data,
                    previous_answer=previous_answer,
                    current_answer=current_answer,
                )
                update_sentence = edge_data.get("nl_sentence", "")

                dist_from_root = self.graph.nodes[upd_source].get("layer", 0)
                try:
                    dist_to_subject = nx.shortest_path_length(_undirected, upd_source, subject)
                except nx.NetworkXNoPath:
                    dist_to_subject = -1

                turns.append({
                    "role": "user",
                    "content": (
                        f"Information update: {update_sentence} "
                        f"Question: {_entailment_q}"
                    ),
                    "serial": serial,
                    "distance": (dist_from_root, dist_to_subject),
                })
                turns.append({
                    "role": "assistant",
                    "content": "",
                    "ground_truth": current_answer,
                    "effect": effect,
                    "task_type": "update_answer",
                })

                previous_answer = current_answer

            conversations.append({
                "id": f"{subject}_{hypothesis_target}_conversation",
                "topology": "linear_inheritance",
                "metadata": {
                    "subject": subject_label,
                    "hypothesis_target": self.graph.nodes[hypothesis_target].get("label", hypothesis_target),
                    "hypothesis_statement": hypothesis_statement,
                    "hypothesis_distance_from_root": hyp_dist_from_root,
                },
                "turns": turns,
            })

        return conversations

    def _build_tree_inheritance_conversations(self) -> List[Dict[str, object]]:
        from path_relation_module.effect_classifier import classify_inheritance_effect

        hypothesis_edges = self.reader.get_edges_by_type("hypothesis_edge")
        if not hypothesis_edges:
            return []

        subject, hypothesis_target, _ = hypothesis_edges[0]
        subject_label = self.graph.nodes[subject].get("label", subject)
        hypothesis_statement = self._make_property_statement(subject, hypothesis_target)

        defeasible_edges = sorted(
            [
                (u, v, data)
                for u, v, data in self.graph.edges(data=True)
                if data.get("type") in {"defeasible_has_property", "defeasible_lacks_property"}
            ],
            key=lambda item: item[2].get("serial", 0),
        )

        backbone_text = self._compose_pretext(include_defeasible=False)

        initial_answer = self._normalize_answer(
            self.resolutor.resolve(self.graph, subject, hypothesis_target, serial_cutoff=0)
        )

        _entailment_q = (
            "Given all the information so far, does the information support the hypothesis?"
        )

        # Pre-compute a class-only undirected subgraph for dist_to_subject.
        # Using the full undirected graph allows a 2-hop shortcut:
        #   reveal_node — Target_Property — subject
        # because every reveal node and the hypothesis subject all connect to
        # Target_Property.  Restricting to class nodes forces the path to go
        # through the actual inheritance tree.
        _class_nodes = [
            n for n, d in self.graph.nodes(data=True) if d.get("type") == "class"
        ]
        class_undirected = self.graph.subgraph(_class_nodes).to_undirected()

        # Use the node's stored layer as depth — avoids the 2-hop shortcut
        # root → property_node → subject that nx.shortest_path_length would find
        # on the undirected graph.
        hyp_dist_from_root = self.graph.nodes[subject].get("layer", -1)

        turns = [
            {
                "role": "user",
                "content": (
                    f"Pretext:\n{backbone_text}\n\n"
                    f"Hypothesis: {hypothesis_statement}\n\n"
                    f"Question: {_entailment_q}"
                ),
            },
            {
                "role": "assistant",
                "content": "",
                "ground_truth": initial_answer,
                "task_type": "initial_answer",
            },
        ]

        for upd_source, _, edge_data in defeasible_edges:
            serial = edge_data.get("serial", 0)
            current_answer = self._normalize_answer(
                self.resolutor.resolve(
                    self.graph, subject, hypothesis_target, serial_cutoff=serial
                )
            )
            effect = classify_inheritance_effect(
                graph=self.graph,
                source_node=upd_source,
                edge_type=edge_data.get("type"),
                subject=subject,
                hypothesis_target=hypothesis_target,
            )
            update_sentence = edge_data.get("nl_sentence", "")

            dist_from_root = self.graph.nodes[upd_source].get("layer", 0)
            try:
                dist_to_subject = nx.shortest_path_length(class_undirected, upd_source, subject)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                dist_to_subject = -1

            turns.append({
                "role": "user",
                "content": (
                    f"Information update:\n{update_sentence}\n\n"
                    f"Question: {_entailment_q}"
                ),
                "serial": serial,
                "distance": (dist_from_root, dist_to_subject),
            })
            turns.append({
                "role": "assistant",
                "content": "",
                "ground_truth": current_answer,
                "effect": effect,
                "task_type": "update_answer",
            })

        return [
            {
                "id": f"{subject}_{hypothesis_target}_conversation",
                "topology": "tree_inheritance",
                "metadata": {
                    "subject": subject_label,
                    "hypothesis_target": self.graph.nodes[hypothesis_target].get(
                        "label", hypothesis_target
                    ),
                    "hypothesis_statement": hypothesis_statement,
                    "hypothesis_distance_from_root": hyp_dist_from_root,
                },
                "turns": turns,
            }
        ]

    def _build_payload(self, source_id: str, target_id: str) -> Dict[str, str]:
        '''
        Builds a payload for generating a natural language sentence based on the source and target nodes.
        The payload includes the subject, object, and their respective predicates, which are derived from the node metadata.
        '''
        source_node = self.graph.nodes[source_id]
        target_node = self.graph.nodes[target_id]
        source_meta = source_node.get("metadata", {})
        target_meta = target_node.get("metadata", {})

        object_predicate = target_meta.get(
            "filled_object_template",
            f"has the property '{target_node.get('label', target_id)}'",
        )
        object_predicate_plural = target_meta.get(
            "filled_template",
            f"have the property '{target_node.get('label', target_id)}'",
        )
        subject_predicate = source_meta.get(
            "filled_template",
            f"has the property '{source_node.get('label', source_id)}'",
        )
        subject_predicate_singular = source_meta.get(
            "filled_object_template",
            f"has the property '{source_node.get('label', source_id)}'",
        )

        return {
            "subject": source_node.get("label", source_id),
            "object": target_node.get("label", target_id),
            "subject_predicate": subject_predicate,
            "subject_predicate_singular": subject_predicate_singular,
            "object_predicate": object_predicate,
            "object_predicate_plural": object_predicate_plural,
        }

    def _generate_sentence(self, edge_type: str, payload: Dict[str, str]) -> str:
        if edge_type == "defeasible_lacks_property":
            return self._generate_negative_update(payload)

        if edge_type == "has_property":
            subject = payload["subject"]
            plural_subject = subject if subject.endswith("s") else subject + "s"
            return f"{plural_subject.capitalize()} {payload['object_predicate_plural']}."

        if edge_type == "hypothesis_edge":
            return self._make_property_question_from_predicate(
                payload["subject"], payload["object_predicate"]
            )

        topology = self.graph.graph.get("topology", "inheritance")

        # Default topology: entity is_a P_root should say "X is radiant in color."
        # rather than using the generic "X is a <label>" template.
        if edge_type == "is_a" and topology == "default":
            return f"{payload['subject']} {payload['object_predicate']}."

        return self.kb.format_with_generator(
            self.llm_sentence_generator,
            edge_type,
            payload,
            topology=topology,
        )

    def _make_property_question_from_predicate(self, subject: str, predicate: str) -> str:
        if predicate.startswith("is "):
            return f"Is {subject} {predicate[3:]}?"
        if predicate.startswith("has "):
            return f"Does {subject} have {predicate[4:]}?"
        if predicate.startswith("can "):
            return f"Can {subject} {predicate[4:]}?"
        if predicate.startswith("serves as "):
            return f"Does {subject} serve as {predicate[10:]}?"
        if predicate.startswith("appears "):
            return f"Does {subject} appear {predicate[8:]}?"
        if predicate.startswith("produces "):
            return f"Does {subject} produce {predicate[9:]}?"
        if predicate.startswith("tastes or smells "):
            return f"Does {subject} taste or smell {predicate[17:]}?"
        return f"Does {subject} {predicate}?"

    def _generate_negative_update(self, payload: Dict[str, str]) -> str:
        predicate = payload["object_predicate"]
        subject = payload["subject"]

        if predicate.startswith("is "):
            return f"{subject} is not {predicate[3:]}."
        if predicate.startswith("has "):
            return f"{subject} does not have {predicate[4:]}."
        if predicate.startswith("can "):
            return f"{subject} cannot {predicate[4:]}."
        if predicate.startswith("serves as "):
            return f"{subject} does not serve as {predicate[10:]}."
        if predicate.startswith("appears "):
            return f"{subject} does not appear {predicate[8:]}."
        if predicate.startswith("produces "):
            return f"{subject} does not produce {predicate[9:]}."
        if predicate.startswith("tastes or smells "):
            return f"{subject} does not taste or smell {predicate[17:]}."
        if predicate.startswith("displays "):
            return f"{subject} does not display {predicate[9:]}."
        if predicate.startswith("thrives "):
            return f"{subject} does not thrive {predicate[8:]}."

        return f"{subject} does not {predicate}."

    def _compose_pretext(self, include_defeasible: bool) -> str:
        relevant_types = {"is_a", "implies", "has_attribute", "inheritance", "has_property"}
        if include_defeasible:
            relevant_types.update({"defeasible_has_property", "defeasible_lacks_property"})

        ordered_edges = sorted(
            self.graph.edges(data=True),
            key=lambda item: (
                item[2].get("serial", 0),
                self.graph.nodes[item[0]].get("layer", 0),
                self.graph.nodes[item[1]].get("layer", 0),
            ),
        )
        sentences = []
        for _, v, data in ordered_edges:
            edge_type = data.get("type")
            target_node_type = self.graph.nodes[v].get("type")

            # If we are NOT including defeasible edges (Base Text phase)
            if not include_defeasible:
                # Skip Type-2 irrelevant updates from the base pretext.
                if edge_type == "has_attribute" and target_node_type == "irrelevant_property":
                    continue

            if edge_type in relevant_types and data.get("nl_sentence"):
                sentences.append(data.get("nl_sentence"))

        return " ".join(sentences)

    def _build_base_graph(self) -> nx.DiGraph:
        """
        Creates the initial state graph containing ONLY the backbone pretext
        and hypothesis edges. No defeasible evidence is included.
        """
        base_graph = nx.DiGraph()
        base_graph.add_nodes_from(self.graph.nodes(data=True))
        base_graph.graph.update(self.graph.graph)

        for u, v, data in self.graph.edges(data=True):
            edge_type = data.get("type")
            target_node_type = self.graph.nodes[v].get("type")

            is_standard_update = edge_type in {"defeasible_has_property", "defeasible_lacks_property"}
            is_type_2_update = (edge_type == "has_attribute" and target_node_type == "irrelevant_property")

            if not is_standard_update and not is_type_2_update:
                base_graph.add_edge(u, v, **data)

        return base_graph

    def _build_revealed_graph(self, serial_cutoff: int):
        revealed_graph = nx.DiGraph()
        revealed_graph.add_nodes_from(self.graph.nodes(data=True))
        revealed_graph.graph.update(self.graph.graph)

        for u, v, data in self.graph.edges(data=True):
            edge_type = data.get("type")
            target_node_type = self.graph.nodes[v].get("type")

            if edge_type == "hypothesis_edge":
                continue

            is_standard_update = edge_type in {"defeasible_has_property", "defeasible_lacks_property"}
            is_type_2_update = (edge_type == "has_attribute" and target_node_type == "irrelevant_property")

            if is_standard_update or is_type_2_update:
                if data.get("serial", 0) > serial_cutoff:
                    continue

            revealed_graph.add_edge(u, v, **data)

        return revealed_graph

    def _make_property_statement(self, subject: str, property_node: str) -> str:
        """Declarative form of the hypothesis: 'Subject predicate.'"""
        subject_label = self.graph.nodes[subject].get("label", subject)
        metadata = self.graph.nodes[property_node].get("metadata", {})
        predicate = metadata.get(
            "filled_object_template",
            f"have the property '{self.graph.nodes[property_node].get('label', property_node)}'",
        )
        return f"{subject_label.capitalize()} {predicate}."

    def _make_property_question(self, subject: str, property_node: str) -> str:
        subject_label = self.graph.nodes[subject].get("label", subject)
        metadata = self.graph.nodes[property_node].get("metadata", {})
        predicate = metadata.get(
            "filled_object_template",
            f"have the property '{self.graph.nodes[property_node].get('label', property_node)}'",
        )
        if predicate.startswith("is "):
            return f"Is {subject_label} {predicate[3:]}?"
        if predicate.startswith("has "):
            return f"Does {subject_label} have {predicate[4:]}?"
        if predicate.startswith("can "):
            return f"Can {subject_label} {predicate[4:]}?"
        if predicate.startswith("serves as "):
            return f"Does {subject_label} serve as {predicate[10:]}?"
        if predicate.startswith("appears "):
            return f"Does {subject_label} appear {predicate[8:]}?"
        if predicate.startswith("produces "):
            return f"Does {subject_label} produce {predicate[9:]}?"
        if predicate.startswith("tastes or smells "):
            return f"Does {subject_label} taste or smell {predicate[17:]}?"
        return f"Does {subject_label} {predicate}?"

    def _classify_effect(
        self,
        update_target: str,
        hypothesis_target: str,
        edge_data: Dict[str, object],
        update_source: Optional[str] = None,
        hyp_subject: Optional[str] = None,
        groups: Optional[Dict[str, List[tuple]]] = None,
        previous_answer: Optional[str] = None,
        current_answer: Optional[str] = None,
    ) -> str:
        # Group-aware path: another member of a group has its individual
        # edge fired. The disjunction collapses, and the hypothesis subject
        # gets the OPPOSITE polarity of the firing edge.
        # Example: group {A has T, B lacks T}, hypothesis on A, B's lacks
        # fires => A is cleared (strengthen).
        if (
            groups is not None
            and hyp_subject is not None
            and update_source is not None
            and update_target in groups
        ):
            group_subjects = {gu for gu, _, _ in groups[update_target]}
            if hyp_subject in group_subjects and hyp_subject != update_source:
                if not self._on_chain(update_target, hypothesis_target):
                    return "no effect"
                if edge_data.get("type") == "defeasible_lacks_property":
                    return "strengthen"
                if edge_data.get("type") == "defeasible_has_property":
                    return "weaken"

        # Answer-delta override: trust PMPM when its answer flipped.
        if previous_answer is not None and current_answer is not None:
            if previous_answer != current_answer:
                if previous_answer in {"no", "unknown"} and current_answer == "yes":
                    return "strengthen"
                if previous_answer == "yes" and current_answer in {"no", "unknown"}:
                    return "weaken"

        # Original structural logic.
        target_node_type = self.graph.nodes[update_target].get("type")
        if target_node_type == "irrelevant_property":
            return "no effect"

        if update_target != hypothesis_target:
            if not self._on_chain(update_target, hypothesis_target):
                return "no effect"

        if edge_data.get("type") == "defeasible_has_property":
            return "strengthen"
        if edge_data.get("type") == "defeasible_lacks_property":
            return "weaken"
        return "no effect"

    def _collect_edge_sentences(self) -> Dict[str, str]:
        sentences = {}
        for _, _, data in self.graph.edges(data=True):
            edge_id = data.get("id")
            sentence = data.get("nl_sentence")
            if edge_id and sentence:
                sentences[edge_id] = sentence
        return sentences

    def _normalize_answer(self, answer: str) -> str:
        mapping = {
            "True": "yes",
            "False": "no",
            "Skeptical/Unknown": "unknown",
        }
        return mapping.get(answer, answer.lower())
    
    def _on_chain(self, node: str, hyp_target: str) -> bool:
        """True if `node` reaches `hyp_target` in the algorithm graph."""
        algo_graph = self.resolutor.build_algorithm_graph(self.graph)
        if node not in algo_graph or hyp_target not in algo_graph:
            return False
        if node == hyp_target:
            return True
        return nx.has_path(algo_graph, node, hyp_target)