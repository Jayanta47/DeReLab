"""
DefaultReasoningQuestionGenerator
===================================
Generates QuestionRecord objects (Types 1–3) and Type4Episode objects for the
four default-reasoning question types from a populated default-topology graph.

Module layout
-------------
QuestionRecord            – dataclass output for Types 1–3
Type4Episode              – dataclass output for Type 4 source-priority disputes
_derive_mode_a()          – Mode A label derivation helper
Type1Handler              – on-chain defeasible evidence
Type2Handler              – irrelevant properties
Type3Handler              – quantified / scoped uncertainty
Type4Handler              – source-priority conflict episodes (submodule; inject into orchestrator)
DefaultReasoningQuestionGenerator – orchestrator

Usage
-----
# With type4 (dependency injection):
t4 = Type4Handler(source_names=["Expert A", "Expert B"])
gen = DefaultReasoningQuestionGenerator(type4_handler=t4)

# Without type4:
gen = DefaultReasoningQuestionGenerator()
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# QuestionRecord
# ---------------------------------------------------------------------------

@dataclass
class QuestionRecord:
    """Single question unit emitted by the default-reasoning generator (Types 1–3)."""

    id: str
    question_type: int            # 1, 2, or 3
    subject: str
    hypothesis_target: str
    pi_add_type: str
    pi_add_node: str
    pi_add_serial: float
    baseline_state: str           # "ENTAILED" | "DEFEATED" | "UNDETERMINED"
    post_update_state: str
    mode_a_label: str             # "STRENGTHENING" | "WEAKENING" | "NO_EFFECT"
    mode_b_label: str             # "YES" | "NO" | "UNKNOWN"
    difficulty: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Type4Episode
# ---------------------------------------------------------------------------

@dataclass
class Type4Episode:
    """
    A source-priority belief-revision episode.

    scenario = "b_first"
        B (lower-priority) speaks first, A (higher-priority) speaks second.

    scenario = "a_first"
        A (higher-priority) speaks first, B (lower-priority) speaks second.
        Because A outranks B, B's claim has no effect on ground-truth.

    scenario_type = "contradiction"
        B denies a chain property; A reinstates it. B-world closes when A fires.

    scenario_type = "bypass"
        B denies rule Pi→Pi+1. A introduces a skip rule Pj→Pk (j<i, k>i+1)
        that routes around B's defeat. B's denial persists after A fires;
        A provides an alternative path rather than cancelling B's claim.

    b_claim_level = "rule"    B's NL claim is about the rule Pi→Pi+1.
    b_claim_level = "object"  B's NL claim is about the subject directly.
    """
    subject: str
    hypothesis_target: str
    p_i: str           # chain node whose incoming edge is denied (chain[i+1] for edge i)
    b_serial: float    # serial of B's claim
    a_serial: float    # serial of A's claim
    b_type: str        # "defeasible_lacks_property" (B always denies)
    a_type: str        # "defeasible_has_property"
    source_a: str      # higher-priority source name
    source_b: str      # lower-priority source name
    scenario: str = "b_first"            # "b_first" | "a_first"
    scenario_type: str = "contradiction" # "contradiction" | "bypass"
    bypass_source: Optional[str] = None  # Pj node id (bypass only)
    bypass_target: Optional[str] = None  # Pk node id (bypass only)
    b_claim_level: str = "rule"          # "rule" | "object"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_STATE_TO_MODE_B: Dict[str, str] = {
    "ENTAILED": "YES",
    "DEFEATED": "NO",
    "UNDETERMINED": "UNKNOWN",
}


def _derive_mode_a(baseline: str, post: str, pi_add_type: str) -> str:
    """
    Derive the Mode-A (belief-update) label.

    Rules (in priority order):
      1. "safe_irr"              → always NO_EFFECT.
      2. "pos"                   → always STRENGTHENING.
      3. "neg" / "exception"     → always WEAKENING.
         (A negative update reduces support regardless of whether the binary
         outcome changes — symmetric to rule 2 for positive updates.)
      4. baseline=ENTAILED, post=DEFEATED or UNDETERMINED → WEAKENING.
      5. baseline != ENTAILED, post=ENTAILED → STRENGTHENING.
      6. baseline=UNDETERMINED, post=DEFEATED → WEAKENING.
      7. All other → NO_EFFECT.
    """
    if pi_add_type == "safe_irr":
        return "NO_EFFECT"

    if pi_add_type == "pos":
        return "STRENGTHENING"

    if pi_add_type in {"neg", "exception"}:
        return "WEAKENING"

    if baseline == "ENTAILED" and post in {"DEFEATED", "UNDETERMINED"}:
        return "WEAKENING"

    if baseline != "ENTAILED" and post == "ENTAILED":
        return "STRENGTHENING"

    if baseline == "UNDETERMINED" and post == "DEFEATED":
        return "WEAKENING"

    return "NO_EFFECT"


def _make_record_id(subject: str, q_type: int, pi_add_type: str, node: str, serial: float) -> str:
    serial_str = f"{serial:.3f}".replace(".", "_")
    return f"{subject}_t{q_type}_{pi_add_type}_{node}_{serial_str}"


# ---------------------------------------------------------------------------
# Type 1 Handler
# ---------------------------------------------------------------------------

class Type1Handler:
    """
    Generates questions from on-chain defeasible evidence edges.

    For each defeasible edge (subject → chain_node, type=pos/neg):
      - pos  → one QuestionRecord with pi_add_type="pos"
      - neg  → ONE QuestionRecord, randomly framed as either:
                  • pi_add_type="neg"       (serial)
                  • pi_add_type="exception" (serial + 0.001)
               Only one per edge — never both — to avoid duplicate question types.

    Edges whose serial appears in type4_serials[subject] are skipped; those
    edges belong to a Type4Episode instead.
    """

    def generate(
        self,
        graph,
        chain: List[str],
        resolver,
        subjects: List[str],
        hypothesis_target: str,
        generation_report: dict,
        type4_serials: Optional[Dict[str, set]] = None,
    ) -> List[QuestionRecord]:
        records: List[QuestionRecord] = []
        chain_set = set(chain)

        for subject in subjects:
            skip = (type4_serials or {}).get(subject, set())

            for _u, v, data in graph.out_edges(subject, data=True):
                etype = data.get("type")
                if etype not in {"defeasible_has_property", "defeasible_lacks_property"}:
                    continue
                if v not in chain_set:
                    continue

                serial = float(data.get("serial", 0))
                if serial in skip:
                    continue  # reserved for Type4

                chain_depth = chain.index(v)

                baseline = resolver.resolve(
                    graph, subject, hypothesis_target, serial_cutoff=serial - 1
                )

                if baseline == "DEFEATED":
                    generation_report["warnings"].append(
                        f"Type1: subject={subject} chain_node={v} serial={serial} "
                        f"— baseline already DEFEATED before this edge is revealed."
                    )

                post = resolver.resolve(
                    graph, subject, hypothesis_target, serial_cutoff=serial
                )

                difficulty = {
                    "chain_depth": chain_depth,
                    "chain_length": len(chain),
                }

                if etype == "defeasible_has_property":
                    mode_a = _derive_mode_a(baseline, post, "pos")
                    mode_b = _STATE_TO_MODE_B[post]
                    records.append(QuestionRecord(
                        id=_make_record_id(subject, 1, "pos", v, serial),
                        question_type=1,
                        subject=subject,
                        hypothesis_target=hypothesis_target,
                        pi_add_type="pos",
                        pi_add_node=v,
                        pi_add_serial=serial,
                        baseline_state=baseline,
                        post_update_state=post,
                        mode_a_label=mode_a,
                        mode_b_label=mode_b,
                        difficulty=difficulty,
                    ))

                else:  # defeasible_lacks_property — pick one framing only
                    use_exception = random.random() < 0.5
                    if use_exception:
                        pi_type = "exception"
                        rec_serial = serial + 0.001
                    else:
                        pi_type = "neg"
                        rec_serial = serial

                    mode_a = _derive_mode_a(baseline, post, pi_type)
                    mode_b = _STATE_TO_MODE_B[post]
                    records.append(QuestionRecord(
                        id=_make_record_id(subject, 1, pi_type, v, rec_serial),
                        question_type=1,
                        subject=subject,
                        hypothesis_target=hypothesis_target,
                        pi_add_type=pi_type,
                        pi_add_node=v,
                        pi_add_serial=rec_serial,
                        baseline_state=baseline,
                        post_update_state=post,
                        mode_a_label=mode_a,
                        mode_b_label=mode_b,
                        difficulty=difficulty,
                    ))

        return records


# ---------------------------------------------------------------------------
# Type 2 Handler
# ---------------------------------------------------------------------------

class Type2Handler:
    """
    Generates questions from irrelevant-property has_attribute edges.

    Tags each irrelevant node as "safe_irr" or "trap_irr" based on semantic
    label overlap with chain nodes (using negation detection heuristics).
    Untagged nodes are skipped with a warning.
    """

    def generate(
        self,
        graph,
        chain: List[str],
        resolver,
        subjects: List[str],
        hypothesis_target: str,
        generation_report: dict,
    ) -> List[QuestionRecord]:
        records: List[QuestionRecord] = []
        subjects_set = set(subjects)

        irr_nodes = [
            n
            for n, d in graph.nodes(data=True)
            if d.get("type") == "irrelevant_property"
        ]

        irr_serial_counter: Dict[str, float] = {}

        for irr_node in irr_nodes:
            for subject, _v, data in graph.in_edges(irr_node, data=True):
                if data.get("type") != "has_attribute":
                    continue
                if subject not in subjects_set:
                    continue

                tag = self._tag_irrelevance(graph, irr_node, chain)
                if tag is None:
                    generation_report["warnings"].append(
                        f"Type2: irr_node={irr_node} subject={subject} — "
                        "cannot determine safe/trap tag; skipping."
                    )
                    continue

                offset_key = subject
                if offset_key not in irr_serial_counter:
                    irr_serial_counter[offset_key] = 0.0
                pi_add_serial = irr_serial_counter[offset_key]
                irr_serial_counter[offset_key] += 0.1

                if tag == "safe_irr":
                    state = resolver.resolve(graph, subject, hypothesis_target)
                    baseline = state
                    post = state
                else:
                    baseline = resolver.resolve(
                        graph, subject, hypothesis_target, serial_cutoff=0
                    )
                    post = "DEFEATED"

                mode_a = _derive_mode_a(baseline, post, tag)
                mode_b = _STATE_TO_MODE_B[post]

                records.append(QuestionRecord(
                    id=_make_record_id(subject, 2, tag, irr_node, pi_add_serial),
                    question_type=2,
                    subject=subject,
                    hypothesis_target=hypothesis_target,
                    pi_add_type=tag,
                    pi_add_node=irr_node,
                    pi_add_serial=pi_add_serial,
                    baseline_state=baseline,
                    post_update_state=post,
                    mode_a_label=mode_a,
                    mode_b_label=mode_b,
                    difficulty={"is_trap": tag == "trap_irr"},
                ))

        return records

    def _tag_irrelevance(
        self, graph, irr_node: str, chain: List[str]
    ) -> Optional[str]:
        irr_label = (
            graph.nodes[irr_node]
            .get("metadata", {})
            .get("selected_value", graph.nodes[irr_node].get("label", ""))
        ).lower().strip()

        chain_labels = []
        for n in chain:
            lbl = (
                graph.nodes[n]
                .get("metadata", {})
                .get("selected_value", graph.nodes[n].get("label", ""))
            ).lower().strip()
            chain_labels.append(lbl)

        for cl in chain_labels:
            if self._labels_conflict(irr_label, cl):
                return "trap_irr"

        return "safe_irr"

    @staticmethod
    def _labels_conflict(irr_label: str, chain_label: str) -> bool:
        neg_prefixes = ("not ", "cannot ", "does not ", "is not ", "are not ")

        for prefix in neg_prefixes:
            if irr_label.startswith(prefix):
                rest = irr_label[len(prefix):]
                if rest == chain_label or chain_label.startswith(rest) or rest.startswith(chain_label):
                    return True

        if chain_label.startswith("can ") and irr_label.startswith("cannot "):
            if chain_label[4:] == irr_label[7:]:
                return True

        if chain_label.startswith("is ") and irr_label.startswith("is not "):
            if chain_label[3:] == irr_label[7:]:
                return True

        for prefix in neg_prefixes:
            if chain_label.startswith(prefix):
                rest = chain_label[len(prefix):]
                if rest == irr_label or irr_label.startswith(rest):
                    return True

        return False


# ---------------------------------------------------------------------------
# Type 3 Handler
# ---------------------------------------------------------------------------

class Type3Handler:
    """
    Generates quantified-uncertainty questions.

    Part A (Type 3-I): Instantiation scope.
    Part B (Type 3-III): Generic scope.
    """

    def generate(
        self,
        graph,
        chain: List[str],
        resolver,
        subjects: List[str],
        hypothesis_target: str,
        generation_report: dict,
    ) -> List[QuestionRecord]:
        records: List[QuestionRecord] = []
        records += self._generate_instantiation(
            graph, chain, resolver, subjects, hypothesis_target, generation_report
        )
        records += self._generate_generic(
            graph, chain, resolver, subjects, hypothesis_target, generation_report
        )
        return records

    def _generate_instantiation(
        self, graph, chain, resolver, subjects, hypothesis_target, generation_report
    ) -> List[QuestionRecord]:
        records: List[QuestionRecord] = []
        subjects_set = set(subjects)

        by_target: Dict[str, List[Tuple[str, str, float]]] = {}
        for subject in subjects:
            for _u, v, data in graph.out_edges(subject, data=True):
                etype = data.get("type")
                if etype not in {"defeasible_has_property", "defeasible_lacks_property"}:
                    continue
                if v not in set(chain):
                    continue
                by_target.setdefault(v, []).append(
                    (subject, etype, float(data.get("serial", 0)))
                )

        for p_i, members in by_target.items():
            if p_i not in set(chain) or p_i == chain[0]:
                continue
            types_present = {m[1] for m in members}
            if len(types_present) < 2 or len(members) < 2:
                continue

            members_pos = {m[0] for m in members if m[1] == "defeasible_has_property"}
            members_neg = {m[0] for m in members if m[1] == "defeasible_lacks_property"}
            scope_set = {m[0] for m in members}
            group_serial = min(m[2] for m in members) - 0.5

            for subject in subjects:
                baseline = resolver.resolve(
                    graph, subject, hypothesis_target,
                    serial_cutoff=max(0, int(group_serial) - 1),
                )
                if subject in scope_set:
                    post = "UNDETERMINED"
                else:
                    post = resolver.resolve(
                        graph, subject, hypothesis_target,
                        serial_cutoff=int(group_serial),
                    )

                mode_a = _derive_mode_a(baseline, post, "quantified_inst")
                mode_b = _STATE_TO_MODE_B[post]

                records.append(QuestionRecord(
                    id=_make_record_id(subject, 3, "quantified_inst", p_i, group_serial),
                    question_type=3,
                    subject=subject,
                    hypothesis_target=hypothesis_target,
                    pi_add_type="quantified_inst",
                    pi_add_node=p_i,
                    pi_add_serial=group_serial,
                    baseline_state=baseline,
                    post_update_state=post,
                    mode_a_label=mode_a,
                    mode_b_label=mode_b,
                    difficulty={"scope_size": len(scope_set)},
                    extra={
                        "scope_set": sorted(scope_set),
                        "group_target": p_i,
                        "kind": "instantiation",
                        "members_pos": sorted(members_pos),
                        "members_neg": sorted(members_neg),
                    },
                ))

        return records

    def _generate_generic(
        self, graph, chain, resolver, subjects, hypothesis_target, generation_report
    ) -> List[QuestionRecord]:
        records: List[QuestionRecord] = []

        if not subjects:
            generation_report["warnings"].append(
                "Type3-III: no subjects in graph; skipping generic scope records."
            )
            return records

        p_root = chain[0]
        max_serial = max(
            (data.get("serial", 0)
             for _, _, data in graph.edges(data=True)
             if data.get("type") in {"defeasible_has_property", "defeasible_lacks_property"}),
            default=0,
        )

        for idx, p_i in enumerate(chain[1:], start=1):
            has_neg_edge = any(
                True
                for subject in subjects
                for _u, v, data in graph.out_edges(subject, data=True)
                if v == p_i and data.get("type") == "defeasible_lacks_property"
            )
            if not has_neg_edge:
                continue

            generic_serial = float(max_serial + 1 + idx)

            for subject in subjects:
                post = "UNDETERMINED"
                baseline = resolver.resolve(
                    graph, subject, hypothesis_target, serial_cutoff=max_serial,
                )
                mode_a = _derive_mode_a(baseline, post, "quantified_gen")
                mode_b = _STATE_TO_MODE_B[post]

                records.append(QuestionRecord(
                    id=_make_record_id(subject, 3, "quantified_gen", p_i, generic_serial),
                    question_type=3,
                    subject=subject,
                    hypothesis_target=hypothesis_target,
                    pi_add_type="quantified_gen",
                    pi_add_node=p_i,
                    pi_add_serial=generic_serial,
                    baseline_state=baseline,
                    post_update_state=post,
                    mode_a_label=mode_a,
                    mode_b_label=mode_b,
                    difficulty={"scope": "generic"},
                    extra={"kind": "generic", "p_root": p_root, "p_i": p_i},
                ))

        return records


# ---------------------------------------------------------------------------
# Type 4 Handler  (produces Type4Episode objects, not QuestionRecords)
# ---------------------------------------------------------------------------

class Type4Handler:
    """
    Creates Type4Episode objects for source-priority belief-revision tests.

    Supports two scenario types, configured via the `scenarios` list:
      "contradiction" — B denies a property A will confirm (or has confirmed).
      "bypass"        — B denies a rule edge; A introduces a skip rule that
                        routes around B's defeat.

    Episodes are cycled through the scenarios list.  For contradiction, a real
    positive defeasible edge from the subject is used as A's anchor.  For
    bypass, all serials are synthetic.

    Bypass uniqueness (Fix 5): a conversation-level `used_rule_claims` set
    tracks (source_a, bypass_src, bypass_tgt, polarity) tuples.  Each bypass
    claim is guaranteed unique within a conversation; if no unique combination
    can be found, the bypass episode is skipped with a warning rather than
    repeating a claim.

    Parameters
    ----------
    source_names  : ordered name list; index 0 = highest reliability.
    scenarios     : cycle list, e.g. ["contradiction", "bypass"].
    b_first_prob  : probability that B fires before A in each episode.
    """

    MAX_EPISODES_PER_SUBJECT = 2

    def __init__(
        self,
        source_names: Optional[List[str]] = None,
        scenarios: Optional[List[str]] = None,
        b_first_prob: float = 0.7,
    ):
        self.source_names: List[str] = source_names or ["Expert A", "Expert B"]
        self.scenarios: List[str] = scenarios or ["contradiction", "bypass"]
        self.b_first_prob = b_first_prob

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def select_and_create_episodes(
        self,
        graph,
        chain: List[str],
        subjects: List[str],
        hypothesis_target: str,
        generation_report: Optional[dict] = None,
    ) -> Tuple[List[Type4Episode], Dict[str, set]]:
        """
        Returns (episodes, type4_serials_by_subject).

        type4_serials_by_subject maps subject → set of real graph-edge serials
        reserved for Type4 contradiction episodes (Type1Handler skips them).

        `used_rule_claims` is maintained across all subjects so the same
        (source_a, bypass_src, bypass_tgt) claim never appears twice in one
        conversation (Fix 5).
        """
        if generation_report is None:
            generation_report = {"warnings": []}

        chain_set = set(chain)
        episodes: List[Type4Episode] = []
        type4_serials_by_subject: Dict[str, set] = {}
        n_src = len(self.source_names)

        # Conversation-level claim uniqueness tracker (Fix 5).
        # Key: (source_a, bypass_src, bypass_tgt, "has")
        used_rule_claims: set = set()

        for subject in subjects:
            subj_edges = [
                (v, data)
                for _, v, data in graph.out_edges(subject, data=True)
                if data.get("type") in {"defeasible_has_property", "defeasible_lacks_property"}
                and v in chain_set
            ]
            subj_edges.sort(key=lambda x: x[1].get("serial", 0))

            max_graph_serial = max(
                (d.get("serial", 0) for _, d in subj_edges), default=0
            ) if subj_edges else 0

            pos_edges = [
                (v, d) for v, d in subj_edges
                if d.get("type") == "defeasible_has_property"
            ]
            n_pos = len(pos_edges)
            second_half_pos = list(pos_edges[n_pos // 2:]) if n_pos >= 1 else []
            random.shuffle(second_half_pos)

            bypass_indices = self._select_bypass_indices(chain, self.MAX_EPISODES_PER_SUBJECT)

            reserved_serials: set = set()
            subj_episodes: List[Type4Episode] = []
            used_b: set = set()
            used_chain_nodes: set = set()
            pairs_created = 0
            scenario_cursor = 0
            bypass_idx_cursor = 0
            pos_cursor = 0

            for _attempt in range(self.MAX_EPISODES_PER_SUBJECT * 4):
                if pairs_created >= self.MAX_EPISODES_PER_SUBJECT:
                    break

                scenario_type = self.scenarios[scenario_cursor % len(self.scenarios)]
                scenario_cursor += 1
                scenario = "b_first" if random.random() < self.b_first_prob else "a_first"
                b_claim_level = random.choice(["rule", "object"])

                if scenario_type == "contradiction":
                    ep = self._make_contradiction_episode(
                        subject, hypothesis_target, chain,
                        second_half_pos, pos_cursor, used_chain_nodes,
                        max_graph_serial, reserved_serials, used_b,
                        scenario, b_claim_level, n_src,
                    )
                    if ep is None:
                        continue
                    pos_cursor += 1

                else:  # bypass
                    if bypass_idx_cursor >= len(bypass_indices):
                        continue
                    edge_idx = bypass_indices[bypass_idx_cursor]
                    bypass_idx_cursor += 1
                    ep = self._make_bypass_episode(
                        subject, hypothesis_target, chain,
                        edge_idx, used_chain_nodes,
                        max_graph_serial, pairs_created, used_b,
                        scenario, b_claim_level, n_src,
                        used_rule_claims, generation_report,
                    )
                    if ep is None:
                        continue

                used_chain_nodes.add(ep.p_i)
                used_b.add(ep.b_serial)
                subj_episodes.append(ep)
                pairs_created += 1

                # Register the bypass claim so later subjects/episodes can't reuse it.
                if ep.scenario_type == "bypass" and ep.bypass_source and ep.bypass_target:
                    used_rule_claims.add((ep.source_a, ep.bypass_source, ep.bypass_target, "has"))

            episodes.extend(subj_episodes)
            type4_serials_by_subject[subject] = reserved_serials

        return episodes, type4_serials_by_subject

    # ------------------------------------------------------------------
    # Episode factories
    # ------------------------------------------------------------------

    def _make_contradiction_episode(
        self,
        subject, hypothesis_target, chain,
        second_half_pos, pos_cursor, used_chain_nodes,
        max_graph_serial, reserved_serials, used_b,
        scenario, b_claim_level, n_src,
    ) -> Optional[Type4Episode]:
        """Create a contradiction episode using a real positive graph edge as A's anchor."""
        avail = [
            (v, d) for v, d in second_half_pos[pos_cursor:]
            if v not in used_chain_nodes
        ]
        if not avail:
            return None
        p_i_node, edge_data = avail[0]
        a_serial = float(edge_data.get("serial", max_graph_serial + 2))
        reserved_serials.add(edge_data.get("serial"))

        if scenario == "b_first":
            pool = [
                i + 0.5
                for i in range(max(1, int(a_serial) // 2))
                if i + 0.5 not in used_b
            ]
            b_serial = random.choice(pool) if pool else max(0.5, a_serial - 0.5)
        else:
            b_serial = a_serial + 0.5
            while b_serial in used_b:
                b_serial += 0.1

        a_idx, b_idx = self._pick_source_pair(n_src)
        return Type4Episode(
            subject=subject,
            hypothesis_target=hypothesis_target,
            p_i=p_i_node,
            b_serial=b_serial,
            a_serial=a_serial,
            b_type="defeasible_lacks_property",
            a_type="defeasible_has_property",
            source_a=self.source_names[a_idx],
            source_b=self.source_names[b_idx],
            scenario=scenario,
            scenario_type="contradiction",
            bypass_source=None,
            bypass_target=None,
            b_claim_level=b_claim_level,
        )

    def _make_bypass_episode(
        self,
        subject, hypothesis_target, chain,
        edge_idx, used_chain_nodes,
        max_graph_serial, pair_offset, used_b,
        scenario, b_claim_level, n_src,
        used_rule_claims: set,
        generation_report: dict,
    ) -> Optional[Type4Episode]:
        """
        Create a bypass episode with synthetic serials.

        Enumerates all valid (bypass_src, bypass_tgt) pairs for the given
        denied-edge index (Fix 5), picks the first combination of (nodes,
        source pair) whose claim key is not yet in used_rule_claims.
        """
        if edge_idx + 1 >= len(chain):
            return None
        p_i_node = chain[edge_idx + 1]
        if p_i_node in used_chain_nodes:
            return None

        bypass_options = self._enumerate_bypass_options(chain, edge_idx)
        if not bypass_options:
            generation_report["warnings"].append(
                f"Type4 bypass: subject={subject} edge_idx={edge_idx} "
                "— no valid bypass node pairs; skipping."
            )
            return None

        # Try each (bypass_src, bypass_tgt) option with a freshly picked source pair.
        # Stop at the first combination not already in used_rule_claims.
        chosen_src = chosen_tgt = None
        chosen_a_idx = chosen_b_idx = None

        for bypass_src, bypass_tgt in bypass_options:
            # Try multiple source pairs to maximise variety.
            for _ in range(min(n_src * 2, 8)):
                a_idx, b_idx = self._pick_source_pair(n_src)
                source_a = self.source_names[a_idx]
                claim_key = (source_a, bypass_src, bypass_tgt, "has")
                if claim_key not in used_rule_claims:
                    chosen_src, chosen_tgt = bypass_src, bypass_tgt
                    chosen_a_idx, chosen_b_idx = a_idx, b_idx
                    break
            if chosen_src is not None:
                break

        if chosen_src is None:
            generation_report["warnings"].append(
                f"Type4 bypass: subject={subject} edge_idx={edge_idx} "
                "— all bypass node/source combinations already used; skipping."
            )
            return None

        base_syn = float(max_graph_serial + 2 + 2 * pair_offset)
        if scenario == "b_first":
            b_serial, a_serial = base_syn, base_syn + 1.0
        else:
            a_serial, b_serial = base_syn, base_syn + 1.0

        while b_serial in used_b:
            b_serial += 0.1

        return Type4Episode(
            subject=subject,
            hypothesis_target=hypothesis_target,
            p_i=p_i_node,
            b_serial=b_serial,
            a_serial=a_serial,
            b_type="defeasible_lacks_property",
            a_type="defeasible_has_property",
            source_a=self.source_names[chosen_a_idx],
            source_b=self.source_names[chosen_b_idx],
            scenario=scenario,
            scenario_type="bypass",
            bypass_source=chosen_src,
            bypass_target=chosen_tgt,
            b_claim_level=b_claim_level,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _select_bypass_indices(self, chain: List[str], num_pairs: int) -> List[int]:
        """
        Return spread-selected chain edge indices valid for bypass.

        Valid range: i in [1, len(chain)-3] so that j=0 < i and
        k=len(chain)-1 >= i+2 (minimum bypass skip distance of 2 edges).
        """
        valid_start = 1
        valid_end = len(chain) - 3
        if valid_start > valid_end:
            return []
        valid_range = valid_end - valid_start + 1
        if num_pairs == 1:
            return [valid_start + valid_range // 2]
        segment_size = valid_range / num_pairs
        indices = []
        for k in range(num_pairs):
            mid = valid_start + int((k + 0.5) * segment_size)
            mid = max(valid_start, min(mid, valid_end))
            indices.append(mid)
        seen: set = set()
        unique = []
        for idx in indices:
            if idx not in seen:
                unique.append(idx)
                seen.add(idx)
        return unique

    def _enumerate_bypass_options(
        self, chain: List[str], denied_edge_idx: int
    ) -> List[Tuple[str, str]]:
        """
        Return all valid (Pj, Pk) bypass pairs for a bypass around denied_edge_idx.

        Constraint: j < denied_edge_idx  AND  k >= denied_edge_idx + 2.
        Options are shuffled so callers get variety across episodes.
        """
        i = denied_edge_idx
        options: List[Tuple[str, str]] = []
        for j in range(0, i):
            for k in range(i + 2, len(chain)):
                options.append((chain[j], chain[k]))
        random.shuffle(options)
        return options

    def _pick_source_pair(self, n_src: int) -> Tuple[int, int]:
        if n_src >= 2:
            a_idx = random.randint(0, n_src - 2)
            b_idx = random.randint(a_idx + 1, n_src - 1)
        else:
            a_idx, b_idx = 0, 0
        return a_idx, b_idx


# ---------------------------------------------------------------------------
# Type 4 Conversation Plan  (rule-based redesign)
# ---------------------------------------------------------------------------

@dataclass
class Type4PlannedEvent:
    """
    One event within a Type4ConversationPlan.

    event_type = "source_claim":
        A source (A or B) makes a claim about a chain property.
        claim_polarity = "pos" → source asserts the property holds.
        claim_polarity = "neg" → source asserts the property is denied.
        claim_level    = "rule"   → NL is a general rule statement.
        claim_level    = "object" → NL is about the specific subject.
        bypass_src / bypass_tgt are only set when the claim introduces a
        bypass rule (A's claim in bypass episodes).

    event_type = "edge_reveal":
        A real defeasible edge from the graph is revealed.  Effect is
        computed the same way as a Type-1 reveal in Phase 1.
        edge_kind = "pos" | "neg" (exception phrasing is not used in T4).
    """
    event_type: str                       # "source_claim" | "edge_reveal"

    # --- source_claim ---
    source: str = ""
    is_source_a: bool = False
    claim_node: str = ""                  # chain node being claimed
    claim_polarity: str = ""              # "pos" | "neg"
    claim_level: str = "rule"             # "rule" | "object"
    bypass_src: Optional[str] = None      # bypass only: bypass source node
    bypass_tgt: Optional[str] = None      # bypass only: bypass target node

    # --- edge_reveal ---
    edge_subject: str = ""
    edge_node: str = ""
    edge_kind: str = ""                   # "pos" | "neg"
    edge_serial: float = 0.0


@dataclass
class Type4ConversationPlan:
    """
    Complete pre-planned conversation for one Type 4 subject section.

    The events list fully determines the turn order; the NL generator
    renders it without making any further logical decisions.

    Contradiction episode (scenario_type="contradiction"):
        Both sources dispute the same contested_node.
        A must have a real positive graph edge at contested_node (anchor).
        B (lower priority) claims negative; A (higher priority) reinstates.

    Bypass episode (scenario_type="bypass"):
        B denies rule b_block_src → b_block_tgt (blocking the chain).
        A introduces skip rule a_bypass_src → a_bypass_tgt that routes
        around B's block.  Constraint:
            idx(a_bypass_src) < idx(b_block_src) < idx(b_block_tgt) < idx(a_bypass_tgt)
    """
    subject: str
    hypothesis_target: str
    source_a: str                         # higher-priority source
    source_b: str                         # lower-priority source
    scenario_type: str                    # "contradiction" | "bypass"
    ordering: str                         # "b_first" | "a_first"

    # contradiction-specific
    contested_node: str = ""
    claim_level: str = "rule"

    # bypass-specific
    b_block_src: str = ""
    b_block_tgt: str = ""
    a_bypass_src: str = ""
    a_bypass_tgt: str = ""

    events: List = field(default_factory=list)   # List[Type4PlannedEvent]


class Type4ConversationPlanner:
    """
    Produces Type4ConversationPlan objects for a list of Type 4 subjects.

    Bypass selection rule
    ---------------------
    1. Select A's bypass pair (a_src, a_tgt) with gap ≥ 3 chain indices.
    2. Select B's blocked pair (b_src, b_tgt) strictly inside A's range:
           chain_idx(a_src) < chain_idx(b_src) < chain_idx(b_tgt) < chain_idx(a_tgt)
    This guarantees A's bypass always circumvents B's block.

    Contradiction selection rule
    ----------------------------
    Requires a real positive defeasible edge from the subject to the
    contested node (used as A's anchor so the resolver restores ENTAILED
    after A fires).  Preference is given to second-half chain nodes.

    Interleaved edge reveals
    ------------------------
    Between the two source claims, one real T1 defeasible edge from the
    subject is revealed (like a Type-1 update, with full PMPM resolution).
    Optionally, one more edge reveal is appended after both claims.
    The reveal slots obey the constraint:
        "edges for nodes before / at B's claim node" (b_first),
        "edges for nodes before / at A's claim node" (a_first).
    Edges beyond the last source claim node are not revealed until after
    both A and B have spoken.
    """

    def __init__(
        self,
        source_names: Optional[List[str]] = None,
        scenarios: Optional[List[str]] = None,
        b_first_prob: float = 0.7,
    ):
        self.source_names: List[str] = source_names or ["Expert A", "Expert B"]
        self.scenarios: List[str] = scenarios or ["contradiction", "bypass"]
        self.b_first_prob = b_first_prob

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def plan_for_subjects(
        self,
        graph,
        chain: List[str],
        subjects: List[str],
        hypothesis_target: str,
        generation_report: Optional[dict] = None,
    ) -> List[Type4ConversationPlan]:
        if generation_report is None:
            generation_report = {"warnings": []}

        plans: List[Type4ConversationPlan] = []
        used_bypass_claims: set = set()

        for subject in subjects:
            plan = self._plan_for_subject(
                graph, subject, hypothesis_target, chain,
                generation_report, used_bypass_claims,
            )
            if plan is not None:
                plans.append(plan)
                if plan.scenario_type == "bypass":
                    used_bypass_claims.add((plan.source_a, plan.a_bypass_src, plan.a_bypass_tgt))

        return plans

    # ------------------------------------------------------------------
    # Per-subject dispatch
    # ------------------------------------------------------------------

    def _plan_for_subject(
        self,
        graph,
        subject: str,
        hypothesis_target: str,
        chain: List[str],
        generation_report: dict,
        used_bypass_claims: set,
    ) -> Optional[Type4ConversationPlan]:
        scenario_type = random.choice(self.scenarios)
        ordering = "b_first" if random.random() < self.b_first_prob else "a_first"
        a_idx, b_idx = self._pick_source_pair_v2()
        source_a = self.source_names[a_idx]
        source_b = self.source_names[b_idx]

        if scenario_type == "contradiction":
            return self._plan_contradiction(
                graph, subject, hypothesis_target, chain,
                ordering, source_a, source_b, generation_report,
            )
        return self._plan_bypass(
            graph, subject, hypothesis_target, chain,
            ordering, source_a, source_b,
            used_bypass_claims, generation_report,
        )

    # ------------------------------------------------------------------
    # Contradiction planner
    # ------------------------------------------------------------------

    def _plan_contradiction(
        self,
        graph,
        subject: str,
        hypothesis_target: str,
        chain: List[str],
        ordering: str,
        source_a: str,
        source_b: str,
        generation_report: dict,
    ) -> Optional[Type4ConversationPlan]:
        chain_set = set(chain)

        pos_edges = sorted(
            [
                (v, d)
                for _, v, d in graph.out_edges(subject, data=True)
                if d.get("type") == "defeasible_has_property" and v in chain_set
            ],
            key=lambda x: x[1].get("serial", 0),
        )
        if not pos_edges:
            generation_report["warnings"].append(
                f"Type4Planner contradiction: subject={subject} — no positive edges; skipping."
            )
            return None

        n = len(pos_edges)
        candidates = pos_edges[n // 2:] if n >= 2 else pos_edges
        contested_node, _ = random.choice(candidates)
        contested_idx = chain.index(contested_node)
        claim_level = "object"

        # T1 edges available for reveals (exclude A's anchor edge)
        all_t1 = [
            (v, d)
            for _, v, d in graph.out_edges(subject, data=True)
            if d.get("type") in {"defeasible_has_property", "defeasible_lacks_property"}
            and v in chain_set
            and not (v == contested_node and d.get("type") == "defeasible_has_property")
        ]

        # Edges eligible for the between-claim reveal slot
        if ordering == "b_first":
            between_pool = [(v, d) for v, d in all_t1 if chain.index(v) < contested_idx]
        else:
            between_pool = [(v, d) for v, d in all_t1 if chain.index(v) <= contested_idx]

        between_edge = random.choice(between_pool) if between_pool else None

        remaining = [e for e in all_t1 if between_edge is None or e[0] != between_edge[0]]
        post_edge = (
            random.choice(remaining)
            if remaining and random.random() < 0.6
            else None
        )

        def _reveal(v, d) -> Type4PlannedEvent:
            kind = "pos" if d.get("type") == "defeasible_has_property" else "neg"
            return Type4PlannedEvent(
                event_type="edge_reveal",
                edge_subject=subject, edge_node=v,
                edge_kind=kind, edge_serial=float(d.get("serial", 0)),
            )

        b_ev = Type4PlannedEvent(
            event_type="source_claim",
            source=source_b, is_source_a=False,
            claim_node=contested_node, claim_polarity="neg",
            claim_level=claim_level,
        )
        a_ev = Type4PlannedEvent(
            event_type="source_claim",
            source=source_a, is_source_a=True,
            claim_node=contested_node, claim_polarity="pos",
            claim_level=claim_level,
        )

        events: List[Type4PlannedEvent] = []
        if ordering == "b_first":
            events.append(b_ev)
            if between_edge:
                events.append(_reveal(*between_edge))
            events.append(a_ev)
        else:
            events.append(a_ev)
            if between_edge:
                events.append(_reveal(*between_edge))
            events.append(b_ev)
        if post_edge:
            events.append(_reveal(*post_edge))

        return Type4ConversationPlan(
            subject=subject,
            hypothesis_target=hypothesis_target,
            source_a=source_a, source_b=source_b,
            scenario_type="contradiction",
            ordering=ordering,
            contested_node=contested_node,
            claim_level=claim_level,
            events=events,
        )

    # ------------------------------------------------------------------
    # Bypass planner
    # ------------------------------------------------------------------

    def _plan_bypass(
        self,
        graph,
        subject: str,
        hypothesis_target: str,
        chain: List[str],
        ordering: str,
        source_a: str,
        source_b: str,
        used_bypass_claims: set,
        generation_report: dict,
    ) -> Optional[Type4ConversationPlan]:
        n = len(chain)
        if n < 5:
            generation_report["warnings"].append(
                f"Type4Planner bypass: chain length {n} < 5; skipping subject={subject}."
            )
            return None

        chain_set = set(chain)

        # Enumerate valid (a_src_idx, a_tgt_idx) with gap >= 3; pick randomly
        a_options = [
            (ai, at)
            for ai in range(0, n - 3)
            for at in range(ai + 3, n)
        ]
        random.shuffle(a_options)

        chosen = None
        for a_src_idx, a_tgt_idx in a_options:
            b_options = [
                (bi, bt)
                for bi in range(a_src_idx + 1, a_tgt_idx - 1)
                for bt in range(bi + 1, a_tgt_idx)
            ]
            if not b_options:
                continue
            b_src_idx, b_tgt_idx = random.choice(b_options)
            a_src_node = chain[a_src_idx]
            a_tgt_node = chain[a_tgt_idx]
            b_src_node = chain[b_src_idx]
            b_tgt_node = chain[b_tgt_idx]
            if (source_a, a_src_node, a_tgt_node) not in used_bypass_claims:
                chosen = (a_src_node, a_tgt_node, b_src_node, b_tgt_node,
                          a_src_idx, a_tgt_idx, b_src_idx, b_tgt_idx)
                break

        if chosen is None:
            generation_report["warnings"].append(
                f"Type4Planner bypass: subject={subject} — no valid bypass pair; skipping."
            )
            return None

        a_bypass_src, a_bypass_tgt, b_block_src, b_block_tgt, \
            a_src_idx, a_tgt_idx, b_src_idx, _ = chosen

        all_t1 = [
            (v, d)
            for _, v, d in graph.out_edges(subject, data=True)
            if d.get("type") in {"defeasible_has_property", "defeasible_lacks_property"}
            and v in chain_set
        ]

        # Between-claim reveal: nodes at or before b_src_idx
        between_pool = [(v, d) for v, d in all_t1 if chain.index(v) <= b_src_idx]
        between_edge = random.choice(between_pool) if between_pool else None

        # Post-claim reveal: nodes at or before a_src_idx (A's bypass source)
        post_pool = [
            (v, d) for v, d in all_t1
            if chain.index(v) <= a_src_idx
            and (between_edge is None or v != between_edge[0])
        ]
        post_edge = random.choice(post_pool) if post_pool else None

        def _reveal(v, d) -> Type4PlannedEvent:
            kind = "pos" if d.get("type") == "defeasible_has_property" else "neg"
            return Type4PlannedEvent(
                event_type="edge_reveal",
                edge_subject=subject, edge_node=v,
                edge_kind=kind, edge_serial=float(d.get("serial", 0)),
            )

        b_ev = Type4PlannedEvent(
            event_type="source_claim",
            source=source_b, is_source_a=False,
            claim_node=b_block_tgt, claim_polarity="neg",
            claim_level="rule",
            bypass_src=b_block_src, bypass_tgt=b_block_tgt,
        )
        a_ev = Type4PlannedEvent(
            event_type="source_claim",
            source=source_a, is_source_a=True,
            claim_node=a_bypass_tgt, claim_polarity="pos",
            claim_level="rule",
            bypass_src=a_bypass_src, bypass_tgt=a_bypass_tgt,
        )

        events: List[Type4PlannedEvent] = []
        if ordering == "b_first":
            events.append(b_ev)
            if between_edge:
                events.append(_reveal(*between_edge))
            events.append(a_ev)
        else:
            events.append(a_ev)
            if between_edge:
                events.append(_reveal(*between_edge))
            events.append(b_ev)
        if post_edge:
            events.append(_reveal(*post_edge))

        return Type4ConversationPlan(
            subject=subject,
            hypothesis_target=hypothesis_target,
            source_a=source_a, source_b=source_b,
            scenario_type="bypass",
            ordering=ordering,
            b_block_src=b_block_src, b_block_tgt=b_block_tgt,
            a_bypass_src=a_bypass_src, a_bypass_tgt=a_bypass_tgt,
            events=events,
        )

    def _pick_source_pair_v2(self) -> Tuple[int, int]:
        n = len(self.source_names)
        if n >= 2:
            a_idx = random.randint(0, n - 2)
            b_idx = random.randint(a_idx + 1, n - 1)
        else:
            a_idx, b_idx = 0, 0
        return a_idx, b_idx


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class DefaultReasoningQuestionGenerator:
    """
    Thin orchestrator: inspects a populated default-topology graph, runs each
    handler, collects QuestionRecord objects and Type4Episode objects.

    Returns (records, episodes, generation_report).

    Parameters
    ----------
    type4_handler : Type4Handler or None
        Pre-constructed Type4Handler submodule.  Pass None (default) to
        disable Type 4 generation entirely.
    type3_enabled : bool
        Enable Type 3 quantified-uncertainty questions (default False).
    type4_enabled : bool
        When False, Type 4 is never generated even if type4_handler is given.
    type4_object_fraction : float
        Fraction of graph subjects reserved exclusively for Type 4 episodes.
        The remaining subjects go to Type 1/2 sections.  Min 1 subject is
        always reserved for Type 4 (provided there are ≥ 2 subjects).
    max_hypothesis_sections : int | None
        Hard cap on the total number of hypothesis sections in the conversation
        (Phase 1 + Phase 2 combined).  None = no cap.
    max_updates_per_section : int | None
        Maximum Type-1/2/3 update records rendered per subject section.
        None = no cap.
    """

    def __init__(
        self,
        type4_handler: Optional["Type4Handler"] = None,
        type3_enabled: bool = False,
        type4_enabled: bool = True,
        type4_object_fraction: float = 0.5,
        max_hypothesis_sections: Optional[int] = None,
        max_updates_per_section: Optional[int] = None,
        type4_planner: Optional["Type4ConversationPlanner"] = None,
    ):
        self.type4_handler = type4_handler
        self.type4_planner = type4_planner
        self.type3_enabled = type3_enabled
        self.type4_enabled = type4_enabled
        self.type4_object_fraction = type4_object_fraction
        self.max_hypothesis_sections = max_hypothesis_sections
        self.max_updates_per_section = max_updates_per_section
        self._last_report: dict = {}

    def generate(
        self, graph
    ) -> Tuple[List[QuestionRecord], List[Type4Episode], dict]:
        """
        Returns (records, type4_episodes, generation_report).

        records       – QuestionRecord list for Types 1, 2, (optionally 3).
        type4_episodes – Type4Episode list; empty if include_type4=False.
        generation_report – audit dict.
        """
        from path_relation_module.default_resolver import DefaultChainResolver

        resolver = DefaultChainResolver()

        generation_report: dict = {
            "chain_length": 0,
            "num_subjects": 0,
            "type12_subjects": [],
            "type4_subjects": [],
            "warnings": [],
            "records_by_type": {1: 0, 2: 0, 3: 0, 4: 0},
        }

        # ------------------------------------------------------------------
        # 1. Discover hypothesis edges
        # ------------------------------------------------------------------
        hypothesis_edges = [
            (u, v, d)
            for u, v, d in graph.edges(data=True)
            if d.get("type") == "hypothesis_edge"
        ]

        if not hypothesis_edges:
            generation_report["warnings"].append("No hypothesis_edge found in graph.")
            self._last_report = generation_report
            return [], [], generation_report

        hyp_targets = list({v for _, v, _ in hypothesis_edges})
        if len(hyp_targets) > 1:
            generation_report["warnings"].append(
                f"Multiple hypothesis targets found ({hyp_targets}); using {hyp_targets[0]}."
            )
        hypothesis_target = hyp_targets[0]

        subjects = list(dict.fromkeys(
            u for u, v, _ in hypothesis_edges if v == hypothesis_target
        ))
        generation_report["num_subjects"] = len(subjects)

        if not subjects:
            generation_report["warnings"].append("No subjects found for hypothesis target.")
            self._last_report = generation_report
            return [], [], generation_report

        # ------------------------------------------------------------------
        # 2. Extract chain
        # ------------------------------------------------------------------
        chain = resolver.extract_chain(graph, subjects[0], hypothesis_target)
        generation_report["chain_length"] = len(chain)

        if len(chain) < 2:
            generation_report["warnings"].append(
                f"Chain length {len(chain)} < 2; cannot generate questions."
            )
            self._last_report = generation_report
            return [], [], generation_report

        # ------------------------------------------------------------------
        # 3. Sanity checks
        # ------------------------------------------------------------------
        for s in subjects:
            bs = resolver.baseline_state(graph, s, hypothesis_target)
            if bs == "DEFEATED":
                generation_report["warnings"].append(
                    f"Subject={s}: baseline already DEFEATED. Graph may be malformed."
                )

        # ------------------------------------------------------------------
        # 4. Subject partitioning (Fix 1)
        #    Subjects are split into disjoint sets: type12_subjects go to
        #    Type-1/2/3 handlers; type4_subjects go exclusively to Type4Handler.
        #    This prevents the same object appearing in multiple hypothesis
        #    sections with carry-forward context contamination.
        # ------------------------------------------------------------------
        type12_subjects, type4_subjects = self._partition_subjects(
            subjects, generation_report
        )

        # Apply max_hypothesis_sections cap across both phases (Fix 8b).
        type12_subjects, type4_subjects = self._apply_section_cap(
            type12_subjects, type4_subjects, generation_report
        )

        generation_report["type12_subjects"] = list(type12_subjects)
        generation_report["type4_subjects"] = list(type4_subjects)

        # ------------------------------------------------------------------
        # 5. Type4 episode selection (must happen before Type1 so we can
        #    pass the reserved serials to Type1Handler)
        # ------------------------------------------------------------------
        type4_episodes: List[Type4Episode] = []
        type4_serials_by_subject: Dict[str, set] = {}

        if self.type4_handler is not None and type4_subjects:
            type4_episodes, type4_serials_by_subject = (
                self.type4_handler.select_and_create_episodes(
                    graph, chain, type4_subjects, hypothesis_target,
                    generation_report=generation_report,
                )
            )
            generation_report["records_by_type"][4] = len(type4_episodes)

        # ------------------------------------------------------------------
        # 6. Run Type1 / Type2 / (optional) Type3 handlers on type12_subjects
        # ------------------------------------------------------------------
        all_records: List[QuestionRecord] = []

        t1 = Type1Handler()
        t1_records = t1.generate(
            graph, chain, resolver, type12_subjects, hypothesis_target,
            generation_report, type4_serials=type4_serials_by_subject,
        )
        all_records += t1_records
        generation_report["records_by_type"][1] = len(t1_records)

        t2 = Type2Handler()
        t2_records = t2.generate(graph, chain, resolver, type12_subjects, hypothesis_target, generation_report)
        all_records += t2_records
        generation_report["records_by_type"][2] = len(t2_records)

        if self.type3_enabled:
            t3 = Type3Handler()
            t3_records = t3.generate(graph, chain, resolver, type12_subjects, hypothesis_target, generation_report)
            all_records += t3_records
            generation_report["records_by_type"][3] = len(t3_records)

        # ------------------------------------------------------------------
        # 7. Apply max_updates_per_section cap (Fix 8b)
        #    Cap the number of records per subject AFTER generation so the
        #    cap is applied uniformly regardless of question type mix.
        # ------------------------------------------------------------------
        if self.max_updates_per_section is not None:
            all_records = self._cap_records_per_subject(
                all_records, type12_subjects, generation_report
            )

        # ------------------------------------------------------------------
        # 8. Sort records (by subject order, then serial) and return
        # ------------------------------------------------------------------
        subject_order = {s: i for i, s in enumerate(type12_subjects)}
        all_records.sort(key=lambda r: (subject_order.get(r.subject, 999), r.pi_add_serial))

        self._last_report = generation_report
        return all_records, type4_episodes, generation_report

    def get_last_report(self) -> dict:
        return self._last_report

    # ------------------------------------------------------------------
    # New plan-based generation path (uses Type4ConversationPlanner)
    # ------------------------------------------------------------------

    def generate_planned(
        self, graph
    ) -> Tuple[List[QuestionRecord], List[Type4ConversationPlan], dict]:
        """
        Like generate(), but uses Type4ConversationPlanner instead of
        Type4Handler.  Returns (records, type4_plans, generation_report).

        The returned type4_plans is a list of Type4ConversationPlan objects
        that the NL generator renders via its plan-based Phase 2 path.
        type4_plans is empty if no type4_planner was provided or if there
        are no type4_subjects.
        """
        from path_relation_module.default_resolver import DefaultChainResolver

        resolver = DefaultChainResolver()

        generation_report: dict = {
            "chain_length": 0,
            "num_subjects": 0,
            "type12_subjects": [],
            "type4_subjects": [],
            "warnings": [],
            "records_by_type": {1: 0, 2: 0, 3: 0, 4: 0},
        }

        hypothesis_edges = [
            (u, v, d)
            for u, v, d in graph.edges(data=True)
            if d.get("type") == "hypothesis_edge"
        ]
        if not hypothesis_edges:
            generation_report["warnings"].append("No hypothesis_edge found in graph.")
            self._last_report = generation_report
            return [], [], generation_report

        hyp_targets = list({v for _, v, _ in hypothesis_edges})
        hypothesis_target = hyp_targets[0]
        subjects = list(dict.fromkeys(u for u, v, _ in hypothesis_edges if v == hypothesis_target))
        generation_report["num_subjects"] = len(subjects)

        if not subjects:
            generation_report["warnings"].append("No subjects found for hypothesis target.")
            self._last_report = generation_report
            return [], [], generation_report

        chain = resolver.extract_chain(graph, subjects[0], hypothesis_target)
        generation_report["chain_length"] = len(chain)
        if len(chain) < 2:
            generation_report["warnings"].append(
                f"Chain length {len(chain)} < 2; cannot generate questions."
            )
            self._last_report = generation_report
            return [], [], generation_report

        # Partition subjects
        use_type4 = (
            self.type4_planner is not None
            and self.type4_enabled
            and len(subjects) >= 2
        )
        if use_type4:
            n_type4 = max(1, round(len(subjects) * self.type4_object_fraction))
            n_type4 = min(n_type4, len(subjects) - 1)
            shuffled = list(subjects)
            random.shuffle(shuffled)
            type4_subjects = shuffled[:n_type4]
            type12_subjects = shuffled[n_type4:]
        else:
            type12_subjects = list(subjects)
            type4_subjects = []

        type12_subjects, type4_subjects = self._apply_section_cap(
            type12_subjects, type4_subjects, generation_report
        )
        generation_report["type12_subjects"] = list(type12_subjects)
        generation_report["type4_subjects"] = list(type4_subjects)

        # Build type4 plans (no serials to reserve since type4 subjects
        # are already excluded from type12_subjects)
        type4_plans: List[Type4ConversationPlan] = []
        if use_type4 and type4_subjects:
            type4_plans = self.type4_planner.plan_for_subjects(
                graph, chain, type4_subjects, hypothesis_target,
                generation_report=generation_report,
            )
            generation_report["records_by_type"][4] = len(type4_plans)

        # Type1/2/3 handlers on type12_subjects
        all_records: List[QuestionRecord] = []

        t1_records = Type1Handler().generate(
            graph, chain, resolver, type12_subjects, hypothesis_target,
            generation_report, type4_serials=None,
        )
        all_records += t1_records
        generation_report["records_by_type"][1] = len(t1_records)

        t2_records = Type2Handler().generate(
            graph, chain, resolver, type12_subjects, hypothesis_target,
            generation_report,
        )
        all_records += t2_records
        generation_report["records_by_type"][2] = len(t2_records)

        if self.type3_enabled:
            t3_records = Type3Handler().generate(
                graph, chain, resolver, type12_subjects, hypothesis_target,
                generation_report,
            )
            all_records += t3_records
            generation_report["records_by_type"][3] = len(t3_records)

        if self.max_updates_per_section is not None:
            all_records = self._cap_records_per_subject(
                all_records, type12_subjects, generation_report
            )

        subject_order = {s: i for i, s in enumerate(type12_subjects)}
        all_records.sort(
            key=lambda r: (subject_order.get(r.subject, 999), r.pi_add_serial)
        )

        self._last_report = generation_report
        return all_records, type4_plans, generation_report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _partition_subjects(
        self, subjects: List[str], generation_report: dict
    ) -> Tuple[List[str], List[str]]:
        """
        Split subjects into (type12_subjects, type4_subjects).

        Type 4 subjects are a disjoint subset so the same object never
        appears in both Phase 1 and Phase 2 of the conversation (Fix 1).
        """
        use_type4 = (
            self.type4_handler is not None
            and self.type4_enabled
            and len(subjects) >= 2
        )
        if not use_type4:
            if self.type4_handler is not None and len(subjects) < 2:
                generation_report["warnings"].append(
                    "Type4 disabled: graph has fewer than 2 hypothesis subjects."
                )
            return list(subjects), []

        n_type4 = max(1, round(len(subjects) * self.type4_object_fraction))
        n_type4 = min(n_type4, len(subjects) - 1)  # always keep ≥1 for Type 1/2

        shuffled = list(subjects)
        random.shuffle(shuffled)
        type4_subjects = shuffled[:n_type4]
        type12_subjects = shuffled[n_type4:]
        return type12_subjects, type4_subjects

    def _apply_section_cap(
        self,
        type12_subjects: List[str],
        type4_subjects: List[str],
        generation_report: dict,
    ) -> Tuple[List[str], List[str]]:
        """
        Enforce max_hypothesis_sections across both phases (Fix 8b).

        Phase 1 gets priority; Type 4 gets the remainder up to the cap.
        Both lists are randomly truncated when they exceed the budget.
        """
        if self.max_hypothesis_sections is None:
            return type12_subjects, type4_subjects

        cap = self.max_hypothesis_sections
        # Allocate: Phase 1 gets at most ceil(cap/2), Phase 2 gets the rest.
        # If no Type 4, Phase 1 gets the full cap.
        if not type4_subjects:
            if len(type12_subjects) > cap:
                generation_report["warnings"].append(
                    f"max_hypothesis_sections={cap}: truncating type12 from "
                    f"{len(type12_subjects)} to {cap} subjects."
                )
                random.shuffle(type12_subjects)
                type12_subjects = type12_subjects[:cap]
            return type12_subjects, type4_subjects

        # With Type 4: split budget proportionally.
        t4_budget = max(1, cap // 2)
        t12_budget = cap - t4_budget

        if len(type4_subjects) > t4_budget:
            generation_report["warnings"].append(
                f"max_hypothesis_sections cap: truncating type4 from "
                f"{len(type4_subjects)} to {t4_budget} subjects."
            )
            random.shuffle(type4_subjects)
            type4_subjects = type4_subjects[:t4_budget]

        if len(type12_subjects) > t12_budget:
            generation_report["warnings"].append(
                f"max_hypothesis_sections cap: truncating type12 from "
                f"{len(type12_subjects)} to {t12_budget} subjects."
            )
            random.shuffle(type12_subjects)
            type12_subjects = type12_subjects[:t12_budget]

        return type12_subjects, type4_subjects

    def _cap_records_per_subject(
        self,
        records: List[QuestionRecord],
        subjects: List[str],
        generation_report: dict,
    ) -> List[QuestionRecord]:
        """
        Limit records to max_updates_per_section per subject (Fix 8b).

        Keeps records with the smallest pi_add_serial values (earliest
        information) so the conversation opens naturally.
        """
        cap = self.max_updates_per_section
        by_subject: Dict[str, List[QuestionRecord]] = {s: [] for s in subjects}
        for r in records:
            if r.subject in by_subject:
                by_subject[r.subject].append(r)

        result: List[QuestionRecord] = []
        for subj in subjects:
            recs = sorted(by_subject[subj], key=lambda r: r.pi_add_serial)
            if len(recs) > cap:
                generation_report["warnings"].append(
                    f"max_updates_per_section={cap}: dropping "
                    f"{len(recs) - cap} records for subject={subj}."
                )
                recs = recs[:cap]
            result.extend(recs)
        return result
