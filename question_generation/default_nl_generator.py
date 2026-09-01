"""
DefaultReasoningNLGenerator
============================
Converts QuestionRecord / Type4Episode objects into a single multi-turn
growing-context conversation for the default-reasoning topology.

Conversation structure
----------------------
The output is always ONE conversation dict (returned as a one-element list).

  Phase 1 — Type 1/2 section, one subject at a time:
    First subject:  "Pretext: … Hypothesis: … Question: …"
    Each subsequent subject: "New hypothesis: … Question: …"
    Followed by the subject's Type1/2 information-update turns.

  Phase 2 — Type 4 section (comes after all Type 1/2 turns):
    Each subject with Type4 episodes gets its own "New hypothesis" turn.
    Type 4 subjects are disjoint from Phase 1 subjects (Fix 1).

Key design notes
----------------
- T1 and T2 records are randomly interleaved within each subject section so
  that irrelevant updates don't always appear first (Fix 2).

- Reveal-order randomisation (per subject):
    With probability `reveal_order_random_prob` the T1 records for a subject
    are shuffled into a random order instead of the default ascending-serial
    order.  This breaks the predictable "pos edges first → always strengthening
    early, weakening late" positional pattern.

    Serial mode  (default, prob = 1 - reveal_order_random_prob):
      T1 edges revealed low→high serial.  State is computed via
      serial_cutoff, exactly as before.

    Random mode  (prob = reveal_order_random_prob):
      T1 edges shuffled into an arbitrary order.  State is computed via
      serial_cutoff=0 (so NO real graph edges are included by serial
      filter — all real serials start at 1) plus an explicit `extra_edges`
      list of only the edges that have been revealed so far.
      This mirrors the PMPM partial-graph approach: the resolver sees the
      backbone (is_a + implies chain) plus exactly the defeasible edges
      disclosed up to that point.  prev_cutoff is NOT advanced in random
      mode for T1 records (the revealed-edge set takes its role).

- prev_cutoff advances only when a T1 record is processed in serial mode,
  keeping resolver correctness intact when T2 items appear between T1 events.
- Bypass Type-4 A-claims always produce "strengthening" (Fix 4).
- Redundant confirmations from lower-priority sources → NO_EFFECT (Fix 8a).
- max_total_turns provides a hard safety cap on turn count (Fix 8b).
"""

from __future__ import annotations

import logging
import random
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_STATE_TO_MODE_B: Dict[str, str] = {
    "ENTAILED": "yes",
    "DEFEATED": "no",
    "UNDETERMINED": "unknown",
}

_EFFECT_QUESTION = (
    "Does this update strengthen, weaken, or have no effect on the support for the hypothesis?"
)


def _state_delta_mode_a(state_before: str, state_after: str) -> str:
    """
    Derive effect label from the state transition. Used for negative edges in
    random-reveal mode, where the state delta is the authoritative source of truth.
    """
    if state_before == state_after:
        return "NO_EFFECT"
    if state_after == "ENTAILED":
        return "STRENGTHENING"
    if state_before == "ENTAILED":
        return "WEAKENING"
    if state_before == "UNDETERMINED" and state_after == "DEFEATED":
        return "WEAKENING"
    return "NO_EFFECT"


def _compute_pos_mode_a(
    chain: list,
    new_node: str,
    prior_evidence: list,
    state_before: str,
    state_after: str,
) -> str:
    """
    Unified effect rule for revealing a positive defeasible edge at new_node.

    Implements the full rulebook for the ENTAILED→ENTAILED case:
      STRENGTHENING  — new entry is viable (forward sub-chain clear) AND no
                       existing positive already sits closer to the hypothesis
                       (at a later chain position) with its own clear sub-chain.
                       In other words, the new evidence shortens the effective path.
      NO_EFFECT      — new entry's path is blocked, OR an existing positive at
                       a later chain index already covers the same path.

    All other before→after transitions use state-delta (DEFEATED→ENTAILED=
    STRENGTHENING, ENTAILED→DEFEATED=WEAKENING, etc.).

    prior_evidence : (etype, node) pairs visible BEFORE adding the new edge.
    """
    if state_before != "ENTAILED" or state_after != "ENTAILED":
        return _state_delta_mode_a(state_before, state_after)

    chain_idx = {n: i for i, n in enumerate(chain)}
    k = chain_idx.get(new_node, -1)
    if k < 0:
        return "NO_EFFECT"

    neg_set = {nd for et, nd in prior_evidence if et == "defeasible_lacks_property"}
    pos_set = {nd for et, nd in prior_evidence if et == "defeasible_has_property"}

    # New entry is viable only when its forward sub-chain is free of known negatives
    if any(chain[j] in neg_set for j in range(k + 1, len(chain))):
        return "NO_EFFECT"

    # If an existing positive already sits at a later (closer-to-hypothesis) chain
    # position with a clear sub-chain, the new entry does not shorten the path
    for nd in pos_set:
        j = chain_idx.get(nd, -1)
        if j > k and not any(chain[m] in neg_set for m in range(j + 1, len(chain))):
            return "NO_EFFECT"

    return "STRENGTHENING"


def _random_mode_mode_a(
    pi_add_type: str,
    pi_add_node: str,
    chain: list,
    revealed_t1_edges_before: list,
    state_before: str,
    state_after: str,
) -> str:
    """
    Compute the effect label for a T1 record in random-reveal mode.

    Positive edge: delegates to _compute_pos_mode_a (unified path-shortened rule).
    Negative / exception: state-delta.
    """
    if pi_add_type == "pos":
        return _compute_pos_mode_a(
            chain, pi_add_node, revealed_t1_edges_before, state_before, state_after
        )
    return _state_delta_mode_a(state_before, state_after)


def _random_interleave(a_list: list, b_list: list) -> list:
    """
    Merge two lists preserving a_list's internal order while inserting
    b_list items at uniformly random positions.

    This is used to interleave T1 records (ordered by graph serial) with T2
    records (no required order) so that irrelevant updates don't cluster at
    the front of each section.
    """
    if not b_list:
        return list(a_list)
    if not a_list:
        return list(b_list)

    result: list = []
    ai, bi = 0, 0
    na, nb = len(a_list), len(b_list)
    while ai < na and bi < nb:
        remaining_a = na - ai
        remaining_b = nb - bi
        # Probability of picking next b item proportional to its remaining count.
        if random.random() < remaining_b / (remaining_a + remaining_b):
            result.append(b_list[bi])
            bi += 1
        else:
            result.append(a_list[ai])
            ai += 1
    result.extend(a_list[ai:])
    result.extend(b_list[bi:])
    return result


class DefaultReasoningNLGenerator:
    """
    Renders QuestionRecord / Type4Episode lists into a single multi-turn
    conversation dict for the default-reasoning topology.
    """

    def __init__(self, kb_manager):
        self.kb = kb_manager

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build_conversations(
        self,
        graph,
        question_records: List,
        type4_episodes: Optional[List] = None,
        type4_plans: Optional[List] = None,
        source_names: Optional[List[str]] = None,
        max_total_turns: Optional[int] = None,
        reveal_order_random_prob: float = 0.0,
    ) -> List[Dict]:
        """
        Build one merged growing-context conversation for all subjects.

        Phase 1 — Type 1/2 events grouped by subject, announced in order.
                   T1 and T2 records are randomly interleaved (Fix 2).
        Phase 2 — Type 4 events for each subject, always at the end with a
                   fresh "New hypothesis" turn.  Type 4 subjects must be
                   disjoint from Phase 1 subjects (Fix 1 — enforced upstream).

        Parameters
        ----------
        graph                   : populated default-topology graph.
        question_records        : QuestionRecord list from DefaultReasoningQuestionGenerator.
        type4_episodes          : Type4Episode list (legacy path; ignored when type4_plans given).
        type4_plans             : Type4ConversationPlan list (new plan-based path).
        source_names            : ordered source-name list (index 0 = most reliable).
        max_total_turns         : hard safety cap on total conversation turns (Fix 8b).
        reveal_order_random_prob: per-subject probability of using randomised reveal
                                  order for T1 edges instead of ascending-serial order.

        When type4_plans is provided, Phase 2 uses the plan-based renderer which
        interleaves source claims with T1-like edge reveals.  The old episode-based
        path is used only when type4_episodes is provided without type4_plans.

        Returns a one-element list containing the single conversation dict.
        """
        from path_relation_module.default_resolver import DefaultChainResolver
        from question_generation.default_question_generator import _derive_mode_a

        resolver = DefaultChainResolver()
        plans = type4_plans or []
        episodes = type4_episodes or [] if not plans else []

        if not question_records and not plans and not episodes:
            return []

        hypothesis_target: Optional[str] = None
        if question_records:
            hypothesis_target = question_records[0].hypothesis_target
        elif plans:
            hypothesis_target = plans[0].hypothesis_target
        elif episodes:
            hypothesis_target = episodes[0].hypothesis_target
        if hypothesis_target is None:
            return []

        # Group records by subject, preserving discovery order
        by_subject: Dict[str, List] = {}
        for rec in question_records:
            by_subject.setdefault(rec.subject, []).append(rec)

        type12_subjects: List[str] = list(dict.fromkeys(by_subject.keys()))
        # type4 subjects from plans (new path) or episodes (legacy path)
        if plans:
            type4_subjects: List[str] = list(dict.fromkeys(p.subject for p in plans))
        else:
            type4_subjects = list(dict.fromkeys(ep.subject for ep in episodes))

        effective_source_names: Optional[List[str]] = None
        if source_names:
            effective_source_names = source_names
        elif plans:
            seen: List[str] = []
            for p in plans:
                for name in (p.source_a, p.source_b):
                    if name not in seen:
                        seen.append(name)
            effective_source_names = seen or None
        elif episodes:
            seen = []
            for ep in episodes:
                for name in (ep.source_a, ep.source_b):
                    if name not in seen:
                        seen.append(name)
            effective_source_names = seen or None

        # Source priority map: index 0 = highest priority (Fix 8a).
        source_priority: Dict[str, int] = {}
        if effective_source_names:
            source_priority = {name: i for i, name in enumerate(effective_source_names)}

        pretext = self._compose_pretext(graph, source_names=effective_source_names)

        turns: List[Dict] = []
        question_types_present: set = set()
        global_event_counter = 0

        subject_cutoffs: Dict[str, int] = {}
        subject_extras: Dict[str, Optional[list]] = {}
        subject_bypass_rules: Dict[str, Optional[list]] = {}

        # Conversation-level claimed_rules tracker for Fix 8a.
        # Maps (p_i, polarity) → source_name of the highest-priority claimer so far.
        claimed_rules: Dict[tuple, str] = {}

        # Reserve turns for Phase 2.  For the new plan-based path, compute exact
        # turns needed (2 hypothesis + 4 per event per plan).  For legacy episodes,
        # fall back to the old estimate of 10 per subject.
        if plans:
            _type4_reserve = sum(2 + len(p.events) * 4 for p in plans)
        else:
            _type4_reserve = len(type4_subjects) * 10 if type4_subjects else 0

        def _turn_budget_ok() -> bool:
            if max_total_turns is None:
                return True
            return len(turns) < max_total_turns - _type4_reserve

        def get_b_extras(serial: float, subj_episodes: list) -> Optional[list]:
            extras = []
            for ep in subj_episodes:
                if ep.scenario_type == "bypass":
                    if ep.b_serial <= serial:
                        extras.append((ep.b_type, ep.p_i))
                else:
                    if ep.b_serial <= serial < ep.a_serial:
                        extras.append((ep.b_type, ep.p_i))
            return extras or None

        def get_bypass_rules(serial: float, subj_episodes: list) -> Optional[list]:
            rules = []
            for ep in subj_episodes:
                if (
                    ep.scenario_type == "bypass"
                    and ep.bypass_source
                    and ep.bypass_target
                    and ep.a_serial <= serial
                ):
                    rules.append((ep.bypass_source, ep.bypass_target))
            return rules or None

        # ------------------------------------------------------------------
        # Phase 1: Type 1/2 events, one subject block at a time
        # ------------------------------------------------------------------

        # Maps pi_add_type → defeasible edge type used in extra_edges lists.
        _T1_ETYPE = {
            "pos":       "defeasible_has_property",
            "neg":       "defeasible_lacks_property",
            "exception": "defeasible_lacks_property",  # same underlying edge, different NL
        }

        first_turn = True
        for subject in type12_subjects:
            if not _turn_budget_ok():
                break

            all_recs = by_subject.get(subject, [])
            if not all_recs:
                continue

            # Decide reveal mode for this subject section.
            is_random_mode = random.random() < reveal_order_random_prob

            # Separate by type.
            t1_recs_sorted = sorted(
                [r for r in all_recs if r.question_type == 1],
                key=lambda r: r.pi_add_serial,
            )
            t2_recs = [r for r in all_recs if r.question_type == 2]
            t3_recs = sorted(
                [r for r in all_recs if r.question_type == 3],
                key=lambda r: r.pi_add_serial,
            )

            if is_random_mode and t1_recs_sorted:
                # Shuffle T1 records; T2 interleaving still random (Fix 2).
                t1_for_merge = list(t1_recs_sorted)
                random.shuffle(t1_for_merge)
            else:
                t1_for_merge = t1_recs_sorted

            merged = _random_interleave(t1_for_merge, t2_recs) + t3_recs

            if not merged:
                continue

            hypothesis_statement = self._make_hyp_statement(graph, subject, hypothesis_target)
            subject_label = self._get_node_label(graph, subject)
            initial_state = resolver.resolve(graph, subject, hypothesis_target, serial_cutoff=0)
            initial_mode_b = _STATE_TO_MODE_B.get(initial_state, "unknown")
            _hyp_q = (
                f"Given all the information so far about {subject_label}, "
                "does the information support the latest hypothesis?"
            )

            if first_turn:
                turns.append({
                    "role": "user",
                    "content": (
                        f"Pretext: {pretext} "
                        f"Hypothesis: {hypothesis_statement} "
                        f"Question: {_hyp_q}"
                    ),
                    "subject": subject_label,
                    "reveal_mode": "random" if is_random_mode else "serial",
                })
                first_turn = False
            else:
                turns.append({
                    "role": "user",
                    "content": (
                        f"We now turn to a different object. "
                        f"New hypothesis: {hypothesis_statement} "
                        f"Question: {_hyp_q}"
                    ),
                    "task_type": "hypothesis_change",
                    "new_subject": subject_label,
                    "subject": subject_label,
                    "reveal_mode": "random" if is_random_mode else "serial",
                })
            turns.append({
                "role": "assistant",
                "content": "",
                "ground_truth": initial_mode_b,
                "task_type": "initial_answer",
                "subject": subject_label,
            })

            prev_cutoff = subject_cutoffs.get(subject, 0)
            prev_extras = subject_extras.get(subject, None)
            prev_bypass_rules = subject_bypass_rules.get(subject, None)

            # Random-mode: track which T1 edges have been explicitly revealed.
            # Each entry is (edge_type_str, chain_node_id) — the extra_edges format.
            revealed_t1_edges: List = []

            subject_chain = resolver.extract_chain(graph, subject, hypothesis_target)
            chain_idx_map = {n: i for i, n in enumerate(subject_chain)}
            _chain_len = len(subject_chain)

            for rec in merged:
                if not _turn_budget_ok():
                    break

                is_t1 = rec.question_type == 1
                q_type = rec.question_type
                question_types_present.add(q_type)
                global_event_counter += 1

                render_neg: Optional[set] = None  # negative nodes revealed before this record

                if is_t1 and is_random_mode:
                    # --- Random-mode resolver calls ---
                    # state_before: backbone + edges revealed so far (before this one)
                    edges_before = revealed_t1_edges  # snapshot before adding current
                    state_before = resolver.resolve(
                        graph, subject, hypothesis_target,
                        serial_cutoff=0,
                        extra_edges=edges_before if edges_before else None,
                        bypass_rules=prev_bypass_rules,
                    )
                    # Add this edge to the revealed set, then compute state_after.
                    etype_reveal = _T1_ETYPE.get(rec.pi_add_type, "defeasible_has_property")
                    revealed_t1_edges = revealed_t1_edges + [(etype_reveal, rec.pi_add_node)]
                    state_after = resolver.resolve(
                        graph, subject, hypothesis_target,
                        serial_cutoff=0,
                        extra_edges=revealed_t1_edges,
                    )
                    mode_b = _STATE_TO_MODE_B.get(state_after, "unknown")
                    mode_a = _random_mode_mode_a(
                        rec.pi_add_type, rec.pi_add_node,
                        subject_chain, edges_before,
                        state_before, state_after,
                    ).lower()
                    # Collect negatives revealed before this record for phrasing check.
                    render_neg = {
                        nid for etype, nid in edges_before
                        if etype == "defeasible_lacks_property"
                    }
                    # prev_cutoff stays unchanged in random mode for T1

                elif is_t1:
                    # --- Serial-mode resolver calls (original behaviour) ---
                    curr_int = int(rec.pi_add_serial)
                    old_cutoff = prev_cutoff  # capture before advancing
                    state_before = resolver.resolve(
                        graph, subject, hypothesis_target,
                        serial_cutoff=prev_cutoff, extra_edges=prev_extras,
                        bypass_rules=prev_bypass_rules,
                    )
                    state_after = resolver.resolve(
                        graph, subject, hypothesis_target,
                        serial_cutoff=curr_int, extra_edges=None,
                    )
                    mode_b = _STATE_TO_MODE_B.get(state_after, "unknown")
                    mode_a = _derive_mode_a(state_before, state_after, rec.pi_add_type).lower()
                    prev_cutoff = curr_int
                    # Collect negatives revealed before this record for phrasing check.
                    render_neg = {
                        v for _, v, d in graph.out_edges(rec.subject, data=True)
                        if d.get("type") == "defeasible_lacks_property"
                        and d.get("serial", 0) <= old_cutoff
                    }

                else:
                    # T2: irrelevant attributes never change the hypothesis state.
                    # mode_b = state after all T1 reveals so far (unchanged by this T2).
                    # In random mode, state is tracked via revealed_t1_edges (prev_cutoff
                    # stays 0 and prev_extras is unused); in serial mode, prev_cutoff
                    # and prev_extras capture the same information.
                    if rec.question_type == 2:
                        if is_random_mode:
                            curr_state = resolver.resolve(
                                graph, subject, hypothesis_target,
                                serial_cutoff=0,
                                extra_edges=revealed_t1_edges or None,
                                bypass_rules=prev_bypass_rules,
                            )
                        else:
                            curr_state = resolver.resolve(
                                graph, subject, hypothesis_target,
                                serial_cutoff=prev_cutoff,
                                extra_edges=prev_extras,
                                bypass_rules=prev_bypass_rules,
                            )
                        mode_b = _STATE_TO_MODE_B.get(curr_state, "unknown")
                        mode_a = "no_effect"
                        precomputed = rec.mode_b_label.lower()
                        if mode_b.lower() != precomputed:
                            logger.warning(
                                "T2 mode_b mismatch: subject=%s node=%s "
                                "resolver=%s precomputed=%s; using resolver.",
                                rec.subject, rec.pi_add_node, mode_b, precomputed,
                            )
                    else:
                        # T3: use pre-computed labels.
                        mode_b = rec.mode_b_label.lower()
                        mode_a = rec.mode_a_label.lower()

                update_sentence = self._render_update_sentence(graph, rec, render_neg)

                _node_idx = chain_idx_map.get(rec.pi_add_node, -1)
                _chain_dist = (_chain_len - 1 - _node_idx) if _node_idx >= 0 else None

                _entailment_q = (
                    f"Given all the information so far about {subject_label}, "
                    "does it support the latest hypothesis?"
                )
                turns.append({
                    "role": "user",
                    "content": (
                        f"Information update: {update_sentence} "
                        f"Question: {_entailment_q}"
                    ),
                    "serial": global_event_counter,
                    "update_type": rec.pi_add_type,
                    "question_type": q_type,
                    "reveal_mode": "random" if is_random_mode else "serial",
                    "subject": subject_label,
                    "hyp_node_distance": _chain_dist,
                })
                turns.append({
                    "role": "assistant",
                    "content": "",
                    "ground_truth": mode_b,
                    "effect": mode_a,
                    "task_type": "update_answer",
                    "subject": subject_label,
                })
                turns.append({"role": "user", "content": f"Question: {_EFFECT_QUESTION}", "subject": subject_label})
                turns.append({
                    "role": "assistant",
                    "content": "",
                    "ground_truth": mode_a,
                    "task_type": "effect_of_update",
                    "subject": subject_label,
                })

                prev_extras = None
                prev_bypass_rules = None

            subject_cutoffs[subject] = prev_cutoff
            subject_extras[subject] = prev_extras
            subject_bypass_rules[subject] = prev_bypass_rules

        # ------------------------------------------------------------------
        # Phase 2: Type 4 — plan-based (new) or episode-based (legacy)
        # ------------------------------------------------------------------
        if plans:
            self._render_type4_plans(
                graph, plans, resolver, turns, question_types_present,
                first_turn, pretext, max_total_turns,
            )
            # The inner method mutates `turns` and `question_types_present`.
            # `first_turn` is only relevant if we had zero Type 1/2 subjects,
            # in which case _render_type4_plans handles it internally.
        else:
            # Legacy episode-based Phase 2 follows below.
            pass

        if not plans:
            for subject in type4_subjects:
                subj_episodes = [ep for ep in episodes if ep.subject == subject]
                if not subj_episodes:
                    continue

                question_types_present.add(4)

                hypothesis_statement = self._make_hyp_statement(graph, subject, hypothesis_target)
                subject_label = self._get_node_label(graph, subject)
                _leg_chain = resolver.extract_chain(graph, subject, hypothesis_target)
                _leg_chain_idx = {n: i for i, n in enumerate(_leg_chain)}
                _leg_chain_len = len(_leg_chain)
                acc_cutoff = subject_cutoffs.get(subject, 0)
                initial_state = resolver.resolve(
                    graph, subject, hypothesis_target, serial_cutoff=acc_cutoff
                )
                initial_mode_b = _STATE_TO_MODE_B.get(initial_state, "unknown")
                _hyp_q = (
                    f"Given all the information so far about {subject_label}, "
                    "does the information support the latest hypothesis?"
                )

                if first_turn:
                    turns.append({
                        "role": "user",
                        "content": (
                            f"Pretext: {pretext} "
                            f"Hypothesis: {hypothesis_statement} "
                            f"Question: {_hyp_q}"
                        ),
                        "subject": subject_label,
                    })
                    first_turn = False
                else:
                    turns.append({
                        "role": "user",
                        "content": (
                            f"We now turn to a different object. "
                            f"New hypothesis: {hypothesis_statement} "
                            f"Question: {_hyp_q}"
                        ),
                        "task_type": "hypothesis_change",
                        "new_subject": subject_label,
                        "subject": subject_label,
                    })
                turns.append({
                    "role": "assistant",
                    "content": "",
                    "ground_truth": initial_mode_b,
                    "task_type": "initial_answer",
                    "subject": subject_label,
                })

                t4_events: List = []
                for ep in subj_episodes:
                    t4_events.append((ep.b_serial, "type4_b", ep))
                    t4_events.append((ep.a_serial, "type4_a", ep))
                t4_events.sort(key=lambda e: e[0])

                prev_cutoff = acc_cutoff
                prev_extras = subject_extras.get(subject, None)
                prev_bypass_rules = subject_bypass_rules.get(subject, None)
                prev_mode_b = initial_mode_b

                _TURNS_PER_EPISODE = 8
                in_progress_episodes: set = set()
                skipped_episodes: set = set()

                for serial, kind, payload in t4_events:
                    ep_id = id(payload)
                    if ep_id in skipped_episodes:
                        continue
                    if ep_id not in in_progress_episodes:
                        if (
                            max_total_turns is not None
                            and len(turns) + _TURNS_PER_EPISODE > max_total_turns
                        ):
                            skipped_episodes.add(ep_id)
                            continue
                        in_progress_episodes.add(ep_id)

                    curr_int = int(serial)
                    extras_after = get_b_extras(serial, subj_episodes)
                    bypass_rules_after = get_bypass_rules(serial, subj_episodes)

                    state_before = resolver.resolve(
                        graph, subject, hypothesis_target,
                        serial_cutoff=prev_cutoff,
                        extra_edges=prev_extras,
                        bypass_rules=prev_bypass_rules,
                    )
                    state_after = resolver.resolve(
                        graph, subject, hypothesis_target,
                        serial_cutoff=curr_int,
                        extra_edges=extras_after,
                        bypass_rules=bypass_rules_after,
                    )

                    global_event_counter += 1

                    if kind == "type4_b":
                        source_name = payload.source_b
                        polarity = "lacks" if payload.b_type == "defeasible_lacks_property" else "has"
                        claim_key = (payload.p_i, polarity)
                        mode_a, mode_b = self._check_redundant_claim(
                            claim_key, source_name, source_priority, claimed_rules,
                            state_before, state_after, payload, prev_mode_b,
                        )
                        if mode_a is None:
                            mode_b = _STATE_TO_MODE_B.get(state_after, "unknown")
                            if payload.scenario == "a_first":
                                mode_a = "no_effect"
                            else:
                                pi_type = "neg" if payload.b_type == "defeasible_lacks_property" else "pos"
                                mode_a = _derive_mode_a(state_before, state_after, pi_type).lower()
                            self._register_claim(claim_key, source_name, source_priority, claimed_rules)
                        claim_sentence = self._render_type4_b_claim(graph, payload)
                        update_sentence = f"{payload.source_b} reports: {claim_sentence}"
                        update_type = "type4_b_claim"
                        source_field = payload.source_b

                    else:
                        source_name = payload.source_a
                        polarity = "has" if payload.a_type == "defeasible_has_property" else "lacks"
                        claim_key_a = (payload.p_i, polarity)
                        mode_a, mode_b = self._check_redundant_claim(
                            claim_key_a, source_name, source_priority, claimed_rules,
                            state_before, state_after, payload, prev_mode_b,
                        )
                        if mode_a is None:
                            mode_b = _STATE_TO_MODE_B.get(state_after, "unknown")
                            if payload.scenario_type == "bypass":
                                mode_a = "strengthening"
                            else:
                                pi_type = "neg" if payload.a_type == "defeasible_lacks_property" else "pos"
                                mode_a = _derive_mode_a(state_before, state_after, pi_type).lower()
                            self._register_claim(claim_key_a, source_name, source_priority, claimed_rules)
                        claim_sentence = self._render_type4_a_claim(graph, payload)
                        update_sentence = f"{payload.source_a} reports: {claim_sentence}"
                        update_type = "type4_a_claim"
                        source_field = payload.source_a

                    _ep_node_idx = _leg_chain_idx.get(payload.p_i, -1)
                    _entailment_q = (
                        f"Given all the information so far about {subject_label}, "
                        "does it support the latest hypothesis?"
                    )
                    turns.append({
                        "role": "user",
                        "content": (
                            f"Information update: {update_sentence} "
                            f"Question: {_entailment_q}"
                        ),
                        "serial": global_event_counter,
                        "update_type": update_type,
                        "question_type": 4,
                        "source": source_field,
                        "type4_scenario": payload.scenario,
                        "type4_scenario_type": payload.scenario_type,
                        "subject": subject_label,
                        "hyp_node_distance": (_leg_chain_len - 1 - _ep_node_idx) if _ep_node_idx >= 0 else None,
                    })
                    turns.append({
                        "role": "assistant",
                        "content": "",
                        "ground_truth": mode_b,
                        "effect": mode_a,
                        "task_type": "update_answer",
                        "subject": subject_label,
                    })
                    turns.append({"role": "user", "content": f"Question: {_EFFECT_QUESTION}", "subject": subject_label})
                    turns.append({
                        "role": "assistant",
                        "content": "",
                        "ground_truth": mode_a,
                        "task_type": "effect_of_update",
                        "subject": subject_label,
                    })

                    prev_cutoff = curr_int
                    prev_extras = extras_after
                    prev_bypass_rules = bypass_rules_after
                    prev_mode_b = mode_b

        # ------------------------------------------------------------------
        # Build output
        # ------------------------------------------------------------------
        all_subjects_ordered = list(dict.fromkeys(type12_subjects + type4_subjects))

        if plans:
            t4_meta = [
                {
                    "subject": self._get_node_label(graph, p.subject),
                    "scenario_type": p.scenario_type,
                    "ordering": p.ordering,
                    "source_a": p.source_a,
                    "source_b": p.source_b,
                    **(
                        {"contested_node": self._get_node_label(graph, p.contested_node),
                         "claim_level": p.claim_level}
                        if p.scenario_type == "contradiction" else {}
                    ),
                    **(
                        {"b_block_src": self._get_node_label(graph, p.b_block_src),
                         "b_block_tgt": self._get_node_label(graph, p.b_block_tgt),
                         "a_bypass_src": self._get_node_label(graph, p.a_bypass_src),
                         "a_bypass_tgt": self._get_node_label(graph, p.a_bypass_tgt)}
                        if p.scenario_type == "bypass" else {}
                    ),
                }
                for p in plans
            ]
        else:
            t4_meta = [
                {
                    "subject": self._get_node_label(graph, ep.subject),
                    "scenario": ep.scenario,
                    "scenario_type": ep.scenario_type,
                    "source_a": ep.source_a,
                    "source_b": ep.source_b,
                    "p_i": self._get_node_label(graph, ep.p_i),
                    "b_claim_level": ep.b_claim_level,
                    **(
                        {
                            "bypass_source": self._get_node_label(graph, ep.bypass_source),
                            "bypass_target": self._get_node_label(graph, ep.bypass_target),
                        }
                        if ep.scenario_type == "bypass" and ep.bypass_source and ep.bypass_target
                        else {}
                    ),
                }
                for ep in episodes
            ]

        return [{
            "id": f"{hypothesis_target}_conversation",
            "topology": "default",
            "metadata": {
                "subjects": [self._get_node_label(graph, s) for s in all_subjects_ordered],
                "hypothesis_target": self._get_node_label(graph, hypothesis_target),
                "question_types_present": sorted(question_types_present),
                **({"type4_plans": t4_meta} if t4_meta else {}),
            },
            "turns": turns,
        }]

    # ------------------------------------------------------------------
    # Plan-based Type 4 Phase 2 renderer
    # ------------------------------------------------------------------

    def _render_type4_plans(
        self,
        graph,
        plans: List,
        resolver,
        turns: List[Dict],
        question_types_present: set,
        first_turn_flag: bool,
        pretext: str,
        max_total_turns: Optional[int],
    ) -> None:
        """
        Render Type4ConversationPlan objects into `turns` (mutates in place).

        Resolver state per subject:
          - serial_cutoff=0  (no real graph edges visible by default)
          - synth_extra      synthetic (etype, node) tuples from source claims
          - t1_revealed      revealed real graph edge (etype, node) tuples
          - bypass_rules     bypass (src, tgt) tuples from A's bypass claims

        Effect computation:
          - Source claim by B (b_first contradiction/bypass): state-delta rule.
          - Source claim by A (contradiction reinstatement): state-delta rule
            (usually STRENGTHENING).
          - Source claim by A (bypass): always STRENGTHENING.
          - Source claim by B (a_first contradiction): NO_EFFECT — A already
            holds the higher-authority positive at the same node; B's lower-
            priority negative is not applied to the resolver state.
          - Source claim by B (a_first bypass): state-delta (B's block is applied
            to the resolver, but A's bypass rule may already neutralise it →
            state may not change → NO_EFFECT via state-delta).
          - Edge reveal: random-mode logic (positive edge: check clear path;
            negative edge: state-delta).
        """
        first_turn = first_turn_flag

        for plan in plans:
            subject = plan.subject
            hypothesis_target = plan.hypothesis_target

            subject_chain = resolver.extract_chain(graph, subject, hypothesis_target)
            chain_idx_map = {n: i for i, n in enumerate(subject_chain)}
            _chain_len = len(subject_chain)

            # --- Hypothesis intro ---
            hypothesis_statement = self._make_hyp_statement(graph, subject, hypothesis_target)
            subject_label = self._get_node_label(graph, subject)
            initial_state = resolver.resolve(
                graph, subject, hypothesis_target, serial_cutoff=0
            )
            initial_mode_b = _STATE_TO_MODE_B.get(initial_state, "unknown")
            _hyp_q = (
                f"Given all the information so far about {subject_label}, "
                "does the information support the latest hypothesis?"
            )

            if first_turn:
                turns.append({
                    "role": "user",
                    "content": (
                        f"Pretext: {pretext} "
                        f"Hypothesis: {hypothesis_statement} "
                        f"Question: {_hyp_q}"
                    ),
                    "task_type": "hypothesis_intro",
                    "question_type": 4,
                    "subject": subject_label,
                })
                first_turn = False
            else:
                turns.append({
                    "role": "user",
                    "content": (
                        f"We now turn to a different object. "
                        f"New hypothesis: {hypothesis_statement} "
                        f"Question: {_hyp_q}"
                    ),
                    "task_type": "hypothesis_change",
                    "question_type": 4,
                    "new_subject": subject_label,
                    "subject": subject_label,
                })
            turns.append({
                "role": "assistant",
                "content": "",
                "ground_truth": initial_mode_b,
                "task_type": "initial_answer",
                "subject": subject_label,
            })

            question_types_present.add(4)

            # Explicit evidence state for the resolver (serial_cutoff stays 0)
            synth_extra: List = []   # source-claim synthetic edges
            t1_revealed: List = []   # graph T1 edge reveals
            bypass_rules: List = []  # A's bypass rules

            # Track whether A has already claimed positive at the contested node
            # (needed to decide if B's later claim has any resolver effect).
            a_claimed_pos_nodes: set = set()

            for event in plan.events:
                combined_extra = (synth_extra + t1_revealed) or None
                br = bypass_rules or None
                state_before = resolver.resolve(
                    graph, subject, hypothesis_target,
                    serial_cutoff=0,
                    extra_edges=combined_extra,
                    bypass_rules=br,
                )

                if event.event_type == "source_claim":
                    is_a = event.is_source_a
                    scenario = plan.scenario_type
                    ordering = plan.ordering

                    # Determine whether B's claim is overridden by A's higher
                    # authority (only in a_first contradiction at the same node).
                    b_overridden = (
                        not is_a
                        and scenario == "contradiction"
                        and ordering == "a_first"
                        and event.claim_node in a_claimed_pos_nodes
                    )

                    if b_overridden:
                        mode_a = "no_effect"
                        mode_b = _STATE_TO_MODE_B.get(state_before, "unknown")
                    elif is_a:
                        if scenario == "bypass":
                            # A introduces the bypass rule
                            bypass_rules = bypass_rules + [(event.bypass_src, event.bypass_tgt)]
                        else:
                            # Contradiction: remove B's negative (if present), add A's positive
                            synth_extra = [
                                (et, nd) for et, nd in synth_extra
                                if not (
                                    et == "defeasible_lacks_property"
                                    and nd == event.claim_node
                                )
                            ]
                            synth_extra = synth_extra + [
                                ("defeasible_has_property", event.claim_node)
                            ]
                        a_claimed_pos_nodes.add(event.claim_node)

                        combined_extra = (synth_extra + t1_revealed) or None
                        br = bypass_rules or None
                        state_after = resolver.resolve(
                            graph, subject, hypothesis_target,
                            serial_cutoff=0,
                            extra_edges=combined_extra,
                            bypass_rules=br,
                        )
                        mode_b = _STATE_TO_MODE_B.get(state_after, "unknown")
                        if scenario == "bypass":
                            mode_a = "strengthening"
                        elif ordering == "a_first":
                            # A speaks before any negatives are in scope → direct
                            # positive evidence always strengthens, even if backbone
                            # was already ENTAILED (same rule as T1 positive edges).
                            mode_a = "strengthening"
                        else:
                            # b_first: A reinstates after B's negative; use state-delta
                            # which correctly returns STRENGTHENING (DEFEATED→ENTAILED)
                            # or NO_EFFECT if another blocker was already on the chain.
                            mode_a = _state_delta_mode_a(state_before, state_after).lower()
                    else:
                        # B's claim — add synthetic negative
                        synth_extra = synth_extra + [
                            ("defeasible_lacks_property", event.claim_node)
                        ]
                        combined_extra = (synth_extra + t1_revealed) or None
                        br = bypass_rules or None
                        state_after = resolver.resolve(
                            graph, subject, hypothesis_target,
                            serial_cutoff=0,
                            extra_edges=combined_extra,
                            bypass_rules=br,
                        )
                        mode_b = _STATE_TO_MODE_B.get(state_after, "unknown")
                        mode_a = _state_delta_mode_a(state_before, state_after).lower()

                    update_sentence = (
                        f"{event.source} reports: "
                        + self._render_t4_plan_claim(graph, plan, event)
                    )
                    _node_idx = chain_idx_map.get(event.claim_node, -1)
                    update_meta = {
                        "update_type": "type4_a_claim" if is_a else "type4_b_claim",
                        "source": event.source,
                        "type4_scenario_type": scenario,
                        "type4_ordering": ordering,
                        "question_type": 4,
                        "hyp_node_distance": (_chain_len - 1 - _node_idx) if _node_idx >= 0 else None,
                    }

                else:  # edge_reveal
                    etype = (
                        "defeasible_has_property"
                        if event.edge_kind == "pos"
                        else "defeasible_lacks_property"
                    )
                    prior_evidence = synth_extra + t1_revealed
                    t1_revealed = t1_revealed + [(etype, event.edge_node)]
                    combined_extra = (synth_extra + t1_revealed) or None
                    br = bypass_rules or None
                    state_after = resolver.resolve(
                        graph, subject, hypothesis_target,
                        serial_cutoff=0,
                        extra_edges=combined_extra,
                        bypass_rules=br,
                    )
                    mode_b = _STATE_TO_MODE_B.get(state_after, "unknown")
                    if event.edge_kind == "pos":
                        mode_a = _compute_pos_mode_a(
                            subject_chain, event.edge_node,
                            prior_evidence, state_before, state_after,
                        ).lower()
                    else:
                        mode_a = _state_delta_mode_a(state_before, state_after).lower()

                    update_sentence = self._render_t4_edge_reveal(graph, event)
                    _node_idx = chain_idx_map.get(event.edge_node, -1)
                    update_meta = {
                        "update_type": "type4_edge_reveal",
                        "question_type": 4,
                        "edge_kind": event.edge_kind,
                        "hyp_node_distance": (_chain_len - 1 - _node_idx) if _node_idx >= 0 else None,
                    }

                # --- Append 4-turn pattern ---
                _entailment_q = (
                    f"Given all the information so far about {subject_label}, "
                    "does it support the latest hypothesis?"
                )
                turns.append({
                    "role": "user",
                    "content": (
                        f"Information update: {update_sentence} "
                        f"Question: {_entailment_q}"
                    ),
                    "subject": subject_label,
                    **update_meta,
                })
                turns.append({
                    "role": "assistant",
                    "content": "",
                    "ground_truth": mode_b,
                    "effect": mode_a,
                    "task_type": "update_answer",
                    "subject": subject_label,
                })
                turns.append({"role": "user", "content": f"Question: {_EFFECT_QUESTION}", "subject": subject_label})
                turns.append({
                    "role": "assistant",
                    "content": "",
                    "ground_truth": mode_a,
                    "task_type": "effect_of_update",
                    "subject": subject_label,
                })

    def _render_t4_plan_claim(self, graph, plan, event) -> str:
        """NL sentence for a source_claim event in a Type4ConversationPlan."""
        subject = plan.subject
        subject_label = self._get_node_label(graph, subject)

        if plan.scenario_type == "bypass":
            # Always rule-level; bypass_src/bypass_tgt set on both B and A events
            src_label = self._get_node_label(graph, event.bypass_src)
            tgt_pred = self._get_property_predicate_plural(graph, event.bypass_tgt)
            if event.is_source_a:
                return f"objects with property {src_label} generally {tgt_pred}."
            else:
                negated = self._negate_rule_predicate(tgt_pred)
                return f"objects with property {src_label} generally {negated}."

        # Contradiction — always object-level
        pred = self._get_property_predicate(graph, event.claim_node)
        if event.claim_polarity == "neg":
            return self._negate_single(subject_label, pred)
        return f"{subject_label.capitalize()} {pred}."

    def _render_t4_edge_reveal(self, graph, event) -> str:
        """NL sentence for an edge_reveal event in a Type4ConversationPlan."""
        etype = (
            "defeasible_has_property"
            if event.edge_kind == "pos"
            else "defeasible_lacks_property"
        )
        return self._get_edge_nl(graph, event.edge_subject, event.edge_node, etype)

    # ------------------------------------------------------------------
    # Redundant-confirmation helpers (Fix 8a)
    # ------------------------------------------------------------------

    def _check_redundant_claim(
        self,
        claim_key: tuple,
        source_name: str,
        source_priority: Dict[str, int],
        claimed_rules: Dict[tuple, str],
        state_before: str,
        state_after: str,
        payload,
        prev_mode_b: str,
    ):
        """
        Return (mode_a, mode_b) if this claim is redundant, else (None, None).

        Redundant = a higher-priority source has already claimed the same
        (p_i, polarity) edge.  In that case the new claim adds no new
        information and the effect is NO_EFFECT.
        """
        prior_source = claimed_rules.get(claim_key)
        if prior_source is None:
            return None, None

        current_priority = source_priority.get(source_name, 999)
        prior_priority = source_priority.get(prior_source, 999)

        if prior_priority < current_priority:
            # Higher-priority source already holds this claim → NO_EFFECT.
            return "no_effect", prev_mode_b

        return None, None

    @staticmethod
    def _register_claim(
        claim_key: tuple,
        source_name: str,
        source_priority: Dict[str, int],
        claimed_rules: Dict[tuple, str],
    ) -> None:
        """Record this claim if it's the first or a higher-priority assertion."""
        prior = claimed_rules.get(claim_key)
        if prior is None or source_priority.get(source_name, 999) < source_priority.get(prior, 999):
            claimed_rules[claim_key] = source_name

    # ------------------------------------------------------------------
    # NL rendering helpers
    # ------------------------------------------------------------------

    def _find_chain_parent_node(self, graph, subject: str, p_i: str) -> Optional[str]:
        """Return the chain node immediately before p_i, or None if p_i is at index 0."""
        from path_relation_module.default_resolver import DefaultChainResolver
        hyp_target = None
        for _u, v, data in graph.out_edges(subject, data=True):
            if data.get("type") == "hypothesis_edge":
                hyp_target = v
                break
        if hyp_target is None:
            return None
        chain = DefaultChainResolver().extract_chain(graph, subject, hyp_target)
        try:
            idx = chain.index(p_i)
        except ValueError:
            return None
        return chain[idx - 1] if idx > 0 else None

    def _render_update_sentence(
        self, graph, record, revealed_neg_nodes: Optional[set] = None
    ) -> str:
        pi_type = record.pi_add_type
        subject = record.subject
        p_i = record.pi_add_node
        subject_label = self._get_node_label(graph, subject)

        if pi_type == "pos":
            return self._get_edge_nl(graph, subject, p_i, "defeasible_has_property")

        if pi_type == "neg":
            return self._get_edge_nl(graph, subject, p_i, "defeasible_lacks_property")

        if pi_type == "exception":
            # Use exception phrasing only when the immediately preceding chain node
            # has NOT been shown as negative.  If it has been revealed as negative
            # (e.g. "X is not erosive" appeared earlier in this conversation),
            # saying "X is an exception to the rule that erosive objects are Y"
            # is contradictory; fall back to the plain negative phrasing instead.
            use_exception = True
            if revealed_neg_nodes:
                parent_node = self._find_chain_parent_node(graph, subject, p_i)
                if parent_node is not None and parent_node in revealed_neg_nodes:
                    use_exception = False
            if use_exception:
                parent_label = self._find_chain_parent_label(graph, subject, p_i)
                p_i_predicate_plural = self._get_property_predicate_plural(graph, p_i)
                return (
                    f"{subject_label} is an exception to the rule that "
                    f"{parent_label} objects {p_i_predicate_plural}."
                )
            # Fallback: plain "X does not have property Y" sentence.
            return self._get_edge_nl(graph, subject, p_i, "defeasible_lacks_property")

        if pi_type in {"safe_irr", "trap_irr"}:
            return self._get_edge_nl(graph, subject, p_i, "has_attribute")

        if pi_type == "quantified_inst":
            scope_set = record.extra.get("scope_set", [])
            scope_labels = [self._get_node_label(graph, n) for n in scope_set]
            p_i_predicate = self._get_property_predicate(graph, p_i)
            joined = self._join_labels(scope_labels)
            return self._negate_predicate_plural(joined, p_i_predicate)

        if pi_type == "quantified_gen":
            p_root = record.extra.get("p_root", "")
            p_root_label = self._get_node_label(graph, p_root) if p_root else "object"
            p_i_predicate = self._get_property_predicate(graph, p_i)
            return self._negate_predicate_generic(p_root_label, p_i_predicate)

        return f"Update regarding {subject_label} and {self._get_node_label(graph, p_i)}."

    def _render_type4_b_claim(self, graph, episode) -> str:
        if episode.b_claim_level == "rule":
            return self._render_rule_level_denial(graph, episode.subject, episode.p_i)
        subject_label = self._get_node_label(graph, episode.subject)
        predicate = self._get_property_predicate(graph, episode.p_i)
        return self._negate_single(subject_label, predicate)

    def _render_type4_a_claim(self, graph, episode) -> str:
        if episode.scenario_type == "bypass" and episode.bypass_source and episode.bypass_target:
            src_label = self._get_node_label(graph, episode.bypass_source)
            tgt_predicate = self._get_property_predicate(graph, episode.bypass_target)
            return f"objects with property {src_label} generally {tgt_predicate}."
        return self._render_rule_level_confirmation(graph, episode.subject, episode.p_i)

    def _render_rule_level_denial(self, graph, subject: str, p_i: str) -> str:
        parent_label = self._find_chain_parent_label(graph, subject, p_i)
        predicate = self._get_property_predicate(graph, p_i)
        negated = self._negate_rule_predicate(predicate)
        return f"objects with property {parent_label} generally {negated}."

    def _render_rule_level_confirmation(self, graph, subject: str, p_i: str) -> str:
        parent_label = self._find_chain_parent_label(graph, subject, p_i)
        predicate = self._get_property_predicate(graph, p_i)
        return f"objects with property {parent_label} generally {predicate}."

    def _negate_rule_predicate(self, predicate: str) -> str:
        if predicate.startswith("is "):
            return f"are not {predicate[3:]}"
        if predicate.startswith("are "):
            return f"are not {predicate[4:]}"
        if predicate.startswith("has "):
            return f"do not have {predicate[4:]}"
        if predicate.startswith("have "):
            return f"do not have {predicate[5:]}"
        if predicate.startswith("can "):
            return f"cannot {predicate[4:]}"
        if predicate.startswith("serves as "):
            return f"do not serve as {predicate[10:]}"
        if predicate.startswith("appears "):
            return f"do not appear {predicate[8:]}"
        if predicate.startswith("produces "):
            return f"do not produce {predicate[9:]}"
        if predicate.startswith("thrives "):
            return f"do not thrive {predicate[8:]}"
        if predicate.startswith("displays "):
            return f"do not display {predicate[9:]}"
        return f"do not {predicate}"

    def _render_predicate_sentence(
        self, subject_label: str, predicate: str, edge_type: str
    ) -> str:
        if edge_type == "defeasible_has_property":
            return f"{subject_label.capitalize()} {predicate}."
        return self._negate_single(subject_label, predicate)

    def _negate_single(self, subject_label: str, predicate: str) -> str:
        if predicate.startswith("is "):
            return f"{subject_label} is not {predicate[3:]}."
        if predicate.startswith("has "):
            return f"{subject_label} does not have {predicate[4:]}."
        if predicate.startswith("can "):
            return f"{subject_label} cannot {predicate[4:]}."
        if predicate.startswith("serves as "):
            return f"{subject_label} does not serve as {predicate[10:]}."
        if predicate.startswith("appears "):
            return f"{subject_label} does not appear {predicate[8:]}."
        if predicate.startswith("produces "):
            return f"{subject_label} does not produce {predicate[9:]}."
        if predicate.startswith("tastes or smells "):
            return f"{subject_label} does not taste or smell {predicate[17:]}."
        if predicate.startswith("displays "):
            return f"{subject_label} does not display {predicate[9:]}."
        if predicate.startswith("thrives "):
            return f"{subject_label} does not thrive {predicate[8:]}."
        return f"{subject_label} does not {predicate}."

    def _negate_predicate_plural(self, joined_subjects: str, predicate: str) -> str:
        if predicate.startswith("is "):
            return f"At least one of {joined_subjects} is not {predicate[3:]}."
        if predicate.startswith("has "):
            return f"At least one of {joined_subjects} does not have {predicate[4:]}."
        if predicate.startswith("can "):
            return f"At least one of {joined_subjects} cannot {predicate[4:]}."
        if predicate.startswith("serves as "):
            return f"At least one of {joined_subjects} does not serve as {predicate[10:]}."
        if predicate.startswith("appears "):
            return f"At least one of {joined_subjects} does not appear {predicate[8:]}."
        if predicate.startswith("produces "):
            return f"At least one of {joined_subjects} does not produce {predicate[9:]}."
        if predicate.startswith("tastes or smells "):
            return f"At least one of {joined_subjects} does not taste or smell {predicate[17:]}."
        if predicate.startswith("displays "):
            return f"At least one of {joined_subjects} does not display {predicate[9:]}."
        if predicate.startswith("thrives "):
            return f"At least one of {joined_subjects} does not thrive {predicate[8:]}."
        return f"At least one of {joined_subjects} does not {predicate}."

    def _negate_predicate_generic(self, p_root_label: str, predicate: str) -> str:
        if predicate.startswith("is "):
            return f"At least one {p_root_label} object is not {predicate[3:]}."
        if predicate.startswith("has "):
            return f"At least one {p_root_label} object does not have {predicate[4:]}."
        if predicate.startswith("can "):
            return f"At least one {p_root_label} object cannot {predicate[4:]}."
        if predicate.startswith("serves as "):
            return f"At least one {p_root_label} object does not serve as {predicate[10:]}."
        if predicate.startswith("appears "):
            return f"At least one {p_root_label} object does not appear {predicate[8:]}."
        if predicate.startswith("produces "):
            return f"At least one {p_root_label} object does not produce {predicate[9:]}."
        if predicate.startswith("tastes or smells "):
            return f"At least one {p_root_label} object does not taste or smell {predicate[17:]}."
        if predicate.startswith("displays "):
            return f"At least one {p_root_label} object does not display {predicate[9:]}."
        if predicate.startswith("thrives "):
            return f"At least one {p_root_label} object does not thrive {predicate[8:]}."
        return f"At least one {p_root_label} object does not {predicate}."

    def _get_edge_nl(self, graph, u: str, v: str, expected_type: str) -> str:
        if graph.has_edge(u, v):
            data = graph.edges[u, v]
            if data.get("type") == expected_type:
                nl = data.get("nl_sentence")
                if nl:
                    return nl

        for _u2, v2, data in graph.out_edges(u, data=True):
            if v2 == v and data.get("type") == expected_type:
                nl = data.get("nl_sentence")
                if nl:
                    return nl

        u_label = self._get_node_label(graph, u)
        v_label = self._get_node_label(graph, v)
        if expected_type == "defeasible_has_property":
            return f"{u_label} {self._get_property_predicate(graph, v)}."
        if expected_type == "defeasible_lacks_property":
            return self._negate_single(u_label, self._get_property_predicate(graph, v))
        return f"{u_label} has attribute {v_label}."

    def _find_chain_parent_label(self, graph, subject: str, p_i: str) -> str:
        from path_relation_module.default_resolver import DefaultChainResolver

        hyp_target = None
        for _u, v, data in graph.out_edges(subject, data=True):
            if data.get("type") == "hypothesis_edge":
                hyp_target = v
                break
        if hyp_target is None:
            return "property"

        resolver = DefaultChainResolver()
        chain = resolver.extract_chain(graph, subject, hyp_target)
        if not chain:
            return "property"

        try:
            idx = chain.index(p_i)
        except ValueError:
            return "property"

        if idx == 0:
            return self._get_node_label(graph, chain[0])
        return self._get_node_label(graph, chain[idx - 1])

    # ------------------------------------------------------------------
    # Pretext composition
    # ------------------------------------------------------------------

    def _compose_pretext(
        self,
        graph,
        source_names: Optional[List[str]] = None,
    ) -> str:
        chain_sentences = []
        instantiation_sentences = []

        ordered_edges = sorted(
            graph.edges(data=True),
            key=lambda item: (
                graph.nodes[item[0]].get("layer", 0),
                graph.nodes[item[1]].get("layer", 0),
            ),
        )

        for u, v, data in ordered_edges:
            etype = data.get("type")
            if etype not in {"is_a", "implies", "has_attribute", "has_property"}:
                continue
            if etype == "has_attribute" and graph.nodes[v].get("type") == "irrelevant_property":
                continue
            nl = data.get("nl_sentence")
            if not nl:
                continue
            if etype == "is_a":
                instantiation_sentences.append(nl)
            else:
                chain_sentences.append(nl)

        pretext = " ".join(chain_sentences + instantiation_sentences)

        if source_names and len(source_names) >= 2:
            pretext += " " + self._format_reliability_statement(source_names)

        return pretext

    @staticmethod
    def _format_reliability_statement(source_names: List[str]) -> str:
        import random as _rnd

        pairs = [
            (source_names[i], source_names[i + 1])
            for i in range(len(source_names) - 1)
        ]
        _rnd.shuffle(pairs)

        sentences = []
        for higher, lower in pairs:
            if _rnd.random() < 0.5:
                sentences.append(f"{higher} is more reliable than {lower}")
            else:
                sentences.append(f"{lower} is less reliable than {higher}")

        first = "For reference, " + sentences[0]
        rest = sentences[1:]
        return ". ".join([first] + rest) + "."

    # ------------------------------------------------------------------
    # Hypothesis statement
    # ------------------------------------------------------------------

    def _make_hyp_statement(self, graph, subject: str, hypothesis_target: str) -> str:
        subject_label = self._get_node_label(graph, subject)
        metadata = graph.nodes[hypothesis_target].get("metadata", {})
        predicate = metadata.get(
            "filled_object_template",
            f"have the property '{self._get_node_label(graph, hypothesis_target)}'",
        )
        return f"{subject_label.capitalize()} {predicate}."

    # ------------------------------------------------------------------
    # Node / property label helpers
    # ------------------------------------------------------------------

    def _get_node_label(self, graph, node_id: str) -> str:
        return graph.nodes[node_id].get("label", node_id)

    def _get_property_predicate(self, graph, node_id: str) -> str:
        return graph.nodes[node_id].get("metadata", {}).get(
            "filled_object_template",
            graph.nodes[node_id].get("label", node_id),
        )

    def _get_property_predicate_plural(self, graph, node_id: str) -> str:
        return graph.nodes[node_id].get("metadata", {}).get(
            "filled_template",
            graph.nodes[node_id].get("label", node_id),
        )

    @staticmethod
    def _join_labels(labels: List[str]) -> str:
        if len(labels) == 1:
            return labels[0]
        if len(labels) == 2:
            return f"{labels[0]} or {labels[1]}"
        return ", ".join(labels[:-1]) + f", or {labels[-1]}"
