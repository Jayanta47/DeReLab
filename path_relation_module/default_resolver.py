"""
DefaultChainResolver
====================
Implements the "defaults carry through" principle for a single linear chain
in the default topology. Independent from PMPM / GraphResolutor.

Graph contract (default topology):
  - Entity nodes (type="entity", layer=0) connect to Property_Root via type="is_a".
  - Properties form a linear chain via type="implies" edges:
      P_root → P_1 → ... → P_n  (hypothesis_target is P_n, the deepest child).
  - implies edges go PARENT → CHILD (P_root is the implied parent).
  - Defeasible edges from entity nodes to intermediate chain nodes carry:
      type = "defeasible_has_property"  (positive evidence, confirms property)
      type = "defeasible_lacks_property" (negative evidence, denies property)
      serial: int  (temporal ordering; lower = revealed earlier)

Resolver states: "ENTAILED" | "DEFEATED" | "UNDETERMINED"
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class DefaultChainResolver:
    """
    Resolves whether subject `x` entails hypothesis_target `P_n` under the
    "defaults carry through" rule for a single linear default chain.

    The algorithm never touches PMPM or GraphResolutor; it operates directly on
    the chain structure and the defeasible evidence collected from the graph.
    """

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def resolve(
        self,
        graph,
        subject: str,
        hypothesis_target: str,
        serial_cutoff: Optional[float] = None,
        extra_edges: Optional[list] = None,
        chain_set: Optional[set] = None,
        bypass_rules: Optional[list] = None,
    ) -> str:
        """
        Resolve the entailment state of (subject → hypothesis_target).

        Parameters
        ----------
        graph : nx.DiGraph
            The fully-populated default-topology graph.
        subject : str
            Node ID of the entity being queried.
        hypothesis_target : str
            Node ID of the terminal property (P_n).
        serial_cutoff : float | None
            When given, only defeasible edges with serial ≤ serial_cutoff are
            considered.  Pass 0 to get a backbone-only view (no defeasible
            evidence; serials start at 1).  Pass None to include all edges.
        extra_edges : list of (edge_type, node_id) | None
            Synthetic object-level assertions injected before resolution.  Each
            tuple is (edge_type_str, chain_node_id) where edge_type_str is
            "defeasible_has_property" or "defeasible_lacks_property".
        bypass_rules : list of (src_node_id, tgt_node_id) | None
            Synthetic rule-level bypass edges introduced by Type-4 source A.
            Each tuple (Pj, Pk) means "if Pj is reachable, Pk is also an
            entry point" — equivalent to path-existence from Pk to hypothesis
            target, exactly as the linear-inheritance path check works once
            the bypass entry is established.

        Returns
        -------
        "ENTAILED" | "DEFEATED" | "UNDETERMINED"
        """
        chain = self.extract_chain(graph, subject, hypothesis_target)
        if len(chain) < 2:
            return "UNDETERMINED"

        effective_chain_set = chain_set if chain_set is not None else set(chain)
        pos_at, neg_at = self._collect_evidence(graph, subject, serial_cutoff, effective_chain_set)

        if extra_edges:
            for etype, nid in extra_edges:
                if etype == "defeasible_has_property":
                    pos_at.add(nid)
                elif etype == "defeasible_lacks_property":
                    neg_at.add(nid)

        return self._run_algorithm(chain, pos_at, neg_at, bypass_rules=bypass_rules)

    def extract_chain(self, graph, subject: str, hypothesis_target: str) -> list:
        """
        Return the ordered property chain [P_root, P_1, ..., P_n=hypothesis_target].

        Walks BACKWARD from hypothesis_target following type="implies" incoming
        edges until arriving at P_root (the node that subject connects to via
        type="is_a").

        Returns an empty list if the chain cannot be reconstructed.
        """
        p_root = self._find_is_a_target(graph, subject)
        if p_root is None:
            logger.debug("extract_chain: no is_a edge found for subject %s", subject)
            return []

        # Walk backwards from hypothesis_target collecting the chain.
        # We stop when we reach p_root.
        reverse_chain = [hypothesis_target]
        current = hypothesis_target

        visited = {hypothesis_target}
        while current != p_root:
            # Find the unique parent via a type="implies" incoming edge.
            parents = [
                u
                for u, _v, data in graph.in_edges(current, data=True)
                if data.get("type") == "implies"
            ]
            if not parents:
                # No parent implies → broken chain.
                logger.debug(
                    "extract_chain: chain broken at %s (no implies parent)", current
                )
                return []
            if len(parents) > 1:
                # Multiple parents — ambiguous; use the one closer to p_root if
                # one of them is p_root, otherwise log a warning and pick first.
                if p_root in parents:
                    parents = [p_root]
                else:
                    logger.debug(
                        "extract_chain: multiple implies parents at %s; using first",
                        current,
                    )
                    parents = [parents[0]]

            parent = parents[0]
            if parent in visited:
                logger.debug("extract_chain: cycle detected at %s", parent)
                return []
            visited.add(parent)
            reverse_chain.append(parent)
            current = parent

        chain = list(reversed(reverse_chain))
        return chain

    def baseline_state(self, graph, subject: str, hypothesis_target: str) -> str:
        """
        Resolve with serial_cutoff=0 — backbone only, no defeasible evidence.

        Equivalent to: "Is the hypothesis entailed by is_a + implies chain alone?"
        Under the defaults-carry-through rule, if subject is_a P_root and the
        chain P_root → ... → P_n exists with no blockers, the answer is ENTAILED.
        """
        return self.resolve(graph, subject, hypothesis_target, serial_cutoff=0)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_is_a_target(self, graph, subject: str) -> Optional[str]:
        """
        Return the node that subject links to via type="is_a", or None.
        """
        for _u, v, data in graph.out_edges(subject, data=True):
            if data.get("type") == "is_a":
                return v
        return None

    def _collect_evidence(
        self,
        graph,
        subject: str,
        serial_cutoff: Optional[float],
        chain_set: Optional[set] = None,
    ) -> tuple:
        """
        Collect positive (pos_at) and negative (neg_at) chain-node sets from
        the defeasible edges of `subject`.

        Only edges whose target is in chain_set are considered (when provided),
        preventing off-chain evidence in branching graphs from corrupting the
        resolution.  Only edges with serial ≤ serial_cutoff are considered when
        serial_cutoff is not None.

        Returns (pos_at: set[str], neg_at: set[str]).
        """
        pos_at: set = set()
        neg_at: set = set()

        for _u, v, data in graph.out_edges(subject, data=True):
            etype = data.get("type")
            if etype not in {"defeasible_has_property", "defeasible_lacks_property"}:
                continue
            if chain_set is not None and v not in chain_set:
                continue
            if serial_cutoff is not None:
                edge_serial = data.get("serial", 0)
                if edge_serial > serial_cutoff:
                    continue
            if etype == "defeasible_has_property":
                pos_at.add(v)
            else:
                neg_at.add(v)

        return pos_at, neg_at

    def _run_algorithm(
        self,
        chain: list,
        pos_at: set,
        neg_at: set,
        bypass_rules: Optional[list] = None,
    ) -> str:
        """
        Core "defaults carry through" resolution.

        chain : [P_root, P_1, ..., P_n]
        pos_at: chain nodes for which positive evidence exists
        neg_at: chain nodes for which negative evidence exists
        bypass_rules : list of (Pj_id, Pk_id) synthetic rule edges (Type 4 bypass)
            Each tuple means: if Pj is reachable from some entry point without
            any blocker in between, then Pk becomes an additional entry point.

        Rules
        -----
        1. If any chain node is in BOTH pos_at AND neg_at → UNDETERMINED
           (direct conflict at that node).
        2. Entry points into the chain:
           - Index 0 (P_root) is always an entry point (via is_a).
           - Any chain node in pos_at is an additional entry point.
        3. Bypass entry points: for each (Pj, Pk) bypass rule, if Pj is
           reachable from an existing entry point with no blocker in [entry+1..j],
           then Pk is added as an entry point (provided Pk itself is not blocked).
        4. For each entry point at index i, check whether any chain[j] with
           j > i is in neg_at.  If no blocker found for this entry → ENTAILED.
        5. If all entry points are blocked → DEFEATED.
        """
        # Rule 1: direct conflict at any node → UNDETERMINED.
        conflict_nodes = pos_at & neg_at
        if conflict_nodes:
            return "UNDETERMINED"

        # Build a set of entry points (chain indices where we can "enter" the
        # default chain and start carrying through).
        entry_indices = {0}  # is_a always grants entry at P_root (index 0)
        chain_index = {node: idx for idx, node in enumerate(chain)}

        for node in pos_at:
            if node in chain_index:
                entry_indices.add(chain_index[node])

        # Rule 3: bypass entry points.
        # A bypass rule (Pj, Pk) adds Pk as an entry point if Pj is reachable
        # from any existing entry without a negation blocker between them.
        if bypass_rules:
            for bypass_src, bypass_tgt in bypass_rules:
                if bypass_src not in chain_index or bypass_tgt not in chain_index:
                    continue
                j_idx = chain_index[bypass_src]
                k_idx = chain_index[bypass_tgt]
                if k_idx <= j_idx:
                    continue
                # Bypass target itself must not be negated (would create an
                # immediately-useless entry point blocked at index k).
                if chain[k_idx] in neg_at:
                    continue
                # Check whether any existing entry can reach Pj unblocked.
                for entry_i in sorted(entry_indices):
                    if entry_i > j_idx:
                        continue
                    # Sub-chain [entry_i+1 .. j_idx] must be free of negations.
                    blocked = any(
                        chain[x] in neg_at for x in range(entry_i + 1, j_idx + 1)
                    )
                    if not blocked:
                        entry_indices.add(k_idx)
                        break

        # Rules 4/5: for each entry point, check if sub-chain to P_n is clear.
        for entry_i in sorted(entry_indices):
            subchain_blocked = any(
                chain[j] in neg_at for j in range(entry_i + 1, len(chain))
            )
            if not subchain_blocked:
                return "ENTAILED"

        # All entry points were blocked — hypothesis is defeated.
        return "DEFEATED"
