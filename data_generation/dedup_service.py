"""
Duplicate-detection service for dataset generation pipelines.

Each DuplicationChecker instance is scoped to ONE difficulty level.
Never share an instance across difficulties.

Linear-inheritance fingerprint:
    (chain_length, root_concept, reveal_sequence)
    reveal_sequence = ((depth, edge_type), ...) ordered by serial.

Tree-inheritance fingerprint:
    (root_concept, canonical_tree_form)
    canonical_tree_form is a nested tuple built with the AHU canonical-form
    algorithm, making the fingerprint insensitive to random UUID node IDs.
    Each node contributes (def_edge_or_None, sorted_child_forms).

Lookup cost: O(1) average via Python hash set.
"""


class DuplicationChecker:

    def __init__(self):
        self._seen: set = set()
        self.total_checked: int = 0
        self.total_duplicates: int = 0

    # ------------------------------------------------------------------
    # Fingerprint computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_fingerprint(graph, chain_length: int, root_concept: str) -> tuple:
        """
        Build a hashable fingerprint from a (populated or raw) graph.

        Only the defeasible edges contribute to uniqueness — the backbone
        inheritance chain is fully determined by chain_length, and the
        property node is always the same.
        """
        defeasible: list[tuple] = []
        for u, _v, data in graph.edges(data=True):
            if data.get("type") in (
                "defeasible_has_property",
                "defeasible_lacks_property",
            ):
                depth = graph.nodes[u].get("layer", 0)
                defeasible.append((data["serial"], depth, data["type"]))

        # Sort ascending by serial so the tuple captures reveal order
        defeasible.sort()
        reveal_seq = tuple((depth, etype) for _, depth, etype in defeasible)
        return (chain_length, root_concept, reveal_seq)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, fingerprint: tuple) -> bool:
        """
        Attempt to register a fingerprint.

        Returns True  → fingerprint is new; sample accepted.
        Returns False → fingerprint already seen; sample is a duplicate.
        """
        self.total_checked += 1
        if fingerprint in self._seen:
            self.total_duplicates += 1
            return False
        self._seen.add(fingerprint)
        return True

    @staticmethod
    def compute_tree_fingerprint(graph, root_concept: str) -> tuple:
        """
        Build a hashable fingerprint for a tree-inheritance graph.

        Uses a recursive canonical form so that two graphs with the same
        branching structure and the same defeasible-edge pattern (same node
        positions, same types, same serial ordering) are detected as
        duplicates regardless of their random UUID node IDs.

        Each node contributes:  (defeasible_annotation, sorted_child_forms)
        defeasible_annotation:  None if the node has no defeasible edge, or
                                (serial, edge_type) otherwise.
        Children are sorted by their own canonical form so the result is
        invariant to insertion order.
        """
        root = next(
            (n for n, d in graph.nodes(data=True)
             if d.get("type") == "class" and d.get("layer", -1) == 0),
            None,
        )
        if root is None:
            return (root_concept,)

        cache: dict = {}

        def _subtree(node) -> tuple:
            if node in cache:
                return cache[node]
            ann = None
            for _, _, data in graph.out_edges(node, data=True):
                if data.get("type") in (
                    "defeasible_has_property",
                    "defeasible_lacks_property",
                ):
                    ann = (data.get("serial", 0), data["type"])
                    break
            children = [
                c for c in graph.predecessors(node)
                if graph.edges[c, node].get("type") == "inheritance"
            ]
            child_forms = tuple(sorted((_subtree(c) for c in children), key=repr))
            result = (ann, child_forms)
            cache[node] = result
            return result

        return (root_concept, _subtree(root))

    @staticmethod
    def compute_default_fingerprint(graph, root_concept: str) -> tuple:
        """
        Build a hashable fingerprint for a default-reasoning graph.

        Captures: chain property count, entity count, irrelevant-attribute
        count, root concept, and the defeasible-edge reveal sequence expressed
        as (entity_index, property_layer, edge_type) tuples sorted by serial.

        Entity node names follow the pattern "Object_<N>", so the index is
        extracted directly; property layers are numeric graph attributes.
        Both are deterministic within one generation pass, making the
        fingerprint stable across otherwise identical structural samples.
        """
        num_properties = sum(1 for _, d in graph.nodes(data=True) if d.get("type") == "property")
        num_entities = sum(1 for _, d in graph.nodes(data=True) if d.get("type") == "entity")
        num_irr = sum(1 for _, d in graph.nodes(data=True) if d.get("type") == "irrelevant_property")

        defeasible: list = []
        for u, v, data in graph.edges(data=True):
            if data.get("type") in ("defeasible_has_property", "defeasible_lacks_property"):
                # Extract entity index from "Object_N" node name.
                try:
                    entity_idx = int(u.rsplit("_", 1)[-1])
                except (ValueError, IndexError):
                    entity_idx = hash(u)
                prop_layer = graph.nodes[v].get("layer", 0)
                defeasible.append((data.get("serial", 0), entity_idx, prop_layer, data["type"]))

        defeasible.sort()
        reveal_seq = tuple((ei, pl, et) for _, ei, pl, et in defeasible)
        return (root_concept, num_properties, num_entities, num_irr, reveal_seq)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def unique_count(self) -> int:
        return len(self._seen)

    def stats(self) -> dict:
        return {
            "total_checked": self.total_checked,
            "total_duplicates": self.total_duplicates,
            "unique_count": self.unique_count,
            "duplicate_rate": (
                self.total_duplicates / self.total_checked
                if self.total_checked > 0
                else 0.0
            ),
        }
