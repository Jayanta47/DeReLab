"""
Tree-inheritance PMPM tests.

Two suites verify that the algorithm is path-sensitive:

Suite A — double flip (True → False → True)
    A1  Backbone only: Canary inherits CanFly via Bird→Animal.        → True
    A2  Add Bird -(defeasible_lacks_property)→ CanFly.
        Bird is on the path, so it breaks Canary's chain.             → False
    A3  Add Canary -(defeasible_has_property)→ CanFly.
        Canary's direct positive overrides Bird's block.              → True

Suite B — off-path changes leave the result untouched
    B1  Same backbone as A1.                                           → True
    B2  Add Mammal -(defeasible_lacks_property)→ CanFly.
        Mammal is in the Dog/Mammal branch, off Canary's path.        → True
    B3  Add Eagle -(defeasible_lacks_property)→ CanFly.
        Eagle is Bird's sibling, not an ancestor of Canary in query.  → True
    B4  Add Dog -(defeasible_lacks_property)→ CanFly.
        Dog is a leaf of the Mammal branch, off Canary's path.        → True

Tree shape (child → parent edges, i.e. DeReLab convention):

         Animal
        /       \\
     Bird       Mammal
    /    \\         \\
 Canary  Eagle      Dog

Property node: CanFly  (Animal -(inheritance)→ CanFly)
Query:         Canary  -(hypothesis_edge)→ CanFly
"""

import json

import networkx as nx

from path_relation_module.graph_resolutor import GraphResolutor


# ──────────────────────────────────────────────────────────────────────────────
# Shared backbone builder
# ──────────────────────────────────────────────────────────────────────────────

def _build_base_tree() -> nx.DiGraph:
    """
    Pure backbone with no defeasible edges.
    Canary's positive path to CanFly: Canary → Bird → Animal → CanFly.
    Eagle and Dog/Mammal form the off-path branches.
    """
    g = nx.DiGraph()
    g.graph["topology"] = "inheritance"

    g.add_nodes_from(["Canary", "Eagle", "Bird", "Dog", "Mammal", "Animal", "CanFly"])

    # Backbone (child → parent, all positive)
    g.add_edge("Canary", "Bird",   type="is_a")
    g.add_edge("Eagle",  "Bird",   type="is_a")
    g.add_edge("Bird",   "Animal", type="inheritance")
    g.add_edge("Dog",    "Mammal", type="is_a")
    g.add_edge("Mammal", "Animal", type="inheritance")
    g.add_edge("Animal", "CanFly", type="inheritance")

    # Hypothesis
    g.add_edge("Canary", "CanFly", type="hypothesis_edge")

    return g


# ──────────────────────────────────────────────────────────────────────────────
# Suite A — double flip
# ──────────────────────────────────────────────────────────────────────────────

def build_A1() -> tuple[nx.DiGraph, str, str, str]:
    """
    A1: Backbone only.
    Canary's unbroken path: Canary → Bird → Animal → CanFly.
    Expected: True
    """
    return _build_base_tree(), "Canary", "CanFly", "True"


def build_A2() -> tuple[nx.DiGraph, str, str, str]:
    """
    A2: Add Bird -(defeasible_lacks_property)→ CanFly.
    Bird is the intermediate node ON Canary's path.
    Its exception blocks the inherited CanFly for everything below it.
    Expected: False
    """
    g, subj, prop, _ = build_A1()
    g.add_edge("Bird", "CanFly", type="defeasible_lacks_property")
    return g, subj, prop, "False"


def build_A3() -> tuple[nx.DiGraph, str, str, str]:
    """
    A3: Add Canary -(defeasible_has_property)→ CanFly.
    Canary is the leaf node ON its own path and is more specific than Bird.
    A direct positive edge from the query subject takes absolute priority.
    Expected: True
    """
    g, subj, prop, _ = build_A2()
    g.add_edge("Canary", "CanFly", type="defeasible_has_property")
    return g, subj, prop, "True"


# ──────────────────────────────────────────────────────────────────────────────
# Suite B — off-path changes have no effect
# ──────────────────────────────────────────────────────────────────────────────

def build_B1() -> tuple[nx.DiGraph, str, str, str]:
    """
    B1: Backbone only (same as A1).
    Expected: True
    """
    return _build_base_tree(), "Canary", "CanFly", "True"


def build_B2() -> tuple[nx.DiGraph, str, str, str]:
    """
    B2: Add Mammal -(defeasible_lacks_property)→ CanFly.
    Mammal is the root of the Dog/Mammal branch — completely off Canary's path.
    PMPM trims the graph to Canary's reachable subgraph, so Mammal never enters.
    Expected: True (unchanged)
    """
    g, subj, prop, _ = build_B1()
    g.add_edge("Mammal", "CanFly", type="defeasible_lacks_property")
    return g, subj, prop, "True"


def build_B3() -> tuple[nx.DiGraph, str, str, str]:
    """
    B3: Also add Eagle -(defeasible_lacks_property)→ CanFly.
    Eagle is Bird's sibling — it shares the Bird parent but is not an
    ancestor of Canary; it lies in a parallel sub-branch.
    Expected: True (unchanged)
    """
    g, subj, prop, _ = build_B2()
    g.add_edge("Eagle", "CanFly", type="defeasible_lacks_property")
    return g, subj, prop, "True"


def build_B4() -> tuple[nx.DiGraph, str, str, str]:
    """
    B4: Also add Dog -(defeasible_lacks_property)→ CanFly.
    Dog is a leaf of the Mammal branch — the furthest off-path node.
    Even three accumulated off-path negatives leave the Canary result intact.
    Expected: True (unchanged)
    """
    g, subj, prop, _ = build_B3()
    g.add_edge("Dog", "CanFly", type="defeasible_lacks_property")
    return g, subj, prop, "True"


# ──────────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────────

def _run_suite(suite_name: str, builders: list) -> bool:
    resolutor = GraphResolutor()
    print(f"\n{'─' * 56}")
    print(f" Suite {suite_name}")
    print(f"{'─' * 56}")
    all_pass = True
    for i, builder in enumerate(builders, 1):
        g, subj, prop, expected = builder()
        result = resolutor.resolve(g, subj, prop)
        passed = result == expected
        all_pass = all_pass and passed
        label = "PASS" if passed else "FAIL"
        print(f"\n  [{label}]  Case {suite_name}{i}: {subj} → {prop}")
        print(f"         expected={expected}  result={result}")
    print(f"\n  Suite {suite_name}: {'ALL PASS ✓' if all_pass else 'FAIL ✗'}")
    return all_pass


if __name__ == "__main__":
    print("Tree Inheritance PMPM Test Suite")
    print("=" * 56)

    a_pass = _run_suite(
        "A — double flip (True→False→True)",
        [build_A1, build_A2, build_A3],
    )
    b_pass = _run_suite(
        "B — off-path changes (all True)",
        [build_B1, build_B2, build_B3, build_B4],
    )

    print(f"\n{'=' * 56}")
    print(f"Overall: {'ALL PASS ✓' if a_pass and b_pass else 'FAIL ✗'}")
