"""
Effect classifier for inheritance-based topologies (linear and tree).

Policy v1: An evidence edge on a class node that is a positive ancestor of
the hypothesis subject strengthens the case if it confirms the property
(defeasible_has_property) and weakens it if it denies the property
(defeasible_lacks_property).  Edges whose source node is NOT on any
positive path from the hypothesis subject have no effect.

This module is isolated from the conversation builder so the policy can be
changed here without touching the rest of the pipeline.
"""

import networkx as nx

_POSITIVE_EDGE_TYPES = frozenset({"inheritance", "is_a", "implies", "has_property"})


def classify_inheritance_effect(
    graph: nx.DiGraph,
    source_node: str,
    edge_type: str,
    subject: str,
    hypothesis_target: str,
) -> str:
    """
    Classify the effect of one defeasible evidence edge on an inheritance hypothesis.

    Parameters
    ----------
    graph           : The full populated graph (linear_inheritance or tree_inheritance).
    source_node     : Class node that carries the defeasible edge.
    edge_type       : 'defeasible_has_property' or 'defeasible_lacks_property'.
    subject         : Hypothesis subject (the leaf node being queried).
    hypothesis_target: Property node being queried.

    Returns
    -------
    'strengthen', 'weaken', or 'no effect'.
    """
    if not _is_on_positive_path(graph, source_node, subject):
        return "no effect"

    if edge_type == "defeasible_has_property":
        return "strengthen"
    if edge_type == "defeasible_lacks_property":
        return "weaken"
    return "no effect"


def _is_on_positive_path(graph: nx.DiGraph, node: str, subject: str) -> bool:
    """
    Return True if `node` is reachable from `subject` via positive edges.

    In DeReLab inheritance graphs, edges point from child to parent
    (Subclass → Superclass), so traversing successors of `subject` walks
    up toward the root.  A source_node that is reachable this way is an
    ancestor of the subject and therefore relevant to the hypothesis.
    """
    if node == subject:
        return True

    visited: set = set()
    queue = [subject]
    while queue:
        curr = queue.pop(0)
        if curr in visited:
            continue
        visited.add(curr)
        for succ in graph.successors(curr):
            if succ in visited:
                continue
            if graph.edges[curr, succ].get("type") in _POSITIVE_EDGE_TYPES:
                if succ == node:
                    return True
                queue.append(succ)
    return False
