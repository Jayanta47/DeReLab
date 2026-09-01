import json

import networkx as nx

from path_relation_module.graph_resolutor import GraphResolutor


def build_nixon_diamond_graph() -> nx.DiGraph:
    """
    Editable Nixon Diamond graph using DeReLab-style edge types.

    Structure:
    - Nixon is a Quaker
    - Nixon is a Republican
    - Quakers are typically pacifists
    - Republicans are typically not pacifists
    - Query: Is Nixon a pacifist?
    """
    graph = nx.DiGraph()
    graph.graph["topology"] = "default"

    graph.add_nodes_from(
        [
            "Nixon",
            "Quaker",
            "Republican",
            "Pacifist",
        ]
    )

    graph.add_edge("Nixon", "Quaker", type="is_a")
    graph.add_edge("Nixon", "Republican", type="is_a")
    graph.add_edge("Quaker", "Pacifist", type="defeasible_has_property")
    graph.add_edge("Republican", "Pacifist", type="defeasible_lacks_property")
    graph.add_edge("Nixon", "Pacifist", type="hypothesis_edge")

    return graph


def run_case():
    graph = build_nixon_diamond_graph()
    resolutor = GraphResolutor()

    subject = "Nixon"
    target = "Pacifist"
    result = resolutor.resolve(graph, subject, target)

    print("Nixon Diamond Test")
    print(json.dumps(
        {
            "query": f"{subject} -> {target}",
            "expected": "Skeptical/Unknown",
            "result": result,
            "pass": result == "Skeptical/Unknown",
        },
        indent=2,
    ))


if __name__ == "__main__":
    run_case()
