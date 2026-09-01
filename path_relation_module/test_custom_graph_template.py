import json
import networkx as nx

from path_relation_module.graph_resolutor import GraphResolutor

def build_case_1() -> tuple[nx.DiGraph, str, str, str]:
    """
    Image 1: Standard inheritance path without contradictions. 
    Expected: True
    """
    graph = nx.DiGraph()
    graph.graph["topology"] = "inheritance"

    nodes = ["Moby", "Whale", "Mammals", "Land-dwellers", "Air-breathers"]
    graph.add_nodes_from(nodes)

    # Backbone (Black arrows)
    graph.add_edge("Moby", "Whale", type="is_a")
    graph.add_edge("Whale", "Mammals", type="inheritance")
    graph.add_edge("Mammals", "Land-dwellers", type="inheritance")
    graph.add_edge("Land-dwellers", "Air-breathers", type="inheritance")

    # Hypothesis (Green arrow)
    graph.add_edge("Moby", "Air-breathers", type="hypothesis_edge")

    return graph, "Moby", "Air-breathers", "True"

def build_case_2() -> tuple[nx.DiGraph, str, str, str]:
    """
    Image 2: Defeasible lack of property breaks the chain. 
    Expected: False
    """
    # Start with the base backbone from Case 1
    graph, subject, target, _ = build_case_1()

    # Exception (Red crossed arrow): Whale lacks property Land-dwellers
    graph.add_edge("Whale", "Land-dwellers", type="defeasible_lacks_property")

    return graph, subject, target, "False"

def build_case_3() -> tuple[nx.DiGraph, str, str, str]:
    """
    Image 3: Direct defeasible property overrides the broken chain. 
    Expected: True
    """
    # Start with the structure from Case 2 (which already has the broken chain)
    graph, subject, target, _ = build_case_2()

    # Exception override (Solid Red arrow): Whale has property Air-breathers
    graph.add_edge("Whale", "Air-breathers", type="defeasible_has_property")

    return graph, subject, target, "True"

def run_all_cases():
    cases = [build_case_1, build_case_2, build_case_3]
    resolutor = GraphResolutor()

    print("--- Derelab Defeasible Reasoning Test Suite ---")
    for i, builder in enumerate(cases, 1):
        graph, subject, target, expected = builder()
        result = resolutor.resolve(graph, subject, target)

        print(f"\nTest Case {i}:")
        print(json.dumps(
            {
                "query": f"{subject} -> {target}",
                "expected": expected,
                "result": result,
                "pass": str(result).lower() == expected.lower(),
            },
            indent=2,
        ))

if __name__ == "__main__":
    run_all_cases()