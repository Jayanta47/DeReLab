import networkx as nx
from typing import Optional

from .pmpm_algo import query_defeasible_graph


class GraphResolutor:
    """
    Adapts DeReLab graphs into the signed PMPM input graph.

    All graph topologies go through the same skeptical path-relation algorithm.
    The only thing that changes is how DeReLab edge types are interpreted as
    positive or negative evidence.

    Use serial_cutoff in build_algorithm_graph / resolve to limit which
    defeasible edges are visible to PMPM at a given point in the reveal
    schedule. Backbone edges (inheritance, is_a, implies) are always included.
    """

    POSITIVE_EDGE_TYPES = {
        "inheritance",
        "is_a",
        "implies",
        "has_property",
        "defeasible_has_property",
    }
    NEGATIVE_EDGE_TYPES = {"defeasible_lacks_property"}
    PASS_THROUGH_EDGE_TYPES = {"hypothesis_edge", "has_attribute"}
    DEFEASIBLE_EDGE_TYPES = {"defeasible_has_property", "defeasible_lacks_property"}

    def infer_topology(self, original_graph) -> str:
        topology = original_graph.graph.get("topology")
        if topology:
            return topology

        edge_types = {data.get("type") for _, _, data in original_graph.edges(data=True)}
        if "inheritance" in edge_types:
            return "inheritance"
        return "default"

    def build_algorithm_graph(self, original_graph, topology=None, serial_cutoff: Optional[int] = None):
        """
        Build the signed graph that PMPM receives.

        serial_cutoff: when provided, defeasible edges with serial > serial_cutoff
        are excluded. Pass 0 to get a backbone-only view (no defeasible evidence).
        Pass None to include all edges in the graph as-is (caller is responsible
        for pre-filtering when needed, e.g. default topology with group events).
        """
        topology = topology or self.infer_topology(original_graph)

        algo_graph = nx.DiGraph()
        algo_graph.add_nodes_from(original_graph.nodes())

        for u, v, data in original_graph.edges(data=True):
            edge_type = data.get("type")
            if edge_type in self.PASS_THROUGH_EDGE_TYPES:
                continue
            if serial_cutoff is not None and edge_type in self.DEFEASIBLE_EDGE_TYPES:
                if data.get("serial", 0) > serial_cutoff:
                    continue
            if edge_type in self.POSITIVE_EDGE_TYPES:
                algo_graph.add_edge(u, v, sign="+")
            elif edge_type in self.NEGATIVE_EDGE_TYPES:
                algo_graph.add_edge(u, v, sign="-")

        return algo_graph

    def resolve(self, original_graph, x, y, topology=None, serial_cutoff: Optional[int] = None):
        topology = topology or self.infer_topology(original_graph)
        clean_graph = self.build_algorithm_graph(original_graph, topology=topology, serial_cutoff=serial_cutoff)
        return query_defeasible_graph(clean_graph, x, y)
