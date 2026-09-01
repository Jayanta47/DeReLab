import networkx as nx

class GraphReader:
    """
    Encapsulates all logic for traversing and reading the NetworkX graph structure.
    """
    def set_graph(self, graph):
        self.graph = graph

    def get_nodes_by_type(self, node_type):
        """Returns a list of (node_id, data) tuples for a specific type."""
        return [
            (n, d) for n, d in self.graph.nodes(data=True) 
            if d.get('type') == node_type
        ]

    def get_edges_by_type(self, edge_type):
        """Returns a list of (u, v, data) for a specific edge type."""
        return [
            (u, v, d) for u, v, d in self.graph.edges(data=True) 
            if d.get('type') == edge_type
        ]

    def get_sorted_layers(self):
        """
        Returns nodes grouped by layer index.
        Useful for processing chains in order (Level 0 -> Level 1 -> ...).
        """
        layers = {}
        for n, d in self.graph.nodes(data=True):
            layer_idx = d.get('layer', 0)
            if layer_idx not in layers:
                layers[layer_idx] = []
            layers[layer_idx].append(n)
        return dict(sorted(layers.items()))

    def get_outgoing_neighbors(self, node_id, edge_type=None):
        """Finds downstream nodes."""
        if edge_type:
            return [
                v for u, v, d in self.graph.out_edges(node_id, data=True) 
                if d.get('type') == edge_type
            ]
        return list(self.graph.successors(node_id))