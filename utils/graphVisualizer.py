import networkx as nx
import matplotlib.pyplot as plt

class GraphVisualizer:
    """
    A standalone tool to visualize the NetworkX graphs.
    """
    def __init__(self, layout_type='spring'):
        self.layout_type = layout_type

    def visualize(self, graph, title="Graph Structure"):
        plt.figure(figsize=(10, 6))
        
        # 1. layout definition
        if self.layout_type == 'spring':
            pos = nx.spring_layout(graph, seed=42)
        elif self.layout_type == 'shell':
            pos = nx.shell_layout(graph)
        else:
            # A custom hierarchical layout often looks best for "Reasoning Chains"
            pos = nx.multipartite_layout(graph, subset_key="layer")

        # 2. Draw Nodes
        # We can color nodes based on their 'type' attribute if it exists
        node_colors = []
        for node in graph.nodes(data=True):
            attr = node[1].get('type', 'default')
            if attr == 'entity':
                node_colors.append('#99ccff') # Light Blue
            elif attr == 'exception':
                node_colors.append('#ff9999') # Light Red
            else:
                node_colors.append('#99ff99') # Light Green

        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=2000, alpha=0.9, edgecolors='black')

        # 3. Draw Edges
        nx.draw_networkx_edges(graph, pos, width=2, arrowstyle='-|>', arrowsize=20, edge_color='gray')

        # 4. Draw Labels
        nx.draw_networkx_labels(graph, pos, font_size=10, font_family='sans-serif', font_weight='bold')

        # 5. Draw Edge Labels (if any specific types exist)
        edge_labels = nx.get_edge_attributes(graph, 'type')
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)

        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()