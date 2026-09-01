import networkx as nx
from typing import Dict, List, Tuple

from utils.models.edge import GraphEdge, EdgeType
from utils.models.node import EntityNode, PropertyNode

class Graph:
    def __init__(self):
  
        self.graph = nx.DiGraph()

    def add_entity(self, name: str, attributes: Dict = None) -> EntityNode:
        if name in self.graph: 
            return self.graph.nodes[name]['data']
        
        if attributes is None: attributes = {}
        node = EntityNode(name, **attributes)
        self.graph.add_node(node.id, data=node)
        return node

    def add_property(self, prop_type: str, prop_value: str, attributes: Dict = None) -> PropertyNode:
        unique_id = f"{prop_type}:{prop_value}"
        if unique_id in self.graph: 
            return self.graph.nodes[unique_id]['data']
        
        if attributes is None: attributes = {}
        node = PropertyNode(prop_type, prop_value, **attributes)
        self.graph.add_node(node.id, data=node)
        return node

    def add_relation(self, source_id: str, target_id: str, relation: EdgeType, 
                     preposition: str = None, grammar_meta: Dict = None):
        
        if grammar_meta is None: grammar_meta = {}
        if preposition: grammar_meta['preposition'] = preposition

        # Standardize Enum to Value
        rel_value = relation.value if hasattr(relation, 'value') else relation

        edge_obj = GraphEdge(
            source=source_id,
            target=target_id,
            relation_type=rel_value,
            metadata=grammar_meta
        )

        self.graph.add_edge(source_id, target_id, data=edge_obj)

        if relation == EdgeType.INHERITANCE:
            self._update_structural_metadata(source_id, target_id)




    def _update_structural_metadata(self, parent_id: str, child_id: str):
        if parent_id not in self.graph or child_id not in self.graph: return

        parent_node = self.graph.nodes[parent_id]['data']
        child_node = self.graph.nodes[child_id]['data']

        if not isinstance(parent_node, EntityNode) or not isinstance(child_node, EntityNode):
            return

        # Recalculate child count
        count = 0
        for _, _, d in self.graph.out_edges(parent_id, data=True):
            if 'data' in d and d['data'].relation_type == EdgeType.INHERITANCE.value:
                count += 1
        parent_node.child_count = count

        # Propagate depth
        new_depth = parent_node.depth + 1
        if new_depth > child_node.depth:
            child_node.depth = new_depth
            self._propagate_depth(child_id)

    def _propagate_depth(self, current_id: str):
        current_node = self.graph.nodes[current_id]['data']
        for succ_id in self.graph.successors(current_id):
            edge_data = self.graph.get_edge_data(current_id, succ_id)
            if not edge_data or 'data' not in edge_data: continue
            
            edge_obj = edge_data['data']
            if edge_obj.relation_type == EdgeType.INHERITANCE.value:
                succ_node = self.graph.nodes[succ_id]['data']
                if isinstance(succ_node, EntityNode):
                    if succ_node.depth < current_node.depth + 1:
                        succ_node.depth = current_node.depth + 1
                        self._propagate_depth(succ_id)



    def get_children_of(self, node_id: str):
        children = []
        if node_id not in self.graph: return children
        for succ in self.graph.successors(node_id):
            edge_data = self.graph[node_id][succ]
            if 'data' in edge_data:
                edge_obj = edge_data['data']
                if edge_obj.relation_type == EdgeType.INHERITANCE.value:
                    children.append(self.graph.nodes[succ]['data'])
        return children

    def get_properties_of(self, node_id: str):
        props = []
        if node_id not in self.graph: return props
        for succ in self.graph.successors(node_id):
            edge_data = self.graph[node_id][succ]
            if 'data' in edge_data:
                edge_obj = edge_data['data']
                if edge_obj.relation_type in [EdgeType.HAS_PROPERTY.value, EdgeType.HAS_NOT_PROPERTY.value]:
                    props.append((self.graph.nodes[succ]['data'], edge_obj))
        return props