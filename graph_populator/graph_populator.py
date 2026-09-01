from .graph_reader import GraphReader
from data_handler.kb_manager import KnowledgeBaseManager


class GraphPopulator:
    """
    Responsible for injecting real-world entities and attribute metadata 
    from the Knowledge Base into the topological graph.
    """
    def __init__(self, kb_manager: KnowledgeBaseManager):
        self.graph = None
        self.reader = GraphReader()
        self.kb = kb_manager

    def populate(self, graph):
        self.graph = graph
        self.reader.set_graph(graph)
        """Main entry point to populate the graph based on its topology."""
        topology = self.graph.graph.get("topology")

        if topology == "linear_inheritance":
            self._populate_inheritance_chain()
        elif topology == "tree_inheritance":
            self._populate_inheritance_tree()
        else:
            self._populate_default_chain()

        return self.graph
    
    def _populate_inheritance_tree(self):
        root_concept = self.graph.graph.get("root_concept", "animal")

        # Find the single root class node (layer 0, type class).
        root_nodes = [
            n for n, d in self.graph.nodes(data=True)
            if d.get("type") == "class" and d.get("layer", -1) == 0
        ]
        if not root_nodes:
            return
        root_node = root_nodes[0]

        # BFS from root following inheritance edges in reverse
        # (edges go child → parent, so predecessors of a node are its children).
        ordered_class_nodes = []
        visited: set = set()
        queue = [root_node]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            if self.graph.nodes[node].get("type") == "class":
                ordered_class_nodes.append(node)
            for child in self.graph.predecessors(node):
                edge = self.graph.edges[child, node]
                if edge.get("type") == "inheritance" and child not in visited:
                    queue.append(child)

        print(f"Tree populator: {len(ordered_class_nodes)} class nodes found.")

        # Request one label per class node: root_concept + nonce words for the rest.
        concept_chain = self.kb.get_concept_chain(
            min_length=len(ordered_class_nodes),
            root_concept=root_concept,
        )
        # print(f"Concept chain retrieved from KB: {concept_chain}")

        for i, node_id in enumerate(ordered_class_nodes):
            label = concept_chain[i] if i < len(concept_chain) else f"concept_{i}"
            self.graph.nodes[node_id]["label"] = label
            self.graph.nodes[node_id]["metadata"] = {
                "concept": label,
                "layer": self.graph.nodes[node_id].get("layer", 0),
            }

        # Property node gets property info from the KB.
        property_info = self.kb.get_full_property_info(
            root_concept=root_concept, limit=1
        )
        # print(f"Property info retrieved from KB: {property_info}")

        property_nodes = self.reader.get_nodes_by_type("property")
        if property_nodes and property_info:
            prop_data = property_info[0]
            for node_id, _ in property_nodes:
                self.graph.nodes[node_id]["label"] = prop_data["selected_value"]
                self.graph.nodes[node_id]["metadata"] = prop_data

    def _populate_inheritance_chain(self):
        layers = self.reader.get_sorted_layers()
        if not layers:
            return

        max_depth = max(layers.keys())
        concept_chain = self.kb.get_concept_chain(
            min_length=max_depth + 1,
            root_concept=self.graph.graph.get("root_concept"),
        )

        # print("Concept chain retrieved from KB:", concept_chain)

        property_info = self.kb.get_full_property_info(
            root_concept=self.graph.graph.get("root_concept"),
            limit=1
        )

        # print("Property info retrieved from KB:", property_info)

        for layer_idx, nodes in layers.items():
            if layer_idx < len(concept_chain):
                concept_text = concept_chain[layer_idx]
                for node_id in nodes:
                    self.graph.nodes[node_id]["label"] = concept_text
                    self.graph.nodes[node_id]["metadata"] = {
                        "concept": concept_text,
                        "layer": layer_idx,
                    }

        property_nodes = self.reader.get_nodes_by_type("property")
        if property_nodes and property_info:
            prop_data = property_info[0]
            for node_id, _ in property_nodes:
                self.graph.nodes[node_id]["label"] = prop_data["selected_value"]
                self.graph.nodes[node_id]["metadata"] = prop_data
        

    def _populate_default_chain(self):
        # 1. Populate Entities (Layer 0)
        object_nodes = self.reader.get_nodes_by_type("entity")
        # entity is the labels for object nodes and how they will be named
        # worthwhile to 
        entity_labels = self.kb.get_entity_labels(
            len(object_nodes),
            root_concept=self.graph.graph.get("root_concept", "animal"),
            allow_nonsensical=self.graph.graph.get("allow_nonsensical_entities", True),
        )
        for i, (node_id, _) in enumerate(object_nodes):
            entity_data = entity_labels[i]
            self.graph.nodes[node_id]["label"] = entity_data["label"]
            self.graph.nodes[node_id]["metadata"] = entity_data

        # 2. Populate Main Properties extracted from the graph for the root concept.
        property_nodes = sorted(
            self.reader.get_nodes_by_type("property"), key=lambda x: x[1]["layer"]
        )
        domain = self.graph.graph.get("root_concept", "Object")
        main_attributes = self.kb.get_default_attributes(
            domain=domain,
            num_attributes=len(property_nodes),
        )
        used_attribute_names = []

        for i, (node_id, _) in enumerate(property_nodes):
            if i < len(main_attributes):
                attr_data = main_attributes[i]
                used_attribute_names.append(attr_data["attribute_name"])

                self.graph.nodes[node_id]["label"] = attr_data["selected_value"]
                self.graph.nodes[node_id]["metadata"] = attr_data

        # 3. Populate Irrelevant Properties
        irrelevant_nodes = self.reader.get_nodes_by_type("irrelevant_property")
        if irrelevant_nodes:
            irr_attributes = self.kb.get_default_irrelevant_attributes(
                domain=domain,
                num_irr_attr=len(irrelevant_nodes),
                exclusion_set=used_attribute_names,
            )
            for i, (node_id, _) in enumerate(irrelevant_nodes):
                if i < len(irr_attributes):
                    attr_data = irr_attributes[i]
                    self.graph.nodes[node_id]["label"] = attr_data["selected_value"]
                    # Store metadata for distractors
                    self.graph.nodes[node_id]["metadata"] = attr_data
