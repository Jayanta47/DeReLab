import networkx as nx
import uuid
import random
import os
import sys
import math

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

sys.path.append(parent_dir)

from configs.graphConfig import GraphGenerationConfig, InheritanceGraphConfig

# ---------------------------------------------------------
# Graph Generator Implementation
# ---------------------------------------------------------


class GraphStructureGenerator:
    """
    Base class for generating graph structures.
    """

    def __init__(self):
        self.graph = nx.DiGraph()

    def reset_graph(self):
        self.graph = nx.DiGraph()

    def get_graph(self):
        return self.graph

    def clear(self):
        self.graph.clear()

    def generate_structure(self, config: GraphGenerationConfig):
        raise NotImplementedError("Subclasses must implement this method.")


class DefaultChainGenerator(GraphStructureGenerator):
    """
    Generates a bounded-branching property tree for Default Reasoning.
    Structure: Objects -> Root Property -> Property Tree ...
    """

    def __init__(self):
        super().__init__()
        self.property_leaf_nodes = []
        self.hypothesis_target = None

    def _generate_attribute_chain(
        self,
        attribute_length,
        root_name="Property_Root",
        max_branching_factor=3,
        branch_probability=0.3,  
    ):
        """
        Creates an organic property tree where only randomly selected nodes 
        are allowed to branch based on a probability threshold.
        """
        self.property_leaf_nodes = []
        self.hypothesis_target = None

        if attribute_length <= 0:
            return root_name

        max_branching_factor = max(1, int(max_branching_factor))
        branch_probability = max(0.0, min(1.0, float(branch_probability)))
        
        # Track node stats: {"node_id": {"children": 0, "capacity": X}}
        node_stats = {}
        
        # The pool of nodes that still have room for children
        open_parents = []
        property_nodes = []

        def initialize_node(node_id):
            """Determines the branching destiny of a node when it is born."""
            # Roll the dice against the threshold
            if max_branching_factor > 1 and random.random() < branch_probability:
                # This node is a BRANCHER. It can have between 2 and max_branching_factor children.
                capacity = random.randint(2, max_branching_factor)
            else:
                # This node is LINEAR. It can only pass the chain forward to 1 child.
                capacity = 1

            node_stats[node_id] = {"children": 0, "capacity": capacity}
            open_parents.append(node_id)
            property_nodes.append(node_id)

        # 1. Setup Root
        self.graph.add_node(root_name, type="property", layer=1)
        initialize_node(root_name)

        # 2. Iteratively build the tree until we hit the exact length
        for i in range(1, attribute_length):
            current_node_id = f"Property_{i}"
            
            # Pick a valid parent from the open pool
            parent_node_id = random.choice(open_parents)
            parent_layer = self.graph.nodes[parent_node_id]["layer"]

            # Add the new node
            self.graph.add_node(
                current_node_id, 
                type="property", 
                layer=parent_layer + 1
            )
            
            # Connect them
            edge_id = str(uuid.uuid4())[:8]
            self.graph.add_edge(
                parent_node_id, current_node_id, id=edge_id, type="implies"
            )

            # Update the parent's stats
            node_stats[parent_node_id]["children"] += 1
            
            # If the parent has reached its rolled capacity, it cannot accept more children
            if node_stats[parent_node_id]["children"] >= node_stats[parent_node_id]["capacity"]:
                open_parents.remove(parent_node_id)

            # Initialize the newly created node
            initialize_node(current_node_id)

        # 3. Identify all nodes that never received children (the endpoints)
        self.property_leaf_nodes = [
            node_id for node_id in property_nodes if self.graph.out_degree(node_id) == 0
        ]

        return root_name

    def _resolve_default_config_value(self, config, field_name, default_value):
        default_config = getattr(config, "default_config", None)
        if default_config is not None and hasattr(default_config, field_name):
            return getattr(default_config, field_name)
        return getattr(config, field_name, default_value)

    def _get_hypothesis_target(self):
        if self.hypothesis_target and self.hypothesis_target in self.graph:
            return self.hypothesis_target

        property_nodes = [
            node_id
            for node_id, data in self.graph.nodes(data=True)
            if data.get("type") == "property"
        ]
        if not property_nodes:
            return None

        leaf_nodes = [
            node_id for node_id in property_nodes if self.graph.out_degree(node_id) == 0
        ]
        candidate_nodes = leaf_nodes or property_nodes
        self.hypothesis_target = max(
            candidate_nodes,
            key=lambda node_id: (
                self.graph.nodes[node_id].get("layer", 0),
                node_id,
            ),
        )
        return self.hypothesis_target

    def _generate_object_nodes(self, num_objects, root_property_node):
        """
        Creates object nodes and connects them to the root of the attribute chain.
        """
        object_nodes = []
        for i in range(num_objects):
            obj_node_id = f"Object_{i}"
            object_nodes.append(obj_node_id)

            # Layer 0 for Objects
            self.graph.add_node(obj_node_id, type="entity", layer=0)

            # Connect Object -> Root Property (e.g., Tweety -> Bird)
            edge_id = str(uuid.uuid4())[:8]
            self.graph.add_edge(
                obj_node_id, root_property_node, id=edge_id, type="is_a"
            )

        return object_nodes

    def _add_irrelevant_attribute(self, object_nodes, num_irr_attr):
        """
        Adds irrelevant 'noise' attributes to random object nodes.
        """
        if not object_nodes:
            return

        for i in range(num_irr_attr):
            # Select a random object to attach noise to
            target_obj = random.choice(object_nodes)

            irr_node_id = f"Irrelevant_{i}"

            # Place irrelevant nodes on Layer 1 (same depth as Root Property but disconnected from chain)
            self.graph.add_node(irr_node_id, type="irrelevant_property", layer=1)

            # Connect Object -> Irrelevant Attribute
            edge_id = str(uuid.uuid4())[:8]
            self.graph.add_edge(
                target_obj, irr_node_id, id=edge_id, type="has_attribute"
            )

    def _add_hypothesis_edges(self, num_hypothesis_subjects=None):
        """
        Connects object nodes to the selected terminal property.

        num_hypothesis_subjects: if given, only this many randomly chosen objects
        receive a hypothesis edge. None means all objects get one.
        """
        object_nodes = [n for n, d in self.graph.nodes(data=True) if d.get("type") == "entity"]
        hypothesis_target = self._get_hypothesis_target()

        if not object_nodes or hypothesis_target is None:
            return

        if num_hypothesis_subjects is not None:
            k = max(1, min(num_hypothesis_subjects, len(object_nodes)))
            object_nodes = random.sample(object_nodes, k)

        for obj in object_nodes:
            edge_id = str(uuid.uuid4())[:8]
            self.graph.add_edge(
                obj, hypothesis_target, id=edge_id, type="hypothesis_edge"
            )

    def _add_defeasible_edges(self, sparsity_factor):
        """
        Adds probabilistic 'yes' and 'no' evidence edges to ALL intermediate properties
        across all branches, assigning a 'serial' attribute for temporal ordering.
        """
        object_nodes = [n for n, d in self.graph.nodes(data=True) if d.get("type") == "entity"]
        hypothesis_target = self._get_hypothesis_target()

        if not object_nodes or hypothesis_target is None:
            return

        # Fetch ALL intermediate property nodes across the entire tree structure.
        # An intermediate node is a property that is NOT the root (layer 1)
        # and NOT a leaf node (out_degree == 0).
        intermediate_props = [
            n for n, d in self.graph.nodes(data=True)
            if d.get("type") == "property" 
            and d.get("layer") > 1 
            and self.graph.out_degree(n) > 0 
        ]

        # Sort by layer so that evidence is revealed logically from general to specific
        intermediate_props.sort(key=lambda n: self.graph.nodes[n].get("layer", 0))

        serial_counter = 1

        for prop in intermediate_props:
            # Shuffle objects so the temporal timeline doesn't always start with Object_0
            random.shuffle(object_nodes)

            for obj in object_nodes:
                if random.random() < sparsity_factor:
                    edge_type = random.choice([
                        "defeasible_has_property", 
                        "defeasible_lacks_property"
                    ])
                    edge_id = str(uuid.uuid4())[:8]

                    self.graph.add_edge(
                        obj, prop, 
                        id=edge_id, 
                        type=edge_type, 
                        serial=serial_counter  # The temporal sequence marker
                    )
                    
                    serial_counter += 1

    def generate_structure(self, config: GraphGenerationConfig):
        """
        Orchestrates the generation of the full graph based on the config.
        """
        self.clear()

        # 1. Build the Property Chain
        chain_length = self._resolve_default_config_value(config, "chain_length", 3)
        num_objects = self._resolve_default_config_value(config, "num_objects", 2)
        num_irrelevant_attributes = self._resolve_default_config_value(
            config, "num_irrelevant_attributes", 1
        )
        max_branching_factor = self._resolve_default_config_value(
            config, "max_branching_factor", None
        )
        if max_branching_factor is None:
            max_branching_factor = self._resolve_default_config_value(
                config, "attribute_branching_factor", 1
            )
        branch_probability = self._resolve_default_config_value(
            config, "branch_probability", 0.0
        )

        root_prop_id = self._generate_attribute_chain(
            attribute_length=chain_length,
            root_name=config.root_name,
            max_branching_factor=max_branching_factor,
            branch_probability=branch_probability,
        )

        # 2. Build Objects and link to Root
        object_nodes = self._generate_object_nodes(
            num_objects=num_objects, root_property_node=root_prop_id
        )

        # 3. Add Irrelevant Attributes
        if num_irrelevant_attributes > 0:
            self._add_irrelevant_attribute(
                object_nodes, num_irrelevant_attributes
            )

        # 4. Inject Evaluation Edges
        num_hyp = self._resolve_default_config_value(config, "num_hypothesis_subjects", None)
        self._add_hypothesis_edges(num_hypothesis_subjects=num_hyp)
        self._add_defeasible_edges(config.sparsity_factor)

        num_sources_range = self._resolve_default_config_value(config, "num_sources_range", (2, 2))
        self.graph.graph["topology"] = "default"
        self.graph.graph["reasoning_mode"] = "defeasible"
        self.graph.graph["allow_nonsensical_entities"] = config.allow_nonsensical_entities
        self.graph.graph["root_concept"] = config.root_concept
        self.graph.graph["num_sources_range"] = num_sources_range
        # Store the full DefaultGraphConfig so downstream stages (question
        # generator, NL generator) can read all config fields without having
        # to re-resolve them from individual graph attributes.
        self.graph.graph["default_config"] = getattr(config, "default_config", None)

        return self.graph



class LinearInheritanceGenerator(GraphStructureGenerator):
    """
    Generates a simple vertical inheritance chain:
    Class A <-(is_a)- Class B <-(is_a)- Class C ...
    """

    def add_defeasible_edges(self, config: GraphGenerationConfig):
        """
        Adds defeasible edges to the inheritance chain to simulate exceptions,
        assigning a serial number for temporal ordering.
        """
        # We can only attach defeasible edges to intermediate subclasses.
        # Exclude Root (0) and the Hypothesis node (max_depth).
        available_depths = list(range(1, config.inheritance_config.max_depth))
        
        if not available_depths:
            return

        # Determine the number of edges with safe bounds
        lower_bound = max(1, config.inheritance_config.max_depth // 2)
        upper_bound = max(lower_bound, config.inheritance_config.max_depth)
        num_edges = random.randint(lower_bound, upper_bound)
        
        # Ensure we don't try to sample more edges than available nodes
        num_edges = min(num_edges, len(available_depths))

        # Randomly select unique depths and sort them ascending.
        # This ensures we start from the "lowest" numerical depth (closest to root)
        # and move downwards toward the most specific subclass.
        selected_depths = sorted(random.sample(available_depths, num_edges))

        serial_counter = 1
        for depth in selected_depths:
            child_node = f"Subclass_{depth}"
            
            # Randomly pick if this subclass has the property or lacks it
            edge_type = random.choice(["defeasible_has_property", "defeasible_lacks_property"])
            edge_id = str(uuid.uuid4())[:8]

            self.graph.add_edge(
                child_node, 
                self.property_node, 
                id=edge_id, 
                type=edge_type, 
                serial=serial_counter
            )
            serial_counter += 1

    def _add_hypothesis_edge(self, config: GraphGenerationConfig):
        """
        Attaches the hypothesis edge strictly to the final node in the chain.
        """
        last_node = f"Subclass_{config.inheritance_config.max_depth}"
        edge_id = str(uuid.uuid4())[:8]
        
        self.graph.add_edge(
            last_node, 
            self.property_node, 
            id=edge_id, 
            type="hypothesis_edge"
        )

    def generate_structure(self, config: GraphGenerationConfig):
        self.clear()

        # Root Class
        parent_node = config.root_name
        
        # We create a singular property node that the whole inheritance chain will refer to
        self.property_node = "property_node" 
        
        self.graph.add_node(parent_node, type="class", layer=0)
        self.graph.add_node(self.property_node, type="property", layer=0)
        
        # The base premise: The root class has the property
        self.graph.add_edge(
            parent_node, 
            self.property_node, 
            id=str(uuid.uuid4())[:8], 
            # type=random.choice(["has_property", "lacks_property"])
            type="has_property"
        )

        # Build the inheritance chain
        for i in range(1, config.inheritance_config.max_depth + 1):
            child_node = f"Subclass_{i}"
            self.graph.add_node(child_node, type="class", layer=i)

            # Inheritance flows UP (Subclass IS_A Superclass)
            edge_id = str(uuid.uuid4())[:8]
            self.graph.add_edge(child_node, parent_node, id=edge_id, type="inheritance")

            parent_node = child_node

        # Inject the defeasible evidence and the final hypothesis
        self.add_defeasible_edges(config)
        self._add_hypothesis_edge(config)

        # Attach standard metadata to the graph object itself
        self.graph.graph["topology"] = "linear_inheritance"
        self.graph.graph["reasoning_mode"] = "inheritance"
        
        # Using getattr as a safe fallback in case config.root_concept isn't set yet
        self.graph.graph["root_concept"] = getattr(config, 'root_concept', config.root_name)

        return self.graph


class TreeInheritanceGenerator(GraphStructureGenerator):
    """
    Generates complex inheritance trees using stochastic branching (Galton-Watson).
    Phases: Skeleton -> Property Rules -> Defeasible Edges -> Hypothesis.
    """

    def __init__(self):
        super().__init__()
        self.leaf_nodes = []        # Track the ends of the chains for hypotheses
        self.property_node = ""     # The core property being reasoned about
        self.hypothesis_leaf = None # The one leaf chosen for the hypothesis edge

    # ---------------------------------------------------------
    # PHASE 1: Skeleton Generation
    # ---------------------------------------------------------
    def _grow_chains_recursively(self, parent_node, current_depth, config: InheritanceGraphConfig):
        """Recursively builds the tree using branching decay and survival checks."""
        
        # 1. Base Case: Max Depth reached
        if current_depth >= config.max_depth:
            self.leaf_nodes.append(parent_node)
            return

        # 2. Extinction Roll (Survival Probability)
        if current_depth > 0:
            if random.random() > config.survival_prob:
                self.leaf_nodes.append(parent_node)
                return  # Chain dies here

        # 3. Calculate Branching Factor
        if current_depth == 0:
            num_children = config.root_breadth
        else:
            # Apply L-System decay logic: max_branching * (decay ^ (depth - 1))
            decayed_max = math.floor(config.max_branching * (config.branching_decay ** (current_depth - 1)))
            actual_max = max(config.min_branching, decayed_max)
            num_children = random.randint(config.min_branching, actual_max)

        # 4. Generate Children
        children_generated = 0
        for i in range(num_children):
            child_id = f"{parent_node}_sub_{uuid.uuid4().hex[:4]}"
            self.graph.add_node(child_id, type="class", layer=current_depth + 1)
            
            # Edge: Child -> Parent (Inheritance flows upwards)
            edge_id = str(uuid.uuid4())[:8]
            self.graph.add_edge(child_id, parent_node, id=edge_id, type="inheritance")
            
            # Recurse
            self._grow_chains_recursively(child_id, current_depth + 1, config)
            children_generated += 1

        # Fallback: if loop ran 0 times due to min_branching=0 roll
        if children_generated == 0:
            self.leaf_nodes.append(parent_node)

    def _build_skeleton(self, root_name, config: InheritanceGraphConfig):
        """Initializes the root and triggers recursion."""
        self.leaf_nodes = []
        self.graph.add_node(root_name, type="class", layer=0)
        self._grow_chains_recursively(root_name, 0, config)

    # ---------------------------------------------------------
    # PHASE 2 & 3 & 4: Properties, Defeasibility, Hypothesis
    # ---------------------------------------------------------
    def _add_property_and_base_rule(self, root_name):
        """Creates the target property and establishes the base rule at the root."""
        self.property_node = "Target_Property"
        self.graph.add_node(self.property_node, type="property", layer=0)
        
        # Base Premise: The Root Class HAS the target property
        self.graph.add_edge(
            root_name, 
            self.property_node, 
            id=str(uuid.uuid4())[:8], 
            type="has_property"
        )

    def _add_hypothesis_edges(self):
        """
        Selects one leaf as the hypothesis subject and records it so
        _add_defeasible_edges can exclude it from the eligible pool.

        With 80% probability the deepest leaf (highest layer) is chosen;
        otherwise a uniformly random leaf is picked.  Ties at max depth are
        broken by uniform random choice among the tied nodes.
        """
        if not self.leaf_nodes:
            return

        if random.random() < 0.8:
            max_depth = max(self.graph.nodes[n].get("layer", 0) for n in self.leaf_nodes)
            deepest = [n for n in self.leaf_nodes if self.graph.nodes[n].get("layer", 0) == max_depth]
            self.hypothesis_leaf = random.choice(deepest)
        else:
            self.hypothesis_leaf = random.choice(self.leaf_nodes)

        self.graph.add_edge(
            self.hypothesis_leaf,
            self.property_node,
            id=str(uuid.uuid4())[:8],
            type="hypothesis_edge",
        )

    def _add_defeasible_edges(self, sparsity_factor):
        """
        Attaches 'yes/no' defeasible evidence to class nodes.
        Eligible: any class node with layer > 0 except the hypothesis leaf.
        Previously only intermediate (non-leaf) nodes were used; now all
        non-root non-hypothesis nodes are eligible.
        Serials are assigned in ascending layer order.
        """
        eligible = [
            n for n, d in self.graph.nodes(data=True)
            if d.get("type") == "class"
            and d.get("layer", 0) > 0
            and n != self.hypothesis_leaf
        ]
        eligible.sort(key=lambda n: self.graph.nodes[n].get("layer", 0))

        serial_counter = 1
        for node in eligible:
            if random.random() < sparsity_factor:
                edge_type = random.choice(["defeasible_has_property", "defeasible_lacks_property"])
                self.graph.add_edge(
                    node,
                    self.property_node,
                    id=str(uuid.uuid4())[:8],
                    type=edge_type,
                    serial=serial_counter,
                )
                serial_counter += 1

    # ---------------------------------------------------------
    # ORCHESTRATION
    # ---------------------------------------------------------
    def generate_structure(self, config: GraphGenerationConfig):
        self.clear()
        self.hypothesis_leaf = None

        # 1. Base Skeleton
        self._build_skeleton(config.root_name, config.inheritance_config)

        # 2. Add the Property Node and Link Root
        self._add_property_and_base_rule(config.root_name)

        # 3. Pick the hypothesis leaf first so defeasible injection can exclude it.
        self._add_hypothesis_edges()

        # 4. Add Defeasible Evidence (Using Sparsity Factor)
        self._add_defeasible_edges(config.sparsity_factor)

        # Finalize Metadata
        self.graph.graph["topology"] = "tree_inheritance"
        self.graph.graph["reasoning_mode"] = "inheritance"
        self.graph.graph["root_concept"] = config.root_concept

        return self.graph


if __name__ == "__main__":
    import sys
    import os

    # Get the absolute path of the parent directory
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

    # Add the parent directory to the system path
    sys.path.append(parent_dir)

    from utils.graphVisualizer import GraphVisualizer

    config = GraphGenerationConfig(
        chain_length=5,
        num_objects=4,
        num_irrelevant_attributes=2,
        root_name="Property_Root",
        sparsity_factor=0.4
    )

    gen = DefaultChainGenerator()
    graph = gen.generate_structure(config)

    # If you use the Visualizer from the previous step:
    # viz = GraphVisualizer(layout_type='multipartite') # Using multipartite to see layers
    # viz.visualize(graph, title="Default Reasoning with Distractors")

    # config = GraphGenerationConfig(
    #     max_depth=4,
    #     initial_breadth="large",  # Level 1 will have 10 nodes
    #     chain_survival_prob=0.4,
    #     root_name="living",
    # )

    # gen = LinearInheritanceGenerator()
    # graph = gen.generate_structure(config)

    # genT = TreeInheritanceGenerator()
    # graphT = genT.generate_structure(config)

    # viz = GraphVisualizer(layout_type="multipartite")
    # viz.visualize(graphT)

    # interactive graph
    from utils.interactive_visualizer import InteractiveGraphVisualizer
    interactive_viz = InteractiveGraphVisualizer()
    interactive_viz.visualize(graph, title="Defeasible Reasoning with Distractors")
