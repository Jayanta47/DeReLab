import os
import random
import sys
import unittest

import networkx as nx


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from configs.graphConfig import (
    DefaultGraphConfig,
    GraphGenerationConfig,
    InheritanceGraphConfig,
    TopologyType,
)
from graph_generation import (
    DefaultChainGenerator,
    LinearInheritanceGenerator,
    TreeInheritanceGenerator,
)
from utils.interactive_visualizer import InteractiveGraphVisualizer


OUTPUT_DIR = os.path.join(CURRENT_DIR, "graph_outputs")


def _select_generator(topology: TopologyType):
    if topology == TopologyType.LINEAR_INHERITANCE:
        return LinearInheritanceGenerator()
    if topology == TopologyType.TREE_INHERITANCE:
        return TreeInheritanceGenerator()
    return DefaultChainGenerator()


TEST_CONFIGS = [

#     GraphGenerationConfig(
#         topology=TopologyType.DEFAULT,
#         root_name="Category_A",
#         root_concept="animal",
#         sparsity_factor=0.8,
#         default_config=DefaultGraphConfig(
#             chain_length=5,
#             num_objects=4,
#             num_irrelevant_attributes=2,
#             max_branching_factor=2,
#             branch_probability=0.8,
#         ),
#     ),
    GraphGenerationConfig(
        topology=TopologyType.LINEAR_INHERITANCE,
        root_name="Animal",
        root_concept="animal",
        inheritance_config=InheritanceGraphConfig(
            max_depth=6,
        ),
    ),
    # GraphGenerationConfig(
    #     topology=TopologyType.TREE_INHERITANCE,
    #     root_name="LivingThing",
    #     root_concept="animal",
    #     inheritance_config=InheritanceGraphConfig.from_preset("jellyfish"),
    #     sparsity_factor=0.6,
    # ),
]


def _build_output_path(config: GraphGenerationConfig, index: int) -> str:
    filename = f"{index:02d}_{config.topology.value}_graph.html"
    return os.path.join(OUTPUT_DIR, filename)


def run_graph_example(config: GraphGenerationConfig, index: int):
    generator = _select_generator(config.topology)
    graph = generator.generate_structure(config)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = _build_output_path(config, index)

    visualizer = InteractiveGraphVisualizer(output_file=output_path)
    visualizer.visualize(graph, title=f"Graph Test: {config.topology.value}")

    print("=" * 60)
    print(f"Topology: {config.topology.value}")
    print(f"Nodes: {graph.number_of_nodes()}")
    print(f"Edges: {graph.number_of_edges()}")
    print(f"Saved HTML: {output_path}")

    return graph


def main():
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Graph output dir: {OUTPUT_DIR}")

    for index, config in enumerate(TEST_CONFIGS, start=1):
        run_graph_example(config, index)


class GraphGenerationConfigTests(unittest.TestCase):
    def test_default_graph_config_uses_linear_defaults(self):
        config = GraphGenerationConfig()

        self.assertEqual(config.default_config.chain_length, 3)
        self.assertEqual(config.default_config.max_branching_factor, 1)
        self.assertEqual(config.default_config.branch_probability, 0.0)
        self.assertEqual(config.default_config.num_objects, 2)
        self.assertEqual(config.default_config.num_irrelevant_attributes, 1)

    def test_tree_inheritance_defaults_to_jellyfish_preset(self):
        config = GraphGenerationConfig()

        self.assertEqual(config.topology, TopologyType.TREE_INHERITANCE)
        self.assertEqual(config.inheritance_config.max_depth, 5)
        self.assertEqual(config.inheritance_config.root_breadth, 8)
        self.assertEqual(config.inheritance_config.min_branching, 1)
        self.assertEqual(config.inheritance_config.max_branching, 1)
        self.assertEqual(config.inheritance_config.survival_prob, 0.4)
        self.assertEqual(config.inheritance_config.branching_decay, 1.0)

    def test_unknown_preset_falls_back_to_jellyfish(self):
        preset = InheritanceGraphConfig.from_preset("unknown-preset")

        self.assertEqual(preset.max_depth, 5)
        self.assertEqual(preset.root_breadth, 8)
        self.assertEqual(preset.min_branching, 1)
        self.assertEqual(preset.max_branching, 1)
        self.assertEqual(preset.survival_prob, 0.4)
        self.assertEqual(preset.branching_decay, 1.0)


class GraphGeneratorTests(unittest.TestCase):
    def test_default_generator_uses_nested_default_config(self):
        config = _with_legacy_default_fields(
            GraphGenerationConfig(
                topology=TopologyType.DEFAULT,
                root_name="Category_A",
                root_concept="animal",
                sparsity_factor=0.4,
                default_config=DefaultGraphConfig(
                    chain_length=5,
                    max_branching_factor=1,
                    branch_probability=0.0,
                    num_objects=4,
                    num_irrelevant_attributes=2,
                ),
            )
        )

        graph = _select_generator(config.topology).generate_structure(config)

        property_nodes = [
            node for node, data in graph.nodes(data=True) if data.get("type") == "property"
        ]
        entity_nodes = [
            node for node, data in graph.nodes(data=True) if data.get("type") == "entity"
        ]
        irrelevant_nodes = [
            node
            for node, data in graph.nodes(data=True)
            if data.get("type") == "irrelevant_property"
        ]

        self.assertEqual(graph.graph["topology"], TopologyType.DEFAULT.value)
        self.assertEqual(graph.graph["reasoning_mode"], "defeasible")
        self.assertEqual(graph.graph["root_concept"], "animal")
        self.assertEqual(len(property_nodes), config.default_config.chain_length)
        self.assertEqual(len(entity_nodes), config.default_config.num_objects)
        self.assertEqual(
            len(irrelevant_nodes), config.default_config.num_irrelevant_attributes
        )

    def test_default_generator_branching_factor_one_stays_linear(self):
        config = _with_legacy_default_fields(
            GraphGenerationConfig(
                topology=TopologyType.DEFAULT,
                root_name="Category_A",
                root_concept="animal",
                sparsity_factor=1.0,
                default_config=DefaultGraphConfig(
                    chain_length=5,
                    max_branching_factor=1,
                    branch_probability=0.0,
                    num_objects=2,
                    num_irrelevant_attributes=0,
                ),
            )
        )

        graph = _select_generator(config.topology).generate_structure(config)
        property_nodes = [
            node for node, data in graph.nodes(data=True) if data.get("type") == "property"
        ]
        root_node = config.root_name
        leaf_nodes = [node for node in property_nodes if graph.out_degree(node) == 0]

        self.assertEqual(len(property_nodes), config.default_config.chain_length)
        self.assertEqual(len(leaf_nodes), 1)
        self.assertEqual(graph.out_degree(root_node), 1)
        self.assertEqual(
            nx.shortest_path_length(graph, root_node, leaf_nodes[0]),
            config.default_config.chain_length - 1,
        )

    def test_default_generator_branching_config_can_create_tree(self):
        random_seed = 7
        config = _with_legacy_default_fields(
            GraphGenerationConfig(
                topology=TopologyType.DEFAULT,
                root_name="Category_A",
                root_concept="animal",
                sparsity_factor=1.0,
                default_config=DefaultGraphConfig(
                    chain_length=7,
                    max_branching_factor=3,
                    branch_probability=0.9,
                    num_objects=3,
                    num_irrelevant_attributes=0,
                ),
            )
        )

        random_state = random.getstate()
        random.seed(random_seed)
        graph = _select_generator(config.topology).generate_structure(config)
        random.setstate(random_state)
        property_nodes = [
            node for node, data in graph.nodes(data=True) if data.get("type") == "property"
        ]
        leaf_nodes = [node for node in property_nodes if graph.out_degree(node) == 0]
        hypothesis_edges = [
            (source, target, data)
            for source, target, data in graph.edges(data=True)
            if data.get("type") == "hypothesis_edge"
        ]
        root_node = config.root_name

        self.assertEqual(len(property_nodes), config.default_config.chain_length)
        self.assertGreaterEqual(graph.out_degree(root_node), 1)
        self.assertGreater(len(leaf_nodes), 1)
        self.assertEqual(len(hypothesis_edges), config.default_config.num_objects)

        hypothesis_targets = {target for _, target, _ in hypothesis_edges}
        self.assertEqual(len(hypothesis_targets), 1)
        self.assertIn(next(iter(hypothesis_targets)), leaf_nodes)

    def test_linear_inheritance_generator_uses_inheritance_config(self):
        config = GraphGenerationConfig(
            topology=TopologyType.LINEAR_INHERITANCE,
            root_name="Animal",
            root_concept="animal",
            inheritance_config=InheritanceGraphConfig(
                max_depth=6,
                root_breadth=8,
                min_branching=1,
                max_branching=1,
                survival_prob=0.4,
                branching_decay=1.0,
            ),
        )

        graph = _select_generator(config.topology).generate_structure(config)

        class_nodes = [
            node for node, data in graph.nodes(data=True) if data.get("type") == "class"
        ]
        property_nodes = [
            node for node, data in graph.nodes(data=True) if data.get("type") == "property"
        ]
        hypothesis_edges = [
            (source, target, data)
            for source, target, data in graph.edges(data=True)
            if data.get("type") == "hypothesis_edge"
        ]

        self.assertEqual(graph.graph["topology"], TopologyType.LINEAR_INHERITANCE.value)
        self.assertEqual(graph.graph["reasoning_mode"], "inheritance")
        self.assertEqual(graph.graph["root_concept"], "animal")
        self.assertEqual(len(class_nodes), config.inheritance_config.max_depth + 1)
        self.assertEqual(len(property_nodes), 1)
        self.assertEqual(len(hypothesis_edges), 1)
        self.assertEqual(hypothesis_edges[0][0], f"Subclass_{config.inheritance_config.max_depth}")

    def test_tree_inheritance_generator_keeps_jellyfish_default(self):
        config = GraphGenerationConfig(
            topology=TopologyType.TREE_INHERITANCE,
            root_name="LivingThing",
            root_concept="animal",
        )

        graph = _select_generator(config.topology).generate_structure(config)

        class_nodes = [
            node for node, data in graph.nodes(data=True) if data.get("type") == "class"
        ]
        property_nodes = [
            node for node, data in graph.nodes(data=True) if data.get("type") == "property"
        ]
        root_children = list(graph.predecessors(config.root_name))
        hypothesis_edge_count = sum(
            1
            for _, _, data in graph.edges(data=True)
            if data.get("type") == "hypothesis_edge"
        )

        self.assertEqual(graph.graph["topology"], TopologyType.TREE_INHERITANCE.value)
        self.assertEqual(graph.graph["reasoning_mode"], "inheritance")
        self.assertEqual(graph.graph["root_concept"], "animal")
        self.assertEqual(config.inheritance_config.root_breadth, 8)
        self.assertGreaterEqual(len(class_nodes), 1 + config.inheritance_config.root_breadth)
        self.assertEqual(len(property_nodes), 1)
        self.assertEqual(len(root_children), config.inheritance_config.root_breadth)
        self.assertGreaterEqual(hypothesis_edge_count, 1)


if __name__ == "__main__":
    main()
