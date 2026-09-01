import json
import os

import networkx as nx

from configs.graphConfig import (
    DefaultGraphConfig,
    GraphGenerationConfig,
    InheritanceGraphConfig,
    TopologyType,
)
from data_handler.kb_manager import KnowledgeBaseManager
from graph_generation import (
    DefaultChainGenerator,
    LinearInheritanceGenerator,
    TreeInheritanceGenerator,
)
from graph_populator.graph_populator import GraphPopulator
from question_generation.nl_generator import NaturalLanguageGenerator
from utils.interactive_visualizer import InteractiveGraphVisualizer
import random
random.seed(88)

# Central toggles for sanity-check runs.
# `debug_stage` lets you stop after a specific pipeline checkpoint.
RUN_OPTIONS = {
    "single_run": True,
    "save_dataset": True,
    "render_interactive_html": False,
    "interactive_html_path": "defeasible_reasoning_graph.html",
    "show_sample_conversation": False,
    "debug_stage": "full",
    "render_structure_graph": False,
    "structure_graph_html_path": "structure_graph.html",
    "render_populated_graph": True,
    "populated_graph_html_path": "populated_graph.html",
    "render_final_graph": False,
    "final_graph_html_path": "defeasible_reasoning_graph.html",
}


PIPELINE_TEST_CONFIGS = {
    "default": GraphGenerationConfig(
        topology=TopologyType.DEFAULT,
        root_name="Property_Root",
        root_concept="Object",
        sparsity_factor=0.4,
        allow_nonsensical_entities=True,
        use_conceptnet=False,
        default_config=DefaultGraphConfig(
            chain_length=8,
            max_branching_factor=1,
            branch_probability=0.0,
            num_objects=5,
            num_irrelevant_attributes=3,
            num_hypothesis_subjects=None,
            num_sources_range=(2, 4),
            type4_scenarios=["contradiction", "bypass"],
            type4_b_first_prob=0.7,
            # Fix 1: disjoint subject partitioning for Type 4
            type4_enabled=True,
            type4_object_fraction=0.5,
            # Fix 8b: conversation length caps (hard mode values for testing)
            max_hypothesis_sections=3,
            max_updates_per_section=6,
            max_total_turns=None,
            reveal_order_random_prob=0.5,
        ),
    ),
    "linear_inheritance": GraphGenerationConfig(
        topology=TopologyType.LINEAR_INHERITANCE,
        root_name="Animal",
        root_concept="animal",
        sparsity_factor=0.4,
        allow_nonsensical_entities=True,
        use_conceptnet=False,
        inheritance_config=InheritanceGraphConfig(
            max_depth=20,
            root_breadth=5,
            min_branching=1,
            max_branching=1,
            survival_prob=0.6,
            branching_decay=1.0,
        ),
    ),
    "tree_inheritance": GraphGenerationConfig(
        topology=TopologyType.TREE_INHERITANCE,
        root_name="Animal",
        root_concept="animal",
        sparsity_factor=0.4,
        allow_nonsensical_entities=True,
        use_conceptnet=False,
        inheritance_config=InheritanceGraphConfig.from_preset("jellyfish"),
    ),
}


def _select_generator(topology: TopologyType):
    # Pick the graph generator that matches the requested logical topology.
    if topology == TopologyType.LINEAR_INHERITANCE:
        return LinearInheritanceGenerator()
    if topology == TopologyType.TREE_INHERITANCE:
        return TreeInheritanceGenerator()
    return DefaultChainGenerator()


def _render_graph_if_enabled(graph, enabled: bool, output_path: str, title: str):
    # Keep visualization behind a flag so we can inspect intermediate stages on demand.
    if not enabled:
        return

    html_output_path = os.path.abspath(output_path)
    visualizer = InteractiveGraphVisualizer(output_file=html_output_path)
    visualizer.visualize(graph, title=title)
    print(f"Saved interactive graph to: {html_output_path}")


def run_single_example():
    # Single-run configuration used for both sanity checks and dataset generation.
    config = PIPELINE_TEST_CONFIGS["default"]

    # Stage 1: create the raw graph structure with defeasible edges already marked.
    generator = _select_generator(config.topology)
    graph = generator.generate_structure(config)
    _render_graph_if_enabled(
        graph,
        RUN_OPTIONS["render_structure_graph"],
        RUN_OPTIONS["structure_graph_html_path"],
        f"DeReLab Structure Graph ({config.topology.value})",
    )

    if RUN_OPTIONS["debug_stage"] == "structure":
        print("--- Stopped After Structure Generation ---")
        print(f"Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
        return

    # Stage 2: inject entity/property labels and metadata from the KB layer.
    kb = KnowledgeBaseManager()
    if not config.use_conceptnet:
        kb.conceptnet = None

    if config.allow_nonsensical_entities:
        kb.allow_nonsensical_entities = True

    populated_graph = GraphPopulator(kb_manager=kb).populate(graph=graph)
    _render_graph_if_enabled(
        populated_graph,
        RUN_OPTIONS["render_populated_graph"],
        RUN_OPTIONS["populated_graph_html_path"],
        f"DeReLab Populated Graph ({config.topology.value})",
    )

    if RUN_OPTIONS["debug_stage"] == "populated":
        print("--- Stopped After Graph Population ---")
        print(
            "Sample node labels: "
            f"{list(nx.get_node_attributes(populated_graph, 'label').items())[:8]}"
        )
        return

    # Stage 3: attach natural-language realizations to the graph edges.
    nl_generator = NaturalLanguageGenerator(kb_manager=kb)
    edge_sentences = nl_generator.generate_edge_sentences(graph=populated_graph, include_defeasible=True)

    if RUN_OPTIONS["debug_stage"] == "nl":
        print("--- Stopped After NL Generation ---")
        print(f"Generated edge sentences: {len(edge_sentences)}")
        sample_sentences = list(edge_sentences.values())[:8]
        for sentence in sample_sentences:
            print(f"- {sentence}")
        return

    # Stage 4: reveal defeasible updates in order and re-query PMPM after each step.
    conversations = nl_generator.build_conversation_dataset(graph = populated_graph)

    output_path = os.path.abspath("generated_defeasible_dataset.json")

    topology_config: dict = {}
    if config.topology == TopologyType.DEFAULT:
        dc = config.default_config
        topology_config = {
            "chain_length": dc.chain_length,
            "max_branching_factor": dc.max_branching_factor,
            "branch_probability": dc.branch_probability,
            "num_objects": dc.num_objects,
            "num_irrelevant_attributes": dc.num_irrelevant_attributes,
            "num_hypothesis_subjects": dc.num_hypothesis_subjects,
            "num_sources_range": list(dc.num_sources_range),
            "type4_scenarios": dc.type4_scenarios,
            "type4_b_first_prob": dc.type4_b_first_prob,
            "type4_enabled": dc.type4_enabled,
            "type4_object_fraction": dc.type4_object_fraction,
            "max_hypothesis_sections": dc.max_hypothesis_sections,
            "max_updates_per_section": dc.max_updates_per_section,
            "max_total_turns": dc.max_total_turns,
            "reveal_order_random_prob": dc.reveal_order_random_prob,
        }
    else:
        topology_config = {
            "max_depth": config.inheritance_config.max_depth,
            "root_breadth": config.inheritance_config.root_breadth,
            "min_branching": config.inheritance_config.min_branching,
            "max_branching": config.inheritance_config.max_branching,
            "survival_prob": config.inheritance_config.survival_prob,
            "branching_decay": config.inheritance_config.branching_decay,
        }

    payload = {
        "config": {
            "topology": config.topology.value,
            "root_name": config.root_name,
            "root_concept": config.root_concept,
            "sparsity_factor": config.sparsity_factor,
            "allow_nonsensical_entities": config.allow_nonsensical_entities,
            "use_conceptnet": config.use_conceptnet,
            "topology_config": topology_config,
        },
        "graph_summary": {
            "num_nodes": populated_graph.number_of_nodes(),
            "num_edges": populated_graph.number_of_edges(),
            "sample_node_labels": list(nx.get_node_attributes(populated_graph, "label").items())[:8],
        },
        "edge_sentences": edge_sentences,
        "conversations": conversations,
    }

    # Persist the final dataset artifact if this run is not visualization-only.
    if RUN_OPTIONS["save_dataset"]:
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    # Final graph view is useful once labels and NL sentences have been attached.
    _render_graph_if_enabled(
        populated_graph,
        RUN_OPTIONS["render_final_graph"] or RUN_OPTIONS["render_interactive_html"],
        RUN_OPTIONS["final_graph_html_path"],
        f"DeReLab Graph Sanity Check ({config.topology.value})",
    )

    # Print a compact summary so single-run debugging stays easy from the terminal.
    print("--- Dataset Generation Complete ---")
    if RUN_OPTIONS["save_dataset"]:
        print(f"Saved dataset to: {output_path}")
    print(f"Generated conversations: {len(conversations)}")
    if conversations and RUN_OPTIONS["show_sample_conversation"]:
        print("--- Sample Conversation ---")
        print(json.dumps(conversations, indent=2))


if __name__ == "__main__":
    run_single_example()
