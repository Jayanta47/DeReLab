"""
Batch dataset generator for the Default Reasoning topology.

Produces two files, one per difficulty level:
    datasets/default_reasoning_easy.json
    datasets/default_reasoning_hard.json

Each file contains a 'metadata' block and a 'samples' list.
Every sample is a full pipeline run:
    structure → population → NL sentences → question records → conversations.

Difficulty dimensions
---------------------
  easy  – short chain (3–5), few objects (2–3), minimal noise (1–2 irr attrs)
  hard  – long chain (7–12), more objects (4–6), more noise (2–4 irr attrs)

Root concept selection
----------------------
ROOT_CONCEPT_WEIGHTS controls how often each semantic domain is chosen as the
root concept for a sample.  Modify the probability values here to shift the
distribution; they must sum to 1.0.

Reproducibility: each difficulty level is seeded independently as
    seed + difficulty_index  (easy=+0, hard=+1).
"""

import json
import os
import random
from dataclasses import dataclass, field

from configs.graphConfig import DefaultGraphConfig, GraphGenerationConfig, TopologyType
from data_generation.dedup_service import DuplicationChecker
from data_handler.kb_manager import KnowledgeBaseManager
from graph_generation.graph_generator import DefaultChainGenerator
from graph_populator.graph_populator import GraphPopulator
from question_generation.nl_generator import NaturalLanguageGenerator


# ---------------------------------------------------------------------------
# Root-concept probability distribution for default reasoning
# ---------------------------------------------------------------------------
# Keys are semantic domains recognised by domain_attributes_values.json.
# Probabilities must sum to 1.0.
ROOT_CONCEPT_WEIGHTS: dict = {
    "Object": 1.0,
    # "Animal": 0.25,
    # "Plant": 0.15,
    # "Tool":   0.10,
}


# ---------------------------------------------------------------------------
# Per-difficulty configuration
# ---------------------------------------------------------------------------

@dataclass
class DefaultDifficultyConfig:
    """
    Parameter ranges sampled uniformly for each generated sample.

    chain_length_range       : (min, max) number of property nodes in the chain.
    num_objects_range        : (min, max) number of entity (object) nodes.
    num_irr_attr_range       : (min, max) irrelevant attribute nodes added as noise.
    sparsity_factor_range    : (min, max) probability that a (entity, chain-node)
                               pair receives a defeasible evidence edge.
    max_hypothesis_sections  : hard cap on total conversation sections (Fix 8b).
    max_updates_per_section  : hard cap on update turns per section (Fix 8b).
    max_total_turns          : safety cap on total conversation turns (Fix 8b).
    type4_enabled            : enable Type-4 source-priority episodes (Fix 1).
    type4_object_fraction    : fraction of objects reserved for Type-4 (Fix 1).
    """
    chain_length_range:    tuple
    num_objects_range:     tuple
    num_irr_attr_range:    tuple
    sparsity_factor_range: tuple
    max_hypothesis_sections:  int  = 3
    max_updates_per_section:  int  = 6
    max_total_turns:          int  = 40
    type4_enabled:            bool = True
    type4_object_fraction:    float = 0.5
    reveal_order_random_prob: float = 0.5


DIFFICULTY_PRESETS: dict = {
    "easy": DefaultDifficultyConfig(
        chain_length_range=(5, 10),
        num_objects_range=(2, 3),
        num_irr_attr_range=(1, 2),
        sparsity_factor_range=(0.30, 0.50),
        max_hypothesis_sections=2,
        max_updates_per_section=None,
        max_total_turns=None,   
        type4_enabled=True,
        type4_object_fraction=0.5,
        reveal_order_random_prob=0.5,
    ),
    "hard": DefaultDifficultyConfig(
        chain_length_range=(10, 20),
        num_objects_range=(4, 7),
        num_irr_attr_range=(2, 4),
        sparsity_factor_range=(0.40, 0.70),
        max_hypothesis_sections=5,
        max_updates_per_section=None,
        max_total_turns=None,  
        type4_enabled=True,
        type4_object_fraction=0.4,
        reveal_order_random_prob=0.5,
    ),
}

_DIFFICULTY_SEED_OFFSETS: dict = {"easy": 0, "hard": 0}


# ---------------------------------------------------------------------------
# Top-level batch configuration
# ---------------------------------------------------------------------------

@dataclass
class BatchGenerationConfig:
    num_samples: int = 150
    difficulties: list = field(default_factory=lambda: ["easy", "hard"])
    root_concept_weights: dict = field(default_factory=lambda: dict(ROOT_CONCEPT_WEIGHTS))
    # How many named sources to include per graph for Type-4 source-priority
    # disputes.  (min, max): if min == max, always use that count; otherwise
    # draw uniformly from [min, max] per graph.  Minimum meaningful value is 2.
    num_sources_range: tuple = (2, 4)
    dedup_enabled: bool = True
    seed: int = 42
    output_dir: str = "datasets"
    max_consecutive_duplicates: int = 15
    use_conceptnet: bool = False
    allow_nonsensical_entities: bool = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_root_concept(weights: dict) -> str:
    """Weighted random draw over the root_concept_weights dict."""
    concepts = list(weights.keys())
    probs = list(weights.values())
    return random.choices(concepts, weights=probs, k=1)[0]


def _run_pipeline(graph, kb: KnowledgeBaseManager):
    """Populate → NL sentences → conversations (via NaturalLanguageGenerator dispatch)."""
    populated = GraphPopulator(kb_manager=kb).populate(graph=graph)
    nl = NaturalLanguageGenerator(kb_manager=kb)
    sentences = nl.generate_edge_sentences(graph=populated, include_defeasible=True)
    conversations = nl.build_conversation_dataset(graph=populated)
    generation_report = nl.get_default_generation_report()
    return populated, sentences, conversations, generation_report


def _count_defeasible(graph) -> int:
    return sum(
        1 for _, _, d in graph.edges(data=True)
        if d.get("type") in ("defeasible_has_property", "defeasible_lacks_property")
    )


def _count_nodes_by_type(graph, node_type: str) -> int:
    return sum(1 for _, d in graph.nodes(data=True) if d.get("type") == node_type)


# ---------------------------------------------------------------------------
# Per-difficulty generation loop
# ---------------------------------------------------------------------------

def _generate_for_difficulty(
    difficulty: str,
    batch_config: BatchGenerationConfig,
    kb: KnowledgeBaseManager,
) -> dict:
    diff_cfg = DIFFICULTY_PRESETS[difficulty]
    dedup = DuplicationChecker() if batch_config.dedup_enabled else None
    generator = DefaultChainGenerator()

    samples: list = []
    consecutive_dupes = 0
    total_attempts = 0
    saturated = False

    target = batch_config.num_samples
    progress_interval = max(1, target // 5)

    print(f"\n[{difficulty.upper()}] Generating {target} samples  "
          f"(dedup={'on' if dedup else 'off'})")

    while len(samples) < target:
        total_attempts += 1

        root_concept = _sample_root_concept(batch_config.root_concept_weights)
        chain_length = random.randint(*diff_cfg.chain_length_range)
        num_objects = random.randint(*diff_cfg.num_objects_range)
        num_irr_attr = random.randint(*diff_cfg.num_irr_attr_range)
        sparsity = random.uniform(*diff_cfg.sparsity_factor_range)

        config = GraphGenerationConfig(
            topology=TopologyType.DEFAULT,
            root_name="Property_Root",
            root_concept=root_concept,
            sparsity_factor=sparsity,
            allow_nonsensical_entities=batch_config.allow_nonsensical_entities,
            use_conceptnet=batch_config.use_conceptnet,
            default_config=DefaultGraphConfig(
                chain_length=chain_length,
                num_objects=num_objects,
                num_irrelevant_attributes=num_irr_attr,
                num_sources_range=batch_config.num_sources_range,
                type4_enabled=diff_cfg.type4_enabled,
                type4_object_fraction=diff_cfg.type4_object_fraction,
                max_hypothesis_sections=diff_cfg.max_hypothesis_sections,
                max_updates_per_section=diff_cfg.max_updates_per_section,
                max_total_turns=diff_cfg.max_total_turns,
                reveal_order_random_prob=diff_cfg.reveal_order_random_prob,
            ),
        )

        graph = generator.generate_structure(config)

        # ---- dedup check (on raw structural graph) ----
        if dedup is not None:
            fp = DuplicationChecker.compute_default_fingerprint(graph, root_concept)
            if not dedup.register(fp):
                consecutive_dupes += 1
                if consecutive_dupes >= batch_config.max_consecutive_duplicates:
                    print(
                        f"[{difficulty.upper()}] Saturation: "
                        f"{consecutive_dupes} consecutive duplicates after "
                        f"{total_attempts} attempts. "
                        f"Stopping at {len(samples)}/{target} samples."
                    )
                    saturated = True
                    break
                continue
            consecutive_dupes = 0

        # ---- full pipeline ----
        try:
            populated, sentences, conversations, gen_report = _run_pipeline(graph, kb)
        except Exception as exc:
            print(f"  [WARN] Pipeline failed for attempt {total_attempts}: {exc}")
            continue

        if not conversations:
            # No questions could be generated for this graph; skip silently.
            continue

        samples.append({
            "id": len(samples),
            "config": {
                "topology": "default",
                "root_concept": root_concept,
                "difficulty": difficulty,
                "chain_length": chain_length,
                "num_objects": num_objects,
                "num_irr_attr": num_irr_attr,
                "sparsity_factor": round(sparsity, 4),
                "num_sources_range": list(batch_config.num_sources_range),
                "type4_enabled": diff_cfg.type4_enabled,
                "type4_object_fraction": diff_cfg.type4_object_fraction,
                "max_hypothesis_sections": diff_cfg.max_hypothesis_sections,
                "max_updates_per_section": diff_cfg.max_updates_per_section,
                "max_total_turns": diff_cfg.max_total_turns,
                "reveal_order_random_prob": diff_cfg.reveal_order_random_prob,
            },
            "graph_summary": {
                "num_property_nodes": _count_nodes_by_type(populated, "property"),
                "num_entity_nodes": _count_nodes_by_type(populated, "entity"),
                "num_irr_nodes": _count_nodes_by_type(populated, "irrelevant_property"),
                "num_nodes": populated.number_of_nodes(),
                "num_edges": populated.number_of_edges(),
                "num_defeasible_edges": _count_defeasible(graph),
                "records_by_type": gen_report.get("records_by_type", {}),
            },
            "edge_sentences": sentences,
            "conversations": conversations,
        })

        n = len(samples)
        if n % progress_interval == 0 or n == target:
            print(f"[{difficulty.upper()}]  {n}/{target} samples generated "
                  f"(attempts so far: {total_attempts})")

    dedup_stats = dedup.stats() if dedup else None

    return {
        "metadata": {
            "version": "1.0",
            "topology": "default",
            "difficulty": difficulty,
            "num_samples_requested": target,
            "num_samples_generated": len(samples),
            "total_attempts": total_attempts,
            "saturated": saturated,
            "seed": batch_config.seed + _DIFFICULTY_SEED_OFFSETS[difficulty],
            "root_concept_weights": batch_config.root_concept_weights,
            "dedup_enabled": batch_config.dedup_enabled,
            "max_consecutive_duplicates": batch_config.max_consecutive_duplicates,
            "difficulty_config": {
                "chain_length_range": list(diff_cfg.chain_length_range),
                "num_objects_range": list(diff_cfg.num_objects_range),
                "num_irr_attr_range": list(diff_cfg.num_irr_attr_range),
                "sparsity_factor_range": list(diff_cfg.sparsity_factor_range),
            },
            "dedup_stats": dedup_stats,
        },
        "samples": samples,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_all(batch_config: BatchGenerationConfig) -> None:
    os.makedirs(batch_config.output_dir, exist_ok=True)

    kb = KnowledgeBaseManager()
    if not batch_config.use_conceptnet:
        kb.conceptnet = None
    kb.allow_nonsensical_entities = batch_config.allow_nonsensical_entities

    all_datasets = {}

    for difficulty in batch_config.difficulties:
        diff_seed = batch_config.seed + _DIFFICULTY_SEED_OFFSETS[difficulty]
        random.seed(diff_seed)

        dataset = _generate_for_difficulty(difficulty, batch_config, kb)
        all_datasets[difficulty] = dataset

        out_path = os.path.join(
            batch_config.output_dir,
            f"default_reasoning_{difficulty}.json",
        )
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(dataset, fh, indent=2)

        n = dataset["metadata"]["num_samples_generated"]
        print(f"[{difficulty.upper()}] Saved {n} samples → {out_path}")

    # ---- Duplicate structure report ----
    print("\n=== Duplicate Graph Structure Report ===")
    for difficulty, dataset in all_datasets.items():
        stats = dataset["metadata"].get("dedup_stats")
        if stats:
            print(
                f"  [{difficulty.upper():6s}]  "
                f"unique={stats['unique_count']}  "
                f"duplicates={stats['total_duplicates']}  "
                f"attempts={stats['total_checked']}  "
                f"dup_rate={stats['duplicate_rate']*100:.1f}%"
            )
        else:
            n = dataset["metadata"]["num_samples_generated"]
            print(f"  [{difficulty.upper():6s}]  dedup disabled, {n} samples generated")

    print("\nAll difficulties complete.")


if __name__ == "__main__":
    generate_all(
        BatchGenerationConfig(
            num_samples=150,
            difficulties=["easy", "hard"],
            root_concept_weights={
                "Object": 1.0,
            },
            num_sources_range=(2, 4),
            dedup_enabled=True,
            seed=42,
            output_dir="datasets",
            max_consecutive_duplicates=15,
            use_conceptnet=False,
            allow_nonsensical_entities=True,
        )
    )
