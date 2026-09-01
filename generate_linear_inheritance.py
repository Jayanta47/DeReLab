"""
Batch dataset generator for the Linear Inheritance topology.

Produces two files, one per difficulty level:
    datasets/linear_inheritance_easy.json
    datasets/linear_inheritance_hard.json

Each file contains a 'metadata' block and a 'samples' list.
Every sample is a full pipeline run: structure → population → NL sentences → conversations.

Root concept selection
----------------------
ROOT_CONCEPT_WEIGHTS controls how often each semantic domain is chosen as the
root concept for a sample.  Modify the probability values here to shift the
distribution; they must sum to 1.0.

Reproducibility: each difficulty level is seeded independently as
    seed + difficulty_index  (easy=+0, hard=+1)
so changing num_samples for one difficulty does not perturb the others.
"""

import json
import os
import random
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Root-concept probability distribution for linear inheritance
# ---------------------------------------------------------------------------
# Modify probability values here; they must sum to 1.0.
ROOT_CONCEPT_WEIGHTS: dict = {
    "animal": 0.2,
    "mammal":  0.15,
    "vertebrate":   0.15,
    "plant": 0.15,
    "insect": 0.15,
    "microorganism": 0.10,
    "bird": 0.10,
}

from data_generation.dedup_service import DuplicationChecker
from data_handler.kb_manager import KnowledgeBaseManager
from graph_generation.graph_generator import LinearInheritanceGenerator
from graph_populator.graph_populator import GraphPopulator
from question_generation.nl_generator import NaturalLanguageGenerator


# ---------------------------------------------------------------------------
# Per-chain-length configuration
# ---------------------------------------------------------------------------

@dataclass
class ChainLengthConfig:
    """
    Difficulty parameters for a single chain length.

    num_defeasible_fraction_range controls how many defeasible edges are
    injected relative to chain_length.  Shorter chains within a difficulty
    band use a higher fraction to keep reasoning complexity comparable.
    """
    chain_length: int
    num_defeasible_fraction_range: tuple  # (min_frac, max_frac)


# ---------------------------------------------------------------------------
# Per-difficulty configuration
# ---------------------------------------------------------------------------

@dataclass
class DifficultyConfig:
    """
    Groups all chain-length variants that belong to one difficulty level.

    serial_random: when True the reveal order of defeasible edges is
        randomised (shuffled), multiplying the configuration space by k!
        where k is the number of defeasible edges.  Set to False to restore
        the original ascending-depth-order serial assignment.

    chain_length_weights: optional dict mapping chain_length → relative weight
        for weighted sampling.  Missing lengths default to weight 1.0.
        When None, all variants are sampled uniformly.
    """
    chain_length_configs: list                              # list[ChainLengthConfig]
    serial_random: bool = True
    sparsity_factor_range: tuple = (0.3, 0.5)              # reserved for future use
    chain_length_weights: Optional[Dict[int, float]] = None # e.g. {3: 0.5, 4: 0.3, 5: 0.2}

    def sample_chain_config(self) -> "ChainLengthConfig":
        if self.chain_length_weights:
            weights = [
                self.chain_length_weights.get(c.chain_length, 1.0)
                for c in self.chain_length_configs
            ]
            return random.choices(self.chain_length_configs, weights=weights, k=1)[0]
        return random.choice(self.chain_length_configs)


# Shorter chains within a difficulty get a higher defeasible fraction so that
# absolute reasoning complexity stays comparable across chain lengths.
DIFFICULTY_PRESETS: dict = {
    "easy": DifficultyConfig(
        chain_length_configs=[
            ChainLengthConfig(chain_length=4, num_defeasible_fraction_range=(0.80, 1)),
            ChainLengthConfig(chain_length=5, num_defeasible_fraction_range=(0.60, 0.80)),
            ChainLengthConfig(chain_length=6, num_defeasible_fraction_range=(0.50, 0.750)),
        ],
        serial_random=True,
        sparsity_factor_range=(0.2, 0.4),
    ),
    "hard": DifficultyConfig(
        chain_length_configs=[
            ChainLengthConfig(chain_length=20, num_defeasible_fraction_range=(0.50, 0.85)),
            ChainLengthConfig(chain_length=22, num_defeasible_fraction_range=(0.50, 0.80)),
            ChainLengthConfig(chain_length=26, num_defeasible_fraction_range=(0.50, 0.75)),
            ChainLengthConfig(chain_length=30, num_defeasible_fraction_range=(0.50, 0.70)),
        ],
        serial_random=True,
        sparsity_factor_range=(0.4, 0.6),
    ),
}

_DIFFICULTY_SEED_OFFSETS = {"easy": 0, "hard": 0}


# ---------------------------------------------------------------------------
# Top-level batch configuration
# ---------------------------------------------------------------------------

@dataclass
class BatchGenerationConfig:
    """
    Top-level config for a batch generation run.

    difficulty_configs: if provided, overrides DIFFICULTY_PRESETS for the
        named difficulties.  Pass a dict mapping difficulty name →
        DifficultyConfig.  Unspecified difficulties still fall back to
        DIFFICULTY_PRESETS (if the key exists there) or raise KeyError.

    Example — generate 200 easy samples with custom chain-length mix:

        generate_all(BatchGenerationConfig(
            num_samples=200,
            difficulties=["easy"],
            difficulty_configs={
                "easy": DifficultyConfig(
                    chain_length_configs=[
                        ChainLengthConfig(chain_length=3, num_defeasible_fraction_range=(0.5, 0.8)),
                        ChainLengthConfig(chain_length=5, num_defeasible_fraction_range=(0.3, 0.6)),
                    ],
                    chain_length_weights={3: 0.6, 5: 0.4},
                ),
            },
        ))
    """
    num_samples: int = 150
    difficulties: list = field(default_factory=lambda: ["easy", "hard"])
    root_concept_weights: dict = field(default_factory=lambda: dict(ROOT_CONCEPT_WEIGHTS))
    dedup_enabled: bool = True
    seed: int = 42
    output_dir: str = "datasets"
    max_consecutive_duplicates: int = 10
    use_conceptnet: bool = False
    allow_nonsensical_entities: bool = True
    difficulty_configs: Optional[Dict[str, DifficultyConfig]] = None  # overrides DIFFICULTY_PRESETS


# ---------------------------------------------------------------------------
# Extended generator: direct parameter control, no config object needed
# ---------------------------------------------------------------------------

class _LinearInheritanceSamplingGenerator(LinearInheritanceGenerator):
    """
    Extends LinearInheritanceGenerator so that chain_length, defeasible edge
    density, and serial ordering can be controlled directly — bypassing
    GraphGenerationConfig / InheritanceGraphConfig.
    """

    def generate_sample(
        self,
        chain_length: int,
        root_name: str,
        root_concept: str,
        num_defeasible_fraction_range: tuple,
        serial_random: bool = True,
    ):
        """
        Build one raw graph.

        serial_random=True  → defeasible edges are revealed in a random
                              order (serials are assigned after shuffling).
        serial_random=False → serials follow ascending depth order
                              (original LinearInheritanceGenerator behaviour).
        """
        self.clear()
        self.property_node = "property_node"

        # Root class node + shared property node
        self.graph.add_node(root_name, type="class", layer=0)
        self.graph.add_node(self.property_node, type="property", layer=0)
        self.graph.add_edge(
            root_name,
            self.property_node,
            id=str(uuid.uuid4())[:8],
            type="has_property",
        )

        # Backbone: Subclass_1 → Subclass_2 → … → Subclass_{chain_length}
        parent = root_name
        for i in range(1, chain_length + 1):
            child = f"Subclass_{i}"
            self.graph.add_node(child, type="class", layer=i)
            self.graph.add_edge(
                child, parent, id=str(uuid.uuid4())[:8], type="inheritance"
            )
            parent = child

        # Defeasible edges on intermediate nodes (depth 1 … chain_length-1)
        available_depths = list(range(1, chain_length))
        if available_depths:
            lo_frac, hi_frac = num_defeasible_fraction_range
            min_edges = max(1, int(lo_frac * chain_length))
            max_edges = max(min_edges, int(hi_frac * chain_length))
            max_edges = min(max_edges, len(available_depths))
            min_edges = min(min_edges, max_edges)

            num_edges = random.randint(min_edges, max_edges)
            selected_depths = random.sample(available_depths, num_edges)

            if serial_random:
                random.shuffle(selected_depths)   # random reveal order
            else:
                selected_depths.sort()            # ascending-depth reveal order

            for serial, depth in enumerate(selected_depths, start=1):
                edge_type = random.choice(
                    ["defeasible_has_property", "defeasible_lacks_property"]
                )
                self.graph.add_edge(
                    f"Subclass_{depth}",
                    self.property_node,
                    id=str(uuid.uuid4())[:8],
                    type=edge_type,
                    serial=serial,
                )

        # Hypothesis edge from the leaf node
        self.graph.add_edge(
            f"Subclass_{chain_length}",
            self.property_node,
            id=str(uuid.uuid4())[:8],
            type="hypothesis_edge",
        )

        self.graph.graph["topology"] = "linear_inheritance"
        self.graph.graph["reasoning_mode"] = "inheritance"
        self.graph.graph["root_concept"] = root_concept
        return self.graph


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def _run_pipeline(graph, kb: KnowledgeBaseManager):
    """Populate → NL sentences → conversations."""
    populated = GraphPopulator(kb_manager=kb).populate(graph=graph)
    nl = NaturalLanguageGenerator(kb_manager=kb)
    sentences = nl.generate_edge_sentences(graph=populated, include_defeasible=True)
    conversations = nl.build_conversation_dataset(graph=populated)
    return populated, sentences, conversations


def _count_defeasible(graph) -> int:
    return sum(
        1 for _, _, d in graph.edges(data=True)
        if d.get("type") in ("defeasible_has_property", "defeasible_lacks_property")
    )


# ---------------------------------------------------------------------------
# Per-difficulty generation loop
# ---------------------------------------------------------------------------

def _generate_for_difficulty(
    difficulty: str,
    batch_config: BatchGenerationConfig,
    kb: KnowledgeBaseManager,
) -> dict:
    custom = (batch_config.difficulty_configs or {}).get(difficulty)
    diff_cfg = custom if custom is not None else DIFFICULTY_PRESETS[difficulty]

    # One dedup checker per difficulty — never shared across difficulties
    dedup = DuplicationChecker() if batch_config.dedup_enabled else None

    generator = _LinearInheritanceSamplingGenerator()
    samples: list = []
    consecutive_dupes = 0
    total_attempts = 0
    saturated = False

    target = batch_config.num_samples
    progress_interval = max(1, target // 5)

    print(f"\n[{difficulty.upper()}] Generating {target} samples  "
          f"(dedup={'on' if dedup else 'off'}, "
          f"serial_random={diff_cfg.serial_random})")

    while len(samples) < target:
        total_attempts += 1

        chain_cfg = diff_cfg.sample_chain_config()
        concepts = list(batch_config.root_concept_weights.keys())
        probs = list(batch_config.root_concept_weights.values())
        root_concept = random.choices(concepts, weights=probs, k=1)[0]
        root_name = root_concept.capitalize()

        graph = generator.generate_sample(
            chain_length=chain_cfg.chain_length,
            root_name=root_name,
            root_concept=root_concept,
            num_defeasible_fraction_range=chain_cfg.num_defeasible_fraction_range,
            serial_random=diff_cfg.serial_random,
        )

        # ---- dedup check (fingerprint computed on the raw structural graph) ----
        if dedup is not None:
            fp = DuplicationChecker.compute_fingerprint(
                graph, chain_cfg.chain_length, root_concept
            )
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
            consecutive_dupes = 0   # reset on any accepted sample

        # ---- full pipeline ----
        populated, sentences, conversations = _run_pipeline(graph, kb)

        samples.append({
            "id": len(samples),
            "config": {
                "topology": "linear_inheritance",
                "root_name": root_name,
                "root_concept": root_concept,
                "chain_length": chain_cfg.chain_length,
                "num_defeasible_edges": _count_defeasible(graph),
                "difficulty": difficulty,
                "serial_random": diff_cfg.serial_random,
                "num_defeasible_fraction_range": list(chain_cfg.num_defeasible_fraction_range),
            },
            "graph_summary": {
                "num_nodes": populated.number_of_nodes(),
                "num_edges": populated.number_of_edges(),
                "num_defeasible_edges": _count_defeasible(graph),
            },
            "edge_sentences": sentences,
            "conversations": conversations,
        })

        n = len(samples)
        if n % progress_interval == 0 or n == target:
            print(f"[{difficulty.upper()}]  {n}/{target} samples generated "
                  f"(attempts so far: {total_attempts})")

    return {
        "metadata": {
            "version": "1.0",
            "topology": "linear_inheritance",
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
                "serial_random": diff_cfg.serial_random,
                "sparsity_factor_range": list(diff_cfg.sparsity_factor_range),
                "chain_length_configs": [
                    {
                        "chain_length": c.chain_length,
                        "num_defeasible_fraction_range": list(c.num_defeasible_fraction_range),
                    }
                    for c in diff_cfg.chain_length_configs
                ],
            },
            "dedup_stats": dedup.stats() if dedup else None,
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
        diff_seed = batch_config.seed + _DIFFICULTY_SEED_OFFSETS.get(
            difficulty, batch_config.difficulties.index(difficulty)
        )
        random.seed(diff_seed)

        dataset = _generate_for_difficulty(difficulty, batch_config, kb)
        all_datasets[difficulty] = dataset

        out_path = os.path.join(
            batch_config.output_dir,
            f"linear_inheritance_{difficulty}.json",
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
            root_concept_weights=ROOT_CONCEPT_WEIGHTS,
            dedup_enabled=True,
            seed=42,
            output_dir="datasets",
            max_consecutive_duplicates=10,
            use_conceptnet=False,
            allow_nonsensical_entities=True,
        )
    )
