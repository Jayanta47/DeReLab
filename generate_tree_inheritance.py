"""
Batch dataset generator for the Tree Inheritance topology.

Produces two files, one per difficulty level:
    datasets/tree_inheritance_easy.json
    datasets/tree_inheritance_hard.json

Each difficulty uses all three preset shapes (jellyfish, balanced, bushy)
scaled to a comparable expected node count for that difficulty tier:

    easy  →  ~5–10  nodes   (small variants of each preset)
    hard  →  ~25–35 nodes   (original balanced/bushy + large jellyfish)

Defeasibility fix: any class node except the root (layer 0) and the
hypothesis leaf is eligible for defeasible edges.

Root concept selection
----------------------
ROOT_CONCEPT_WEIGHTS controls how often each semantic domain is chosen as the
root concept for a sample.  Modify the probability values here to shift the
distribution; they must sum to 1.0.

Serial order is randomised by default (serial_random=True) to maximise
configuration variety; set to False to restore ascending-layer ordering.

Reproducibility: each difficulty is seeded as  seed + {easy:0, hard:1}.
"""

import json
import os
import random
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Root-concept probability distribution for tree inheritance
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

from configs.graphConfig import GraphGenerationConfig, InheritanceGraphConfig, TopologyType
from data_generation.dedup_service import DuplicationChecker
from data_handler.kb_manager import KnowledgeBaseManager
from graph_generation.graph_generator import TreeInheritanceGenerator
from graph_populator.graph_populator import GraphPopulator
from question_generation.nl_generator import NaturalLanguageGenerator


# ---------------------------------------------------------------------------
# Per-preset-variant configuration
# ---------------------------------------------------------------------------

@dataclass
class PresetVariant:
    """One specific InheritanceGraphConfig variant, labelled by base preset name."""
    preset_name: str               # "jellyfish" | "balanced" | "bushy"
    config: InheritanceGraphConfig
    expected_nodes: tuple          # (approx_min, approx_max) — informational only


# ---------------------------------------------------------------------------
# Per-difficulty configuration
# ---------------------------------------------------------------------------

@dataclass
class DifficultyConfig:
    """
    Groups all preset variants that belong to one difficulty level.

    serial_random: True  → defeasible edges are revealed in a random order
                           (multiplies configuration space by k!).
                  False  → ascending layer order (original behaviour).

    preset_weights: optional dict mapping preset_name → relative weight for
                    weighted sampling.  Missing names default to weight 1.0.
                    When None, all variants are sampled uniformly.
    """
    preset_variants: list                              # list[PresetVariant]
    sparsity_factor_range: tuple                       # (min, max) — controls defeasible edge density
    serial_random: bool = True
    preset_weights: Optional[Dict[str, float]] = None  # e.g. {"jellyfish": 0.4, "bushy": 0.4, "balanced": 0.2}

    def sample_preset(self) -> "PresetVariant":
        if self.preset_weights:
            weights = [
                self.preset_weights.get(v.preset_name, 1.0)
                for v in self.preset_variants
            ]
            return random.choices(self.preset_variants, weights=weights, k=1)[0]
        return random.choice(self.preset_variants)


# ---------------------------------------------------------------------------
# Difficulty presets
# ---------------------------------------------------------------------------
#
# Expected node counts (excluding property node) per variant:
#
#   jellyfish-small  depth=3 breadth=4 survival=0.40  →  ~4–8  nodes
#   balanced-small   depth=2 breadth=2 full binary    →  exactly 7
#   bushy-small      depth=2 breadth=3 survival=0.70  →  ~5–9  nodes
#
#   jellyfish        depth=5 breadth=8 survival=0.40  →  ~13–16 nodes  (original)
#   balanced-medium  depth=3 breadth=2 full binary    →  exactly 15
#   bushy-medium     depth=3 breadth=3 survival=0.75  →  ~12–18 nodes
#
#   jellyfish-large  depth=7 breadth=12 survival=0.50 →  ~24–32 nodes
#   balanced         depth=4 breadth=2  full binary   →  exactly 31    (original)
#   bushy            depth=4 breadth=4  survival=0.80 →  ~20–35 nodes  (original)
#
# Within medium, jellyfish (≈14 nodes on average) is slightly harder than
# balanced-medium (exactly 15) because the stochastic branching of jellyfish
# creates more unpredictable reasoning chains despite a similar node budget.

DIFFICULTY_PRESETS: dict = {
    "easy": DifficultyConfig(
        preset_variants=[
            PresetVariant(
                preset_name="jellyfish",
                config=InheritanceGraphConfig(
                    max_depth=4, root_breadth=5,
                    min_branching=1, max_branching=1,
                    survival_prob=0.40, branching_decay=1.0,
                ),
                expected_nodes=(4, 8),
            ),
            PresetVariant(
                preset_name="balanced",
                config=InheritanceGraphConfig(
                    max_depth=3, root_breadth=2,
                    min_branching=2, max_branching=2,
                    survival_prob=1.0, branching_decay=1.0,
                ),
                expected_nodes=(7, 7),
            ),
            PresetVariant(
                preset_name="bushy",
                config=InheritanceGraphConfig(
                    max_depth=2, root_breadth=3,
                    min_branching=1, max_branching=2,
                    survival_prob=0.70, branching_decay=1.0,
                ),
                expected_nodes=(5, 9),
            ),
        ],
        sparsity_factor_range=(0.5, 0.8),
        serial_random=True,
    ),
    "hard": DifficultyConfig(
        preset_variants=[
            PresetVariant(
                preset_name="jellyfish",
                config=InheritanceGraphConfig(
                    max_depth=8, root_breadth=13,
                    min_branching=1, max_branching=1,
                    survival_prob=0.6, branching_decay=1.0,
                ),
                expected_nodes=(24, 50),
            ),
            PresetVariant(
                preset_name="balanced",
                config=InheritanceGraphConfig(     # original balanced preset
                    max_depth=4, root_breadth=3,
                    min_branching=2, max_branching=2,
                    survival_prob=1.0, branching_decay=1.0,
                ),
                expected_nodes=(46, 46),
            ),
            PresetVariant(
                preset_name="bushy",
                config=InheritanceGraphConfig(     # original bushy preset
                    max_depth=4, root_breadth=4,
                    min_branching=1, max_branching=4,
                    survival_prob=0.80, branching_decay=0.75,
                ),
                expected_nodes=(20, 60),
            ),
        ],
        sparsity_factor_range=(0.4, 0.6),
        serial_random=True,
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

    Example — generate 200 easy samples with custom preset mix:

        generate_all(BatchGenerationConfig(
            num_samples=200,
            difficulties=["easy"],
            difficulty_configs={
                "easy": DifficultyConfig(
                    preset_variants=[...],
                    preset_weights={"jellyfish": 0.4, "bushy": 0.4, "balanced": 0.2},
                    sparsity_factor_range=(0.30, 0.55),
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
# Extended generator with serial-randomisation support
# ---------------------------------------------------------------------------

class _TreeInheritanceSamplingGenerator(TreeInheritanceGenerator):
    """
    Extends TreeInheritanceGenerator to accept per-sample parameters
    (InheritanceGraphConfig, sparsity_factor, serial_random) directly,
    bypassing GraphGenerationConfig.

    Defeasibility: any non-root class node except the hypothesis leaf.
    Serial order:  random shuffle when serial_random=True (default),
                   ascending-layer order otherwise.
    """

    def generate_sample(
        self,
        root_name: str,
        root_concept: str,
        inheritance_config: InheritanceGraphConfig,
        sparsity_factor: float,
        serial_random: bool = True,
    ):
        self.clear()
        self.leaf_nodes = []
        self.hypothesis_leaf = None

        self._build_skeleton(root_name, inheritance_config)
        self._add_property_and_base_rule(root_name)
        self._add_hypothesis_edges()                         # picks hypothesis_leaf first
        self._add_defeasible_edges_sampler(sparsity_factor, serial_random)

        self.graph.graph["topology"] = "tree_inheritance"
        self.graph.graph["reasoning_mode"] = "inheritance"
        self.graph.graph["root_concept"] = root_concept
        return self.graph

    def _add_defeasible_edges_sampler(self, sparsity_factor: float, serial_random: bool):
        """
        Eligible nodes: all class nodes with layer > 0, excluding the hypothesis leaf.
        Order: shuffled (serial_random=True) or ascending layer (serial_random=False).
        """
        eligible = [
            n for n, d in self.graph.nodes(data=True)
            if d.get("type") == "class"
            and d.get("layer", 0) > 0
            and n != self.hypothesis_leaf
        ]

        selected = [n for n in eligible if random.random() < sparsity_factor]

        if serial_random:
            random.shuffle(selected)
        else:
            selected.sort(key=lambda n: self.graph.nodes[n].get("layer", 0))

        for serial, node in enumerate(selected, start=1):
            edge_type = random.choice(
                ["defeasible_has_property", "defeasible_lacks_property"]
            )
            self.graph.add_edge(
                node,
                self.property_node,
                id=str(uuid.uuid4())[:8],
                type=edge_type,
                serial=serial,
            )


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
    dedup = DuplicationChecker() if batch_config.dedup_enabled else None
    generator = _TreeInheritanceSamplingGenerator()

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

        preset = diff_cfg.sample_preset()
        concepts = list(batch_config.root_concept_weights.keys())
        probs = list(batch_config.root_concept_weights.values())
        root_concept = random.choices(concepts, weights=probs, k=1)[0]
        root_name = root_concept.capitalize()
        sparsity = random.uniform(*diff_cfg.sparsity_factor_range)

        graph = generator.generate_sample(
            root_name=root_name,
            root_concept=root_concept,
            inheritance_config=preset.config,
            sparsity_factor=sparsity,
            serial_random=diff_cfg.serial_random,
        )

        # ---- dedup check (on raw structural graph before population) ----
        if dedup is not None:
            fp = DuplicationChecker.compute_tree_fingerprint(graph, root_concept)
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
        populated, sentences, conversations = _run_pipeline(graph, kb)

        num_class_nodes = sum(
            1 for _, d in graph.nodes(data=True) if d.get("type") == "class"
        )

        samples.append({
            "id": len(samples),
            "config": {
                "topology": "tree_inheritance",
                "root_name": root_name,
                "root_concept": root_concept,
                "preset_name": preset.preset_name,
                "difficulty": difficulty,
                "serial_random": diff_cfg.serial_random,
                "sparsity_factor": round(sparsity, 4),
                "inheritance_config": {
                    "max_depth": preset.config.max_depth,
                    "root_breadth": preset.config.root_breadth,
                    "min_branching": preset.config.min_branching,
                    "max_branching": preset.config.max_branching,
                    "survival_prob": preset.config.survival_prob,
                    "branching_decay": preset.config.branching_decay,
                },
            },
            "graph_summary": {
                "num_class_nodes": num_class_nodes,
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
            "topology": "tree_inheritance",
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
                "preset_variants": [
                    {
                        "preset_name": v.preset_name,
                        "expected_nodes": list(v.expected_nodes),
                        "inheritance_config": {
                            "max_depth": v.config.max_depth,
                            "root_breadth": v.config.root_breadth,
                            "min_branching": v.config.min_branching,
                            "max_branching": v.config.max_branching,
                            "survival_prob": v.config.survival_prob,
                            "branching_decay": v.config.branching_decay,
                        },
                    }
                    for v in diff_cfg.preset_variants
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
            f"tree_inheritance_{difficulty}.json",
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
