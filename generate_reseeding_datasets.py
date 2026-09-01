"""
Reseeding robustness experiment — dataset generator.

Generates 30-sample "easy" datasets for both default_reasoning and
linear_inheritance topologies across 20 seeds.  The first seed is 42
(matching the main evaluation datasets).  The remaining 19 seeds are
drawn deterministically from a separate random stream so that re-running
this script always produces the same seed list.

Output layout
-------------
  datasets/reseeding/
    seeds.json                              ← manifest of all 20 seeds
    seed_42/
      default_reasoning_easy.json           ← 30 samples, seed 42
      linear_inheritance_easy.json          ← 30 samples, seed 42
    seed_<S2>/
      ...
    ...

Usage
-----
  python generate_reseeding_datasets.py               # all 20 seeds
  python generate_reseeding_datasets.py --seed_index 0  # single seed by position
  python generate_reseeding_datasets.py --only_topology default
  python generate_reseeding_datasets.py --num_samples 30 --output_dir datasets/reseeding
"""

import argparse
import json
import os
import random
import sys

import generate_default_reasoning as dr_gen
import generate_linear_inheritance as li_gen


# ---------------------------------------------------------------------------
# Seed configuration
# ---------------------------------------------------------------------------

_FIXED_SEED_0 = 42          # first seed matches existing evaluation datasets
_SEED_RNG_SEED = 7          # seeds the generator that picks the 19 extras
_NUM_SEEDS = 20
_SEED_RANGE = (1_000, 99_999)


def build_seed_list() -> list[int]:
    """Return the canonical list of 20 seeds (always the same order)."""
    rng = random.Random(_SEED_RNG_SEED)
    pool = [s for s in range(*_SEED_RANGE) if s != _FIXED_SEED_0]
    extras = rng.sample(pool, _NUM_SEEDS - 1)
    return [_FIXED_SEED_0] + extras


# ---------------------------------------------------------------------------
# Config — copied directly from the generator modules so the reseeding
# experiment uses identical settings.  Update here if the generators change.
# ---------------------------------------------------------------------------

# Copied from generate_default_reasoning.ROOT_CONCEPT_WEIGHTS
_DR_ROOT_WEIGHTS: dict = {
    "Object": 1.0,
}

# Copied from generate_linear_inheritance.ROOT_CONCEPT_WEIGHTS
_LI_ROOT_WEIGHTS: dict = {
    "animal":        0.20,
    "mammal":        0.15,
    "vertebrate":    0.15,
    "plant":         0.15,
    "insect":        0.15,
    "microorganism": 0.10,
    "bird":          0.10,
}


def _dr_config(seed: int, num_samples: int, output_dir: str) -> dr_gen.BatchGenerationConfig:
    # All fields match the BatchGenerationConfig defaults in
    # generate_default_reasoning.py — only seed, num_samples, difficulties,
    # and output_dir differ from the main generation run.
    return dr_gen.BatchGenerationConfig(
        num_samples=num_samples,
        difficulties=["easy"],
        root_concept_weights=_DR_ROOT_WEIGHTS,
        num_sources_range=(2, 4),
        dedup_enabled=True,
        seed=seed,
        output_dir=output_dir,
        max_consecutive_duplicates=15,
        use_conceptnet=False,
        allow_nonsensical_entities=True,
    )


def _li_config(seed: int, num_samples: int, output_dir: str) -> li_gen.BatchGenerationConfig:
    # All fields match the BatchGenerationConfig defaults in
    # generate_linear_inheritance.py — only seed, num_samples, difficulties,
    # and output_dir differ.
    return li_gen.BatchGenerationConfig(
        num_samples=num_samples,
        difficulties=["easy"],
        root_concept_weights=_LI_ROOT_WEIGHTS,
        dedup_enabled=True,
        seed=seed,
        output_dir=output_dir,
        max_consecutive_duplicates=10,
        use_conceptnet=False,
        allow_nonsensical_entities=True,
    )


# ---------------------------------------------------------------------------
# Per-seed generation
# ---------------------------------------------------------------------------

def generate_for_seed(
    seed: int,
    num_samples: int,
    base_output_dir: str,
    topologies: list[str],
) -> None:
    seed_dir = os.path.join(base_output_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    print(f"\n{'=' * 60}", flush=True)
    print(f"Seed {seed}  →  {seed_dir}", flush=True)

    if "default" in topologies:
        dr_gen.generate_all(_dr_config(seed, num_samples, seed_dir))

    if "linear_inheritance" in topologies:
        li_gen.generate_all(_li_config(seed, num_samples, seed_dir))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python generate_reseeding_datasets.py",
        description="Generate reseeding robustness datasets "
                    "(20 seeds × 30 samples × 2 topologies)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--output_dir", default="datasets/reseeding",
        help="Root output directory. Default: datasets/reseeding",
    )
    p.add_argument(
        "--num_samples", type=int, default=30,
        help="Samples per seed per topology. Default: 30",
    )
    p.add_argument(
        "--seed_index", type=int, default=None, metavar="I",
        help="Generate only the seed at position I in the 20-seed list (0-based). "
             "Omit to generate all 20.",
    )
    p.add_argument(
        "--only_topology", choices=["default", "linear_inheritance"], default=None,
        help="Generate only one topology. Default: both.",
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    seeds = build_seed_list()
    topologies = (
        [args.only_topology] if args.only_topology
        else ["default", "linear_inheritance"]
    )

    os.makedirs(args.output_dir, exist_ok=True)

    manifest_path = os.path.join(args.output_dir, "seeds.json")
    with open(manifest_path, "w") as fh:
        json.dump({"seeds": seeds, "seed_rng_seed": _SEED_RNG_SEED,
                   "fixed_seed_0": _FIXED_SEED_0}, fh, indent=2)
    print(f"Seed manifest: {manifest_path}")
    print(f"Seeds: {seeds}")

    if args.seed_index is not None:
        if args.seed_index >= len(seeds):
            print(f"ERROR: seed_index {args.seed_index} out of range (0–{len(seeds)-1})",
                  file=sys.stderr)
            sys.exit(1)
        target_seeds = [seeds[args.seed_index]]
        print(f"Generating seed index {args.seed_index} only: seed={target_seeds[0]}")
    else:
        target_seeds = seeds
        print(f"Generating all {len(target_seeds)} seeds × "
              f"{args.num_samples} samples × {len(topologies)} topologies = "
              f"{len(target_seeds) * args.num_samples * len(topologies)} total samples")

    for seed in target_seeds:
        generate_for_seed(
            seed=seed,
            num_samples=args.num_samples,
            base_output_dir=args.output_dir,
            topologies=topologies,
        )

    print(f"\nAll done. Output in: {args.output_dir}/")


if __name__ == "__main__":
    main()
