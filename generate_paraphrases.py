#!/usr/bin/env python3
"""
Paraphrase generator for DeReLab datasets.

For each dataset, produces two paraphrased copies by rewriting the
Pretext and Hypothesis in each conversation's opening turn.
Update facts, question templates, entity names, and ground-truth answers
are left unchanged so the existing evaluation pipeline runs unmodified.

What is paraphrased (turn 0 only):
  "Pretext: <rules/hierarchy> Hypothesis: <claim>. Question: ..."
                ↑ this part is rewritten                ↑ question kept fixed

Output files (written to --output_dir):
  {name}_para1.json   — first paraphrase version
  {name}_para2.json   — second paraphrase version

Batching (HuggingFace):
  --batch_size N  (default 5) batches N samples per LLM call.
  Total calls per dataset = ceil(150 / N).  For all 6 datasets and
  batch_size=5 that is 180 calls.

Batching (OpenAI):
  Same batch prompt structure.  No GPU required — runs on any CPU node.

Usage:
    # GPT-5.1 (default)
    python generate_paraphrases.py --provider openai

    # HuggingFace
    python generate_paraphrases.py --provider hf \\
        --model_id allenai/OLMo-2-1124-32B-Instruct

CHPC:
    sbatch slurm_paraphrase_openai.sh   # GPT-5.1, no GPU
    sbatch slurm_paraphrase.sh          # HuggingFace, GPU
"""

import argparse
import copy
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# OpenAI model wrapper — same .respond() interface as HuggingFaceModel
# ---------------------------------------------------------------------------

class _SimpleResponse:
    def __init__(self, text: str):
        self.response_text = text


class _OpenAIParaphraseModel:
    """Thin OpenAI wrapper that exposes the same .respond() interface as HuggingFaceModel."""

    def __init__(self, model_id: str, api_key: str | None,
                 max_new_tokens: int, temperature: float):
        import openai
        self._client = openai.OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY")
        )
        self._model_id = model_id
        self._max_tokens = max_new_tokens
        self._temperature = temperature

    def respond(self, messages: list) -> _SimpleResponse:
        response = self._client.chat.completions.create(
            model=self._model_id,
            max_completion_tokens=self._max_tokens,
            temperature=self._temperature,
            messages=[{"role": "system", "content": _SYSTEM_PROMPT}] + messages,
        )
        return _SimpleResponse(response.choices[0].message.content.strip())


# ---------------------------------------------------------------------------
# Dataset list
# ---------------------------------------------------------------------------

ALL_DATASETS = [
    "linear_inheritance_easy.json",
    "linear_inheritance_hard.json",
    "tree_inheritance_easy.json",
    "tree_inheritance_hard.json",
    "default_reasoning_easy.json",
    "default_reasoning_hard.json",
]

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a precise paraphrasing assistant. "
    "You rewrite text while preserving every logical relationship and fact exactly. "
    "Entity names (capitalised proper nouns such as 'Goatels', 'Oenamo') must remain "
    "character-for-character identical. Never add, remove, or change any fact."
)

_USER_TEMPLATE = """\
For each BLOCK below, write exactly two paraphrases.
Each paraphrase must preserve every entity name, every relationship, and every fact.
Only the wording and sentence structure may change.

Respond with this exact format and nothing else:
BLOCK_1_V1:
<first paraphrase of block 1>
BLOCK_1_V2:
<second paraphrase of block 1>
BLOCK_2_V1:
<first paraphrase of block 2>
BLOCK_2_V2:
<second paraphrase of block 2>
(continue for all blocks)

{blocks}"""


# ---------------------------------------------------------------------------
# Helpers: splitting and reconstructing turn 0
# ---------------------------------------------------------------------------

def split_turn0(content: str):
    """
    Split turn 0 into (paraphrase_target, question_suffix).

    Turn 0 is always:
      "Pretext: <...> Hypothesis: <...>. Question: <...>"
    Everything from "Question:" onward is kept verbatim.
    """
    q_idx = content.find("Question:")
    if q_idx == -1:
        return content, ""
    prefix = content[:q_idx].rstrip()
    suffix = content[q_idx:]          # "Question: ..."
    return prefix, suffix


def rebuild_turn0(paraphrased_prefix: str, question_suffix: str) -> str:
    return paraphrased_prefix.rstrip() + " " + question_suffix


# ---------------------------------------------------------------------------
# Helpers: building and parsing the batch prompt
# ---------------------------------------------------------------------------

def build_prompt(prefixes: list) -> str:
    """Construct the batched paraphrase prompt for N pretext blocks."""
    block_strs = [f"BLOCK_{i}:\n{text}" for i, text in enumerate(prefixes, 1)]
    return _USER_TEMPLATE.format(blocks="\n\n".join(block_strs))


def parse_response(response_text: str, n_blocks: int) -> dict:
    """
    Extract (v1, v2) for each block from model output.

    Returns {1: (v1, v2), 2: (v1, v2), ...}.
    Values are None when a block could not be parsed.
    """
    results = {}
    for i in range(1, n_blocks + 1):
        # Each version ends when the next BLOCK_*_V* tag starts or at EOF
        v1_pat = rf"BLOCK_{i}_V1:\s*(.*?)(?=BLOCK_\d+_V[12]:|$)"
        v2_pat = rf"BLOCK_{i}_V2:\s*(.*?)(?=BLOCK_\d+_V[12]:|$)"
        v1_m = re.search(v1_pat, response_text, re.DOTALL)
        v2_m = re.search(v2_pat, response_text, re.DOTALL)
        results[i] = (
            v1_m.group(1).strip() if v1_m else None,
            v2_m.group(1).strip() if v2_m else None,
        )
    return results


# ---------------------------------------------------------------------------
# Core: process one dataset
# ---------------------------------------------------------------------------

def process_dataset(model, dataset_path: str, output_dir: str,
                    batch_size: int, max_samples: Optional[int]) -> None:
    print(f"\n{'=' * 60}", flush=True)
    print(f"Dataset : {os.path.basename(dataset_path)}", flush=True)

    with open(dataset_path, encoding="utf-8") as fh:
        raw = json.load(fh)

    samples = raw.get("samples", [])
    if max_samples:
        samples = samples[:max_samples]

    n_samples = len(samples)
    n_batches = math.ceil(n_samples / batch_size)
    print(f"Samples : {n_samples}  |  batch_size={batch_size}  |  calls={n_batches}",
          flush=True)

    para1_samples = copy.deepcopy(samples)
    para2_samples = copy.deepcopy(samples)

    fail_count = 0
    t_start = time.time()

    for batch_idx in range(n_batches):
        lo = batch_idx * batch_size
        hi = min(lo + batch_size, n_samples)
        batch_range = range(lo, hi)

        # Collect turn-0 prefixes for this batch
        prefixes = []
        q_suffixes = []
        for s_idx in batch_range:
            turns = samples[s_idx].get("conversations", [{}])[0].get("turns", [])
            content = turns[0]["content"] if turns else ""
            prefix, q_suffix = split_turn0(content)
            prefixes.append(prefix)
            q_suffixes.append(q_suffix)

        # Single LLM call for the whole batch
        prompt = build_prompt(prefixes)
        try:
            response = model.respond(
                [{"role": "user", "content": prompt}]
            )
            parsed = parse_response(response.response_text, len(prefixes))
        except Exception as exc:
            print(f"  [WARN] batch {batch_idx + 1}/{n_batches} failed: {exc} — "
                  "keeping originals for this batch.", flush=True)
            fail_count += len(batch_range)
            continue

        # Apply paraphrases
        for local_i, (s_idx, q_suffix) in enumerate(zip(batch_range, q_suffixes), 1):
            v1, v2 = parsed.get(local_i, (None, None))
            if v1:
                turn0_p1 = rebuild_turn0(v1, q_suffix)
                para1_samples[s_idx]["conversations"][0]["turns"][0]["content"] = turn0_p1
            else:
                fail_count += 1
                print(f"  [WARN] sample {s_idx}: v1 parse failed — keeping original.",
                      flush=True)
            if v2:
                turn0_p2 = rebuild_turn0(v2, q_suffix)
                para2_samples[s_idx]["conversations"][0]["turns"][0]["content"] = turn0_p2
            else:
                fail_count += 1
                print(f"  [WARN] sample {s_idx}: v2 parse failed — keeping original.",
                      flush=True)

        elapsed = time.time() - t_start
        print(
            f"  batch {batch_idx + 1}/{n_batches}  "
            f"samples {lo}–{hi - 1}  "
            f"elapsed={elapsed / 60:.1f}min",
            flush=True,
        )

    elapsed = time.time() - t_start
    print(f"Done in {elapsed / 60:.1f} min  |  parse failures: {fail_count}", flush=True)

    # Write output files
    stem = Path(dataset_path).stem           # e.g. "default_reasoning_easy"
    os.makedirs(output_dir, exist_ok=True)
    for suffix, para_samples in [("para1", para1_samples), ("para2", para2_samples)]:
        out = dict(raw)
        out["metadata"] = dict(raw.get("metadata", {}))
        out["metadata"]["paraphrase_source"] = os.path.basename(dataset_path)
        out["metadata"]["paraphrase_version"] = suffix
        out["samples"] = para_samples
        out_path = os.path.join(output_dir, f"{stem}_{suffix}.json")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        print(f"Saved → {out_path}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        prog="python generate_paraphrases.py",
        description="Generate paraphrased copies of DeReLab datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # ── provider selection ────────────────────────────────────────────────────
    p.add_argument("--provider", choices=["openai", "hf"], default="openai",
                   help="LLM backend: 'openai' (default) or 'hf' (HuggingFace).")
    p.add_argument("--model_id", default=None,
                   help="Model ID. Defaults to 'gpt-5.1' for openai provider, "
                        "required for hf provider.")
    p.add_argument("--api_key", default=None,
                   help="OpenAI API key. Falls back to OPENAI_API_KEY env var.")

    # ── dataset / output ──────────────────────────────────────────────────────
    p.add_argument("--datasets_dir", default="datasets",
                   help="Directory containing source dataset JSONs. Default: datasets/")
    p.add_argument("--output_dir", default="datasets",
                   help="Directory for output para1/para2 files. Default: datasets/")
    p.add_argument("--datasets", nargs="+", default=None, metavar="FILE",
                   help="Specific dataset filenames to process. Default: all 6.")
    p.add_argument("--batch_size", type=int, default=5,
                   help="Samples per LLM call. Default: 5 (30 calls per dataset).")
    p.add_argument("--max_new_tokens", type=int, default=2048,
                   help="Max generation tokens per call. Default: 2048.")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="Sampling temperature for paraphrase diversity. Default: 0.7.")
    p.add_argument("--max_samples", type=int, default=None, metavar="N",
                   help="Limit to first N samples per file (smoke test).")

    # ── HuggingFace-only args ─────────────────────────────────────────────────
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"],
                   default="bfloat16", help="(hf only) torch dtype.")
    p.add_argument("--device_map", default="auto",
                   help="(hf only) device_map for model loading.")
    p.add_argument("--hf_token", default=None,
                   help="(hf only) HuggingFace access token. Falls back to HF_TOKEN env var.")
    p.add_argument("--quantization", choices=["none", "8bit", "4bit"], default="none",
                   help="(hf only) quantization mode.")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    dataset_files = args.datasets or ALL_DATASETS
    dataset_paths = []
    for fname in dataset_files:
        path = os.path.join(args.datasets_dir, fname)
        if not os.path.isfile(path):
            print(f"ERROR: dataset not found: {path}", file=sys.stderr)
            sys.exit(1)
        dataset_paths.append(path)

    if args.provider == "openai":
        model_id = args.model_id or "gpt-5.1"
        api_key  = args.api_key or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            print("ERROR: OPENAI_API_KEY not set. Pass --api_key or export the env var.",
                  file=sys.stderr)
            sys.exit(1)
        print(f"Provider    : OpenAI", flush=True)
        print(f"Model       : {model_id}", flush=True)
        model = _OpenAIParaphraseModel(
            model_id=model_id,
            api_key=api_key,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        print("OpenAI client ready.\n", flush=True)

    else:  # hf
        if not args.model_id:
            print("ERROR: --model_id is required for --provider hf.", file=sys.stderr)
            sys.exit(1)
        hf_token = (args.hf_token or os.environ.get("HF_TOKEN") or "").strip() or None
        import torch
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                     "float32": torch.float32}
        torch_dtype = dtype_map[args.dtype]
        print(f"Provider    : HuggingFace", flush=True)
        print(f"Loading model: {args.model_id}", flush=True)
        from evaluation.models.hf_model import HuggingFaceModel
        model = HuggingFaceModel(
            model_id=args.model_id,
            system_prompt=_SYSTEM_PROMPT,
            device_map=args.device_map,
            torch_dtype=torch_dtype,
            max_new_tokens=args.max_new_tokens,
            hf_token=hf_token,
            load_in_8bit=args.quantization == "8bit",
            load_in_4bit=args.quantization == "4bit",
            temperature=args.temperature,
        )
        print("Model loaded.\n", flush=True)

    for dataset_path in dataset_paths:
        process_dataset(
            model=model,
            dataset_path=dataset_path,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
        )

    print("\nAll datasets complete.", flush=True)


if __name__ == "__main__":
    main()
