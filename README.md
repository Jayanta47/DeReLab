# DeReLab

**Official repository for:**

> **DERELAB: Probing Defeasible Reasoning and Confirmation Bias in LLMs with a Generative Benchmark**
> Jayanta Sadhu, Sayem Shahad, Kenneth Marino — 2026
> [arXiv:2608.30413](https://arxiv.org/abs/2608.30413)

---

## Overview

DeReLab is a generative benchmark framework for probing *defeasible reasoning* and *confirmation bias* in large language models. Rather than relying on static datasets, DeReLab generates multi-turn conversations from formal reasoning graphs whose ground-truth answers are computed by a skeptical path-relation algorithm (PMPM). Every claim in the benchmark is verifiable against this formal ground truth, enabling precise measurement of when and how models fail to revise their beliefs under incongruent evidence.

The framework supports three reasoning topologies:

| Topology | Description |
|---|---|
| **Default** | Property-chain reasoning with defeasible updates, distractors, and source-priority conflicts |
| **Linear inheritance** | Taxonomic chains where properties propagate along a single lineage |
| **Tree inheritance** | Branching hierarchies where updates on one branch can, or cannot, affect sibling paths |

Each topology is evaluated at **easy** and **hard** difficulty tiers, yielding six reasoning conditions per model. The benchmark includes both an *entailment question* (yes / no / unknown) and an *effect-of-update question* (strengthening / weakening / no-effect) per turn, enabling measurement of metacognitive dissociation.

---

## Key Findings

- **Confirmation bias is near-universal**: 17 of 19 model–reasoning-type cells show statistically significant bias (Holm–Bonferroni corrected). Bias magnitude spans from BiasGap = 0.08 (Gemma-4-12B) to 0.94 (GPT-5.1).
- **Know-but-don't-output dissociation**: Several models correctly label an update as *weakening* on the majority of incongruent turns yet almost never revise their entailment answer (e.g., GPT-5-mini: 95.4% effect accuracy, 20.3% entailment accuracy on C2 turns).
- **Effect labelling is scaffolded**: Reversing the question order (effect question first) causes C2-effect accuracy to collapse for all tested models, revealing that correct effect labelling in the standard format was scaffolded by prior engagement with the entailment question.
- **Extended thinking suppresses bias**: Qwen3-32B with extended thinking reduces default BiasGap from 0.67 to 0.008, while inheritance bias remains partial (Gap = 0.065, n.s.).
- **Confidence decays with inference depth**: On-path correct-answer confidence declines significantly with distance for most models under hard tree-inheritance conditions (Gemma-4-31B ρ = −0.298, Llama-3.1-8B ρ = −0.255, Llama-3.1-70B ρ = −0.207).
- **Benchmark scores are stable**: ICC = 0.949 across 20 independently seeded draws; CV < 4% for five of eight models on both default and linear-inheritance conditions.

---

## Models Evaluated

| Model | Notes |
|---|---|
| GPT-5.1 | |
| GPT-5-mini | |
| Qwen3-32B | Standard (non-thinking) |
| Qwen3-32B* | Extended-thinking variant |
| Gemma-4-31B | |
| Gemma-4-12B | |
| Gemma-4-MoE (26B-A4B) | |
| Gemma-4-E4B | |
| Llama-3.1-70B | |
| Llama-3.1-8B | |

---

## Repository Structure

```
DeReLab/
├── generate_default_reasoning.py      # Dataset generation — default topology
├── generate_linear_inheritance.py     # Dataset generation — linear inheritance
├── generate_tree_inheritance.py       # Dataset generation — tree inheritance
├── generate_paraphrases.py            # Surface-form paraphrase generation
├── generate_reseeding_datasets.py     # Seeded draw generation for robustness analysis
│
├── knowledge_graph/                   # Knowledge base and entity population
│   ├── KBbuilder.py
│   └── data_sources/
│
├── path_relation_module/              # PMPM skeptical reasoning algorithm
│   ├── pmpm_algo.py                   # Core truth engine
│   ├── graph_resolutor.py             # DeReLab graph → PMPM adapter
│   └── effect_classifier.py
│
├── question_generation/               # NL generation from graph edges
│   ├── nl_generator.py
│   ├── default_nl_generator.py
│   └── default_question_generator.py
│
├── evaluation/
│   ├── main.py                        # Run inference for a model
│   ├── evaluator.py                   # Turn-level accuracy scorer
│   ├── models/                        # Model adapters (OpenAI, Gemini, HF, etc.)
│   ├── analyze_performance.py         # Main heatmap and depth-line figures
│   ├── analyze_reseeding.py           # ICC / CV / Kruskal–Wallis robustness stats
│   ├── cogsci/                        # Confirmation bias analysis
│   │   └── main.py                    # BiasGap, OR, metacognitive, anchoring figures
│   ├── path_confidence/               # Spearman ρ vs. inference depth
│   ├── real_entity/                   # Real-entity generalisation experiment
│   └── reverse/                       # Reversed question-order experiment
│
├── interpretability/                  # Probing experiments (forward passes)
├── reseeding_scripts/                 # Shell scripts for seeded evaluation runs
├── test_data/                         # Sample generated datasets (one per topology)
└── utils/
    └── visualization/                 # Heatmap and graph visualisation utilities
```

---

## Installation

Requires Python 3.10+.

```bash
git clone https://github.com/your-org/DeReLab.git
cd DeReLab
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

API keys (set in `.env` or environment):

```
OPENAI_API_KEY=...
GOOGLE_API_KEY=...       # for Gemini / Gemma-4 via Google GenAI
HF_TOKEN=...             # for Llama / Qwen via Hugging Face
```

---

## Generating the Benchmark

Sample datasets for all three topologies are included in `test_data/`. To generate fresh datasets:

```bash
# Default reasoning
python generate_default_reasoning.py

# Linear inheritance
python generate_linear_inheritance.py

# Tree inheritance
python generate_tree_inheritance.py
```

Each script writes a JSON file of multi-turn conversations with PMPM-verified ground-truth labels.

---

## Running Evaluation

```bash
python -m evaluation.main \
    --model      gpt-5.1 \
    --dataset    datasets/default_hard.json \
    --output_dir evaluation/results
```

See `python -m evaluation.main --help` for all options. Model adapters live in `evaluation/models/`; add a new file there to support a new provider.

---

## Reproducing Paper Results

All scripts run from the repo root (`python -m <module>`). Every script accepts `--help`. Detailed per-script output listings are in [EXPERIMENTS.md](EXPERIMENTS.md).

### 1 — Main benchmark performance

Produces the accuracy heatmap (model × topology × difficulty), depth-accuracy curves, and the appendix accuracy table.

```bash
python -m evaluation.analyze_performance \
    --results_dir evaluation/results \
    --output_dir  evaluation/analysis
```

To reproduce the real-entity generalisation results (Table in §Appendix), run on each subset separately:

```bash
python -m evaluation.analyze_performance \
    --results_dir evaluation/real_entity/real_entity_test/non_sensical \
    --output_dir  evaluation/analysis/real_entity_non_sensical

python -m evaluation.analyze_performance \
    --results_dir evaluation/real_entity/real_entity_test/sensical \
    --output_dir  evaluation/analysis/real_entity_sensical
```

**Key outputs:** `evaluation/analysis/heatmap.pdf`, `depth_lines.pdf`, `appendix_table.pdf/.tex`

---

### 2 — Reseeding robustness (ICC, CV, Kruskal–Wallis)

Runs stability analysis across 20 independently seeded draws. Produces ICC(2,1), CV table, and strip plots.

```bash
python -m evaluation.analyze_reseeding \
    --results_dir evaluation/reseeding \
    --output_dir  evaluation/reseeding_analysis
```

To analyse a specific task type rather than overall accuracy:

```bash
python -m evaluation.analyze_reseeding \
    --results_dir evaluation/reseeding \
    --output_dir  evaluation/reseeding_analysis \
    --task_type   update_answer
# choices: initial_answer | update_answer | effect_of_update
```

**Key outputs:** `kruskal_wallis.csv`, `icc_summary.txt`, `reseeding_stats.csv`, figures in `reseeding_analysis/figures/`

---

### 3 — Dataset statistics

Reports sample counts, task-type distributions, ground-truth balance, chain lengths, and graph structure for any dataset directory.

```bash
# Main benchmark datasets
python -m evaluation.dataset_stats \
    --datasets_dir datasets \
    --output       evaluation/analysis/dataset_stats.txt

# Real-entity subsets
python -m evaluation.dataset_stats \
    --datasets_dir datasets/real_entity_test/non_sensical \
    --output       evaluation/analysis/real_entity_nonsensical_stats.txt

python -m evaluation.dataset_stats \
    --datasets_dir datasets/real_entity_test/sensical \
    --output       evaluation/analysis/real_entity_sensical_stats.txt

# Single reseeding seed (sanity check)
python -m evaluation.dataset_stats \
    --datasets_dir datasets/reseeding/seed_0
```

---

### 4 — Confirmation bias analysis

Labels every `update_answer` turn with a congruence code (C1–C5), computes BiasGap, odds ratios, anchoring rates, and metacognitive dissociation metrics, and produces all paper figures.

```bash
# Standard question order (main paper results)
python -m evaluation.cogsci.main \
    --results_dir evaluation/results \
    --output_dir  evaluation/cogsci/output

# Reversed question order (effect question first — §Reversed order experiment)
python -m evaluation.cogsci.main \
    --results_dir evaluation/reverse \
    --output_dir  evaluation/cogsci/output_reversed
```

**Key outputs in `evaluation/cogsci/output/`:**
- `bias_metrics_per_model.csv` — BiasGap, OR with 95% CI, McNemar p, anchoring rate, metacognitive accuracy
- `congruence_labels.jsonl` — one labelled row per `update_answer` turn (input to §4a below)
- `figures/bias_gap_bars.pdf`, `or_forest.pdf`, `anchoring_bars.pdf`, `semantic_override_bars.pdf`, `metacognitive_bars.pdf`, `condition_distribution.pdf`

#### 4a — C1×C2 confusion matrices

Requires `congruence_labels.jsonl` from step 4 above.

```bash
python -m evaluation.cogsci.c1_c2_confusion_matrix \
    --labels_file evaluation/cogsci/output/congruence_labels.jsonl \
    --metrics_csv  evaluation/cogsci/output/bias_metrics_per_model.csv \
    --output_dir   evaluation/cogsci/output
```

**Outputs:** `c1_c2_confusion_matrix.pdf` (all-model grid), `c1_c2_confusion_matrix_gpt51.pdf` (detail panel with OR verification)

---

### 5 — Bias annotation (per-turn augmentation)

Writes annotated copies of result files with a `bias_annotation` field on every turn. Useful for downstream analysis without re-running the full pipeline.

```bash
# Annotate all result files
python -m evaluation.cogsci.augment_bias \
    --results_dir evaluation/results \
    --output_dir  evaluation/cogsci/augmented

# Annotate a single file
python -m evaluation.cogsci.augment_bias \
    --results_dir evaluation/results \
    --output_dir  evaluation/cogsci/augmented \
    --file        evaluation/results/gpt-5.1/default_reasoning_easy__20260501.json
```

Original files are not modified. Output mirrors the input directory tree under `output_dir`.

---

### 6 — Path confidence analysis

Correlates model confidence with update-to-hypothesis distance in tree-inheritance graphs. Also measures off-path intrusion and inference depth effects.

```bash
# Main path confidence (Spearman ρ between distance and confidence)
python -m evaluation.path_confidence.path_confidence \
    --results_dir evaluation/results \
    --output_dir  evaluation/path_confidence/output \
    --figures_dir evaluation/path_confidence/figures

# Off-path intrusion analysis (intrusion_rate per model)
python -m evaluation.path_confidence.off_path_analysis \
    --results_dir evaluation/results \
    --output_dir  evaluation/path_confidence/output

# Inference depth analysis (on-path only)
python -m evaluation.path_confidence.inference_depth_analysis \
    --results_dir evaluation/results \
    --output_dir  evaluation/path_confidence/output
```

**Key outputs:** `path_confidence_turns.csv`, `path_confidence_stats.csv`, `offpath_stats.csv`, `inference_depth_stats.csv`, and plots in `path_confidence/figures/`

---

## PMPM Algorithm

The core truth engine is a skeptical path-relation algorithm (PMPM) implemented in `path_relation_module/pmpm_algo.py`. It computes whether a hypothesis is entailed, defeated, or unknown given a signed defeasible graph. Runnable self-tests covering classic exception handling, symmetric conflict, and pure positive inheritance are included:

```bash
python path_relation_module/pmpm_algo.py
```

---

## Citation

If you use DeReLab in your research, please cite:

```bibtex
@misc{sadhu2026derelabprobingdefeasiblereasoning,
      title={DERELAB: Probing Defeasible Reasoning and Confirmation Bias in LLMs with a Generative Benchmark},
      author={Jayanta Sadhu and Sayem Shahad and Kenneth Marino},
      year={2026},
      eprint={2608.30413},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2608.30413},
}
```

---

## License

See [LICENSE](LICENSE) for details.
