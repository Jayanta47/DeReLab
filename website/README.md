# DeReLab website (self-contained module)

This folder is the entire public website for the DeReLab paper: a landing
page plus a live, in-browser dataset simulator. It has **no dependency on
anything else in this repository** — no build step, no server, no shared
imports outside this folder. You can copy `website/` verbatim into a new,
empty repository and it will work exactly as it does here.

It currently lives inside the research repo (`DeReLab/website/`) only
because that's where development happened. See **Transferring to the
public website repo** below for how to move it out.

## What's here

```
website/
├── index.html          Landing page: findings, topologies, models, citation
├── simulator.html       The interactive dataset simulator
├── css/
│   └── style.css        Shared stylesheet (white / beige / forest-green theme)
├── js/
│   ├── data.js           Embedded KB data (domain attributes, nonce-entity word banks)
│   ├── pmpm.js            Port of the PMPM skeptical reasoning algorithm
│   ├── generator.js       Port of tree-inheritance structure/population/NL/conversation generation
│   ├── graph-render.js     SVG renderer for the simulator's graph view
│   └── simulator.js       Page controller: wires the stage stepper to generator + renderer
└── assets/
    └── figures/          Result charts embedded in the landing page's findings section
```

Everything is plain HTML/CSS/JS with no bundler and no `node_modules`.
Scripts are loaded as ordinary `<script>` tags (not ES modules) specifically
so the pages also work when opened directly from disk (`file://…`), not
just when served over HTTP.

## How the simulator works

The simulator is **not a canned demo** — every click generates a fresh
random example using the real generation algorithm, ported from the Python
pipeline into JavaScript, for all three topologies (selectable in the
toolbar):

| Simulator stage | Tree / linear inheritance, ported from | Default reasoning, ported from |
|---|---|---|
| 1. Structure | `graph_generation/graph_generator.py` → `TreeInheritanceGenerator` | `graph_generation/graph_generator.py` → `DefaultChainGenerator` |
| 2. Populate | `graph_populator/graph_populator.py` → `_populate_inheritance_tree` | `graph_populator/graph_populator.py` → `_populate_default_chain` |
| 3. Verbalize | `question_generation/nl_generator.py` → `generate_edge_sentences` | same |
| 4. Reason | `question_generation/nl_generator.py` → `_build_tree_inheritance_conversations`, backed by `path_relation_module/pmpm_algo.py` | simplified port of `_build_default_conversations` (see below), backed by the same PMPM module |

**Linear inheritance is not a separate port.** Structurally it's a
tree-inheritance graph forced into a single unbranching chain
(`root_breadth=1`, `min_branching=max_branching=1`, `survival_prob=1.0` —
see `LINEAR_PRESET` in `js/data.js`), so it reuses 100% of the tree code
path; only the topology label differs.

The PMPM algorithm port (`js/pmpm.js`) was validated against all five
canonical test cases in `path_relation_module/pmpm_algo.py`'s `__main__`
block, and the tree-inheritance pipeline port was cross-checked against a
live run of the actual Python pipeline
(`generate_tree_inheritance.py`'s `_TreeInheritanceSamplingGenerator`) to
confirm identical sentence templates, pretext ordering, and effect
classification. All three topologies were stress-tested for 200 random
generations each with zero errors (turn/answer sanity, unique edge ids).

Parameters are fixed to each topology's **easy-tier** configuration (see
`EASY_PRESETS`, `LINEAR_PRESET`, `DEFAULT_EASY_PRESET` in `js/data.js`).

### Deliberate simplifications from the real pipeline

1. **Default reasoning is simplified to one hypothesis subject per graph.**
   The real pipeline (`DefaultReasoningQuestionGenerator` +
   `DefaultReasoningNLGenerator` + `Type4ConversationPlanner`, none of
   which are ported) supports multiple hypothesis subjects per graph,
   Type-3 disjunctive "at least one of X or Y is not..." exception groups,
   and a full Type-4 source-priority sub-system (named sources,
   contradiction/bypass scenarios, reversed speaker order). The simulator
   instead generates one hypothesis subject with its property chain, a
   staged reveal of defeasible evidence restricted to that subject, and
   one irrelevant distractor attribute interleaved into the reveal
   sequence — enough to demonstrate property-chain reasoning and
   distractor-suppression without the added UI complexity of
   multi-section, multi-source conversations.
2. **Nonce-entity word-bank fallback.** In the real pipeline, a root
   concept without its own entry in `non_sensical_entities.json` (i.e.
   anything other than `animal` or `plant` among the seven root concepts)
   falls back to a **uniformly random** word bank — including totally
   unrelated domains like `Tool` or `Food`. That reads as a bug rather
   than a feature in a public demo, so the simulator instead reuses the
   `Animal` word bank for `mammal`, `vertebrate`, `insect`,
   `microorganism`, and `bird` roots. This only affects the *cosmetic*
   choice of invented label words; the graph structure, population
   mechanism, sentence templates, and PMPM reasoning are otherwise
   faithful.
3. **Trimmed word banks.** `js/data.js` embeds a 180-word sample (fixed
   seed) from each 1000-word nonce-entity pool, and only the seven domains
   reachable from the root-concept distribution in
   `domain_attributes_values.json` — enough variety for a demo, without
   inlining the full multi-thousand-word source files.
4. **`use_conceptnet` is always off**, matching the real easy-tier config
   (`use_conceptnet=False` in `BatchGenerationConfig`), so the simulator
   never needs network access.
5. **Linear inheritance's defeasible-edge selection** uses tree
   inheritance's per-node Bernoulli sampling (`sparsity_factor`) rather
   than `LinearInheritanceGenerator.add_defeasible_edges`'s distinct
   "sample N unique depths without replacement" algorithm. Same
   mechanism (a random subset of intermediate nodes gets evidence),
   slightly different edge-count distribution.

If you use this simulator for anything beyond illustration (e.g. citing
specific generated examples), regenerate the same example with the actual
Python pipeline to confirm it — the JS port is for demonstration, the
Python pipeline in the main repo is the source of truth for the released
dataset.

## Result charts on the landing page

The three findings on the landing page that show a chart
(`assets/figures/bias_gap_bars_grouped.png`, `metacognitive_bars.png`,
`model_accuracy_default.png`) are rasterized directly from the **same PDFs
compiled into the paper**, at `DeReLab_latex/latex/images/` in the parent
repo — not regenerated from raw `evaluation/*/figures/` output, which can
drift from what's actually in the paper (different run, different figure
never used in the final draft, etc.). If the paper's figures change, re-run
the conversion from the updated PDFs, e.g.:

```python
import fitz  # pip install pymupdf
doc = fitz.open("DeReLab_latex/latex/images/bias_gap_bars_grouped.pdf")
pix = doc[0].get_pixmap(matrix=fitz.Matrix(200/72, 200/72))  # ~200 DPI
pix.save("website/assets/figures/bias_gap_bars_grouped.png")
```

Not every finding has a chart — some are clearer as plain text, and the
question-order reversal finding uses a small hand-built CSS bar comparison
(`.mini-chart` in `css/style.css`) instead of an image, since no dedicated
figure for that experiment exists in the paper.

## Running locally

No build step. Either:

```bash
open index.html          # macOS, opens directly from disk
```

or serve it (recommended, avoids any `file://` quirks in some browsers):

```bash
python3 -m http.server 8000
# then open http://localhost:8000/
```

## Before publishing

All GitHub links already point to the paper's public repo:
`https://github.com/Jayanta47/DeReLab`. If that URL ever changes, update it
across all files with:

```bash
grep -rln "github.com/Jayanta47/DeReLab" . | xargs sed -i '' 's|github.com/Jayanta47/DeReLab|github.com/NEW-PATH|g'
```

## Deploying (current setup: this repo, via GitHub Actions)

The site deploys straight from this repo's `website/` subfolder using
`.github/workflows/deploy-website.yml`. GitHub Pages' built-in "deploy from
branch" mode can only serve a whole branch root or `/docs`, not an
arbitrary subfolder, so a subfolder deploy needs a GitHub Actions workflow
that packages `website/` as the Pages artifact.

One-time setup, done in the GitHub UI:

1. Repo **Settings → Pages → Build and deployment → Source: GitHub
   Actions** (not "Deploy from a branch").
2. Push to `main` with changes under `website/` (or run the workflow
   manually from the **Actions** tab — it has `workflow_dispatch` enabled).
3. Watch it run under the **Actions** tab; once green, the live URL
   appears both in **Settings → Pages** and in the workflow run's summary.
4. GitHub Pages requires the repo to be **public**, unless the account is
   on GitHub Pro/Team/Enterprise. If this repo is private, either make it
   public or move the code to a public repo before Pages will serve
   anything.

The resulting URL is `https://<owner>.github.io/<repo-name>/` — the
contents of `website/` become the *root* of that URL, not a `/website/`
subpath, since the workflow uploads that folder's contents directly as the
Pages artifact.

## Alternative: separate dedicated repo

If the site is ever split out of this repo, a dedicated repo needs no
GitHub Actions at all:

1. Create the new (empty) public repository, e.g. `derelab-website`.
2. Copy this folder's contents to the repo root (not the `website/`
   folder itself — its contents):
   ```bash
   cp -R /path/to/DeReLab/website/. /path/to/derelab-website/
   ```
3. Commit and push.
4. Enable GitHub Pages: repo **Settings → Pages → Source: Deploy from a
   branch → `main` / root**. No workflow file is needed since there's no
   build step and no subfolder to select.
5. The site will be live at `https://<org-or-user>.github.io/derelab-website/`
   (or a custom domain, if configured under **Settings → Pages → Custom
   domain**).

## Design notes

- **Palette**: warm white background, beige surfaces, forest-green accent
  (site chrome, CTAs) — per the requested theme.
- **Reasoning-state colors are intentionally *not* part of the green/beige
  palette.** The simulator's yes/no and strengthen/weaken indicators use
  the colorblind-safe Okabe-Ito blue (`#0072B2`) / amber (`#E69F00`) pair
  instead of red/green, consistent with the color choices already made for
  the paper's own figures.
- Typefaces: **Fraunces** (display/headings) + **Inter** (UI/body) +
  **JetBrains Mono** (code/citation), loaded from Google Fonts.
- A `prefers-color-scheme: dark` variant is included; `data-theme="dark"`
  / `data-theme="light"` attributes on `<html>` can force a specific theme
  if you later add a toggle.
