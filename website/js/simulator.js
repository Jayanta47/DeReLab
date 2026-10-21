/**
 * simulator.js
 * ---------------------------------------------------------------------
 * UI controller for simulator.html. Wires the ported generator/PMPM
 * modules and the SVG renderer to the stage-stepper interface. Supports
 * all three topologies: default, linear inheritance, tree inheritance.
 * ---------------------------------------------------------------------
 */
(function () {
  "use strict";

  const GEN = window.DeReLabGenerator;
  const RENDER = window.DeReLabRender;
  const D = window.DeReLabData;

  const STAGES = ["structure", "populate", "verbalize", "reason"];

  // Per-topology tunable parameters. Only the knobs that meaningfully
  // change the demo are exposed -- e.g. minBranching/branchingDecay for
  // tree inheritance are left fixed.
  const PARAM_SCHEMAS = {
    tree_inheritance: [
      { key: "maxDepth", label: "Max depth", min: 2, max: 6, step: 1, default: 4, desc: "How many levels the tree grows." },
      { key: "rootBreadth", label: "Root breadth", min: 1, max: 6, step: 1, default: 3, desc: "Children at the root." },
      { key: "survivalProb", label: "Branch survival", min: 0.2, max: 1, step: 0.1, default: 0.6, desc: "Chance a branch keeps growing at each level." },
      { key: "sparsity", label: "Evidence density", min: 0.2, max: 0.9, step: 0.05, default: 0.65, desc: "Chance each node gets a defeasible update." },
    ],
    linear_inheritance: [
      { key: "maxDepth", label: "Chain length", min: 2, max: 6, step: 1, default: 4, desc: "How many classes in the chain." },
      { key: "sparsity", label: "Evidence density", min: 0.2, max: 0.9, step: 0.05, default: 0.65, desc: "Chance each node gets a defeasible update." },
    ],
    default: [
      { key: "chainLength", label: "Chain length", min: 3, max: 7, step: 1, default: 5, desc: "How many properties in the chain." },
      { key: "branchProbability", label: "Branch chance", min: 0, max: 0.5, step: 0.05, default: 0, desc: "Chance a property node branches instead of continuing the chain." },
      { key: "numObjects", label: "Objects", min: 1, max: 3, step: 1, default: 1, desc: "Total objects; only one is asked about, the rest are decoys." },
      { key: "sparsityFactor", label: "Evidence density", min: 0.2, max: 0.9, step: 0.05, default: 0.65, desc: "Chance each property gets defeasible evidence." },
      { key: "numIrrelevantAttributes", label: "Distractor attributes", min: 0, max: 2, step: 1, default: 1, desc: "Irrelevant facts mixed into the reveal." },
    ],
  };

  function defaultParamValues(topology) {
    const values = {};
    PARAM_SCHEMAS[topology].forEach((p) => { values[p.key] = p.default; });
    return values;
  }

  const state = {
    topology: "tree_inheritance", // "default" | "linear_inheritance" | "tree_inheritance"
    graph: null,
    conversation: null,
    stageIndex: 0,
    revealCount: 0, // number of update turns revealed in "reason" stage
    paramsOpen: false,
    paramValues: {
      tree_inheritance: defaultParamValues("tree_inheritance"),
      linear_inheritance: defaultParamValues("linear_inheritance"),
      default: defaultParamValues("default"),
    },
  };

  const els = {};

  function $(id) {
    return document.getElementById(id);
  }

  function cacheEls() {
    els.topologySelect = $("topology-select");
    els.rootConceptSelect = $("root-concept-select");
    els.generateBtn = $("generate-btn");
    els.graphContainer = $("graph-container");
    els.stagePanel = $("stage-panel");
    els.stageSteps = document.querySelectorAll(".stage-step");
    els.prevBtn = $("prev-stage-btn");
    els.nextBtn = $("next-stage-btn");
    els.answerBadge = $("answer-badge");
    els.rawToggleBtn = $("raw-toggle-btn");
    els.rawPanel = $("raw-panel");
    els.simLayout = $("sim-layout");
    els.paramsToggleBtn = $("params-toggle-btn");
    els.paramsPanel = $("params-panel");
    els.paramsFields = $("params-fields");
  }

  function answerBadgeHTML(answer) {
    const cls = answer === "yes" ? "badge-yes" : answer === "no" ? "badge-no" : "badge-unknown";
    const text = answer === "yes" ? "Yes" : answer === "no" ? "No" : "Unknown";
    return `<span class="answer-badge ${cls}">${text}</span>`;
  }

  function effectBadgeHTML(effect) {
    const cls = effect === "strengthen" ? "badge-yes" : effect === "weaken" ? "badge-no" : "badge-unknown";
    return `<span class="answer-badge ${cls} small">${effect}</span>`;
  }

  function esc(s) {
    const d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
  }

  function isDefault() {
    return state.topology === "default";
  }

  // ---------------------------------------------------------------------
  // Generation
  // ---------------------------------------------------------------------

  function generateNew() {
    const rootSel = els.rootConceptSelect.value;
    const rootOpt = rootSel === "random" ? {} : { rootConcept: rootSel };
    const custom = state.paramsOpen ? state.paramValues[state.topology] : null;

    if (state.topology === "default") {
      const preset = custom ? Object.assign({}, D.DEFAULT_EASY_PRESET, custom) : undefined;
      state.graph = GEN.generateDefaultStructure(Object.assign({}, rootOpt, preset ? { preset } : {}));
    } else if (state.topology === "linear_inheritance") {
      const preset = custom
        ? Object.assign({}, D.LINEAR_PRESET, { maxDepth: custom.maxDepth })
        : D.LINEAR_PRESET;
      const opts = Object.assign({}, rootOpt, { preset, topology: "linear_inheritance" });
      if (custom) opts.sparsity = custom.sparsity;
      state.graph = GEN.generateStructure(opts);
    } else {
      const opts = Object.assign({}, rootOpt, { topology: "tree_inheritance" });
      if (custom) {
        opts.preset = {
          name: "custom", maxDepth: custom.maxDepth, rootBreadth: custom.rootBreadth,
          minBranching: 1, maxBranching: 2, survivalProb: custom.survivalProb, branchingDecay: 1.0,
        };
        opts.sparsity = custom.sparsity;
      }
      state.graph = GEN.generateStructure(opts);
    }
    state.conversation = null;
    state.stageIndex = 0;
    state.revealCount = 0;
    renderAll();
  }

  // ---------------------------------------------------------------------
  // Stage panel content (kept short: one line of context + a compact
  // stat list, not paragraphs of explanation)
  // ---------------------------------------------------------------------

  function countByType(edges, type) {
    return edges.filter((e) => e.type === type).length;
  }

  function renderStructurePanel(g) {
    if (isDefault()) {
      const numProps = g.propertyNodeIds.length;
      const numDefeasible = countByType(g.edges, "defeasible_has_property") + countByType(g.edges, "defeasible_lacks_property");
      els.stagePanel.innerHTML = `
        <h3>1. Structure</h3>
        <p>A branching property chain grows from a root property; objects attach to it.</p>
        <dl class="stat-list">
          <dt>Root concept</dt><dd>${esc(g.rootConcept)}</dd>
          <dt>Property nodes</dt><dd>${numProps}</dd>
          <dt>Objects</dt><dd>${g.objectIds.length}</dd>
          <dt>Defeasible edges (subject only)</dt><dd>${numDefeasible}</dd>
        </dl>
        <p class="hint">Labels are hidden until the next stage.</p>
      `;
    } else {
      const numClass = [...g.nodes.values()].filter((n) => n.type === "class").length;
      const numDefeasible = countByType(g.edges, "defeasible_has_property") + countByType(g.edges, "defeasible_lacks_property");
      els.stagePanel.innerHTML = `
        <h3>1. Structure</h3>
        <p>A taxonomy ${g.topology === "linear_inheritance" ? "chain" : "tree"} grows from a root class.</p>
        <dl class="stat-list">
          <dt>Root concept</dt><dd>${esc(g.rootConcept)}</dd>
          <dt>Class nodes</dt><dd>${numClass}</dd>
          <dt>Leaf nodes</dt><dd>${g.leafNodes.length}</dd>
          <dt>Defeasible edges</dt><dd>${numDefeasible}</dd>
        </dl>
        <p class="hint">Labels are hidden until the next stage.</p>
      `;
    }
  }

  function renderPopulatePanel(g) {
    if (isDefault()) {
      const rootMeta = g.nodes.get(g.rootPropertyNode).metadata;
      const hypMeta = g.nodes.get(g.hypothesisTarget).metadata;
      els.stagePanel.innerHTML = `
        <h3>2. Populate</h3>
        <p>Objects get invented words; each property node gets a real attribute value.</p>
        <dl class="stat-list">
          <dt>Root property</dt><dd>${esc(rootMeta.selected_value)}</dd>
          <dt>Hypothesis subject</dt><dd><strong>${esc(g.nodes.get(g.hypothesisSubject).label)}</strong></dd>
          <dt>Hypothesis property</dt><dd>${esc(hypMeta.selected_value)}</dd>
        </dl>
      `;
    } else {
      const sample = [...g.nodes.values()].filter((n) => n.type === "class" && n.id !== g.rootName).slice(0, 4);
      els.stagePanel.innerHTML = `
        <h3>2. Populate</h3>
        <p>Each node gets a label: the root keeps its concept name, the rest are invented words.</p>
        <dl class="stat-list">
          <dt>Target property</dt><dd>${esc(g.nodes.get(g.propertyNode).label)}</dd>
          <dt>Sample labels</dt><dd>${sample.map((n) => esc(n.label)).join(", ")}</dd>
          <dt>Hypothesis subject</dt><dd><strong>${esc(g.nodes.get(g.hypothesisLeaf).label)}</strong></dd>
        </dl>
      `;
    }
  }

  function backboneSentences(g) {
    if (isDefault()) {
      return g.edges
        .filter((e) => e.type === "is_a" || e.type === "implies")
        .slice()
        .sort((a, b) => (g.nodes.get(a.u).layer || 0) - (g.nodes.get(b.u).layer || 0))
        .map((e) => e.nl_sentence);
    }
    return g.edges
      .filter((e) => e.type === "inheritance" || e.type === "has_property")
      .slice()
      .sort((a, b) => (g.nodes.get(a.u).layer || 0) - (g.nodes.get(b.u).layer || 0))
      .map((e) => e.nl_sentence);
  }

  function renderVerbalizePanel(g) {
    const pretext = backboneSentences(g).join(" ");
    const hypEdge = g.edges.find((e) => e.type === "hypothesis_edge");
    els.stagePanel.innerHTML = `
      <h3>3. Verbalize</h3>
      <p>Every edge becomes a sentence. This is the pretext the model sees first:</p>
      <blockquote class="pretext-block">${esc(pretext)}</blockquote>
      <p><strong>Question:</strong> &ldquo;${esc(hypEdge.nl_sentence)}&rdquo;</p>
    `;
  }

  function renderReasonPanel(g, convo) {
    const total = convo.turns.filter((t) => t.task_type === "update_answer").length;
    els.stagePanel.innerHTML = `
      <h3>4. Reason</h3>
      <p>Each update is revealed one at a time; the answer is recomputed after every reveal.</p>
      <div class="transcript" id="transcript"></div>
      <div class="reveal-controls">
        <button id="reveal-next-btn" class="btn btn-primary" ${state.revealCount >= total ? "disabled" : ""}>
          Reveal next (${state.revealCount}/${total})
        </button>
        <button id="reveal-reset-btn" class="btn btn-ghost">Restart</button>
      </div>
    `;

    const transcript = $("transcript");
    const backboneTurn = convo.turns[0];
    const initialTurn = convo.turns[1];
    transcript.appendChild(
      turnBlock("Pretext + question", backboneTurn.content, `Initial answer: ${answerBadgeHTML(initialTurn.ground_truth)}`)
    );

    let updateIdx = 0;
    for (let i = 2; i < convo.turns.length; i += 2) {
      if (updateIdx >= state.revealCount) break;
      const userTurn = convo.turns[i];
      const assistantTurn = convo.turns[i + 1];
      transcript.appendChild(
        turnBlock(
          `Update ${updateIdx + 1}`,
          userTurn.content,
          `${answerBadgeHTML(assistantTurn.ground_truth)} &nbsp; ${effectBadgeHTML(assistantTurn.effect)}`
        )
      );
      updateIdx++;
    }
    transcript.scrollTop = transcript.scrollHeight;

    $("reveal-next-btn").addEventListener("click", () => {
      state.revealCount = Math.min(state.revealCount + 1, total);
      renderAll();
    });
    $("reveal-reset-btn").addEventListener("click", () => {
      state.revealCount = 0;
      renderAll();
    });

    const currentAnswer =
      state.revealCount === 0 ? initialTurn.ground_truth : convo.turns[2 + (state.revealCount - 1) * 2 + 1].ground_truth;
    els.answerBadge.innerHTML = `Current answer: ${answerBadgeHTML(currentAnswer)}`;
    els.answerBadge.hidden = false;
  }

  function turnBlock(label, content, footerHTML) {
    const wrap = document.createElement("div");
    wrap.className = "turn-block";
    wrap.innerHTML = `
      <div class="turn-label">${esc(label)}</div>
      <div class="turn-content">${esc(content).replace(/\n/g, "<br>")}</div>
      <div class="turn-footer">${footerHTML}</div>
    `;
    return wrap;
  }

  // ---------------------------------------------------------------------
  // Master render
  // ---------------------------------------------------------------------

  function updateTurnsOf(convo) {
    return convo.turns.filter((t) => t.role === "user" && t.edgeId);
  }

  function renderAll() {
    const g = state.graph;
    const stage = STAGES[state.stageIndex];

    els.stageSteps.forEach((stepEl, i) => {
      stepEl.classList.toggle("active", i === state.stageIndex);
      stepEl.classList.toggle("done", i < state.stageIndex);
    });
    els.prevBtn.disabled = state.stageIndex === 0;
    els.nextBtn.textContent = state.stageIndex === STAGES.length - 1 ? "Finished" : "Next stage →";
    els.nextBtn.disabled = state.stageIndex === STAGES.length - 1;

    if (stage !== "reason") els.answerBadge.hidden = true;

    const revealedEdgeIds = new Set();
    let activeEdgeId = null;
    if (stage === "reason" && state.conversation) {
      const updateTurns = updateTurnsOf(state.conversation);
      for (let i = 0; i < state.revealCount && i < updateTurns.length; i++) {
        revealedEdgeIds.add(updateTurns[i].edgeId);
      }
      if (state.revealCount > 0 && updateTurns[state.revealCount - 1]) {
        activeEdgeId = updateTurns[state.revealCount - 1].edgeId;
      }
    }

    RENDER.render(els.graphContainer, g, { stage, revealedEdgeIds, activeEdgeId });

    if (stage === "structure") renderStructurePanel(g);
    else if (stage === "populate") renderPopulatePanel(g);
    else if (stage === "verbalize") renderVerbalizePanel(g);
    else if (stage === "reason") {
      if (!state.conversation) {
        state.conversation = isDefault() ? GEN.buildDefaultConversation(g) : GEN.buildConversation(g);
      }
      renderReasonPanel(g, state.conversation);
    }

    renderRawPanel(g);
  }

  function renderRawPanel(g) {
    if (!els.rawPanel || els.rawPanel.hidden) return;
    const summary = {
      topology: g.topology,
      root_concept: g.rootConcept,
      nodes: [...g.nodes.values()].map((n) => ({ id: n.id, type: n.type, layer: n.layer, label: n.label })),
      edges: g.edges.map((e) => ({ u: e.u, v: e.v, type: e.type, serial: e.serial, nl_sentence: e.nl_sentence })),
      conversation: state.conversation,
    };
    els.rawPanel.textContent = JSON.stringify(summary, null, 2);
  }

  // ---------------------------------------------------------------------
  // Wiring
  // ---------------------------------------------------------------------

  function goStage(delta) {
    const nextIndex = state.stageIndex + delta;
    if (nextIndex < 0 || nextIndex >= STAGES.length) return;

    const g = state.graph;
    const targetStage = STAGES[nextIndex];
    if (delta > 0) {
      if (targetStage === "populate") isDefault() ? GEN.populateDefault(g) : GEN.populate(g);
      if (targetStage === "verbalize") GEN.generateEdgeSentences(g);
      if (targetStage === "reason" && !state.conversation) {
        state.conversation = isDefault() ? GEN.buildDefaultConversation(g) : GEN.buildConversation(g);
      }
    }
    state.stageIndex = nextIndex;
    renderAll();
  }

  // ---------------------------------------------------------------------
  // Parameters panel
  // ---------------------------------------------------------------------

  function renderParamsPanel() {
    const schema = PARAM_SCHEMAS[state.topology];
    const values = state.paramValues[state.topology];
    els.paramsFields.innerHTML = schema
      .map(
        (p) => `
        <div class="param-row">
          <label for="param-${p.key}">${esc(p.label)} <span class="param-val" id="param-${p.key}-val">${values[p.key]}</span></label>
          <input type="range" id="param-${p.key}" min="${p.min}" max="${p.max}" step="${p.step}" value="${values[p.key]}">
          <div class="param-desc">${esc(p.desc)}</div>
        </div>
      `
      )
      .join("");

    schema.forEach((p) => {
      const input = $(`param-${p.key}`);
      const val = $(`param-${p.key}-val`);
      input.addEventListener("input", () => {
        val.textContent = input.value;
      });
      input.addEventListener("change", () => {
        state.paramValues[state.topology][p.key] = Number(input.value);
        generateNew();
      });
    });
  }

  function toggleParamsPanel(open) {
    state.paramsOpen = open;
    els.paramsPanel.hidden = !open;
    els.simLayout.classList.toggle("params-open", open);
    els.paramsToggleBtn.setAttribute("aria-expanded", String(open));
    if (open) renderParamsPanel();
  }

  function init() {
    cacheEls();

    const options = ['<option value="random">Random (weighted)</option>'];
    Object.keys(D.ROOT_CONCEPT_WEIGHTS).forEach((k) => {
      options.push(`<option value="${k}">${k.charAt(0).toUpperCase() + k.slice(1)}</option>`);
    });
    els.rootConceptSelect.innerHTML = options.join("");

    els.topologySelect.addEventListener("change", () => {
      state.topology = els.topologySelect.value;
      if (state.paramsOpen) renderParamsPanel();
      generateNew();
    });
    els.generateBtn.addEventListener("click", generateNew);
    els.paramsToggleBtn.addEventListener("click", () => toggleParamsPanel(!state.paramsOpen));
    els.prevBtn.addEventListener("click", () => goStage(-1));
    els.nextBtn.addEventListener("click", () => goStage(1));
    els.stageSteps.forEach((stepEl, i) => {
      stepEl.addEventListener("click", () => {
        if (i <= state.stageIndex) {
          state.stageIndex = i;
          renderAll();
        }
      });
    });
    if (els.rawToggleBtn) {
      els.rawToggleBtn.addEventListener("click", () => {
        els.rawPanel.hidden = !els.rawPanel.hidden;
        els.rawToggleBtn.textContent = els.rawPanel.hidden ? "Show raw sample (JSON)" : "Hide raw sample (JSON)";
        renderRawPanel(state.graph);
      });
    }

    generateNew();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
