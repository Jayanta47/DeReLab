/**
 * simulator.js
 * ---------------------------------------------------------------------
 * UI controller for simulator.html. Wires the ported generator/PMPM
 * modules and the SVG renderer to the stage-stepper interface.
 * ---------------------------------------------------------------------
 */
(function () {
  "use strict";

  const GEN = window.DeReLabGenerator;
  const RENDER = window.DeReLabRender;
  const D = window.DeReLabData;

  const STAGES = ["structure", "populate", "verbalize", "reason"];
  const STAGE_LABELS = {
    structure: "1 · Structure",
    populate: "2 · Populate",
    verbalize: "3 · Verbalize",
    reason: "4 · Reason",
  };

  const state = {
    graph: null,
    conversation: null,
    stageIndex: 0,
    revealCount: 0, // number of defeasible turns revealed in "reason" stage
  };

  const els = {};

  function $(id) {
    return document.getElementById(id);
  }

  function cacheEls() {
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

  // ---------------------------------------------------------------------
  // Generation
  // ---------------------------------------------------------------------

  function generateNew() {
    const rootSel = els.rootConceptSelect.value;
    const opts = rootSel === "random" ? {} : { rootConcept: rootSel };
    state.graph = GEN.generateStructure(opts);
    state.conversation = null;
    state.stageIndex = 0;
    state.revealCount = 0;
    renderAll();
  }

  // ---------------------------------------------------------------------
  // Stage panel content
  // ---------------------------------------------------------------------

  function countByType(edges, type) {
    return edges.filter((e) => e.type === type).length;
  }

  function renderStructurePanel(g) {
    const numClass = [...g.nodes.values()].filter((n) => n.type === "class").length;
    const numDefeasible = countByType(g.edges, "defeasible_has_property") + countByType(g.edges, "defeasible_lacks_property");
    els.stagePanel.innerHTML = `
      <h3>Structure generation</h3>
      <p>A Galton&ndash;Watson branching process grew a taxonomy tree from a
      root class down to <strong>${g.preset.maxDepth}</strong> levels deep,
      using the <strong>${esc(g.preset.name)}</strong> shape preset from the
      easy-tier configuration (root breadth ${g.preset.rootBreadth}, survival
      probability ${g.preset.survivalProb}). One leaf was selected as the
      <em>hypothesis subject</em> &mdash; the node whose relationship to the
      target property the whole conversation will probe.</p>
      <dl class="stat-list">
        <dt>Root concept</dt><dd>${esc(g.rootConcept)}</dd>
        <dt>Class nodes</dt><dd>${numClass}</dd>
        <dt>Leaf nodes</dt><dd>${g.leafNodes.length}</dd>
        <dt>Hypothesis subject</dt><dd><code>${esc(g.hypothesisLeaf)}</code> (unlabeled until populated)</dd>
        <dt>Defeasible edges sampled</dt><dd>${numDefeasible} (sparsity ${g.sparsity.toFixed(2)})</dd>
      </dl>
      <p class="hint">Node labels are hidden at this stage &mdash; the graph
      is still a bare skeleton of typed nodes and edges, matching
      <code>TreeInheritanceGenerator.generate_structure()</code>.</p>
    `;
  }

  function renderPopulatePanel(g) {
    const sample = [...g.nodes.values()].filter((n) => n.type === "class" && n.id !== g.rootName).slice(0, 4);
    const propMeta = g.nodes.get(g.propertyNode).metadata;
    els.stagePanel.innerHTML = `
      <h3>Population</h3>
      <p>Every class node was assigned a label drawn from the knowledge
      base: the root keeps the semantic root concept (<strong>${esc(g.rootConcept)}</strong>),
      and each descendant class draws a fictional, phonotactically-plausible
      nonce word &mdash; so the model cannot lean on real-world priors about
      any specific taxon.</p>
      <dl class="stat-list">
        <dt>Target property</dt><dd><strong>${esc(propMeta.attribute_name)}</strong> &rarr; &ldquo;${esc(propMeta.filled_object_template)}&rdquo;</dd>
        <dt>Sample class labels</dt><dd>${sample.map((n) => esc(n.label)).join(", ")}${g.leafNodes.length > 4 ? ", &hellip;" : ""}</dd>
        <dt>Hypothesis subject label</dt><dd><strong>${esc(g.nodes.get(g.hypothesisLeaf).label)}</strong></dd>
      </dl>
      <p class="hint">This mirrors <code>GraphPopulator._populate_inheritance_tree()</code>:
      a BFS from the root assigns one concept-chain label per class node,
      then the property node is filled from a domain-specific attribute
      pool keyed to the root concept.</p>
    `;
  }

  function renderVerbalizePanel(g) {
    const backboneEdges = g.edges.filter((e) => e.type === "inheritance" || e.type === "has_property");
    const backboneSorted = backboneEdges.slice().sort((a, b) => (g.nodes.get(a.u).layer || 0) - (g.nodes.get(b.u).layer || 0));
    const pretext = backboneSorted.map((e) => e.nl_sentence).join(" ");
    const hypEdge = g.edges.find((e) => e.type === "hypothesis_edge");
    els.stagePanel.innerHTML = `
      <h3>Natural-language verbalization</h3>
      <p>Every backbone and defeasible edge was converted into a surface
      sentence via template filling
      (<code>NaturalLanguageGenerator.generate_edge_sentences()</code>).
      The backbone sentences below form the pretext the model sees before
      any update is revealed:</p>
      <blockquote class="pretext-block">${esc(pretext)}</blockquote>
      <p><strong>Hypothesis question:</strong><br>
      &ldquo;${esc(hypEdge.nl_sentence)}&rdquo;</p>
      <p class="hint">Defeasible-edge sentences (the per-turn evidence) are
      revealed one at a time in the next stage.</p>
    `;
  }

  function renderReasonPanel(g, convo) {
    const total = convo.turns.filter((t) => t.task_type === "update_answer").length;
    let html = `
      <h3>Staged reasoning</h3>
      <p>Backbone facts are revealed first, then each defeasible update is
      shown one at a time, in serial order. After every reveal the PMPM
      skeptical algorithm is re-queried over only the edges disclosed so
      far &mdash; exactly the process used to generate ground-truth labels
      for the benchmark.</p>
      <div class="transcript" id="transcript"></div>
      <div class="reveal-controls">
        <button id="reveal-next-btn" class="btn btn-primary" ${state.revealCount >= total ? "disabled" : ""}>
          Reveal next update (${state.revealCount}/${total})
        </button>
        <button id="reveal-reset-btn" class="btn btn-ghost">Restart reveal</button>
      </div>
    `;
    els.stagePanel.innerHTML = html;

    const transcript = $("transcript");
    const backboneTurn = convo.turns[0];
    const initialTurn = convo.turns[1];
    transcript.appendChild(
      turnBlock(
        "Pretext + Hypothesis",
        backboneTurn.content,
        `Initial answer: ${answerBadgeHTML(initialTurn.ground_truth)}`
      )
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
          `${answerBadgeHTML(assistantTurn.ground_truth)} &nbsp; effect: ${effectBadgeHTML(assistantTurn.effect)}`
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
      state.revealCount === 0
        ? initialTurn.ground_truth
        : convo.turns[2 + (state.revealCount - 1) * 2 + 1].ground_truth;
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

    if (stage !== "reason") {
      els.answerBadge.hidden = true;
    }

    let revealedSerials = new Set();
    let activeSerial = null;
    if (stage === "reason" && state.conversation) {
      const defeasibleEdgesSorted = g.edges
        .filter((e) => e.type === "defeasible_has_property" || e.type === "defeasible_lacks_property")
        .sort((a, b) => (a.serial || 0) - (b.serial || 0));
      for (let i = 0; i < state.revealCount; i++) {
        revealedSerials.add(defeasibleEdgesSorted[i].serial);
      }
      if (state.revealCount > 0) activeSerial = defeasibleEdgesSorted[state.revealCount - 1].serial;
    }

    RENDER.render(els.graphContainer, g, { stage, revealedSerials, activeSerial });

    if (stage === "structure") renderStructurePanel(g);
    else if (stage === "populate") renderPopulatePanel(g);
    else if (stage === "verbalize") renderVerbalizePanel(g);
    else if (stage === "reason") {
      if (!state.conversation) state.conversation = GEN.buildConversation(g);
      renderReasonPanel(g, state.conversation);
    }

    renderRawPanel(g);
  }

  function renderRawPanel(g) {
    if (!els.rawPanel || els.rawPanel.hidden) return;
    const summary = {
      root_concept: g.rootConcept,
      preset: g.preset.name,
      sparsity_factor: Number(g.sparsity.toFixed(4)),
      hypothesis_subject: g.hypothesisLeaf,
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
      if (targetStage === "populate") GEN.populate(g);
      if (targetStage === "verbalize") GEN.generateEdgeSentences(g);
      if (targetStage === "reason" && !state.conversation) state.conversation = GEN.buildConversation(g);
    }
    state.stageIndex = nextIndex;
    renderAll();
  }

  function init() {
    cacheEls();

    const options = ['<option value="random">Random (weighted)</option>'];
    Object.keys(D.ROOT_CONCEPT_WEIGHTS).forEach((k) => {
      options.push(`<option value="${k}">${k.charAt(0).toUpperCase() + k.slice(1)}</option>`);
    });
    els.rootConceptSelect.innerHTML = options.join("");

    els.generateBtn.addEventListener("click", generateNew);
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
