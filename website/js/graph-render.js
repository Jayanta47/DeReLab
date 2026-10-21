/**
 * graph-render.js
 * ---------------------------------------------------------------------
 * Pure-SVG renderer for the simulator. Not a port of anything in the
 * Python pipeline -- purely a visual aid layered on top of the ported
 * generator/PMPM logic. Dispatches on graph.topology:
 *   "tree_inheritance" | "linear_inheritance"  -> renderInheritance
 *   "default"                                  -> renderDefault
 * ---------------------------------------------------------------------
 */
window.DeReLabRender = (function () {
  "use strict";

  const SVG_NS = "http://www.w3.org/2000/svg";

  function el(name, attrs, children) {
    const node = document.createElementNS(SVG_NS, name);
    if (attrs) {
      Object.keys(attrs).forEach((k) => node.setAttribute(k, attrs[k]));
    }
    (children || []).forEach((c) => node.appendChild(c));
    return node;
  }

  const COLORS = {
    root: "#3D6B4A",
    classNode: "#8FAE93",
    entityNode: "#3D6B4A",
    irrelevantNode: "#B9B2A0",
    hypothesisRing: "#B4783C",
    property: "#C9A227",
    edgeStructural: "#B9B2A0",
    edgeBackbone: "#C9A227",
    edgeHypothesis: "#B4783C",
    edgeStrengthen: "#0072B2",
    edgeWeaken: "#E69F00",
    edgeNoEffect: "#B9B2A0",
    edgePending: "#DDD6C4",
    labelText: "#2B2A26",
    labelMuted: "#8A8577",
  };

  const FONT = "var(--font-ui, sans-serif)";
  const NODE_LABEL_SIZE = 19;
  const TAG_LABEL_SIZE = 19;

  function textEl(x, y, size, fill, str, extra) {
    return el(
      "text",
      Object.assign(
        { x, y, "text-anchor": "middle", "font-size": size, fill, "font-family": FONT },
        extra || {}
      ),
      [document.createTextNode(str)]
    );
  }

  // =====================================================================
  // Inheritance topologies (tree + linear): root at top, chain/tree grows
  // down via "inheritance" edges, all defeasible/hypothesis edges point to
  // one shared property node placed to the right.
  // =====================================================================

  function computeInheritanceLayout(graph) {
    const childrenOf = new Map();
    graph.edges.forEach((e) => {
      if (e.type === "inheritance") {
        if (!childrenOf.has(e.v)) childrenOf.set(e.v, []);
        childrenOf.get(e.v).push(e.u);
      }
    });

    let nextX = 0;
    const pos = new Map();

    function place(nodeId, depth) {
      const children = childrenOf.get(nodeId) || [];
      if (children.length === 0) {
        pos.set(nodeId, { x: nextX, y: depth });
        nextX += 1;
        return pos.get(nodeId).x;
      }
      const xs = children.map((c) => place(c, depth + 1));
      const x = xs.reduce((a, b) => a + b, 0) / xs.length;
      pos.set(nodeId, { x, y: depth });
      return x;
    }
    place(graph.rootName, 0);

    const maxLeafX = Math.max(1, nextX - 1);
    const maxDepth = Math.max(...[...pos.values()].map((p) => p.y), graph.preset.maxDepth);
    return { pos, maxLeafX, maxDepth };
  }

  function renderInheritance(container, graph, opts) {
    const stage = opts.stage;
    const revealedEdgeIds = opts.revealedEdgeIds;
    const activeEdgeId = opts.activeEdgeId;
    const isLinear = graph.topology === "linear_inheritance";

    const { pos, maxLeafX, maxDepth } = computeInheritanceLayout(graph);

    const colGap = 185;
    const rowGap = 160;
    const propertyColX = (maxLeafX + 1.6) * colGap;
    const width = propertyColX + colGap;
    const height = (maxDepth + 1) * rowGap + 90;

    const svg = el("svg", {
      viewBox: `0 0 ${width} ${height}`,
      width: "100%",
      height: Math.max(540, height),
      role: "img",
      "aria-label": `Generated ${isLinear ? "linear inheritance" : "tree inheritance"} reasoning graph`,
    });

    function px(nodeId) {
      const p = pos.get(nodeId);
      return { x: p.x * colGap + colGap / 2, y: p.y * rowGap + 54 };
    }
    const propertyPos = { x: propertyColX, y: ((maxDepth + 1) * rowGap) / 2 + 26 };

    const edgeLayer = el("g", { class: "edge-layer" });
    const nodeLayer = el("g", { class: "node-layer" });

    graph.edges
      .filter((e) => e.type === "inheritance")
      .forEach((e) => {
        const a = px(e.u);
        const b = px(e.v);
        edgeLayer.appendChild(
          el("line", { x1: a.x, y1: a.y, x2: b.x, y2: b.y, stroke: COLORS.edgeStructural, "stroke-width": 2.5 })
        );
      });

    if (stage === "verbalize" || stage === "reason") {
      const root = px(graph.rootName);
      edgeLayer.appendChild(
        el("line", {
          x1: root.x, y1: root.y, x2: propertyPos.x, y2: propertyPos.y,
          stroke: COLORS.edgeBackbone, "stroke-width": 2.5, "stroke-dasharray": "2 5",
        })
      );
    }

    if ((stage === "verbalize" || stage === "reason") && graph.hypothesisLeaf) {
      const subj = px(graph.hypothesisLeaf);
      edgeLayer.appendChild(
        el("line", {
          x1: subj.x, y1: subj.y, x2: propertyPos.x, y2: propertyPos.y,
          stroke: COLORS.edgeHypothesis, "stroke-width": 3.5,
        })
      );
    }

    if (stage === "reason") {
      graph.edges
        .filter((e) => e.type === "defeasible_has_property" || e.type === "defeasible_lacks_property")
        .forEach((e) => {
          const a = px(e.u);
          const isRevealed = revealedEdgeIds.has(e.id);
          const isActive = e.id === activeEdgeId;
          const strengthColor = e.type === "defeasible_has_property" ? COLORS.edgeStrengthen : COLORS.edgeWeaken;
          edgeLayer.appendChild(
            el("line", {
              x1: a.x, y1: a.y, x2: propertyPos.x, y2: propertyPos.y,
              stroke: isRevealed ? strengthColor : COLORS.edgePending,
              "stroke-width": isActive ? 4.5 : isRevealed ? 3 : 1.5,
              opacity: isRevealed ? 1 : 0.35,
            })
          );
        });
    }

    graph.nodes.forEach((n) => {
      if (n.type !== "class") return;
      const p = px(n.id);
      const isRoot = n.id === graph.rootName;
      const isHyp = n.id === graph.hypothesisLeaf;
      const r = isRoot ? 30 : 24;

      const g = el("g", { class: "graph-node", "data-node-id": n.id });
      if (isHyp) {
        g.appendChild(
          el("circle", { cx: p.x, cy: p.y, r: r + 7, fill: "none", stroke: COLORS.hypothesisRing, "stroke-width": 3.5 })
        );
      }
      g.appendChild(
        el("circle", { cx: p.x, cy: p.y, r, fill: isRoot ? COLORS.root : COLORS.classNode, stroke: "#fff", "stroke-width": 2.5 })
      );

      const showLabel = stage !== "structure";
      const labelText = showLabel ? n.label || n.id : isRoot ? "root" : "?";
      g.appendChild(textEl(p.x, p.y + r + 24, NODE_LABEL_SIZE, showLabel ? COLORS.labelText : COLORS.labelMuted, labelText));
      nodeLayer.appendChild(g);
    });

    {
      const showLabel = stage !== "structure";
      const g = el("g", { class: "graph-node property-node" });
      g.appendChild(
        el("rect", {
          x: propertyPos.x - 38, y: propertyPos.y - 28, width: 76, height: 56, rx: 14,
          fill: COLORS.property, stroke: "#fff", "stroke-width": 3,
        })
      );
      g.appendChild(
        textEl(
          propertyPos.x, propertyPos.y + 48, NODE_LABEL_SIZE, showLabel ? COLORS.labelText : COLORS.labelMuted,
          showLabel ? `"${graph.nodes.get(graph.propertyNode).label}"` : "target property"
        )
      );
      nodeLayer.appendChild(g);
    }

    svg.appendChild(edgeLayer);
    svg.appendChild(nodeLayer);
    container.appendChild(svg);
  }

  // =====================================================================
  // Default reasoning: property chain grows DOWNWARD (vertically) via
  // "implies" edges, root at top -- same orientation as the inheritance
  // topologies, but visually distinct because every is_a / defeasible /
  // hypothesis / distractor edge fans out from a small column of OBJECT
  // nodes to one side, converging on different points down the chain,
  // rather than converging on one shared property box from many leaves.
  // =====================================================================

  function computeDefaultLayout(graph) {
    const childrenOf = new Map();
    graph.edges.forEach((e) => {
      if (e.type === "implies") {
        if (!childrenOf.has(e.u)) childrenOf.set(e.u, []);
        childrenOf.get(e.u).push(e.v);
      }
    });
    let nextX = 0;
    const pos = new Map();
    function place(nodeId, depth) {
      const children = childrenOf.get(nodeId) || [];
      if (children.length === 0) {
        pos.set(nodeId, { x: nextX, y: depth });
        nextX += 1;
        return pos.get(nodeId).x;
      }
      const xs = children.map((c) => place(c, depth + 1));
      const x = xs.reduce((a, b) => a + b, 0) / xs.length;
      pos.set(nodeId, { x, y: depth });
      return x;
    }
    place(graph.rootPropertyNode, 0);
    const maxLeafX = Math.max(1, nextX - 1);
    const maxDepth = Math.max(...[...pos.values()].map((p) => p.y));
    return { pos, maxLeafX, maxDepth };
  }

  function renderDefault(container, graph, opts) {
    const stage = opts.stage;
    const revealedEdgeIds = opts.revealedEdgeIds;
    const activeEdgeId = opts.activeEdgeId;

    const { pos, maxLeafX, maxDepth } = computeDefaultLayout(graph);

    const colGap = 175;
    const rowGap = 155;
    const chainTopY = 60;
    const objectColX = 110;
    const chainBaseX = 300;
    const objectStep = 160;

    const chainHeight = (maxDepth + 1) * rowGap;
    const chainWidth = (maxLeafX + 1) * colGap;

    const numObjects = graph.objectIds.length;
    const objectsHeight = (numObjects - 1) * objectStep;
    const objectStartY = chainTopY + chainHeight / 2 - objectsHeight / 2;
    const objectPos = new Map();
    graph.objectIds.forEach((id, i) => {
      objectPos.set(id, { x: objectColX, y: objectStartY + i * objectStep });
    });
    const lowestObjectY = objectStartY + Math.max(0, numObjects - 1) * objectStep;
    const irrelevantPos = { x: objectColX - 20, y: lowestObjectY + objectStep };

    const width = Math.max(chainBaseX + chainWidth + colGap, 620);
    const height = Math.max(chainTopY + chainHeight + 80, irrelevantPos.y + 70);

    const svg = el("svg", {
      viewBox: `0 0 ${width} ${height}`,
      width: "100%",
      height: Math.max(540, height),
      role: "img",
      "aria-label": "Generated default-reasoning property chain graph",
    });

    function chainPx(nodeId) {
      const p = pos.get(nodeId);
      return { x: chainBaseX + p.x * colGap + colGap / 2, y: chainTopY + p.y * rowGap };
    }

    const edgeLayer = el("g", { class: "edge-layer" });
    const nodeLayer = el("g", { class: "node-layer" });

    // property chain backbone (implies edges) — always visible
    graph.edges
      .filter((e) => e.type === "implies")
      .forEach((e) => {
        const a = chainPx(e.u);
        const b = chainPx(e.v);
        edgeLayer.appendChild(
          el("line", { x1: a.x, y1: a.y, x2: b.x, y2: b.y, stroke: COLORS.edgeStructural, "stroke-width": 3 })
        );
      });

    // is_a (every object -> root property) — visible from "verbalize" on
    if (stage === "verbalize" || stage === "reason") {
      graph.edges
        .filter((e) => e.type === "is_a")
        .forEach((e) => {
          const a = objectPos.get(e.u);
          const b = chainPx(e.v);
          edgeLayer.appendChild(
            el("line", {
              x1: a.x, y1: a.y, x2: b.x, y2: b.y,
              stroke: COLORS.edgeBackbone, "stroke-width": 2.5, "stroke-dasharray": "2 5",
            })
          );
        });
    }

    // hypothesis edge (subject -> hypothesis target property)
    if ((stage === "verbalize" || stage === "reason") && graph.hypothesisSubject) {
      const a = objectPos.get(graph.hypothesisSubject);
      const b = chainPx(graph.hypothesisTarget);
      edgeLayer.appendChild(
        el("line", { x1: a.x, y1: a.y, x2: b.x, y2: b.y, stroke: COLORS.edgeHypothesis, "stroke-width": 3.5 })
      );
    }

    // defeasible edges (fan from the subject into the chain) + irrelevant
    // distractor edge (off to the side, never touching the chain) —
    // "reason" stage only
    if (stage === "reason") {
      const subjectPos = objectPos.get(graph.hypothesisSubject);
      graph.edges
        .filter((e) => e.type === "defeasible_has_property" || e.type === "defeasible_lacks_property")
        .forEach((e) => {
          const b = chainPx(e.v);
          const isRevealed = revealedEdgeIds.has(e.id);
          const isActive = e.id === activeEdgeId;
          const strengthColor = e.type === "defeasible_has_property" ? COLORS.edgeStrengthen : COLORS.edgeWeaken;
          edgeLayer.appendChild(
            el("line", {
              x1: subjectPos.x, y1: subjectPos.y, x2: b.x, y2: b.y,
              stroke: isRevealed ? strengthColor : COLORS.edgePending,
              "stroke-width": isActive ? 4.5 : isRevealed ? 3 : 1.5,
              opacity: isRevealed ? 1 : 0.35,
            })
          );
        });

      graph.edges
        .filter((e) => e.type === "has_attribute")
        .forEach((e) => {
          const isRevealed = revealedEdgeIds.has(e.id);
          const isActive = e.id === activeEdgeId;
          edgeLayer.appendChild(
            el("line", {
              x1: subjectPos.x, y1: subjectPos.y, x2: irrelevantPos.x, y2: irrelevantPos.y,
              stroke: isRevealed ? COLORS.edgeNoEffect : COLORS.edgePending,
              "stroke-width": isActive ? 4 : 2,
              opacity: isRevealed ? 1 : 0.3,
              "stroke-dasharray": "3 4",
            })
          );
        });
    }

    const showLabel = stage !== "structure";

    // property chain nodes
    graph.propertyNodeIds.forEach((id) => {
      const p = chainPx(id);
      const isRoot = id === graph.rootPropertyNode;
      const isHyp = id === graph.hypothesisTarget;
      const r = isRoot ? 30 : 24;
      const g = el("g", { class: "graph-node", "data-node-id": id });
      if (isHyp) {
        g.appendChild(el("circle", { cx: p.x, cy: p.y, r: r + 7, fill: "none", stroke: COLORS.hypothesisRing, "stroke-width": 3.5 }));
      }
      g.appendChild(el("circle", { cx: p.x, cy: p.y, r, fill: isRoot ? COLORS.root : COLORS.property, stroke: "#fff", "stroke-width": 3 }));
      const label = showLabel ? `"${graph.nodes.get(id).label}"` : isRoot ? "root" : "?";
      g.appendChild(textEl(p.x + r + 14, p.y + 5, NODE_LABEL_SIZE, showLabel ? COLORS.labelText : COLORS.labelMuted, label, { "text-anchor": "start" }));
      nodeLayer.appendChild(g);
    });

    // object nodes (hypothesis subject + any decoys)
    graph.objectIds.forEach((id) => {
      const p = objectPos.get(id);
      const isHyp = id === graph.hypothesisSubject;
      const g = el("g", { class: "graph-node", "data-node-id": id });
      if (isHyp) {
        g.appendChild(el("circle", { cx: p.x, cy: p.y, r: 33, fill: "none", stroke: COLORS.hypothesisRing, "stroke-width": 3.5 }));
      }
      g.appendChild(el("circle", { cx: p.x, cy: p.y, r: 26, fill: COLORS.entityNode, stroke: "#fff", "stroke-width": 3, opacity: isHyp ? 1 : 0.6 }));
      const label = showLabel ? graph.nodes.get(id).label : "?";
      g.appendChild(textEl(p.x, p.y + 54, NODE_LABEL_SIZE, showLabel ? COLORS.labelText : COLORS.labelMuted, label));
      if (isHyp) {
        g.appendChild(textEl(p.x, p.y - 40, TAG_LABEL_SIZE - 3, COLORS.hypothesisRing, "subject", { "font-weight": 600 }));
      }
      nodeLayer.appendChild(g);
    });

    // irrelevant distractor node
    {
      const g = el("g", { class: "graph-node" });
      g.appendChild(
        el("rect", {
          x: irrelevantPos.x - 38, y: irrelevantPos.y - 26, width: 76, height: 52, rx: 12,
          fill: COLORS.irrelevantNode, stroke: "#fff", "stroke-width": 3,
        })
      );
      const irrNode = [...graph.nodes.values()].find((n) => n.type === "irrelevant_property");
      const label = showLabel && irrNode ? `"${irrNode.label}"` : "distractor";
      g.appendChild(textEl(irrelevantPos.x, irrelevantPos.y + 44, NODE_LABEL_SIZE, showLabel ? COLORS.labelText : COLORS.labelMuted, label));
      nodeLayer.appendChild(g);
    }

    svg.appendChild(edgeLayer);
    svg.appendChild(nodeLayer);
    container.appendChild(svg);
  }

  // =====================================================================
  // Dispatch
  // =====================================================================

  /**
   * Render the graph into `container` (a DOM element). `stage` is one of
   * "structure" | "populate" | "verbalize" | "reason".
   * `revealedEdgeIds`: Set<string> of graph edge ids to draw as active
   * (stage "reason" only); edges outside this set render dimmed.
   * `activeEdgeId`: the edge id currently being highlighted, if any.
   */
  function render(container, graph, opts) {
    opts = opts || {};
    opts.revealedEdgeIds = opts.revealedEdgeIds || new Set();
    container.innerHTML = "";
    if (graph.topology === "default") {
      renderDefault(container, graph, opts);
    } else {
      renderInheritance(container, graph, opts);
    }
  }

  return { render, COLORS };
})();
