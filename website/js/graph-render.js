/**
 * graph-render.js
 * ---------------------------------------------------------------------
 * Pure-SVG renderer for the tree-inheritance simulator. Not a port of
 * anything in the Python pipeline -- purely a visual aid layered on top
 * of the ported generator/PMPM logic.
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

  function computeLayout(graph) {
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

  const COLORS = {
    root: "#3D6B4A",
    classNode: "#8FAE93",
    hypothesisRing: "#B4783C",
    property: "#C9A227",
    edgeInheritance: "#B9B2A0",
    edgeHasProperty: "#C9A227",
    edgeHypothesis: "#B4783C",
    edgeStrengthen: "#0072B2",
    edgeWeaken: "#E69F00",
    edgePending: "#DDD6C4",
    labelText: "#2B2A26",
    labelMuted: "#8A8577",
  };

  /**
   * Render the tree into `container` (a DOM element). `stage` is one of
   * "structure" | "populate" | "verbalize" | "reason".
   * `revealedSerials`: Set<number> of defeasible-edge serials to draw as
   * active (stage "reason" only); edges beyond this set render dimmed.
   * `activeSerial`: the serial currently being highlighted, if any.
   */
  function render(container, graph, opts) {
    opts = opts || {};
    const stage = opts.stage || "structure";
    const revealedSerials = opts.revealedSerials || new Set();
    const activeSerial = opts.activeSerial;

    container.innerHTML = "";
    const { pos, maxLeafX, maxDepth } = computeLayout(graph);

    const nodeCount = maxLeafX + 1;
    const colGap = 108;
    const rowGap = 96;
    const propertyColX = (maxLeafX + 1.5) * colGap;
    const width = propertyColX + colGap;
    const height = (maxDepth + 1) * rowGap + 60;

    const svg = el("svg", {
      viewBox: `0 0 ${width} ${height}`,
      width: "100%",
      height: Math.max(320, height),
      role: "img",
      "aria-label": "Generated tree-inheritance reasoning graph",
    });

    function px(nodeId) {
      const p = pos.get(nodeId);
      return { x: p.x * colGap + colGap / 2, y: p.y * rowGap + 44 };
    }
    const propertyPos = { x: propertyColX, y: ((maxDepth + 1) * rowGap) / 2 + 20 };

    const edgeLayer = el("g", { class: "edge-layer" });
    const nodeLayer = el("g", { class: "node-layer" });

    // ---- inheritance edges (always visible from stage 1) ----
    graph.edges
      .filter((e) => e.type === "inheritance")
      .forEach((e) => {
        const a = px(e.u);
        const b = px(e.v);
        edgeLayer.appendChild(
          el("line", {
            x1: a.x, y1: a.y, x2: b.x, y2: b.y,
            stroke: COLORS.edgeInheritance, "stroke-width": 2,
          })
        );
      });

    // ---- has_property backbone edge (root -> property), visible from "verbalize" on ----
    if (stage === "verbalize" || stage === "reason") {
      const root = px(graph.rootName);
      edgeLayer.appendChild(
        el("line", {
          x1: root.x, y1: root.y, x2: propertyPos.x, y2: propertyPos.y,
          stroke: COLORS.edgeHasProperty, "stroke-width": 2, "stroke-dasharray": "2 4",
        })
      );
    }

    // ---- hypothesis edge (subject -> property), visible from "verbalize" on ----
    if ((stage === "verbalize" || stage === "reason") && graph.hypothesisLeaf) {
      const subj = px(graph.hypothesisLeaf);
      edgeLayer.appendChild(
        el("line", {
          x1: subj.x, y1: subj.y, x2: propertyPos.x, y2: propertyPos.y,
          stroke: COLORS.edgeHypothesis, "stroke-width": 3,
        })
      );
    }

    // ---- defeasible edges, only meaningful in "reason" stage ----
    if (stage === "reason") {
      graph.edges
        .filter((e) => e.type === "defeasible_has_property" || e.type === "defeasible_lacks_property")
        .forEach((e) => {
          const a = px(e.u);
          const isRevealed = revealedSerials.has(e.serial);
          const isActive = e.serial === activeSerial;
          const strengthColor = e.type === "defeasible_has_property" ? COLORS.edgeStrengthen : COLORS.edgeWeaken;
          edgeLayer.appendChild(
            el("line", {
              x1: a.x, y1: a.y, x2: propertyPos.x, y2: propertyPos.y,
              stroke: isRevealed ? strengthColor : COLORS.edgePending,
              "stroke-width": isActive ? 4 : isRevealed ? 2.5 : 1.5,
              opacity: isRevealed ? 1 : 0.35,
            })
          );
        });
    }

    // ---- class + root nodes ----
    graph.nodes.forEach((n) => {
      if (n.type !== "class") return;
      const p = px(n.id);
      const isRoot = n.id === graph.rootName;
      const isHyp = n.id === graph.hypothesisLeaf;
      const r = isRoot ? 22 : 17;

      const g = el("g", { class: "graph-node", "data-node-id": n.id });
      if (isHyp) {
        g.appendChild(
          el("circle", { cx: p.x, cy: p.y, r: r + 5, fill: "none", stroke: COLORS.hypothesisRing, "stroke-width": 2.5 })
        );
      }
      g.appendChild(
        el("circle", {
          cx: p.x, cy: p.y, r,
          fill: isRoot ? COLORS.root : COLORS.classNode,
          stroke: "#fff", "stroke-width": 2,
        })
      );

      const showLabel = stage !== "structure";
      const labelText = showLabel ? n.label || n.id : isRoot ? "root" : "?";
      g.appendChild(
        el("text", {
          x: p.x, y: p.y + r + 16, "text-anchor": "middle",
          "font-size": 11.5, fill: showLabel ? COLORS.labelText : COLORS.labelMuted,
          "font-family": "var(--font-ui, sans-serif)",
        }, [document.createTextNode(labelText)])
      );
      nodeLayer.appendChild(g);
    });

    // ---- property node ----
    {
      const showLabel = stage !== "structure";
      const g = el("g", { class: "graph-node property-node" });
      g.appendChild(
        el("rect", {
          x: propertyPos.x - 26, y: propertyPos.y - 20, width: 52, height: 40, rx: 10,
          fill: COLORS.property, stroke: "#fff", "stroke-width": 2,
        })
      );
      g.appendChild(
        el("text", {
          x: propertyPos.x, y: propertyPos.y + 36, "text-anchor": "middle",
          "font-size": 11.5, fill: showLabel ? COLORS.labelText : COLORS.labelMuted,
          "font-family": "var(--font-ui, sans-serif)",
        }, [document.createTextNode(showLabel ? `"${graph.nodes.get(graph.propertyNode).label}"` : "target property")])
      );
      nodeLayer.appendChild(g);
    }

    svg.appendChild(edgeLayer);
    svg.appendChild(nodeLayer);
    container.appendChild(svg);
  }

  return { render, computeLayout, COLORS };
})();
