/**
 * pmpm.js
 * ---------------------------------------------------------------------
 * Direct JavaScript port of the PMPM skeptical path-relation algorithm
 * (path_relation_module/pmpm_algo.py) and the graph resolver that signs
 * DeReLab edges before handing them to it
 * (path_relation_module/graph_resolutor.py).
 *
 * Operates on a plain adjacency-list digraph:
 *   { nodes: Set<string>,
 *     edges: Map<string, Array<{to: string, sign: '+'|'-'}>> }   // successors
 *     predecessors: Map<string, Array<{from: string, sign}>>
 *
 * `queryDefeasibleGraph(graph, x, y)` returns "True" | "False" |
 * "Skeptical/Unknown", exactly matching the Python return values so the
 * rest of the pipeline can reuse the same normalisation step.
 * ---------------------------------------------------------------------
 */
window.DeReLabPMPM = (function () {
  "use strict";

  function makeSignedGraph() {
    return {
      nodes: new Set(),
      succ: new Map(), // node -> [{to, sign}]
      pred: new Map(), // node -> [{from, sign}]
    };
  }

  function addNode(g, n) {
    if (!g.nodes.has(n)) {
      g.nodes.add(n);
      g.succ.set(n, []);
      g.pred.set(n, []);
    }
  }

  function addEdge(g, u, v, sign) {
    addNode(g, u);
    addNode(g, v);
    g.succ.get(u).push({ to: v, sign });
    g.pred.get(v).push({ from: u, sign });
  }

  function hasEdge(g, u, v) {
    const list = g.succ.get(u) || [];
    return list.find((e) => e.to === v) || null;
  }

  // ---- GraphResolutor.build_algorithm_graph port ----
  const POSITIVE_EDGE_TYPES = new Set(["inheritance", "is_a", "implies", "has_property", "defeasible_has_property"]);
  const NEGATIVE_EDGE_TYPES = new Set(["defeasible_lacks_property"]);
  const PASS_THROUGH_EDGE_TYPES = new Set(["hypothesis_edge", "has_attribute"]);
  const DEFEASIBLE_EDGE_TYPES = new Set(["defeasible_has_property", "defeasible_lacks_property"]);

  /**
   * rawEdges: Array<{u, v, type, serial}>
   * serialCutoff: number|null -- defeasible edges with serial > cutoff are excluded.
   *               Pass 0 for a backbone-only view.
   */
  function buildAlgorithmGraph(rawNodes, rawEdges, serialCutoff) {
    const g = makeSignedGraph();
    rawNodes.forEach((n) => addNode(g, n));
    for (const { u, v, type, serial } of rawEdges) {
      if (PASS_THROUGH_EDGE_TYPES.has(type)) continue;
      if (serialCutoff !== null && DEFEASIBLE_EDGE_TYPES.has(type)) {
        if ((serial || 0) > serialCutoff) continue;
      }
      if (POSITIVE_EDGE_TYPES.has(type)) addEdge(g, u, v, "+");
      else if (NEGATIVE_EDGE_TYPES.has(type)) addEdge(g, u, v, "-");
    }
    return g;
  }

  function resolve(rawNodes, rawEdges, x, y, serialCutoff) {
    const g = buildAlgorithmGraph(rawNodes, rawEdges, serialCutoff);
    return queryDefeasibleGraph(g, x, y);
  }

  // ---- pmpm_algo.query_defeasible_graph port ----
  function queryDefeasibleGraph(G, x, y) {
    function getTrimmedNodes() {
      const m1 = new Set([x]);
      const queue = [x];
      while (queue.length) {
        const curr = queue.shift();
        for (const { to, sign } of G.succ.get(curr) || []) {
          if (sign === "+" && !m1.has(to)) {
            m1.add(to);
            queue.push(to);
          }
        }
      }

      const m2 = new Set();
      const yPreds = G.pred.get(y) || [];
      if (yPreds.some((e) => m1.has(e.from))) {
        m2.add(y);
      }
      if (!m2.has(y)) return new Set();

      const q2 = [y];
      while (q2.length) {
        const curr = q2.shift();
        for (const { from: parent } of G.pred.get(curr) || []) {
          if (m1.has(parent) && !m2.has(parent)) {
            m2.add(parent);
            q2.push(parent);
          }
        }
      }
      return m2;
    }

    function resolveConflict(tentativePos, tentativeNeg, provenTrue, provenFalse) {
      const mPre = new Set();
      const queue = [...tentativePos, ...tentativeNeg];
      while (queue.length) {
        const curr = queue.shift();
        for (const { to: superclass, sign } of G.succ.get(curr) || []) {
          if (sign !== "+") continue;
          if (provenTrue.has(superclass) && !provenFalse.has(superclass)) {
            if (!mPre.has(superclass)) {
              mPre.add(superclass);
              queue.push(superclass);
            }
          }
        }
      }

      const validPos = [...tentativePos].filter((n) => !mPre.has(n));
      const validNeg = [...tentativeNeg].filter((n) => !mPre.has(n));

      if (validPos.length && !validNeg.length) return "True";
      if (validNeg.length && !validPos.length) return "False";
      return "Skeptical/Unknown";
    }

    const trimmedNodes = getTrimmedNodes();
    if (trimmedNodes.size === 0) return "Skeptical/Unknown";

    // Induced subgraph restricted to trimmedNodes, topologically sorted.
    const inSub = (n) => trimmedNodes.has(n);
    const subSucc = new Map();
    const subPred = new Map();
    trimmedNodes.forEach((n) => {
      subSucc.set(n, (G.succ.get(n) || []).filter((e) => inSub(e.to)));
      subPred.set(n, (G.pred.get(n) || []).filter((e) => inSub(e.from)));
    });

    const topoOrder = topologicalSort(trimmedNodes, subSucc);
    if (topoOrder === null) {
      throw new Error("Graph contains cycles. Skeptical algorithm requires a DAG.");
    }

    const provenTrue = new Set();
    const provenFalse = new Set();

    for (const node of topoOrder) {
      if (node === x) {
        provenTrue.add(node);
        continue;
      }

      const direct = hasEdge(G, x, node);
      if (direct) {
        if (direct.sign === "+") {
          provenTrue.add(node);
          continue;
        } else if (direct.sign === "-") {
          provenFalse.add(node);
          continue;
        }
      }

      const tentativePos = new Set();
      const tentativeNeg = new Set();
      for (const { from: parent, sign } of subPred.get(node) || []) {
        if (!provenTrue.has(parent)) continue;
        if (sign === "+") tentativePos.add(parent);
        else if (sign === "-") tentativeNeg.add(parent);
      }

      if (tentativePos.size && !tentativeNeg.size) {
        provenTrue.add(node);
      } else if (tentativeNeg.size && !tentativePos.size) {
        provenFalse.add(node);
      } else if (tentativePos.size && tentativeNeg.size) {
        const resolution = resolveConflict(tentativePos, tentativeNeg, provenTrue, provenFalse);
        if (resolution === "True") provenTrue.add(node);
        else if (resolution === "False") provenFalse.add(node);
      } else {
        const blocked = (subPred.get(node) || []).some(
          ({ from: parent, sign }) => provenFalse.has(parent) && sign === "+"
        );
        if (blocked) provenFalse.add(node);
      }
    }

    if (provenTrue.has(y)) return "True";
    if (provenFalse.has(y)) return "False";
    return "Skeptical/Unknown";
  }

  // Kahn's algorithm; returns null on a cycle (mirrors nx.NetworkXUnfeasible).
  function topologicalSort(nodeSet, succMap) {
    const inDegree = new Map();
    nodeSet.forEach((n) => inDegree.set(n, 0));
    nodeSet.forEach((n) => {
      (succMap.get(n) || []).forEach((e) => {
        inDegree.set(e.to, (inDegree.get(e.to) || 0) + 1);
      });
    });
    const queue = [...nodeSet].filter((n) => inDegree.get(n) === 0);
    const order = [];
    while (queue.length) {
      const n = queue.shift();
      order.push(n);
      (succMap.get(n) || []).forEach((e) => {
        inDegree.set(e.to, inDegree.get(e.to) - 1);
        if (inDegree.get(e.to) === 0) queue.push(e.to);
      });
    }
    return order.length === nodeSet.size ? order : null;
  }

  function normalizeAnswer(answer) {
    const mapping = { True: "yes", False: "no", "Skeptical/Unknown": "unknown" };
    return mapping[answer] || answer.toLowerCase();
  }

  return { resolve, buildAlgorithmGraph, queryDefeasibleGraph, normalizeAnswer };
})();
