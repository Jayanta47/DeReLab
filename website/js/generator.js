/**
 * generator.js
 * ---------------------------------------------------------------------
 * In-browser port of the tree-inheritance pipeline, staged so the
 * simulator UI can stop after each phase:
 *
 *   1. structure()   -- graph_generation/graph_generator.py: TreeInheritanceGenerator
 *   2. populate()     -- graph_populator/graph_populator.py: _populate_inheritance_tree
 *   3. verbalize()    -- question_generation/nl_generator.py: generate_edge_sentences
 *   4. converse()     -- question_generation/nl_generator.py: _build_tree_inheritance_conversations
 *
 * Only the easy-tier tree-inheritance configuration is implemented,
 * matching DIFFICULTY_PRESETS["easy"] in generate_tree_inheritance.py.
 * ---------------------------------------------------------------------
 */
window.DeReLabGenerator = (function () {
  "use strict";

  const D = window.DeReLabData;
  const PMPM = window.DeReLabPMPM;

  function rand() {
    return Math.random();
  }
  function randInt(min, max) {
    // inclusive of both ends, mirrors Python's random.randint
    return Math.floor(rand() * (max - min + 1)) + min;
  }
  function choice(arr) {
    return arr[Math.floor(rand() * arr.length)];
  }
  function sampleWithoutReplacement(arr, n) {
    const pool = arr.slice();
    const out = [];
    const count = Math.min(n, pool.length);
    for (let i = 0; i < count; i++) {
      const idx = Math.floor(rand() * pool.length);
      out.push(pool.splice(idx, 1)[0]);
    }
    return out;
  }
  function shuffle(arr) {
    const a = arr.slice();
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  }
  function weightedChoice(items, weights) {
    const total = weights.reduce((a, b) => a + b, 0);
    let r = rand() * total;
    for (let i = 0; i < items.length; i++) {
      r -= weights[i];
      if (r <= 0) return items[i];
    }
    return items[items.length - 1];
  }
  function uid() {
    return Math.random().toString(16).slice(2, 10);
  }

  // =====================================================================
  // PHASE 1: Structure -- TreeInheritanceGenerator
  // =====================================================================

  function generateStructure(opts) {
    const rootConcept =
      opts && opts.rootConcept
        ? opts.rootConcept
        : weightedChoice(Object.keys(D.ROOT_CONCEPT_WEIGHTS), Object.values(D.ROOT_CONCEPT_WEIGHTS));
    const preset = (opts && opts.preset) || choice(D.EASY_PRESETS);
    const sparsity =
      opts && opts.sparsity != null
        ? opts.sparsity
        : D.EASY_SPARSITY_RANGE[0] + rand() * (D.EASY_SPARSITY_RANGE[1] - D.EASY_SPARSITY_RANGE[0]);

    const rootName = rootConcept.charAt(0).toUpperCase() + rootConcept.slice(1);

    const nodes = new Map(); // id -> {type, layer}
    const edges = []; // {u, v, type, serial?, id}
    const leafNodes = [];

    function addNode(id, type, layer) {
      nodes.set(id, { id, type, layer });
    }
    function addEdge(u, v, type, extra) {
      edges.push(Object.assign({ u, v, type, id: uid() }, extra || {}));
    }

    // ---- _build_skeleton / _grow_chains_recursively ----
    addNode(rootName, "class", 0);

    function growChains(parentId, depth) {
      if (depth >= preset.maxDepth) {
        leafNodes.push(parentId);
        return;
      }
      if (depth > 0) {
        if (rand() > preset.survivalProb) {
          leafNodes.push(parentId);
          return;
        }
      }
      let numChildren;
      if (depth === 0) {
        numChildren = preset.rootBreadth;
      } else {
        const decayedMax = Math.floor(preset.maxBranching * Math.pow(preset.branchingDecay, depth - 1));
        const actualMax = Math.max(preset.minBranching, decayedMax);
        numChildren = randInt(preset.minBranching, actualMax);
      }
      let generated = 0;
      for (let i = 0; i < numChildren; i++) {
        const childId = `${parentId}_sub_${uid().slice(0, 4)}`;
        addNode(childId, "class", depth + 1);
        addEdge(childId, parentId, "inheritance");
        growChains(childId, depth + 1);
        generated++;
      }
      if (generated === 0) leafNodes.push(parentId);
    }
    growChains(rootName, 0);

    // ---- _add_property_and_base_rule ----
    const propertyNode = "Target_Property";
    addNode(propertyNode, "property", 0);
    addEdge(rootName, propertyNode, "has_property");

    // ---- _add_hypothesis_edges ----
    let hypothesisLeaf = null;
    if (leafNodes.length) {
      if (rand() < 0.8) {
        const maxDepth = Math.max(...leafNodes.map((n) => nodes.get(n).layer));
        const deepest = leafNodes.filter((n) => nodes.get(n).layer === maxDepth);
        hypothesisLeaf = choice(deepest);
      } else {
        hypothesisLeaf = choice(leafNodes);
      }
      addEdge(hypothesisLeaf, propertyNode, "hypothesis_edge");
    }

    // ---- _add_defeasible_edges_sampler (serial_random=True for easy tier) ----
    const eligible = [...nodes.values()].filter(
      (d) => d.type === "class" && d.layer > 0 && d.id !== hypothesisLeaf
    );
    const selected = shuffle(eligible.filter(() => rand() < sparsity));
    selected.forEach((node, i) => {
      const edgeType = choice(["defeasible_has_property", "defeasible_lacks_property"]);
      addEdge(node.id, propertyNode, edgeType, { serial: i + 1 });
    });

    return {
      rootName,
      rootConcept,
      propertyNode,
      hypothesisLeaf,
      preset,
      sparsity,
      nodes,
      edges,
      leafNodes,
    };
  }

  // =====================================================================
  // PHASE 2: Population -- GraphPopulator._populate_inheritance_tree
  // =====================================================================

  function populate(graph) {
    const domainKey = graph.rootConcept.charAt(0).toUpperCase() + graph.rootConcept.slice(1);
    const noncePool = D.NONCE_ENTITIES[domainKey] || D.NONCE_ENTITIES.Animal;

    // BFS from root following inheritance edges in reverse (children first).
    const childrenByParent = new Map();
    graph.edges.forEach((e) => {
      if (e.type === "inheritance") {
        if (!childrenByParent.has(e.v)) childrenByParent.set(e.v, []);
        childrenByParent.get(e.v).push(e.u);
      }
    });
    const ordered = [];
    const visited = new Set();
    const queue = [graph.rootName];
    while (queue.length) {
      const node = queue.shift();
      if (visited.has(node)) continue;
      visited.add(node);
      if (graph.nodes.get(node).type === "class") ordered.push(node);
      (childrenByParent.get(node) || []).forEach((c) => {
        if (!visited.has(c)) queue.push(c);
      });
    }

    // Root label is the lowercase root_concept string itself (matches
    // KnowledgeBaseManager.get_concept_chain, which prepends the raw
    // root_concept -- not the capitalised node id -- to the nonce chain);
    // the rest draw nonce words.
    const nonceNeeded = ordered.length - 1;
    const nonceWords = sampleWithoutReplacement(noncePool, nonceNeeded);
    ordered.forEach((nodeId, i) => {
      const label = i === 0 ? graph.rootConcept : nonceWords[i - 1] || `concept_${i}`;
      const n = graph.nodes.get(nodeId);
      n.label = label;
      n.metadata = { concept: label, layer: n.layer };
    });

    // Property node: one random attribute from the domain pool.
    const domainAttrs = D.DOMAIN_ATTRIBUTES[domainKey] || D.DOMAIN_ATTRIBUTES.Animal;
    const attrKeys = Object.keys(domainAttrs);
    const attrKey = choice(attrKeys);
    const attrData = domainAttrs[attrKey];
    const selectedValue = choice(attrData.examples).toLowerCase();
    const propInfo = {
      attribute_name: attrKey,
      selected_value: selectedValue,
      filled_template: attrData.template.replace("[value]", selectedValue),
      filled_object_template: attrData.object_template.replace("[value]", selectedValue),
    };
    const propNode = graph.nodes.get(graph.propertyNode);
    propNode.label = propInfo.selected_value;
    propNode.metadata = propInfo;

    return graph;
  }

  // =====================================================================
  // PHASE 3: Verbalization -- NaturalLanguageGenerator.generate_edge_sentences
  // =====================================================================

  function buildPayload(graph, sourceId, targetId) {
    const source = graph.nodes.get(sourceId);
    const target = graph.nodes.get(targetId);
    const sMeta = source.metadata || {};
    const tMeta = target.metadata || {};

    const objectPredicate = tMeta.filled_object_template || `has the property '${target.label || targetId}'`;
    const objectPredicatePlural = tMeta.filled_template || `have the property '${target.label || targetId}'`;
    const subjectPredicate = sMeta.filled_template || `has the property '${source.label || sourceId}'`;

    return {
      subject: source.label || sourceId,
      object: target.label || targetId,
      subject_predicate: subjectPredicate,
      object_predicate: objectPredicate,
      object_predicate_plural: objectPredicatePlural,
    };
  }

  function makePropertyQuestionFromPredicate(subject, predicate) {
    if (predicate.startsWith("is ")) return `Is ${subject} ${predicate.slice(3)}?`;
    if (predicate.startsWith("has ")) return `Does ${subject} have ${predicate.slice(4)}?`;
    if (predicate.startsWith("can ")) return `Can ${subject} ${predicate.slice(4)}?`;
    if (predicate.startsWith("serves as ")) return `Does ${subject} serve as ${predicate.slice(10)}?`;
    if (predicate.startsWith("appears ")) return `Does ${subject} appear ${predicate.slice(8)}?`;
    if (predicate.startsWith("produces ")) return `Does ${subject} produce ${predicate.slice(9)}?`;
    if (predicate.startsWith("tastes or smells "))
      return `Does ${subject} taste or smell ${predicate.slice(17)}?`;
    return `Does ${subject} ${predicate}?`;
  }

  function generateNegativeUpdate(payload) {
    const predicate = payload.object_predicate;
    const subject = payload.subject;
    if (predicate.startsWith("is ")) return `${subject} is not ${predicate.slice(3)}.`;
    if (predicate.startsWith("has ")) return `${subject} does not have ${predicate.slice(4)}.`;
    if (predicate.startsWith("can ")) return `${subject} cannot ${predicate.slice(4)}.`;
    if (predicate.startsWith("serves as ")) return `${subject} does not serve as ${predicate.slice(10)}.`;
    if (predicate.startsWith("appears ")) return `${subject} does not appear ${predicate.slice(8)}.`;
    if (predicate.startsWith("produces ")) return `${subject} does not produce ${predicate.slice(9)}.`;
    if (predicate.startsWith("tastes or smells "))
      return `${subject} does not taste or smell ${predicate.slice(17)}.`;
    if (predicate.startsWith("displays ")) return `${subject} does not display ${predicate.slice(9)}.`;
    if (predicate.startsWith("thrives ")) return `${subject} does not thrive ${predicate.slice(8)}.`;
    return `${subject} does not ${predicate}.`;
  }

  function generateSentence(graph, edgeType, payload) {
    if (edgeType === "defeasible_lacks_property") return generateNegativeUpdate(payload);

    if (edgeType === "has_property") {
      const subject = payload.subject;
      const pluralSubject = subject.endsWith("s") ? subject : subject + "s";
      const capitalized = pluralSubject.charAt(0).toUpperCase() + pluralSubject.slice(1);
      return `${capitalized} ${payload.object_predicate_plural}.`;
    }

    if (edgeType === "hypothesis_edge") {
      return makePropertyQuestionFromPredicate(payload.subject, payload.object_predicate);
    }

    if (edgeType === "inheritance") {
      return `${payload.subject} is a kind of ${payload.object}.`;
    }

    if (edgeType === "defeasible_has_property") {
      return `${payload.subject} ${payload.object_predicate}.`;
    }

    return `${payload.subject} is related to ${payload.object}.`;
  }

  function generateEdgeSentences(graph) {
    graph.edges.forEach((edge) => {
      const payload = buildPayload(graph, edge.u, edge.v);
      let sentence = generateSentence(graph, edge.type, payload);
      if (sentence) {
        sentence = sentence.charAt(0).toUpperCase() + sentence.slice(1);
        edge.nl_sentence = sentence;
      }
    });
    return graph;
  }

  // =====================================================================
  // PHASE 4: Conversation -- _build_tree_inheritance_conversations
  // =====================================================================

  const POSITIVE_EDGE_TYPES_FOR_PATH = new Set(["inheritance", "is_a", "implies", "has_property"]);

  function isOnPositivePath(graph, node, subject) {
    if (node === subject) return true;
    const succByNode = new Map();
    graph.edges.forEach((e) => {
      if (!succByNode.has(e.u)) succByNode.set(e.u, []);
      succByNode.get(e.u).push(e);
    });
    const visited = new Set();
    const queue = [subject];
    while (queue.length) {
      const curr = queue.shift();
      if (visited.has(curr)) continue;
      visited.add(curr);
      for (const e of succByNode.get(curr) || []) {
        if (visited.has(e.v)) continue;
        if (POSITIVE_EDGE_TYPES_FOR_PATH.has(e.type)) {
          if (e.v === node) return true;
          queue.push(e.v);
        }
      }
    }
    return false;
  }

  function classifyInheritanceEffect(graph, sourceNode, edgeType, subject) {
    if (!isOnPositivePath(graph, sourceNode, subject)) return "no effect";
    if (edgeType === "defeasible_has_property") return "strengthen";
    if (edgeType === "defeasible_lacks_property") return "weaken";
    return "no effect";
  }

  function makePropertyStatement(graph, subjectId, propertyNodeId) {
    const subjectLabel = graph.nodes.get(subjectId).label || subjectId;
    const meta = graph.nodes.get(propertyNodeId).metadata || {};
    const predicate = meta.filled_object_template || `have the property '${graph.nodes.get(propertyNodeId).label}'`;
    return `${subjectLabel.charAt(0).toUpperCase() + subjectLabel.slice(1)} ${predicate}.`;
  }

  function composePretext(graph) {
    const relevantTypes = new Set(["inheritance", "has_property"]);
    const ordered = graph.edges
      .filter((e) => relevantTypes.has(e.type) && e.nl_sentence)
      .slice()
      .sort((a, b) => {
        const la = graph.nodes.get(a.u).layer || 0;
        const lb = graph.nodes.get(b.u).layer || 0;
        return la - lb;
      });
    return ordered.map((e) => e.nl_sentence).join(" ");
  }

  function buildConversation(graph) {
    const hypEdge = graph.edges.find((e) => e.type === "hypothesis_edge");
    if (!hypEdge) return null;
    const subject = hypEdge.u;
    const hypothesisTarget = hypEdge.v;
    const subjectLabel = graph.nodes.get(subject).label || subject;
    const hypothesisStatement = makePropertyStatement(graph, subject, hypothesisTarget);

    const defeasibleEdges = graph.edges
      .filter((e) => e.type === "defeasible_has_property" || e.type === "defeasible_lacks_property")
      .slice()
      .sort((a, b) => (a.serial || 0) - (b.serial || 0));

    const backboneText = composePretext(graph);

    const rawNodes = [...graph.nodes.keys()];
    const rawEdges = graph.edges.map((e) => ({ u: e.u, v: e.v, type: e.type, serial: e.serial }));

    const entailmentQ = "Given all the information so far, does the information support the hypothesis?";

    const initialAnswer = PMPM.normalizeAnswer(PMPM.resolve(rawNodes, rawEdges, subject, hypothesisTarget, 0));

    const hypDistFromRoot = graph.nodes.get(subject).layer;

    // Undirected class-node adjacency (inheritance edges only), for the
    // dist_to_subject field -- mirrors the class_undirected subgraph in
    // _build_tree_inheritance_conversations.
    const classAdj = new Map();
    graph.edges.forEach((e) => {
      if (e.type !== "inheritance") return;
      if (!classAdj.has(e.u)) classAdj.set(e.u, []);
      if (!classAdj.has(e.v)) classAdj.set(e.v, []);
      classAdj.get(e.u).push(e.v);
      classAdj.get(e.v).push(e.u);
    });
    function distToSubject(node) {
      if (node === subject) return 0;
      const visited = new Set([node]);
      const queue = [[node, 0]];
      while (queue.length) {
        const [curr, d] = queue.shift();
        for (const nb of classAdj.get(curr) || []) {
          if (visited.has(nb)) continue;
          if (nb === subject) return d + 1;
          visited.add(nb);
          queue.push([nb, d + 1]);
        }
      }
      return -1;
    }

    const turns = [
      {
        role: "user",
        content: `Pretext:\n${backboneText}\n\nHypothesis: ${hypothesisStatement}\n\nQuestion: ${entailmentQ}`,
        stage: "backbone",
      },
      {
        role: "assistant",
        content: "",
        ground_truth: initialAnswer,
        task_type: "initial_answer",
      },
    ];

    defeasibleEdges.forEach((edge) => {
      const serial = edge.serial || 0;
      const currentAnswer = PMPM.normalizeAnswer(
        PMPM.resolve(rawNodes, rawEdges, subject, hypothesisTarget, serial)
      );
      const effect = classifyInheritanceEffect(graph, edge.u, edge.type, subject);
      turns.push({
        role: "user",
        content: `Information update:\n${edge.nl_sentence}\n\nQuestion: ${entailmentQ}`,
        serial,
        sourceNode: edge.u,
        distance: [graph.nodes.get(edge.u).layer, distToSubject(edge.u)],
      });
      turns.push({
        role: "assistant",
        content: "",
        ground_truth: currentAnswer,
        effect,
        task_type: "update_answer",
      });
    });

    return {
      id: `${subject}_${hypothesisTarget}_conversation`,
      topology: "tree_inheritance",
      metadata: {
        subject: subjectLabel,
        hypothesis_target: graph.nodes.get(hypothesisTarget).label,
        hypothesis_statement: hypothesisStatement,
        hypothesis_distance_from_root: hypDistFromRoot,
      },
      turns,
    };
  }

  return {
    generateStructure,
    populate,
    generateEdgeSentences,
    buildConversation,
  };
})();
