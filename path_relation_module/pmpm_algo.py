import networkx as nx


def query_defeasible_graph(G, x, y):
    """
    Evaluates whether x inherits property y in an acyclic nonmonotonic graph.

    Edges must have a 'sign' attribute:
    '+' for supporting links and '-' for defeating links.
    """

    def get_trimmed_nodes():
        m1 = {x}
        queue = [x]
        while queue:
            curr = queue.pop(0)
            for neighbor in G.successors(curr):
                if G[curr][neighbor].get("sign") == "+" and neighbor not in m1:
                    m1.add(neighbor)
                    queue.append(neighbor)

        m2 = set()
        for parent in G.predecessors(y):
            if parent in m1:
                m2.add(y)
                break

        if y not in m2:
            return set()

        queue = [y]
        while queue:
            curr = queue.pop(0)
            for parent in G.predecessors(curr):
                if parent in m1 and parent not in m2:
                    m2.add(parent)
                    queue.append(parent)

        return m2

    def resolve_conflict(
        tentative_pos_sources, tentative_neg_sources, proven_true, proven_false
    ):
        m_pre = set()
        queue = list(tentative_pos_sources.union(tentative_neg_sources))

        while queue:
            curr = queue.pop(0)
            for superclass in G.successors(curr):
                if G[curr][superclass].get("sign") != "+":
                    continue
                if superclass in proven_true and superclass not in proven_false:
                    if superclass not in m_pre:
                        m_pre.add(superclass)
                        queue.append(superclass)

        valid_pos = tentative_pos_sources - m_pre
        valid_neg = tentative_neg_sources - m_pre

        if valid_pos and not valid_neg:
            return "True"
        if valid_neg and not valid_pos:
            return "False"
        return "Skeptical/Unknown"

    trimmed_nodes = get_trimmed_nodes()
    if not trimmed_nodes:
        return "Skeptical/Unknown"

    g_trimmed = G.subgraph(trimmed_nodes)

    try:
        topo_order = list(nx.topological_sort(g_trimmed))
    except nx.NetworkXUnfeasible as exc:
        raise ValueError("Graph contains cycles. Skeptical algorithm requires a DAG.") from exc

    proven_true = set()
    proven_false = set()

    for node in topo_order:
        if node == x:
            proven_true.add(node)
            continue

        # Direct edges from x take absolute priority (pseudocode steps 6–9:
        # off-head[M_T, M_F] in steps 10–13 means already-classified nodes
        # are never reconsidered by inherited paths).
        if G.has_edge(x, node):
            if G[x][node].get("sign") == "+":
                proven_true.add(node)
                continue
            elif G[x][node].get("sign") == "-":
                proven_false.add(node)
                continue

        tentative_pos = set()
        tentative_neg = set()
        for parent in g_trimmed.predecessors(node):
            if parent not in proven_true:
                continue
            sign = G[parent][node].get("sign")
            if sign == "+":
                tentative_pos.add(parent)
            elif sign == "-":
                tentative_neg.add(parent)

        if tentative_pos and not tentative_neg:
            proven_true.add(node)
        elif tentative_neg and not tentative_pos:
            proven_false.add(node)
        elif tentative_pos and tentative_neg:
            resolution = resolve_conflict(
                tentative_pos,
                tentative_neg,
                proven_true,
                proven_false,
            )
            if resolution == "True":
                proven_true.add(node)
            elif resolution == "False":
                proven_false.add(node)
        else:
            # No proven_true parents supply any support at all.
            # If every contributing predecessor is proven_false with a positive
            # link, the chain is uniformly blocked and the node inherits M_F.
            if any(
                parent in proven_false and G[parent][node].get("sign") == "+"
                for parent in g_trimmed.predecessors(node)
            ):
                proven_false.add(node)

    if y in proven_true:
        return "True"
    if y in proven_false:
        return "False"
    return "Skeptical/Unknown"


if __name__ == "__main__":
    def run_case(title, graph, source, target, expected):
        result = query_defeasible_graph(graph, source, target)
        print(f"{title}")
        print(f"Query: {source} -> {target}")
        print(f"Expected: {expected}")
        print(f"Result:   {result}")
        print(f"PASS:     {result == expected}")
        print("-" * 40)

    # Case 1: Classic exception structure
    # Tweety is a penguin, penguins are birds, birds fly, but penguins do not fly.
    g1 = nx.DiGraph()
    g1.add_nodes_from(["Tweety", "Penguin", "Bird", "Fly"])
    g1.add_edge("Tweety", "Penguin", sign="+")
    g1.add_edge("Penguin", "Bird", sign="+")
    g1.add_edge("Bird", "Fly", sign="+")
    g1.add_edge("Penguin", "Fly", sign="-")
    run_case(
        "Case 1: Penguin exception blocks inherited flying",
        g1,
        "Tweety",
        "Fly",
        "False",
    )

    # Case 2: Pure positive inheritance chain
    # Nemo is a fish, fish swim.
    g2 = nx.DiGraph()
    g2.add_nodes_from(["Nemo", "Fish", "Swim"])
    g2.add_edge("Nemo", "Fish", sign="+")
    g2.add_edge("Fish", "Swim", sign="+")
    run_case(
        "Case 2: Positive chain supports the conclusion",
        g2,
        "Nemo",
        "Swim",
        "True",
    )

    # Case 3: Conflicting defaults from equally supported parents
    # Alex is both an indoor_pet and a farm_animal.
    # Indoor pets are quiet; farm animals are not quiet.
    g3 = nx.DiGraph()
    g3.add_nodes_from(["Alex", "IndoorPet", "FarmAnimal", "Quiet"])
    g3.add_edge("Alex", "IndoorPet", sign="+")
    g3.add_edge("Alex", "FarmAnimal", sign="+")
    g3.add_edge("IndoorPet", "Quiet", sign="+")
    g3.add_edge("FarmAnimal", "Quiet", sign="-")
    run_case(
        "Case 3: Symmetric conflict yields skeptical/unknown",
        g3,
        "Alex",
        "Quiet",
        "Skeptical/Unknown",
    )

    # Case 4: Unreachable target
    # There is no relevant path from Widget to Warm.
    g4 = nx.DiGraph()
    g4.add_nodes_from(["Widget", "Metal", "Heavy", "Warm"])
    g4.add_edge("Widget", "Metal", sign="+")
    g4.add_edge("Metal", "Heavy", sign="+")
    run_case(
        "Case 4: Unreachable target stays unknown",
        g4,
        "Widget",
        "Warm",
        "Skeptical/Unknown",
    )

    # Case 5: Direct negative fact overrides longer positive support
    # Robo is a bird, birds move, moving things are noisy,
    # but Robo is directly known to not be noisy.
    g5 = nx.DiGraph()
    g5.add_nodes_from(["Robo", "Bird", "Move", "Noisy"])
    g5.add_edge("Robo", "Bird", sign="+")
    g5.add_edge("Bird", "Move", sign="+")
    g5.add_edge("Move", "Noisy", sign="+")
    g5.add_edge("Robo", "Noisy", sign="-")
    run_case(
        "Case 5: Direct negative evidence defeats inherited support",
        g5,
        "Robo",
        "Noisy",
        "False",
    )
