import networkx as nx
from pyvis.network import Network
from collections import defaultdict


class InteractiveGraphVisualizer:
    """
    Generates an interactive HTML visualization of the reasoning graph.

    Default topology  — fixed column layout:
      far-left: irrelevant_property nodes (red boxes)
      centre:   entity / object nodes (blue circles)
      right:    property chain nodes (red rounded boxes), top-to-bottom in
                chain order, connected by solid red arrows
      defeasible edges rendered as curved black arrows

    Tree / linear inheritance — hierarchical top-down layout (unchanged).
    """

    def __init__(self, output_file="reasoning_graph.html"):
        self.output_file = output_file

    # ------------------------------------------------------------------
    # Topology detection
    # ------------------------------------------------------------------

    def _topology(self, graph) -> str:
        return graph.graph.get("topology", "default")

    # ------------------------------------------------------------------
    # Default topology: chain order
    # ------------------------------------------------------------------

    def _chain_order(self, graph) -> list:
        """Return property nodes sorted from chain root to leaf."""
        prop_nodes = [n for n, d in graph.nodes(data=True) if d.get("type") == "property"]
        implies_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get("type") == "implies"]
        targets = {v for _, v in implies_edges}
        roots = [n for n in prop_nodes if n not in targets]

        order, visited = [], set()
        current = roots[0] if roots else (prop_nodes[0] if prop_nodes else None)
        while current is not None and current not in visited:
            order.append(current)
            visited.add(current)
            nxt = [v for u, v in implies_edges if u == current and v not in visited]
            current = nxt[0] if nxt else None

        for n in prop_nodes:
            if n not in visited:
                order.append(n)
        return order

    # ------------------------------------------------------------------
    # Default topology: fixed positions
    # ------------------------------------------------------------------

    def _default_positions(self, graph) -> dict:
        chain = self._chain_order(graph)
        entities = [n for n, d in graph.nodes(data=True) if d.get("type") == "entity"]
        irrelevants = [n for n, d in graph.nodes(data=True) if d.get("type") == "irrelevant_property"]

        X_IRR, X_ENT, X_CHAIN = -450, 0, 450
        V_STEP = 160

        pos = {}

        # Chain: evenly spaced, top to bottom
        for i, n in enumerate(chain):
            pos[n] = (X_CHAIN, i * V_STEP)

        # Entities: spread across the chain's vertical span
        chain_span = max((len(chain) - 1) * V_STEP, 1)
        for i, n in enumerate(entities):
            y = (i * chain_span / max(len(entities) - 1, 1)) if len(entities) > 1 else chain_span / 2
            pos[n] = (X_ENT, y)

        # Irrelevants: align with the entity they belong to
        entity_set = set(entities)
        owner_of: dict = {}
        for u, v, d in graph.edges(data=True):
            if d.get("type") == "has_attribute" and u in entity_set:
                owner_of[v] = u

        owner_children: dict = defaultdict(list)
        unowned = []
        for irr in irrelevants:
            if irr in owner_of:
                owner_children[owner_of[irr]].append(irr)
            else:
                unowned.append(irr)

        for owner, children in owner_children.items():
            oy = pos.get(owner, (0, 0))[1]
            for j, irr in enumerate(children):
                offset = (j - (len(children) - 1) / 2) * 100
                pos[irr] = (X_IRR, oy + offset)

        for j, irr in enumerate(unowned):
            pos[irr] = (X_IRR, j * V_STEP)

        return pos

    # ------------------------------------------------------------------
    # Node styling
    # ------------------------------------------------------------------

    def _node_style_default(self, node_id, node_data) -> dict:
        ntype = node_data.get("type", "")
        label = node_data.get("label", str(node_id))

        if ntype == "property":
            return {
                "label": label,
                "title": f"ID: {node_id}\nType: chain property",
                "shape": "box",
                "color": {"background": "#ff6b6b", "border": "#c0392b",
                          "highlight": {"background": "#ff4444", "border": "#922b21"}},
                "font": {"color": "#ffffff", "size": 14, "bold": True},
                "borderWidth": 2,
                "physics": False,
            }
        if ntype == "entity":
            return {
                "label": label,
                "title": f"ID: {node_id}\nType: object",
                "shape": "circle",
                "color": {"background": "#5dade2", "border": "#1a5276",
                          "highlight": {"background": "#3498db", "border": "#1a5276"}},
                "font": {"color": "#ffffff", "size": 13, "bold": True},
                "borderWidth": 2,
                "size": 30,
                "physics": False,
            }
        if ntype == "irrelevant_property":
            return {
                "label": label,
                "title": f"ID: {node_id}\nType: irrelevant attribute",
                "shape": "box",
                "color": {"background": "#fadbd8", "border": "#e74c3c",
                          "highlight": {"background": "#f1948a", "border": "#c0392b"}},
                "font": {"color": "#922b21", "size": 13},
                "borderWidth": 2,
                "physics": False,
            }
        return {
            "label": label,
            "title": f"ID: {node_id}\nType: {ntype}",
            "shape": "box",
            "color": "#e0e0e0",
            "physics": False,
        }

    def _node_style_inheritance(self, node_id, node_data, graph) -> dict:
        ntype = node_data.get("type", "default")
        color_map = {
            "entity": "#99ccff",
            "exception": "#ff9999",
            "property": "#99ff99",
            "class": "#f2f2f2",
        }
        label = node_data.get("label", str(node_id))
        hover = f"ID: {node_id}\nType: {ntype}"
        if "metadata" in node_data:
            hover += f"\nValue: {node_data['metadata'].get('value', '')}"

        style = {
            "label": label,
            "title": hover,
            "color": color_map.get(ntype, "#e0e0e0"),
            "shape": "box",
        }
        if self._topology(graph) == "tree_inheritance":
            if ntype == "class":
                style["level"] = node_data.get("layer", 0)
                style["mass"] = 2
            elif ntype == "property":
                max_layer = max(
                    (d.get("layer", 0) for _, d in graph.nodes(data=True) if d.get("type") == "class"),
                    default=0,
                )
                style["level"] = max_layer + 1
                style["shape"] = "ellipse"
                style["color"] = "#c8e6c9"
                style["size"] = 30
            else:
                style["level"] = node_data.get("layer", 0)
        return style

    # ------------------------------------------------------------------
    # Edge styling
    # ------------------------------------------------------------------

    def _edge_style_default(self, edge_data) -> dict:
        etype = edge_data.get("type", "")

        if etype == "implies":
            return {
                "color": "#c0392b",
                "width": 2.5,
                "arrows": {"to": {"enabled": True, "scaleFactor": 1.2}},
                "smooth": {"enabled": False},
                "label": "",
                "title": "implies",
                "physics": False,
            }

        if etype in ("is_a", "hypothesis_edge"):
            return {
                "color": "#1a6fa8",
                "width": 2,
                "arrows": {"to": {"enabled": True, "scaleFactor": 1.1}},
                "smooth": {"enabled": True, "type": "curvedCCW", "roundness": 0.2},
                "label": "",
                "title": etype,
                "physics": False,
            }

        if etype == "has_attribute":
            return {
                "color": "#1a6fa8",
                "width": 1.5,
                "arrows": {"to": {"enabled": True}},
                "smooth": {"enabled": True, "type": "curvedCW", "roundness": 0.2},
                "label": "",
                "title": "has attribute",
                "physics": False,
            }

        if etype in ("defeasible_has_property", "defeasible_lacks_property"):
            sign = "+" if etype == "defeasible_has_property" else "−"
            return {
                "color": "#1c1c1c",
                "width": 2,
                "dashes": [6, 4],
                "arrows": {"to": {"enabled": True, "scaleFactor": 1.1}},
                "smooth": {"enabled": True, "type": "curvedCW", "roundness": 0.45},
                "label": sign,
                "font": {"size": 14, "color": "#1c1c1c", "bold": True},
                "title": etype,
                "physics": False,
            }

        return {
            "color": "gray",
            "label": etype,
            "title": etype,
            "physics": False,
        }

    def _edge_style_inheritance(self, edge_data, is_tree) -> dict:
        etype = edge_data.get("type", "unknown")
        style = {
            "title": etype,
            "label": etype,
            "color": "gray",
            "font": {"size": 10, "align": "middle"},
        }
        if etype == "hypothesis_edge":
            style["color"] = "#d32f2f"
        elif etype == "defeasible_has_property":
            style["color"] = "#388e3c"

        if not is_tree:
            return style

        if etype == "inheritance":
            style.update({"color": "#455a64", "width": 2.5})
        elif etype == "has_property":
            style.update({"color": "#66bb6a", "dashes": True, "physics": False,
                          "label": "", "smooth": {"enabled": True, "type": "curvedCW", "roundness": 0.2}})
        elif etype in {"defeasible_has_property", "defeasible_lacks_property"}:
            style.update({"dashes": True, "physics": False, "label": "",
                          "smooth": {"enabled": True, "type": "curvedCW", "roundness": 0.25},
                          "color": "#ef6c00" if etype == "defeasible_lacks_property" else "#352dd7"})
        elif etype == "hypothesis_edge":
            style.update({"dashes": True, "physics": False, "label": "", "width": 1.5,
                          "smooth": {"enabled": True, "type": "curvedCW", "roundness": 0.35}})
        return style

    # ------------------------------------------------------------------
    # Layout configuration
    # ------------------------------------------------------------------

    def _options_default(self, net):
        net.set_options("""
        var options = {
          "physics": { "enabled": false },
          "edges": { "smooth": false },
          "interaction": {
            "dragNodes": true,
            "dragView": true,
            "zoomView": true,
            "hover": true
          }
        }
        """)

    def _options_tree(self, net):
        net.set_options("""
        var options = {
          "layout": {
            "hierarchical": {
              "enabled": true,
              "direction": "UD",
              "sortMethod": "directed",
              "levelSeparation": 150,
              "nodeSpacing": 170,
              "treeSpacing": 220,
              "blockShifting": true,
              "edgeMinimization": true,
              "parentCentralization": true
            }
          },
          "physics": { "enabled": false },
          "edges": { "smooth": false },
          "interaction": {
            "dragNodes": true,
            "dragView": true,
            "zoomView": true,
            "hover": true
          }
        }
        """)

    def _options_physics(self, net):
        net.set_options("""
        var options = {
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -5000,
              "centralGravity": 0.1,
              "springLength": 180,
              "springConstant": 0.05
            },
            "minVelocity": 0.75
          }
        }
        """)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def visualize(self, graph, title="Defeasible Reasoning Graph"):
        topo = self._topology(graph)
        is_tree = topo == "tree_inheritance"
        is_default = topo == "default"

        net = Network(height="800px", width="100%", directed=True, heading=title)

        if is_default:
            positions = self._default_positions(graph)

            for node_id, node_data in graph.nodes(data=True):
                style = self._node_style_default(node_id, node_data)
                x, y = positions.get(node_id, (0, 0))
                net.add_node(node_id, x=x, y=y, **style)

            for u, v, edge_data in graph.edges(data=True):
                net.add_edge(u, v, **self._edge_style_default(edge_data))

            self._options_default(net)

        else:
            for node_id, node_data in graph.nodes(data=True):
                net.add_node(
                    node_id,
                    **self._node_style_inheritance(node_id, node_data, graph)
                )
            for u, v, edge_data in graph.edges(data=True):
                net.add_edge(u, v, **self._edge_style_inheritance(edge_data, is_tree))

            if is_tree:
                self._options_tree(net)
            else:
                self._options_physics(net)

        net.show(self.output_file, notebook=False)
        print(f"Interactive graph saved to: '{self.output_file}'")
