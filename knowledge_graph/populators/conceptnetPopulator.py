
from knowledge_graph.populators.basePopulator import BaseGraph
from knowledge_graph.db_fetch.conceptnet.conceptnet import Conceptnet_db


class ConceptNetGraph(BaseGraph):
    """Builds and appends ConceptNet taxonomic chains into the graph using a local SQLite DB."""
    
    def __init__(self, root_word, json_filepath="taxonomy_graph.json", max_depth=2):
        super().__init__(root_word, json_filepath)
        self.max_depth = max_depth
        self._visited_concepts = set()
        self.db = Conceptnet_db()

    def build(self):
        """Initializes the DB, finds root, builds the graph via BFS, and saves to JSON."""
        print(f"\nResolving '{self.root_word}' on ConceptNet (Local DB)...")

        root_concept = self.root_word.strip().lower().replace(" ", "_")
        root_id = f"cnet:{root_concept}"
        
        # Add Root Node
        self._add_node(
            kb_id=root_id, 
            kb_source="conceptnet", 
            canonical_label=root_concept.replace("_", " "), 
            aliases=[]
        )
        
        # Start Breadth-First Search
        self._populate_children_bfs([root_concept], current_depth=0)
        
        # Save output
        self._save_graph()

    def _populate_children_bfs(self, concept_list, current_depth):
        """Uses Breadth-First Search to traverse subclasses and properties, limited by max_depth."""
        if current_depth >= self.max_depth or not concept_list:
            return
            
        next_level_concepts = []
        
        for concept in concept_list:
            if concept in self._visited_concepts:
                continue
            self._visited_concepts.add(concept)
            
            parent_id = f"cnet:{concept}"
            
            print(f"Querying ConceptNet subclasses for '{concept}' (Depth {current_depth+1}/{self.max_depth})...")
            subclasses = self.db.get_subclasses(concept)
            
            for sub in subclasses:
                child_id = f"cnet:{sub}"
                child_label = sub.replace("_", " ")
                
                # Add child node
                self._add_node(child_id, "conceptnet", child_label, aliases=[])
                
                # Add Subclass edge
                self._add_edge(
                    source_id=parent_id,
                    target_id=child_id,
                    relation_type="subclass",
                    context="ConceptNet IsA relation"
                )
                
                # Queue for next depth level
                next_level_concepts.append(sub)

            properties = self.db.get_properties(concept)
            
            for prop in properties:
                prop_id = f"cnet:{prop}"
                prop_label = prop.replace("_", " ")
                
                # Add property node
                self._add_node(prop_id, "conceptnet", prop_label, aliases=[])
                
                # Add Property edge
                self._add_edge(
                    source_id=parent_id,
                    target_id=prop_id,
                    relation_type="property",
                    context="ConceptNet HasProperty relation"
                )
                # we don't want to find subclasses of an adjective.

        # Recurse to next level
        self._populate_children_bfs(next_level_concepts, current_depth + 1)


