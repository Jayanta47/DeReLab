import os
import json
import time
import requests
from abc import ABC, abstractmethod
import nltk
from nltk.corpus import wordnet as wn

# need to be inside a folder
class BaseGraph(ABC):
    """Abstract base class for building taxonomic knowledge graphs."""
    
    def __init__(self, root_word, json_filepath="knowledge_graph.json"):
        self.root_word = root_word
        self.json_filepath = json_filepath
        
        self.nodes = {}
        self.edges = []
        self.edges_set = set() 
        
        self._load_graph()

    def _load_graph(self):
        """Loads the graph from the JSON file if it exists."""
        if os.path.exists(self.json_filepath):
            with open(self.json_filepath, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    self.nodes = data.get("nodes", {})
                    self.edges = data.get("edges", [])
                    self.edges_set = {(e["source"], e["target"]) for e in self.edges}
                    print(f"Loaded existing graph: {len(self.nodes)} nodes, {len(self.edges)} edges from {self.json_filepath}.")
                except json.JSONDecodeError:
                    print("Warning: JSON file is corrupted or empty. Starting fresh.")
        else:
            print(f"No existing graph found at '{self.json_filepath}'. Starting fresh.")

    def _save_graph(self):
        """Saves the updated graph to the JSON file."""
        with open(self.json_filepath, 'w', encoding='utf-8') as f:
            json.dump({
                "nodes": self.nodes,
                "edges": self.edges
            }, f, indent=2, ensure_ascii=False)
        print(f"Graph successfully saved to '{self.json_filepath}'.")

    def _get_plural(self, word):
        """A simple heuristic rule-based pluralizer."""
        if not word: return ""
        word_lower = word.lower()
        if word_lower.endswith(('s', 'sh', 'ch', 'x', 'z')):
            return word + 'es'
        elif word_lower.endswith('y') and len(word) > 1 and word_lower[-2] not in 'aeiou':
            return word[:-1] + 'ies'
        else:
            return word + 's'

    def _add_node(self, kb_id, kb_source, canonical_label, aliases):
        """Adds a clean node strictly adhering to the specified schema."""
        if kb_id in self.nodes:
            return 
            
        singular = canonical_label
        plural = self._get_plural(singular)
        indef_article = "an" if singular and singular[0].lower() in "aeiou" else "a"
        
        self.nodes[kb_id] = {
            "kind": "class",
            "kb_source": kb_source,
            "kb_id": kb_id,
            "canonical_label": canonical_label,
            "surface": {
                "singular": singular,
                "plural": plural,
                "indef_article": indef_article
            },
            "semantic_type": self.root_word,
            "domain": self._get_plural(self.root_word),
            "aliases": aliases,
            "mask_label": None
        }

    def _add_edge(self, source_id, target_id, relation_type, context=""):
        """Adds an edge if it does not already exist."""
        if (source_id, target_id) not in self.edges_set:
            self.edges_set.add((source_id, target_id))
            self.edges.append({
                "source": source_id,
                "target": target_id,
                "relation": relation_type,
                "context": context
            })



    @abstractmethod
    def build(self):
        """Main execution method to be implemented by subclasses."""
        pass


