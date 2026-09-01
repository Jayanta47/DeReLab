import os
import json
import time
import requests
from abc import ABC, abstractmethod
from knowledge_graph.populators.basePopulator import BaseGraph
import nltk
from nltk.corpus import wordnet as wn

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


class WordNetGraph(BaseGraph):
    """Builds and appends WordNet taxonomic chains into the graph."""
    
    def __init__(self, root_word, json_filepath="knowledge_graph.json"):
        super().__init__(root_word, json_filepath)
        self._visited_synsets = set()

    def _infer_edge_relationship(self, parent_synset, child_synset):
        definition = child_synset.definition().lower()
        property_keywords = [
            'young', 'immature', 'baby', 'juvenile', 'adult', 
            'female', 'male', 'castrated', 
            'used for', 'kept for', 'trained', 'raised', 'having powers of'
        ]
        
        is_property = any(w in definition for w in property_keywords)
        is_lexname_shift = child_synset.lexname() != parent_synset.lexname()
        
        return "property" if (is_property or is_lexname_shift) else "subclass"

    def _process_synset(self, synset):
        lemmas = synset.lemmas()
        canonical_label = lemmas[0].name().replace('_', ' ')
        aliases = [l.name().replace('_', ' ') for l in lemmas[1:]]
        self._add_node(synset.name(), "oewn", canonical_label, aliases)

    def build(self):
        synsets = wn.synsets(self.root_word, pos=wn.NOUN)
        if not synsets:
            raise ValueError(f"Error: No entity found for '{self.root_word}' in WordNet")
        
        root_synset = synsets[0]
        self._process_synset(root_synset)
        self._visited_synsets.add(root_synset)
        
        self._populate_children(root_synset)
        self._save_graph()

    def _populate_children(self, current_synset):
        for hyponym in current_synset.hyponyms():
            self._process_synset(hyponym)
            
            relation_type = self._infer_edge_relationship(current_synset, hyponym)
            self._add_edge(
                source_id=current_synset.name(), 
                target_id=hyponym.name(), 
                relation_type=relation_type, 
                context=hyponym.definition()
            )
            
            if hyponym not in self._visited_synsets:
                self._visited_synsets.add(hyponym)
                self._populate_children(hyponym)


