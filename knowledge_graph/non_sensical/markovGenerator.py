import random
from typing import List, Dict, Optional

import json
from knowledge_graph.non_sensical.markovGeneratorHelper.entityFilter import EntityFilter
from knowledge_graph.non_sensical.markovGeneratorHelper.markovModel import MarkovModel
from knowledge_graph.non_sensical.markovGeneratorHelper.corpusProvider.base import CorpusProvider
from knowledge_graph.non_sensical.baseGenerator import EntityGenerator

class MarkovEntityGenerator(EntityGenerator):
    """Generates single entities using a trained MarkovModel and EntityFilter."""
    def __init__(self, model: MarkovModel, filter_rules: EntityFilter, seed_provider: Optional[CorpusProvider] = None):
        self.model = model
        self.filter = filter_rules
        self.seed_corpus = seed_provider.get_corpus() if seed_provider else[]

    def generate(self) -> str:
        attempts = 0
        while attempts < 100:
            attempts += 1
            
            # 50% chance to seed the generation with a prefix from the original corpus (if provided)
            seed = None
            if self.seed_corpus and random.random() > 0.5:
                ref_word = random.choice(self.seed_corpus)
                seed = "^" + ref_word[:2].lower() if len(ref_word) >= 2 else None

            candidate = self.model.generate_raw(seed=seed, max_len=self.filter.max_len)
            
            
            if self.filter.is_valid(candidate):
                return candidate
                
        return "Fallback" 

class CompoundEntityGenerator(EntityGenerator):
    """Combines two abstract generators to create compound entities (e.g. Adj-Noun)."""
    def __init__(self, left_generator: EntityGenerator, right_generator: EntityGenerator, separator: str = "-"):
        self.left_generator = left_generator
        self.right_generator = right_generator
        self.separator = separator

    def generate(self) -> str:
        left = self.left_generator.generate()
        right = self.right_generator.generate()
        return f"{left}{self.separator}{right}"
