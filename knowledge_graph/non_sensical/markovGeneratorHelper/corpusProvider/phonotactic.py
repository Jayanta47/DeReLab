import random
import json
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from collections import defaultdict
from knowledge_graph.non_sensical.markovGeneratorHelper.corpusProvider.base import CorpusProvider

class PhonotacticCorpusProvider(CorpusProvider):
    """Pathway 2: Generates a completely synthetic zero-bias corpus using syllable rules."""
    def __init__(self, corpus_size: int = 2000):
        self.corpus_size = corpus_size
        self.onsets =["b", "c", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z", 
                       "br", "cr", "dr", "fr", "gr", "pr", "tr", "ch", "sh", "th"]
        self.nuclei =["a", "e", "i", "o", "u", "ou", "ie", "ea", "ae", "oo"]
        self.codas =["", "", "", "b", "d", "g", "k", "l", "m", "n", "p", "r", "s", "t", "x", "th", "nt", "st"]

    def _generate_syllable(self) -> str:
        return random.choice(self.onsets) + random.choice(self.nuclei) + random.choice(self.codas)

    def get_corpus(self) -> List[str]:
        corpus = []
        for _ in range(self.corpus_size):
            syllables = random.choices([2, 3], weights=[0.8, 0.2])[0]
            word = "".join(self._generate_syllable() for _ in range(syllables))
            corpus.append(word)
        return corpus