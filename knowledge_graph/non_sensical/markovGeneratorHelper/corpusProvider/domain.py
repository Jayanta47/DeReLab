import requests
from abc import ABC, abstractmethod
from typing import List, Set, Optional
from knowledge_graph.db_fetch.datamuse import DatamuseExpander
from knowledge_graph.non_sensical.markovGeneratorHelper.corpusProvider.base import CorpusProvider
# ==========================================
class RootwordCorpusProvider(CorpusProvider):
    """
    Acts as an Adapter, using DatamuseExpander to fetch a seed corpus 
    and satisfy the CorpusProvider interface for the Markov model.
    """
    def __init__(self, seed_word: str, expander: Optional[DatamuseExpander] = None, max_results_per_query: int = 100):
        self.seed_word = seed_word.lower().strip()
        
        # If an expander isn't injected, instantiate a default one
        self.expander = expander or DatamuseExpander(
            default_max=max_results_per_query, 
            only_single_words=True
        )
        
        # Build and cache the corpus immediately upon initialization
        self._corpus = self.expander.build_mega_corpus(self.seed_word)
        print(f"[+] RootwordCorpusProvider ready: {len(self._corpus)} words cached.")

    def get_corpus(self) -> List[str]:
        """Returns the pre-fetched cached corpus."""
        return self._corpus