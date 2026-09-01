import requests
from typing import List, Set, Optional

class DatamuseExpander:
    """
    A comprehensive wrapper for the Datamuse API to fetch semantic, 
    orthographic, and contextual expansions of a seed word.
    Perfect for generating high-entropy seed corpora for Markov chains.
    """
    BASE_URL = "https://api.datamuse.com/words"

    def __init__(self, default_max: int = 100, only_single_words: bool = True):
        self.default_max = default_max
        self.only_single_words = only_single_words

    def _query(self, params: dict) -> List[str]:
        """Internal helper to safely make API calls and parse out clean words."""
        if 'max' not in params:
            params['max'] = self.default_max
            
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            words =[]
            for item in data:
                word = item['word']
                # Filter out multi-word phrases (e.g., "grizzly bear") if required
                if self.only_single_words and not word.isalpha():
                    continue
                words.append(word)
                
            return words
            
        except requests.exceptions.RequestException as e:
            print(f"[!] Datamuse API Error: {e}")
            return[]

    # ==========================================
    # 1. PRIMARY TARGETS (From your prompt)
    # ==========================================

    def get_similar_words(self, word: str) -> List[str]:
        """Fetches words with similar meaning (Means Like)."""
        return self._query({"ml": word})
        
    def get_synonyms(self, word: str) -> List[str]:
        """Strictly fetches WordNet synonyms."""
        return self._query({"rel_syn": word})

    def get_subclasses(self, word: str) -> List[str]:
        """Fetches hyponyms (More specific types / Subclasses). E.g., animal -> dog."""
        return self._query({"rel_gen": word})

    def get_properties(self, word: str) -> List[str]:
        """Fetches popular adjectives used to describe the word. E.g., animal -> wild."""
        return self._query({"rel_jja": word})

    # ==========================================
    # 2. HIERARCHY & PARTS (WordNet Topology)
    # ==========================================

    def get_superclasses(self, word: str) -> List[str]:
        """Fetches hypernyms (Broader categories). E.g., dog -> animal."""
        return self._query({"rel_spc": word})

    def get_components(self, word: str) -> List[str]:
        """Fetches meronyms (Parts that make up the word). E.g., car -> accelerator."""
        return self._query({"rel_com": word})

    def get_part_of(self, word: str) -> List[str]:
        """Fetches holonyms (The larger whole this word belongs to). E.g., trunk -> tree."""
        return self._query({"rel_par": word})

    # ==========================================
    # 3. SOUND & SPELLING (Orthographic)
    # ==========================================

    def get_sounds_like(self, word: str) -> List[str]:
        """Fetches phonetically similar words."""
        return self._query({"sl": word})

    def get_rhymes(self, word: str, approximate: bool = False) -> List[str]:
        """Fetches perfect or approximate rhymes."""
        param = "rel_nry" if approximate else "rel_rhy"
        return self._query({param: word})

    def get_spelled_like(self, pattern: str) -> List[str]:
        """Fetches words by spelling pattern using wildcards (* and ?). E.g., 'ani*'"""
        return self._query({"sp": pattern})

    # ==========================================
    # 4. CONTEXT & COLLOCATION (Google Ngrams)
    # ==========================================

    def get_associated_triggers(self, word: str) -> List[str]:
        """Fetches words statistically associated with the word in text."""
        return self._query({"rel_trg": word})

    def get_frequent_followers(self, word: str) -> List[str]:
        """Fetches words that frequently appear immediately AFTER the word."""
        return self._query({"rel_bga": word})

    def get_frequent_predecessors(self, word: str) -> List[str]:
        """Fetches words that frequently appear immediately BEFORE the word."""
        return self._query({"rel_bgb": word})

    def get_antonyms(self, word: str) -> List[str]:
        """Fetches opposite words (Antonyms)."""
        return self._query({"rel_ant": word})

    # ==========================================
    # 5. MEGA-CORPUS AGGREGATOR
    # ==========================================

    def build_mega_corpus(self, word: str) -> List[str]:
        """
        Runs multiple API queries to build a massive, deduplicated corpus 
        around a single seed concept. Highly recommended for Markov Model training.
        """
        print(f"[*] Building mega-corpus for '{word}'...")
        corpus_set: Set[str] = set()

        # Execute queries and combine results
        corpus_set.update(self.get_similar_words(word))
        corpus_set.update(self.get_subclasses(word))
        # corpus_set.update(self.get_superclasses(word))
        # corpus_set.update(self.get_properties(word))
        corpus_set.update(self.get_components(word))
        corpus_set.update(self.get_sounds_like(word))
        corpus_set.update(self.get_associated_triggers(word))

        # Add the seed word itself
        corpus_set.add(word)

        final_corpus = list(corpus_set)
        print(f"[+] Successfully aggregated {len(final_corpus)} unique words.")
        return final_corpus
    

# write a main
if __name__ == "__main__":
    expander = DatamuseExpander()
    seed_word = "animal"
    mega_corpus = expander.build_mega_corpus(seed_word)
    print(f"Mega Corpus for '{seed_word}': {mega_corpus}")



