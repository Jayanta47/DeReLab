import random
import json
import re
from typing import List
from wuggy import WuggyGenerator



class NeutralEntityGenerator:
    
    def __init__(self, language: str = "orthographic_english",json_filepath: str = "non_sensical_refs.json"):
        # 1. Initialization and Plugin Loading
        self.generator = WuggyGenerator()
        self.language = language
        self._load_language()
        data = {}
        if json_filepath:
            try:
                with open(json_filepath, 'r') as f:
                    data = json.load(f)
            except FileNotFoundError:
                print(f"Warning: {json_filepath} not found. Using defaults.")
        self.adjective_references = data.get("adjectives", ["glim", "silept", "tragen", "florid"])
        self.noun_references = data.get("nouns", ["dragon", "tiger", "eagle", "spider", "whale", "badger"])

    def _load_language(self):
        """Loads the phonetic/orthographic rules for the target language safely."""
        try:
            self.generator.load(self.language)
            print(f"[*] Successfully loaded Wuggy language plugin: {self.language}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load language '{self.language}'. "
                f"Supported official plugins: {self.generator.supported_official_language_plugin_names}. "
                f"Original error: {e}"
            )

    def generate_from_reference(self, reference_word: str, limit: int = 15) -> List[str]:
        """
        Core generation function: Generates pseudowords that structurally 
        match a real reference word but carry zero meaning.
        """
        candidates =[]
        try:

            results = self.generator.generate_classic([reference_word])
            
            for item in results:

                pseudo = item.get("pseudoword", item.get("pseudo_word", str(item)))
                
                if pseudo and pseudo.lower() != reference_word.lower():
                    candidates.append(pseudo.capitalize())
                    
                if len(candidates) >= limit:
                    break
                    
        except Exception as e:
            print(f"[!] Warning: Wuggy failed to generate for '{reference_word}'. Error: {e}")
            
        return candidates

    def filter_entities(self, 
                        words: List[str], 
                        min_len: int = 4, 
                        max_len: int = 10, 
                        no_repeating_chars: bool = True) -> List[str]:
        """
        Sanitization: Applies constraints to ensure words are aesthetically 
        pleasing and optimal for LLM tokenization.
        """
        filtered =[]
        for w in words:

            if not (min_len <= len(w) <= max_len):
                continue
                

            if no_repeating_chars and re.search(r'(.)\1{2,}', w):
                continue
                
            if not w.isalpha():
                continue
                
            filtered.append(w)
            
        return list(set(filtered)) 

    def build_compound_entity(self, separator: str = "-") -> str:
        """
        Compound Generation: Merges a pseudoword derived from an adjective 
        with one derived from a noun (e.g., 'Silept-Tragen').
        """
        adj_ref = random.choice(self.adjective_references)
        noun_ref = random.choice(self.noun_references)
        
        # Generate raw candidates
        adj_candidates = self.generate_from_reference(adj_ref, limit=20)
        noun_candidates = self.generate_from_reference(noun_ref, limit=20)
        
        # Filter candidates for aesthetics
        adj_filtered = self.filter_entities(adj_candidates)
        noun_filtered = self.filter_entities(noun_candidates)
        
        # Apply fallbacks in case Wuggy yields empty results based on strict constraints
        final_adj = random.choice(adj_filtered) if adj_filtered else self._procedural_fallback(adj_ref)
        final_noun = random.choice(noun_filtered) if noun_filtered else self._procedural_fallback(noun_ref)
        
        return f"{final_adj}{separator}{final_noun}"

    def generate_batch(self, count: int, compound: bool = False) -> List[str]:
        """
        Batch Processing: Generates a large dataset of unique entities.
        """
        entities = set()
        attempts = 0
        max_attempts = count * 10 
        
        while len(entities) < count and attempts < max_attempts:
            if compound:
                entities.add(self.build_compound_entity())
            else:
                ref = random.choice(self.noun_references)
                candidates = self.generate_from_reference(ref, limit=10)
                filtered = self.filter_entities(candidates)
                if filtered:
                    entities.add(random.choice(filtered))
            attempts += 1
            
        return list(entities)
        
    def _procedural_fallback(self, seed_word: str) -> str:
        """
        Fallback mechanism: A simple vowel/consonant replacer if 
        Wuggy fails to generate a valid word for a highly specific reference.
        """
        vowels = "aeiou"
        consonants = "bcdfghjklmnpqrstvwxyz"
        word = "".join(
            random.choice(vowels) if char in vowels else random.choice(consonants)
            for char in seed_word
        )
        return word.capitalize()
    


# call generate_batch with compound=True to get compound entities
if __name__ == "__main__":
    generator = NeutralEntityGenerator()
    compound_entities = generator.generate_batch(count=20, compound=True)
    for entity in compound_entities:
        print(entity)