import re
import nltk

# Download necessary NLTK datasets
nltk.download('words', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True) 

from nltk.corpus import words, wordnet

class EntityFilter:
    """Handles the sanitization and phonetic validation of generated entities."""
    def __init__(self, min_len: int = 4, max_len: int = 10, no_repeating_chars: bool = True, max_consonants: int = 3):
        self.min_len = min_len
        self.max_len = max_len
        self.no_repeating_chars = no_repeating_chars
        self.max_consonants = max_consonants
        
        # Base dictionary (fast, exact matches)
        self.english_words = set(w.lower() for w in words.words())

    def is_valid(self, word: str) -> bool:
        # Check length
        if not word or not (self.min_len <= len(word) <= self.max_len):
            return False
            
        # Check if characters are purely alphabetic
        if not word.isalpha():
            return False
            
        # Check for repeating characters (e.g., 'aaa')
        if self.no_repeating_chars and re.search(r'(.)\1{2,}', word):
            return False
            
        # Check for unpronounceable consonant strings
        if re.search(f'[^aeiouyAEIOUY]{{{self.max_consonants + 1},}}', word):
            return False
            
        # Check if it is an actual English word
        if self._is_english_word(word):
            return False
            
        return True

    def _is_english_word(self, word: str) -> bool:
        word_lower = word.lower()
        
        # 1. Fast check against the basic dictionary
        if word_lower in self.english_words:
            return True
            
        # 2. Check WordNet (This catches plurals, adverbs, and verb conjugations)
        # e.g., 'jumping', 'swords', 'stranger'
        if wordnet.synsets(word_lower):
            return True
            
        return False