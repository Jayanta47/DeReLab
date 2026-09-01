import random
import json
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from collections import defaultdict
from knowledge_graph.non_sensical.markovGeneratorHelper.corpusProvider.base import CorpusProvider


class ReferenceCorpusProvider(CorpusProvider):
    def __init__(self, default_words: List[str] = None, json_filepath: str = None, json_key: str = None, json_field: str = None):
        self.corpus = default_words or[]

        # if the json is a single dict with a key that contains a list of words, we can directly use that list
        if json_filepath and json_key:
            try:
                with open(json_filepath, 'r') as f:
                    data = json.load(f)
                    if json_key in data:
                        raw_data = data[json_key]
                        
                        # Use the recursive flattener to extract all words
                        extracted_words = self._flatten_data(raw_data)
                        
                        # Only overwrite the defaults if we actually found words
                        if extracted_words:
                            self.corpus = extracted_words
                print(f"[+] Loaded {len(self.corpus)} words from {json_filepath} under key '{json_key}'.")
                            
            except FileNotFoundError:
                print(f"[!] Warning: {json_filepath} not found. Using defaults.")
            except Exception as e:
                print(f"[!] Error loading JSON: {e}. Using defaults.")

        # if the json is an array of dicts, and we need to extract a specific field from each dict
        elif json_filepath and json_field:
            try:
                with open(json_filepath, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and json_field in item:
                                raw_value = item[json_field]
                                extracted_words = self._flatten_data(raw_value)
                                self.corpus.extend(extracted_words)
                print(f"[+] Loaded {len(self.corpus)} words from {json_filepath} by extracting field '{json_field}'.")
            except FileNotFoundError:
                print(f"[!] Warning: {json_filepath} not found. Using defaults.")
            except Exception as e:
                print(f"[!] Error loading JSON: {e}. Using defaults.")

    def _flatten_data(self, item) -> List[str]:
        """
        Recursively digs through dictionaries, lists, and nested objects 
        to extract all string values into a single flat list.
        """
        result =[]
        
        if isinstance(item, dict):
            # If it's a dict (e.g., "adjectives": {"flimsy": [...]}), dig into the values
            for value in item.values():
                result.extend(self._flatten_data(value))
                
        elif isinstance(item, list):
            # If it's a list, dig into each item
            for value in item:
                result.extend(self._flatten_data(value))
                
        elif isinstance(item, str):
            # Base case:found a string. 
            # strip whitespace and split by space just in case there are phrases (e.g., "dark velvet")

            words = item.strip().split()
            for word in words:
                clean_word = ''.join(char for char in word if char.isalpha())
                if clean_word:
                    result.append(clean_word)
                    
        return result

    def get_corpus(self) -> List[str]:
        return self.corpus