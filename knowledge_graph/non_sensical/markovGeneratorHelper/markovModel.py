import random
from collections import defaultdict
from knowledge_graph.non_sensical.markovGeneratorHelper.corpusProvider.base import CorpusProvider

class MarkovModel:
    """The Machine Learning model that learns transitions from a CorpusProvider."""
    def __init__(self, order: int = 2):
        self.order = order
        self.transitions = defaultdict(list)
        self.is_trained = False

    def train(self, provider: CorpusProvider):
        corpus = provider.get_corpus()
        for word in corpus:
            padded = "^" + word.lower() + "$"
            for i in range(len(padded) - self.order):
                state = padded[i : i + self.order]
                next_char = padded[i + self.order]
                self.transitions[state].append(next_char)
        self.is_trained = True

    def generate_raw(self, seed: str = None, max_len: int = 15) -> str:
        if not self.is_trained:
            raise RuntimeError("Model must be trained before generating.")

        if seed and len(seed) >= self.order:
            current_state = seed[:self.order].lower()
            result = current_state
        else:
            starts =[state for state in self.transitions.keys() if state.startswith("^")]
            current_state = random.choice(starts) if starts else "^a"
            result = current_state.replace("^", "")

        for _ in range(max_len):
            options = self.transitions.get(current_state)
            if not options:
                break
            
            next_char = random.choice(options)
            if next_char == "$":
                break
                
            result += next_char
            current_state = result[-self.order:]
            
            if current_state not in self.transitions:
                current_state = "^" + current_state[-1] if len(current_state) == 1 else current_state[-self.order:]
                
        return result.capitalize()

