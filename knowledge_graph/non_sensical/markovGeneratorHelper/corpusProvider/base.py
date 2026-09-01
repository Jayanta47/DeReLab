from abc import ABC, abstractmethod
from typing import List, Dict, Optional

class CorpusProvider(ABC):
    """Abstract Base Class for any system that can provide a training corpus."""
    @abstractmethod
    def get_corpus(self) -> List[str]:
        pass