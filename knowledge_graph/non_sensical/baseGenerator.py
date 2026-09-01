from abc import ABC, abstractmethod

class EntityGenerator(ABC):
    """Abstract Base Class for generating an entity."""
    @abstractmethod
    def generate(self) -> str:
        pass