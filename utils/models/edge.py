from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Dict, Any

class EdgeType(Enum):
    INHERITANCE = "is_a"
    NO_INHERITANCE = "is_not_a"
    HAS_PROPERTY = "has"
    HAS_NOT_PROPERTY = "lacks"

@dataclass
class GraphEdge:
    source: str
    target: str
    relation_type: str 
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self):
        d = asdict(self)
        d['metadata'] = self.metadata.copy()
        return d
    
    @staticmethod
    def from_dict(data: Dict[str, Any]):
        rel_type = data.get('relation_type')
        if isinstance(rel_type, EdgeType):
            rel_type = rel_type.value
            
        return GraphEdge(
            source=data['source'],
            target=data['target'],
            relation_type=rel_type,
            metadata=data.get('metadata', {})
        )

    def get_preposition(self):
        return self.metadata.get('preposition', None)