from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Dict, Any

class NodeType(Enum):
    ENTITY = "entity"
    PROPERTY = "property"

@dataclass
class GraphNode:
    id: str
    node_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        # Convert dataclass to dict
        d = asdict(self)
        # Ensure metadata copy to avoid mutation issues
        d['metadata'] = self.metadata.copy()
        return d

    @staticmethod
    def from_dict(data: Dict[str, Any]):
        node_type = data.get('node_type')
        
        # 1. Reconstruct EntityNode
        if node_type == NodeType.ENTITY.value:
            meta = data.get('metadata', {}).copy()
            # Extract explicit fields if present in top-level or metadata
            depth = data.get('depth', meta.pop('depth', 0))
            child_count = data.get('child_count', meta.pop('child_count', 0))
            
            return EntityNode(
                name=data['id'],
                depth=depth,
                child_count=child_count,
                **meta
            )
        
        # 2. Reconstruct PropertyNode
        elif node_type == NodeType.PROPERTY.value:
            meta = data.get('metadata', {}).copy()
            prop_type = data.get('property_type', meta.pop('property_type', ''))
            prop_value = data.get('property_value', meta.pop('property_value', ''))
            
            return PropertyNode(
                prop_type=prop_type,
                prop_value=prop_value,
                **meta
            )
        
        # 3. Fallback generic
        return GraphNode(
            id=data['id'],
            node_type=node_type,
            metadata=data.get('metadata', {})
        )

@dataclass
class EntityNode(GraphNode):
    depth: int = 0
    child_count: int = 0

    def __init__(self, name: str, depth: int = 0, child_count: int = 0, **kwargs):
        # Mirror explicit fields into metadata for persistence consistency
        kwargs['depth'] = depth
        kwargs['child_count'] = child_count
        super().__init__(id=name, node_type=NodeType.ENTITY.value, metadata=kwargs)
        self.depth = depth
        self.child_count = child_count

@dataclass
class PropertyNode(GraphNode):
    property_type: str = ""
    property_value: str = ""

    def __init__(self, prop_type: str, prop_value: str, **kwargs):
        unique_id = f"{prop_type}:{prop_value}"
        kwargs['property_type'] = prop_type
        kwargs['property_value'] = prop_value
        super().__init__(id=unique_id, node_type=NodeType.PROPERTY.value, metadata=kwargs)
        self.property_type = prop_type
        self.property_value = prop_value