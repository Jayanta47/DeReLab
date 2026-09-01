from enum import Enum
from dataclasses import dataclass
import networkx as nx
from graph_generation.base import InheritanceGraphBuilder



class BeliefState(Enum):
    SUPPORTED = "strengthened"
    WEAKLY_SUPPORTED = "weakened"
    CONTRADICTED = "contradicted"
    UNKNOWN = "no_change"

