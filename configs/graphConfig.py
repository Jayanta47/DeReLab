from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class TopologyType(Enum):
    DEFAULT = "default"
    LINEAR_INHERITANCE = "linear_inheritance"
    TREE_INHERITANCE = "tree_inheritance"


@dataclass
class DefaultGraphConfig:
    """
    Configuration for the Default (defeasible chain) topology.

    Graph structure
    ---------------
    chain_length            : Number of property nodes in the linear default chain
                              P_root → P_1 → … → P_n (hypothesis target).
                              Larger values make reasoning harder.
    max_branching_factor    : Maximum number of branches at any chain node.
                              Keep at 1 for a purely linear chain.
    branch_probability      : Probability that a chain split occurs at each node.
                              Only meaningful when max_branching_factor > 1.
    num_objects             : Number of entity (object) nodes that instantiate
                              P_root via an is_a edge.
    num_irrelevant_attributes: Number of irrelevant/noise attribute nodes attached
                              to entity nodes via has_attribute edges.  These feed
                              Type-2 (distractor) questions.
    num_hypothesis_subjects : How many of the num_objects entities receive a
                              hypothesis edge pointing to P_n (the target).
                              None means all objects get one.

    Source / Type-4 settings
    ------------------------
    num_sources_range       : (min, max) number of named sources included in
                              the conversation pretext for source-priority
                              (Type-4) questions.  Drawn uniformly at generation
                              time.  Minimum meaningful value is 2.
    type4_scenarios         : Ordered list of Type-4 scenario types to cycle
                              through per subject.  Supported values:
                                "contradiction" — B denies a property; A reinstates.
                                "bypass"        — B denies a rule edge; A introduces
                                                  a skip rule that routes around it.
                              None defaults to ["contradiction", "bypass"].
    type4_b_first_prob      : Probability that the lower-priority source B speaks
                              first in a Type-4 episode.  When B speaks first the
                              model must update its belief after A's later claim.
    type4_enabled           : When True a disjoint subset of objects is reserved
                              exclusively for Type-4 episodes and never appears
                              again in Type-1/2 sections, preventing carry-forward
                              context contamination across hypothesis sections.
                              Automatically disabled when num_objects < 2.
    type4_object_fraction   : Fraction of objects allocated to Type 4 (rounded to
                              nearest int, minimum 1 if type4_enabled is True).
                              E.g. 0.5 with 4 objects → 2 for Type-4, 2 for T1/2.

    Reveal-order randomisation
    --------------------------
    reveal_order_random_prob: Probability (per object section) that the defeasible
                              edge reveals for that section are shown in a RANDOM
                              order rather than the original serial order.
                              At 0.5 (default), half the sections are serial and
                              half are randomised.
                              - Serial mode  : edges revealed low→high serial;
                                resolver uses serial_cutoff (standard approach).
                              - Random mode  : T1 edges shuffled into a random
                                order; resolver is called with serial_cutoff=0
                                and an explicit revealed-edge set so that only the
                                graph backbone + edges disclosed so far are visible
                                (partial-graph / PMPM-style reasoning).

    Conversation length caps
    ------------------------
    max_hypothesis_sections : Hard cap on the total number of hypothesis sections
                              (subjects) rendered across Phase 1 + Phase 2.
                              None = no cap.
    max_updates_per_section : Maximum Type-1/2/3 update turns rendered per subject
                              section.  Earlier-serial records are kept.  None = no cap.
    max_total_turns         : Safety cap on the total number of conversation turns.
                              Generation stops cleanly once this limit is reached.
                              None = no cap.
    """

    # ---- Graph structure ----
    chain_length: int = 3
    max_branching_factor: int = 1
    branch_probability: float = 0.0
    num_objects: int = 2
    num_irrelevant_attributes: int = 1
    num_hypothesis_subjects: Optional[int] = None

    # ---- Source / Type-4 settings ----
    num_sources_range: tuple = (2, 2)
    type4_scenarios: Optional[List[str]] = None
    type4_b_first_prob: float = 0.7
    type4_enabled: bool = True
    type4_object_fraction: float = 0.5

    # ---- Reveal-order randomisation ----
    reveal_order_random_prob: float = 0.5

    # ---- Conversation length caps ----
    max_hypothesis_sections: Optional[int] = None
    max_updates_per_section: Optional[int] = None
    max_total_turns: Optional[int] = None


@dataclass
class InheritanceGraphConfig:
    """
    Configuration for both Linear and Tree inheritance topologies.

    max_depth       : Maximum depth of the class hierarchy from root to leaf.
    root_breadth    : Number of direct children of the root class.
    min_branching   : Minimum number of children a non-leaf class can have.
    max_branching   : Maximum number of children a non-leaf class can have.
    survival_prob   : Probability that a class node spawns at least one child
                      at the next depth level (controls tree sparsity).
    branching_decay : Multiplier applied to branching limits at each depth level.
                      Values < 1 make deeper levels narrower.
    """

    max_depth: int = 4
    root_breadth: int = 5
    min_branching: int = 1
    max_branching: int = 3
    survival_prob: float = 0.6
    branching_decay: float = 1.0

    @classmethod
    def from_preset(cls, preset: str):
        presets = {
            "jellyfish": {
                "max_depth": 5,
                "root_breadth": 8,
                "min_branching": 1,
                "max_branching": 1,
                "survival_prob": 0.4,
                "branching_decay": 1.0,
            },
            "balanced": {
                "max_depth": 4,
                "root_breadth": 2,
                "min_branching": 2,
                "max_branching": 2,
                "survival_prob": 1.0,
                "branching_decay": 1.0,
            },
            "bushy": {
                "max_depth": 4,
                "root_breadth": 4,
                "min_branching": 1,
                "max_branching": 4,
                "survival_prob": 0.8,
                "branching_decay": 0.75,
            },
        }
        chosen = presets.get(preset.lower(), presets["jellyfish"])
        return cls(**chosen)


@dataclass
class GraphGenerationConfig:
    """
    Top-level configuration object passed to a graph generator.

    topology                : Which graph topology to generate.
    root_name               : Label of the root node (property root or class root).
    default_config          : Detailed config for the Default topology.
    inheritance_config      : Detailed config for Linear / Tree inheritance topologies.
    sparsity_factor         : Controls defeasible-edge density.  For Default topology,
                              this is the probability that a (subject, chain-node) pair
                              receives a defeasible evidence edge.
    root_concept            : Semantic domain seed used by the KB for realistic labels
                              (e.g. "Object", "Animal", "Plant", "Tool").
    use_conceptnet          : When True, ConceptNet is queried for semantically coherent
                              chain properties.  Slower but more realistic.
    allow_nonsensical_entities: When True, entity names may be arbitrary identifiers
                              rather than real-world nouns.
    """

    topology: TopologyType = TopologyType.TREE_INHERITANCE
    root_name: str = "Root_Class"
    default_config: DefaultGraphConfig = field(default_factory=DefaultGraphConfig)
    inheritance_config: InheritanceGraphConfig = field(
        default_factory=lambda: InheritanceGraphConfig.from_preset("jellyfish")
    )
    sparsity_factor: float = 0.4
    root_concept: str = "animal"
    use_conceptnet: bool = True
    allow_nonsensical_entities: bool = True
