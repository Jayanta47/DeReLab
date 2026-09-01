from dataclasses import dataclass, field
from typing import List, Optional, Union

@dataclass
class FilterConfig:
    """Configuration for phonetic and aesthetic constraints."""
    min_len: int = 4
    max_len: int = 10
    max_consonants: int = 3
    no_repeating_chars: bool = True

@dataclass
class CorpusConfig:
    """Configuration for how the Markov Model gets its training data."""
    corpus_type: str = "reference"  # Accepts "reference" or "phonotactic"
    corpus_size: int = 2000         # Only used if corpus_type == "phonotactic"
    default_words: List[str] = field(default_factory=list)
    json_filepath: Optional[str] = None
    json_key: Optional[str] = None
    json_field: Optional[str] = None

@dataclass
class SingleGeneratorConfig:
    """Configuration for a standard (single word) Markov Generator."""
    corpus: CorpusConfig
    filter_rules: FilterConfig = field(default_factory=FilterConfig)
    markov_order: int = 2
    use_provider_as_seed: bool = False

@dataclass
class CompoundGeneratorConfig:
    """Configuration for combining two generators (e.g., Adjective-Noun)."""
    left_part: SingleGeneratorConfig
    right_part: SingleGeneratorConfig
    separator: str = "-"

@dataclass
class TaskConfig:
    """Defines a specific batch generation task to be executed."""
    task_name: str
    count: int
    generator: Union[SingleGeneratorConfig, CompoundGeneratorConfig]

@dataclass
class BatchConfig:
    """The Root Configuration object containing a list of tasks."""
    tasks: List[TaskConfig] = field(default_factory=list)


COMPOUND_WORD_JSON_PATH = "KnowledgeGraph/data_sources/non_sensical_corpus.json"
DOMAIN_JSON_PATH = "KnowledgeGraph/data_sources/domains.json"


class GeneratorPresets:



    # WAY-1: PHONENOTACTIC SYLLABLES (ZERO-BIAS)
    @staticmethod
    def zero_bias_dataset_task(count: int = 10) -> TaskConfig:
        """
        PRESET 1: Pure Zero-Bias Entities.
        Uses the phonotactic syllable generator to ensure 0% LLM semantic bias.
        Perfect for replacing real-world entities in training/testing datasets.
        """
        return TaskConfig(
            task_name="Zero_Bias_Phonetactic_Entities",
            count=count,
            generator=SingleGeneratorConfig(
                corpus=CorpusConfig(
                    corpus_type="phonotactic",
                    corpus_size=3000
                ),
                # Kept to 5-8 chars so tokenizers don't shatter them
                filter_rules=FilterConfig(min_len=5, max_len=8, max_consonants=2),
                markov_order=2,
                use_provider_as_seed=False
            )
        )

    # WAY-2: COMPOUND NAMES (ADJ-NOUN) USING REFERENCE WORDS
    @staticmethod
    def fantasy_compound_task(count: int = 10, json_path: str = COMPOUND_WORD_JSON_PATH) -> TaskConfig:
        """
        PRESET 2: Fantasy / Faction Names.
        Creates hyphenated names (e.g., "Glim-Dragon") using adjectives and nouns.
        """
        return TaskConfig(
            task_name="Compound_Names",
            count=count,
            generator=CompoundGeneratorConfig(
                separator="-",
                left_part=SingleGeneratorConfig(
                    corpus=CorpusConfig(
                        corpus_type="reference",
                        json_filepath=json_path,
                        json_key="adjectives",
                        default_words=["silent", "tragic", "florid", "mystic", "crimson"]
                    ),
                    filter_rules=FilterConfig(min_len=4, max_len=6),
                    markov_order=2,
                    use_provider_as_seed=True
                ),
                right_part=SingleGeneratorConfig(
                    corpus=CorpusConfig(
                        corpus_type="reference",
                        json_filepath=json_path,
                        json_key="nouns",
                        default_words=["dragon", "tiger", "eagle", "spider", "raven"]
                    ),
                    filter_rules=FilterConfig(min_len=4, max_len=7),
                    markov_order=2,
                    use_provider_as_seed=False
                )
            )
        )

    # WAY-3: USE DOMAIN-NAMES AS SEEDS
    @staticmethod
    def domain_task(count: int = 10, json_path: str = DOMAIN_JSON_PATH, json_field: str = "domain_name") -> TaskConfig:
        """
        PRESET 3: Clinical / Sci-Fi Terminology.
        Generates longer, highly structured words that sound official or alien 
        (e.g., "Mivolax", "Brelinta").
        """
        return TaskConfig(
            task_name="Food",
            count=count,
            generator=SingleGeneratorConfig(
                corpus=CorpusConfig(
                    corpus_type="rootword",
                    default_words=["food", "cuisine", "dish", "meal", "snack"],
                    corpus_size=int(count * 0.1), 
                ),
                filter_rules=FilterConfig(min_len=5, max_len=7, max_consonants=2),
                markov_order=2, # Higher order makes it sound more deliberate/latin-like
                use_provider_as_seed=True
            )
        )

    @staticmethod
    def full_evaluation_batch(count_per_task: int = 5) -> BatchConfig:
        """
        PRESET 4: The Ultimate Test Batch.
        Returns a complete BatchConfig containing one of every preset, 
        perfect for quickly testing the whole system.
        """
        return BatchConfig(
            tasks=[
                # GeneratorPresets.zero_bias_dataset_task(count=count_per_task),
                # GeneratorPresets.fantasy_compound_task(count=count_per_task),
                GeneratorPresets.domain_task(count=count_per_task)
            ]
        )