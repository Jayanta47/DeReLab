
from html import entities
from typing import Dict, Any, List
from knowledge_graph.non_sensical.baseGenerator import EntityGenerator
from knowledge_graph.non_sensical.markovGeneratorHelper.corpusProvider.base import CorpusProvider
from knowledge_graph.non_sensical.markovGeneratorHelper.corpusProvider.phonotactic import PhonotacticCorpusProvider
from knowledge_graph.non_sensical.markovGeneratorHelper.corpusProvider.reference import ReferenceCorpusProvider
from knowledge_graph.non_sensical.markovGeneratorHelper.corpusProvider.domain import RootwordCorpusProvider
from knowledge_graph.non_sensical.markovGeneratorHelper.markovModel import MarkovModel
from knowledge_graph.non_sensical.markovGeneratorHelper.entityFilter import EntityFilter
from knowledge_graph.non_sensical.markovGenerator import MarkovEntityGenerator, CompoundEntityGenerator
from configs.nonsenseConfig import BatchConfig, TaskConfig, CorpusConfig, SingleGeneratorConfig, CompoundGeneratorConfig
import json



REF_JSON_PATH = "KnowledgeGraph/data_sources/non_sensical_corpus.json"
SAVE_OUTPUT_PATH = "KnowledgeGraph/data_sources/non_sensical_entities.json"

class NonSensicalGeneratorService:
    """
    Reads a Class-based configuration, wires up the necessary Providers, 
    Models, and Generators, and executes the batch generation tasks.
    """
    def __init__(self, config: BatchConfig):
        self.config = config

    def _build_provider(self, corpus_cfg: CorpusConfig) -> CorpusProvider:
        if corpus_cfg.corpus_type == "phonotactic":
            return PhonotacticCorpusProvider(corpus_size=corpus_cfg.corpus_size)
        elif corpus_cfg.corpus_type == "rootword":
            return RootwordCorpusProvider(seed_word=corpus_cfg.default_words[0] if corpus_cfg.default_words else "animal", max_results_per_query=corpus_cfg.corpus_size)
        else:
            json_filepath = corpus_cfg.json_filepath or REF_JSON_PATH
            json_key = corpus_cfg.json_key 
            json_field = corpus_cfg.json_field
            if json_key:
                return ReferenceCorpusProvider(
                    default_words=corpus_cfg.default_words,
                    json_filepath=json_filepath,
                    json_key=json_key
                )
            elif json_field:
                return ReferenceCorpusProvider(
                    default_words=corpus_cfg.default_words,
                    json_filepath=json_filepath,
                    json_field=json_field
                )
            return ReferenceCorpusProvider(default_words=corpus_cfg.default_words, json_filepath=REF_JSON_PATH, json_key="nouns")

    def _build_single_generator(self, gen_cfg: SingleGeneratorConfig) -> MarkovEntityGenerator:
        # 1. Build Provider
        provider = self._build_provider(gen_cfg.corpus)
        
        # 2. Train Model
        model = MarkovModel(order=gen_cfg.markov_order)
        model.train(provider)
        
        # 3. Build Filter
        entity_filter = EntityFilter(
            min_len=gen_cfg.filter_rules.min_len,
            max_len=gen_cfg.filter_rules.max_len,
            max_consonants=gen_cfg.filter_rules.max_consonants,
            no_repeating_chars=gen_cfg.filter_rules.no_repeating_chars
        )
        
        # 4. Return wired generator
        return MarkovEntityGenerator(
            model=model, 
            filter_rules=entity_filter, 
            seed_provider=provider if gen_cfg.use_provider_as_seed else None
        )

    def _process_task(self, task: TaskConfig) -> List[str]:
        if isinstance(task.generator, CompoundGeneratorConfig):
            left_gen = self._build_single_generator(task.generator.left_part)
            right_gen = self._build_single_generator(task.generator.right_part)
            generator = CompoundEntityGenerator(
                left_generator=left_gen, 
                right_generator=right_gen, 
                separator=task.generator.separator
            )
        else:
            generator = self._build_single_generator(task.generator)

        entities = set()
        max_attempts = task.count * 20
        attempts = 0
        
        while len(entities) < task.count and attempts < max_attempts:
            entities.add(generator.generate())
            attempts += 1
        
        return list(entities)

    def run(self) -> Dict[str, List[str]]:
        """Executes all tasks defined in the configuration."""
        results = {}
        for task in self.config.tasks:
            results[task.task_name] = self._process_task(task)
        # save in a json
        with open(SAVE_OUTPUT_PATH, "a") as f:
            json.dump(results, f, indent=4)
        return results
