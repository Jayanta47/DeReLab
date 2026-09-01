import os
import random
import json
from typing import Callable, Dict, List, Optional

from .attribute_manager import AttributeManager
from .domain_attribute_manager import DomainAttributeManager
from .kb_fetcher import KbFetcher


class KnowledgeBaseManager:
    allow_nonsensical_entities: bool = True

    def __init__(
        self,
        attribute_path: str = "knowledge_graph/data_sources/attributes.json",
        domain_path: str = "knowledge_graph/data_sources/domains.json",
        domain_attributes_path: str = "knowledge_graph/data_sources/domain_attributes_values.json",
        kb_path: str = "knowledge_graph/data_sources/knowledge_graph.json",
        nonsense_json_path: str = "knowledge_graph/data_sources/non_sensical_entities.json",
        names_path: str = "knowledge_graph/data_sources/names.txt",
        only_nonsensical: bool = True,
    ):
        self.attr_manager = AttributeManager(attribute_path=attribute_path, domain_path=domain_path)
        self.domain_attr_manager = DomainAttributeManager(path=domain_attributes_path)
        self.kb_fetcher = KbFetcher(kb_path)
        self.only_nonsensical= only_nonsensical

        self.nonsensical_entities =[]
        if os.path.exists(nonsense_json_path):
            with open(nonsense_json_path, "r", encoding="utf-8") as f:
                try:
                    self.nonsensical_entities = json.load(f)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON from {nonsense_json_path}. No nonsensical entities loaded.")

        self._source_names: List[str] = []
        if os.path.exists(names_path):
            with open(names_path, "r", encoding="utf-8") as f:
                self._source_names = [ln.strip() for ln in f if ln.strip()]

        # Templates for inheritance topology (taxa/category relationships).
        self.templates = {
            "is_a": "{subject} is a {object}.",
            "implies": "Objects that {subject_predicate} generally {object_predicate_plural}.",
            "conflict": "However, {subject} does not {object}.",
            "has_attribute": "{subject} {object_predicate}.",
            "inheritance": "{subject} is a kind of {object}.",
            "defeasible_has_property": "{subject} {object_predicate}.",
            "defeasible_lacks_property": "{subject} does not {object_predicate}.",
            "hypothesis_edge": "Does {subject} {object_predicate}?",
        }

        # Templates for default-reasoning topology (property-chain relationships).
        # Note: is_a is intentionally absent — entity instantiations use the
        # property node's filled_object_template directly in nl_generator.py.
        self.default_templates = {
            "implies": "Objects that {subject_predicate} generally {object_predicate_plural}.",
            "has_attribute": "{subject} {object_predicate}.",
            "defeasible_has_property": "{subject} {object_predicate}.",
            "defeasible_lacks_property": "{subject} does not {object_predicate}.",
            "hypothesis_edge": "Does {subject} {object_predicate}?",
        }

    def get_template(self, edge_type: str, topology: str = "inheritance") -> str:
        template_set = (
            self.default_templates if topology == "default" else self.templates
        )
        template = template_set.get(edge_type) or self.templates.get(edge_type)
        if not template:
            return "{subject} is related to {object}."
        return template


    # randomly choose from the allowed attributes of the domain------------------------------
    def get_attributes(self, num_attributes: int, domain: str = None) -> List[Dict[str, str]]:
        return self.attr_manager.get_attributes(num_attributes)

    def get_irrelevant_attributes(
        self, num_irr_attr: int, exclusion_set: List[str]
    ) -> List[Dict[str, str]]:
        return self.attr_manager.get_irrelevant_attributes(num_irr_attr, exclusion_set)

    # --------------default reasoning attribute methods (use domain_attributes_values.json)-------

    def get_default_attributes(
        self, domain: str, num_attributes: int
    ) -> List[Dict[str, str]]:
        """
        Return attributes for default-reasoning chain nodes, drawn from the
        domain-specific pool in domain_attributes_values.json.
        Falls back to the flat attributes.json pool when the domain is unknown.
        """
        if self.domain_attr_manager.has_domain(domain):
            return self.domain_attr_manager.get_attributes(domain, num_attributes)
        return self.attr_manager.get_attributes(num_attributes)

    def get_default_irrelevant_attributes(
        self,
        domain: str,
        num_irr_attr: int,
        exclusion_set: List[str],
    ) -> List[Dict[str, str]]:
        """
        Return irrelevant attributes for default-reasoning distractor nodes.
        Prefers cross-domain attributes (semantically distant from the chain).
        Falls back to the flat attributes.json pool when the domain is unknown.
        """
        if self.domain_attr_manager.has_domain(domain):
            return self.domain_attr_manager.get_irrelevant_attributes(
                domain, num_irr_attr, exclusion_set
            )
        return self.attr_manager.get_irrelevant_attributes(num_irr_attr, exclusion_set)

    # --------------to find the concept chain for inheritance graph--------------
    def get_concept_chain(
        self, min_length: int = 3, root_concept: Optional[str] = None
    ) -> List[str]:
        if root_concept and not self.only_nonsensical:
            chain = self._get_ontology_chain(root_concept, min_length)
            if chain:
                return chain

        if self.allow_nonsensical_entities:
            nonce_chain = self._get_nonce_entity(limit=min_length - 1, root_concept=root_concept)
            if root_concept:
                return [root_concept] + nonce_chain
            return nonce_chain
            
        return[]


    # --------------to find the concepts for default reasoning graph--------------
    def get_entity_labels(
        self,
        num_entities: int,
        root_concept: str = None,
        allow_nonsensical: bool = True,
    ) -> List[Dict[str, str]]:
        
        sensible_pool =[] if self.only_nonsensical else self._get_sensible_entities(root_concept)
        nonsensical_pool = self._get_nonce_entity(limit=num_entities, root_concept=root_concept) if (allow_nonsensical or self.only_nonsensical) else[]
        
        if self.only_nonsensical:
            candidates =[{"label": label, "semantic_type": "nonsensical"} for label in nonsensical_pool]
        else:
            candidates =[{"label": label, "semantic_type": "sensical"} for label in sensible_pool]
            if allow_nonsensical:
                candidates.extend([{"label": label, "semantic_type": "nonsensical"} for label in nonsensical_pool]
                )

        if not candidates:
            candidates =[{"label": f"entity_{idx}", "semantic_type": "synthetic"} for idx in range(num_entities)]

        if len(candidates) >= num_entities:
            selected = random.sample(candidates, num_entities)
        else:
            selected = list(candidates)
            while len(selected) < num_entities:
                seed = random.choice(candidates)
                copy_idx = len(selected) + 1
                selected.append(
                    {
                        "label": f"{seed['label']} #{copy_idx}",
                        "semantic_type": seed["semantic_type"],
                    }
                )

        return selected


    def get_source_names(self, n: int) -> List[str]:
        """
        Sample n distinct person names (without replacement) from names.txt.
        Falls back to generic "Source 1", "Source 2", … when the pool is too small.
        """
        pool = self._source_names
        if len(pool) >= n:
            return random.sample(pool, n)
        selected = list(pool)
        idx = len(selected) + 1
        while len(selected) < n:
            selected.append(f"Source {idx}")
            idx += 1
        return selected

    # property info (with template structure) for the inheritance topology, domain-aware----
    def get_full_property_info(self, root_concept: str, limit: int = 10) -> List[Dict[str, str]]:
        return self.get_default_attributes(root_concept, limit)

    def format_with_generator(
        self,
        generator: Optional[Callable[[str, Dict[str, str]], str]],
        edge_type: str,
        payload: Dict[str, str],
        topology: str = "inheritance",
    ) -> str:
        if generator:
            generated = generator(edge_type, payload)
            if generated:
                return generated

        template = self.get_template(edge_type, topology=topology)
        return template.format(**payload)


    # PRIVATE METHODS

    def _get_ontology_chain(self, root_concept: str, min_length: int) -> List[str]:
        """
        Builds a hierarchy chain by recursively finding subclasses of subclasses.
        If no path reaches min_length, it takes the longest available path
        and fills the remaining gaps with nonce entities.
        """
        if not self.kb_fetcher or min_length <= 0:
            return[]

        longest_path =[]

        # Helper function for Depth-First Search (DFS)
        def dfs(current_concept: str, path: List[str]) -> bool:
            nonlocal longest_path
            
            # Keep track of the deepest chain we've found so far
            if len(path) > len(longest_path):
                longest_path = list(path)
                
            if len(path) == min_length:
                return True
                
            subclasses = self.kb_fetcher.get_subclasses(current_concept)
            random.shuffle(subclasses)
            
            for child in subclasses:
                if child not in path:
                    path.append(child)
                    if dfs(child, path):
                        return True
                    path.pop()
                    
            return False

        dfs(root_concept, [root_concept])

        cleaned_chain = [concept.replace("_", " ") for concept in longest_path]


        gap = min_length - len(cleaned_chain)
        if gap > 0:
            # Fetch nonce words to fill the gap
            nonce_words = self._get_nonce_entity(limit=gap)
            cleaned_chain.extend(nonce_words)

            while len(cleaned_chain) < min_length:
                cleaned_chain.append(f"nonce_entity_{len(cleaned_chain)}")

        return cleaned_chain[:min_length]

    def _get_sensible_entities(self, root_concept: str) -> List[str]:
        """
        Fetches sensible entities directly relying on get_subclasses.
        """
        subclasses = self.kb_fetcher.get_subclasses(root_concept)
        return [concept.replace("_", " ") for concept in subclasses]
    
    def _get_nonce_entity(self, limit: int = 10, root_concept: str = None) -> List[str]:
        """
        Samples nonsensical entities from the parsed JSON list.
        """
        if not self.nonsensical_entities:
            return[]

        nonsense_type = None

        # 2. Find the matching key (case-insensitive) 
        if root_concept:
            root_lower = root_concept.lower()
            for key in self.nonsensical_entities.keys():
                if key.lower() == root_lower:
                    nonsense_type = key
                    break
   
        if nonsense_type is None:
            nonsense_type = random.choice(list(self.nonsensical_entities.keys()))

        nonce_entities = self.nonsensical_entities.get(nonsense_type,[]) 


        sampled = random.sample(nonce_entities, min(limit, len(nonce_entities)))
        
        return sampled