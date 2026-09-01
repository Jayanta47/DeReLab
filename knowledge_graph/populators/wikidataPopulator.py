import os
import json
import time
import requests
from abc import ABC, abstractmethod
import nltk
from nltk.corpus import wordnet as wn
from knowledge_graph.populators.basePopulator import BaseGraph



class WikidataGraph(BaseGraph):
    """Builds and appends Wikidata taxonomic chains into the graph."""
    
    def __init__(self, root_word, json_filepath="taxonomy_graph.json", max_depth=2):
        super().__init__(root_word, json_filepath)
        self.max_depth = max_depth
        self._visited_qids = set()
        # Wikimedia APIs strictly require a User-Agent header
        self.headers = {
            "User-Agent": "TaxonomyBuilder/1.0 (educational data science script)"
        }

    def _get_root_entity(self):
        url = "https://www.wikidata.org/w/api.php"
        params = {
            "action": "wbsearchentities",
            "format": "json",
            "language": "en",
            "search": self.root_word
        }
        
        # Added headers and raise_for_status to handle 403 Forbidden errors properly
        response = requests.get(url, params=params, headers=self.headers)
        response.raise_for_status() 
        
        data = response.json()
        if not data.get('search'):
            raise ValueError(f"Error: No Wikidata entity found for '{self.root_word}'")
        
        best_match = data['search'][0]
        return {
            "id": best_match['id'],
            "label": best_match.get('label', self.root_word),
            "description": best_match.get('description', ''),
            "aliases": best_match.get('aliases', [])
        }

    def _infer_edge_relationship(self, description):
        if not description:
            return "subclass"
            
        desc = description.lower()
        property_keywords = [
            'young', 'immature', 'baby', 'juvenile', 'adult', 
            'female', 'male', 'castrated', 
            'used for', 'kept for', 'trained', 'raised', 'having powers of',
            'role', 'state', 'condition', 'ability'
        ]
        
        return "property" if any(w in desc for w in property_keywords) else "subclass"

    def _query_subclasses(self, qid):
        url = "https://query.wikidata.org/sparql"
        query = f"""
        SELECT ?child ?childLabel ?childDescription (GROUP_CONCAT(?altLabel; separator="|") as ?aliases)
        WHERE {{
          ?child wdt:P279 wd:{qid}.
          OPTIONAL {{
            ?child skos:altLabel ?altLabel .
            FILTER(LANG(?altLabel) = "en")
          }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        GROUP BY ?child ?childLabel ?childDescription
        LIMIT 100
        """
        
        headers = self.headers.copy()
        headers["Accept"] = "application/sparql-results+json"
        
        try:
            response = requests.get(url, params={'query': query}, headers=headers)
            response.raise_for_status()
            return response.json()['results']['bindings']
        except Exception as e:
            print(f"Failed to query {qid}: {e}")
            return []

    def build(self):
        print(f"Resolving '{self.root_word}' on Wikidata...")
        root_data = self._get_root_entity()
        root_qid = root_data["id"]
        
        print(f"Found Root: {root_data['label']} ({root_qid}) - {root_data['description']}")
        self._add_node(root_qid, "wikidata", root_data["label"], root_data["aliases"])
        
        self._populate_children_bfs([root_qid], current_depth=0)
        self._save_graph()

    def _populate_children_bfs(self, qid_list, current_depth):
        if current_depth >= self.max_depth or not qid_list:
            return
            
        next_level_qids = []
        for qid in qid_list:
            if qid in self._visited_qids:
                continue
            self._visited_qids.add(qid)
            
            print(f"Querying subclasses for {qid} (Depth {current_depth+1}/{self.max_depth})...")
            results = self._query_subclasses(qid)
            time.sleep(0.5) # Polite API usage
            
            for row in results:
                child_url = row.get('child', {}).get('value', '')
                child_id = child_url.split('/')[-1]
                child_label = row.get('childLabel', {}).get('value', '')
                child_desc = row.get('childDescription', {}).get('value', '')
                
                alias_str = row.get('aliases', {}).get('value', '')
                aliases = list(set([a.strip() for a in alias_str.split('|') if a.strip()]))
                
                if not child_label or child_label.startswith('Q'):
                    continue
                    
                self._add_node(child_id, "wikidata", child_label, aliases)
                
                relation_type = self._infer_edge_relationship(child_desc)
                self._add_edge(
                    source_id=qid,
                    target_id=child_id,
                    relation_type=relation_type,
                    context=child_desc
                )
                
                next_level_qids.append(child_id)

        self._populate_children_bfs(next_level_qids, current_depth + 1)

