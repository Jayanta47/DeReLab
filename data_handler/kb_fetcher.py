import json



#  all the functions here return a list of canonical labels or predicate_lemmas/properties
class KbFetcher:
    def __init__(self, ontology_path, properties_path=None):
        """
        Initializes the analyzer by reading ontology and (optionally) properties data.

        :param ontology_path: Path to the JSON file containing 'nodes' and 'edges'
        :param properties_path: Optional path to domain_properties.json; omit when
                                 property fetching is handled elsewhere.
        """

        with open(ontology_path, 'r', encoding='utf-8') as f:
            ontology_json = json.load(f)

        self.properties = []
        if properties_path:
            with open(properties_path, 'r', encoding='utf-8') as f:
                self.properties = json.load(f)
            
        self.nodes = ontology_json.get("nodes", {})
        self.edges = ontology_json.get("edges",[])
        
        # Precompute a mapping from canonical_label to a list of node_ids
        self.label_to_ids = {}
        for node_id, node_data in self.nodes.items():
            label = node_data.get("canonical_label")
            if label:
                label_lower = label.lower()
                if label_lower not in self.label_to_ids:
                    self.label_to_ids[label_lower] =[]
                self.label_to_ids[label_lower].append(node_id)

    def get_subclasses(self, canonical_label):
        """
        Returns a list of canonical labels that are subclasses of the given label.
        Relation rule: source = parent (superclass), target = subclass
        """
        label_lower = canonical_label.lower()
        if label_lower not in self.label_to_ids:
            return[]

        parent_ids = self.label_to_ids[label_lower]
        subclasses = set()

        for edge in self.edges:
            # If the given label is the source (parent), the target is its subclass
            if edge.get("relation") == "subclass" and edge.get("source") in parent_ids:
                target_id = edge.get("target")
                if target_id in self.nodes:
                    target_label = self.nodes[target_id].get("canonical_label")
                    if target_label:
                        subclasses.add(target_label)

        return list(subclasses)

    def get_superclasses(self, canonical_label):
        """
        Returns a list of canonical labels that are superclasses of the given label.
        Relation rule: source = parent (superclass), target = subclass
        """
        label_lower = canonical_label.lower()
        if label_lower not in self.label_to_ids:
            return[]

        child_ids = self.label_to_ids[label_lower]
        superclasses = set()

        for edge in self.edges:
            if edge.get("relation") == "subclass" and edge.get("target") in child_ids:
                source_id = edge.get("source")
                if source_id in self.nodes:
                    source_label = self.nodes[source_id].get("canonical_label")
                    if source_label:
                        superclasses.add(source_label)

        return list(superclasses)

    def get_properties(self, canonical_label):
        """
        Returns a list of predicate_lemmas (properties) where the given label 
        matches either the 'domain' or is inside 'allowed_subject_types'.
        """
        label_lower = canonical_label.lower()
        matched_properties = set()

        for prop in self.properties:
            domain = prop.get("domain", "")
            allowed_subjects =[subj.lower() for subj in prop.get("allowed_subject_types", [])]

            if label_lower == domain.lower() or label_lower in allowed_subjects:
                predicate = prop.get("predicate_lemma")
                if predicate:
                    matched_properties.add(predicate)

        return list(matched_properties)
    
    def get_complete_properties(self,canonical_label):
        # instead of only returning the predicate lemma like get_properties, return the full object
        # do not keep duplicates, if two properties have the same predicate lemma and frame_type, only keep one of them
        label_lower = canonical_label.lower()
        matched_properties = {}

        for prop in self.properties:
            domain = prop.get("domain", "")
            allowed_subjects =[subj.lower() for subj in prop.get("allowed_subject_types", [])]

            if label_lower == domain.lower() or label_lower in allowed_subjects:
                predicate = prop.get("predicate_lemma")
                frame_type = prop.get("frame_type")
                if predicate and frame_type:
                    key = (predicate, frame_type)
                    if key not in matched_properties:
                        matched_properties[key] = prop

        return list(matched_properties.values())



# write a main

if __name__ == "__main__":

    fetcher = KbFetcher('../knowledge_graph/data_sources/knowledge_graph.json', '../knowledge_graph/data_sources/domain_properties.json')

    print("Subclasses of nymph:", fetcher.get_subclasses("nymph")) 

    print("Superclasses of larva:", fetcher.get_superclasses("larva")) 

    print("Properties of insect:", fetcher.get_properties("insect")) 

    print("Properties of animals:", fetcher.get_properties("animal"))

    print("Complete properties of animals:", fetcher.get_complete_properties("animal"))