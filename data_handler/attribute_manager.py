import json
import random
from typing import List, Dict, Any
import os

class AttributeManager:
    def __init__(self, attribute_path: str, domain_path: str):
        """
        Initializes the manager by loading the attributes and domains JSON files.
        """

        try:
            with open(attribute_path, "r", encoding="utf-8") as f:
                self.attrData = json.load(f)
            self.all_attr_keys = list(self.attrData.keys())
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find the file at {attribute_path}")
        except json.JSONDecodeError:
            raise ValueError(f"The file at {attribute_path} is not valid JSON.")
        
  
        try:
            with open(domain_path, "r", encoding="utf-8") as f:
                domain_list = json.load(f)
            
            # Map domain names to their allowed attributes for O(1) lookup
            self.domains = {}
            for d in domain_list:
                domain_name = d.get("domain_name")
                if domain_name:
                    self.domains[domain_name] = d.get("allowed_attributes",[])
                    
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find the file at {domain_path}")
        except json.JSONDecodeError:
            raise ValueError(f"The file at {domain_path} is not valid JSON.")

    def _process_attribute(self, key: str) -> Dict[str, str]:
        """
        Internal helper to select a random example for a given attribute key
        and fill in its templates.
        """
        attr_data = self.attrData[key]

        selected_value = random.choice(attr_data["examples"]).lower()

        filled_template = attr_data["template"].replace("[value]", selected_value)
        filled_obj_template = attr_data["object_template"].replace(
            "[value]", selected_value
        )

        return {
            "attribute_name": key,
            "selected_value": selected_value,
            "filled_template": filled_template,
            "filled_object_template": filled_obj_template,
        }

    def get_attributes(self, num_attributes: int, domain: str = None) -> List[Dict[str, str]]:
        """
        Selects `num_attributes` keys, prioritizing the domain's allowed attributes.
        If more attributes are requested than allowed by the domain, it falls back
        to other available attributes.
        """
        count = min(num_attributes, len(self.all_attr_keys))
        if count == 0:
            return[]


        domain_allowed_keys =[]
        if domain and domain in self.domains:
            # Only consider allowed keys that actually exist in attrData
            domain_allowed_keys =[k for k in self.domains[domain] if k in self.attrData]

        selected_keys =[]

        if domain_allowed_keys:
            num_from_domain = min(count, len(domain_allowed_keys))
            selected_keys.extend(random.sample(domain_allowed_keys, num_from_domain))

        # 2. Fallback to other attributes if we still need more
        if len(selected_keys) < count:
            needed = count - len(selected_keys)
            remaining_pool =[k for k in self.all_attr_keys if k not in selected_keys]
            selected_keys.extend(random.sample(remaining_pool, needed))


        results =[]
        for key in selected_keys:
            results.append(self._process_attribute(key))

        return results

    def get_irrelevant_attributes(
        self, num_irr_attr: int, exclusion_set: List[str], domain: str = None
    ) -> List[Dict[str, str]]:
        """
        Selects attributes that are NOT in the exclusion_set AND NOT in the 
        domain's allowed_attributes (to ensure they are truly irrelevant).
        """

        full_exclusion = set(exclusion_set)
        if domain and domain in self.domains:
            full_exclusion.update(self.domains[domain])


        available_pool =[k for k in self.all_attr_keys if k not in full_exclusion]

        count = min(num_irr_attr, len(available_pool))
        if count == 0:
            return []

        selected_keys = random.sample(available_pool, count)

        results =[]
        for key in selected_keys:
            results.append(self._process_attribute(key))

        return results



if __name__ == "__main__":

    test_attributes = {
        "material": {
            "examples": ["Gold", "Plastic"],
            "template": "are made of [value]",
            "object_template": "is made of[value]",
        },
        "color": {
            "examples": ["Red", "Blue"],
            "template": "are [value] in color",
            "object_template": "is [value] in color",
        },
        "size": {
            "examples": ["Huge", "Tiny"],
            "template": "are [value] in size",
            "object_template": "is [value] in size",
        },
        "origin": {
            "examples": ["Alien", "Earth"],
            "template": "are of [value] origin",
            "object_template": "is of [value] origin",
        },
        "habitat": {
            "examples": ["Desert", "Ocean"],
            "template": "live in [value]",
            "object_template": "lives in [value]"
        }
    }


    test_domains =[
        {
            "domain_name": "Animal",
            "allowed_attributes": ["color", "size", "habitat"]
        },
        {
            "domain_name": "Tool",
            "allowed_attributes":["material", "color", "size"]
        }
    ]

    with open("attributes.json", "w") as f:
        json.dump(test_attributes, f)
    with open("domains.json", "w") as f:
        json.dump(test_domains, f)


    manager = AttributeManager("attributes.json", "domains.json")

    print("--- Getting 2 Attributes for Domain 'Tool' ---")

    attrs = manager.get_attributes(2, domain="Tool")
    for a in attrs:
        print(a["attribute_name"], "->", a["selected_value"])

    print("\n--- Getting 4 Attributes for Domain 'Animal' (Requires Fallback) ---")

    attrs = manager.get_attributes(4, domain="Animal")
    for a in attrs:
        print(a["attribute_name"], "->", a["selected_value"])

    print("\n--- Getting 1 Irrelevant Attribute for 'Animal' ---")

    irr_attrs = manager.get_irrelevant_attributes(1, exclusion_set=["material"], domain="Animal")
    for a in irr_attrs:
        print(a["attribute_name"], "->", a["selected_value"])

    # Cleanup test files
    os.remove("attributes.json")
    os.remove("domains.json")