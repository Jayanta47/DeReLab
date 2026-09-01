"""
DomainAttributeManager
======================
Reads domain_attributes_values.json — a combined file that maps each domain
directly to its allowed attributes and their domain-specific examples:

    {
      "Tool": {
        "capability": {"examples": [...], "template": "...", "object_template": "..."},
        ...
      },
      "Animal": { ... }
    }

Used for default-reasoning graph population where attribute selection must be
domain-aware. Inheritance population continues to use the original
AttributeManager + domains.json + attributes.json.
"""

import json
import random
from typing import Dict, List, Optional


class DomainAttributeManager:

    def __init__(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            self._data: Dict[str, Dict] = json.load(f)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def has_domain(self, domain: str) -> bool:
        return self._resolve_domain(domain) is not None

    def get_attributes(self, domain: str, num_attributes: int) -> List[Dict]:
        """
        Return `num_attributes` randomly sampled attribute dicts for `domain`.

        If the domain's own pool has fewer unique keys than requested, the
        remaining slots are filled with attributes from other domains (to
        avoid repeating the same attribute key in a single chain).
        Falls back to a flattened union of all domains when the domain is not
        in the file.
        """
        resolved = self._resolve_domain(domain)
        primary_pool = self._domain_pool(resolved or domain)

        if len(primary_pool) >= num_attributes:
            return self._sample(primary_pool, num_attributes)

        # Primary domain exhausted — take all of it and supplement cross-domain.
        result = self._sample(primary_pool, len(primary_pool))
        used_keys = {r["attribute_name"] for r in result}

        cross_pool: Dict[str, Dict] = {}
        for d_name, d_attrs in self._data.items():
            if d_name == domain:
                continue
            for key, attr_data in d_attrs.items():
                if key not in used_keys:
                    cross_pool.setdefault(key, attr_data)

        remaining = num_attributes - len(result)
        result.extend(self._sample(cross_pool, remaining))
        return result

    def get_irrelevant_attributes(
        self,
        domain: str,
        num_irr_attr: int,
        exclusion_set: List[str],
    ) -> List[Dict]:
        """
        Return attributes that are NOT in `exclusion_set`.

        Preference: attributes from domains OTHER than `domain` (semantically
        distant).  Falls back to same-domain attributes if the cross-domain
        pool is too small.
        """
        excl = set(exclusion_set)
        resolved = self._resolve_domain(domain) or domain

        # Cross-domain pool (other domains first, for semantic distance).
        cross: Dict[str, Dict] = {}
        same: Dict[str, Dict] = {}
        for d_name, d_attrs in self._data.items():
            target = cross if d_name != resolved else same
            for key, attr_data in d_attrs.items():
                if key not in excl:
                    target.setdefault(key, attr_data)

        pool = {**cross, **same}  # cross-domain keys take precedence
        return self._sample(pool, num_irr_attr)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_domain(self, domain: str) -> Optional[str]:
        """Return the exact key matching `domain` (case-insensitive), or None."""
        domain_lower = domain.lower()
        for key in self._data:
            if key.lower() == domain_lower:
                return key
        return None

    def _domain_pool(self, domain: str) -> Dict[str, Dict]:
        """Return the attribute dict for `domain`, or a flattened fallback."""
        if domain in self._data:
            return self._data[domain]
        # Flatten all domains as a fallback.
        merged: Dict[str, Dict] = {}
        for d_attrs in self._data.values():
            for key, attr_data in d_attrs.items():
                merged.setdefault(key, attr_data)
        return merged

    def _process(self, attr_key: str, attr_data: Dict) -> Dict:
        selected_value = random.choice(attr_data["examples"]).lower()
        return {
            "attribute_name": attr_key,
            "selected_value": selected_value,
            "filled_template": attr_data["template"].replace("[value]", selected_value),
            "filled_object_template": attr_data["object_template"].replace(
                "[value]", selected_value
            ),
        }

    def _sample(self, pool: Dict[str, Dict], n: int) -> List[Dict]:
        keys = list(pool.keys())
        count = min(n, len(keys))
        if count == 0:
            return []
        return [self._process(k, pool[k]) for k in random.sample(keys, count)]
