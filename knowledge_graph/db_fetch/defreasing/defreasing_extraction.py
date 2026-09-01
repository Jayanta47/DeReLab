import csv
import json
import re
from collections import defaultdict


ISA_PATTERN = re.compile(
    r"^(?:A |An )?(?P<child>[\w\s\-]+?)\s+(?:is|are)\s+(?:a |an )?(?P<parent>[\w\s\-]+?)$",
    re.IGNORECASE
)

PROPERTY_PATTERN = re.compile(
    r"^(?P<entity>[\w\s\-]+?)\s+(?P<property>.+)$",
    re.IGNORECASE
)

count=0


entities = defaultdict(lambda: {
    "subclasses": set(),
    "properties": set(),
    "sentences": set()
})



def normalize(text: str) -> str:
    return text.strip().rstrip(".")


def process_sentence(sentence: str):
    sentence = normalize(sentence)

    # 1. isA check (highest priority)
    isa_match = ISA_PATTERN.match(sentence)
    if isa_match:
        child = normalize(isa_match.group("child"))
        parent = normalize(isa_match.group("parent"))

        entities[parent]["subclasses"].add(child)
        entities[parent]["sentences"].add(sentence)
        entities.setdefault(child, entities[child])

        return

    # 2. property extraction
    prop_match = PROPERTY_PATTERN.match(sentence)
    if prop_match:
        global count
        entity = normalize(prop_match.group("entity"))
        count+=1
        property_ = normalize(prop_match.group("property"))

        entities[entity]["properties"].add(property_)
        entities[entity]["sentences"].add(sentence)

def process_csv(csv_path: str):
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header

        for row in reader:
            premise = row[0]
            sentences = [s for s in premise.split(".") if s.strip()]
            for s in sentences:
                process_sentence(s)





process_csv("./db/defreasing.csv")

final_json = {
    "entities": {
        k: {
            "subclasses": sorted(v["subclasses"]),
            "properties": sorted(v["properties"]),
            "sentences": sorted(v["sentences"]),
        }
        for k, v in entities.items()
    }
}

# total unique properties extracted
unique_properties = set()
for v in final_json["entities"].values():
    unique_properties.update(v["properties"])
print(f"Total unique properties extracted: {len(unique_properties)}")


with open("./db/defreasing_extraction.json", "w", encoding="utf-8") as f:
    json.dump(final_json, f, indent=4, ensure_ascii=False)
