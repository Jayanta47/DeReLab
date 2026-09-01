from nltk.corpus import wordnet as wn
from knowledge_graph.populators.wordnetPopulator import WordNetGraph
from knowledge_graph.populators.wikidataPopulator import WikidataGraph
from knowledge_graph.populators.conceptnetPopulator import ConceptNetGraph
from utils.visualizer import  KGVisualizer

OUTPUT_FILE = "KnowledgeGraph/knowledge_graph.json"

# # 1. Run WordNet
# print("\n=== Extracting from WordNet ===")
# wn_graph = WordNetGraph("animal", json_filepath=OUTPUT_FILE)
# wn_graph.build()

# # 2. Run Wikidata (Appends seamlessly to the same file)
# print("\n=== Extracting from Wikidata ===")
# wd_graph = WikidataGraph("animal", json_filepath=OUTPUT_FILE, max_depth=2)
# wd_graph.build()


print("\n=== Extracting from ConceptNet ===")

# Restrict to depth 2, because ConceptNet is highly interconnected!
cnet_graph = ConceptNetGraph("animal", json_filepath=OUTPUT_FILE, max_depth=2)
cnet_graph.build()

visualizer = KGVisualizer(OUTPUT_FILE)
visualizer.visualize("taxonomy_visualization.html")