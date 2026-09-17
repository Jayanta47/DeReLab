/**
 * data.js
 * ---------------------------------------------------------------------
 * Static knowledge-base data ported from the Python pipeline so the
 * in-browser simulator can populate a tree-inheritance graph without a
 * backend. Trimmed subsets of the real data sources used at
 * evaluation.dataset generation time:
 *
 *   - knowledge_graph/data_sources/domain_attributes_values.json
 *   - knowledge_graph/data_sources/non_sensical_entities.json
 *
 * Only the seven domains reachable from the tree-inheritance root-concept
 * distribution are included. Two of those domains (Animal, Plant) have a
 * dedicated nonsensical-entity word bank in the source data; the other
 * five (Mammal, Vertebrate, Insect, Microorganism, Bird) reuse the Animal
 * bank here for legibility -- the original Python fallback draws from a
 * uniformly random domain (including unrelated ones like Tool or Food)
 * when no exact match exists, which reads as a bug rather than a feature
 * in a public demo.
 * ---------------------------------------------------------------------
 */
window.DeReLabData = (function () {
  "use strict";

  // Root-concept sampling weights, mirroring ROOT_CONCEPT_WEIGHTS in
  // generate_tree_inheritance.py.
  const ROOT_CONCEPT_WEIGHTS = {
    animal: 0.2,
    mammal: 0.15,
    vertebrate: 0.15,
    plant: 0.15,
    insect: 0.15,
    microorganism: 0.1,
    bird: 0.1,
  };

  // domain_attributes_values.json, trimmed to the seven tree-inheritance
  // root domains.
  const DOMAIN_ATTRIBUTES = {
    Animal: {
      behavior: { examples: ["Solitary", "Curious", "Energetic", "Friendly", "Gentle", "Playful", "Obedient", "Calm", "Nimble", "Wild", "Slow", "Fierce", "Agile", "Domesticated", "Cuddly", "Docile", "Territorial", "Active", "Brave", "Tame", "Feral", "Stray", "Captive", "Trained"], template: "are [value] by nature", object_template: "is [value] by nature" },
      capability: { examples: ["Fly", "Swim", "Echolocate", "Camouflage", "Regenerate", "Hibernate", "Produce Venom", "Climb", "Mimic Sounds", "Glide", "Sprint", "Change Color", "Spin Webs", "Slither", "Hover", "breathe"], template: "can [value]", object_template: "can [value]" },
      diet: { examples: ["Carnivore", "Herbivore", "Omnivore", "Insectivore", "Scavenger", "Frugivore", "Piscivore", "Detritivore", "Nectarivore", "Folivore"], template: "are [value] in diet", object_template: "is [value] in diet" },
      domestication_status: { examples: ["Domesticated", "Wild", "Feral", "Stray", "Captive", "Trained", "Exotic", "Untamed"], template: "are [value] in nature", object_template: "is [value] in nature" },
      habitat: { examples: ["Marine", "Freshwater", "Desert", "Tundra", "Tropical", "Savanna", "Alpine", "Subterranean", "Urban", "Wetland", "Coastal", "Boreal", "Temperate", "Estuarine"], template: "lives in [value] environments", object_template: "lives in a [value] environment" },
      location: { examples: ["the Wild", "Captivity", "the Northern Hemisphere", "the Southern Hemisphere", "the Tropics", "the Arctic", "the Antarctic", "the Canopy", "the Deep Sea", "the Amazon Basin", "the Indo-Pacific", "the Sub-Saharan Africa"], template: "are found in [value]", object_template: "is found in [value]" },
      physical_trait: { examples: ["Striped skin", "Fur", "Feathers", "Scale", "Tail", "Wing", "Claw", "Whiskers"], template: "have [value]", object_template: "has [value]" },
      scarcity: { examples: ["Abundant", "Rare", "Common", "Scarce", "Endangered", "Extinct", "Plentiful"], template: "are [value] in nature", object_template: "is [value] in nature" },
    },
    Mammal: {
      biology: { examples: ["Endothermic", "Viviparous", "Placental", "Monotreme", "Warm-blooded", "Lactating"], template: "are [value]", object_template: "is [value]" },
      covering: { examples: ["Furry", "Hairless", "Woolly", "Bristly", "Pelted", "Quilled", "Glandular"], template: "have a [value] covering", object_template: "has a [value] covering" },
      diet: { examples: ["Carnivorous", "Herbivorous", "Omnivorous", "Insectivorous", "Piscivorous", "Sanguivorous"], template: "are [value] in diet", object_template: "is [value] in diet" },
      locomotion: { examples: ["Digitigrade", "Plantigrade", "Unguligrade", "Bipedal", "Quadrupedal", "Arboreal", "Fossorial", "Volant", "Marine"], template: "utilize [value] locomotion", object_template: "utilizes [value] locomotion" },
    },
    Vertebrate: {
      skeleton: { examples: ["Bony", "Cartilaginous", "Endoskeletal", "Ossified", "Chondric"], template: "possess a [value] skeleton", object_template: "possesses a [value] skeleton" },
      anatomy: { examples: ["Spined", "Cranial", "Jawed", "Jawless", "Bilateral", "Tetrapodal"], template: "exhibit [value] anatomical traits", object_template: "exhibits [value] anatomical traits" },
      respiration: { examples: ["Pulmonary", "Branchial", "Cutaneous", "Bimodal"], template: "utilize [value] respiration", object_template: "utilizes [value] respiration" },
      blood_circulation: { examples: ["Single-loop", "Double-loop", "Closed"], template: "feature a [value] circulatory system", object_template: "features a [value] circulatory system" },
    },
    Insect: {
      development: { examples: ["Holometabolous", "Hemimetabolous", "Ametabolous", "Paurometabolous"], template: "undergo [value] development", object_template: "undergoes [value] development" },
      mouthpart: { examples: ["Chewing", "Piercing-sucking", "Siphoning", "Sponging", "Mandibulate", "Haustellate"], template: "possess [value] mouthparts", object_template: "possesses [value] mouthparts" },
      anatomy: { examples: ["Six-legged", "Antennaed", "Segmented", "Exoskeletal", "Compound-eyed", "Winged", "Apterous"], template: "feature [value] anatomy", object_template: "features [value] anatomy" },
      social_structure: { examples: ["Eusocial", "Solitary", "Gregarious", "Subsocial", "Parasitic"], template: "exhibit [value] behaviors", object_template: "exhibits [value] behaviors" },
    },
    Microorganism: {
      cellular_structure: { examples: ["Unicellular", "Multicellular", "Prokaryotic", "Eukaryotic", "Acellular"], template: "have a [value] cellular structure", object_template: "has a [value] cellular structure" },
      morphology: { examples: ["Coccus", "Bacillus", "Spirillum", "Vibrio", "Spirochete", "Pleomorphic", "Filamentous", "Icosahedral"], template: "exhibit a [value] morphology", object_template: "exhibits a [value] morphology" },
      metabolism: { examples: ["Aerobic", "Anaerobic", "Facultative", "Phototrophic", "Chemotrophic", "Autotrophic", "Heterotrophic"], template: "rely on [value] metabolism", object_template: "relies on [value] metabolism" },
      reproduction: { examples: ["Binary Fission", "Budding", "Spore-forming", "Conjugation", "Viral Replication", "Mitotic"], template: "utilize [value] for reproduction", object_template: "utilizes [value] for reproduction" },
    },
    Bird: {
      behavior: { examples: ["Curious", "Energetic", "Friendly", "Gentle", "Playful", "Obedient", "Calm", "Nimble", "Fierce", "Agile", "Solitary", "Domesticated", "Cuddly", "Docile", "Territorial", "Active", "Tame", "Feral", "Stray", "Captive"], template: "are [value] by nature", object_template: "is [value] by nature" },
      diet: { examples: ["Carnivore", "Herbivore", "Omnivore", "Insectivore", "Scavenger", "Frugivore", "Piscivore", "Detritivore", "Nectarivore", "Folivore"], template: "are [value] in diet", object_template: "is [value] in diet" },
      habitat: { examples: ["Forest", "Coastal", "Marine", "Urban", "Desert", "Tundra", "Alpine", "Tropical", "Scrubland", "Pelagic", "Woodland", "Estuarine"], template: "found in [value] environments", object_template: "found in a [value] environment" },
      physical_trait: { examples: ["Webbed", "Crested", "Taloned", "Tufted", "Long-legged", "Hook-billed", "Iridescent", "Fork-tailed", "Broad-winged", "Plumed", "Spurred", "Short-billed", "Straight-billed"], template: "have the feature of being [value]", object_template: "has the feature of being [value]" },
      scarcity: { examples: ["Extinct", "Abundant", "Rare", "Common", "Scarce", "Endangered", "Plentiful"], template: "are [value] in nature", object_template: "is [value] in nature" },
    },
    Plant: {
      habitat: { examples: ["Forest", "Wetland", "Coastal", "Marine", "Grassland", "Urban", "Desert", "Tundra", "Alpine", "Riparian", "Tropical", "Scrubland", "Pelagic", "Woodland", "Estuarine"], template: "thrive in [value] environments", object_template: "thrives in a [value] environment" },
      scarcity: { examples: ["Abundant", "Rare", "Common", "Scarce", "Endangered", "Extinct", "Plentiful"], template: "are [value] in the wild", object_template: "is [value] in the wild" },
      shape: { examples: ["Branching", "Oval", "Jagged", "Tapered"], template: "are [value] in shape", object_template: "is [value] in shape" },
      size: { examples: ["Tiny", "Massive", "Microscopic", "Monumental"], template: "are [value] in size", object_template: "is [value] in size" },
      weight: { examples: ["Light", "Heavy"], template: "are [value] in weight", object_template: "is [value] in weight" },
      texture: { examples: ["Smooth", "Rough", "Fuzzy", "Spiky"], template: "have a [value] texture", object_template: "has a [value] texture" },
      biology: { examples: ["Vascular", "Non-vascular"], template: "have [value] tissues", object_template: "has [value] tissues" },
      affordance: { examples: ["Edible", "Medicinal", "Toxic", "Biodegradable"], template: "are considered [value]", object_template: "is considered [value]" },
      capability: { examples: ["Photosynthesize", "Bloom", "Climb", "Regenerate"], template: "can [value]", object_template: "can [value]" },
    },
  };

  // non_sensical_entities.json word banks (180-word sample per domain,
  // sampled once with a fixed seed from the full 1000-word pool).
  const NONCE_ENTITIES = {
    Animal: ["Ammal", "Andatte", "Aniet", "Animasm", "Animaug", "Animelt", "Anivore", "Annamel", "Anory", "Beltens", "Berte", "Biryo", "Bivora", "Bovil", "Bovione", "Brupets", "Caliss", "Candal", "Captus", "Capur", "Cavione", "Colock", "Crets", "Crinse", "Crupig", "Dinake", "Dinale", "Dinet", "Dinvera", "Domele", "Donemal", "Dulate", "Dumale", "Dummate", "Einve", "Enale", "Enami", "Enaming", "Enese", "Ennamel", "Enormia", "Equad", "Equidat", "Farinel", "Farry", "Faunter", "Feerie", "Felfaux", "Fiste", "Fleptic", "Flogint", "Floolf", "Furva", "Gicarry", "Gicary", "Ginie", "Gonel", "Gonsur", "Goodog", "Gooter", "Gricary", "Grine", "Gritima", "Hogic", "Humal", "Humales", "Humals", "Hummal", "Hunam", "Inalle", "Inaml", "Inemal", "Ingele", "Ingets", "Inging", "Inimale", "Inimel", "Inmal", "Inmon", "Invishe", "Lamal", "Lamele", "Lanie", "Lapur", "Lapured", "Larey", "Larmia", "Launet", "Launt", "Lianic", "Licave", "Liete", "Lifeed", "Lific", "Listol", "Maleur", "Modone", "Monce", "Mutrie", "Mutrins", "Nomasm", "Nomeles", "Nonhum", "Nuadry", "Nuadult", "Nualle", "Numat", "Nummel", "Oenals", "Oenamo", "Oenat", "Oenatte", "Oenele", "Oenemal", "Oengult", "Oenome", "Oenonal", "Oentie", "Omals", "Omats", "Omels", "Omiant", "Omnimal", "Oretie", "Orseat", "Petock", "Phine", "Plary", "Prebrut", "Preed", "Priger", "Prossue", "Quadul", "Qualed", "Qualion", "Quall", "Qualy", "Quarmal", "Quidats", "Quinals", "Quinama", "Quinele", "Reptice", "Rescat", "Ritie", "Sacre", "Sandeer", "Seasm", "Seate", "Sechory", "Skinger", "Snale", "Stimal", "Storse", "Suelie", "Tabeed", "Tanomal", "Tebles", "Theef", "Thelly", "Thelty", "Thora", "Tivor", "Unamel", "Unamele", "Unarry", "Unels", "Varmic", "Verat", "Verte", "Vetigel", "Vetiger", "Welle", "Whance", "Whavist", "Wolic", "Wolle", "Wolood", "Younael", "Zooll"],
    Plant: ["Abien", "Ablut", "Actopic", "Anked", "Annuste", "Apain", "Apaines", "Apeck", "Apott", "Appower", "Aquad", "Aqual", "Aquitt", "Assub", "Autop", "Bieng", "Biest", "Blaing", "Blene", "Bletio", "Blins", "Bloren", "Blume", "Botatem", "Brecill", "Bureet", "Cenaten", "Centat", "Checory", "Cheds", "Chinet", "Clumbed", "Copitt", "Crall", "Crowerb", "Culter", "Culties", "Cultint", "Denus", "Depice", "Deplant", "Depower", "Deranty", "Deveg", "Devic", "Diolow", "Droph", "Drower", "Elwored", "Enuseed", "Facry", "Famen", "Finere", "Flotae", "Flotat", "Flotate", "Forante", "Foryo", "Gameck", "Gamenty", "Genatt", "Gente", "Gereed", "Getit", "Grached", "Graleg", "Grall", "Grang", "Grapal", "Grappot", "Grash", "Groele", "Grogerp", "Gropict", "Groweed", "Growees", "Hablut", "Heckint", "Heedle", "Heeleut", "Heops", "Hogen", "Hounty", "Imanty", "Imenatt", "Imental", "Imett", "Ingus", "Intat", "Intem", "Laate", "Lanarin", "Lantand", "Lanty", "Leatina", "Leatio", "Lerming", "Letal", "Leynt", "Macen", "Machint", "Macra", "Mante", "Mateaft", "Moinary", "Mointa", "Moned", "Myrming", "Namin", "Natital", "Nuatery", "Orear", "Orens", "Orestit", "Oroal", "Phyter", "Piett", "Plana", "Planeds", "Plante", "Planus", "Plear", "Pleneed", "Pleynt", "Plore", "Reatt", "Refint", "Renty", "Replang", "Replous", "Reveget", "Revir", "Setive", "Shutogy", "Smelwor", "Soility", "Soirub", "Sowery", "Spers", "Spoduce", "Stice", "Stiono", "Stits", "Stivar", "Stuce", "Sturic", "Tabitt", "Tation", "Teatt", "Toinatt", "Treae", "Trevens", "Trint", "Trinty", "Troger", "Trogy", "Troweed", "Trowint", "Undaned", "Undepor", "Unter", "Urefing", "Uriery", "Urint", "Vegent", "Vegenty", "Vegery", "Washe", "Wastat", "Wastion", "Weetive", "Weraft", "Werap", "Werow", "Worall", "Worent", "Wornate", "Wortal", "Wortio", "Wortion"],
  };
  // Non-Animal, non-Plant root concepts borrow the Animal word bank for
  // their descendant class labels (see module docstring above).
  ["Mammal", "Vertebrate", "Insect", "Microorganism", "Bird"].forEach((k) => {
    NONCE_ENTITIES[k] = NONCE_ENTITIES.Animal;
  });

  // Easy-tier preset variants, mirroring DIFFICULTY_PRESETS["easy"] in
  // generate_tree_inheritance.py.
  const EASY_PRESETS = [
    { name: "jellyfish", maxDepth: 4, rootBreadth: 5, minBranching: 1, maxBranching: 1, survivalProb: 0.4, branchingDecay: 1.0 },
    { name: "balanced", maxDepth: 3, rootBreadth: 2, minBranching: 2, maxBranching: 2, survivalProb: 1.0, branchingDecay: 1.0 },
    { name: "bushy", maxDepth: 2, rootBreadth: 3, minBranching: 1, maxBranching: 2, survivalProb: 0.7, branchingDecay: 1.0 },
  ];
  const EASY_SPARSITY_RANGE = [0.5, 0.8];

  return {
    ROOT_CONCEPT_WEIGHTS,
    DOMAIN_ATTRIBUTES,
    NONCE_ENTITIES,
    EASY_PRESETS,
    EASY_SPARSITY_RANGE,
  };
})();
