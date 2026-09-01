import random

animal_properties = [
    "hasFur",
    "hasFeathers",
    "hasScales",
    "isWarmBlooded",
    "isColdBlooded",
    "canFly",
    "canSwim",
    "canRunFast",
    "hasSharpTeeth",
    "hasClaws",
    "hasWings",
    "hasTail",
    "hasNightVision",
    "isNocturnal",
    "isDiurnal",
    "isCarnivorous",
    "isHerbivorous",
    "isOmnivorous",
    "isDomesticated",
    "isWild",
    "isSocial",
    "isSolitary",
    "hasCamouflage",
    "isTerritorial",
    "hasKeenSmell",
    "hasGoodHearing",
    "isEndangered",
    "hasMigrationBehavior",
    "hasVenom",
    "hasHorns",
    "hasHooves",
    "hasGills",
    "laysEggs",
    "givesLiveBirth",
    "hasComplexCommunication"
]


def get_random_animal_properties(num_properties=5):
    return random.sample(animal_properties, num_properties)