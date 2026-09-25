"""Label vocabularies shared by the engine and both front-ends."""

# datasets/build_metadata.py: 0=normal, 1=fire/smoke, 2=collapse/flood, 3=other_disaster
DISASTER_CLASSES = ["normal", "fire / smoke", "collapse / flood", "other disaster"]
VICTIM_CLASSES = ["no victim", "victim present"]
# models/full_model.py NuisanceAwareHead.NUISANCE_CLASSES
NUISANCE_CLASSES = [
    "clean", "solar heating", "RGB false fire", "audio false alarm", "partial occlusion",
]


# Disaster-relevant AudioSet classes (index -> display group). Indices follow
# weights/audioset_labels.csv (official AudioSet class_labels_indices.csv).
AUDIOSET_RELEVANT = {
    0: "voice", 8: "voice", 11: "voice", 13: "voice", 14: "distress",
    22: "distress",
    283: "weather", 286: "weather", 287: "weather", 288: "water", 289: "weather",
    293: "water",
    298: "fire", 299: "fire",
    310: "alarm", 323: "siren", 324: "siren", 325: "siren",
    335: "aircraft", 336: "aircraft", 339: "aircraft",
    343: "engine",
    388: "alarm", 396: "siren", 397: "siren", 399: "alarm", 400: "alarm",
    426: "explosion", 441: "collapse", 469: "collapse",
}
