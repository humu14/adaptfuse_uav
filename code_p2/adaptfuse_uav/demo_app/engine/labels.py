"""Label vocabularies shared by the engine and both front-ends."""

# datasets/build_metadata.py: 0=normal, 1=fire/smoke, 2=collapse/flood, 3=other_disaster
DISASTER_CLASSES = ["normal", "fire / smoke", "collapse / flood", "other disaster"]
VICTIM_CLASSES = ["no victim", "victim present"]
# models/full_model.py NuisanceAwareHead.NUISANCE_CLASSES
NUISANCE_CLASSES = [
    "clean", "solar heating", "RGB false fire", "audio false alarm", "partial occlusion",
]


# Zero-shot hazard vocabulary (engine/hazard.py). Each event maps to one DISASTER_CLASSES index;
# its text embedding is the mean over prompts x templates (built by scripts/export_hazard.py).
HAZARD_TEMPLATES = ["a photo of {}.", "an aerial photo of {}.", "a drone photo of {}.", "a news photo of {}."]
HAZARD_EVENTS = [
    {"name": "normal", "coarse": 0, "prompts": [
        "a city street with intact buildings", "a quiet residential neighborhood", "a green forest",
        "a dense pine forest seen from above", "treetops of a forest", "farmland and fields",
        "a calm river", "a lake surrounded by trees", "a highway with normal traffic", "a park with trees",
        "mountains and a valley on a clear day", "a parking lot", "a town seen from above",
        "people walking outdoors"]},
    {"name": "fire", "coarse": 1, "prompts": [
        "a wildfire with flames", "a burning building", "flames and fire", "a forest fire burning trees",
        "a house on fire", "a car on fire"]},
    {"name": "smoke", "coarse": 1, "prompts": [
        "thick smoke rising into the sky", "a large plume of smoke from a fire", "smoke over a forest",
        "black smoke billowing from a building"]},
    {"name": "explosion", "coarse": 1, "prompts": [
        "an explosion with a fireball", "a huge fireball and a mushroom cloud of smoke",
        "an explosion at an industrial plant"]},
    {"name": "collapsed building", "coarse": 2, "prompts": [
        "a collapsed building", "buildings destroyed by an earthquake", "rubble and debris of a collapsed building",
        "a pile of concrete rubble", "a destroyed house", "ruins of buildings after a disaster"]},
    {"name": "flood", "coarse": 2, "prompts": [
        "a flooded area", "flooded streets and houses", "a river overflowing with muddy flood water",
        "a flash flood with rushing brown water", "houses surrounded by flood water"]},
    {"name": "landslide", "coarse": 2, "prompts": [
        "a landslide", "a mudslide destroying houses on a hillside", "a hillside collapse with mud and debris"]},
    {"name": "traffic accident", "coarse": 3, "prompts": [
        "a car crash", "a traffic accident on a road", "a vehicle collision with damaged cars",
        "an overturned truck on a highway", "a crashed car on the road"]},
]
HAZARD_NAMES = [e["name"] for e in HAZARD_EVENTS]
# Events drawn as image regions; fire / smoke regions come from the fire/smoke detector instead.
ZONE_EVENTS = {"collapsed building": "collapsed building", "flood": "flood water",
               "landslide": "landslide", "traffic accident": "accident"}

# COCO classes kept from the person/vehicle detector (index -> name)
COCO_KEEP = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 8: "boat"}

# Sound events: AudioSet classes (weights/audioset_labels.csv indices) pooled into the groups
# shown by the app. Group score = max over its classes. "hazard" groups can raise an alert;
# the others describe the background. "victim" marks human-distress cues.
SOUND_GROUPS = {
    "fire crackling":     {"hazard": True,  "idx": [298, 299]},
    "explosion / blast":  {"hazard": True,  "idx": [426, 427, 430, 432, 433, 434, 436, 466]},
    "siren / alarm":      {"hazard": True,  "idx": [322, 323, 324, 325, 388, 396, 397, 399, 400, 310]},
    "screaming / crying": {"hazard": True,  "idx": [8, 9, 11, 13, 14, 22, 23, 24, 25, 38], "victim": True},
    "crash / collapse":   {"hazard": True,  "idx": [358, 440, 443, 460, 469, 470, 478, 493]},
    "rushing water":      {"hazard": True,  "idx": [288, 292, 293, 294, 295, 297, 445]},
    "rain / thunder":     {"hazard": True,  "idx": [286, 287, 289, 290, 291]},
    "helicopter / drone": {"hazard": False, "idx": [335, 336, 337, 338, 339, 340]},
    "vehicles / traffic": {"hazard": False, "idx": [300, 301, 304, 306, 307, 308, 312, 313, 314, 316, 318,
                                                     326, 327, 329, 343, 348, 349]},
    "wind":               {"hazard": False, "idx": [283, 284, 285]},
    "speech":             {"hazard": False, "idx": [0, 1, 2, 3, 4, 5, 70]},
    "music":              {"hazard": False, "idx": [137]},
}
