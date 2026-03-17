import copy
import json
import os

MAX_SOURCE = 50
INPUT_PATH = os.path.join("datasets", "bio-ml", "ncit-doid.disease", "om.json")
OUTPUT_PATH = os.path.join("datasets", "test-small", "ncit-doid", "om.json")

print(f"Loading {INPUT_PATH} ...")
with open(INPUT_PATH) as f:
    dataset = json.load(f)

print(f"Full source concepts : {len(dataset['source'])}")
print(f"Full target concepts : {len(dataset['target'])}")

# fetch all iris of the sources present in the equiv reference alignment only.
ref_source_iris = set()
for split in dataset["reference"]["equiv"]:
    for pair in dataset["reference"]["equiv"][split]:
        ref_source_iris.add(pair["source"])

print(f"Source IRIs with references: {len(ref_source_iris)}")

# Select max_sources amount that are
selected_sources = [s for s in dataset["source"] if s["iri"] in ref_source_iris][:MAX_SOURCE]
selected_iris = {s["iri"] for s in selected_sources}

print(f"Selected {len(selected_sources)} source concepts")

# ── Filter reference to only selected sources
filtered_reference = {}
for ref_type in dataset["reference"]:
    filtered_reference[ref_type] = {}
    for split in dataset["reference"][ref_type]:
        filtered_reference[ref_type][split] = [
            pair for pair in dataset["reference"][ref_type][split]
            if pair["source"] in selected_iris
        ]

# Build output dataset 
subset = {
    "dataset-info": copy.deepcopy(dataset["dataset-info"]),
    "source": selected_sources,
    "target": dataset["target"],   # keep all targets
    "reference": filtered_reference,
}
 
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(subset, f, indent=2)

print(f"\nSaved test subset to {OUTPUT_PATH}")
print(f"  source  : {len(subset['source'])}")
print(f"  target  : {len(subset['target'])}")
for ref_type in subset["reference"]:
    for split, pairs in subset["reference"][ref_type].items():
        print(f"  reference[{ref_type}][{split}]: {len(pairs)} pairs")
