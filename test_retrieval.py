import argparse
import json
import os
import traceback

from ontomap.encoder.lightweight import IRILabelInLightweightEncoder
from ontomap.evaluation.evaluator import evaluator
from ontomap.ontology_matchers import MatcherCatalog
from ontomap.postprocess import process

DATASET_PATH = "datasets/test-small/ncit-doid/om.json"
DEVICE = "cuda"
TOP_K = 5

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Model name from MatcherCatalog['retrieval']")
args = parser.parse_args()

if args.model not in MatcherCatalog["retrieval"]:
    print(f"Unknown model '{args.model}'. Available: {list(MatcherCatalog['retrieval'].keys())}")
    exit(1)

# Load
print(f"Loading dataset...")
with open(DATASET_PATH, encoding="utf-8") as f:
    dataset = json.load(f)
print(f"Source: {len(dataset['source'])}  Target: {len(dataset['target'])}")

# Encode
print("Encoding inputs...")
encoder = IRILabelInLightweightEncoder()
encoded_inputs = encoder(**dataset)

# Run retrieval
print(f"Loading {args.model}...")
try:
    model_class = MatcherCatalog["retrieval"][args.model]
    model = model_class(top_k=TOP_K, device=DEVICE)

    print("Generating predictions...")
    predicts = model.generate(input_data=encoded_inputs)

    predicts = process.eval_preprocess_ir_outputs(predicts=predicts)
    results = evaluator(
        track="bio-ml",
        predicts=predicts,
        references=dataset["reference"],
    )

    print(f"\n{'='*60}")
    print(f"RESULTS — {args.model} on ncit-doid test subset")
    print(f"{'='*60}")
    for split_name, split_results in results.items():
        print(f"\n  {split_name}:")
        for metric, value in split_results.items():
            print(f"    {metric}: {value:.4f}")
    print(f"{'='*60}")

    # Save results
    RESULTS_PATH = "experiments/results/retrieval-test-results.json"
    existing = {}
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, encoding="utf-8") as f:
            existing = json.load(f)
    existing[args.model] = results
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

except Exception as e:
    print(f"Failed: {e}")
    traceback.print_exc()
