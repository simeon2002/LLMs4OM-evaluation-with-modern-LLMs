import argparse
import json
import os
import traceback

from ontomap.encoder.rag import IRILabelInRAGEncoder
from ontomap.evaluation.evaluator import evaluator
from ontomap.ontology_matchers import MatcherCatalog
from ontomap.postprocess import process

DATASET_PATH = "datasets/test-small/ncit-doid/om.json"
DEVICE = "cuda"
BATCH_SIZE = 32

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Model name from MatcherCatalog['rag']")
args = parser.parse_args()

if args.model not in MatcherCatalog["rag"]:
    print(f"Unknown model '{args.model}'. Available: {list(MatcherCatalog['rag'].keys())}")
    exit(1)

# Load
print("Loading dataset...")
with open(DATASET_PATH, encoding="utf-8") as f:
    dataset = json.load(f)
print(f"Source: {len(dataset['source'])}  Target: {len(dataset['target'])}")

# Encode
print("Encoding inputs...")
encoder = IRILabelInRAGEncoder()
encoded_inputs = encoder(**dataset)

# Build config
model_config = {
    "retriever-config": {"top_k": 5, "device": DEVICE},
    "llm-config": {
        "max_token_length": 1,
        "tokenizer_max_length": 500,
        "num_beams": 1,
        "device": DEVICE,
        "truncation": True,
        "top_p": 0.95,
        "temperature": 0.8,
        "batch_size": BATCH_SIZE,
        "padding": "max_length",
    },
    "nshots": None,
}

# Run RAG pipeline
print(f"Loading {args.model}...")
try:
    model_class = MatcherCatalog["rag"][args.model]
    model = model_class(**model_config)

    print("Generating predictions...")
    predicts = model.generate(input_data=encoded_inputs)

    predicts, _ = process.postprocess_hybrid(
        predicts=predicts,
        llm_confidence_th=0.7,
        ir_score_threshold=0.9,
    )

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
    RESULTS_PATH = "experiments/results/rag-test-results.json"
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
