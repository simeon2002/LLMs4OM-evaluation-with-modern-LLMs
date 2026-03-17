"""
Standalone test for Qwen3-Embedding-0.6B retrieval on the small ncit-doid test subset.

Run scripts/create_test_dataset.py first to generate the subset.

Usage:
    python scripts/create_test_dataset.py   # once
    python test_qwen3_retrieval.py
"""

import traceback

from ontomap.encoder.lightweight import IRILabelInLightweightEncoder
from ontomap.evaluation.evaluator import evaluator
from ontomap.ontology_matchers.retrieval.models import Qwen3EmbeddingRetrieval
from ontomap.postprocess import process
from ontomap.utils import io

DATASET_PATH = "datasets/test-small/ncit-doid/om.json"
DEVICE = "cuda"
TOP_K = 5

# ── Load ──────────────────────────────────────────────────────────────────────
print(f"Loading {DATASET_PATH} ...")
dataset = io.read_json(DATASET_PATH)
print(f"Source: {len(dataset['source'])}  Target: {len(dataset['target'])}")

# ── Encode ────────────────────────────────────────────────────────────────────
print("Encoding inputs...")
encoder = IRILabelInLightweightEncoder()
encoded_inputs = encoder(**dataset)

# ── Run retrieval ─────────────────────────────────────────────────────────────
print("Loading Qwen3-Embedding-0.6B...")
try:
    model = Qwen3EmbeddingRetrieval(top_k=TOP_K, device=DEVICE)

    print("Generating predictions...")
    predicts = model.generate(input_data=encoded_inputs)

    predicts = process.eval_preprocess_ir_outputs(predicts=predicts)
    results = evaluator(
        track="bio-ml",
        predicts=predicts,
        references=dataset["reference"],
    )

    print(f"\n{'='*60}")
    print(f"RESULTS — Qwen3EmbeddingRetrieval on ncit-doid test subset")
    print(f"{'='*60}")
    for split_name, split_results in results.items():
        print(f"\n  {split_name}:")
        for metric, value in split_results.items():
            print(f"    {metric}: {value:.4f}")
    print(f"{'='*60}")

except Exception as e:
    print(f"Failed: {e}")
    traceback.print_exc()
