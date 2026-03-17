import json
import traceback

from ontomap.encoder.rag import IRILabelInRAGEncoder
from ontomap.evaluation.evaluator import evaluator
from ontomap.ontology_matchers.rag.models import LLaMA3Qwen3RAG
from ontomap.postprocess import process

DATASET_PATH = "datasets/test-small/ncit-doid/om.json"
DEVICE = "cuda"
BATCH_SIZE = 16

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
print("Loading LLaMA3 + Qwen3-Embedding-0.6B RAG...")
try:
    model = LLaMA3Qwen3RAG(**model_config)

    print("Generating predictions...")
    predicts = model.generate(input_data=encoded_inputs)

    # Post-process
    predicts, _ = process.postprocess_hybrid(
        predicts=predicts,
        llm_confidence_th=0.7,
        ir_score_threshold=0.9,
    )

    # Evaluate
    results = evaluator(
        track="bio-ml",
        predicts=predicts,
        references=dataset["reference"],
    )

    print(f"\n{'='*60}")
    print(f"RESULTS — LLaMA3Qwen3RAG on ncit-doid test subset")
    print(f"{'='*60}")
    for split_name, split_results in results.items():
        print(f"\n  {split_name}:")
        for metric, value in split_results.items():
            print(f"    {metric}: {value:.4f}")
    print(f"{'='*60}")

except Exception as e:
    print(f"Failed: {e}")
    traceback.print_exc()
