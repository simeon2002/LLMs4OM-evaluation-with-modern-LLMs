'''
This script tests the full pipeline on a small subset of the ncit-doid dataset, using the MistralBertRAG model.
in order for this to work, truncate once again the reference files (ncit-doid) to only include matches for the first 50 source concepts that have references. This ensures we are evaluating on a manageable subset while still testing the full pipeline end-to-end.
It performs the following steps:
1. Loads the dataset from JSON.
2. Applies smart truncation to select only source concepts that have references, up to a maximum of 50.
3. Encodes the inputs using IRILabelInRAGEncoder.
4. Loads the MistralBertRAG model and generates predictions.
5. Post-processes the predictions using the original thresholds (IR score >= 0.9, LLM confidence >= 0.7).
6. Evaluates the results against the reference matches for the selected sources.
7. Prints the evaluation results.'''

from ontomap.ontology import ontology_matching
from ontomap.ontology_matchers.rag.models import MistralLLMBertRAG
from ontomap.encoder import IRILabelInRAGEncoder
from ontomap.evaluation.evaluator import evaluator
from ontomap.postprocess import process
from ontomap.utils import io
import os
import copy
import traceback

# Load dataset
dataset_class = ontology_matching['bio-ml'][0]
config_root = "datasets"

print("Loading dataset from JSON...")
om_json_path = os.path.join(config_root, dataset_class.working_dir, "om.json")
dataset = io.read_json(om_json_path)

original_dataset = copy.deepcopy(dataset)

print(f"Full dataset source concepts: {len(dataset['source'])}")
print(f"Full dataset target concepts: {len(dataset['target'])}")

# =============================================
# SMART TRUNCATION: Pick sources that actually 
# have matching targets in the reference
# =============================================
MAX_SOURCE = 50

# Collect source IRIs that appear in any reference
ref_source_iris = set()
for ref_type in dataset['reference']:
    for split in dataset['reference'][ref_type]:
        for pair in dataset['reference'][ref_type][split]:
            ref_source_iris.add(pair['source'])

print(f"Source IRIs with references: {len(ref_source_iris)}")

# Filter source concepts to only those with references
source_with_refs = [s for s in dataset['source'] if s['iri'] in ref_source_iris]
source_without_refs = [s for s in dataset['source'] if s['iri'] not in ref_source_iris]

print(f"Source concepts with references: {len(source_with_refs)}")

# Take first MAX_SOURCE that have references
dataset['source'] = source_with_refs[:MAX_SOURCE]
# Keep ALL targets
print(f"Selected {len(dataset['source'])} source concepts (with refs), keeping all {len(dataset['target'])} targets")

# Encode
print("\nEncoding inputs...")
encoder = IRILabelInRAGEncoder()
encoded_inputs = encoder(**dataset)

# Model config (same as original study)
model_config = {
    "retriever-config": { 
        "top_k": 5,
        "device": "cuda"
    },
    "llm-config": {
        "max_token_length": 1,
        "tokenizer_max_length": 500,
        "num_beams": 1,
        "device": "cuda",
        "truncation": True,
        "top_p": 0.95,
        "temperature": 0.8,
        "batch_size": 16,
        "padding": "max_length"
    },
    "nshots": 0
}

print("\nLoading MistralBertRAG model...")
try:
    model = MistralLLMBertRAG(**model_config)

    print("Generating predictions...")
    predicts = model.generate(input_data=encoded_inputs)
    print(f"Generated {len(predicts)} prediction groups")

    # Post-process with ORIGINAL thresholds
    print("\nPost-processing (original thresholds: ir=0.9, llm=0.7)...")
    predicts, _ = process.postprocess_hybrid(
        predicts,
        llm_confidence_th=0.7,
        ir_score_threshold=0.9
    )
    print(f"Post-processed predictions: {len(predicts)} matches")

    # Evaluate against ONLY the references for our selected sources
    print("\nEvaluating results...")
    results = evaluator(
        track='bio-ml',
        predicts=predicts,
        references=original_dataset["reference"]
    )

    print(f"\n{'='*60}")
    print(f"RESULTS for MistralBertRAG on ncit-doid ({MAX_SOURCE} sources with refs)")
    print(f"{'='*60}")
    for split_name, split_results in results.items():
        print(f"\n  {split_name}:")
        for metric, value in split_results.items():
            print(f"    {metric}: {value:.4f}")
    print(f"{'='*60}")

except Exception as e:
    print(f"✗ Failed: {str(e)}")
    traceback.print_exc()