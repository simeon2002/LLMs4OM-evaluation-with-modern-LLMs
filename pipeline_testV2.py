from ontomap.ontology.bioml import SNOMEDNCITPharmOMDataset
from ontomap.encoder import IRILabelInRAGEncoder
from ontomap.ontology_matchers.rag.models import MistralLLMBertRAG
from ontomap.evaluation.evaluator import evaluator
from ontomap.postprocess import process
import json
import os
import traceback
from datetime import datetime

# =============================================
# 1. CONFIG
# =============================================
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
        "batch_size": 64,
        "padding": "max_length"
    },
    "nshots": 0
}

# =============================================
# 2. LOAD DATASET
# =============================================
print("Loading snomed-ncit-pharm dataset...")
ontology = SNOMEDNCITPharmOMDataset().collect(root_dir="datasets")
print(f"Source: {len(ontology['source'])}, Target: {len(ontology['target'])}")

# =============================================
# 3. ENCODE
# =============================================
print("Encoding inputs...")
encoded_inputs = IRILabelInRAGEncoder()(**ontology)

# =============================================
# 4. RUN MODEL
# =============================================
print("Loading MistralLLMBertRAG...")
try:
    model = MistralLLMBertRAG(**model_config)

    print("Generating predictions...")
    predicts = model.generate(input_data=encoded_inputs)
    print(f"Generated {len(predicts)} prediction groups")

    # =============================================
    # 5. SAVE RAW PREDICTIONS
    # =============================================
    timestamp = datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
    output_dir = "experiments/output-full-MistralBertRAG/bio-ml/snomed-ncit.pharm"
    os.makedirs(output_dir, exist_ok=True)

    raw_path = os.path.join(output_dir, f"rag-MistralBertRAG-label-{timestamp}.json")
    with open(raw_path, 'w') as f:
        json.dump({"generated-output": predicts}, f, indent=4)
    print(f"Saved raw predictions to {raw_path}")

    # =============================================
    # 6. POST-PROCESS (original thresholds)
    # =============================================
    print("Post-processing (ir=0.9, llm=0.7)...")
    predicts, _ = process.postprocess_hybrid(
        predicts=predicts,
        llm_confidence_th=0.7,
        ir_score_threshold=0.9
    )
    print(f"Post-processed: {len(predicts)} matches")

    # =============================================
    # 7. EVALUATE
    # =============================================
    print("Evaluating...")
    results = evaluator(
        track='bio-ml',
        predicts=predicts,
        references=ontology["reference"]
    )

    print(f"\n{'='*60}")
    print(f"RESULTS: MistralBertRAG on omim-ordo (FULL)")
    print(f"{'='*60}")
    for split_name, split_results in results.items():
        print(f"\n  {split_name}:")
        for metric, value in split_results.items():
            print(f"    {metric}: {value:.4f}")
    print(f"{'='*60}")

    # Save evaluation results
    eval_path = os.path.join(output_dir, f"eval-{timestamp}.json")
    with open(eval_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved evaluation to {eval_path}")

except Exception as e:
    print(f"✗ Failed: {str(e)}")
    traceback.print_exc()