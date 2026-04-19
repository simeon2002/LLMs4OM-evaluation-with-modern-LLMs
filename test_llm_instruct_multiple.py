# -*- coding: utf-8 -*-
"""
Sanity check for instruct LLM: uses BERT retriever to fetch real top-k candidates
for a source concept, then runs the LLM on each to see how many get 'yes'.

Usage:
    python test_llm_instruct_multiple.py --model LLaMA3InstructDecoderLM
"""
import argparse
import importlib
import json
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="LLM class name from ontomap.ontology_matchers.rag.models")
parser.add_argument("--n-sources", type=int, default=20, help="Number of source concepts to test (default: 20)")
parser.add_argument("--top-k", type=int, default=5, help="Number of retriever candidates (default: 5)")
args = parser.parse_args()

# ── Load dataset ──────────────────────────────────────────────────────────────
DATASET_PATH = "datasets/test-small/ncit-doid/om.json"
print(f"Loading dataset from {DATASET_PATH}...")
with open(DATASET_PATH, encoding="utf-8") as f:
    dataset = json.load(f)

source_label_map = {s["iri"]: s["label"] for s in dataset["source"]}
target_label_map = {t["iri"]: t["label"] for t in dataset["target"]}

# Build ground truth: source IRI → set of correct target IRIs
reference = dataset["reference"]["equiv"]["test"]
gt_map = {}
for ref in reference:
    src = ref["source"]
    tgt = ref["target"]
    gt_map.setdefault(src, set()).add(tgt)

# ── Run BERT retriever ────────────────────────────────────────────────────────
print(f"Running BERTRetrieval (top_k={args.top_k}) on all source concepts...")
from ontomap.ontology_matchers.retrieval.models import BERTRetrieval
from ontomap.encoder.lightweight import IRILabelInLightweightEncoder

retriever = BERTRetrieval(top_k=args.top_k, device="cuda")
lightweight_encoder = IRILabelInLightweightEncoder()
encoded = lightweight_encoder(**dataset)  # [source_ontos, target_ontos]
retrieval_output = retriever.generate(input_data=encoded)

print(f"\nRunning LLM on first {args.n_sources} source concepts, top-{args.top_k} candidates each...\n")

# ── Load LLM ─────────────────────────────────────────────────────────────────
config = {
    "max_token_length": 1,
    "tokenizer_max_length": 500,
    "num_beams": 1,
    "device": "cuda",
    "truncation": True,
    "top_p": 0.95,
    "temperature": 0.8,
    "batch_size": 1,
    "padding": "max_length",
}

print(f"Loading {args.model}...")
rag_models = importlib.import_module("ontomap.ontology_matchers.rag.models")
if not hasattr(rag_models, args.model):
    print(f"ERROR: '{args.model}' not found in rag.models")
    sys.exit(1)

llm_class = getattr(rag_models, args.model)
llm = llm_class(**config)

if not llm.answer_sets_token_id["yes"] or not llm.answer_sets_token_id["no"]:
    print("WARNING: yes or no token IDs are empty!")
    sys.exit(1)

# ── Run LLM on all source concepts ────────────────────────────────────────────
total_yes = 0
total_candidates = 0
total_correct_in_topk = 0
total_correct_found = 0

for entry in retrieval_output[:args.n_sources]:
    source_iri = entry["source"]
    candidate_iris = entry["target-cands"]
    candidate_scores = entry["score-cands"]

    source_label = source_label_map.get(source_iri, source_iri)
    correct_iris = gt_map.get(source_iri, set())
    correct_in_topk = correct_iris & set(candidate_iris)

    print(f"\nSource: '{source_label}'")
    print(f"GT match(es) in top-{args.top_k}: {[target_label_map.get(i, i) for i in correct_in_topk] or 'none'}")
    print(f"{'Rank':<6} {'IR Score':<10} {'Candidate':<40} {'GT?':<6} {'Answer':<8} {'Confidence'}")
    print("-" * 85)

    yes_count = 0
    correct_found = 0
    for rank, (cand_iri, ir_score) in enumerate(zip(candidate_iris, candidate_scores), 1):
        cand_label = target_label_map.get(cand_iri, cand_iri)
        is_correct = cand_iri in correct_iris

        prompt = (
            "You are an ontology matching expert. Determine whether the following two concepts refer to the same real-world entity.\n\n"
            f"Concept 1: {source_label}\n"
            f"Concept 2: {cand_label}\n\n"
            "Answer with exactly one word: yes or no."
        )
        sequences, probas = llm.generate([prompt])
        answer = sequences[0]
        confidence = probas[0]

        if answer == "yes":
            yes_count += 1
        if is_correct and answer == "yes":
            correct_found += 1

        gt_str = "YES" if is_correct else ""
        print(f"{rank:<6} {ir_score:<10.4f} {cand_label:<40} {gt_str:<6} {answer:<8} {confidence:.4f}")

    print("-" * 85)
    print(f"Yes answers: {yes_count}/{len(candidate_iris)} | Correct matches found: {correct_found}/{len(correct_in_topk)}")

    total_yes += yes_count
    total_candidates += len(candidate_iris)
    total_correct_in_topk += len(correct_in_topk)
    total_correct_found += correct_found

print(f"\n{'='*85}")
print(f"SUMMARY over {args.n_sources} source concepts:")
print(f"  Total 'yes' answers : {total_yes} / {total_candidates}")
print(f"  Correct matches found by LLM: {total_correct_found} / {total_correct_in_topk}")
print(f"  Avg yes per source  : {total_yes / args.n_sources:.2f} / {args.top_k}")
print(f"{'='*85}")
