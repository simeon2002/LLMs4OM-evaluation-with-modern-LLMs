# -*- coding: utf-8 -*-
"""
Quick sanity check for instruct/chat RAG LLM decoder classes.
Uses instruct-style prompt (no ### Answer: completion style).

Usage:
    python test_llm_instruct.py --model LLaMA3InstructDecoderLM
    python test_llm_instruct.py --model Qwen35_9BInstructDecoderLM
"""
import argparse
import importlib
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="LLM class name from ontomap.ontology_matchers.rag.models")
args = parser.parse_args()

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

print(f"Tokenizer padding_side: {llm.tokenizer.padding_side}")
print(f"Tokenizer pad_token: {llm.tokenizer.pad_token}")
print(f"Tokenizer eos_token: {llm.tokenizer.eos_token}")
print(f"Tokenizer bos_token: {llm.tokenizer.bos_token}")
print(f"Vocab size: {llm.tokenizer.vocab_size}")

print("\nyes/no token IDs found:")
print(f"  yes set: {llm.answer_sets_token_id['yes']}")
print(f"  no  set: {llm.answer_sets_token_id['no']}")

for yes_token_id in llm.answer_sets_token_id['yes']:
    yes_tokens = llm.tokenizer.convert_ids_to_tokens(yes_token_id)
    print(f"  yes token ID {yes_token_id} corresponds to tokens: {yes_tokens}")

for no_token_id in llm.answer_sets_token_id['no']:
    no_tokens = llm.tokenizer.convert_ids_to_tokens(no_token_id)
    print(f"  no token ID {no_token_id} corresponds to tokens: {no_tokens}")

if not llm.answer_sets_token_id["yes"] or not llm.answer_sets_token_id["no"]:
    print("\nWARNING: yes or no token IDs are empty — check tokenizer vocab!")
    sys.exit(1)

prompt = (
    "You are an ontology matching expert. Determine whether the following two concepts refer to the same real-world entity.\n\n"
    "Concept 1: malignant neoplasm of lung\n"
    "Concept 2: lung cancer\n\n"
    "Answer with exactly one word: yes or no."
)

print(f"\nTest prompt:\n{prompt}")
sequences, probas = llm.generate([prompt])
print(f"\nAnswer: {sequences[0]}  (confidence: {probas[0]:.4f})")
print("\nSanity check passed!")
