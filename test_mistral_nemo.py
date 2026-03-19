# -*- coding: utf-8 -*-
"""
Quick sanity check for any RAG LLM decoder class:
- Verifies tokenizer loads with correct padding settings
- Verifies yes/no tokens are found in the vocab
- Runs a single yes/no prompt

Usage:
    python test_mistral_nemo.py --model MistralNemoDecoderLM
    python test_mistral_nemo.py --model Qwen25_7BDecoderLM
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
print(f"Vocab size: {llm.tokenizer.vocab_size}")

print("\nyes/no token IDs found:")
print(f"  yes set: {llm.answer_sets_token_id['yes']}")
print(f"  no  set: {llm.answer_sets_token_id['no']}")

# IMPORTANT CHECK: yes/no tokens must tokenize to a single token (max_token_length=1)
if not llm.answer_sets_token_id["yes"] or not llm.answer_sets_token_id["no"]:
    print("\nWARNING: yes or no token IDs are empty — check tokenizer vocab!")
    sys.exit(1)

prompt = (
    "Do the following two concepts refer to the same thing? "
    "Source: 'Malignant neoplasm of lung'. Target: 'lung cancer'. "
    "Answer yes or no.\n"
)

print(f"\nTest prompt: {prompt}")
sequences, probas = llm.generate([prompt])
print(f"Answer: {sequences[0]}  (confidence: {probas[0]:.4f})")
print("\nSanity check passed!")
