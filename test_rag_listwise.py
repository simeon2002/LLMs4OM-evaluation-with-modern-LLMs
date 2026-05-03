"""
Standalone test for listwise ranking + RRF RAG approach.

Runs BERT retrieval to get top-k candidates per source concept, then sends
one listwise prompt per source to an LLM, parses the ranking, applies RRF,
and reports results vs IR-only baseline.

Usage:
    python test_rag_listwise.py --model LLaMA3ListwiseBertRAG
                                --dataset ncit-doid   (or omim-ordo)
                                --n-sources 20
                                --top-k 5
"""
import argparse
import json
import os
import sys
from pathlib import Path

from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from ontomap.ontology_matchers.rag.dataset_listwise import LabelListwiseRAGDataset
from ontomap.ontology_matchers.rag.rag_listwise import (
    LLaMA3ListwiseBertRAG,
    Gemma4ListwiseBertRAG,
    apply_rrf,
    parse_ranking,
)
from ontomap.ontology_matchers.retrieval.models import BERTRetrieval
from ontomap.encoder.lightweight import IRILabelInLightweightEncoder

MODELS = {
    "LLaMA3ListwiseBertRAG": LLaMA3ListwiseBertRAG,
    "Gemma4ListwiseBertRAG": Gemma4ListwiseBertRAG,
}

DATASETS = {
    "ncit-doid": "datasets/bio-ml/ncit-doid.disease/om.json",
    "omim-ordo": "datasets/bio-ml/omim-ordo.disease/om.json",
}


def load_dataset(path: str):
    with open(path) as f:
        return json.load(f)


def build_iri_index(onto: list) -> dict:
    return {concept["iri"]: i for i, concept in enumerate(onto)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--dataset", default="ncit-doid", choices=list(DATASETS.keys()))
    parser.add_argument("--n-sources", type=int, default=300)
    parser.add_argument("--no-per-source", action="store_true", help="suppress per-source output")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    device = "cuda"
    dataset_path = DATASETS[args.dataset]
    ds = load_dataset(dataset_path)

    source_onto = ds["source"]
    target_onto = ds["target"]
    reference = {pair["source"]: pair["target"] for pair in ds["reference"]["equiv"]["full"]}

    s2i = build_iri_index(source_onto)
    t2i = build_iri_index(target_onto)

    # ── IR retrieval ─────────────────────────────────────────────────────────
    print("Running BERT retrieval...")
    retriever_kwargs = {
        "device": device,
        "top_k": args.top_k,
        "batch_size": 32,
        "truncation": True,
        "padding": True,
        "max_length": 128,
    }
    retriever = BERTRetrieval(**retriever_kwargs)
    retriever.load()

    encoder = IRILabelInLightweightEncoder()
    task_args = {"source": source_onto, "target": target_onto}
    encoded = encoder(**task_args)
    ir_output = retriever.generate(input_data=encoded)
    # ir_output: list of {"source": iri, "target-cands": [...], "score-cands": [...]}

    # ── Load LLM ─────────────────────────────────────────────────────────────
    print(f"Loading LLM ({args.model})...")
    llm_kwargs = {
        "device": device,
        "batch_size": 1,
        "truncation": True,
        "padding": True,
        "tokenizer_max_length": 2048,
        "max_token_length": 40,
        "num_beams": 1,
        "temperature": 1.0,
        "top_p": 1.0,
    }
    ModelClass = MODELS[args.model]
    # Instantiate the ListwiseRAG to get the LLM (we use LLM directly here)
    rag_kwargs = {
        "device": device,
        "retriever-config": retriever_kwargs,
        "llm-config": llm_kwargs,
    }
    model = ModelClass(**rag_kwargs)
    llm = model.LLM

    # ── Build listwise inputs for first n sources ─────────────────────────────
    ir_subset = ir_output[:args.n_sources]

    listwise_inputs = []
    for item in ir_subset:
        src_iri = item["source"]
        src_concept = source_onto[s2i[src_iri]]
        tgt_concepts = [target_onto[t2i[iri]] for iri in item["target-cands"]]
        listwise_inputs.append({
            "source": src_concept,
            "targets": tgt_concepts,
            "source_iri": src_iri,
            "target_iris": item["target-cands"],
            "ir_scores": item["score-cands"],
        })

    dataset = LabelListwiseRAGDataset(data=listwise_inputs)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    # ── Run and display ───────────────────────────────────────────────────────
    SEP = "=" * 95
    ir_correct = 0
    rrf_correct = 0
    gt_in_topk_count = 0
    ir_correct_of_topk = 0
    rrf_correct_of_topk = 0
    rrf_disagrees_count = 0
    improvement_cases = []
    llm_rank1_correct = 0             # LLM ranked GT first (LLM alone would be correct)
    ir_wrong_llm_right = 0            # IR wrong but LLM ranked GT first
    ir_wrong_llm_right_rrf_wrong = 0  # LLM was right but RRF still failed
    ir_right_llm_wrong_rrf_right = 0  # IR right, LLM wrong, but RRF still correct (IR weight saved it)

    # build a lookup from source_iri → listwise_input item for per-source display
    item_by_iri = {item["source_iri"]: item for item in listwise_inputs}

    print(f"\n{SEP}")
    for batch in tqdm(dataloader):
        texts = batch["texts"]
        source_iris = batch["source_iris"]
        target_iris_list = batch["target_iris"]
        ir_scores_list = batch["ir_scores"]

        ranking_texts = llm.generate(texts)

        for src_iri, target_iris, ir_scores, ranking_text in zip(
            source_iris, target_iris_list, ir_scores_list, ranking_texts
        ):
            item = item_by_iri[src_iri]
            src_label = item["source"]["label"]
            gt_target = reference.get(src_iri)

            n = len(target_iris)
            llm_ranking = parse_ranking(ranking_text, n)
            rrf_scores = apply_rrf(ir_scores, llm_ranking)
            best_rrf_idx = max(range(n), key=lambda i: rrf_scores[i])

            ir_best_iri = target_iris[0]
            rrf_best_iri = target_iris[best_rrf_idx]

            if best_rrf_idx != 0:
                rrf_disagrees_count += 1
            gt_in_topk = (gt_target is not None) and (gt_target in target_iris)
            ir_hit = (gt_target is not None) and (ir_best_iri == gt_target)
            rrf_hit = (gt_target is not None) and (rrf_best_iri == gt_target)
            if gt_in_topk:
                gt_in_topk_count += 1
                if ir_hit:
                    ir_correct_of_topk += 1
                if rrf_hit:
                    rrf_correct_of_topk += 1
            if ir_hit:
                ir_correct += 1
            if rrf_hit:
                rrf_correct += 1
            if rrf_hit and not ir_hit:
                improvement_cases.append(src_label)

            # LLM-specific counters (only meaningful when GT is in top-k)
            if gt_in_topk:
                llm_top1_iri = target_iris[llm_ranking[0]]
                llm_hit = (llm_top1_iri == gt_target)
                if llm_hit:
                    llm_rank1_correct += 1
                if not ir_hit and llm_hit:
                    ir_wrong_llm_right += 1
                    if not rrf_hit:
                        ir_wrong_llm_right_rrf_wrong += 1
                if ir_hit and not llm_hit and rrf_hit:
                    ir_right_llm_wrong_rrf_right += 1

            if not args.no_per_source:
                llm_rank_of = [0] * n
                for rank, idx in enumerate(llm_ranking):
                    llm_rank_of[idx] = rank + 1
                rrf_order = sorted(range(n), key=lambda i: rrf_scores[i], reverse=True)
                rrf_rank_of = [0] * n
                for rank, idx in enumerate(rrf_order):
                    rrf_rank_of[idx] = rank + 1

                gt_display = [item["targets"][i]["label"] for i, iri in enumerate(target_iris) if iri == gt_target]
                gt_in_topk_str = f"'{gt_display[0]}'" if gt_display else "NOT IN TOP-K"
                ir_result  = "CORRECT" if ir_hit  else ("WRONG" if gt_in_topk else "GT NOT IN TOP-K")
                rrf_result = "CORRECT" if rrf_hit else ("WRONG" if gt_in_topk else "GT NOT IN TOP-K")

                print(f"\nSource : '{src_label}'")
                print(f"GT     : {gt_in_topk_str}")
                print(f"")
                print(f"  {'IR Rank':>7}  {'IR Score':>8}  {'Candidate':<42}  {'GT?':>4}  {'LLM Rank':>8}  {'RRF Rank':>8}  {'RRF Score':>9}")
                print(f"  {'-'*7}  {'-'*8}  {'-'*42}  {'-'*4}  {'-'*8}  {'-'*8}  {'-'*9}")
                for rank, (iri, concept, ir_sc, rrf_sc) in enumerate(
                    zip(target_iris, item["targets"], ir_scores, rrf_scores)
                ):
                    gt_marker  = " <GT>" if iri == gt_target else ""
                    ir_sel     = " <-- IR picks this" if rank == 0 else ""
                    rrf_sel    = " <-- RRF picks this" if rank == best_rrf_idx else ""
                    label      = concept['label'][:42]
                    print(
                        f"  {rank+1:>7}  {ir_sc:>8.4f}  {label:<42}  {gt_marker:>4}  "
                        f"{llm_rank_of[rank]:>8}  {rrf_rank_of[rank]:>8}  {rrf_sc:>9.4f}"
                        f"{ir_sel}{rrf_sel}"
                    )
                print(f"")
                print(f"  LLM raw output : \"{ranking_text.strip()}\"")
                print(f"  IR  result     : {ir_result}")
                print(f"  RRF result     : {rrf_result}")
                print(SEP)

    total = args.n_sources
    unreachable = total - gt_in_topk_count
    pct = lambda a, b: f"{100*a/b:.1f}%" if b else "N/A"

    print(f"\n{'SUMMARY':=^95}")
    print(f"")
    print(f"  Total sources evaluated        : {total}")
    print(f"  GT in top-{args.top_k} (reachable)    : {gt_in_topk_count} / {total}  ({pct(gt_in_topk_count, total)})  —  ceiling for both IR and RRF")
    print(f"  GT not in top-{args.top_k} (unreachable): {unreachable} / {total}  ({pct(unreachable, total)})  —  neither method can win here")
    print(f"  RRF overrides IR               : {rrf_disagrees_count} / {total}  ({pct(rrf_disagrees_count, total)})")
    print(f"")
    print(f"  {'Metric':<35}  {'IR-only':>10}  {'RRF':>10}  {'Delta':>8}")
    print(f"  {'-'*35}  {'-'*10}  {'-'*10}  {'-'*8}")
    print(f"  {'Correct (all sources)':<35}  {ir_correct:>4} / {total}  {rrf_correct:>4} / {total}  {rrf_correct-ir_correct:>+8d}")
    print(f"  {'Correct (GT in top-k only)':<35}  {ir_correct_of_topk:>4} / {gt_in_topk_count}  {rrf_correct_of_topk:>4} / {gt_in_topk_count}  {rrf_correct_of_topk-ir_correct_of_topk:>+8d}")
    print(f"  {'Accuracy (GT in top-k only)':<35}  {pct(ir_correct_of_topk, gt_in_topk_count):>10}  {pct(rrf_correct_of_topk, gt_in_topk_count):>10}")
    print(f"")
    print(f"  -- LLM signal analysis (of {gt_in_topk_count} reachable sources) --")
    print(f"  {'LLM rank-1 correct':<50} : {llm_rank1_correct} / {gt_in_topk_count}  ({pct(llm_rank1_correct, gt_in_topk_count)})  — if we used LLM alone")
    print(f"  {'IR wrong but LLM rank-1 correct':<50} : {ir_wrong_llm_right} / {gt_in_topk_count}  ({pct(ir_wrong_llm_right, gt_in_topk_count)})  — cases LLM could fix")
    print(f"  {'IR wrong + LLM right + RRF still wrong':<50} : {ir_wrong_llm_right_rrf_wrong} / {gt_in_topk_count}  ({pct(ir_wrong_llm_right_rrf_wrong, gt_in_topk_count)})  — RRF suppressed LLM signal")
    print(f"  {'IR right + LLM wrong + RRF still correct':<50} : {ir_right_llm_wrong_rrf_right} / {gt_in_topk_count}  ({pct(ir_right_llm_wrong_rrf_right, gt_in_topk_count)})  — IR weight saved result (robustness)")
    if improvement_cases:
        print(f"")
        print(f"  RRF fixed these IR errors ({len(improvement_cases)}): {improvement_cases}")
    print("=" * 95)


if __name__ == "__main__":
    main()
