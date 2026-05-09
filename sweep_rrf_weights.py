"""
Offline RRF weight sweep — no GPU needed.

Loads saved LLM rankings from a JSON file (produced by test_rag_listwise.py
with --save-rankings) and tries different IR/LLM weight combinations.

Usage:
    python sweep_rrf_weights.py --rankings experiments/outputs/bio-ml/omim-ordo.disease/listwise-rankings-....json
"""
import argparse
import json
import os
from pathlib import Path

from ontomap.ontology_matchers.rag.rag_listwise import apply_rrf


def sweep(rankings, ir_weights):
    print(f"\n{'IR w':>6}  {'LLM w':>6}  {'IR correct':>12}  {'RRF correct':>12}  {'Delta':>6}  {'Set C (IR saves)':>18}  {'Suppressed':>12}")
    print("-" * 90)

    total = len(rankings)
    reachable = [r for r in rankings if r["gt_target"] is not None and r["gt_target"] in r["target_iris"]]
    n_reachable = len(reachable)

    ir_correct = sum(1 for r in reachable if r["target_iris"][0] == r["gt_target"])

    results = []
    for ir_w in ir_weights:
        llm_w = round(1.0 - ir_w, 2)
        rrf_correct = 0
        set_c = 0
        suppressed = 0

        for r in reachable:
            gt = r["gt_target"]
            target_iris = r["target_iris"]
            ir_scores = r["ir_scores"]
            llm_ranking = r["llm_ranking"]
            n = len(target_iris)

            rrf_scores = apply_rrf(ir_scores, llm_ranking, k=1, ir_weight=ir_w, llm_weight=llm_w)
            best_rrf_idx = max(range(n), key=lambda i: rrf_scores[i])

            ir_hit  = target_iris[0] == gt
            llm_hit = target_iris[llm_ranking[0]] == gt
            rrf_hit = target_iris[best_rrf_idx] == gt

            if rrf_hit:
                rrf_correct += 1
            if ir_hit and not llm_hit and rrf_hit:
                set_c += 1
            if not ir_hit and llm_hit and not rrf_hit:
                suppressed += 1

        delta = rrf_correct - ir_correct
        print(f"  {ir_w:>4}  {llm_w:>6}  {ir_correct:>5}/{n_reachable}  {rrf_correct:>5}/{n_reachable}  {delta:>+6}  {set_c:>8} / {n_reachable}  {suppressed:>6} / {n_reachable}")
        results.append({
            "ir_weight": ir_w,
            "llm_weight": llm_w,
            "ir_correct": ir_correct,
            "rrf_correct": rrf_correct,
            "n_reachable": n_reachable,
            "total_sources": total,
            "delta": delta,
            "ir_accuracy": round(ir_correct / n_reachable, 4),
            "rrf_accuracy": round(rrf_correct / n_reachable, 4),
            "set_c": set_c,
            "suppressed": suppressed,
        })

    print(f"\n  IR-only correct: {ir_correct} / {n_reachable}  ({100*ir_correct/n_reachable:.1f}%)")
    print(f"  Total sources: {total}  |  Reachable (GT in top-k): {n_reachable}")
    return results, ir_correct, n_reachable, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rankings", required=True, help="JSON file from test_rag_listwise.py --save-rankings")
    args = parser.parse_args()

    with open(args.rankings) as f:
        rankings = json.load(f)

    print(f"Loaded {len(rankings)} source rankings from {args.rankings}")

    ir_weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results, ir_correct, n_reachable, total = sweep(rankings, ir_weights)

    out_path = Path(args.rankings).with_name(
        Path(args.rankings).stem.replace("listwise-rankings", "rrf-sweep") + ".json"
    )
    output = {
        "rankings_file": args.rankings,
        "ir_only_correct": ir_correct,
        "n_reachable": n_reachable,
        "total_sources": total,
        "ir_only_accuracy": round(ir_correct / n_reachable, 4),
        "sweep": results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Sweep results saved to {out_path}")


if __name__ == "__main__":
    main()
