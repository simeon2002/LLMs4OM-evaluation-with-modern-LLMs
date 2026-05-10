"""
Offline sweep of RRF k values and IR/LLM weights on saved listwise rankings.
No GPU needed — just loads the rankings JSON and re-applies RRF with different params.

Usage:
    python sweep_rrf_k.py --rankings <path-to-rankings.json>
    python sweep_rrf_k.py --rankings <path-to-rankings.json> --sweep-weights
"""
import argparse
import json


def apply_rrf(ir_scores, llm_ranking, k=1, ir_weight=0.3, llm_weight=0.7):
    n = len(ir_scores)
    ir_order = sorted(range(n), key=lambda i: ir_scores[i], reverse=True)
    ir_rank = [0] * n
    for rank, idx in enumerate(ir_order):
        ir_rank[idx] = rank
    llm_rank = [0] * n
    for rank, idx in enumerate(llm_ranking):
        llm_rank[idx] = rank
    return [ir_weight / (k + ir_rank[i] + 1) + llm_weight / (k + llm_rank[i] + 1) for i in range(n)]


def evaluate(rankings, k, ir_weight=0.3, llm_weight=0.7):
    ir_correct = ir_correct_topk = 0
    rrf_correct = rrf_correct_topk = 0
    gt_in_topk = 0
    set_c = 0  # IR right, LLM wrong, RRF wrong
    set_b = 0  # IR wrong, LLM right, RRF right

    for entry in rankings:
        target_iris = entry["target_iris"]
        ir_scores = entry["ir_scores"]
        llm_ranking = entry["llm_ranking"]
        gt = entry["gt_target"]

        n = len(target_iris)
        ir_best = target_iris[0]
        llm_best = target_iris[llm_ranking[0]]

        rrf_scores = apply_rrf(ir_scores, llm_ranking, k=k, ir_weight=ir_weight, llm_weight=llm_weight)
        rrf_best = target_iris[max(range(n), key=lambda i: rrf_scores[i])]

        in_topk = gt is not None and gt in target_iris
        ir_hit = gt == ir_best
        llm_hit = gt == llm_best
        rrf_hit = gt == rrf_best

        if in_topk:
            gt_in_topk += 1
            if ir_hit:
                ir_correct_topk += 1
            if rrf_hit:
                rrf_correct_topk += 1
            if ir_hit and not llm_hit and not rrf_hit:
                set_c += 1
            if not ir_hit and llm_hit and rrf_hit:
                set_b += 1

        if ir_hit:
            ir_correct += 1
        if rrf_hit:
            rrf_correct += 1

    total = len(rankings)
    return {
        "total": total,
        "gt_in_topk": gt_in_topk,
        "ir_correct": ir_correct,
        "rrf_correct": rrf_correct,
        "ir_correct_topk": ir_correct_topk,
        "rrf_correct_topk": rrf_correct_topk,
        "set_c": set_c,
        "set_b": set_b,
        "delta": rrf_correct - ir_correct,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rankings", required=True)
    parser.add_argument("--ir-weight", type=float, default=0.3)
    parser.add_argument("--llm-weight", type=float, default=0.7)
    parser.add_argument("--sweep-weights", action="store_true", help="2D sweep over k and IR/LLM weight combinations")
    args = parser.parse_args()

    with open(args.rankings) as f:
        rankings = json.load(f)

    print(f"Loaded {len(rankings)} sources from {args.rankings}\n")

    k_values = [1, 2, 3, 5, 10, 20, 60]

    if args.sweep_weights:
        weight_pairs = [
            (0.1, 0.9),
            (0.2, 0.8),
            (0.3, 0.7),
            (0.4, 0.6),
            (0.5, 0.5),
            (0.6, 0.4),
            (0.7, 0.3),
        ]

        all_results = []
        total = None
        for ir_w, llm_w in weight_pairs:
            for k in k_values:
                r = evaluate(rankings, k, ir_w, llm_w)
                total = r["total"]
                all_results.append((ir_w, llm_w, k, r))

        # print grouped by weight pair
        print(f"{'IR w':>5}  {'LLM w':>5}  {'k':>4}  {'IR cor':>8}  {'RRF cor':>9}  {'Delta':>6}  {'Set C':>6}  {'Set B':>6}")
        print(f"{'':->5}  {'':->5}  {'':->4}  {'':->8}  {'':->9}  {'':->6}  {'':->6}  {'':->6}")
        prev_w = None
        for ir_w, llm_w, k, r in all_results:
            if (ir_w, llm_w) != prev_w:
                print()
                prev_w = (ir_w, llm_w)
            marker = " <-- current default" if (abs(ir_w - 0.3) < 1e-9 and abs(llm_w - 0.7) < 1e-9 and k == 1) else ""
            print(
                f"{ir_w:>5.1f}  {llm_w:>5.1f}  {k:>4}  "
                f"{r['ir_correct']:>4}/{total}  {r['rrf_correct']:>5}/{total}  "
                f"{r['delta']:>+6}  {r['set_c']:>6}  {r['set_b']:>6}{marker}"
            )

        # top 10 by delta
        top = sorted(all_results, key=lambda x: (-x[3]["delta"], x[3]["set_c"]))[:10]
        print(f"\n--- Top 10 configurations by delta ---")
        print(f"{'IR w':>5}  {'LLM w':>5}  {'k':>4}  {'Delta':>6}  {'Set C':>6}  {'Set B':>6}")
        print(f"{'':->5}  {'':->5}  {'':->4}  {'':->6}  {'':->6}  {'':->6}")
        for ir_w, llm_w, k, r in top:
            marker = " <-- current default" if (abs(ir_w - 0.3) < 1e-9 and abs(llm_w - 0.7) < 1e-9 and k == 1) else ""
            print(f"{ir_w:>5.1f}  {llm_w:>5.1f}  {k:>4}  {r['delta']:>+6}  {r['set_c']:>6}  {r['set_b']:>6}{marker}")

    else:
        print(f"IR weight: {args.ir_weight}  LLM weight: {args.llm_weight}\n")
        print(f"{'k':>5}  {'IR correct':>10}  {'RRF correct':>11}  {'Delta':>6}  {'Set C (IR✓ LLM✗ RRF✗)':>22}  {'Set B (IR✗ LLM✓ RRF✓)':>22}")
        print(f"{'':->5}  {'':->10}  {'':->11}  {'':->6}  {'':->22}  {'':->22}")

        total = None
        for k in k_values:
            r = evaluate(rankings, k, args.ir_weight, args.llm_weight)
            total = r["total"]
            print(
                f"{k:>5}  {r['ir_correct']:>4}/{total}  {r['rrf_correct']:>5}/{total}  "
                f"{r['delta']:>+6}  {r['set_c']:>22}  {r['set_b']:>22}"
            )


if __name__ == "__main__":
    main()
