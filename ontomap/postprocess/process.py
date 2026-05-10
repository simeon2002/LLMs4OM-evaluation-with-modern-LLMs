# -*- coding: utf-8 -*-
from typing import Dict, List

import numpy as np
from tqdm import tqdm


def eval_preprocess_ir_outputs(predicts: List) -> List:
    predicts_temp = []
    predict_map = {}
    for predict in tqdm(predicts):
        source = predict["source"]
        target_cands = predict["target-cands"]
        score_cands = predict["score-cands"]
        for target, score in zip(target_cands, score_cands):
            if score > 0:
                adjusted = False
                if predict_map.get(f"{source}-{target}", "NA") != "NA":
                    adjusted = True
                    break
                if not adjusted:
                    predicts_temp.append({"source": source, "target": target, "score": score})
                    predict_map[f"{source}-{target}"] = f"{source}-{target}"
    return predicts_temp


def preprocess_ir_outputs(predicts: List) -> List:
    predicts_temp = []
    for predict in tqdm(predicts):
        source, target_cands, score_cands = predict["source"], predict["target-cands"], predict["score-cands"]
        for target, score in zip(target_cands, score_cands):
            if score > 0:
                predicts_temp.append({"source": source, "target": target, "score": score})
    return predicts_temp


def threshold_finder(dictionary: dict, index: int, use_lst: bool = False) -> float:
    scores_dict = {}
    scores_list = []
    for outputs in dictionary.values():
        for output in outputs:
            scores_list.append(output[index])
            if scores_dict.get(output[0], 0) != 0:
                if scores_dict.get(output[0], 0) < output[index]:
                    scores_dict[output[0]] = output[index]
            else:
                scores_dict[output[0]] = output[index]
    if not use_lst:
        scores_list = list(scores_dict.values())
    threshold = sum(scores_list) / len(scores_list) if len(scores_list) != 0 else 0
    return threshold


def build_outputdict(llm_outputs: List, ir_outputs: List) -> Dict:
    outputdict = {}
    for llm_output in tqdm(llm_outputs):
        for ir_output in ir_outputs:
            if llm_output["source"] == ir_output["source"] and llm_output["target"] == ir_output["target"]:
                confidence_ratio = llm_output["score"] * ir_output["score"]
                predicts_list = [llm_output["target"], ir_output["score"], llm_output["score"], confidence_ratio]
                if llm_output["source"] not in list(outputdict.keys()):
                    outputdict[llm_output["source"]] = [predicts_list]
                else:
                    outputdict[llm_output["source"]].append(predicts_list)
    return outputdict


def confidence_score_ratio_based_filtering(outputdict: Dict, topk_confidence_ratio: int, cr_threshold: float) -> Dict:
    outputdict_confidence_ratios = {}
    for source_iri, target_cands in outputdict.items():
        top_k_items = sorted(target_cands, key=lambda X: X[3] >= cr_threshold, reverse=True)[:topk_confidence_ratio]
        outputdict_confidence_ratios[source_iri] = top_k_items
    return outputdict_confidence_ratios


def confidence_score_based_filtering(
    outputdict_confidence_ratios: Dict,
    topk_confidence_score: int,
    llm_confidence_threshold: float,
    ir_score_threshold: float,
) -> List:
    filtered_predicts = []
    for source_iri, target_cands in outputdict_confidence_ratios.items():
        top_k_items = sorted(target_cands, key=lambda X: (X[2] >= llm_confidence_threshold), reverse=True)[:topk_confidence_score]
        for target, ir_score, llm_confidence, confidence_ratio in top_k_items:
            if ir_score >= ir_score_threshold:
                filtered_predicts.append(
                    {
                        "source": source_iri,
                        "target": target,
                        "score": ir_score,
                        "confidence": llm_confidence,
                        "confidence-ratio": confidence_ratio,
                    }
                )
    return filtered_predicts


def postprocess_heuristic(predicts: List, topk_confidence_ratio: int = 3, topk_confidence_score: int = 1) -> [List, Dict]:
    ir_outputs = predicts[0]["ir-outputs"]
    llm_outputs = predicts[1]["llm-output"]

    ir_outputs = preprocess_ir_outputs(predicts=ir_outputs)
    outputdict = build_outputdict(llm_outputs=llm_outputs, ir_outputs=ir_outputs)

    cr_threshold = threshold_finder(outputdict, index=3, use_lst=False)  # 3=confidence_ratio index
    outputdict_confidence_ratios = confidence_score_ratio_based_filtering(
        outputdict=outputdict,
        topk_confidence_ratio=topk_confidence_ratio,
        cr_threshold=cr_threshold,
    )

    ir_score_threshold = threshold_finder(outputdict_confidence_ratios, index=1, use_lst=True)  # 1=ir_score index
    llm_confidence_threshold = threshold_finder(outputdict, index=2, use_lst=False)  # 2=confidence_ratio index

    filtered_predicts = confidence_score_based_filtering(
        outputdict_confidence_ratios=outputdict_confidence_ratios,
        topk_confidence_score=topk_confidence_score,
        llm_confidence_threshold=llm_confidence_threshold,
        ir_score_threshold=ir_score_threshold,
    )
    configs = {
        "topk-confidence-ratio": topk_confidence_ratio,
        "topk-confidence-score": topk_confidence_score,
        "confidence-ratio-th": cr_threshold,
        "ir-score-th": ir_score_threshold,
        "llm-confidence-th": llm_confidence_threshold,
    }
    return filtered_predicts, configs


def postprocess_hybrid(predicts: List, ir_score_threshold: float = 0.9, llm_confidence_th: float = 0.7) -> [List, Dict]:
    ir_outputs = predicts[0]["ir-outputs"]
    llm_outputs = predicts[1]["llm-output"]
    ir_cleaned_outputs_id = []
    ir_cleaned_outputs = []
    for ir in ir_outputs:
        if ir["source"] not in ir_cleaned_outputs_id:
            ir_cleaned_outputs_id.append(ir["source"])
            ir_cleaned_outputs.append(ir)
    ir_outputs = ir_cleaned_outputs
    # ir_outputs = preprocess_ir_outputs(predicts=ir_outputs)
    targets = [target for index, ir_output in enumerate(ir_outputs) for target in ir_output["target-cands"]]
    targets = list(set(targets))
    target2index = {target: index for index, target in enumerate(targets)}
    source2index = {ir_output["source"]: index for index, ir_output in enumerate(ir_outputs)}

    ir_dict = {ir_output["source"]: ir_output for ir_output in ir_outputs}
    ir_dict_based_llm = {ir_output["source"]: np.zeros(len(target2index)) for ir_output in ir_outputs}
    llm_dict = {ir_output["source"]: np.zeros(len(target2index)) for ir_output in ir_outputs}

    outputdict = {}
    for llm_output in tqdm(llm_outputs):
        if llm_output["source"] not in list(outputdict.keys()):
            outputdict[llm_output["source"]] = []
        if llm_output["score"] > llm_confidence_th:
            ir_output = ir_dict.get(llm_output["source"])
            for index, (ir_cand, ir_cand_score) in enumerate(zip(ir_output["target-cands"], ir_output["score-cands"])):
                if ir_cand == llm_output["target"]:
                    confidence_ratio = (llm_output["score"] * 0.2 + 0.8 * ir_cand_score) / 2
                    predicts_list = [llm_output["target"], ir_cand_score, llm_output["score"], confidence_ratio]
                    outputdict[llm_output["source"]].append(predicts_list)
                    ir_dict_based_llm[llm_output["source"]][target2index[ir_cand]] = ir_cand_score
                    llm_dict[llm_output["source"]][target2index[ir_cand]] = confidence_ratio
                    break

    ir_matrix = np.zeros((len(source2index), len(target2index)))

    for iri, output in ir_dict.items():
        for ir_cand, ir_score in zip(output["target-cands"], output["score-cands"]):
            ir_matrix[source2index[iri], target2index[ir_cand]] = ir_score

    ir_matrix_based_llm = np.zeros((len(source2index), len(target2index)))
    for iri, output in ir_dict_based_llm.items():
        ir_matrix_based_llm[source2index[iri], :] = output

    llm_matrix = np.zeros((len(source2index), len(target2index)))
    for iri, output in llm_dict.items():
        llm_matrix[source2index[iri], :] = output

    for col_idx in range(ir_matrix_based_llm.shape[1]):
        col = ir_matrix_based_llm[:, col_idx]
        max_index = np.argmax(col)
        ir_matrix_based_llm[:, col_idx] = np.where(np.arange(len(col)) != max_index, 0, col)

    for col_idx in range(llm_matrix.shape[1]):
        col = llm_matrix[:, col_idx]
        max_index = np.argmax(col)
        llm_matrix[:, col_idx] = np.where(np.arange(len(col)) != max_index, 0, col)

    for row_idx in range(ir_matrix_based_llm.shape[0]):
        row = ir_matrix_based_llm[row_idx, :]
        max_index = np.argmax(row)
        ir_matrix_based_llm[row_idx, :] = np.where(np.arange(len(row)) != max_index, 0, row)

    for row_idx in range(llm_matrix.shape[0]):
        row = llm_matrix[row_idx, :]
        max_index = np.argmax(row)
        llm_matrix[row_idx, :] = np.where(np.arange(len(row)) != max_index, 0, row)

    index2source = {index: source for source, index in source2index.items()}
    index2target = {index: target for target, index in target2index.items()}
    rows, cols = ir_matrix_based_llm.nonzero()
    final_predict = []
    for row, col in zip(rows, cols):
        if ir_matrix_based_llm[row, col] >= ir_score_threshold:
            final_predict.append({
                "source": index2source[row],
                "target": index2target[col],
                "score": ir_matrix_based_llm[row, col]
            })
    configs = {
        "llm-confidence-th": llm_confidence_th,
        "ir-score-th": ir_score_threshold,
    }
    return final_predict, configs


def apply_rrf(
    ir_rank_matrix: np.ndarray,
    llm_rank_matrix: np.ndarray,
    ir_weight: float = 0.3,
    llm_weight: float = 0.7,
    k: int = 1,
) -> np.ndarray:
    n_t = ir_rank_matrix.shape[1]
    lr = np.where(llm_rank_matrix != np.inf, llm_rank_matrix, n_t)
    irr = np.where(ir_rank_matrix != np.inf, ir_rank_matrix, n_t)
    return np.where(
        llm_rank_matrix != np.inf,
        ir_weight / (k + irr + 1) + llm_weight / (k + lr + 1),
        0.0,
    )


def postprocess_listwise(
    predicts: List,
    ir_score_threshold: float = 0.9,
    ir_weight: float = 0.3,
    llm_weight: float = 0.7,
    ir_confidence_gate: float = 0.9,
) -> [List, Dict]:
    ir_outputs = predicts[0]["ir-outputs"]
    llm_outputs = predicts[1]["llm-output"]

    seen = set()
    ir_cleaned = []
    for ir in ir_outputs:
        if ir["source"] not in seen:
            seen.add(ir["source"])
            ir_cleaned.append(ir)
    ir_outputs = ir_cleaned

    ir_dict = {ir["source"]: ir for ir in ir_outputs}
    targets = list(set(t for ir in ir_outputs for t in ir["target-cands"]))
    target2index = {t: i for i, t in enumerate(targets)}
    source2index = {ir["source"]: i for i, ir in enumerate(ir_outputs)}
    n_s, n_t = len(source2index), len(target2index)

    llm_rank_matrix = np.full((n_s, n_t), np.inf)
    ir_score_matrix = np.zeros((n_s, n_t))
    ir_rank_matrix = np.full((n_s, n_t), np.inf)

    for pred in llm_outputs:
        src, tgt = pred["source"], pred["target"]
        if src not in source2index or tgt not in target2index:
            continue
        si, ti = source2index[src], target2index[tgt]
        llm_rank_matrix[si, ti] = pred["score"]  # 0-indexed LLM rank, 0 = best

    for iri, ir_out in ir_dict.items():
        si = source2index[iri]
        ir_cands = ir_out["target-cands"]
        ir_scores = ir_out["score-cands"]
        ir_order = sorted(range(len(ir_cands)), key=lambda idx: ir_scores[idx], reverse=True)
        for ir_rank_pos, cand_idx in enumerate(ir_order):
            tgt = ir_cands[cand_idx]
            if tgt in target2index:
                ti = target2index[tgt]
                ir_score_matrix[si, ti] = ir_scores[cand_idx]
                ir_rank_matrix[si, ti] = ir_rank_pos

    rrf_matrix = apply_rrf(ir_rank_matrix, llm_rank_matrix, ir_weight, llm_weight)

    # IR confidence gate: if top IR score >= gate, force IR top-1 (override LLM)
    if ir_confidence_gate > 0.0:
        for si in range(n_s):
            best_ir_ti = int(np.argmax(ir_score_matrix[si, :]))
            if ir_score_matrix[si, best_ir_ti] >= ir_confidence_gate:
                rrf_matrix[si, :] = 0.0
                rrf_matrix[si, best_ir_ti] = 1.0 + ir_score_matrix[si, best_ir_ti]

    for col in range(n_t):
        best = np.argmax(rrf_matrix[:, col])
        mask = np.arange(n_s) != best
        rrf_matrix[mask, col] = 0
        ir_score_matrix[mask, col] = 0

    for row in range(n_s):
        best = np.argmax(rrf_matrix[row, :])
        mask = np.arange(n_t) != best
        rrf_matrix[row, mask] = 0
        ir_score_matrix[row, mask] = 0

    index2source = {v: k for k, v in source2index.items()}
    index2target = {v: k for k, v in target2index.items()}

    final_predict = []
    for si, ti in zip(*rrf_matrix.nonzero()):
        if ir_score_matrix[si, ti] >= ir_score_threshold:
            final_predict.append({
                "source": index2source[si],
                "target": index2target[ti],
                "score": float(rrf_matrix[si, ti]),
            })
    configs = {
        "ir-score-th": ir_score_threshold,
        "llm-confidence-th": 0.0,
        "ir-weight": ir_weight,
        "llm-weight": llm_weight,
        "ir-confidence-gate": ir_confidence_gate,
    }
    return final_predict, configs
