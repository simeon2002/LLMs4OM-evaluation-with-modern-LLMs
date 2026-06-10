import sys, glob, json
import pandas as pd

sys.path.insert(0, "/vsc-hard-mounts/leuven-data/385/vsc38504/thesis/llms4om/LLMs4OM")

from ontomap.postprocess.process import postprocess_hybrid
from ontomap.evaluation.metrics import evaluation_report

BASE     = "/vsc-hard-mounts/leuven-data/385/vsc38504/thesis/llms4om/LLMs4OM/experiments/outputs/bio-ml"
DATA_DIR = "/vsc-hard-mounts/leuven-data/385/vsc38504/thesis/llms4om/LLMs4OM/datasets/bio-ml"
OUT_CSV  = "/vsc-hard-mounts/leuven-data/385/vsc38504/thesis/llms4om/LLMs4OM/experiments/threshold_sweep_Qwen35_9BQwen34BRAG.csv"

task_labels = {
    "ncit-doid.disease":   "NCIT-DOID",
    "omim-ordo.disease":   "OMIM-ORDO",
    "snomed-fma.body":     "SNOMED-FMA",
    "snomed-ncit.neoplas": "SNOMED-NCIT (N)",
    "snomed-ncit.pharm":   "SNOMED-NCIT (P)",
}

encoders = {
    "C":  "label-2",
    "CC": "label-children-2",
    "CP": "label-parent-2",
}

thresholds = [round(x * 0.05, 2) for x in range(0, 20)]  # 0.00 to 0.95

rows = []
for task, label in task_labels.items():
    with open(f"{DATA_DIR}/{task}/om.json") as fp:
        reference = json.load(fp)["reference"]["equiv"]["full"]

    for enc_key, enc_pattern in encoders.items():
        files = sorted(glob.glob(f"{BASE}/{task}/rag-Qwen35_9BQwen34BRAG-{enc_pattern}*.json"))
        if not files:
            print(f"[SKIP] No file for {label} / {enc_key}")
            continue
        f = files[-1]
        print(f"[{label} / {enc_key}] {f.split('/')[-1]}", flush=True)

        with open(f) as fp:
            predicts = json.load(fp)["generated-output"]

        for th in thresholds:
            print(f"  threshold={th:.2f}", flush=True)
            processed, _ = postprocess_hybrid(
                predicts,
                ir_score_threshold=th,
                llm_confidence_th=0.7,
            )
            res = evaluation_report(processed, reference)
            rows.append({
                "dataset":   label,
                "encoder":   enc_key,
                "threshold": th,
                "precision": round(res["precision"], 2),
                "recall":    round(res["recall"], 2),
                "f1":        round(res["f-score"], 2),
                "n_preds":   res["predictions-len"],
            })

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)
print(f"\nSaved to {OUT_CSV}")
print(df.pivot_table(index=["dataset", "encoder"], columns="threshold", values="f1").to_string())
