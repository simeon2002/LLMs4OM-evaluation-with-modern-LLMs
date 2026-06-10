# Ontology Matching with Large Language Models: An Experimental Comparison

This repository contains the code for the Master's thesis **"Ontology Matching with Large Language Models: An Experimental Comparison"** (KU Leuven, 2025–2026), authored by **Simeon Serafimov**, supervised by Prof. dr ir Anastasia Dimou, with co-supervisors Ali Elhalawati and Duo Yang.

It is a fork and extension of the original [LLMs4OM](https://github.com/HamedBabaei/LLMs4OM) framework by Babaei Giglou et al. (2024). The thesis conducts a controlled re-evaluation of LLMs4OM on the **OAEI 2024 Bio-ML track** using contemporary retrieval and LLM models, and proposes a new **listwise ranking architecture** for instruction-tuned LLMs.

---

## Overview

The original LLMs4OM framework implements a retrieve-and-rank RAG pipeline for ontology matching:

<div align="center">
 <img src="images/LLMs4OM.jpg" width="800" height="200"/>
</div>

1. **Concept representation**: Encode ontology concepts as C (label only), CP (label + parents), or CC (label + children)
2. **Retrieval**: Use an embedding model to retrieve top-K candidate matches from the target ontology
3. **LLM matching**: Use an LLM to classify equivalence for each (source, candidate) pair. pairwise matchign score prediction for pairwise approach vs. LLM ranking for listwise approach
4. **Post-processing**: Filter by similarity and matching score threhsold and enforce one-to-one mappings for pairwise approach, filter by similarity threshold of highest ranking target candidate in LLM ranking and enforce one-to-one mappings.

This thesis extends the framework in three ways:
- Evaluates **7 retrieval models** and **8 LLMs** on the OAEI 2024 Bio-ML track
- Adds a **listwise ranking pipeline** for instruction-tuned LLMs with Reciprocal Rank Fusion (RRF) post-processing
- Analyzes the relative contribution of the retrieval model vs. the LLM to final matching performance
- Analyzes the computational runtime and LLM call count for both pipelines.
- Provides a comparison against OM baselines.

---

## Key Findings

- **Qwen-3-Embedding 4B** is the strongest retrieval model overall while **SBERT** remains a strong and competitive baseline
- Most LLMs converge to **near-identical F1-scores** in the pairwise pipeline. The `Ssim ≥ 0.9` post-processing threshold is the dominant filter and not the LLM's matching score
- The proposed **listwise pipeline** underperforms the pairwise pipeline by ~6 F1 points on NCIT-DOID, primarily due to LLM re-ranking quality degrading recall
- General-purpose LLMs are not yet competitive with specialized OM systems (e.g., HybridOM, LogMapBio) on harder Bio-ML datasets such as SNOMED-FMA

## Thesis Contributions

### 1. Listwise Ranking Architecture (new)

An alternative matching architecture for instruction-tuned LLMs. Instead of classifying each (source, candidate) pair separately, the LLM ranks all K candidates in a single prompt and outputs an ordered list (e.g., `"3, 1, 5, 2, 4"`). Post-processing uses **Reciprocal Rank Fusion (RRF)** to combine the retrieval ranking and LLM ranking:
> RRF = ir_weight / (k + ir_rank + 1) + llm_weight / (k + llm_rank + 1)

Relevant files:
- `ontomap/ontology_matchers/rag/rag_listwise.py` — listwise LLM architectures and model classes
- `ontomap/ontology_matchers/rag/dataset_listwise.py` — listwise prompt templates (C, CP, CC)
- `ontomap/postprocess/process.py` — `apply_rrf()` and `postprocess_listwise()`
- `ontomap/encoder/rag.py` — listwise concept encoders
### 2. New Retrieval Models
| Model | Class | Parameters |
|---|---|---|
| Qwen-3-Embedding 0.6B | `Qwen3EmbeddingRetrieval` | 0.6B |
| Qwen-3-Embedding 4B | `Qwen3Embedding4BRetrieval` | 4B |
| Embedding-Gemma | `EmbeddingGemma300MRetrieval` | 300M |
| Llama-Embed-Nemotron | `LlamaNemotronEmbeddingRetrieval` | 8B |
| NV-Embed-v2 | `NVEmbedV2Retrieval` | — |
### 3. New LLM Models
| Model | Class | Type |
|---|---|---|
| LLaMA-3 8B | `LLaMA3DecoderLM` | Base |
| LLaMA-3 8B Instruct | `LLaMA3InstructDecoderLM` | Instruct |
| Qwen-3.5 9B | `Qwen35_9BDecoderLM` | Base |
| Qwen-3.5 9B Instruct | (listwise) | Instruct |
| Gemma-2 9B | `Gemma2_9BDecoderLM` | Base |
| Gemma-2 2B | `Gemma2_2BDecoderLM` | Base |
| Gemma-4 26B-A4B | `Gemma4_26B_A4BDecoderLM` | Base (MoE) |
| Gemma-4 26B-A4B Instruct | (listwise) | Instruct (MoE) |
| Mamba 3B | `Mamba3BSSMLLM` | SSM |
| Mistral-Nemo 12B | `MistralNemoDecoderLM` | Base |
| Qwen-2.5 7B | `Qwen25_7BDecoderLM` | Base |
| Qwen-2.5 3B | `Qwen25_3BDecoderLM` | Base |
### 4. Hyperparameter Analysis Scripts
- `sweep_rrf_weights.py` — sweeps RRF weights (`ir_weight`, `llm_weight`) over NCIT-DOID and OMIM-ORDO
- `sweep_threshold_Qwen34_Qwen359B.py` — sweeps `Ssim` threshold from 0.0 to 0.95 for the best pipeline configuration, outputs CSV
- other hyperparameter scripts are evaluated in `experiments/experiments.ipynb` jupyter notebook
### 5. Experimental analysis and table production
- All analysis for the experiments performed and additional analyses mentioned in the thesis are avaialbe in **`experiments/experiments.ipynb`**
---
## Datasets

All experiments use the **OAEI 2024 Bio-ML track** — five equivalence matching tasks on biomedical ontologies:

| Task | Source | Target | Domain | #Source | #Target | #Refs |
|---|---|---|---|---|---|---|
| NCIT-DOID | NCIT | DOID | Disease | 15,762 | 8,465 | 4,686 |
| OMIM-ORDO | OMIM | ORDO | Rare Disease | 9,648 | 9,275 | 3,721 |
| SNOMED-FMA | SNOMED CT | FMA | Anatomy | 34,418 | 88,955 | 7,256 |
| SNOMED-NCIT (neoplas) | SNOMED CT | NCIT | Neoplasm | 22,971 | 20,247 | 3,804 |
| SNOMED-NCIT (pharm) | SNOMED CT | NCIT | Pharmacology | 29,500 | 22,136 | 5,803 |
---

## Installation

```bash
git clone https://github.com/simeon2002/LLMs4OM-evaluation-with-modern-LLMs.git
cd LLMs4OM-evaluation-with-modern-LLMs

pip install -r requirements.txt
mv .env-example .env
```

Update tokens in `.env` for LLaMA-2 or GPT-based models, or leave dummy values if not using those.

---

### Pairwise RAG Pipeline (original architecture)

Quick example with `Qwen-3.5 9B` + `Qwen-3-Embedding 4B` on NCIT-DOID:

```python
from ontomap.ontology.bioml import NCITDOIDDiseaseOMDataset
from ontomap.base import BaseConfig
from ontomap.encoder.rag import IRILabelInRAGEncoder
from ontomap.ontology_matchers.rag.models import Qwen35_9BQwen34BRAG
from ontomap.postprocess import process
from ontomap.evaluation.evaluator import evaluator

config = BaseConfig(approach='rag').get_args(device='cuda', batch_size=16)
config.root_dir = "datasets"

ontology = NCITDOIDDiseaseOMDataset().load_from_json(root_dir=config.root_dir)
encoded_inputs = IRILabelInRAGEncoder()(ontology)

model = Qwen35_9BQwen34BRAG(config.Qwen35_9BQwen34BRAG)
predicts = model.generate(input_data=encoded_inputs)

predicts, _ = process.postprocess_hybrid(
    predicts=predicts,
    llm_confidence_th=0.7,
    ir_score_threshold=0.9
)

results = evaluator(track='bio-ml', predicts=predicts, references=ontology["reference"])
print(results)
```

### Listwise Ranking Pipeline (thesis contribution)

Quick example with `Qwen-3.5 9B Instruct` (listwise) + SBERT on NCIT-DOID:

```python
from ontomap.ontology.bioml import NCITDOIDDiseaseOMDataset
from ontomap.base import BaseConfig
from ontomap.encoder.rag import IRILabelInListwiseEncoder
from ontomap.ontology_matchers.rag.rag_listwise import Qwen35_9BListwiseBertRAG
from ontomap.postprocess import process
from ontomap.evaluation.evaluator import evaluator

config = BaseConfig(approach='listwise').get_args(device='cuda', batch_size=16)
config.root_dir = "datasets"

ontology = NCITDOIDDiseaseOMDataset().load_from_json(root_dir=config.root_dir)
encoded_inputs = IRILabelInListwiseEncoder()(ontology)

model = Qwen35_9BListwiseBertRAG(config.Qwen35_9BListwiseBertRAG)
predicts = model.generate(input_data=encoded_inputs)

predicts, _ = process.postprocess_listwise(
    predicts=predicts,
    ir_score_threshold=0.9,
    ir_weight=0.3,
    llm_weight=0.7,
    k=1
)

results = evaluator(track='bio-ml', predicts=predicts, references=ontology["reference"])
print(results)
```

### Retrieval-Only Evaluation

```python
from ontomap.ontology.bioml import NCITDOIDDiseaseOMDataset
from ontomap.base import BaseConfig
from ontomap.encoder.lightweight import IRILabelInLightweightEncoder
from ontomap.ontology_matchers.retrieval.models import Qwen3Embedding4BRetrieval
from ontomap.postprocess import process
from ontomap.evaluation.evaluator import evaluator

config = BaseConfig(approach='retrieval').get_args(device='cuda')
config.root_dir = "datasets"

ontology = NCITDOIDDiseaseOMDataset().load_from_json(root_dir=config.root_dir)
encoded_inputs = IRILabelInLightweightEncoder()(ontology)

model = Qwen3Embedding4BRetrieval(config.Qwen3Embedding4BRetrieval)
predicts = model.generate(input_data=encoded_inputs)

predicts = process.eval_preprocess_ir_outputs(predicts=predicts)
results = evaluator(track='bio-ml', predicts=predicts, references=ontology["reference"])
print(results)
```

### Batch Scripts (ran on HPC with SLURM)

SLURM batch scripts for each model are at the repo root (`.batch` files).
---

## Computational Resources

All experiments were executed on the **Vlaams Supercomputer Centrum (VSC)** HPC infrastructure (KU Leuven Tier-2 clusters) using NVIDIA H100 and V100 GPUs via the Slurm workload manager.

---
