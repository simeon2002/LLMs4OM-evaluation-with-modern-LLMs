# Pipeline Explanation Plan

## Goal
Step-by-step walkthrough of the full RAG ontology matching pipeline — from raw OWL ontology files to final F1 scores — and how the new listwise variant differs.

---

## Step 1 — Data Loading

**Files:** `ontomap/base/ontology.py`, `ontomap/base/dataset.py`, `ontomap/ontology/bioml.py`, `ontomap/ontology/__init__.py`  
**Trigger in pipeline:** `task_obj.load_from_json()` or `task_obj.collect()`

### Class Hierarchy (4 layers)

```
BaseOntologyParser          ← owlready2 OWL parsing logic (base/ontology.py)
    └── BioOntology         ← BioML synonym/comment extraction (ontology/bioml.py)

OMDataset (ABC)             ← abstract dataset interface (base/dataset.py)
    └── BioMLOMDataset      ← BioML file path overrides (ontology/bioml.py)
            └── NCITDOIDDiseaseOMDataset   ← concrete: ncit.owl ↔ doid.owl
            └── OMIMORDODiseaseOMDataset   ← concrete: omim.owl ↔ ordo.owl
            └── SNOMEDFMABodyOMDataset     ← etc.
```

### Two data paths

| Method | How | When |
|--------|-----|-------|
| `collect()` | Parses raw OWL files with owlready2 | Fresh parse from disk |
| `load_from_json()` | Reads pre-cached `om.json` | Fast path (used in pipeline) |

### OWL Parsing flow (`collect` path)

1. `owlready2.World().get_ontology(file).load()` — loads OWL file
2. Iterate all classes with a label (`is_contain_label()`)
3. For each class, extract:

```python
{
    "name": "C0001",                                    # OWL class name/ID
    "iri": "http://purl.obolibrary.org/obo/NCIT_C0001",
    "label": "lymphoma",                                # first label string
    "childrens": [{"iri": ..., "name": ..., "label": ...}],  # subclasses
    "parents":   [{"iri": ..., "name": ..., "label": ...}],  # is_a relations
    "synonyms":  ["lymphoma neoplasm"],                 # hasRelatedSynonym + hasExactSynonym
    "comment":   ["A malignant neoplasm of lymphoid tissue"]
}
```

4. `duplicate_removals()` — removes self-references from parents/children

### Full `task_owl` structure

```python
{
  "dataset-info": {"track": "bio-ml", "ontology-name": "ncit-doid.disease"},
  "source": [ {name, iri, label, childrens, parents, synonyms, comment}, ... ],
  "target": [ {name, iri, label, childrens, parents, synonyms, comment}, ... ],
  "reference": {
    "equiv": {
      "full":  [{"source": iri, "target": iri}, ...],  # all ground truth pairs
      "test":  [...],                                    # test split
      "train": [...]                                     # train split
    },
    "subs": {"test-cands": [...], "train": [...]}       # subsumption relations
  }
}
```

### Registration

`ontomap/ontology/__init__.py` defines `ontology_matching` dict — currently only `NCITDOIDDiseaseOMDataset` active (others commented out). Pipeline iterates this dict.

---

## Step 2 — Encoding

**Files:** `ontomap/base/encoder.py`, `ontomap/encoder/lightweight.py`, `ontomap/encoder/rag.py`, `ontomap/encoder/__init__.py`  
**Trigger in pipeline:** `encoder_module()(**task_owl)`

### Class Hierarchy

```
BaseEncoder (ABC)                        ← base/encoder.py
    └── LightweightEncoder               ← encoder/lightweight.py
            ├── IRILabelInLightweightEncoder          → text = label
            ├── IRILabelDescInLightweightEncoder      → text = label + comment
            ├── IRILabelChildrensInLightweightEncoder → text = label + children labels
            └── IRILabelParentsInLightweightEncoder   → text = label + parent labels
    └── RAGEncoder                       ← encoder/rag.py
            ├── IRILabelInRAGEncoder          llm_encoder = "LabelRAGDataset"
            ├── IRILabelChildrensInRAGEncoder llm_encoder = "LabelChildrenRAGDataset"
            ├── IRILabelParentsInRAGEncoder   llm_encoder = "LabelParentRAGDataset"
            ├── IRILabelInRAGInstructEncoder          (instruct variants)
            ├── IRILabelChildrensInRAGInstructEncoder
            └── IRILabelParentsInRAGInstructEncoder
```

### Two encoder roles (both called during RAG)

**1. LightweightEncoder** — used inside IR retrieval  
Converts each concept to a plain text string for BERT embedding:

| Class | Output text |
|-------|-------------|
| `IRILabelInLightweightEncoder` | `"lymphoma"` |
| `IRILabelParentsInLightweightEncoder` | `"lymphoma  malignant neoplasm, cancer"` |
| `IRILabelChildrensInLightweightEncoder` | `"lymphoma  b-cell lymphoma, t-cell lymphoma"` |

Preprocessing: underscores → spaces, lowercase.

**2. RAGEncoder** — the `encoder_module` called in the pipeline  
Does NOT embed anything. Just packages pointers + IRI→index maps:

```python
{
    "retriever-encoder": IRILabelInLightweightEncoder,  # class (called later in IR)
    "llm-encoder": "LabelParentRAGDataset",             # string (eval'd later for prompts)
    "task-args": task_owl,                              # full ontology data
    "source-onto-iri2index": {"http://...NCIT_C1": 0, ...},  # fast IRI lookup
    "target-onto-iri2index": {"http://...DOID_1": 0, ...}
}
```

`source-onto-iri2index` / `target-onto-iri2index`: O(1) lookup — given an IRI from retrieval results, find the full concept dict instantly without scanning the whole list.

### Registration

`ontomap/encoder/__init__.py` defines `EncoderCatalog`:
```python
EncoderCatalog = {
    "rag": {
        "label":        IRILabelInRAGEncoder,
        "label-parent": IRILabelParentsInRAGEncoder,
        "label-children": IRILabelChildrensInRAGEncoder,
        ...
    },
    "lightweight": { ... },
    "fewshot": { ... },
}
```
Pipeline selects encoder via `EncoderCatalog[kwargs["encoder"]][encoder_id]`.

---

## Step 3 — IR Retrieval

**Files:** `ontomap/base/model.py`, `ontomap/ontology_matchers/retrieval/retrieval.py`, `ontomap/ontology_matchers/retrieval/models.py`  
**Trigger in pipeline:** `MODEL.ir_generate()` → `retrieval_encoder()(**task_args)` → `Retrieval.generate()`

### Class Hierarchy

```
BaseOMModel (ABC)                        ← base/model.py
    └── Retrieval                        ← retrieval/retrieval.py
            ├── BiEncoderRetrieval       ← encodes all at once, GPU cosine similarity matrix
            │       ├── BERTRetrieval              path: sentence-transformers/multi-qa-mpnet-base-dot-v1
            │       ├── SpecterBERTRetrieval        path: allenai/specter_plus_plus
            │       ├── Qwen3EmbeddingRetrieval     path: Qwen/Qwen3-Embedding-0.6B  (batch_size=4)
            │       ├── Qwen3Embedding4BRetrieval   path: Qwen/Qwen3-Embedding-4B    (batch_size=4)
            │       ├── EmbeddingGemma300MRetrieval path: google/embeddinggemma-300m
            │       ├── LlamaNemotronEmbeddingRetrieval path: nvidia/llama-embed-nemotron-8b
            │       ├── NVEmbedV2Retrieval          path: nvidia/NV-Embed-v2 (adds EOS token)
            │       ├── FlanT5XLRetrieval           path: google/flan-t5-xl
            │       ├── FlanT5XXLRetrieval          path: google/flan-t5-xxl
            │       └── AdaRetrieval               loads pre-computed numpy embeddings from disk
            ├── TFIDFRetrieval           ← sklearn TfidfVectorizer (no neural model)
            ├── BM25Retrieval            ← BM25Okapi probabilistic ranking
            └── MLRetrieval
                    └── SVMBERTRetrieval ← trains LinearSVC per source concept (very slow)
```

### `generate()` flow (BiEncoderRetrieval — the one we use)

```
input_data = [source_ontos, target_ontos]   ← from LightweightEncoder
      ↓
fit(target texts)    → embed ALL targets at once  → shape (n_target, embed_dim)
transform(source texts) → embed ALL sources at once → shape (n_source, embed_dim)
      ↓
cosine_similarity(sources_embed, targets_embed) → matrix (n_source × n_target)
      ↓
for each source row: torch.topk(k=5) → top-5 target indices + scores
      ↓
map indices → IRIs using target_ontology list
```

### Final `ir_output`

```python
[
  {
    "source": "http://...NCIT_C1",
    "target-cands": ["http://...DOID_1", "http://...DOID_5", ...],  # top-k IRIs
    "score-cands":  [0.92, 0.87, 0.81, 0.74, 0.68]                 # cosine scores
  },
  ...  # one entry per source concept
]
```

Then `preprocess_ir_outputs()` flattens to individual pairs (score > 0 only):
```python
[{"source": iri, "target": iri1, "score": 0.92}, {"source": iri, "target": iri2, "score": 0.87}, ...]
```

### Key design note
All RAG encoders use `IRILabelInLightweightEncoder` for retrieval (label-only text). Parent/children info is **not** fed to BERT — only to the LLM prompt in step 4. Rationale: BERT's job is recall (get the right candidate into top-5), LLM's job is precision (verify the right one). Adding parent/children to BERT can hurt by matching on unrelated parent labels.

---

## Step 4 — LLM Generation (Pairwise yes/no)

**Files:** `ontomap/ontology_matchers/rag/rag.py`, `ontomap/ontology_matchers/rag/dataset.py`, `ontomap/ontology_matchers/llm/llm.py`  
**Trigger in pipeline:** `RAG.llm_generate(input_data, ir_output)`

### Input

`ir_output` = flat list of individual pairs from `preprocess_ir_outputs()`:
```python
[{"source": "http://...NCIT_C1", "target": "http://...DOID_1", "score": 0.92}, ...]
# one entry per (source, candidate) pair — up to n_sources × top_k entries
```

### Class Hierarchy

```
RAGDataset (Dataset)                        ← dataset.py — base prompt + collate_fn
    ├── LabelRAGDataset                     → prompt: label only
    ├── LabelParentRAGDataset               → prompt: label + parents
    ├── LabelChildrenRAGDataset             → prompt: label + children
    ├── LabelRAGInstructDataset             → instruct-style prompt, label only
    ├── LabelParentRAGInstructDataset       → instruct-style, label + parents
    └── LabelChildrenRAGInstructDataset     → instruct-style, label + children

LLM (ABC)                                   ← llm/llm.py — load(), tokenize(), generate() dispatcher
    └── BaseLLMArch                         ← 8-bit load_model, text generation + batch_decode
            ├── EncoderDecoderLLMArch       ← for FlanT5 (encoder-decoder architecture)
            ├── DecoderLLMArch              ← for decoder-only models (generic)
            └── LLaMA2DecoderLLMArch        ← HF models: padding_side=left, HF token, 8-bit quant
                    └── RAGBasedDecoderLLMArch       ← rag.py — overrides generate: yes/no logit scoring
                            └── RAGBasedInstructDecoderLLMArch  ← overrides tokenize: adds chat template
                                    └── LLaMA3InstructDecoderLM, Gemma4..., Qwen35...

RAGDataset (Dataset)                        ← dataset.py — base prompt + collate_fn
``` 

### `llm_generate()` flow step by step

**1. `build_llm_inputs()`** — maps IRIs back to full concept dicts

```python
for retrieved_item in ir_output:
    source_concept = source_onto[source_onto_iri2index[retrieved_item["source"]]]
    target_concept = target_onto[target_onto_iri2index[retrieved_item["target"]]]
    llm_inputs.append({"source": source_concept, "target": target_concept, "ir-scores": score})
```

Result: list of `{"source": {label, parents, children, ...}, "target": {...}, "ir-scores": 0.92}`

**2. `build_llm_encoder()`** — instantiates the dataset class

```python
dataset = eval(input_data["llm-encoder"])(data=llm_inputs)
# input_data["llm-encoder"] = "LabelParentRAGDataset"  (string from RAGEncoder)
# eval() turns string into class → instantiates it
```

**3. Dataset `__getitem__` / `fill_one_sample()`** — builds one prompt per pair

Example for `LabelParentRAGDataset`:
```
Classify if two concepts refer to the same real world entity or not (answer only yes or no).
### First concept:
spinal meningocele
Parents: neural tube defect
### Second concept:
meningocele
Parents: spinal cord disease
### Answer:
```

For instruct models (`LabelRAGInstructDataset`), the prompt is instead:
```
You are an ontology matching expert. Determine whether the following two concepts refer to the same real-world entity.

Concept 1: spinal meningocele
Concept 2: meningocele

Answer with exactly one word: yes or no.
```

**4. DataLoader batching** — `collate_fn` groups into `{"texts": [...], "iris": [...]}`

**5. `RAGBasedDecoderLLMArch.generate(texts)`** — yes/no logit scoring

```python
# tokenize texts → input_ids
# model.generate(..., max_new_tokens=1, output_scores=True, return_dict_in_generate=True)
# outputs.scores[0] → shape (batch_size, vocab_size) — logits for position 0

# ANSWER_SET maps synonyms to token IDs:
# "yes" set: ["yes", "correct", "true", "positive", ...]
# "no"  set: ["no",  "incorrect", "false", "negative", ...]

probas_yes_no = outputs.scores[0][:, yes_token_ids + no_token_ids].softmax(-1)
yes_proba = probas_yes_no[:, :len(yes_set)].sum(dim=1)   # sum over all "yes" synonyms
no_proba  = probas_yes_no[:, len(yes_set):].sum(dim=1)   # sum over all "no" synonyms

# take the winning label + its probability as confidence
sequences = ["yes" or "no" per batch item]
sequence_probas = [float confidence per batch item]
```

**6. Filter: keep only "yes" predictions**

```python
for label, proba, iri_pair in zip(sequences, sequence_probas, iris):
    if label == "yes":
        predictions.append({"source": iri_pair[0], "target": iri_pair[1], "score": proba})
```

### Final `RAG.generate()` output (stored as JSON)

```python
[
  {"ir-outputs": [{"source": iri, "target-cands": [...], "score-cands": [...]}, ...]},
  {"llm-output": [{"source": iri, "target": iri, "score": 0.95}, ...]}
]
```

### Key design note — why logits not text generation

The model is forced to generate **1 token** and the logit scores over the yes/no vocabulary are read directly instead of decoding text. This gives a calibrated probability (0–1) as confidence score. This is more reliable than decoding "yes"/"no" text and avoids issues with models that generate extra tokens (e.g. "yes, they match").

For instruct models this breaks down — they always produce a high probability for their "preferred" token due to RLHF training, making confidence scores cluster near 0 or 1. That's why the instruct variant still uses logit scoring but the scores are less informative.

---

## Step 5 — Postprocessing

**File:** `ontomap/postprocess/process.py`  
**Trigger in evaluation:** `evaluator_module()` → `postprocess_hybrid()` (or `postprocess_heuristic()`)

### Two postprocessing strategies

| Function | When used | Strategy |
|----------|-----------|----------|
| `postprocess_hybrid()` | Current default | Matrix-based 1-to-1 enforcement + fixed thresholds |
| `postprocess_heuristic()` | Older variant | Dynamic threshold from score distribution |

### `postprocess_hybrid()` step by step

Input: `predicts = [{"ir-outputs": [...]}, {"llm-output": [...]}]`  
Parameters: `ir_score_threshold=0.9`, `llm_confidence_th=0.7`

**1. Deduplicate IR outputs** — keep only first entry per source IRI

**2. Build index maps**
```python
target2index = {"http://...DOID_1": 0, "http://...DOID_2": 1, ...}
source2index = {"http://...NCIT_C1": 0, "http://...NCIT_C2": 1, ...}
```

**3. For each LLM "yes" prediction where score > 0.7:**
```python
confidence_ratio = (llm_score * 0.2 + ir_score * 0.8) / 2
# IR score weighted 4× higher than LLM confidence
```
Fill `ir_matrix_based_llm[source_idx, target_idx] = ir_score`  
Fill `llm_matrix[source_idx, target_idx] = confidence_ratio`

**4. Enforce one-to-one constraint (column pass):**  
For each target column: keep only the source with the highest score, zero out the rest.

**5. Enforce one-to-one constraint (row pass):**  
For each source row: keep only the target with the highest score, zero out the rest.

**6. Final filter:**  
Keep only cells where `ir_score >= ir_score_threshold (0.9)`.

Result:
```python
[{"source": iri, "target": iri, "score": 0.92}, ...]  # final 1-to-1 alignments
```

### Key design note

The 0.9 IR score threshold is strict — it means only high-confidence BERT matches that the LLM also said "yes" to are kept. This maximises precision at the cost of recall. The `llm_confidence_th=0.7` default filters out uncertain LLM predictions. Both thresholds are tunable via the pipeline config.

Result: `[{"source": iri, "target": iri, "score": 0.92}]`

---

## Step 6 — Evaluation

**File:** `ontomap/evaluation/metrics.py`

Compare predictions against `reference["equiv"]["full"]`:
- **Precision** = correct predictions / total predicted
- **Recall** = correct predictions / total reference
- **F1** = harmonic mean

---

## Listwise Variant — What Changes

| Step | Normal | Listwise |
|------|--------|----------|
| 1–3 | Same | Same |
| 4 | 1 prompt per **(source, candidate) pair** → yes/no | 1 prompt per **source** listing all k candidates → ranking string "3,1,5,2,4" |
| 5 | LLM confidence + IR score → threshold filter → 1-to-1 matrix | RRF(IR rank, LLM rank) → top-1 per source directly |
| 6 | Same | Same |

**Key file:** `ontomap/ontology_matchers/rag/rag_listwise.py` + `dataset_listwise.py`

**Current finding:** LLaMA3-8B delta=0 (never overrides IR). Gemma4-26B test pending on H100.

---

# Plan: Listwise Ranking + RRF for RAG Pipeline

## Context
Current RAG pipeline uses **pairwise binary yes/no** per (source, candidate) pair, then combines LLM score + IR score via a confidence ratio. For instruct models this collapses to ~0 or ~1, making LLM score useless — the final ranking is determined entirely by the IR score.

Copromotor suggestion (Duo Yang): replace with **listwise ranking + RRF**.
- One prompt per source concept listing all top-k candidates
- LLM outputs a ranked ordering (e.g. "3, 1, 5, 2, 4")
- RRF (Reciprocal Rank Fusion) combines IR rank + LLM rank → final score
- Take top-1 per source as the predicted alignment

**Branch:** `feature/listwise-rrf` (already created)
**Copromotor says:** keep it separate, test on same dataset + LLM (LLaMA3 instruct), merge later if results are good.
**Motivation:** the bad case shown in demo (`Spinal Meningocele` → rank 1 by IR was wrong, rank 2 was correct) shows IR-only is sometimes wrong; listwise LLM can correct it.

---

## Critical Files (all new — nothing in existing files changes)
- `LLMs4OM/ontomap/ontology_matchers/rag/dataset_listwise.py` — new: `LabelListwiseRAGDataset`
- `LLMs4OM/ontomap/ontology_matchers/rag/rag_listwise.py` — new: LLM arch + `ListwiseRAG` + helpers
- `LLMs4OM/test_rag_listwise.py` — new: standalone test script (interactive output)
- `LLMs4OM/test_rag_listwise.batch` — new: SLURM batch script

Existing files NOT touched (keeps homogeneity of main pipeline).

---

## Architecture Overview

**Current flow (pairwise):**
```
IR retrieval → flat (source, cand) pairs → LLM yes/no per pair → confidence_ratio → threshold → alignments
```

**New flow (listwise):**
```
IR retrieval → group by source → one prompt per source with all k candidates → LLM ranking text →
parse ranking → RRF(IR rank, LLM rank) → top-1 per source → alignments
```

---

## Step 1 — `dataset_listwise.py`: LabelListwiseRAGDataset

One dataset item = one source concept + all its top-k candidates in a single prompt.

```python
class LabelListwiseRAGDataset(Dataset):
    prompt = (
        "You are an ontology matching expert.\n\n"
        "Source concept: {source}\n\n"
        "Rank the following {n} candidate concepts from most to least likely to refer to "
        "the same real-world entity as the source concept.\n\n"
        "Candidates:\n{candidates}\n\n"
        "Output only the candidate numbers ranked from best to worst, separated by commas "
        "(e.g. \"3, 1, 5, 2, 4\"). No explanation."
    )
```

`__getitem__` returns: `{texts, source_iri, target_iris: List[str], ir_scores: List[float]}`
`collate_fn` batches these into lists (not tensors — iris and scores stay as Python lists).

---

## Step 2 — `rag_listwise.py`: LLM arch + RAG class + helpers

### 2a. `RAGBasedListwiseLLMArch(LLaMA2DecoderLLMArch)`
Overrides `generate(input_data)` to do **text generation** (not logit scoring):
```python
def generate(self, input_data):
    tokenized = self.tokenize(input_data)
    input_len = tokenized["input_ids"].shape[1]
    with torch.no_grad():
        outputs = self.model.generate(**tokenized, pad_token_id=self.tokenizer.eos_token_id,
                                      max_new_tokens=40, do_sample=False)
    new_tokens = outputs[:, input_len:]
    return self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    # Returns list of strings like ["3, 1, 5, 2, 4", ...]
```
No yes/no token setup in `__init__` — clean break from `RAGBasedDecoderLLMArch`.

### 2b. `RAGBasedListwiseInstructLLMArch(RAGBasedListwiseLLMArch)`
Adds chat template in `tokenize()`:
```python
formatted = [self.tokenizer.apply_chat_template(
    [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
    for text in input_data]
```

### 2c. `LLaMA3ListwiseDecoderLM(RAGBasedListwiseInstructLLMArch)`
LLaMA3-specific `load_tokenizer` + `load_model` (8-bit quantization, HF token).
Reuses same HF path and setup as `LLaMA3InstructDecoderLM` in `models.py`.

### 2d. `LLaMA3ListwiseBertRAG(ListwiseRAG)`
Sets `Retrieval = BERTRetrieval`, `LLM = LLaMA3ListwiseDecoderLM`.

### 2e. `parse_ranking(text, n_candidates) -> List[int]`
Parses "3, 1, 5, 2, 4" → [2, 0, 4, 1, 3] (0-indexed, position = rank).
Handles partial/invalid output by appending missing candidates at the end.

### 2f. `apply_rrf(ir_scores, llm_ranking, k=60) -> List[float]`
```
rrf[i] = 1/(k + ir_rank[i] + 1) + 1/(k + llm_rank[i] + 1)
```

### 2g. `ListwiseRAG(RAG)`
Overrides `llm_generate`:
1. `build_listwise_inputs(input_data, ir_output)` → groups candidates by source
2. `LabelListwiseRAGDataset` with `batch_size=1` (prompts are long)
3. For each batch: get LLM ranking text → parse → RRF → take top-1
4. Returns `[{"source": iri, "target": iri, "score": rrf_score}]`

Final output of `generate()` stays as `[{"ir-outputs": ...}, {"llm-output": rrf_predictions}]`
so existing evaluation machinery works.

---

## Step 3 — `test_rag_listwise.py`: Interactive test script

Standalone script (no pipeline catalog needed). Tests on `ncit-doid.disease`, first 20 source concepts.

Shows per source:
```
Source: 'Spinal Meningocele'
GT match(es) in top-5: ['meningocele']

Rank  IR Score  Candidate                     GT?   LLM Rank  RRF Score  Final?
1     0.8631    spinal meningioma                   3         0.0155
2     0.7840    meningocele               YES  1         0.0323     ← SELECTED
...

LLM raw output: "2, 1, 3, 5, 4"
Correct match selected: YES
```

Summary at the end: IR-only correct / RRF correct / improvement cases.

Args: `--model LLaMA3ListwiseBertRAG`, `--dataset ncit-doid`, `--n-sources 20`, `--top-k 5`

---

## Step 4 — `test_rag_listwise.batch`: SLURM script

Mirrors `test_rag.batch`:
- H100 node, 1 GPU
- `MODEL` env var
- Calls `python test_rag_listwise.py --model ${MODEL}`

---

## Verification
1. Submit: `sbatch --export=ALL,MODEL=LLaMA3ListwiseBertRAG test_rag_listwise.batch`
2. Check output:
   - LLM ranking texts look like "3, 1, 2, 5, 4" (not garbage)
   - `parse_ranking` handles edge cases (fewer numbers, out-of-range)
   - RRF scores are non-zero and differentiated (not all equal)
   - For `Spinal Meningocele` case: RRF selects rank-2 (meningocele) over IR rank-1 (spinal meningioma)
3. Compare summary: IR-only correct count vs RRF correct count over 20 sources

---

## Old Plan (Instruct models — partially done, lower priority)

Already done: `LLaMA3InstructBertRAG` ✓

Still pending (separate task, not this branch):
- `Gemma4_26B_A4B_itBertRAG` + `Qwen35_9BInstructBertRAG` (Steps 4–7 below)
- `pipeline_rag_instruct.py`

---

## Step 1 (old) — Add 3 instruct dataset classes to `dataset.py`

Add after existing dataset classes:

```python
class LabelRAGInstructDataset(RAGDataset):
    prompt = """You are an ontology matching expert. Determine whether the following two concepts refer to the same real-world entity.

Concept 1: {source}
Concept 2: {target}

Answer with exactly one word: yes or no."""

    def fill_one_sample(self, input_data: Any) -> str:
        source = self.preprocess(input_data["source"]["label"])
        target = self.preprocess(input_data["target"]["label"])
        return self.prompt.replace("{source}", source).replace("{target}", target)


class LabelParentRAGInstructDataset(RAGDataset):
    prompt = """You are an ontology matching expert. Determine whether the following two concepts refer to the same real-world entity.

Concept 1: {source}
Parents of Concept 1: {source_parents}

Concept 2: {target}
Parents of Concept 2: {target_parents}

Answer with exactly one word: yes or no."""

    def fill_one_sample(self, input_data: Any) -> str:
        source = self.preprocess(input_data["source"]["label"])
        target = self.preprocess(input_data["target"]["label"])
        source_parents = ", ".join([self.preprocess(p["label"]) for p in input_data["source"]["parents"]])
        target_parents = ", ".join([self.preprocess(p["label"]) for p in input_data["target"]["parents"]])
        return (self.prompt
            .replace("{source}", source)
            .replace("{target}", target)
            .replace("{source_parents}", source_parents)
            .replace("{target_parents}", target_parents))


class LabelChildrenRAGInstructDataset(RAGDataset):
    prompt = """You are an ontology matching expert. Determine whether the following two concepts refer to the same real-world entity.

Concept 1: {source}
Children of Concept 1: {source_children}

Concept 2: {target}
Children of Concept 2: {target_children}

Answer with exactly one word: yes or no."""

    def fill_one_sample(self, input_data: Any) -> str:
        source = self.preprocess(input_data["source"]["label"])
        target = self.preprocess(input_data["target"]["label"])
        source_children = ", ".join([self.preprocess(c["label"]) for c in input_data["source"]["childrens"]])
        target_children = ", ".join([self.preprocess(c["label"]) for c in input_data["target"]["childrens"]])
        return (self.prompt
            .replace("{source}", source)
            .replace("{target}", target)
            .replace("{source_children}", source_children)
            .replace("{target_children}", target_children))
```

---

## Step 2 — Add 3 instruct encoder classes to `encoder/rag.py`

Current encoder classes (for reference):
```python
class IRILabelInRAGEncoder(RAGEncoder):
    items_in_owl = "(Label)"
    retrieval_encoder = IRILabelInLightweightEncoder
    llm_encoder: str = "LabelRAGDataset"
```

Add after existing 3 classes:
```python
class IRILabelInRAGInstructEncoder(RAGEncoder):
    items_in_owl = "(Label)"
    retrieval_encoder = IRILabelInLightweightEncoder
    llm_encoder: str = "LabelRAGInstructDataset"

class IRILabelChildrensInRAGInstructEncoder(RAGEncoder):
    items_in_owl = "(Label, Children)"
    retrieval_encoder = IRILabelInLightweightEncoder
    llm_encoder: str = "LabelChildrenRAGInstructDataset"

class IRILabelParentsInRAGInstructEncoder(RAGEncoder):
    items_in_owl = "(Label, Parent)"
    retrieval_encoder = IRILabelInLightweightEncoder
    llm_encoder: str = "LabelParentRAGInstructDataset"
```

---

## Step 3 — Register instruct encoders in `encoder/__init__.py`

Import the 3 new encoder classes and add to `EncoderCatalog["rag"]`:

```python
# Add to imports from encoder.rag:
IRILabelInRAGInstructEncoder,
IRILabelChildrensInRAGInstructEncoder,
IRILabelParentsInRAGInstructEncoder,

# Add to EncoderCatalog["rag"]:
"label-instruct": IRILabelInRAGInstructEncoder,
"label-children-instruct": IRILabelChildrensInRAGInstructEncoder,
"label-parent-instruct": IRILabelParentsInRAGInstructEncoder,
```

---

## Step 4 — Add 3 LLM + 3 RAG instruct classes to `rag/models.py`

### LLaMA 3 8B Instruct
```python
class LLaMA3InstructDecoderLM(RAGBasedInstructDecoderLLMArch):
    tokenizer = AutoTokenizer
    model = AutoModelForCausalLM
    path = "meta-llama/Meta-Llama-3-8B-Instruct"

    def __str__(self):
        return super().__str__() + "-LLaMA-3-8B-Instruct"

    def load_tokenizer(self) -> None:
        self.tokenizer = self.tokenizer.from_pretrained(
            self.path,
            token=os.environ["HUGGINGFACE_ACCESS_TOKEN"],
            padding_side="left",
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_model(self) -> None:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        self.model = self.model.from_pretrained(
            self.path,
            quantization_config=quantization_config,
            device_map="balanced",
            token=os.environ["HUGGINGFACE_ACCESS_TOKEN"],
        )


class LLaMA3InstructBertRAG(RAG):
    Retrieval = BERTRetrieval
    LLM = LLaMA3InstructDecoderLM

    def __str__(self):
        return super().__str__() + "-LLaMA3InstructBertRAG"
```

### Gemma 4 26B-A4B Instruct (`enable_thinking=False` required)
```python
class Gemma4_26B_A4B_itDecoderLM(RAGBasedInstructDecoderLLMArch):
    tokenizer = AutoTokenizer
    model = AutoModelForCausalLM
    path = "google/gemma-4-26B-A4B-it"

    def __str__(self):
        return super().__str__() + "-Gemma-4-26B-A4B-it"

    def load_tokenizer(self) -> None:
        self.tokenizer = self.tokenizer.from_pretrained(
            self.path,
            padding_side="left",
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_model(self) -> None:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        self.model = self.model.from_pretrained(
            self.path,
            quantization_config=quantization_config,
            device_map="balanced",
            attn_implementation="eager",
        )

    def tokenize(self, input_data: List) -> Any:
        formatted = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for text in input_data
        ]
        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
            truncation=self.kwargs["truncation"],
            max_length=self.kwargs["tokenizer_max_length"],
            padding=self.kwargs["padding"],
        )
        inputs.to(self.kwargs["device"])
        return inputs


class Gemma4_26B_A4B_itBertRAG(RAG):
    Retrieval = BERTRetrieval
    LLM = Gemma4_26B_A4B_itDecoderLM

    def __str__(self):
        return super().__str__() + "-Gemma4_26B_A4B_itBertRAG"
```

### Qwen3.5 9B Instruct (no BOS + `enable_thinking=False`)
```python
class Qwen35_9BInstructDecoderLM(RAGBasedInstructDecoderLLMArch):
    tokenizer = AutoTokenizer
    model = AutoModelForCausalLM
    path = "Qwen/Qwen3.5-9B"

    def __str__(self):
        return super().__str__() + "-Qwen3.5-9B-Instruct"

    def load_tokenizer(self) -> None:
        self.tokenizer = self.tokenizer.from_pretrained(
            self.path,
            token=os.environ["HUGGINGFACE_ACCESS_TOKEN"],
            padding_side="left",
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def check_answer_set_tokenizer(self, answer: str) -> bool:
        return len(self.tokenizer(answer).input_ids) == 1

    def tokenize(self, input_data: List) -> Any:
        formatted = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for text in input_data
        ]
        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
            truncation=self.kwargs["truncation"],
            max_length=self.kwargs["tokenizer_max_length"],
            padding=self.kwargs["padding"],
        )
        inputs.to(self.kwargs["device"])
        return inputs


class Qwen35_9BInstructBertRAG(RAG):
    Retrieval = BERTRetrieval
    LLM = Qwen35_9BInstructDecoderLM

    def __str__(self):
        return super().__str__() + "-Qwen35_9BInstructBertRAG"
```

---

## Step 5 — Register in `__init__.py`

Add to imports from `rag.models`:
```python
LLaMA3InstructBertRAG,
Gemma4_26B_A4B_itBertRAG,
Qwen35_9BInstructBertRAG,
```

Add to `MatcherCatalog["rag"]`:
```python
"LLaMA3InstructBertRAG": LLaMA3InstructBertRAG,
"Gemma4_26B_A4B_itBertRAG": Gemma4_26B_A4B_itBertRAG,
"Qwen35_9BInstructBertRAG": Qwen35_9BInstructBertRAG,
```

---

## Step 6 — Register in `configs.py`

Add to `rag_icv_models` list:
```python
"LLaMA3InstructBertRAG", "Gemma4_26B_A4B_itBertRAG", "Qwen35_9BInstructBertRAG",
```

---

## Step 7 — Create `pipeline_rag_instruct.py`

New pipeline script identical to `pipeline_rag.py` but with instruct encoders:
```python
"approach-encoders-to-consider": ["label-instruct", "label-children-instruct", "label-parent-instruct"],
```

Use with:
```bash
sbatch --export=ALL,MODEL=LLaMA3InstructBertRAG pipeline_rag_instruct.batch
```

---

## Verification
- Run `test_rag.py --model LLaMA3InstructBertRAG` on a P100 node
- Confirm formatted prompt contains `<|start_header_id|>` (LLaMA 3 chat markers)
- Confirm yes/no token IDs are found (non-empty `answer_sets_token_id`)
- Confirm confidence scores returned (not all 1.0)
- Same for Qwen3.5 instruct and Gemma4 it (H100 only)
