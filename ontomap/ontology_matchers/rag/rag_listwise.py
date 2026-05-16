# -*- coding: utf-8 -*-
import re
from typing import Any, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ontomap.ontology_matchers.llm.llm import LLaMA2DecoderLLMArch
from ontomap.ontology_matchers.rag.dataset_listwise import (
    LabelListwiseRAGDataset,
    LabelParentListwiseRAGDataset,
    LabelChildrenListwiseRAGDataset,
)
from ontomap.ontology_matchers.rag.rag import RAG
from ontomap.ontology_matchers.retrieval.models import BERTRetrieval


# ── Helpers ──────────────────────────────────────────────────────────────────

def parse_ranking(text: str, n_candidates: int):
    """Parse LLM output into a 0-indexed ranking list, or None if the model abstained.

    Returns None if the model output '0' (no match).
    Otherwise returns a list of length n_candidates where index 0 is the best candidate.
    Missing or out-of-range numbers are appended at the end as fallback.
    """
    tokens = re.findall(r"\d+", text.strip())
    # Abstention: model output only "0"
    if tokens and tokens[0] == "0" and all(t == "0" for t in tokens):
        return None
    seen: set = set()
    ranking: List[int] = []
    for token in tokens:
        idx = int(token) - 1  # 1-indexed → 0-indexed
        if 0 <= idx < n_candidates and idx not in seen:
            ranking.append(idx)
            seen.add(idx)
    for i in range(n_candidates):
        if i not in seen:
            ranking.append(i)
    return ranking  # ranking[rank_position] = candidate_index


# ── LLM architectures ────────────────────────────────────────────────────────

class RAGBasedListwiseLLMArch(LLaMA2DecoderLLMArch):
    """Text-generating LLM for listwise ranking.

    Skips yes/no logit extraction entirely — decodes the generated tokens
    directly to get the ranking string (e.g. "3, 1, 5, 2, 4").
    """

    def __str__(self):
        return "RAGBasedListwiseLLMArch"

    def generate(self, input_data: List) -> List[str]:
        tokenized = self.tokenize(input_data)
        input_len = tokenized["input_ids"].shape[1]
        with torch.no_grad():
            outputs = self.model.generate(
                **tokenized,
                pad_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=30,
                do_sample=False,
            )
        new_tokens = outputs[:, input_len:]
        return self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)


class RAGBasedListwiseInstructLLMArch(RAGBasedListwiseLLMArch):
    """Listwise LLM arch with chat-template wrapping for instruct models."""

    def __str__(self):
        return "RAGBasedListwiseInstructLLMArch"

    def tokenize(self, input_data: List) -> Any:
        formatted = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
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


class LLaMA3ListwiseDecoderLM(RAGBasedListwiseInstructLLMArch):
    path = "meta-llama/Meta-Llama-3-8B-Instruct"

    def __str__(self):
        return super().__str__() + "-LLaMA3-8B-Instruct-Listwise"

    def load_tokenizer(self) -> None:
        import os
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.path,
            token=os.environ["HUGGINGFACE_ACCESS_TOKEN"],
            padding_side="left",
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_model(self) -> None:
        import os
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.path,
            quantization_config=quantization_config,
            device_map="balanced",
            token=os.environ["HUGGINGFACE_ACCESS_TOKEN"],
        )


class Gemma4ListwiseDecoderLM(RAGBasedListwiseLLMArch):
    """Gemma 4 26B-A4B-it listwise LLM — uses AutoProcessor (not AutoTokenizer)."""
    path = "google/gemma-4-26B-A4B-it"

    def __str__(self):
        return super().__str__() + "-Gemma4-26B-A4B-it-Listwise"

    def load_tokenizer(self) -> None:
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.path,
            padding_side="left",
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_model(self) -> None:
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(
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


# ── RAG class ─────────────────────────────────────────────────────────────────

class ListwiseRAG(RAG):
    """RAG variant using listwise ranking + RRF instead of pairwise yes/no."""

    def __str__(self):
        return "ListwiseRAG"

    def generate(self, input_data: List) -> List:
        # Skip preprocess_ir_outputs — listwise needs raw target-cands/score-cands format
        ir_output = self.ir_generate(input_data=input_data)
        llm_predictions = self.llm_generate(input_data=input_data, ir_output=ir_output)
        return [{"ir-outputs": ir_output}, {"llm-output": llm_predictions}]

    def build_listwise_inputs(self, input_data: Any, ir_output: Any) -> List:
        source_onto = input_data["task-args"]["source"]
        target_onto = input_data["task-args"]["target"]
        s2i = input_data["source-onto-iri2index"]
        t2i = input_data["target-onto-iri2index"]

        listwise_inputs = []
        for item in ir_output:
            source_iri = item["source"]
            source_concept = source_onto[s2i[source_iri]]
            target_concepts = [target_onto[t2i[iri]] for iri in item["target-cands"]]
            listwise_inputs.append({
                "source": source_concept,
                "targets": target_concepts,
                "source_iri": source_iri,
                "target_iris": item["target-cands"],
                "ir_scores": item["score-cands"],
            })
        return listwise_inputs

    def llm_generate(self, input_data: Any, ir_output: Any) -> List:
        listwise_inputs = self.build_listwise_inputs(input_data, ir_output)
        dataset = eval(input_data["llm-encoder"])(data=listwise_inputs)
        dataloader = DataLoader(
            dataset,
            batch_size=self.kwargs["llm-config"]["batch_size"],
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )

        predictions = []
        for batch in tqdm(dataloader):
            texts = batch["texts"]
            source_iris = batch["source_iris"]
            target_iris_list = batch["target_iris"]

            ranking_texts = self.LLM.generate(texts)

            for source_iri, target_iris, ranking_text in zip(
                source_iris, target_iris_list, ranking_texts
            ):
                n = len(target_iris)
                llm_ranking = parse_ranking(ranking_text, n)
                if llm_ranking is None:
                    # LLM abstained — no match for this source
                    continue
                # llm_ranking[rank_pos] = candidate_idx → invert to get rank of each candidate
                llm_rank_of_candidate = [0] * n
                for rank_pos, cand_idx in enumerate(llm_ranking):
                    llm_rank_of_candidate[cand_idx] = rank_pos
                for i, tgt in enumerate(target_iris):
                    predictions.append({
                        "source": source_iri,
                        "target": tgt,
                        "score": llm_rank_of_candidate[i],  # 0-indexed rank, 0 = best
                    })

        return predictions


class LLaMA3ListwiseBertRAG(ListwiseRAG):
    Retrieval = BERTRetrieval
    LLM = LLaMA3ListwiseDecoderLM

    def __str__(self):
        return super().__str__() + "-LLaMA3ListwiseBertRAG"


class Gemma4ListwiseBertRAG(ListwiseRAG):
    Retrieval = BERTRetrieval
    LLM = Gemma4ListwiseDecoderLM

    def __str__(self):
        return super().__str__() + "-Gemma4ListwiseBertRAG"


class Qwen35_9BListwiseDecoderLM(RAGBasedListwiseLLMArch):
    path = "Qwen/Qwen3.5-9B"

    def __str__(self):
        return super().__str__() + "-Qwen3.5-9B-Listwise"

    def load_tokenizer(self) -> None:
        import os
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.path,
            token=os.environ.get("HUGGINGFACE_ACCESS_TOKEN"),
            padding_side="left",
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_model(self) -> None:
        import os
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.path,
            quantization_config=quantization_config,
            device_map="balanced",
            attn_implementation="eager",
            token=os.environ.get("HUGGINGFACE_ACCESS_TOKEN"),
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


class Qwen35_9BListwiseBertRAG(ListwiseRAG):
    Retrieval = BERTRetrieval
    LLM = Qwen35_9BListwiseDecoderLM

    def __str__(self):
        return super().__str__() + "-Qwen35_9BListwiseBertRAG"
