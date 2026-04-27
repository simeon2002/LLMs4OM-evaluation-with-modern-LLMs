# -*- coding: utf-8 -*-
from typing import Any, List

from torch.utils.data import Dataset


class LabelListwiseRAGDataset(Dataset):
    prompt = (
        "You are an ontology matching expert.\n\n"
        "Source concept: {source}\n\n"
        "Rank the following {n} candidate concepts from most to least likely to refer to "
        "the same real-world entity as the source concept.\n\n"
        "Candidates:\n{candidates}\n\n"
        "You MUST output all {n} numbers ranked from best to worst, separated by commas "
        "(e.g. \"3, 1, 5, 2, 4\"). Output exactly {n} numbers. No explanation."
    )

    def __init__(self, data: List):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def preprocess(self, text: str) -> str:
        return text.replace("_", " ").lower()

    def fill_one_sample(self, item: Any) -> str:
        source = self.preprocess(item["source"]["label"])
        n = len(item["targets"])
        cand_lines = "\n".join(
            f"{i + 1}. {self.preprocess(t['label'])}"
            for i, t in enumerate(item["targets"])
        )
        return (
            self.prompt
            .replace("{source}", source)
            .replace("{n}", str(n))
            .replace("{candidates}", cand_lines)
        )

    def __getitem__(self, index: int):
        item = self.data[index]
        return {
            "texts": self.fill_one_sample(item),
            "source_iri": item["source_iri"],
            "target_iris": item["target_iris"],
            "ir_scores": item["ir_scores"],
        }

    def collate_fn(self, batches: List) -> dict:
        return {
            "texts": [b["texts"] for b in batches],
            "source_iris": [b["source_iri"] for b in batches],
            "target_iris": [b["target_iris"] for b in batches],
            "ir_scores": [b["ir_scores"] for b in batches],
        }
