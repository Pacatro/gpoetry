from datasets import load_dataset, Dataset as HFDataset
from torch.utils.data import Dataset as TorchDataset
import torch
from enum import Enum

from .tokenization import Tokenizer


class DatasetType(Enum):
    HUGGINGFACE = "huggingface"
    TXT = "txt"


class SpanishPoetryDataset(TorchDataset):
    def __init__(self, texts: list[str], tokenizer: Tokenizer):
        self.dataset = [tokenizer.encode(text) for text in texts]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        return torch.tensor(self.dataset[idx], dtype=torch.long)


def load_from_hf(
    url: str, column_name: str, max_samples: int | None = None
) -> list[str]:
    ds = load_dataset(url, split="train")
    assert isinstance(ds, HFDataset)

    if max_samples and max_samples > 0:
        ds = ds.select(range(max_samples))

    return [text for text in ds[column_name] if text]


def load_from_txt(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    parts = content.split("<|inicio|>")

    poems = []
    for part in parts[1:]:
        if "<|fin|>" in part:
            poem = part.split("<|fin|>")[0].strip()
            poems.append(poem)

    return poems
