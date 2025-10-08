from datasets import load_dataset, Dataset as HFDataset
from torch.utils.data import Dataset as TorchDataset
import torch

from .tokenization import Tokenizer


class SpanishPoetryDataset(TorchDataset):
    def __init__(
        self,
        hf_url: str,
        tokenizer: Tokenizer,
        max_samples: int | None = None,
    ):
        ds = load_dataset(hf_url, split="train")
        assert isinstance(ds, HFDataset)

        if max_samples and max_samples > 0:
            ds = ds.select(range(max_samples))

        texts = [text for text in ds["content"] if text]

        tokenizer.fit(texts)

        self.dataset = [tokenizer.encode(text) for text in texts]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        return torch.tensor(self.dataset[idx], dtype=torch.long)
