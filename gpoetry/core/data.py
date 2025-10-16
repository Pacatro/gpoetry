from datasets import load_dataset, Dataset as HFDataset
from torch.utils.data import Dataset as TorchDataset
import torch

from .tokenization import Tokenizer


class SpanishPoetryDataset(TorchDataset):
    def __init__(self, corpus: str, tokenizer: Tokenizer, block_size: int):
        self.tokens = tokenizer.encode(corpus)
        self.block_size = block_size

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # We use all tokens except the last one (x) for training the model
        # Then we use all tokens (y) to use it in the loss function
        x = torch.tensor(self.tokens[idx : idx + self.block_size], dtype=torch.long)
        y = torch.tensor(
            self.tokens[idx + 1 : idx + self.block_size + 1], dtype=torch.long
        )
        return x, y


def load_from_hf(
    url: str,
    column_name: str,
    init_token: str,
    end_token: str,
    max_samples: int | None = None,
) -> str:
    ds = load_dataset(url, split="train")
    assert isinstance(ds, HFDataset)

    if max_samples and max_samples > 0:
        ds = ds.select(range(max_samples))

    texts = [f"{init_token}\n{text}\n{end_token}" for text in ds[column_name] if text]

    return "".join(texts)
