from datasets import load_dataset, Dataset as HFDataset
from torch.utils.data import Dataset as TorchDataset
import torch

from .tokenization import Tokenizer


class SpanishPoetryDataset(TorchDataset):
    """A PyTorch dataset for Spanish poetry."""

    def __init__(self, corpus: str, tokenizer: Tokenizer, block_size: int):
        """Initializes the dataset.

        Args:
            corpus (str): The corpus to use.
            tokenizer (Tokenizer): The tokenizer to use.
            block_size (int): The block size for the model.
        """
        self.tokens = tokenizer.encode(corpus)
        self.block_size = block_size

    def __len__(self):
        """Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns a single item from the dataset.

        Args:
            idx (int): The index of the item to return.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the input and target tensors.
        """
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
    """Loads a dataset from the Hugging Face Hub.

    Args:
        url (str): The URL of the dataset.
        column_name (str): The name of the column containing the text.
        init_token (str): The initial token to add to each text.
        end_token (str): The end token to add to each text.
        max_samples (int | None, optional): The maximum number of samples to load. Defaults to None.

    Returns:
        str: The loaded corpus as a single string.
    """
    ds = load_dataset(url, split="train")
    assert isinstance(ds, HFDataset)

    if max_samples and max_samples > 0:
        ds = ds.select(range(max_samples))

    texts = [
        f"{init_token}\n{text.strip()}\n{end_token}\n\n"
        for text in ds[column_name]
        if text
    ]

    return "".join(texts)
