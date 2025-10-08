from gpoetry.main import SpanishPoetryDataset
from gpoetry import config
from gpoetry.tokenization import WordTokenizer
import torch


def test_dataset_loads():
    """Test that the dataset loads correctly and is not empty."""
    tokenizer = WordTokenizer()
    ds = SpanishPoetryDataset(config.DATASET_URL, tokenizer=tokenizer)
    assert len(ds) > 0


def test_dataset_getitem():
    """Test that __getitem__ returns a tensor of token indices."""
    tokenizer = WordTokenizer()
    ds = SpanishPoetryDataset(config.DATASET_URL, tokenizer=tokenizer)
    item = ds[0]
    assert isinstance(item, torch.Tensor)
    assert item.dim() == 1
    assert item.numel() > 0
