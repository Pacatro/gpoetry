from gpoetry.main import SpanishPoetryDataset
from gpoetry import config


def test_dataset_loads():
    """Test that the dataset loads correctly"""
    ds = SpanishPoetryDataset(config.DATASET_URL)
    assert len(ds) > 0


def test_dataset_getitem():
    ds = SpanishPoetryDataset(config.DATASET_URL)
    item = ds[0]
    assert isinstance(item, str)
    assert len(item) > 0
