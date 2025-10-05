from . import config
from .data import SpanishPoetryDataset

from .tokenization import Tokenization, Tokenizer


def main():
    ds = SpanishPoetryDataset(config.DATASET_URL, Tokenizer(Tokenization.WORD))
    print(ds[0])


if __name__ == "__main__":
    main()
