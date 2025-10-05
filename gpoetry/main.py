from . import config
from .data import SpanishPoetryDataset


def main():
    ds = SpanishPoetryDataset(config.DATASET_URL)
    print(ds[0])


if __name__ == "__main__":
    main()
