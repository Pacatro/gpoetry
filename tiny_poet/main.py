from datasets import load_dataset, Dataset as HFDataset
from torch.utils.data import Dataset as TorchDataset

from . import config


class SpanishPoetryDataset(TorchDataset):
    def __init__(self, hf_url: str, max_samples: int | None = None):
        ds = load_dataset(hf_url, split="train")
        assert isinstance(ds, HFDataset)

        if max_samples and max_samples > 0:
            ds = ds.select(range(max_samples))

        ds.set_format(type="torch", columns=["content"])

        self.dataset = ds

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        return self.dataset[idx]["content"]


def main():
    ds = SpanishPoetryDataset(config.DATASET_URL)
    print(ds[0])


if __name__ == "__main__":
    main()
