from . import config
from .data import SpanishPoetryDataset
from .tokenization import Tokenization, Tokenizer
from .model import GPoeTry
from .train import train


def main():
    print(f"Using device: {config.DEVICE}")
    tokenizer = Tokenizer(Tokenization.WORD)
    ds = SpanishPoetryDataset(config.DATASET_URL, tokenizer, max_samples=10)

    model = GPoeTry(
        vocab_size=tokenizer.vocab_size,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
        block_size=config.BLOCK_SIZE,
        emb_dim=config.EMB_DIM,
        p=config.DROPOUT_P,
    )

    train(
        model,
        ds,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        lr=config.LR,
        device=config.DEVICE,
    )


if __name__ == "__main__":
    main()
