from . import config
from .data import SpanishPoetryDataset, load_from_file, load_poems
from .tokenization import WordTokenizer, CharTokenizer, TokenizerType
from .model import GPoeTry
from .train import train


def main():
    print(f"Using device: {config.DEVICE}")
    print(f"Using dataset: {config.DATASET_URL}")
    print(f"Using max samples: {config.MAX_SAMPLES}")
    print(
        f"Using model config: num_heads={config.NUM_HEADS}, num_layers={config.NUM_LAYERS}, block_size={config.BLOCK_SIZE}, emb_dim={config.EMB_DIM}, dropout_p={config.DROPOUT_P}"
    )
    print(
        f"Using training config: epochs={config.EPOCHS}, batch_size={config.BATCH_SIZE}, lr={config.LR}"
    )

    match config.TOKENIZER_TYPE:
        case TokenizerType.WORD:
            tokenizer = WordTokenizer()
        case TokenizerType.CHAR:
            tokenizer = CharTokenizer()

    # texts = load_poems(config.DATASET_URL, max_samples=config.MAX_SAMPLES)
    texts = load_from_file("data/datos_sancho_mini.txt")

    tokenizer.fit(texts)

    ds = SpanishPoetryDataset(texts=texts, tokenizer=tokenizer)

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
        train_size=config.TRAIN_SIZE,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        lr=config.LR,
        device=config.DEVICE,
    )


if __name__ == "__main__":
    main()
