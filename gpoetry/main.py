from . import config
from .data import SpanishPoetryDataset, load_from_hf, load_from_txt, DatasetType
from .model import GPTModel, GPTConfig
from .tokenization import CharTokenizer, TokenizerType, WordTokenizer
from .train import train


def main():
    match config.TOKENIZER_TYPE:
        case TokenizerType.WORD:
            tokenizer = WordTokenizer()
        case TokenizerType.CHAR:
            tokenizer = CharTokenizer()

    match config.DATASET_TYPE:
        case DatasetType.HUGGINGFACE:
            texts = load_from_hf(
                config.DATASET_URL,
                column_name=config.DATASET_TEXT_COLUMN,
                max_samples=config.MAX_SAMPLES,
            )
        case DatasetType.TXT:
            texts = load_from_txt("data/datos_sancho_mini.txt")

    tokenizer.fit(texts)

    ds = SpanishPoetryDataset(texts=texts, tokenizer=tokenizer)

    print("Dataset length:", len(ds))

    gpt_config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
        block_size=config.BLOCK_SIZE,
        emb_dim=config.EMB_DIM,
        p=config.DROPOUT_P,
    )

    print("Model configuration:")
    print(f"Vocab size: {gpt_config.vocab_size}")
    print(f"Heads: {gpt_config.num_heads}")
    print(f"Layers: {gpt_config.num_layers}")
    print(f"BLocak size: {gpt_config.block_size}")
    print(f"Embbeding dim: {gpt_config.emb_dim}")
    print(f"p: {gpt_config.p}")

    model = GPTModel(config=gpt_config)
    print(f"Model:\n{model}")

    print("Model parameters:", sum(p.numel() for p in model.parameters()))

    train(
        model,
        ds,
        train_size=config.TRAIN_SIZE,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        lr=config.LR,
        device=config.DEVICE,
        block_size=config.BLOCK_SIZE,
    )


if __name__ == "__main__":
    main()
