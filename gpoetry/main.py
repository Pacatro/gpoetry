import torch
import json
from pathlib import Path
from dataclasses import asdict

from . import config
from .data import SpanishPoetryDataset, load_from_hf
from .model import GPTModel, GPTConfig
from .tokenization import CharTokenizer, TokenizerType, WordTokenizer
from .train import train
from .generation import generate


def save_model_config(gpt_config: GPTConfig) -> None:
    with open(config.MODEL_CONFIG_PATH, "w") as f:
        json.dump(asdict(gpt_config), f, indent=4)


def main():
    print(f"Device: {config.DEVICE}")

    match config.TOKENIZER_TYPE:
        case TokenizerType.WORD:
            tokenizer = WordTokenizer()
        case TokenizerType.CHAR:
            tokenizer = CharTokenizer()

    texts = load_from_hf(
        config.DATASET_URL,
        init_token=config.INIT_TOKEN,
        end_token=config.END_TOKEN,
        column_name=config.DATASET_TEXT_COLUMN,
        max_samples=config.MAX_SAMPLES,
    )

    tokenizer.fit(texts)

    ds = SpanishPoetryDataset(
        text=texts, tokenizer=tokenizer, block_size=config.BLOCK_SIZE
    )

    print("Dataset length:", len(ds))

    gpt_config = GPTConfig(vocab_size=tokenizer.vocab_size)

    print(f"Vocab size: {gpt_config.vocab_size}")
    print(f"Heads: {gpt_config.num_heads}")
    print(f"Layers: {gpt_config.num_layers}")
    print(f"Block size: {gpt_config.block_size}")
    print(f"Embbeding dim: {gpt_config.emb_dim}")
    print(f"p: {gpt_config.p}")

    model = GPTModel(config=gpt_config).to(config.DEVICE)

    # FIXME: Can't compile the model because my GPU is too old
    # model.compile()

    print("Model parameters:", sum(p.numel() for p in model.parameters()))

    if not Path(config.MODEL_PATH).exists():
        print("Training model...")
        train(
            model,
            ds,
            train_size=config.TRAIN_SIZE,
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE,
            lr=config.LR,
            device=config.DEVICE,
        )

        model.eval()
        torch.save(model.state_dict(), config.MODEL_PATH)
        save_model_config(gpt_config)

    print("Generating...\n")
    generate(
        model_path=config.MODEL_PATH,
        gpt_config_path=config.MODEL_CONFIG_PATH,
        tokenizer=tokenizer,
        temperature=config.TEMPERATURE,
        top_k=config.TOP_K,
        device=config.DEVICE,
        block_size=config.BLOCK_SIZE,
    )


if __name__ == "__main__":
    main()
