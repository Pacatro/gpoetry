import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from safetensors.torch import save_file

from ..core import config
from ..core.data import SpanishPoetryDataset, load_from_hf
from ..core.model import GPTConfig, GPTModel
from ..core.tokenization import (
    CharTokenizer,
    TokenizerConfig,
    TokenizerType,
    WordTokenizer,
)
from ..core.training import train

train_app = typer.Typer()


def save_model(model: GPTModel, tokenizer_config: TokenizerConfig) -> None:
    model_name = f"gpoetry_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_dir = Path(config.MODELS_FOLDER) / model_name

    model_dir.mkdir(exist_ok=True, parents=True)
    save_file(model.state_dict(), model_dir / f"{model_name}.safetensors")

    with open(model_dir / "config.json", "w") as f:
        json.dump(asdict(model.config), f, indent=4)

    with open(model_dir / "tokenizer.json", "w") as f:
        json.dump(asdict(tokenizer_config), f, indent=4)


@train_app.command(name="train", help="Train the model")
def train_cli(
    tokenization: Annotated[
        TokenizerType, typer.Option("--tokenization", "-t", help="The tokenizer type")
    ],
    max_samples: Annotated[
        int | None,
        typer.Option(
            "--max-samples", "-m", help="The maximum number of samples from the dataset"
        ),
    ] = config.MAX_SAMPLES,
    batch_size: Annotated[
        int, typer.Option("--batch-size", "-b", help="The training batch_size")
    ] = config.BATCH_SIZE,
    epochs: Annotated[
        int, typer.Option("--epochs", "-e", help="The number of epochs")
    ] = config.EPOCHS,
    lr: Annotated[
        float, typer.Option("--lr", "-l", help="The learning rate")
    ] = config.LR,
    train_size: Annotated[
        float, typer.Option("--train-size", "-s", help="The training size")
    ] = config.TRAIN_SIZE,
) -> None:
    print(f"Device: {config.DEVICE}")

    match tokenization:
        case TokenizerType.WORD:
            tokenizer = WordTokenizer()
        case TokenizerType.CHAR:
            tokenizer = CharTokenizer()

    print(f"Tokenizer: {tokenization.value}")

    corpus = load_from_hf(
        config.DATASET_URL,
        init_token=config.INIT_TOKEN,
        end_token=config.END_TOKEN,
        column_name=config.DATASET_TEXT_COLUMN,
        max_samples=max_samples,
    )

    if max_samples:
        print(f"Max samples: {max_samples}")

    tokenizer.fit(corpus)

    ds = SpanishPoetryDataset(
        corpus=corpus, tokenizer=tokenizer, block_size=config.BLOCK_SIZE
    )

    gpt_config = GPTConfig(vocab_size=tokenizer.config.vocab_size)

    print(f"Vocab size: {gpt_config.vocab_size}")
    print(f"Heads: {gpt_config.num_heads}")
    print(f"Layers: {gpt_config.num_layers}")
    print(f"Block size: {gpt_config.block_size}")
    print(f"Embbeding dim: {gpt_config.emb_dim}")
    print(f"p: {gpt_config.p}")

    model = GPTModel(config=gpt_config).to(config.DEVICE)

    # Can't compile the model because my GPU is too old :'D
    # model.compile()

    print("Model parameters:", sum(p.numel() for p in model.parameters()))

    print("Training model...")
    train(
        model,
        ds,
        train_size=train_size,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=config.DEVICE,
    )

    save_model(model, tokenizer.config)
