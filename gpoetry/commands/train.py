import typer
from typing import Annotated

from ..core.generation import generate
from ..core import config
from ..core.data import SpanishPoetryDataset, load_from_hf
from ..core.model import GPTConfig, GPTModel
from ..core.tokenization import TokenizerConfig, TokenizerType, get_tokenizer
from ..core.training import train
from ..core.model_io import save_model

train_app = typer.Typer()


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
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Print verbose output")
    ] = False,
) -> None:
    """Trains the model.

    Args:
        tokenization (Annotated[TokenizerType, typer.Option]): The tokenizer type.
        max_samples (Annotated[int | None, typer.Option, optional): The maximum number of samples from the dataset. Defaults to config.MAX_SAMPLES.
        batch_size (Annotated[int, typer.Option, optional): The training batch size. Defaults to config.BATCH_SIZE.
        epochs (Annotated[int, typer.Option, optional): The number of epochs. Defaults to config.EPOCHS.
        lr (Annotated[float, typer.Option, optional): The learning rate. Defaults to config.LR.
        train_size (Annotated[float, typer.Option, optional): The training size. Defaults to config.TRAIN_SIZE.
    """

    tokenizer = get_tokenizer(TokenizerConfig(tk_type=tokenization.value))

    corpus = load_from_hf(
        config.DATASET_URL,
        init_token=config.INIT_TOKEN,
        end_token=config.END_TOKEN,
        column_name=config.DATASET_TEXT_COLUMN,
        max_samples=max_samples,
    )

    tokenizer.fit(corpus)

    ds = SpanishPoetryDataset(
        corpus=corpus, tokenizer=tokenizer, block_size=config.BLOCK_SIZE
    )

    gpt_config = GPTConfig(vocab_size=tokenizer.config.vocab_size)
    model = GPTModel(config=gpt_config).to(config.DEVICE)

    # Can't compile the model because my GPU is too old :'D
    # model.compile()

    if verbose:
        print(f"Device: {config.DEVICE}")

        if max_samples:
            print(f"Max samples: {max_samples}")

        print(f"Vocab size: {gpt_config.vocab_size}")
        print(f"Heads: {gpt_config.num_heads}")
        print(f"Layers: {gpt_config.num_layers}")
        print(f"Block size: {gpt_config.block_size}")
        print(f"Embbeding dim: {gpt_config.emb_dim}")
        print(f"p: {gpt_config.p}")
        print(f"Tokenizer: {tokenization.value}")

        print("Model parameters:", sum(p.numel() for p in model.parameters()))

    print("Training model...")

    try:
        train(
            model,
            ds,
            train_size=train_size,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=config.DEVICE,
        )
    except KeyboardInterrupt:
        generate(
            model=model,
            init_text=config.INIT_TOKEN,
            tokenizer=tokenizer,
            device=config.DEVICE,
            temperature=config.TEMPERATURE,
            top_k=config.TOP_K,
            gen_limit=200,
            block_size=config.BLOCK_SIZE,
        )

    save_model(model, tokenizer)
