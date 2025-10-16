import typer
from typing import Annotated
from safetensors.torch import load_file
from pathlib import Path
import json

from ..core import config
from ..core.tokenization import (
    TokenizerType,
    TokenizerConfig,
    Tokenizer,
    WordTokenizer,
    CharTokenizer,
)
from ..core.model import GPTModel, GPTConfig
from ..core.generation import generate

inference_app = typer.Typer()


def decode_tokenizer_config(dct: dict) -> dict:
    if "itos" and "stoi" in dct:
        dct["itos"] = {int(k): v for k, v in dct["itos"].items()}
        dct["stoi"] = {k: int(v) for k, v in dct["stoi"].items()}

    return dct


def load_model_tokenizer() -> tuple[GPTModel, Tokenizer]:
    models_folder = Path(config.MODELS_FOLDER)

    if not models_folder.exists():
        raise FileNotFoundError(f"Models folder not found: {models_folder}")

    models_dirs = [d for d in models_folder.iterdir() if d.is_dir()]

    if len(models_dirs) == 0:
        raise FileNotFoundError("No models found")

    model_dir = max(models_dirs)

    model_path = model_dir / f"{model_dir.name}.safetensors"
    config_path = model_dir / "config.json"
    tokenizer_config_path = model_dir / "tokenizer.json"

    with open(config_path, "r") as f:
        config_data = json.load(f)

    loaded_state_dict = load_file(model_path)
    gpt_config = GPTConfig(**config_data)
    model = GPTModel(gpt_config).to(config.DEVICE)
    model.load_state_dict(loaded_state_dict)

    with open(tokenizer_config_path, "r") as f:
        tokenizer_config_data = json.load(f, object_hook=decode_tokenizer_config)

    tokenizer_config = TokenizerConfig(**tokenizer_config_data)

    match tokenizer_config.tk_type:
        case TokenizerType.WORD.value:
            tokenizer = WordTokenizer(tokenizer_config)
        case TokenizerType.CHAR.value:
            tokenizer = CharTokenizer(tokenizer_config)

    return model, tokenizer


@inference_app.command(name="inference", help="Run the inference")
def inference_cli(
    temperature: Annotated[
        float, typer.Option("--temperature", "-t", help="The temperature")
    ] = config.TEMPERATURE,
    top_k: Annotated[
        int, typer.Option("--top-k", "-k", help="The top k")
    ] = config.TOP_K,
    gen_limit: Annotated[
        int, typer.Option("--gen-limit", "-l", help="The generation limit")
    ] = config.GEN_LIMIT,
) -> None:
    if config.MAX_SAMPLES:
        print(f"Max samples: {config.MAX_SAMPLES}")

    model, tokenizer = load_model_tokenizer()

    generate(
        model=model,
        tokenizer=tokenizer,
        top_k=top_k,
        temperature=temperature,
        device=config.DEVICE,
        block_size=config.BLOCK_SIZE,
        gen_limit=gen_limit,
    )
