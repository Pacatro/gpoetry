import json
from pathlib import Path
from dataclasses import asdict
from datetime import datetime
from safetensors.torch import save_file, load_file

from .model import GPTModel, GPTConfig
from .tokenization import (
    TokenizerConfig,
    Tokenizer,
    TokenizerType,
    WordTokenizer,
    CharTokenizer,
)
from . import config


def save_model(model: GPTModel, tokenizer: Tokenizer) -> None:
    """Saves the model and tokenizer configuration.

    The weights of the model are saved in .safetensors format and configuration (either model and tokenizer) is saved in JSON format.

    Args:
        model (GPTModel): The model to save.
        tokenizer_config (TokenizerConfig): The tokenizer configuration to save.
    """
    model_name = f"gpoetry_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_dir = Path(config.MODELS_FOLDER) / model_name

    model_dir.mkdir(exist_ok=True, parents=True)
    save_file(model.state_dict(), model_dir / f"{model_name}.safetensors")

    with open(model_dir / "config.json", "w") as f:
        json.dump(asdict(model.config), f, indent=4)

    with open(model_dir / "tokenizer.json", "w") as f:
        json.dump(asdict(tokenizer.config), f, indent=4)


def decode_tokenizer_config(dct: dict) -> dict:
    """Decodes the tokenizer configuration from a dictionary.

    Args:
        dct (dict): The dictionary to decode.

    Returns:
        dict: The decoded dictionary.
    """
    if "itos" and "stoi" in dct:
        dct["itos"] = {int(k): v for k, v in dct["itos"].items()}
        dct["stoi"] = {k: int(v) for k, v in dct["stoi"].items()}

    return dct


def load_model() -> tuple[GPTModel, Tokenizer]:
    """Loads the model and tokenizer from the models folder.

    Raises:
        FileNotFoundError: If the models folder is not found.
        FileNotFoundError: If no models are found.

    Returns:
        tuple[GPTModel, Tokenizer]: The loaded model and tokenizer.
    """
    models_folder = Path(config.MODELS_FOLDER)

    if not models_folder.exists():
        raise FileNotFoundError(f"Models folder not found: {models_folder}")

    models_dirs = [d for d in models_folder.iterdir() if d.is_dir()]

    if len(models_dirs) == 0:
        raise FileNotFoundError("No models found")

    model_dir = max(models_dirs)

    print(f"Loading model from {model_dir}")

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
