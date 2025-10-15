import torch

from .tokenization import TokenizerType

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset
DATASET_URL: str = "andreamorgar/spanish_poetry"
DATASET_TEXT_COLUMN: str = "content"
MAX_SAMPLES: int | None = 200
TOKENIZER_TYPE: TokenizerType = TokenizerType.CHAR
INIT_TOKEN: str = "<|start|>"
END_TOKEN: str = "<|end|>"

# Model config
NUM_HEADS: int = 4
BLOCK_SIZE: int = 512
NUM_LAYERS: int = 4
EMB_DIM: int = 512
DROPOUT_P: float = 0.2
MODEL_PATH: str = "gpoetry.pt"
MODEL_CONFIG_PATH: str = "gpoetry.json"

# Training config
EPOCHS: int = 5
BATCH_SIZE: int = 32
LR: float = 3e-4
TRAIN_SIZE: float = 0.8

# Generation config
TEMPERATURE: float = 0.6
TOP_K: int = 50
