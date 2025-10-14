import torch

from .tokenization import TokenizerType
from .data import DatasetType

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset
DATASET_URL: str = "andreamorgar/spanish_poetry"
DATASET_TEXT_COLUMN: str = "content"
MAX_SAMPLES: int | None = 100
TOKENIZER_TYPE: TokenizerType = TokenizerType.CHAR
DATASET_TYPE: DatasetType = DatasetType.TXT
INIT_TOKEN: str = "<|start|>"
END_TOKEN: str = "<|end|>"

# Model config
NUM_HEADS: int = 4
BLOCK_SIZE: int = 256
NUM_LAYERS: int = 4
EMB_DIM: int = 256
DROPOUT_P: float = 0.2

# Training config
EPOCHS: int = 5
BATCH_SIZE: int = 32
LR: float = 3e-4
TRAIN_SIZE: float = 0.8

# Generation config
TEMPERATURE: float = 0.4
TOP_K: int = 50
