import torch

from .tokenization import TokenizerType
from .data import DatasetType

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset
DATASET_URL: str = "andreamorgar/spanish_poetry"
DATASET_TEXT_COLUMN: str = "content"
MAX_SAMPLES: int | None = None
TOKENIZER_TYPE: TokenizerType = TokenizerType.WORD
DATASET_TYPE: DatasetType = DatasetType.HUGGINGFACE

# Model config
NUM_HEADS: int = 4
BLOCK_SIZE: int = 256
NUM_LAYERS: int = 2
EMB_DIM: int = 128
DROPOUT_P: float = 0.2

# Training config
EPOCHS: int = 100
BATCH_SIZE: int = 16
LR: float = 3e-4
TRAIN_SIZE: float = 0.8

# Generation config
TEMPERATURE: float = 0.4
TOP_K: int = 50
START_STRING: str = "<start>"
