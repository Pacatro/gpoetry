import torch

from .tokenization import TokenizerType

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset
DATASET_URL: str = "andreamorgar/spanish_poetry"
MAX_SAMPLES: int | None = None
TOKENIZER_TYPE: TokenizerType = TokenizerType.CHAR

# Model config
NUM_HEADS: int = 8
BLOCK_SIZE: int = 1024
NUM_LAYERS: int = 4
EMB_DIM: int = 256
DROPOUT_P: float = 0.2

# Training config
EPOCHS: int = 5000
BATCH_SIZE: int = 32
LR: float = 3e-4
TRAIN_SIZE: float = 0.9

# Generation config
TEMPERATURE: float = 0.4
TOP_K: int = 50
START_STRING: str = "<start>"
