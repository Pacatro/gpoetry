import torch

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset
DATASET_URL: str = "andreamorgar/spanish_poetry"

# Model config
NUM_HEADS: int = 4
BLOCK_SIZE: int = 1024
NUM_LAYERS: int = 4
EMB_DIM: int = 512
DROPOUT_P: float = 0.2

# Training config
EPOCHS: int = 10
BATCH_SIZE: int = 64
LR: float = 1e-3
