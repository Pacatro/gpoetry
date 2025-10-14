# GPoeTry

A tiny GPT model to generate Spanish poetry, built from scratch.

## Getting Started

> [!NOTE]
> To run this project, you need to have the [`uv`](https://docs.astral.sh/uv/) package manager installed.

Follow these steps to run the project:

1. **Clone the repository**

    ```bash
    git clone https://github.com/Pacatro/gpoetry.git
    cd gpoetry
    ```

2. **Install dependencies and create a virtual environment**

    ```bash
    uv sync
    ```

4. Run tests

    ```bash
    uv run pytest
    ```

3. **Run the application**

    To see all available commands and options, run:

    ```bash
    uv run gpoetry
    ```

## Usage

The main entry point for the application is `gpoetry/main.py`. Running `uv run gpoetry` will execute the following workflow:

1. **Configuration**: Loads settings from `gpoetry/config.py`.
2. **Tokenization**: Fits a tokenizer (`CharTokenizer` or `WordTokenizer`) on the loaded text.
3. **Training**: If a trained model file (specified by `MODEL_PATH`) is not found, it will train a new model using the parameters in the configuration file and save it.
4. **Generation**: Generates new text using the trained model.

## Configuration

The entire project can be configured by modifying the values in `gpoetry/config.py`. Here are some of the key options:

| Parameter | Description | Default |
| --- | --- | --- |
| `DEVICE` | The device to run the model on. | `"cuda"` or `"cpu"` |
| `TOKENIZER_TYPE` | The tokenization strategy to use (`CHAR` or `WORD`). | `TokenizerType.CHAR` |
| `MODEL_PATH` | Path to save and load the trained model. | `"gpoetry.pt"` |
| `NUM_HEADS` | Number of attention heads in the Transformer blocks. | `4` |
| `NUM_LAYERS` | Number of Transformer blocks. | `4` |
| `EMB_DIM` | Embedding dimension for tokens and positions. | `512` |
| `EPOCHS` | Number of training epochs. | `10` |
| `BATCH_SIZE` | Batch size for training. | `32` |
| `LR` | Learning rate for the optimizer. | `3e-4` |
| `TEMPERATURE` | Controls the randomness of the generated text. | `0.4` |
| `TOP_K` | Samples from the top K most likely next tokens. | `50` |

## Model Architecture

GPoeTry uses a standard GPT (Generative Pre-trained Transformer) architecture, implemented in `gpoetry/model.py`. It consists of:

- **Token and Positional Embeddings**: To represent the input tokens and their positions in the sequence.
- **Transformer Blocks**: A stack of `NUM_LAYERS` blocks. Each block contains:
  - A multi-head self-attention mechanism (`MHSelfAttention`).
  - A feed-forward neural network (`MLP`).
  - Layer normalization and residual connections.
- **Language Model Head**: A final linear layer that maps the Transformer's output to vocabulary-sized logits.

## Dataset

This project use the [`andreamorgar/spanish_poetry`](https://huggingface.co/datasets/andreamorgar/spanish_poetry) dataset from HuggingFace.

