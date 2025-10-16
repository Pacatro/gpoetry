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

3. **Run tests**

    ```bash
    uv run pytest
    ```

4. **Run the application**

    To see all available commands and options, run:

    ```bash
    uv run gpoetry --help
    ```

## Usage

The application is structured as a CLI with two main commands: `train` and `inference`.

### Training the model

To train a new model, use the `train` command. This will train a new model using the parameters in the configuration file and save it to the `models` directory.

```bash
uv run gpoetry train [OPTIONS]
```

**Options:**

| Option | Description | Default |
| --- | --- | --- |
| `-t`, `--tokenization` | The tokenizer type (`word` or `char`). | `char` |
| `-b`, `--batch-size` | The training batch size. | `32` |
| `-e`, `--epochs` | The number of epochs. | `5` |
| `-l`, `--lr` | The learning rate. | `3e-4` |
| `-s`, `--train-size` | The training split size. | `0.8` |

Example of training with word tokenization:

```bash
uv run gpoetry train --tokenization word
```

### Generating poetry

To generate poetry with a trained model, use the `inference` command. This command will load the latest model from the `models` directory and generate text.

```bash
uv run gpoetry inference [OPTIONS]
```

**Options:**

| Option | Description | Default |
| --- | --- | --- |
| `-t`, `--temperature` | Controls the randomness of the generated text. | `0.6`|
| `-k`, `--top-k` | Samples from the top K most likely next tokens. | `50` |
| `-l`, `--gen-limit` | The generation limit in tokens. | `1000` |

Example of generating text with a higher temperature:

```bash
uv run gpoetry inference --temperature 0.8
```

## Model Architecture

GPoeTry uses a standard GPT (Generative Pre-trained Transformer) architecture, implemented in `gpoetry/core/model.py`. It consists of:

- **Token and Positional Embeddings**: To represent the input tokens and their positions in the sequence.
- **Transformer Blocks**: A stack of `NUM_LAYERS` blocks. Each block contains:
  - A multi-head self-attention mechanism (`MHSelfAttention`).
  - A feed-forward neural network (`MLP`).
  - Layer normalization and residual connections.
- **Language Model Head**: A final linear layer that maps the Transformer's output to vocabulary-sized logits.

## Dataset

This project uses the [`andreamorgar/spanish_poetry`](https://huggingface.co/datasets/andreamorgar/spanish_poetry) dataset from HuggingFace.

