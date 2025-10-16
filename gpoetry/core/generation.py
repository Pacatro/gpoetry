import torch
import torch.nn.functional as F
import time

from .model import GPTModel
from .tokenization import Tokenizer
from . import config


def generate(
    model: GPTModel,
    tokenizer: Tokenizer,
    temperature: float,
    top_k: int,
    device: str,
    block_size: int,
    gen_limit: int = 1000,
) -> None:
    """Generates text using the model.

    Args:
        model (GPTModel): The model to use for generation.
        tokenizer (Tokenizer): The tokenizer to use for encoding and decoding.
        temperature (float): The temperature for sampling.
        top_k (int): The top-k sampling value.
        device (str): The device to use for generation.
        block_size (int): The block size for the model.
        gen_limit (int, optional): The generation limit. Defaults to 1000.
    """
    # Load the model
    model.eval()

    # We start with an initial context with only the first token
    start_token = tokenizer.encode(config.INIT_TOKEN)[0]
    context = torch.tensor([[start_token]], dtype=torch.long, device=device)
    initial_str = tokenizer.decode(context[0].tolist())

    print(initial_str, end="", flush=True)

    generated_text = ""
    with torch.no_grad():
        for _ in range(gen_limit):
            if config.END_TOKEN in generated_text:
                break

            # Select the last "block_size" tokens
            x = context[:, -block_size:]
            logits = model(x)
            logits = logits[:, -1, :] / temperature

            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                # Remove tokens with a probability less than the last token of the top-k
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            # Select and index in base of the probabilities
            next_token = torch.multinomial(probs, num_samples=1)
            text = tokenizer.decode(next_token[0].tolist())

            print(text, end="", flush=True)

            generated_text += text

            # Add the new generated token to the context
            context = torch.cat((context, next_token), dim=1)
            time.sleep(0.02)
