import torch
import torch.nn.functional as F
import sys
import time

from .model import GPTModel, GPTConfig
from .tokenization import Tokenizer
from . import config


def generate(
    model_path: str,
    gpt_config: GPTConfig,
    tokenizer: Tokenizer,
    temperature: float,
    top_k: int,
    device: str,
    block_size: int,
    gen_limit: int = 1000,
) -> None:
    # Load the model
    model = GPTModel(gpt_config)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # We start with an initial context with only the first token
    start_token = tokenizer.encode(config.INIT_TOKEN)[0]
    context = torch.tensor([[start_token]], dtype=torch.long, device=device)
    initial_str = tokenizer.decode(context[0].tolist())

    print(initial_str, end="")
    sys.stdout.flush()

    with torch.no_grad():
        for _ in range(gen_limit):
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
            idx_next = torch.multinomial(probs, num_samples=1)
            char = tokenizer.decode(idx_next[0].tolist())

            print(char, end="")
            sys.stdout.flush()
            # Add the new generated token to the context
            context = torch.cat((context, idx_next), dim=1)
            time.sleep(0.02)
