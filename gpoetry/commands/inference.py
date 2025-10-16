import typer
from typing import Annotated

from ..core import config
from ..core.generation import generate
from ..core.model_io import load_model

inference_app = typer.Typer()


@inference_app.command(name="inference", help="Run the inference")
def inference_cli(
    temperature: Annotated[
        float, typer.Option("--temperature", "-t", help="The temperature")
    ] = config.TEMPERATURE,
    top_k: Annotated[
        int, typer.Option("--top-k", "-k", help="The top k")
    ] = config.TOP_K,
    gen_limit: Annotated[
        int, typer.Option("--gen-limit", "-l", help="The generation limit")
    ] = config.GEN_LIMIT,
) -> None:
    """Runs the inference process.

    Args:
        temperature (Annotated[ float, typer.Option, optional): The temperature for generation. Defaults to config.TEMPERATURE.
        top_k (Annotated[ int, typer.Option, optional): The top-k sampling value. Defaults to config.TOP_K.
        gen_limit (Annotated[ int, typer.Option, optional): The generation limit. Defaults to config.GEN_LIMIT.
    """
    if config.MAX_SAMPLES:
        print(f"Max samples: {config.MAX_SAMPLES}")

    model, tokenizer = load_model()

    generate(
        model=model,
        tokenizer=tokenizer,
        top_k=top_k,
        temperature=temperature,
        device=config.DEVICE,
        block_size=config.BLOCK_SIZE,
        gen_limit=gen_limit,
    )
