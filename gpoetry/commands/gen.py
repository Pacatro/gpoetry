from typing import Annotated

import typer

from ..core import config
from ..core.generation import generate
from ..core.model_io import load_model

gen_app = typer.Typer()


@gen_app.command(
    name="gen", help="Generate a new poem using the most recent trained model."
)
def gen_cli(
    init_text: Annotated[
        str, typer.Option("--init-text", "-i", help="The initial text")
    ] = config.INIT_TOKEN,
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
    """Generate a new poem using the most recent trained model.

    Args:
        init_text (Annotated[ str, typer.Option, optional): The initial text. Defaults to config.INIT_TOKEN.
        temperature (Annotated[ float, typer.Option, optional): The temperature for generation. Defaults to config.TEMPERATURE.
        top_k (Annotated[ int, typer.Option, optional): The top-k sampling value. Defaults to config.TOP_K.
        gen_limit (Annotated[ int, typer.Option, optional): The generation limit. Defaults to config.GEN_LIMIT.
    """
    model, tokenizer = load_model()

    generate(
        model=model,
        init_text=init_text,
        tokenizer=tokenizer,
        top_k=top_k,
        temperature=temperature,
        device=config.DEVICE,
        block_size=config.BLOCK_SIZE,
        gen_limit=gen_limit,
    )
