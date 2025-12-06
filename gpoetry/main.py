from pathlib import Path

import typer

from .commands.gen import gen_app
from .commands.train import train_app
from .core import config

app = typer.Typer(
    rich_markup_mode=None,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)

app.add_typer(gen_app)
app.add_typer(train_app)


@app.callback()
def main():
    Path(config.MODELS_FOLDER).mkdir(exist_ok=True, parents=True)


if __name__ == "__main__":
    app()
