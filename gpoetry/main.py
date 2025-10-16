import typer
from pathlib import Path

from .core import config
from .commands.inference import inference_app
from .commands.train import train_app


app = typer.Typer(
    rich_markup_mode=None,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)

app.add_typer(inference_app)
app.add_typer(train_app)


@app.callback()
def main():
    Path(config.MODELS_FOLDER).mkdir(exist_ok=True, parents=True)


if __name__ == "__main__":
    app()
