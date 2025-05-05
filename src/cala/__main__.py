import os
from typing import Annotated

import typer
import uvicorn

app = typer.Typer()


@app.command()
def main(
    config_path: Annotated[
        str, typer.Option(help="Path to your Cala configuration yaml file.")
    ] = "cala_config.yaml",
) -> None:
    os.environ["CALA_CONFIG_PATH"] = config_path
    uvicorn.run("cala.dashboard:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    app()
