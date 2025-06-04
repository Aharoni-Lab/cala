import os
from pathlib import Path
from typing import Annotated

import typer
import uvicorn
from dotenv import load_dotenv

load_dotenv()
app = typer.Typer()


@app.command()
def main(
    config_path: Annotated[Path, typer.Option(help="Path to your Cala configuration yaml file.")],
) -> None:
    os.environ["CALA_CONFIG_PATH"] = str(config_path)
    uvicorn.run("cala.gui.__main__:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    app()
