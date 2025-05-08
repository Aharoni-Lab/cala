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
    config_path: Annotated[
        Path, typer.Option(help="Path to your Cala configuration yaml file.")
    ] = "cala_config.yaml",
    dashboard: Annotated[bool, typer.Option(help="Launch the dashboard.")] = False,
) -> None:
    if dashboard:
        os.environ["CALA_CONFIG_PATH"] = str(config_path)
        uvicorn.run("cala.gui.__main__:app", host="127.0.0.1", port=8000, reload=True)
    else:
        import asyncio

        from cala.config import Config
        from cala.main import run_pipeline

        config = Config.from_yaml(config_path)

        # loop = asyncio.get_event_loop()
        # loop.run_until_complete(run_pipeline(config))
        asyncio.run(run_pipeline(config))


if __name__ == "__main__":
    app()
