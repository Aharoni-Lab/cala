from pathlib import Path
from typing import Annotated

import typer
import uvicorn
from noob import SynchronousRunner, Tube

from cala.gui.app import get_app
from cala.logging import init_logger

logger = init_logger(__name__)

cli = typer.Typer()

try:
    app = get_app()
except TypeError as e:
    logger.warning(f"Failed to load gui app: {e}")


@cli.command()
def main(spec: str, gui: Annotated[bool, typer.Option()] = False) -> None:
    if gui:
        uvicorn.run("cala.main:app", reload=True, reload_dirs=[Path(__file__).parent])
    else:
        tube = Tube.from_specification(spec)
        runner = SynchronousRunner(tube=tube)
        runner.run()
