from typing import Annotated

import typer
import uvicorn
from noob import SynchronousRunner, Tube

from cala.gui.app import get_app

cli = typer.Typer()
app = get_app()


@cli.command()
def main(spec: str, gui: Annotated[bool, typer.Option()] = False) -> None:
    if gui:
        uvicorn.run("cala.main:app", reload=True)
    else:
        tube = Tube.from_specification(spec)
        runner = SynchronousRunner(tube=tube)
        runner.run()
