import typer
import uvicorn

from cala.gui.app import get_app

cli = typer.Typer()
app = get_app()


@cli.command()
def main(spec: str, gui: bool) -> None:  # put options as main params (look at Typer docs)
    if gui:
        uvicorn.run("cala.main:app", reload=True)
