import typer

app = typer.Typer()


@app.command()
def main() -> None:  # put options as main params (look at Typer docs)
    pass


if __name__ == "__main__":
    app()
