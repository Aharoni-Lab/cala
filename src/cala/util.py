from collections.abc import Sequence
from pathlib import Path
from shutil import rmtree
from uuid import uuid4


def create_id() -> str:
    return uuid4().hex


def combine_attr_replaces(attrs: Sequence[dict[str, list[str]]], context: None = None) -> dict:
    repl = [item for attr in attrs for item in attr.get("replaces", [])]
    return {"replaces": repl} if repl else {}


def clear_dir(directory: Path | str) -> None:
    for path in Path(directory).glob("**/*"):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            rmtree(path)
