from pathlib import Path
from typing import Annotated

from fastapi import Depends

from cala.models.gui import GUISpec


def get_frontend_dir() -> Path: ...


async def get_spec() -> GUISpec: ...


Spec = Annotated[GUISpec, Depends(get_spec)]
