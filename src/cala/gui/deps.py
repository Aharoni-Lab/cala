from typing import Annotated

import yaml
from fastapi import Depends

from cala.gui.const import SPEC_FILE
from cala.models.gui import GUISpec


async def get_spec() -> GUISpec:
    with open(SPEC_FILE) as f:
        data = yaml.safe_load(f)
    spec = GUISpec(**data)
    return spec


Spec = Annotated[GUISpec, Depends(get_spec)]
