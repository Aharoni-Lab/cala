from typing import Annotated

import yaml
from fastapi import Depends
from noob.tube import TubeSpecification

from cala.gui.const import SPEC_FILE


async def get_spec() -> TubeSpecification:
    with open(SPEC_FILE) as f:
        data = yaml.safe_load(f)
    spec = TubeSpecification(**data)
    return spec


Spec = Annotated[TubeSpecification, Depends(get_spec)]
