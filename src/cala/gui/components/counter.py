from cala.assets import AXIS
from cala.assets.assets import Traces


def component_counter(index: int, traces: Traces) -> dict[str, int]:
    if traces.array is None:
        return {"index": index, "count": 0}

    payload = {"index": index, "count": traces.array.sizes[AXIS.component_dim]}
    return payload
