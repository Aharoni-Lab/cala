from cala.assets import Traces
from cala.models import AXIS


def component_counter(index: int, traces: Traces) -> dict[str, int]:
    if traces.array is None:
        return {"index": index, "count": 0}

    payload = {"index": index, "count": traces.array.sizes[AXIS.component_dim]}
    return payload
