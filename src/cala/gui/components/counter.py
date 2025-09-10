import anyio
from starlette.websockets import WebSocket

from cala.assets import Footprints
from cala.models import AXIS


def component_counter(footprints: Footprints, websocket: WebSocket) -> None:
    component_count_ = footprints.sizes[AXIS.component_dim]

    payload = {
        "type_": "component_count",
        "count": component_count_,
    }
    anyio.run(websocket.send_json, payload)
