from cala.assets import Footprints
from cala.gui.util import send_through
from cala.models import AXIS


class StreamCnt:
    def send(self, footprints: Footprints) -> None:
        component_count_ = footprints.sizes[AXIS.component_dim]

        payload = {
            "type_": "component_count",
            "count": component_count_,
        }

        send_through(payload)
