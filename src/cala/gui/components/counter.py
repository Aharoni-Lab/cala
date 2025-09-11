from cala.assets import Footprints
from cala.gui.util import QManager
from cala.models import AXIS


def component_counter(queue_id: str, footprints: Footprints, q_manager: QManager) -> None:
    if footprints.array is None:
        return None

    payload = {"count": footprints.array.sizes[AXIS.component_dim]}
    q_manager.get_queue(queue_id).put(payload)
    return None
