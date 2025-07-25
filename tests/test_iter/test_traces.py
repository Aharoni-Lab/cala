import numpy as np
from noob.node import NodeSpecification

from cala.nodes.iter.traces import Tracer
from cala.testing.toy import FrameDims, Position, Toy


def test_init() -> None:

    n_frames = 50

    toy = Toy(
        n_frames=n_frames,
        frame_dims=FrameDims(width=50, height=50),
        cell_radii=3,
        cell_positions=[Position(width=15, height=15), Position(width=35, height=35)],
        cell_traces=[
            np.array(range(n_frames), dtype=float),
            np.array(range(n_frames, 0, -1), dtype=float),
        ],
    )

    tracer = Tracer.from_specification(
        spec=NodeSpecification(
            id="test", type="cala.nodes.iter.traces.Tracer", params={"tolerance": 1e-3}
        )
    )

    traces = tracer.initialize(footprints=toy.footprints, movie=toy.make_movie())

    np.testing.assert_array_equal(traces.array, toy.traces.array)


def test_ingest_frame() -> None: ...
