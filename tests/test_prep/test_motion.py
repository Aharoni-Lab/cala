import numpy as np
import pytest

from cala.models import AXIS
from cala.nodes.prep.motion import Shift, Stabilizer
from cala.testing.toy import FrameDims, Position, Toy


@pytest.mark.parametrize(
    "params",
    [
        {
            "drift_speed": 1,
            "pcc_kwargs": {"upsample_factor": 100},
            "pcc_filter": "difference_of_gaussians",
            "filter_kwargs": {"low_sigma": 1},
        }
    ],
)
def test_motion_estimation(params) -> None:

    stab = Stabilizer(**params)

    n_frames = 50

    toy = Toy(
        n_frames=n_frames,
        frame_dims=FrameDims(width=50, height=50),
        cell_radii=3,
        cell_positions=[Position(width=15, height=15), Position(width=35, height=35)],
        cell_traces=[np.array(range(n_frames)), np.array(range(n_frames, 0, -1))],
        emit_frames=True,
    )

    shifts = [
        Shift(width=x, height=y)
        for x, y in zip(np.random.randint(-7, 7, n_frames), np.random.randint(-7, 7, n_frames))
    ]

    gen = toy.movie_gen()

    result = []

    for shift, frame in zip(shifts, iter(gen)):
        frame.array = frame.array.roll(
            shifts={AXIS.width_dim: int(shift.width), AXIS.height_dim: int(shift.height)}
        )
        frame.array.attrs = {
            "shift": Shift(
                width=shifts[0].width - shift.width, height=shifts[0].height - shift.height
            )
        }
        result.append(stab.stabilize(frame))

    estimate = -np.array([(m.width, m.height) for m in stab.motions_])
    expected = np.array([(m.width, m.height) for m in shifts]) - (shifts[0].width, shifts[0].height)

    # Allow 1 pixel absolute tolerance
    np.testing.assert_allclose(estimate, expected, atol=1.0)


def test_rigid_translator_preserves_neuron_traces(): ...
