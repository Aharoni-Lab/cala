import numpy as np
import pytest

from cala.models import AXIS
from cala.nodes.prep import blur
from cala.nodes.prep.motion import Anchor, Shift
from cala.testing.toy import FrameDims, Position, Toy


@pytest.mark.parametrize(
    "params",
    [
        {
            "upsample_factor": 10,
            "dog_kwargs": {"low_sigma": 3},
            "gauss_kwargs": {"ksize": (11, 11), "sigmaX": 20},
        }
    ],
)
def test_motion_estimation(params) -> None:

    stab = Anchor(**params)

    n_frames = 50

    toy = Toy(
        n_frames=n_frames,
        frame_dims=FrameDims(width=100, height=100),
        cell_radii=6,
        cell_positions=[
            Position(width=30, height=30),
            Position(width=50, height=60),
            Position(width=70, height=70),
        ],
        cell_traces=[
            np.array(range(n_frames)),
            np.array(range(n_frames, 0, -1)),
            np.array([25] * n_frames),
        ],
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
        frame = blur(frame, method="gaussian", kwargs={"ksize": (5, 5), "sigmaX": 2})
        result.append(stab.stabilize(frame))

    estimate = -np.array([(m.width, m.height) for m in stab._history])
    expected = np.array([(m.width, m.height) for m in shifts]) - (shifts[0].width, shifts[0].height)

    # Allow 1 pixel absolute tolerance
    np.testing.assert_allclose(estimate, expected[1:], atol=1.0)


def test_rigid_translator_preserves_neuron_traces(): ...
