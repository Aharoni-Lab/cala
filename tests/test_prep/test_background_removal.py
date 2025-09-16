from typing import Any

import numpy as np
import pytest

from cala.nodes.prep import remove_background
from cala.testing.toy import FrameDims, Position, Toy


@pytest.mark.parametrize(
    "params, sum",
    [({"method": "uniform", "kernel_size": 2}, 16), ({"method": "tophat", "kernel_size": 4}, 29)],
)
def test_background_removal(params: dict[str, Any], sum) -> None:
    """Test consistency of streaming background removal"""
    toy = Toy(
        n_frames=10,
        frame_dims=FrameDims(width=11, height=11),
        cell_radii=3,
        cell_positions=[Position(width=5, height=5)],
        cell_traces=[np.ones(10)],
        emit_frames=True,
    )

    gen = toy.movie_gen()

    frame = next(gen)

    result = remove_background(frame, **params)

    assert result.array.sum() == sum
