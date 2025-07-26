import numpy as np

from cala.models import AXIS
from cala.nodes.prep.glow_removal import GlowRemover
from cala.testing.toy import FrameDims, Position, Toy


def test_glow_removal():
    yeah_glo = GlowRemover()

    toy = Toy(
        n_frames=10,
        frame_dims=FrameDims(width=11, height=11),
        cell_radii=1,
        cell_positions=[Position(width=5, height=5)],
        cell_traces=[np.array([5, 4, 3, 2, 1, 1, 2, 3, 4, 5])],
    )

    gen = toy.movie_gen()
    movie = toy.make_movie()

    expected_base = movie.array.min(dim=AXIS.frames_dim)

    res = []
    for frame, br in zip(iter(gen), [5, 4, 3, 2, 1, 1, 1, 1, 1, 1]):
        res.append(yeah_glo.process(frame))
        assert yeah_glo.base_brightness_.max() == br

    assert np.array_equal(expected_base, yeah_glo.base_brightness_)
