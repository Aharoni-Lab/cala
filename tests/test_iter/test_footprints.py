import numpy as np
import pytest
import xarray as xr
from noob.node import Node, NodeSpecification
from scipy.ndimage import binary_dilation, generate_binary_structure, grey_erosion

from cala.assets import Footprints
from cala.models.axis import AXIS
from cala.testing.toy import FrameDims, Position, Toy


@pytest.fixture
def separate_cells() -> Toy:
    n_frames = 50

    return Toy(
        n_frames=n_frames,
        frame_dims=FrameDims(width=50, height=50),
        cell_radii=3,
        cell_positions=[
            Position(width=15, height=15),
            Position(width=15, height=35),
            Position(width=25, height=25),
            Position(width=35, height=35),
        ],
        cell_traces=[
            np.random.randint(low=0, high=n_frames, size=n_frames).astype(float),
            np.ones(n_frames, dtype=float),
            np.array(range(n_frames), dtype=float),
            np.array(range(n_frames - 1, -1, -1), dtype=float),
        ],
    )


@pytest.fixture
def fpter() -> Node:
    return Node.from_specification(
        NodeSpecification(
            id="test_footprinter",
            type="cala.nodes.footprints.Footprinter",
            params={"bep": None, "tol": 1e-7},
        )
    )


@pytest.mark.parametrize("toy", ["separate_cells"])
def test_ingest_frame(fpter, toy, request):
    toy = request.getfixturevalue(toy)

    pixstats = Node.from_specification(
        NodeSpecification(id="test_pixstats", type="cala.nodes.pixel_stats.initialize")
    ).process(traces=toy.traces, frames=toy.make_movie())
    compstats = Node.from_specification(
        NodeSpecification(id="test_compstats", type="cala.nodes.component_stats.initialize")
    ).process(traces=toy.traces)

    result = fpter.process(
        footprints=toy.footprints, pixel_stats=pixstats, component_stats=compstats
    )

    expected = toy.footprints.copy()

    assert result == expected


@pytest.fixture
def xpander() -> Node:
    return Node.from_specification(
        NodeSpecification(
            id="test_footprinter",
            type="cala.nodes.footprints.Footprinter",
            params={"bep": 2, "tol": 1e-7},
        )
    )


@pytest.mark.parametrize("toy", ["separate_cells"])
def test_expand_boundary(xpander, toy, request):
    """
    what would be the circumstances of needing boundary expansion:
    existing footprint is too small.
    does not affect component_stats
    pixel_stats may care. if the correlation with a component and a pixel is high,
    the pixel_stat would be high.
    boundary-expansion would be literally trying to add pixel_stats (normalized) around the
    boundary of the current footprint. (basically how many times pixel and trace coincided)
    the thing is, pixel_stat never goes below zero. so you're always sort of adding the boundary
    pixels.
    this means this phenomenon needs to be regulated by another mechanism, i.e. pixel
    value going to zero somehow.
    it would be pretty hard to ensure the cell boundary does not forever expand, since
    the longer the video, the more coincidences with any pixel and any trace will occur,
    so expansion is almost guaranteed every single loop.
    this means after the expansion, we need to rely on removal of "overexpanded" pixels.
    does that occur naturally at W - AM?
    """
    toy = request.getfixturevalue(toy)

    pixstats = Node.from_specification(
        NodeSpecification(id="test_pixstats", type="cala.nodes.pixel_stats.initialize")
    ).process(traces=toy.traces, frames=toy.make_movie())
    compstats = Node.from_specification(
        NodeSpecification(id="test_compstats", type="cala.nodes.component_stats.initialize")
    ).process(traces=toy.traces)

    footprint = generate_binary_structure(2, 1)

    eroded_fps = xr.apply_ufunc(
        grey_erosion,
        toy.footprints.array,
        kwargs={"footprint": footprint},
        vectorize=True,
        input_core_dims=[AXIS.spatial_dims],
        output_core_dims=[AXIS.spatial_dims],
    )

    result = xpander.process(
        footprints=Footprints.from_array(eroded_fps),
        pixel_stats=pixstats,
        component_stats=compstats,
    )

    # import matplotlib.pyplot as plt
    #
    # for idx, fps in enumerate(zip(eroded_fps, toy.footprints.array, result.array)):
    #     ero, exp, res = fps
    #     plt.imsave(f"eroded_{idx}.png", ero)
    #     plt.imsave(f"expect_{idx}.png", exp)
    #     plt.imsave(f"result_{idx}.png", res)

    assert result == toy.footprints
