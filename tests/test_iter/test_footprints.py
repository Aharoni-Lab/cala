import numpy as np
import pytest
import xarray as xr
from noob.node import Node, NodeSpecification
from scipy.ndimage import grey_dilation, grey_erosion
from skimage.morphology import disk

from cala.assets import Footprints
from cala.models.axis import AXIS
from cala.testing.toy import FrameDims, Position, Toy


@pytest.fixture
def separate_cells() -> Toy:
    n_frames = 50

    return Toy(
        n_frames=n_frames,
        frame_dims=FrameDims(width=50, height=50),
        cell_radii=1,
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
def connected_cells() -> Toy:
    n_frames = 50

    return Toy(
        n_frames=n_frames,
        frame_dims=FrameDims(width=50, height=50),
        cell_radii=8,
        cell_positions=[
            Position(width=15, height=15),
            Position(width=15, height=35),
            Position(width=25, height=25),
            Position(width=35, height=35),
        ],
        cell_traces=[
            np.random.randint(low=0, high=n_frames, size=n_frames).astype(float),
            np.abs(np.sin(np.linspace(-np.pi, np.pi, n_frames)) * n_frames).astype(float),
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
            params={"bep": 0, "tol": 1e-7},
        )
    )


@pytest.mark.parametrize("toy", ["separate_cells", "connected_cells"])
def test_ingest_frame(fpter, toy, request):
    toy = request.getfixturevalue(toy)

    pixstats = Node.from_specification(
        NodeSpecification(id="test_pixstats", type="cala.nodes.pixel_stats.initialize")
    ).process(traces=toy.traces, frames=toy.make_movie())
    compstats = Node.from_specification(
        NodeSpecification(id="test_compstats", type="cala.nodes.component_stats.initialize")
    ).process(traces=toy.traces)

    result = fpter.process(
        footprints=toy.footprints, pixel_stats=pixstats, component_stats=compstats, index=0
    ).array.as_numpy()

    expected = toy.footprints.array.as_numpy()

    xr.testing.assert_allclose(result, expected)


@pytest.fixture
def xpander() -> Node:
    return Node.from_specification(
        NodeSpecification(
            id="test_footprinter",
            type="cala.nodes.footprints.Footprinter",
            params={"bep": 2, "tol": 1e-7},
        )
    )


@pytest.mark.parametrize("defect", [grey_erosion, grey_dilation])
@pytest.mark.parametrize("toy", ["separate_cells", "connected_cells"])
def test_boundary_morph(xpander, defect, toy, request):
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
    Not exactly.

    W: width height comp, A: width height comp M: comp comp
    M: avg dot product of traces
    AM: footprint x how correlated other cells are
    """
    toy = request.getfixturevalue(toy)

    pixstats = Node.from_specification(
        NodeSpecification(id="test_pixstats", type="cala.nodes.pixel_stats.initialize")
    ).process(traces=toy.traces, frames=toy.make_movie())
    compstats = Node.from_specification(
        NodeSpecification(id="test_compstats", type="cala.nodes.component_stats.initialize")
    ).process(traces=toy.traces)

    footprint = disk(radius=1)

    modded_fps = xr.apply_ufunc(
        defect,
        toy.footprints.array.as_numpy(),
        kwargs={"footprint": footprint},
        vectorize=True,
        input_core_dims=[AXIS.spatial_dims],
        output_core_dims=[AXIS.spatial_dims],
    )

    result = xpander.process(
        footprints=Footprints.from_array(modded_fps),
        pixel_stats=pixstats,
        component_stats=compstats,
        index=0,
    )

    # expansion breaks when a trace is all-zero and overlaps with another component.
    # not sure when an all-zero trace would occur (esp with noise), so probably ok.
    xr.testing.assert_allclose(result.array.as_numpy(), toy.footprints.array.as_numpy(), atol=1e-3)
