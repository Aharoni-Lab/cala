import pytest
import xarray as xr
from noob.node import Node, NodeSpecification

from cala.assets import AXIS, Footprints, Frame, PixStats, PopSnap, Traces


@pytest.fixture(scope="function")
def init() -> Node:
    return Node.from_specification(
        spec=NodeSpecification(id="ps_init_test", type="cala.nodes.omf.pixel_stats.initialize")
    )


def test_init(init, four_separate_cells) -> None:
    traces = four_separate_cells.traces.array
    movie = four_separate_cells.make_movie().array
    footprints = four_separate_cells.footprints.array

    results = init.process(
        traces=traces,
        frames=movie,
        footprints=four_separate_cells.footprints.array,
    )

    for result, trace, shape in zip(
        results.transpose(AXIS.component_dim, ...),
        traces.transpose(AXIS.component_dim, ...),
        footprints.transpose(AXIS.component_dim, ...),
    ):
        # it's going to be only identical where the footprint exists
        # and otherwise zero
        mask = shape > 0
        expected = xr.where(mask, (trace @ movie) / four_separate_cells.n_frames, 0)
        xr.testing.assert_allclose(result.as_numpy(), expected.as_numpy())


@pytest.fixture(scope="function")
def frame_update() -> Node:
    return Node.from_specification(
        spec=NodeSpecification(id="ps_frame_test", type="cala.nodes.omf.pixel_stats.ingest_frame")
    )


def test_ingest_frame(init, frame_update, four_separate_cells) -> None:
    traces = four_separate_cells.traces.array
    movie = four_separate_cells.make_movie().array
    footprints = four_separate_cells.footprints.array

    pre_ingest = init.process(
        traces=traces.isel({AXIS.frame_dim: slice(None, -1)}),
        frames=movie.isel({AXIS.frame_dim: slice(None, -1)}),
        footprints=footprints,
    )

    result = frame_update.process(
        pixel_stats=PixStats.from_array(pre_ingest),
        frame=Frame.from_array(movie.isel({AXIS.frame_dim: -1})),
        new_traces=PopSnap.from_array(traces.isel({AXIS.frame_dim: -1})),
        footprints=Footprints.from_array(footprints),
    ).array

    expected = init.process(traces=traces, frames=movie, footprints=footprints)
    xr.testing.assert_allclose(expected, result.as_numpy())


@pytest.fixture(scope="function")
def comp_update() -> Node:
    return Node.from_specification(
        spec=NodeSpecification(
            id="ps_comp_test", type="cala.nodes.omf.pixel_stats.ingest_component"
        )
    )


def test_ingest_component(init, comp_update, four_separate_cells):
    slice_idx = -2
    traces = four_separate_cells.traces.array
    footprints = four_separate_cells.footprints.array
    movie = four_separate_cells.make_movie()

    pre_ingest = init.process(
        traces=traces.isel({AXIS.component_dim: slice(None, slice_idx)}),
        frames=movie.array,
        footprints=footprints.isel({AXIS.component_dim: slice(None, slice_idx)}),
    )

    result = comp_update.process(
        pixel_stats=PixStats.from_array(pre_ingest),
        frames=movie,
        new_traces=Traces.from_array(traces.isel({AXIS.component_dim: slice(slice_idx, None)})),
        new_footprints=Footprints.from_array(
            footprints.isel({AXIS.component_dim: slice(slice_idx, None)})
        ),
    ).array

    expected = init.process(traces=traces, frames=movie.array, footprints=footprints)

    xr.testing.assert_allclose(expected.as_numpy(), result.as_numpy())
