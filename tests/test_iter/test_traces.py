import numpy as np
import pytest
import xarray as xr
from noob.node import Node, NodeSpecification

from cala.assets import Frame, Overlaps, Traces
from cala.models import AXIS


@pytest.fixture
def frame_update() -> Node:
    return Node.from_specification(
        spec=NodeSpecification(
            id="frame_test",
            type="cala.nodes.omf.traces.Tracer",
            params={"max_iter": 100, "tol": 1e-4},
        )
    )


@pytest.mark.parametrize(
    "zarr_setup",
    [
        {"zarr_path": None, "peek_size": 50, "flush_interval": None},
        {"zarr_path": "tmp", "peek_size": 30, "flush_interval": 40},
    ],
)
@pytest.mark.parametrize("toy", ["four_separate_cells", "four_connected_cells"])
def test_ingest_frame(frame_update, toy, zarr_setup, request, tmp_path) -> None:
    """
    Frame ingestion step adds new traces to the Traces instance
    that matches true brightness, overlapping and non-overlapping alike.

    """
    toy = request.getfixturevalue(toy)
    zarr_setup["zarr_path"] = tmp_path if zarr_setup["zarr_path"] else None

    xray = Node.from_specification(
        spec=NodeSpecification(id="test", type="cala.nodes.omf.overlap.initialize")
    )

    traces = Traces(array_=None, **zarr_setup)
    traces.array = toy.traces.array.isel({AXIS.frames_dim: slice(None, -1)})

    frame = Frame.from_array(toy.make_movie().array.isel({AXIS.frames_dim: -1}))
    overlap = xray.process(overlaps=Overlaps(), footprints=toy.footprints)

    result = frame_update.process(
        traces=traces, footprints=toy.footprints, frame=frame, overlaps=overlap
    ).array
    expected = toy.traces.array.isel({AXIS.frames_dim: -1})

    xr.testing.assert_allclose(result, expected, atol=1e-3)


@pytest.fixture
def comp_update() -> Node:
    return Node.from_specification(
        NodeSpecification(id="comp_test", type="cala.nodes.omf.traces.ingest_component")
    )


@pytest.mark.parametrize(
    "zarr_setup",
    [
        {"zarr_path": None, "peek_size": 40, "flush_interval": None},
        {"zarr_path": "tmp", "peek_size": 30, "flush_interval": 40},
    ],
)
@pytest.mark.parametrize("toy", ["four_separate_cells"])
def test_ingest_component(comp_update, toy, request, zarr_setup, tmp_path) -> None:
    """
    *New component is always the same length as peek_size.

    I can add components that...
        1. is single and new
        2. is single and replacing
        3. is multiple
        4. is multiple and replacing

    """
    toy = request.getfixturevalue(toy)
    zarr_setup["zarr_path"] = tmp_path if zarr_setup["zarr_path"] else None

    traces = Traces(array_=None, **zarr_setup)
    traces.array = toy.traces.array_.isel({AXIS.component_dim: slice(None, -1)})

    new_traces = toy.traces.array_.isel(
        {AXIS.component_dim: [-1], AXIS.frames_dim: slice(-zarr_setup["peek_size"], None)}
    )

    new_traces.attrs["replaces"] = ["cell_0"]
    result = comp_update.process(traces, Traces.from_array(new_traces))

    expected = toy.traces.array.drop_sel({AXIS.component_dim: 0})
    expected.loc[
        {AXIS.component_dim: -1, AXIS.frames_dim: slice(None, -zarr_setup["peek_size"])}
    ] = np.nan

    assert result.full_array().equals(expected)
