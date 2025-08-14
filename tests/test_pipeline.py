import pytest
import xarray as xr
from noob import Cube, SynchronousRunner, Tube
from noob.node import NodeSpecification, Node

from cala.models import AXIS


@pytest.fixture(params=["single_cell_source", "two_cells_source", "two_overlapping_source"])
def tube(request):
    tube = Tube.from_specification("cala-odl")
    source = Node.from_specification(
        NodeSpecification(
            id="source", type=f"cala.testing.{request.param}", params={"n_frames": 50}
        )
    )
    tube.nodes["source"] = source

    return tube


@pytest.fixture
def cube():
    return Cube.from_specification("cala-odl")


@pytest.fixture
def runner(tube, cube, request):
    return SynchronousRunner(tube=tube, cube=cube)


def test_process(runner) -> None:
    """Start with noisy suff stats"""
    runner.init()
    runner.process()

    assert runner.cube.assets["buffer"].obj.array.size > 0


@pytest.mark.xfail
def test_iter(runner) -> None:
    gen = runner.iter(n=runner.tube.nodes["source"].spec.params["n_frames"])

    movie = []
    for _, exp in enumerate(gen):
        movie.append(exp[0].array)
        fps = runner.cube.assets["footprints"].obj
        trs = runner.cube.assets["traces"].obj

    expected = xr.concat(movie, dim=AXIS.frames_dim)
    result = (fps.array @ trs.array).transpose(*expected.dims)

    xr.testing.assert_allclose(expected, result, atol=1e-5, rtol=1e-5)


@pytest.mark.xfail
def test_run(runner) -> None:
    result = runner.run(n=5)

    assert result


@pytest.mark.xfail
def test_combined_footprint() -> None:
    """Start with two footprints combined"""
    raise AssertionError("Not implemented")


@pytest.mark.xfail
def test_redundant_footprint() -> None:
    """start with redundant footprints"""
    raise AssertionError("Not implemented")
