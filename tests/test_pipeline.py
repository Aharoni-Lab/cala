import pytest
import xarray as xr
from noob import Cube, SynchronousRunner, Tube

from cala.models import AXIS


@pytest.fixture(params=["cala-single-cell"])
def tube(request):
    return Tube.from_specification(request.param)


@pytest.fixture
def cube():
    return Cube.from_specification("cala-single-cell")


@pytest.fixture
def runner(tube, cube, request):
    return SynchronousRunner(tube=tube, cube=cube)


def test_process(runner) -> None:
    """Start with noisy suff stats"""
    runner.init()
    runner.process()

    assert runner.cube.assets["buffer"].obj.array.size > 0


def test_iter(runner) -> None:
    gen = runner.iter(n=30)

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
    """
    Currently, runner.cube gets deinit'd with tube as the runner finishes iteration/run.
    This prevents persisting cube assets beyond the runner lifecycle,
    which might not be what we want. Maybe the whole point of cube is to live
    _beyond_ the lifecycle of tube.
    """
    result = runner.run(n=5)

    assert result
    assert runner.cube.assets["footprints"].obj.array.size > 0


@pytest.mark.xfail
def test_combined_footprint() -> None:
    """Start with two footprints combined"""
    raise AssertionError("Not implemented")


@pytest.mark.xfail
def test_dilating_footprint() -> None:
    """start with binary-eroded footprints"""
    raise AssertionError("Not implemented")


@pytest.mark.xfail
def test_eroding_footprint() -> None:
    """start with binary-dilated footprints"""
    raise AssertionError("Not implemented")


@pytest.mark.xfail
def test_redundant_footprint() -> None:
    """start with redundant footprints"""
    raise AssertionError("Not implemented")
