import pytest
import xarray as xr
from noob import Cube, SynchronousRunner, Tube

from cala.models import AXIS


@pytest.fixture
def odl_tube():
    return Tube.from_specification("cala-odl")


@pytest.fixture
def odl_cube():
    return Cube.from_specification("cala-odl")


@pytest.fixture
def odl_runner(odl_tube, odl_cube):
    return SynchronousRunner(tube=odl_tube, cube=odl_cube)


def test_process(odl_runner) -> None:
    """Start with noisy suff stats"""
    odl_runner.init()
    odl_runner.process()

    assert odl_runner.cube.assets["buffer"].obj.array.size > 0


def test_iter(odl_runner) -> None:
    gen = odl_runner.iter(n=30)

    movie = []
    for _, exp in enumerate(gen):
        movie.append(exp[0].array)
        fps = odl_runner.cube.assets["footprints"].obj
        trs = odl_runner.cube.assets["traces"].obj

    expected = xr.concat(movie, dim=AXIS.frames_dim)
    result = (fps.array @ trs.array).transpose(*expected.dims)

    xr.testing.assert_allclose(expected, result, atol=1e-5, rtol=1e-5)


@pytest.mark.xfail
def test_run(odl_runner) -> None:
    result = odl_runner.run(n=5)

    assert result


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
