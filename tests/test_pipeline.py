import pytest
from noob import Cube, SynchronousRunner, Tube
from scipy.ndimage import binary_dilation, binary_erosion


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


@pytest.mark.xfail
def test_iter(odl_runner) -> None:
    gen = odl_runner.iter()

    result = next(gen)

    assert result


@pytest.mark.xfail
def test_run(odl_runner) -> None:
    result = odl_runner.run(n=5)

    assert result


@pytest.mark.xfail
def test_combined_footprint() -> None:
    """Start with two footprints combined"""
    assert NotImplementedError()


@pytest.mark.xfail
def test_dilating_footprint() -> None:
    """start with binary-eroded footprints"""
    assert NotImplementedError()


@pytest.mark.xfail
def test_eroding_footprint() -> None:
    """start with binary-dilated footprints"""
    assert NotImplementedError()


@pytest.mark.xfail
def test_redundant_footprint() -> None:
    """start with redundant footprints"""
    assert NotImplementedError()
