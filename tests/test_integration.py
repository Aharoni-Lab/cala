import pytest
from noob import Cube, SynchronousRunner, Tube
from scipy.ndimage import binary_dilation, binary_erosion


def test_imperfect_condition() -> None:
    """Start with noisy suff stats"""
    cube = Cube.from_specification("cala-odl")
    tube = Tube.from_specification("cala-odl")
    runner = SynchronousRunner(tube=tube, cube=cube)

    runner.init()
    runner.process()

    assert cube.assets["buffer"].obj.array.size > 0


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
