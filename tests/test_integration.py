import pytest
from scipy.ndimage import binary_dilation, binary_erosion
from noob import Tube, Cube, SynchronousRunner


def test_imperfect_condition() -> None:
    """Start with noisy suff stats"""
    tube = Tube.from_specification()
    cube = Cube.from_specification()
    runner = SynchronousRunner(tube=tube, cube=cube)


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
