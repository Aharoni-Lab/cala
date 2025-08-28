import pytest

from cala.testing import ConnectedSource, SeparateSource, SingleCellSource, SplitOffSource
from cala.testing.toy import Toy


@pytest.fixture
def single_cell() -> Toy:
    source = SingleCellSource()
    return source._toy


@pytest.fixture
def separate_cells() -> Toy:
    source = SeparateSource()
    return source._toy


@pytest.fixture
def connected_cells() -> Toy:
    source = ConnectedSource()
    return source._toy


@pytest.fixture
def splitoff_cells() -> Toy:
    source = SplitOffSource()
    return source._toy
