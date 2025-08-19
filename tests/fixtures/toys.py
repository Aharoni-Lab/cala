import pytest

from cala.testing import SingleCellSource, SeparateSource, ConnectedSource
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
