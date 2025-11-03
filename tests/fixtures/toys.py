import pytest

from cala.testing import (
    ConnectedSource,
    SeparateSource,
    SingleCellSource,
    SplitOffSource,
    GradualOnSource,
    TwoCellsSource,
)
from cala.testing.toy import Toy


@pytest.fixture
def single_cell() -> Toy:
    source = SingleCellSource()
    return source.toy


@pytest.fixture
def two_cells() -> Toy:
    source = SingleCellSource()
    return source.toy


@pytest.fixture
def two_connected_cells() -> Toy:
    source = TwoCellsSource()
    return source.toy


@pytest.fixture
def four_separate_cells() -> Toy:
    source = SeparateSource()
    return source.toy


@pytest.fixture
def four_connected_cells() -> Toy:
    source = ConnectedSource()
    return source.toy


@pytest.fixture
def gradualon_cells() -> Toy:
    source = GradualOnSource()
    return source.toy


@pytest.fixture
def splitoff_cells() -> Toy:
    source = SplitOffSource()
    return source.toy
