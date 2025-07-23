from graphlib import CycleError

import pytest

from cala.models.spec import Pipe


def test_from_id() -> None:
    assert Pipe.from_specification("cala-basic")


def test_cycle_detection() -> None:
    pipe = Pipe.from_specification("cala-cyclic")
    graph = pipe.graph(pipe.init)
    with pytest.raises(CycleError):
        graph.prepare()


def test_dependency_resolution() -> None:
    expected = ["a", "b", "c", "d"]

    pipe = Pipe.from_specification("cala-jumbled")
    graph = pipe.graph(pipe.init)
    assert list(graph.static_order()) == expected
