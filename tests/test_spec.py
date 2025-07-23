from graphlib import CycleError

import pytest

from cala.models.spec import Pipe


def test_from_id() -> None:
    pipe = Pipe.from_specification("cala-basic")
    assert (
        list(pipe.prep.nodes.keys())
        == list(pipe.init.nodes.keys())
        == list(pipe.iter.nodes.keys())
        == ["A", "B", "C"]
    )


def test_cycle_detection() -> None:
    pipe = Pipe.from_specification("cala-cyclic")
    graph = pipe.init.graph()
    with pytest.raises(CycleError):
        graph.prepare()


def test_dependency_resolution() -> None:
    expected = ["A", "B", "C", "D"]

    pipe = Pipe.from_specification("cala-jumbled")
    graph = pipe.init.graph()
    assert list(graph.static_order()) == expected
