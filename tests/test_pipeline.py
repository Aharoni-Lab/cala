import pytest
import xarray as xr
from noob import Cube, SynchronousRunner, Tube
from noob.node import Node, NodeSpecification

from cala.models import AXIS


@pytest.fixture(
    params=[
        "SingleCellSource",
        "TwoCellsSource",
        "TwoOverlappingSource",
        "SeparateSource",
        "GradualOnSource",
    ]
)
def tube(request):
    tube = Tube.from_specification("cala-odl")
    source = Node.from_specification(
        NodeSpecification(id="source", type=f"cala.testing.{request.param}")
    )
    tube.nodes["source"] = source

    return tube


@pytest.fixture
def cube():
    return Cube.from_specification("cala-odl")


@pytest.fixture
def runner(tube, cube, request):
    return SynchronousRunner(tube=tube, cube=cube)


def test_process(runner) -> None:
    """Start with noisy suff stats"""
    runner.init()
    runner.process()

    assert runner.cube.assets["buffer"].obj.array.size > 0


def test_iter(runner) -> None:
    gen = runner.iter(n=runner.tube.nodes["source"].instance.n_frames)

    movie = []
    for _, exp in enumerate(gen):
        movie.append(exp[0].array)
        fps = runner.cube.assets["footprints"].obj
        trs = runner.cube.assets["traces"].obj

    expected = xr.concat(movie, dim=AXIS.frames_dim)
    result = (fps.array @ trs.array).transpose(*expected.dims)

    src_node = runner.tube.nodes["source"].spec.type_.split(".")[-1]

    if src_node == "TwoOverlappingSource":
        diff = expected - result
        for d_fr, e_fr in zip(diff, expected):
            assert d_fr.max() <= e_fr.quantile(0.98) * 2e-2
    else:
        xr.testing.assert_allclose(expected, result, atol=1e-5, rtol=1e-5)

    n_discoverable = {
        "SingleCellSource": 1,
        "TwoCellsSource": 2,
        "TwoOverlappingSource": 2,
        "SeparateSource": 2,
    }
    assert fps.array.sizes[AXIS.component_dim] == n_discoverable[src_node]


def test_run(runner) -> None:
    result = runner.run(n=5)

    assert result
