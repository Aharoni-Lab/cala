import numpy as np
import pytest
import xarray as xr
from noob import Cube, SynchronousRunner, Tube
from noob.node import Node, NodeSpecification

from cala.models import AXIS


@pytest.fixture(
    params=[
        "SingleCellSource",
        "TwoCellsSource",
        "SeparateSource",
        "TwoOverlappingSource",
        "GradualOnSource",
        "SplitOffSource",
    ]
)
def source(request):
    return Node.from_specification(
        NodeSpecification(id="source", type=f"cala.testing.{request.param}")
    )


@pytest.fixture
def tube(source):
    tube = Tube.from_specification("cala-odl")
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


def test_odl(runner, source) -> None:
    gen = runner.iter(n=source.instance.n_frames)
    src_name = source.spec.type_.split(".")[-1]
    toy = source.instance._toy.model_copy()

    preprocessed_frames = []
    for fr in gen:
        preprocessed_frames.append(fr[0].array)
        fps = runner.cube.assets["footprints"].obj
        trs = runner.cube.assets["traces"].obj

    # Correct component count
    if src_name != "SeparateSource":
        assert toy.traces.array.sizes[AXIS.component_dim] == trs.array.sizes[AXIS.component_dim]
    else:
        # 2 is the # of discoverable cells (non-constant) for SeparateSource
        assert trs.array.sizes[AXIS.component_dim] == 2

    if src_name in ["TwoOverlappingSource", "GradualOnSource", "SplitOffSource"]:
        # Traces are reasonably similar
        tr_corr = xr.corr(
            toy.traces.array, trs.array.rename(AXIS.component_rename), dim=AXIS.frame_coord
        )
        for corr in tr_corr:
            assert np.isclose(corr.max(), 1, atol=1e-2)

    else:
        expected = xr.concat(preprocessed_frames, dim=AXIS.frame_coord)
        result = (fps.array @ trs.array).transpose(*expected.dims)

        xr.testing.assert_allclose(expected, result, atol=1e-5, rtol=1e-5)


def test_with_avi() -> None:
    cube = Cube.from_specification("cala-io")
    tube = Tube.from_specification("cala-io")

    runner = SynchronousRunner(tube=tube, cube=cube)

    import matplotlib.pyplot as plt

    gen = runner.iter()
    for i, fr in enumerate(gen):
        raw = fr["raw"].array
        prep = fr["prep"].array / fr["prep"].array.max() * raw.max()
        plt.imsave(f"out{i}.png", xr.concat([raw, prep], dim=AXIS.height_dim))

    assert runner.cube.assets["buffer"].obj.array.size > 0
