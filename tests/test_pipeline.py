import numpy as np
import pytest
import xarray as xr
from noob import SynchronousRunner, Tube
from noob.node import Node, NodeSpecification

from cala.models import AXIS


@pytest.fixture(
    params=[
        # "SingleCellSource",
        # "TwoCellsSource",
        # "SeparateSource",
        # "TwoOverlappingSource",
        # "GradualOnSource",
        "SplitOffSource",
    ]
)
def source(request):
    return Node.from_specification(
        NodeSpecification(id="source", type=f"cala.testing.{request.param}")
    )


@pytest.fixture
def tube(source, tmp_path):
    tube = Tube.from_specification("cala-odl")
    tube.nodes["source"] = source
    # tube.cube.assets["traces"].params["zarr_path"] = tmp_path / "traces"
    # tube.cube.assets["traces"].params["peek_size"] = 100

    return tube


@pytest.fixture
def runner(tube, request):
    return SynchronousRunner(tube=tube)


def test_process(runner) -> None:
    runner.init()
    runner.process()

    assert runner.tube.cube.assets["buffer"].obj.array.size > 0


@pytest.mark.xfail(raises=NotImplementedError)
def test_odl(runner, source) -> None:
    gen = runner.iter(n=source.instance.n_frames)
    src_name = source.spec.type_.split(".")[-1]
    toy = source.instance._toy.model_copy()

    preprocessed_frames = []
    for fr in gen:
        preprocessed_frames.append(fr["prep"].array)
        fps = runner.tube.cube.assets["footprints"].obj
        trs = runner.tube.cube.assets["traces"].obj

    # Correct component count
    if src_name not in ["SeparateSource", "SplitOffSource"]:
        assert toy.traces.array.sizes[AXIS.component_dim] == trs.array.sizes[AXIS.component_dim]
    elif src_name == "SeparateSource":
        # 2 is the # of discoverable cells (non-constant) for SeparateSource
        assert trs.array.sizes[AXIS.component_dim] == 2
    elif src_name == "SplitOffSource":
        # 3 because one should be deprecated
        assert trs.array.sizes[AXIS.component_dim] == 3

    if src_name in ["TwoOverlappingSource", "GradualOnSource"]:
        # Traces are reasonably similar
        tr_corr = xr.corr(
            toy.traces.array, trs.array.rename(AXIS.component_rename), dim=AXIS.frame_coord
        )
        for corr in tr_corr:
            assert np.isclose(corr.max(), 1, atol=1e-2)

    elif src_name in ["SingleCellSource", "TwoCellsSource", "SeparateSource"]:
        expected = xr.concat(preprocessed_frames, dim=AXIS.frame_coord)
        result = (fps.array @ trs.array).transpose(*expected.dims)

        xr.testing.assert_allclose(expected, result, atol=1e-5, rtol=1e-5)

    elif src_name == "SplitOffSource":
        expected = xr.concat(preprocessed_frames, dim=AXIS.frame_coord)
        result = (fps.array @ trs.array).transpose(*expected.dims)
        raise NotImplementedError("Deprecation not implemented")


# def test_with_src():
#     tube = Tube.from_specification("cala-with-ca1")
#     runner = SynchronousRunner(tube=tube)
#     runner.run()
#
#     fps = runner.tube.cube.assets["footprints"].obj
#     assert fps
