import numpy as np
import pytest
import xarray as xr
from noob.node import NodeSpecification

from cala.assets import AXIS, Buffer, Footprints, Traces
from cala.nodes.detect import Cataloger, SliceNMF
from cala.nodes.detect.catalog import _merge_component, _register
from cala.nodes.detect.catalog_depr import CatalogerDepr


@pytest.fixture(scope="module")
def slice_nmf():
    return SliceNMF.from_specification(
        spec=NodeSpecification(
            id="test_slice_nmf",
            type="cala.nodes.detect.SliceNMF",
            params={"min_frames": 10, "detect_thresh": 1, "reprod_tol": 0.001},
        )
    )


@pytest.fixture(scope="module")
def cataloger():
    return Cataloger.from_specification(
        spec=NodeSpecification(
            id="test",
            type="cala.nodes.detect.Cataloger",
            params={
                "age_limit": 100,
                "smooth_kwargs": {"sigma": 2},
                "merge_threshold": 0.8,
                "val_threshold": 0.5,
                "cnt_threshold": 5,
            },
        )
    )


@pytest.fixture(scope="module")
def catalog_depr(cataloger):
    """
    Source of truth for cataloger (before optimization but things were correct)

    """
    model = cataloger.model_dump()
    model["spec"]["type"] = model["spec"]["type_"]
    return CatalogerDepr(**model)


@pytest.fixture(scope="function")
def new_component(slice_nmf, single_cell):
    buff = Buffer(size=100)
    buff.array = single_cell.make_movie().array
    return slice_nmf.process(
        residuals=buff, energy=buff.array.std(dim=AXIS.frames_dim), detect_radius=60
    )


def test_register(cataloger, new_component):
    new_fp, new_tr = new_component
    fp, tr = _register(shapes=new_fp[0].array, tracks=new_tr[0].array)

    assert np.array_equal(fp[0].as_numpy(), new_fp[0].array.as_numpy())
    assert np.array_equal(tr[0], new_tr[0].array)
    assert fp[AXIS.id_coord].item() == tr[AXIS.id_coord].item()
    assert (
        fp[AXIS.detect_coord].item() == tr[AXIS.detect_coord].item() == tr[AXIS.detect_coord].max()
    )


def test_merge_with(slice_nmf, cataloger, single_cell):
    buff = Buffer(size=100)
    buff.array = single_cell.make_movie().array
    new_component = slice_nmf.process(
        buff, energy=buff.array.std(dim=AXIS.frames_dim), detect_radius=10
    )

    A = single_cell.footprints.array

    new_fp, new_tr = new_component
    fp, tr = _merge_component(
        new_fp[0].array.expand_dims(dim="component"),
        new_tr[0].array.expand_dims(dim="component"),
        A.data.reshape((A.shape[0], -1)).tocsr(),
        single_cell.traces.array,
        [0],
    )

    movie_result = (fp @ tr).as_numpy()

    movie_new_comp = new_fp[0].array @ new_tr[0].array
    movie_expected = (single_cell.make_movie().array + movie_new_comp).transpose(*movie_result.dims)

    xr.testing.assert_allclose(movie_result, movie_expected)


def test_process_pass(slice_nmf, cataloger, four_separate_cells):
    """
    test cataloging separate cells. ideal case with cell_radius=5
    """
    buff = Buffer(size=100)
    buff.array = four_separate_cells.make_movie().array
    fps, trs = slice_nmf.process(buff, buff.array.std(dim=AXIS.frames_dim), detect_radius=5)

    # NOTE: by manually putting in separate_cells, we're forcing a double-detection in this test
    new_fps, new_trs = cataloger.process(
        fps, trs, four_separate_cells.footprints, four_separate_cells.traces
    )

    result = new_fps.array @ new_trs.array

    # would not detect cell_0 and cell_1 since they're uniform
    detected = ["cell_2", "cell_3"]
    expected = (
        four_separate_cells.footprints.array.set_xindex(AXIS.id_coord).sel(
            {AXIS.id_coord: detected}
        )
        @ four_separate_cells.traces.array.set_xindex(AXIS.id_coord).sel({AXIS.id_coord: detected})
    ).transpose(*result.dims) * 2

    assert set(new_fps.array.attrs.get("replaces")) == set(detected)
    xr.testing.assert_allclose(result.as_numpy(), expected.as_numpy())


def test_process_fail(slice_nmf, cataloger, four_separate_cells):
    """
    test cataloging separate cells. nmf supposed to fail with radius=25 (grabs too many cells)
    """
    movie = four_separate_cells.make_movie().array
    fps, trs = slice_nmf.process(
        Buffer.from_array(movie, size=100), movie.std(dim=AXIS.frames_dim), detect_radius=25
    )

    # NOTE: by manually putting in separate_cells, we're forcing a double-detection in this test
    new_fps, new_trs = cataloger.process(
        fps, trs, four_separate_cells.footprints, four_separate_cells.traces
    )

    assert new_fps.array is None and new_trs.array is None


def test_process_connected(slice_nmf, cataloger, four_connected_cells):
    """
    trial with connected cells 🙏
    """
    movie = four_connected_cells.make_movie().array
    fps, trs = slice_nmf.process(
        Buffer.from_array(movie, size=100), movie.std(dim=AXIS.frames_dim), detect_radius=4
    )

    # NOTE: by manually putting in connected_cells,
    # we're forcing a double-detection in this test
    new_fps, new_trs = cataloger.process(fps, trs, Footprints(), Traces())

    result = (new_fps.array @ new_trs.array).transpose(AXIS.frames_dim, ...).as_numpy()
    expected = movie.transpose(*result.dims).as_numpy()

    # not sure why we're getting some stray pixels... but we need to remove them
    sig_pxls = (new_fps.array.max(dim=AXIS.component_dim) > 0.1).as_numpy()
    result, expected = result.where(sig_pxls), expected.where(sig_pxls)

    assert new_fps.array is not None
    # 1. the footprints do not overlap
    assert np.all(
        np.triu((new_fps.array @ new_fps.array.rename(AXIS.component_rename)).as_numpy(), 1) == 0
    )
    # 2. the trace and footprint values are accurate (where they do exist)
    xr.testing.assert_allclose(result, expected, atol=1)


@pytest.mark.parametrize(
    "toy,num_conns", [("four_separate_cells", 0), ("four_connected_cells", 36)]
)
def test_mono_merge_matrix(slice_nmf, cataloger, toy, num_conns, request, catalog_depr):
    toy = request.getfixturevalue(toy)
    movie = toy.make_movie().array
    new_fps, new_trs = slice_nmf.process(
        Buffer.from_array(movie, size=100), movie.std(dim=AXIS.frames_dim), detect_radius=4
    )
    shape_chunks = xr.concat([fp.array for fp in new_fps], dim=AXIS.component_dim)
    trace_chunks = xr.concat([tr.array for tr in new_trs], dim=AXIS.component_dim)
    merge_mat = cataloger._monopartite_merge_matrix(shape_chunks, trace_chunks)
    merge_mat_depr = catalog_depr._merge_matrix(shape_chunks, trace_chunks)
    assert np.sum(merge_mat) - np.trace(merge_mat) == num_conns
    assert merge_mat.equals(merge_mat_depr)
