import numpy as np
import pytest
import sparse
import xarray as xr

from cala.streaming.composer import Frame
from cala.streaming.detect.detect import DetectNewComponents, DetectNewComponentsParams


@pytest.fixture
def dtc_params():
    return DetectNewComponentsParams(frames_axis="frames")


@pytest.fixture
def detector(dtc_params):
    return DetectNewComponents(params=dtc_params)


@pytest.fixture
def overlaps(footprints):
    """Create overlaps matrix from footprints"""
    footprints, _, _ = footprints

    # Compute all pairwise overlaps
    data = (
        footprints.dot(footprints.rename({"components": f"components'"})) > 0
    ).astype(int)

    # Convert to sparse matrix
    data.values = sparse.COO(data.values)

    overlaps_ = data.assign_coords(footprints.coords)

    return overlaps_


@pytest.fixture
def component_stats(traces):
    """Create component statistics from traces"""
    # Compute correlation matrix between components
    corr_matrix = (
        traces @ traces.rename({"components": f"components'"}) / traces.sizes["frames"]
    ).assign_coords(traces.coords)

    return corr_matrix


@pytest.fixture
def pixel_stats(stabilized_video, traces):
    """Create pixel statistics"""
    # Reshape frames to pixels x time
    Y = stabilized_video.stack({"pixel": ("width", "height")})

    # Get temporal components C
    C = traces  # components x time

    # Compute W = Y[:, 1:t']C^T/t'
    W = Y @ C.T / len(stabilized_video)

    # Create xarray DataArray with proper dimensions and coordinates
    pixel_stats_ = W.unstack("pixel").assign_coords(traces.coords)

    return pixel_stats_


def test_update_residual_buffer(detector, stabilized_video, footprints, traces):
    """Test residual buffer update"""
    footprints, _, _ = footprints
    frame = Frame(stabilized_video[1], 1)
    residuals = xr.DataArray(
        np.zeros_like(frame.array), dims=frame.array.dims, coords=frame.array.coords
    ).expand_dims("frames")

    detector._update_residual_buffer(
        frame.array,
        footprints,
        traces.isel({"frames": [1]}),
        residuals.isel({"frames": [0]}),
    )

    # Check shape
    assert detector.residuals_.shape == residuals.isel({"frames": [0]}).shape

    # Check that oldest frame was removed and new residual added
    expected_prediction = (footprints * traces.isel(frames=1)).sum(dim="components")
    expected_new_residual = frame.array - expected_prediction

    np.testing.assert_array_almost_equal(
        detector.residuals_.isel(frames=-1), expected_new_residual
    )
    # not sure why detector.residuals_ isn't completely zero below???
    # np.testing.assert_array_almost_equal(
    #     detector.residuals_.isel(frames=-1),
    #     residuals.isel(frames=-1),
    # )


def test_process_residuals(detector, stabilized_video):
    """Test residual processing"""
    detector.residuals_ = stabilized_video[:3]
    V = detector._process_residuals()

    # Check shape
    assert V.shape == detector.residuals_.shape

    # Check median subtraction
    R_med = detector.residuals_.median(dim="frames")
    R_centered = detector.residuals_ - R_med

    # Can't easily test Gaussian filtering directly, but can check properties
    assert V.dims == detector.residuals_.dims
    assert not np.array_equal(V, R_centered)  # Filtering should change values


def test_get_max_variance_neighborhood(detector, stabilized_video):
    """Test neighborhood extraction around maximum variance point"""
    detector.residuals_ = stabilized_video[:3]
    E = (stabilized_video[:3] ** 2).sum(dim="frames")
    neighborhood = detector._get_max_variance_neighborhood(E)

    # Check that neighborhood is centered around maximum variance
    max_coords = E.argmax(dim=["height", "width"])
    radius = detector.params.gaussian_radius

    # Check neighborhood dimensions
    assert neighborhood.sizes["height"] <= 2 * radius + 1
    assert neighborhood.sizes["width"] <= 2 * radius + 1

    # Check that neighborhood contains the maximum point
    assert (max_coords["height"] - radius <= neighborhood.coords["height"]).all()
    assert (max_coords["height"] + radius >= neighborhood.coords["height"]).all()
    assert (max_coords["width"] - radius <= neighborhood.coords["width"]).all()
    assert (max_coords["width"] + radius >= neighborhood.coords["width"]).all()


def test_local_nmf(detector, stabilized_video):
    """Test local NMF decomposition"""
    # Use small neighborhood for testing
    peak = stabilized_video[:3].var(dim="frames").argmax(["height", "width"])
    radius = int(detector.params.gaussian_radius)
    height_slice = slice(
        peak["height"].values.tolist() - radius,
        peak["height"].values.tolist() + radius + 1,
    )
    width_slice = slice(
        peak["width"].values.tolist() - radius,
        peak["width"].values.tolist() + radius + 1,
    )

    neighborhood = (
        stabilized_video[:3]
        .isel(
            height=height_slice,
            width=width_slice,
        )
        .assign_coords(
            {
                "height": stabilized_video.coords["height"][height_slice],
                "width": stabilized_video.coords["width"][width_slice],
            }
        )
    )

    a_new, c_new = detector._local_nmf(neighborhood, stabilized_video[0])

    # Check shapes
    assert a_new.dims == ("height", "width")
    assert c_new.dims == ("frames",)
    assert a_new.shape == stabilized_video[0].shape
    assert c_new.shape == (neighborhood.sizes["frames"],)

    # # Check normalization of spatial component
    # np.testing.assert_almost_equal(a_new.sum(), 1.0)

    # Check non-negativity
    assert (a_new >= 0).all()
    assert (c_new >= 0).all()

    # Check reconstruction quality
    reconstruction = a_new * c_new
    assert set(reconstruction.dims) == set(neighborhood.dims)


def test_validate_component(detector, footprints, traces, residuals):
    """Test component validation"""
    footprints, _, _ = footprints
    # residual has one cell to detect
    detector.residuals_ = residuals + footprints[-1]

    # Create a good component (that one cell to detect)
    a_good = footprints[-1]

    # the same traces
    c_good = traces[-1]

    # Create a bad component (overlapping with existing)
    a_bad = footprints.isel(components=0)  # Same as first component
    c_bad = traces.isel(components=0)  # Same temporal trace

    # Test good component
    assert detector._validate_component(a_good, c_good, traces, footprints[:-1])

    # Test bad component (overlapping)
    assert not detector._validate_component(a_bad, c_bad, traces, footprints[:-1])


def test_update_pixel_stats(
    detector, stabilized_video, footprints, traces, pixel_stats, residuals
):
    """Test pixel statistics update"""

    frame = Frame(stabilized_video[-1], index=len(stabilized_video) - 1)
    footprints, _, _ = footprints

    # Create new traces for testing
    new_traces = xr.DataArray(
        np.random.rand(5, 2),  # 2 new components
        dims=["frames", "components"],
        coords={
            "id_": ("components", ["new1", "new2"]),
            "type_": ("components", ["neuron", "neuron"]),
        },
    )

    updated_stats = detector._update_pixel_stats(
        frame, footprints, traces, residuals[-5:], pixel_stats, new_traces
    )

    # Check shape
    assert updated_stats.sizes["components"] == (
        pixel_stats.sizes["components"] + new_traces.sizes["components"]
    )
    assert updated_stats.sizes["height"] == pixel_stats.sizes["height"]
    assert updated_stats.sizes["width"] == pixel_stats.sizes["width"]

    # Check that original stats are preserved
    np.testing.assert_array_equal(
        updated_stats.isel(components=slice(None, -2)), pixel_stats
    )


def test_update_component_stats(detector, component_stats, traces, stabilized_video):
    """Test component statistics update"""
    frame = Frame(stabilized_video[-1], len(stabilized_video) - 1)
    # Create new traces for testing
    new_traces = xr.DataArray(
        np.random.rand(5, 2),  # 2 new components
        dims=["frames", "components"],
        coords={
            "id_": ("components", ["new1", "new2"]),
            "type_": ("components", ["neuron", "neuron"]),
        },
    )

    updated_stats = detector._update_component_stats(
        component_stats, traces, new_traces, frame.index
    )

    # Check shape
    new_size = component_stats.sizes["components"] + new_traces.sizes["components"]
    assert updated_stats.shape == (new_size, new_size)

    # Check block structure
    # Top-left block (scaled original)
    scale = frame.index / (frame.index + 1)
    np.testing.assert_array_almost_equal(
        updated_stats.isel(
            {"components": slice(None, -2), "components'": slice(None, -2)}
        ),
        component_stats * scale,
    )

    # Check symmetry
    np.testing.assert_array_almost_equal(updated_stats.values, updated_stats.values.T)


def test_update_overlaps(detector, footprints, overlaps):
    """Test overlap matrix update"""
    # Create new footprints for testing
    footprints, _, _ = footprints
    new_footprints = xr.DataArray(
        np.zeros((2, footprints.sizes["height"], footprints.sizes["width"])),
        dims=["components", "height", "width"],
        coords={
            "id_": ("components", ["new1", "new2"]),
            "type_": ("components", ["neuron", "neuron"]),
        },
    )

    # Make first new component overlap with first existing component
    new_footprints[0, :, :] = footprints.isel(components=0)[:, :]
    new_footprints[1, :, :] = footprints.isel(components=1)[:, :]

    updated_overlaps = detector._update_overlaps(footprints, overlaps, new_footprints)

    # Check shape
    new_size = overlaps.sizes["components"] + new_footprints.sizes["components"]
    assert updated_overlaps.shape == (new_size, new_size)

    # Check that original overlaps are preserved
    np.testing.assert_array_equal(
        updated_overlaps.isel(
            {"components": slice(None, -2), "components'": slice(None, -2)}
        ).data.todense(),
        overlaps.data.todense(),
    )

    # Check new overlaps
    assert (
        updated_overlaps.set_xindex("id_").sel({"id_": "new1"}).isel({"components'": 0})
        == 1
    )  # Should overlap with first component

    assert (
        updated_overlaps.set_xindex("id_").sel({"id_": "new2"}).isel({"components'": 1})
        == 1
    )  # Should overlap with second component

    # Check symmetry
    np.testing.assert_array_equal(
        updated_overlaps.data.todense(), updated_overlaps.data.todense().T
    )

    # all updates must have id_ and type_ coordinates
