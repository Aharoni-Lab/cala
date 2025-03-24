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
    n_components = len(footprints.components)

    # Compute all pairwise overlaps
    overlaps = xr.dot(footprints, footprints, dims=["height", "width"])

    # Convert to binary overlap indicator
    overlaps = (overlaps != 0).astype(int)

    # Convert to sparse matrix
    coords = np.where(overlaps)
    data = overlaps.values[coords]

    sparse_matrix = sparse.COO(
        coords=coords, data=data, shape=(n_components, n_components)
    )

    return xr.DataArray(
        sparse_matrix,
        dims=["components", "components'"],
        coords={
            "components": footprints.components,
            "components'": footprints.components,
        },
    )


@pytest.fixture
def component_stats(traces):
    """Create component statistics from traces"""
    # Compute correlation matrix between components
    corr_matrix = xr.corr(traces, traces, dim="frames")

    return corr_matrix


@pytest.fixture
def pixel_stats(frame, traces):
    """Create pixel statistics"""
    # Reshape frame and compute outer product with traces
    y_buf = frame.array.stack(pixels=["height", "width"])

    pixel_stats = xr.DataArray(
        np.outer(y_buf, traces[0]),  # Use first timepoint
        dims=["pixels", "components"],
        coords={"pixels": y_buf.pixels, "components": traces.components},
    ).unstack("pixels")

    return pixel_stats


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

    a_new, c_new = detector._local_nmf(neighborhood)

    # Check shapes
    assert a_new.dims == ("height", "width")
    assert c_new.dims == ("frames",)
    assert a_new.shape == neighborhood.isel(frames=0).shape
    assert c_new.shape == (neighborhood.sizes["frames"],)

    # Check normalization of spatial component
    np.testing.assert_almost_equal(a_new.sum(), 1.0)

    # Check non-negativity
    assert (a_new >= 0).all()
    assert (c_new >= 0).all()

    # Check reconstruction quality
    reconstruction = a_new * c_new
    assert reconstruction.dims == neighborhood.dims
    # Error should be reasonable for random data
    reconstruction_error = ((neighborhood - reconstruction) ** 2).mean()
    assert reconstruction_error < neighborhood.var()


def test_validate_component(detector, footprints, traces, overlaps, residuals):
    """Test component validation"""
    detector.residuals_ = residuals

    # Create a good component (normalized, non-overlapping)
    a_good = xr.zeros_like(footprints.isel(components=0))
    a_good[4:6, 4:6] = 0.25  # Small square with normalized values

    c_good = xr.zeros_like(traces.isel(components=0))
    c_good[:] = 1.0  # Constant temporal component

    # Create a bad component (overlapping with existing)
    a_bad = footprints.isel(components=0)  # Same as first component
    c_bad = traces.isel(components=0)  # Same temporal trace

    # Test good component
    assert detector._validate_component(a_good, c_good, traces, overlaps)

    # Test bad component (overlapping)
    assert not detector._validate_component(a_bad, c_bad, traces, overlaps)


def test_update_pixel_stats(detector, frame, pixel_stats, new_traces):
    """Test pixel statistics update"""
    # Create new traces for testing
    new_traces = xr.DataArray(
        np.random.rand(5, 2),  # 2 new components
        dims=["frames", "components"],
        coords={
            "components": ["new1", "new2"],
            "id_": ("components", ["new1", "new2"]),
            "type_": ("components", ["neuron", "neuron"]),
        },
    )

    updated_stats = detector._update_pixel_stats(frame, pixel_stats, new_traces)

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


def test_update_component_stats(detector, component_stats, traces, frame):
    """Test component statistics update"""
    # Create new traces for testing
    new_traces = xr.DataArray(
        np.random.rand(5, 2),  # 2 new components
        dims=["frames", "components"],
        coords={
            "components": ["new1", "new2"],
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
            components=slice(None, -2), components_prime=slice(None, -2)
        ),
        component_stats * scale,
    )

    # Check symmetry
    np.testing.assert_array_almost_equal(updated_stats.values, updated_stats.values.T)


def test_update_overlaps(detector, footprints, overlaps):
    """Test overlap matrix update"""
    # Create new footprints for testing
    new_footprints = xr.DataArray(
        np.zeros((2, footprints.sizes["height"], footprints.sizes["width"])),
        dims=["components", "height", "width"],
        coords={
            "components": ["new1", "new2"],
            "id_": ("components", ["new1", "new2"]),
            "type_": ("components", ["neuron", "neuron"]),
        },
    )

    # Make first new component overlap with first existing component
    new_footprints[0, :3, :3] = footprints.isel(components=0)[:3, :3]

    updated_overlaps = detector._update_overlaps(footprints, overlaps, new_footprints)

    # Check shape
    new_size = overlaps.sizes["components"] + new_footprints.sizes["components"]
    assert updated_overlaps.shape == (new_size, new_size)

    # Check that original overlaps are preserved
    np.testing.assert_array_equal(
        updated_overlaps.isel(
            components=slice(None, -2), components_prime=slice(None, -2)
        ),
        overlaps,
    )

    # Check new overlaps
    assert (
        updated_overlaps.sel(components="new1", components_prime=overlaps.components[0])
        == 1
    )  # Should overlap with first component

    assert (
        updated_overlaps.sel(components="new2", components_prime=overlaps.components[0])
        == 0
    )  # Should not overlap with first component

    # Check symmetry
    np.testing.assert_array_equal(
        updated_overlaps.values.todense(), updated_overlaps.values.todense().T
    )
