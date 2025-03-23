import numpy as np
import pytest
import sparse
import xarray as xr

from cala.streaming.composer import Frame
from cala.streaming.detect.detect import DetectNewComponents, DetectNewComponentsParams
from tests.fixtures import (
    footprints,
    traces,
    residuals,
)


@pytest.fixture
def params():
    return DetectNewComponentsParams()


@pytest.fixture
def detector(params):
    return DetectNewComponents(params=params)


@pytest.fixture
def frame(residuals):
    """Create a test frame"""
    array = residuals[0]  # Take first frame
    return Frame(array=array, index=0)


@pytest.fixture
def test_footprints(footprints):
    """Use footprints directly from fixtures"""
    footprints_xr, _, _ = footprints
    return footprints_xr


@pytest.fixture
def test_traces(traces):
    """Use traces directly from fixtures"""
    return traces


@pytest.fixture
def test_residuals(residuals):
    """Use residuals directly from fixtures"""
    return residuals[-5:]  # Use last 5 frames as residual buffer


@pytest.fixture
def test_overlaps(test_footprints):
    """Create overlaps matrix from footprints"""
    n_components = len(test_footprints.components)

    # Compute all pairwise overlaps
    overlaps = xr.dot(test_footprints, test_footprints, dims=["height", "width"])

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
        dims=["components", "components_prime"],
        coords={
            "components": test_footprints.components,
            "components_prime": test_footprints.components,
        },
    )


@pytest.fixture
def test_component_stats(test_traces):
    """Create component statistics from traces"""
    # Compute correlation matrix between components
    corr_matrix = xr.corr(test_traces, test_traces, dim="frames")

    return corr_matrix


@pytest.fixture
def test_pixel_stats(frame, test_traces):
    """Create pixel statistics"""
    # Reshape frame and compute outer product with traces
    y_buf = frame.array.stack(pixels=["height", "width"])

    pixel_stats = xr.DataArray(
        np.outer(y_buf, test_traces[0]),  # Use first timepoint
        dims=["pixels", "components"],
        coords={"pixels": y_buf.pixels, "components": test_traces.components},
    ).unstack("pixels")

    return pixel_stats


def test_update_residual_buffer(
    detector, frame, test_footprints, test_traces, test_residuals
):
    """Test residual buffer update"""
    detector._update_residual_buffer(
        frame.array, test_footprints, test_traces, test_residuals
    )

    # Check shape
    assert detector.residuals_.shape == test_residuals.shape

    # Check that oldest frame was removed and new residual added
    expected_prediction = (test_footprints * test_traces.isel(frames=-1)).sum(
        dim="components"
    )
    expected_new_residual = frame.array - expected_prediction

    np.testing.assert_array_almost_equal(
        detector.residuals_.isel(frames=-1), expected_new_residual
    )
    np.testing.assert_array_almost_equal(
        detector.residuals_.isel(frames=slice(None, -1)),
        test_residuals.isel(frames=slice(1, None)),
    )


def test_process_residuals(detector, test_residuals):
    """Test residual processing"""
    detector.residuals_ = test_residuals
    V = detector._process_residuals()

    # Check shape
    assert V.shape == test_residuals.shape

    # Check median subtraction
    R_med = test_residuals.median(dim="frames")
    R_centered = test_residuals - R_med

    # Can't easily test Gaussian filtering directly, but can check properties
    assert V.dims == test_residuals.dims
    assert not np.array_equal(V, R_centered)  # Filtering should change values


def test_get_max_variance_neighborhood(detector, test_residuals):
    """Test neighborhood extraction around maximum variance point"""
    detector.residuals_ = test_residuals
    E = (test_residuals**2).sum(dim="frames")
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


def test_local_nmf(detector, test_residuals):
    """Test local NMF decomposition"""
    # Use small neighborhood for testing
    neighborhood = test_residuals.isel(height=slice(0, 3), width=slice(0, 3))

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


def test_validate_component(
    detector, test_footprints, test_traces, test_overlaps, test_residuals
):
    """Test component validation"""
    detector.residuals_ = test_residuals

    # Create a good component (normalized, non-overlapping)
    a_good = xr.zeros_like(test_footprints.isel(components=0))
    a_good[4:6, 4:6] = 0.25  # Small square with normalized values

    c_good = xr.zeros_like(test_traces.isel(components=0))
    c_good[:] = 1.0  # Constant temporal component

    # Create a bad component (overlapping with existing)
    a_bad = test_footprints.isel(components=0)  # Same as first component
    c_bad = test_traces.isel(components=0)  # Same temporal trace

    # Test good component
    assert detector._validate_component(a_good, c_good, test_traces, test_overlaps)

    # Test bad component (overlapping)
    assert not detector._validate_component(a_bad, c_bad, test_traces, test_overlaps)


def test_update_pixel_stats(detector, frame, test_pixel_stats, new_traces):
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

    updated_stats = detector._update_pixel_stats(frame, test_pixel_stats, new_traces)

    # Check shape
    assert updated_stats.sizes["components"] == (
        test_pixel_stats.sizes["components"] + new_traces.sizes["components"]
    )
    assert updated_stats.sizes["height"] == test_pixel_stats.sizes["height"]
    assert updated_stats.sizes["width"] == test_pixel_stats.sizes["width"]

    # Check that original stats are preserved
    np.testing.assert_array_equal(
        updated_stats.isel(components=slice(None, -2)), test_pixel_stats
    )


def test_update_component_stats(detector, test_component_stats, test_traces, frame):
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
        test_component_stats, test_traces, new_traces, frame.index
    )

    # Check shape
    new_size = test_component_stats.sizes["components"] + new_traces.sizes["components"]
    assert updated_stats.shape == (new_size, new_size)

    # Check block structure
    # Top-left block (scaled original)
    scale = frame.index / (frame.index + 1)
    np.testing.assert_array_almost_equal(
        updated_stats.isel(
            components=slice(None, -2), components_prime=slice(None, -2)
        ),
        test_component_stats * scale,
    )

    # Check symmetry
    np.testing.assert_array_almost_equal(updated_stats.values, updated_stats.values.T)


def test_update_overlaps(detector, test_footprints, test_overlaps):
    """Test overlap matrix update"""
    # Create new footprints for testing
    new_footprints = xr.DataArray(
        np.zeros((2, test_footprints.sizes["height"], test_footprints.sizes["width"])),
        dims=["components", "height", "width"],
        coords={
            "components": ["new1", "new2"],
            "id_": ("components", ["new1", "new2"]),
            "type_": ("components", ["neuron", "neuron"]),
        },
    )

    # Make first new component overlap with first existing component
    new_footprints[0, :3, :3] = test_footprints.isel(components=0)[:3, :3]

    updated_overlaps = detector._update_overlaps(
        test_footprints, test_overlaps, new_footprints
    )

    # Check shape
    new_size = test_overlaps.sizes["components"] + new_footprints.sizes["components"]
    assert updated_overlaps.shape == (new_size, new_size)

    # Check that original overlaps are preserved
    np.testing.assert_array_equal(
        updated_overlaps.isel(
            components=slice(None, -2), components_prime=slice(None, -2)
        ),
        test_overlaps,
    )

    # Check new overlaps
    assert (
        updated_overlaps.sel(
            components="new1", components_prime=test_overlaps.components[0]
        )
        == 1
    )  # Should overlap with first component

    assert (
        updated_overlaps.sel(
            components="new2", components_prime=test_overlaps.components[0]
        )
        == 0
    )  # Should not overlap with first component

    # Check symmetry
    np.testing.assert_array_equal(
        updated_overlaps.values.todense(), updated_overlaps.values.todense().T
    )
