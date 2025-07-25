import numpy as np
import pytest
import sparse
import xarray as xr

from cala.gui.plots import Plotter
from cala.nodes.iter.overlap import Overlapper


def test_init(
    initializer: Overlapper,
    mini_footprints: xr.DataArray,
    plotter: Plotter,
) -> None:
    """Test the correctness of overlap detection."""
    plotter.plot_footprints(mini_footprints, subdir="init/overlap")

    initializer.learn_one(mini_footprints)
    result = initializer.transform_one()

    result.values = result.data.todense()
    plotter.plot_overlaps(result, mini_footprints, subdir="init/overlap")
    # Convert to dense for testing

    # Test expected overlap patterns
    assert result[0, 1] == 1  # Components 0 and 1 overlap
    assert result[1, 0] == 1  # Symmetric
    assert np.sum(result[2]) == 1  # Component 2 only overlaps with itself
    assert result[1, 4] == 1  # Components 3 and 4 overlap
    assert result[4, 1] == 1  # Components 3 and 4 overlap
    assert result[3, 4] == 1  # Components 3 and 4 overlap
    assert result[4, 3] == 1  # Symmetric

    assert np.allclose(dense_matrix, dense_matrix.T)
    assert np.allclose(np.diag(dense_matrix), 1)
