from typing import Any

import numpy as np
import pytest
import xarray as xr

from cala.nodes.iter.component_stats import ComponentStats
from cala.util.new import package_frame


def test_init(
    initializer,
    sample_data: dict[str, Any],
) -> None:
    """Test the correctness of the component correlation computation."""
    # Prepare data
    traces = sample_data["traces"]
    frames = sample_data["frames"]

    # Run computation
    initializer.learn_one(traces, frames)
    result = initializer.transform_one()

    # Manual computation for verification
    C = traces.values
    expected_M = C @ C.T / frames.shape[0]

    # Compare results
    assert np.allclose(result.values, expected_M)

    # Check specific correlation patterns from our constructed data
    assert np.allclose(result.values[0, 1], 0.5)  # Perfect correlation
    assert np.allclose(result.values[0, 2], -0.5)  # Perfect anti-correlation
    assert np.allclose(np.diag(result.values), 0.5)  # Self-correlation

    # Test symmetry
    assert np.allclose(result.values, result.values.T)

    # Test diagonal elements
    assert np.allclose(np.diag(result.values), 0.5)


def test_ingest_frame(
    updater,
    mini_footprints: xr.DataArray,
    mini_traces: xr.DataArray,
    prev_comp_stats: xr.DataArray,
    mini_denoised: xr.DataArray,
    initializer,
) -> None:

    updater.learn_one(
        frame=package_frame(mini_denoised[-1].values, len(mini_denoised) - 1),
        traces=mini_traces,
        component_stats=prev_comp_stats,
    )
    new_comp_stats = updater.transform_one()

    late_init_cs = initializer.learn_one(
        mini_traces,
        frame=mini_denoised,
    ).transform_one()

    assert np.allclose(late_init_cs, new_comp_stats)


def test_ingest_component(): ...
