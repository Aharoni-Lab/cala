import numpy as np
import xarray as xr
from scipy.ndimage import binary_dilation, binary_erosion


def update_footprint(
    footprints: xr.DataArray,
    traces: xr.DataArray,
    pixel_stats: xr.DataArray,
    component_stats: xr.DataArray,
    mini_footprints: xr.DataArray,
    mini_denoised: xr.DataArray,
) -> xr.DataArray:
    """Helper to visualize iteration results."""

    # Run updater and plot results
    updater = Footprinter(boundary_expansion_pixels=1)
    updater.learn_one(
        footprints=footprints, pixel_stats=pixel_stats, component_stats=component_stats
    )
    new_footprints = updater.transform_one().transpose(*mini_footprints.dims)

    # Visualize movies
    preconstructed_movie = (footprints @ traces).transpose(*mini_denoised.dims)
    postconstructed_movie = (new_footprints @ traces).transpose(*mini_denoised.dims)
    residual = mini_denoised - postconstructed_movie

    return new_footprints


def get_stats(
    footprints: xr.DataArray, denoised: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Helper to compute traces and stats for modified footprints."""
    t_init = Tracer()
    traces = t_init.learn_one(footprints, denoised).transform_one()

    ps = PixelStats()
    pixel_stats = ps.learn_one(traces=traces, frame=denoised).transform_one()

    cs = ComponentStats()
    component_stats = cs.learn_one(traces=traces, frame=denoised).transform_one()

    return traces, pixel_stats, component_stats


def test_perfect_condition(
    mini_footprints: xr.DataArray,
    mini_traces: xr.DataArray,
    mini_denoised: xr.DataArray,
) -> None:
    ps = PixelStats()
    mini_pixel_stats = ps.learn_one(traces=mini_traces, frame=mini_denoised).transform_one()

    cs = CompStats()
    mini_component_stats = cs.learn_one(traces=mini_traces, frame=mini_denoised).transform_one()

    new_footprints = update_footprint(
        mini_footprints,
        mini_traces,
        mini_pixel_stats,
        mini_component_stats,
        mini_footprints,
        mini_denoised,
    )
    assert np.allclose(new_footprints, mini_footprints.transpose(*new_footprints.dims), atol=1e-3)


def test_imperfect_condition(mini_footprints: xr.DataArray, mini_denoised: xr.DataArray) -> None:
    # Add noise to stats
    traces, pixel_stats, component_stats = get_stats(mini_footprints, mini_denoised)
    noisy_pixel_stats = pixel_stats + 0.1 * np.random.rand(*pixel_stats.shape)
    noisy_component_stats = component_stats + 0.1 * np.random.rand(*component_stats.shape)

    update_footprint(
        mini_footprints,
        traces,
        noisy_pixel_stats,
        noisy_component_stats,
        mini_footprints,
        mini_denoised,
    )


def test_wrong_footprint(mini_footprints: xr.DataArray, mini_denoised: xr.DataArray) -> None:
    wrong_footprints = mini_footprints.copy()[:4]
    wrong_footprints[3] = mini_footprints[3] + mini_footprints[4]

    traces, pixel_stats, component_stats = get_stats(wrong_footprints, mini_denoised)

    update_footprint(
        wrong_footprints, traces, pixel_stats, component_stats, mini_footprints, mini_denoised
    )


def test_small_footprint(mini_footprints: xr.DataArray, mini_denoised: xr.DataArray) -> None:
    small_footprints = mini_footprints.copy()
    small_footprints[1] = binary_erosion(small_footprints[1])

    traces, pixel_stats, component_stats = get_stats(small_footprints, mini_denoised)

    update_footprint(
        small_footprints, traces, pixel_stats, component_stats, mini_footprints, mini_denoised
    )


def test_oversized_footprint(mini_footprints: xr.DataArray, mini_denoised: xr.DataArray) -> None:
    oversized_footprints = mini_footprints.copy()
    oversized_footprints[1] = binary_dilation(oversized_footprints[1])

    traces, pixel_stats, component_stats = get_stats(oversized_footprints, mini_denoised)

    update_footprint(
        oversized_footprints, traces, pixel_stats, component_stats, mini_footprints, mini_denoised
    )


def test_redundant_footprint(mini_footprints: xr.DataArray, mini_denoised: xr.DataArray) -> None:
    redundant_footprints = mini_footprints.copy()
    rolled = xr.DataArray(np.roll(mini_footprints[-1], -1), dims=("height", "width"))
    rolled = rolled.expand_dims("component").assign_coords(
        {"id_": ("component", ["id5"]), "type_": ("component", [Component.NEURON.value])}
    )
    redundant_footprints = xr.concat(
        [redundant_footprints, rolled],
        dim="component",
    )

    traces, pixel_stats, component_stats = get_stats(redundant_footprints, mini_denoised)

    update_footprint(
        redundant_footprints, traces, pixel_stats, component_stats, mini_footprints, mini_denoised
    )
