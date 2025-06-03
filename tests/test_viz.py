import numpy as np
import pytest
import xarray as xr

from cala.gui.plots import Plotter


@pytest.mark.viz
def test_video_generation(
    plotter: Plotter,
    raw_calcium_video: xr.DataArray,
    preprocessed_video: xr.DataArray,
    stabilized_video: xr.DataArray,
) -> None:
    """Test video generation with visualizations."""

    plotter.write_movie(raw_calcium_video, subdir="fixtures", name="raw_calcium_video")
    plotter.write_movie(preprocessed_video, subdir="fixtures", name="preprocessed_video")
    plotter.write_movie(stabilized_video, subdir="fixtures", name="stabilized_video")


@pytest.mark.viz
def test_plot_observable_fixtures(
    plotter: Plotter,
    footprints: xr.DataArray,
    traces: xr.DataArray,
    spikes: xr.DataArray,
    positions: np.ndarray,
    radii: np.ndarray,
) -> None:
    """Test plotting of observable fixtures with visualizations."""

    # Show visualizations
    plotter.plot_footprints(footprints, positions, radii, subdir="fixtures")
    plotter.plot_traces(traces, spikes, indices=[0, 1, 2], subdir="fixtures")
    plotter.plot_trace_stats(traces, indices=[0, 1, 2], subdir="fixtures")
    plotter.plot_trace_pair_analysis(traces, comp1_idx=0, comp2_idx=1, subdir="fixtures")
    plotter.plot_component_clustering(traces, subdir="fixtures")
