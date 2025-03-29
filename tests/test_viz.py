import pytest


@pytest.mark.viz
def test_video_generation(
    visualizer, raw_calcium_video, preprocessed_video, stabilized_video
):
    """Test video generation with visualizations."""

    visualizer.write_movie(raw_calcium_video, "raw_calcium_video.mp4")
    visualizer.write_movie(preprocessed_video, "preprocessed_video.mp4")
    visualizer.write_movie(stabilized_video, "stabilized_video.mp4")


@pytest.mark.viz
def test_plot_observable_fixtures(visualizer, footprints, traces, spikes):
    """Test plotting of observable fixtures with visualizations."""
    footprints_xr, positions, radii = footprints

    # Show visualizations
    visualizer.plot_footprints(footprints_xr, positions, radii)
    visualizer.plot_traces(traces, spikes, indices=[0, 1, 2])
    visualizer.plot_trace_stats(traces, indices=[0, 1, 2])
    visualizer.plot_trace_pair_analysis(traces, comp1_idx=0, comp2_idx=1)
    visualizer.plot_component_clustering(traces)
