import pytest


@pytest.mark.viz
def test_video_generation(
    visualizer, raw_calcium_video, preprocessed_video, stabilized_video
):
    """Test video generation with visualizations."""

    visualizer.write_movie(
        raw_calcium_video, subdir="fixtures", name="raw_calcium_video"
    )
    visualizer.write_movie(
        preprocessed_video, subdir="fixtures", name="preprocessed_video"
    )
    visualizer.write_movie(stabilized_video, subdir="fixtures", name="stabilized_video")


@pytest.mark.viz
def test_plot_observable_fixtures(
    visualizer, footprints, traces, spikes, positions, radii
):
    """Test plotting of observable fixtures with visualizations."""

    # Show visualizations
    visualizer.plot_footprints(footprints, positions, radii, subdir="fixtures")
    visualizer.plot_traces(traces, spikes, indices=[0, 1, 2], subdir="fixtures")
    visualizer.plot_trace_stats(traces, indices=[0, 1, 2], subdir="fixtures")
    visualizer.plot_trace_pair_analysis(
        traces, comp1_idx=0, comp2_idx=1, subdir="fixtures"
    )
    visualizer.plot_component_clustering(traces, subdir="fixtures")
