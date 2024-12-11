from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np


class VisualHelper:
    @staticmethod
    def write_movie(test_movie_fixture, filepath: str | Path):
        """Test visualization of stabilized calcium video to verify motion correction."""
        video, ground_truth, metadata = test_movie_fixture

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            filepath,
            fourcc,
            24.0,
            (video.sizes["width"], video.sizes["height"]),
        )

        max_brightness = float(video.max())

        for t in video.frames:
            frame = video.sel(frames=t).values  # Extract a single frame
            # If frame is float, convert/scaling to uint8:
            frame_8bit = (frame / max_brightness * 255).astype(np.uint8)

            # grayscale, so convert to BGR color:
            frame_8bit = cv2.cvtColor(frame_8bit, cv2.COLOR_GRAY2BGR)

            out.write(frame_8bit)

        out.release()

    @staticmethod
    def test_visualize_calcium_traces(
        test_movie_fixture,
        filepath: str | Path,
        window_size: int = 5,
        n_traces: int = 5,
        title: Optional[str] = "Calcium Traces from Raw Video",
    ) -> None:
        """
        Visualize calcium traces at specified positions.

        Parameters
        ----------
        test_movie_fixture : pytest.fixture
            Video data
        window_size : int
            Size of window to average around each position
        n_traces : int
            Number of traces to show
        title : str, optional
            Title for the plot

        Returns
        -------
        matplotlib.figure.Figure
            Figure with the visualization
        """
        video, ground_truth, metadata = test_movie_fixture

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[1, 3])

        # Show max projection with selected positions
        max_proj = video.max(dim="frames")
        vmin, vmax = np.percentile(max_proj, [1, 99])
        ax1.imshow(max_proj, cmap="gray", vmin=vmin, vmax=vmax)

        # Randomly select positions if there are too many
        if len(ground_truth) > n_traces:
            selected_positions = ground_truth.sample(n=n_traces, random_state=42)
        else:
            selected_positions = ground_truth

        # Plot selected positions
        ax1.scatter(
            selected_positions["width"],
            selected_positions["height"],
            color="r",
            marker="x",
            s=100,
        )
        ax1.axis("off")
        ax1.set_title("Selected Positions")

        # Extract and plot traces
        times = np.arange(video.sizes["frames"])
        for i, (_, pos) in enumerate(selected_positions.iterrows()):
            y, x = int(pos["height"]), int(pos["width"])
            y_slice = slice(
                max(0, y - window_size), min(video.sizes["height"], y + window_size + 1)
            )
            x_slice = slice(
                max(0, x - window_size), min(video.sizes["width"], x + window_size + 1)
            )

            # Extract mean intensity in window
            trace = video.isel(height=y_slice, width=x_slice).mean(["height", "width"])

            # Plot trace
            ax2.plot(times, trace, label=f"Position {i+1}")

        ax2.set_xlabel("Frame")
        ax2.set_ylabel("Intensity")
        ax2.legend()
        ax2.grid(True)

        if title:
            fig.suptitle(title, fontsize=14)

        plt.tight_layout()

        fig.savefig(filepath)
        plt.close(fig)
