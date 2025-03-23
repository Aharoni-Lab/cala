from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


class VisualHelper:
    @staticmethod
    def write_movie(test_movie_fixture, filepath: str | Path):
        """Test visualization of stabilized calcium video to verify motion stabilization."""
        video = test_movie_fixture

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
    def visualize_traces(
        traces: xr.DataArray,
        spikes: xr.DataArray,
        filepath: str | Path,
        n_traces: int = 5,
        title: Optional[str] = "Calcium Traces and Spikes",
    ) -> None:
        """
        Visualize calcium traces with spike times.

        Parameters
        ----------
        traces : xarray.DataArray
            Calcium traces for each component
        spikes : xarray.DataArray
            Binary spike times for each component
        n_traces : int
            Number of traces to show
        title : str, optional
            Title for the plot
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        # Randomly select components if there are too many
        n_components = traces.sizes["components"]
        selected_idx = np.random.choice(
            n_components, min(n_traces, n_components), replace=False
        )

        # Plot traces and spikes
        times = np.arange(traces.sizes["frames"])
        for i, idx in enumerate(selected_idx):
            trace = traces[idx]
            spike_times = times[spikes[idx].values.astype(bool)]

            # Plot trace
            ax.plot(times, trace + i, label=f"Component {idx}")

            # Plot spike times as red vertical lines
            ax.vlines(spike_times, i, i + 0.5, color="red", alpha=0.5)

        ax.set_xlabel("Frame")
        ax.set_ylabel("Intensity (offset for visibility)")
        ax.legend()
        ax.grid(True)

        if title:
            fig.suptitle(title, fontsize=14)

        plt.tight_layout()
        fig.savefig(filepath)
        plt.close(fig)
