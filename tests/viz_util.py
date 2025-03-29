from pathlib import Path
from typing import Optional, Tuple, Callable, List

import cv2
import matplotlib.pyplot as plt
import numpy as np


class Visualizer:
    """Utility class for visualization."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_fig(self, name: str, subdir: Optional[str] = None) -> None:
        """Save current figure to output directory."""
        save_dir = self.output_dir
        if subdir:
            save_dir = save_dir / subdir
            save_dir.mkdir(exist_ok=True)

        plt.savefig(save_dir / f"{name}.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_footprints(
        self,
        footprints,
        positions: np.ndarray | None = None,
        radii: np.ndarray | None = None,
        name: str = "footprints",
        title: Optional[str] = None,
        subdir: Optional[str] = None,
        highlight_indices: Optional[List[int]] = None,
    ) -> None:
        """Plot spatial footprints with flexible highlighting options."""
        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot composite image
        composite = footprints.sum(dim="component")
        im = ax.imshow(composite, cmap="viridis")

        if positions is not None and radii is not None:
            # Plot circles for all neurons
            for i, (pos, r) in enumerate(zip(positions, radii)):
                color = "y" if highlight_indices and i in highlight_indices else "r"
                alpha = 0.8 if highlight_indices and i in highlight_indices else 0.5
                circle = plt.Circle(pos[::-1], r, fill=False, color=color, alpha=alpha)
                ax.add_patch(circle)

        plt.colorbar(im)
        ax.set_title(title or f"Spatial Footprints (n={len(footprints)})")
        self.save_fig(name, subdir)

    def plot_traces(
        self,
        traces,
        spikes=None,
        indices: Optional[List[int]] = None,
        name: str = "traces",
        subdir: Optional[str] = None,
        additional_signals: Optional[List[Tuple[np.ndarray, dict]]] = None,
    ) -> None:
        """
        Plot calcium traces with optional spikes and additional signals.

        Parameters:
        -----------
        additional_signals : List of (signal, plot_kwargs) tuples
            Additional signals to plot on same axes as traces
        """
        if indices is None:
            indices = list(range(min(5, len(traces))))

        fig, axes = plt.subplots(len(indices), 1, figsize=(15, 3 * len(indices)))
        if len(indices) == 1:
            axes = [axes]

        for i, idx in enumerate(indices):
            ax = axes[i]
            # Plot main trace
            ax.plot(traces[idx], label="Calcium trace")

            # Plot spikes if provided
            if spikes is not None:
                spike_times = np.where(spikes[idx])[0]
                ax.vlines(
                    spike_times,
                    0,
                    traces[idx].max(),
                    color="r",
                    alpha=0.5,
                    label="Spikes",
                )

            # Plot additional signals if provided
            if additional_signals:
                for signal, kwargs in additional_signals:
                    ax.plot(signal[idx], **kwargs)

            ax.set_title(f"Neuron {idx}")
            ax.legend()

        plt.tight_layout()
        self.save_fig(name, subdir)

    def write_movie(self, video, filepath: str | Path):
        """Test visualization of stabilized calcium video to verify motion stabilization."""

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            self.output_dir / filepath,
            fourcc,
            24.0,
            (video.sizes["width"], video.sizes["height"]),
        )

        max_brightness = float(video.max())

        for frame in video:
            # If frame is float, convert/scaling to uint8:
            frame_8bit = (frame / max_brightness * 255).astype(np.uint8)

            # grayscale, so convert to BGR color:
            frame_8bit = cv2.cvtColor(frame_8bit.values, cv2.COLOR_GRAY2BGR)

            out.write(frame_8bit)

        out.release()

    def save_video_frames(
        self,
        video,
        name: str = "video",
        subdir: Optional[str] = None,
        frame_processor: Optional[Callable] = None,
    ) -> None:
        """
        Save video frames with optional processing function.

        Parameters:
        -----------
        frame_processor : Callable
            Function to process each frame before saving
            Should take (frame, index) as arguments
        """
        save_dir = self.output_dir
        if subdir:
            save_dir = save_dir / subdir
        save_dir = save_dir / name
        save_dir.mkdir(parents=True, exist_ok=True)

        vmin, vmax = np.percentile(video, [2, 98])

        for i, frame in enumerate(video):
            fig, ax = plt.subplots(figsize=(8, 8))

            # Apply frame processing if provided
            if frame_processor:
                frame = frame_processor(frame, i)

            ax.imshow(frame, cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_title(f"Frame {i}")
            plt.savefig(save_dir / f"frame_{i:04d}.png", dpi=150, bbox_inches="tight")
            plt.close()

        # Optionally create gif
        try:
            import imageio

            frames = []
            for i in range(len(video)):
                frames.append(imageio.imread(save_dir / f"frame_{i:04d}.png"))
            imageio.mimsave(save_dir / f"{name}.gif", frames, fps=30)
        except ImportError:
            print("imageio not installed - skipping gif creation")
