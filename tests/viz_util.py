from pathlib import Path
from typing import Optional, Tuple, Callable, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import find_contours
import xarray as xr


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

        # Draw circles if positions and radii provided
        if positions is not None and radii is not None:
            for i, (pos, r) in enumerate(zip(positions, radii)):
                color = "y" if highlight_indices and i in highlight_indices else "r"
                alpha = 0.8 if highlight_indices and i in highlight_indices else 0.5
                circle = plt.Circle(pos[::-1], r, fill=False, color=color, alpha=alpha)
                ax.add_patch(circle)

        # Draw contours and labels for each component
        for i, component in enumerate(footprints):
            # Find contours at level 0 (boundary between zero and positive values)
            contours = find_contours(component, 0)

            # Draw each contour
            color = "y" if highlight_indices and i in highlight_indices else "w"
            for contour in contours:
                ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=1)

            # Add component number at centroid of largest contour
            if contours:
                # Use largest contour for label placement
                largest_contour = max(contours, key=len)
                center_y = largest_contour[:, 0].mean()
                center_x = largest_contour[:, 1].mean()
                ax.text(
                    center_x,
                    center_y,
                    str(i),
                    color="w",
                    ha="center",
                    va="center",
                    bbox=dict(facecolor="black", alpha=0.5, pad=1),
                )

        plt.colorbar(im)
        ax.set_title(title or f"Spatial Footprints (n={len(footprints)})")
        self.save_fig(name, subdir)

    def plot_pixel_stats(
        self,
        pixel_stats: xr.DataArray,
        footprints: xr.DataArray,
        name: str = "pixel_stats",
        subdir: Optional[str] = None,
        n_cols: int = 4,
    ) -> None:
        """
        Plot correlation maps between components and pixels.

        Parameters
        ----------
        pixel_stats : xr.DataArray
            DataArray with dims (components, height, width) showing correlation
            between each component's trace and each pixel's intensity
        footprints : xr.DataArray
            DataArray with dims (components, height, width) showing the spatial
            footprints of each component
        name : str
            Name for the saved figure
        subdir : Optional[str]
            Subdirectory within viz_outputs to save the figure
        n_cols : int
            Number of columns in the subplot grid
        """
        n_components = len(pixel_stats)
        n_rows = (n_components + n_cols - 1) // n_cols  # Ceiling division

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False
        )

        # Find global min/max for consistent colormap scaling
        vmin, vmax = pixel_stats.min(), pixel_stats.max()

        for idx in range(n_components):
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row, col]

            # Plot correlation map for this component
            im = ax.imshow(
                pixel_stats[idx],
                cmap="RdBu_r",  # Red-Blue diverging colormap
                vmin=vmin,
                vmax=vmax,
            )

            # Add component ID and type as title
            comp_id = pixel_stats.coords["id_"].values[idx]
            comp_type = pixel_stats.coords["type_"].values[idx]
            ax.set_title(f"{comp_id}\n({comp_type})")

            # Add colorbar
            plt.colorbar(im, ax=ax)

            # Remove ticks for cleaner look
            ax.set_xticks([])
            ax.set_yticks([])

        # Hide empty subplots
        for idx in range(n_components, n_rows * n_cols):
            row, col = idx // n_cols, idx % n_cols
            axes[row, col].set_visible(False)

        # Add overall title
        fig.suptitle("Component-Pixel Correlation Maps", fontsize=16, y=1.02)

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Save figure
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

    def write_movie(self, video, subdir: str | Path | None = None, name: str = "movie"):
        """Test visualization of stabilized calcium video to verify motion stabilization."""
        save_dir = self.output_dir
        if subdir:
            save_dir = save_dir / subdir
            save_dir.mkdir(exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            save_dir / f"{name}.mp4",
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
