from collections.abc import Callable
from pathlib import Path

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr
from scipy.sparse.csgraph import connected_components
from skimage.measure import find_contours

from cala.streaming.core import Axis


class Visualizer:
    """Utility class for visualization."""

    def __init__(self, output_dir: Path | str):
        if isinstance(output_dir, str):
            try:
                self.output_dir = Path(output_dir)
            except ValueError as e:
                raise ValueError from e
        else:
            self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Enhanced style configuration
        self.style_config = {
            "style": "whitegrid",
            "context": "notebook",
            "font_scale": 1.2,
            "palette": "deep",
        }
        sns.set_theme(**self.style_config)

        # Define color palettes for different use cases
        self.colors = {
            "main": sns.color_palette("husl", n_colors=10),
            "sequential": sns.color_palette("rocket", n_colors=10),
            "diverging": sns.color_palette("vlag", n_colors=10),
            "categorical": sns.color_palette("Set2", n_colors=8),
        }

        # Define common plot settings
        self.plot_defaults = {
            "figure.figsize": (10, 6),
            "axes.titlesize": 14,
            "axes.labelsize": 12,
        }

    def save_fig(self, name: str, subdir: str | None = None) -> None:
        """Save current figure to output directory."""
        save_dir = self.output_dir
        if subdir:
            save_dir = save_dir / subdir
            save_dir.mkdir(parents=True, exist_ok=True)

        plt.savefig(save_dir / f"{name}.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_component_contours(
        self,
        ax: plt.Axes,
        component: np.ndarray,
        color: str = "w",
        label: str | None = None,
    ) -> None:
        """
        Helper method to plot contours of a component.

        Parameters
        ----------
        ax : plt.Axes
            Axes to plot on
        component : np.ndarray
            2D array of component footprint
        color : str
            Color for contour and label
        label : Optional[str]
            Label to add at component center (e.g., component number)
        """
        # Find contours at level 0 (boundary between zero and positive values)
        contours = find_contours(component, 0)

        # Draw each contour
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=1)

        # Add label at centroid of largest contour if requested
        if label and contours:
            largest_contour = max(contours, key=len)
            center_y = largest_contour[:, 0].mean()
            center_x = largest_contour[:, 1].mean()
            ax.text(
                center_x,
                center_y,
                label,
                color=color,
                ha="center",
                va="center",
                bbox=dict(facecolor="black", alpha=0.5, pad=1),
            )

    def plot_footprints(
        self,
        footprints: np.ndarray | xr.DataArray,
        positions: np.ndarray | None = None,
        radii: np.ndarray | None = None,
        name: str = "footprints",
        title: str | None = None,
        subdir: str | None = None,
        highlight_indices: list[int] | None = None,
    ) -> None:
        """Plot spatial footprints with flexible highlighting options."""
        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot composite image
        composite = footprints.sum(dim="component")
        sns.heatmap(composite, cmap="viridis", cbar_kws={"label": "Component Intensity"})

        # Draw circles if positions and radii provided
        if positions is not None and radii is not None:
            for i, (pos, r) in enumerate(zip(positions, radii)):
                color = (
                    self.colors["categorical"][1]
                    if highlight_indices and i in highlight_indices
                    else self.colors["categorical"][0]
                )
                alpha = 0.8 if highlight_indices and i in highlight_indices else 0.5
                circle = plt.Circle(pos[::-1], r, fill=False, color=color, alpha=alpha)
                ax.add_patch(circle)

        # Draw contours and labels for each component
        for idx, footprint in enumerate(footprints):
            color = "y" if highlight_indices and idx in highlight_indices else "w"
            self._plot_component_contours(ax, footprint.values, color=color, label=str(idx))

        ax.set_title(title or f"Spatial Footprints (n={len(footprints)})")
        self.save_fig(name, subdir)

    def plot_all_footprints(
        self,
        footprints: np.ndarray | xr.DataArray,
        title: str | None = None,
        subdir: str | None = None,
    ) -> None:
        for idx, fp in enumerate(footprints.transpose(Axis.component_axis, *Axis.spatial_axes)):
            fig, ax = plt.subplots(figsize=(10, 10))
            plt.imshow(fp)
            ax.set_title(
                title
                or f"Spatial Footprints ({idx + 1} of {footprints.sizes[Axis.component_axis]})"
            )
            self.save_fig(f"footprint_{idx}", subdir)

    def plot_pixel_stats(
        self,
        pixel_stats: xr.DataArray,
        footprints: xr.DataArray = None,
        name: str = "pixel_stats",
        subdir: str | None = None,
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

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)

        # Find global min/max for consistent colormap scaling
        vmin, vmax = pixel_stats.min(), pixel_stats.max()

        if footprints is not None:
            footprints = footprints.transpose(*pixel_stats.dims)

        for idx in range(n_components):
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row, col]

            # Plot correlation map for this component
            sns.heatmap(
                pixel_stats[idx],
                cmap="rocket",  # seaborn's improved heat colormap
                center=0,
                cbar_kws={"label": "Inner Product"},
                ax=ax,
                vmin=vmin,
                vmax=vmax,
                annot=n_components < 10,
            )

            if footprints is not None:
                # Add contour of the component's footprint
                self._plot_component_contours(
                    ax,
                    footprints[idx].values,
                    color="y",  # Yellow contours for contrast
                    label=None,  # Skip labels as we have titles
                )

            # Add component ID and type as title
            comp_id = pixel_stats.coords["id_"].values[idx]
            comp_type = pixel_stats.coords["type_"].values[idx]
            ax.set_title(f"{comp_id}\n({comp_type})")

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
        traces: np.ndarray | xr.DataArray,
        spikes: np.ndarray | xr.DataArray | None = None,
        indices: list[int] | None = None,
        name: str = "traces",
        subdir: str | None = None,
        additional_signals: list[tuple[np.ndarray, dict]] | None = None,
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

        sns.set_style("ticks")

        for i, idx in enumerate(indices):
            ax = axes[i]
            # Plot main trace
            sns.lineplot(data=traces[idx], ax=ax, label="Calcium trace")

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
            sns.despine(ax=ax)

        plt.tight_layout()
        self.save_fig(name, subdir)

    def write_movie(
        self, video: xr.DataArray, subdir: str | Path | None = None, name: str = "movie"
    ) -> None:
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
        videos: xr.DataArray | list[tuple[xr.DataArray, str]],
        name: str = "video",
        subdir: str | None = None,
        frame_processor: Callable | None = None,
        n_cols: int | None = None,
    ) -> None:
        """
        Save video frames with optional processing function. Can handle single or multiple videos.

        Parameters:
        -----------
        videos : Union[xr.DataArray, List[Tuple[xr.DataArray, str]]]
            Either a single video DataArray or list of (video, title) tuples for comparison
        name : str
            Base name for saved files
        subdir : Optional[str]
            Subdirectory within output directory
        frame_processor : Optional[Callable]
            Function to process each frame before saving
            Should take (frame, index) as arguments
        n_cols : Optional[int]
            Number of columns when displaying multiple videos. If None, tries to make square grid
        """
        save_dir = self.output_dir
        if subdir:
            save_dir = save_dir / subdir
        save_dir = save_dir / name
        save_dir.mkdir(parents=True, exist_ok=True)

        # Handle single video case
        if isinstance(videos, xr.DataArray):
            videos = [(videos, "")]

        n_videos = len(videos)
        if n_cols is None:
            n_cols = int(np.ceil(np.sqrt(n_videos))) if n_videos > 1 else 1
        n_rows = int(np.ceil(n_videos / n_cols))

        # Get global min/max for consistent scaling
        all_mins = [np.percentile(video, 2) for video, _ in videos]
        all_maxs = [np.percentile(video, 98) for video, _ in videos]
        vmin, vmax = min(all_mins), max(all_maxs)

        # Verify all videos have same number of frames
        n_frames = len(videos[0][0])
        if not all(len(video) == n_frames for video, _ in videos):
            raise ValueError("All videos must have the same number of frames")

        for frame_idx in range(n_frames):
            if n_videos == 1:
                fig, ax = plt.subplots(figsize=(8, 8))
                axes = [[ax]]
            else:
                fig, axes = plt.subplots(
                    n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False
                )

            # Plot each video frame
            for vid_idx, (video, title) in enumerate(videos):
                row = vid_idx // n_cols
                col = vid_idx % n_cols
                ax = axes[row][col]

                frame = video[frame_idx]

                # Apply frame processing if provided
                if frame_processor:
                    frame = frame_processor(frame, frame_idx)

                ax.imshow(frame, cmap="gray", vmin=vmin, vmax=vmax)
                if title:
                    ax.set_title(f"{title}\nFrame {frame_idx}")
                else:
                    ax.set_title(f"Frame {frame_idx}")
                ax.axis("off")

            # Hide empty subplots
            if n_videos > 1:
                for idx in range(n_videos, n_rows * n_cols):
                    row = idx // n_cols
                    col = idx % n_cols
                    axes[row][col].set_visible(False)

            plt.tight_layout()
            plt.savefig(save_dir / f"frame_{frame_idx:04d}.png", dpi=150, bbox_inches="tight")
            plt.close()

        # Create gif
        frames = []
        for i in range(n_frames):
            frames.append(imageio.imread(save_dir / f"frame_{i:04d}.png"))
        imageio.mimsave(save_dir / f"{name}.gif", frames, fps=30)

    def plot_trace_correlations(
        self,
        traces: xr.DataArray,
        name: str = "trace_correlations",
        subdir: str | None = None,
    ) -> None:
        """
        Create pairplot of trace correlations between components.

        Parameters
        ----------
        traces : xr.DataArray
            DataArray with dims (component, frame) containing component traces
        """
        # Convert to pandas DataFrame for seaborn
        df = traces.to_pandas().T  # Transpose to get components as columns

        # Use component IDs as column names if available
        if "id_" in traces.coords:
            df.columns = traces.coords["id_"].values

        # Create pairplot
        g = sns.pairplot(
            df,
            diag_kind="kde",  # Kernel density plots on diagonal
            plot_kws={"alpha": 0.6},
        )

        # Add title
        g.fig.suptitle("Component Trace Correlations", y=1.02)

        # Save figure
        self.save_fig(name, subdir)

    def plot_component_stats(
        self,
        component_stats: xr.DataArray,
        name: str = "component_stats",
        subdir: str | None = None,
        cmap: str = "RdBu_r",
    ) -> None:
        """
        Create heatmap of component correlation statistics.

        Parameters
        ----------
        component_stats : xr.DataArray
            DataArray with dims (component, component') containing correlation matrix
        cmap : str
            Colormap to use for heatmap
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Get component IDs and types for labels
        comp_ids = component_stats.coords["id_"].values
        comp_types = component_stats.coords["type_"].values
        labels = [f"{id_}\n({type_})" for id_, type_ in zip(comp_ids, comp_types)]

        # Create heatmap
        sns.heatmap(
            component_stats.values,
            ax=ax,
            cmap=cmap,
            center=0,  # Center colormap at 0 for correlation matrix
            # vmin=-1,
            # vmax=1,
            square=True,  # Make cells square
            xticklabels=labels,
            yticklabels=labels,
            annot=True,  # Show correlation values
            fmt=".2f",  # Format for correlation values
            cbar_kws={"label": "Correlation"},
        )

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        # Add title
        plt.title("Component Statistics Matrix")

        # Adjust layout to prevent label cutoff
        plt.tight_layout()

        # Save figure
        self.save_fig(name, subdir)

    def plot_trace_pair_analysis(
        self,
        traces: xr.DataArray,
        comp1_idx: int,
        comp2_idx: int,
        name: str = "trace_pair_analysis",
        subdir: str | None = None,
    ) -> None:
        """
        Create detailed analysis of two component traces using JointGrid.

        Parameters
        ----------
        traces : xr.DataArray
            DataArray with dims (component, frame) containing component traces
        comp1_idx, comp2_idx : int
            Indices of components to compare
        """
        # Extract the two traces
        trace1 = traces[comp1_idx]
        trace2 = traces[comp2_idx]

        # Create JointGrid
        g = sns.JointGrid(data=None, x=trace1, y=trace2)

        # Add scatter plot with hexbin
        g.plot_joint(sns.lineplot, alpha=0.6, color=self.colors["main"][0], markers=True)

        # Add marginal distributions
        g.plot_marginals(sns.histplot, kde=True)

        # Add correlation coefficient
        corr = np.corrcoef(trace1, trace2)[0, 1]
        g.figure.suptitle(f"Correlation: {corr:.3f}", y=1.02)

        # Get component IDs if available, otherwise use indices
        comp1_label = (
            f"Component {traces.coords['id_'].values[comp1_idx]}"
            if "id_" in traces.coords
            else f"Component {comp1_idx}"
        )
        comp2_label = (
            f"Component {traces.coords['id_'].values[comp2_idx]}"
            if "id_" in traces.coords
            else f"Component {comp2_idx}"
        )

        g.ax_joint.set_xlabel(f"{comp1_label} Intensity")
        g.ax_joint.set_ylabel(f"{comp2_label} Intensity")

        # Save figure
        self.save_fig(name, subdir)

    def plot_trace_stats(
        self,
        traces: xr.DataArray,
        indices: list[int] | None = None,
        name: str = "trace_stats",
        subdir: str | None = None,
    ) -> None:
        """
        Enhanced trace visualization with statistical features.
        """
        if indices is None:
            indices = list(range(min(5, len(traces))))

        # Create figure with subplots
        fig = plt.figure(figsize=(15, 4 * len(indices)))
        gs = fig.add_gridspec(len(indices), 2, width_ratios=[3, 1])

        for i, idx in enumerate(indices):
            # Time series plot
            ax_time = fig.add_subplot(gs[i, 0])
            sns.lineplot(
                data=traces[idx],
                ax=ax_time,
                color=self.colors["main"][i % len(self.colors["main"])],
                label=f"Component {idx}",
            )

            # Distribution plot
            ax_dist = fig.add_subplot(gs[i, 1])
            sns.histplot(
                data=traces[idx],
                ax=ax_dist,
                kde=True,
                color=self.colors["main"][i % len(self.colors["main"])],
            )

            # Add statistical annotations
            stats_text = (
                f"μ = {traces[idx].mean():.2f}\n"
                f"σ = {traces[idx].std():.2f}\n"
                f"max = {traces[idx].max():.2f}"
            )
            ax_dist.text(
                0.95,
                0.95,
                stats_text,
                transform=ax_dist.transAxes,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(facecolor="white", alpha=0.8),
            )

            sns.despine(ax=ax_time)
            sns.despine(ax=ax_dist)

        plt.tight_layout()
        self.save_fig(name, subdir)

    def plot_component_clustering(
        self,
        traces: xr.DataArray,
        name: str = "component_clustering",
        subdir: str | None = None,
    ) -> None:
        """
        Create clustering visualization of components based on trace similarity.
        """
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(traces)

        # Create clustermap
        g = sns.clustermap(
            corr_matrix,
            cmap=self.colors["diverging"],
            center=0,
            figsize=(12, 12),
            dendrogram_ratio=0.1,
            cbar_pos=(0.02, 0.8, 0.03, 0.2),
            cbar_kws={"label": "Correlation"},
            annot=len(traces) < 10,
            fmt=".2f",
        )

        # Add title
        g.figure.suptitle("Component Clustering by Trace Similarity", y=1.02)

        # Save figure
        self.save_fig(name, subdir)

    def plot_overlaps(
        self,
        overlap_matrix: np.ndarray,
        footprints: xr.DataArray,
        name: str = "component_overlap",
        subdir: str | None = None,
    ) -> None:
        """
        Plot footprints with overlapping components highlighted by group.

        Parameters
        ----------
        overlap_matrix : np.ndarray
            Square binary matrix where (i,j) = 1 if components i and j overlap
        footprints : xr.DataArray
            DataArray with dims (component, height, width) showing spatial footprints
        name : str
            Name for the output file
        subdir : Optional[str]
            Subdirectory for saving the figure
        """
        # Find connected components (groups of overlapping neurons)
        n_groups, labels = connected_components(overlap_matrix, directed=False)

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

        # Plot 1: Overlap matrix with group boundaries
        sns.heatmap(
            overlap_matrix,
            ax=ax1,
            cmap="crest",
            cbar_kws={"label": "Overlap"},
            square=True,
            xticklabels=True,
            yticklabels=True,
        )

        ax1.set_title("Overlap Matrix")

        # Plot 2: Spatial footprints with overlapping groups colored
        composite = footprints.max(dim="component")
        ax2.grid(False)
        im = ax2.imshow(composite, cmap="flare")
        plt.colorbar(im, ax=ax2, label="Component Intensity")

        # Plot contours for each group
        for group_idx in range(n_groups):
            group_mask = labels == group_idx
            if sum(group_mask) > 1:  # Only mark groups with multiple components
                color = self.colors["categorical"][group_idx % len(self.colors["categorical"])]
                group_indices = np.where(group_mask)[0]

                # Plot contours for all components in this group
                for comp_idx in group_indices:
                    self._plot_component_contours(
                        ax2,
                        footprints[comp_idx].values,
                        color=color,
                        label=str(comp_idx),
                    )

        # Plot contours for non-overlapping components in white
        for comp_idx in range(len(footprints)):
            if sum(overlap_matrix[comp_idx]) <= 1:  # Only self-overlap
                self._plot_component_contours(
                    ax2, footprints[comp_idx].values, color="w", label=str(comp_idx)
                )

        ax2.set_title("Spatial Footprints with Overlap Groups")

        # Add overall title
        fig.suptitle("Component Overlap Analysis", y=1.02, fontsize=14)

        plt.tight_layout()
        self.save_fig(name, subdir)

    def plot_comparison(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        name: str = "comparison",
        subdir: str | None = None,
        titles: tuple[str, str] = ("Label", "Prediction"),
    ) -> None:
        """
        Plot two images side by side with their difference.

        Parameters
        ----------
        image1, image2 : np.ndarray
            Images to compare
        name : str
            Name for the saved figure
        subdir : Optional[str]
            Subdirectory for saving the figure
        titles : Tuple[str, str]
            Titles for the two input images
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        # Plot first image
        im1 = ax1.imshow(image1)
        ax1.set_title(titles[0])
        plt.colorbar(im1, ax=ax1)

        # Plot second image
        im2 = ax2.imshow(image2)
        ax2.set_title(titles[1])
        plt.colorbar(im2, ax=ax2)

        # Plot difference
        diff = image1 - image2
        im3 = ax3.imshow(diff, cmap="RdBu_r")
        ax3.set_title("Difference")
        plt.colorbar(im3, ax=ax3)

        # Turn off axes for cleaner look
        for ax in (ax1, ax2, ax3):
            ax.axis("off")

        plt.tight_layout()
        self.save_fig(name, subdir)
