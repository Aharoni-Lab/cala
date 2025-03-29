from pathlib import Path
from typing import Optional, Tuple, Callable, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr
from skimage.measure import find_contours


class Visualizer:
    """Utility class for visualization."""

    def __init__(self, output_dir: Path):
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

    def save_fig(self, name: str, subdir: Optional[str] = None) -> None:
        """Save current figure to output directory."""
        save_dir = self.output_dir
        if subdir:
            save_dir = save_dir / subdir
            save_dir.mkdir(exist_ok=True)

        plt.savefig(save_dir / f"{name}.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_component_contours(
        self,
        ax: plt.Axes,
        component: np.ndarray,
        color: str = "w",
        label: Optional[str] = None,
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
        sns.heatmap(
            composite, cmap="viridis", cbar_kws={"label": "Component Intensity"}
        )

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
            self._plot_component_contours(
                ax, footprint.values, color=color, label=str(idx)
            )

        ax.set_title(title or f"Spatial Footprints (n={len(footprints)})")
        self.save_fig(name, subdir)

    def plot_pixel_stats(
        self,
        pixel_stats: xr.DataArray,
        footprints: xr.DataArray = None,
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
                cbar_kws={"label": "Correlation"},
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

    def plot_trace_correlations(
        self,
        traces: xr.DataArray,
        name: str = "trace_correlations",
        subdir: Optional[str] = None,
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
        subdir: Optional[str] = None,
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
        subdir: Optional[str] = None,
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
        g.plot_joint(
            sns.lineplot, alpha=0.6, color=self.colors["main"][0], markers=True
        )

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
        indices: Optional[List[int]] = None,
        name: str = "trace_stats",
        subdir: Optional[str] = None,
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
        subdir: Optional[str] = None,
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
        g.fig.suptitle("Component Clustering by Trace Similarity", y=1.02)

        # Save figure
        self.save_fig(name, subdir)
