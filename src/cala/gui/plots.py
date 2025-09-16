from pathlib import Path

import cv2
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr

sns.set_theme(style="whitegrid", context="notebook", font_scale=1.2, palette="deep")


def write_movie(video: xr.DataArray, path: str | Path) -> None:
    """Test visualization of stabilized calcium video to verify motion stabilization."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, 24.0, (video.sizes["width"], video.sizes["height"]))

    max_brightness = video.max().item()

    for frame in video:
        frame_8bit = (frame / max_brightness * 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame_8bit.values, cv2.COLOR_GRAY2BGR)
        out.write(frame_bgr)

    out.release()


def write_gif(
    videos: xr.DataArray | list[xr.DataArray],
    path: str | Path,
    n_cols: int | None = None,
) -> None:
    """
    Save video frames with optional processing function. Can handle single or multiple videos.

    Parameters:
    -----------
    videos : Union[xr.DataArray, List[Tuple[xr.DataArray, str]]]
        Either a single video DataArray or list of (video, title) tuples for comparison
    n_cols : Optional[int]
        Number of columns when displaying multiple videos. If None, tries to make square grid
    """
    # Handle single video case
    if isinstance(videos, xr.DataArray):
        videos = [videos]

    # Verify all videos have same number of frames
    n_frames = len(videos[0][0])
    if not all(len(video) == n_frames for video in videos):
        raise ValueError("All videos must have the same number of frames")

    n_videos = len(videos)
    if n_cols is None:
        n_cols = int(np.ceil(np.sqrt(n_videos))) if n_videos > 1 else 1
    n_rows = int(np.ceil(n_videos / n_cols))

    # Get global min/max for consistent scaling
    vmin = np.min([np.min(video) for video in videos])
    vmax = np.max([np.max(video) for video in videos])

    for frame_idx in range(n_frames):
        if n_videos == 1:
            fig, ax = plt.subplots(figsize=(8, 8))
            axes = [[ax]]
        else:
            fig, axes = plt.subplots(
                n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False
            )

        for vid_idx, (video, title) in enumerate(videos):
            last_row = vid_idx // n_cols
            remn_col = vid_idx % n_cols
            ax = axes[last_row][remn_col]

            frame = video[frame_idx]

            ax.imshow(frame, cmap="gray", vmin=vmin, vmax=vmax)
            if title:
                ax.set_title(f"{title}\nFrame {frame_idx}")
            else:
                ax.set_title(f"Frame {frame_idx}")
            ax.axis("off")

        # Hide empty subplots
        if n_videos > 1:
            for idx in range(n_videos, n_rows * n_cols):
                last_row = idx // n_cols
                remn_col = idx % n_cols
                axes[last_row][remn_col].set_visible(False)

        plt.tight_layout()
        plt.savefig(path / f"{frame_idx:04d}.png", dpi=150, bbox_inches="tight")

    # Create gif
    frames = []
    for i in range(n_frames):
        frames.append(imageio.imread(path / f"{i:04d}.png"))
    imageio.mimsave(path, frames, fps=30)
