from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.ndimage import gaussian_filter


@dataclass
class CalciumVideoParams:
    """Parameters for synthetic calcium imaging video generation."""

    # Video dimensions
    frames: int = 100
    height: int = 512
    width: int = 512

    # Noise and background
    noise_level: float = 0.05
    baseline: float = 0.2
    drift_magnitude: float = 0.1

    # Neuron properties
    num_neurons: int = 50
    neuron_size_range: Tuple[int, int] = (8, 15)  # min/max radius in pixels
    neuron_shape_irregularity: float = 0.2  # 0=perfect circles, 1=very irregular
    margin: int = 50  # border margin for neuron placement

    # Calcium dynamics
    decay_time_range: Tuple[float, float] = (10, 20)  # frames
    firing_rate_range: Tuple[float, float] = (0.05, 0.15)  # probability per frame
    amplitude_range: Tuple[float, float] = (0.5, 1.0)

    # Motion
    motion_amplitude: Tuple[float, float] = (2, 2)  # pixels in y, x
    motion_frequency: float = 1  # cycles per video

    # Optical properties
    blur_sigma: float = 1.0

    # Artifacts
    photobleaching_decay: float = 0.3  # exponential decay rate
    dead_pixel_fraction: float = 0  # 0.001  # fraction of pixels that are dead
    hot_pixel_fraction: float = 0  # 0.0005  # fraction of pixels that are hot
    hot_pixel_intensity: float = 2.0  # intensity multiplier for hot pixels
    glow_intensity: float = 0.3  # intensity of the broad glow artifact
    glow_sigma: float = 0.5  # relative spread of glow (as fraction of width)


@pytest.fixture
def params():
    """Return default parameters for video generation."""
    return CalciumVideoParams()


@pytest.fixture
def raw_calcium_video(params: CalciumVideoParams):
    """Generate synthetic calcium imaging data that mimics real neuronal activity."""
    video = np.random.normal(
        params.baseline,
        params.noise_level,
        (params.frames, params.height, params.width),
    )

    # Add baseline drift
    t = np.linspace(0, 4 * np.pi, params.frames)
    drift = params.drift_magnitude * np.sin(t)
    video += drift[:, np.newaxis, np.newaxis]

    # Generate random neuron positions
    neuron_positions = np.random.randint(
        params.margin,
        min(params.height, params.width) - params.margin,
        size=(params.num_neurons, 2),
    )

    # Generate neuron properties
    radii = np.random.uniform(*params.neuron_size_range, params.num_neurons)
    decay_time = np.random.uniform(*params.decay_time_range, params.num_neurons)
    firing_rate = np.random.uniform(*params.firing_rate_range, params.num_neurons)
    amplitude = np.random.uniform(*params.amplitude_range, params.num_neurons)

    ground_truth = pd.DataFrame(
        {
            "height": neuron_positions[:, 0],
            "width": neuron_positions[:, 1],
            "radius": radii,
            "decay_time": decay_time,
            "firing_rate": firing_rate,
            "amplitude": amplitude,
        }
    )

    # Add calcium dynamics
    calcium_traces = np.zeros((params.num_neurons, params.frames))
    spatial_profiles = []

    for n in range(params.num_neurons):
        # Generate spike times
        spikes = np.random.random(params.frames) < firing_rate[n]

        # Create calcium trace
        for f in range(params.frames):
            if spikes[f]:
                calcium_traces[n, f:] += amplitude[n] * np.exp(
                    -(np.arange(params.frames - f)) / decay_time[n]
                )

        # Create irregular spatial profile
        spatial_profiles.append(
            create_irregular_neuron(int(radii[n]), params.neuron_shape_irregularity)
        )

    # Add neurons to video with motion
    for f in range(params.frames):
        motion_y = int(
            params.motion_amplitude[0]
            * np.sin(2 * np.pi * f / (params.frames / params.motion_frequency))
        )
        motion_x = int(
            params.motion_amplitude[1]
            * np.cos(2 * np.pi * f / (params.frames / params.motion_frequency))
        )

        for n in range(params.num_neurons):
            radius = int(radii[n])
            y_pos = neuron_positions[n, 0] + motion_y
            x_pos = neuron_positions[n, 1] + motion_x

            y_slice = slice(y_pos - radius, y_pos + radius + 1)
            x_slice = slice(x_pos - radius, x_pos + radius + 1)

            if (
                0 <= y_pos - radius
                and y_pos + radius + 1 <= params.height
                and 0 <= x_pos - radius
                and x_pos + radius + 1 <= params.width
            ):
                video[f, y_slice, x_slice] += calcium_traces[n, f] * spatial_profiles[n]

    # Apply Gaussian blur
    for f in range(params.frames):
        video[f] = gaussian_filter(video[f], sigma=params.blur_sigma)

    # Add artifacts
    video = add_artifacts(video, params)

    video_xr = xr.DataArray(
        video,
        dims=["frames", "height", "width"],
        coords={
            "frames": np.arange(params.frames),
            "height": np.arange(params.height),
            "width": np.arange(params.width),
        },
    )

    # Additional metadata
    metadata = {
        "calcium_traces": calcium_traces,
        "spatial_profiles": spatial_profiles,
        "motion": {
            "y": [
                int(
                    params.motion_amplitude[0]
                    * np.sin(2 * np.pi * f / (params.frames / params.motion_frequency))
                )
                for f in range(params.frames)
            ],
            "x": [
                int(
                    params.motion_amplitude[1]
                    * np.cos(2 * np.pi * f / (params.frames / params.motion_frequency))
                )
                for f in range(params.frames)
            ],
        },
    }

    return video_xr, ground_truth, metadata


@pytest.fixture
def preprocessed_video(raw_calcium_video, params: CalciumVideoParams):
    """Calcium imaging video with artifacts removed except photobleaching."""
    video, ground_truth, metadata = raw_calcium_video
    frames, height, width = video.shape

    clean = np.zeros(
        (params.frames, params.height, params.width),
    )

    # Add neurons with their calcium dynamics
    calcium_traces = metadata["calcium_traces"]
    spatial_profiles = metadata["spatial_profiles"]
    motion = metadata["motion"]

    # Add neurons to video with motion
    for f in range(frames):
        motion_y = motion["y"][f]
        motion_x = motion["x"][f]

        for n in range(params.num_neurons):
            y_pos = ground_truth["height"].iloc[n] + motion_y
            x_pos = ground_truth["width"].iloc[n] + motion_x
            radius = int(ground_truth["radius"].iloc[n])

            y_slice = slice(y_pos - radius, y_pos + radius + 1)
            x_slice = slice(x_pos - radius, x_pos + radius + 1)

            if (
                0 <= y_pos - radius
                and y_pos + radius + 1 <= params.height
                and 0 <= x_pos - radius
                and x_pos + radius + 1 <= params.width
            ):
                clean[f, y_slice, x_slice] += calcium_traces[n, f] * spatial_profiles[n]

    # Apply Gaussian blur
    for f in range(frames):
        clean[f] = gaussian_filter(clean[f], sigma=params.blur_sigma)

    # Keep photobleaching artifact
    if params.photobleaching_decay > 0:
        decay = np.exp(-np.arange(frames) * params.photobleaching_decay / frames)
        clean *= decay[:, np.newaxis, np.newaxis]

    clean_xr = xr.DataArray(clean, dims=video.dims, coords=video.coords)

    return clean_xr, ground_truth, metadata


@pytest.fixture
def stabilized_video(preprocessed_video, params: CalciumVideoParams):
    """Motion-corrected calcium imaging video."""
    video, ground_truth, metadata = preprocessed_video

    # remove the artificial motion
    stabilized = np.zeros_like(video)

    for f in range(params.frames):
        motion_y = -metadata["motion"]["y"][f]
        motion_x = -metadata["motion"]["x"][f]

        if motion_y >= 0:
            y_src = slice(None, -motion_y) if motion_y else slice(None)
            y_dst = slice(motion_y, None) if motion_y else slice(None)
        else:
            y_src = slice(-motion_y, None)
            y_dst = slice(None, motion_y)

        if motion_x >= 0:
            x_src = slice(None, -motion_x) if motion_x else slice(None)
            x_dst = slice(motion_x, None) if motion_x else slice(None)
        else:
            x_src = slice(-motion_x, None)
            x_dst = slice(None, motion_x)

        stabilized[f, y_dst, x_dst] = video[f, y_src, x_src]

    # Convert to xarray with same coordinates
    stabilized_xr = xr.DataArray(stabilized, dims=video.dims, coords=video.coords)

    return stabilized_xr, ground_truth, metadata


def create_irregular_neuron(radius: int, irregularity: float) -> np.ndarray:
    """Create an irregular neuron shape using random perturbations of a circle."""
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]

    # Convert to polar
    angles = np.arctan2(y, x)
    distances = np.sqrt(x * x + y * y)

    # Generate random perturbations around the circle
    num_perturbations = 8
    perturbation_angles = np.linspace(0, 2 * np.pi, num_perturbations, endpoint=False)
    perturbation_magnitudes = 1 + irregularity * np.random.uniform(
        -1, 1, num_perturbations
    )

    # Interpolate perturbations for all angles
    from scipy.interpolate import interp1d

    perturbation_func = interp1d(
        np.concatenate([perturbation_angles, [2 * np.pi]]),
        np.concatenate([perturbation_magnitudes, [perturbation_magnitudes[0]]]),
        kind="cubic",
    )

    # Apply perturbations
    perturbed_radius = perturbation_func(np.mod(angles, 2 * np.pi)) * radius
    mask = distances <= perturbed_radius

    # Create gaussian profile within the mask
    profile = np.exp(-(distances[mask] ** 2) / (2 * (radius / 2) ** 2))
    result = np.zeros_like(distances)
    result[mask] = profile

    return result


def add_artifacts(video: np.ndarray, params: CalciumVideoParams) -> np.ndarray:
    """Add realistic microscopy artifacts to the video."""
    frames, height, width = video.shape

    # Add broad glow artifact
    y, x = np.ogrid[0:height, 0:width]
    glow_center_y = height // 2 + np.sin(np.linspace(0, 2 * np.pi, frames)) * (
        height // 4
    )
    glow_center_x = width // 2 + np.cos(np.linspace(0, 2 * np.pi, frames)) * (
        width // 4
    )

    for f in range(frames):
        # Create broad gaussian glow
        dist_sq = (y - glow_center_y[f]) ** 2 + (x - glow_center_x[f]) ** 2
        glow = params.glow_intensity * np.exp(
            -dist_sq / (2 * (width * params.glow_sigma) ** 2)
        )
        # Add slight temporal variation to glow intensity
        glow *= 1 + 0.2 * np.sin(2 * np.pi * f / (frames / 2))
        video[f] += glow

    # Photobleaching
    if params.photobleaching_decay > 0:
        decay = np.exp(-np.arange(frames) * params.photobleaching_decay / frames)
        video *= decay[:, np.newaxis, np.newaxis]

    # Dead pixels (always black)
    num_dead = int(height * width * params.dead_pixel_fraction)
    dead_y = np.random.randint(0, height, num_dead)
    dead_x = np.random.randint(0, width, num_dead)
    video[:, dead_y, dead_x] = 0

    # Hot pixels (always bright)
    num_hot = int(height * width * params.hot_pixel_fraction)
    hot_y = np.random.randint(0, height, num_hot)
    hot_x = np.random.randint(0, width, num_hot)
    hot_values = params.hot_pixel_intensity * (1 + 0.2 * np.random.randn(num_hot))
    video[:, hot_y, hot_x] = hot_values[np.newaxis, :]

    return video
