from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
import pytest
import xarray as xr
from scipy.ndimage import gaussian_filter
from scipy.stats.qmc import PoissonDisk

from cala.streaming.core import Component


@dataclass
class CalciumVideoParams:
    """Parameters for synthetic calcium imaging video generation."""

    # Video dimensions
    frames: int = 100
    height: int = 512
    width: int = 512

    # Noise and background
    noise_level: float = 0.5
    baseline: float = 0.1
    drift_magnitude: float = 0.1

    # Neuron properties
    num_neurons: int = 50
    neuron_size_range: tuple[int, int] = (40, 60)  # min/max radius in pixels
    neuron_shape_irregularity: float = 1.8  # 0=perfect circles, higher = more irregular

    # Calcium dynamics
    decay_time_range: tuple[float, float] = (10, 20)  # frames
    firing_rate_range: tuple[float, float] = (0.05, 0.15)  # probability per frame
    amplitude_range: tuple[float, float] = (0.5, 1.0)

    # Motion
    motion_amplitude: tuple[float, float] = (7, 7)  # pixels in y, x
    motion_frequency: float = 1  # cycles per video

    # Optical properties
    blur_sigma: float = 1.0

    # Artifacts
    photobleaching_decay: float = 0.3  # exponential decay rate
    dead_pixel_fraction: float = 0.001  # fraction of pixels that are dead
    hot_pixel_fraction: float = 0.0005  # fraction of pixels that are hot
    hot_pixel_intensity: float = 2.0  # intensity multiplier for hot pixels
    glow_intensity: float = 0.3  # intensity of the broad glow artifact
    glow_sigma: float = 0.5  # relative spread of glow (as fraction of width)


@pytest.fixture(scope="session")
def params() -> CalciumVideoParams:
    """Return default parameters for video generation."""
    return CalciumVideoParams()


@pytest.fixture(scope="session")
def ids(params: CalciumVideoParams) -> list[str]:
    return [f"comp_{i}" for i in range(params.num_neurons)]


@pytest.fixture(scope="session")
def types(params: CalciumVideoParams) -> list[str]:
    return [Component.NEURON.value] * params.num_neurons


@pytest.fixture(scope="session")
def radii(params: CalciumVideoParams) -> np.ndarray:
    return np.random.uniform(*params.neuron_size_range, params.num_neurons)


@pytest.fixture(scope="session")
def frame_coords(params: CalciumVideoParams) -> dict:
    return {
        "frame_": ("frame", [i for i in range(params.frames)]),
        "time_": ("frame", [f"t_{i}" for i in range(params.frames)]),
    }


@pytest.fixture(scope="session")
def positions(params: CalciumVideoParams, radii: np.ndarray) -> np.ndarray:
    max_radius = np.max(radii)

    # Calculate the available space after accounting for margins
    available_height = params.height - 2 * max_radius
    available_width = params.width - 2 * max_radius

    # Create sampler for the available space
    sampler = PoissonDisk(d=2, radius=max_radius / (2 * max(available_height, available_width)))

    # Sample points and scale them to our available dimensions
    points = sampler.random(params.num_neurons)
    positions = np.zeros_like(points).astype(int)
    positions[:, 0] = np.round(points[:, 0] * available_height + max_radius).astype(int)
    positions[:, 1] = np.round(points[:, 1] * available_width + max_radius).astype(int)

    return positions


@pytest.fixture(scope="session")
def footprints(
    params: CalciumVideoParams,
    ids: list[str],
    types: list[Enum],
    radii: np.ndarray,
    positions: np.ndarray,
) -> xr.DataArray:
    """Generate spatial footprints for neurons."""
    cell_shapes = []
    # Generate spatial profiles
    for n in range(params.num_neurons):
        profile = create_irregular_neuron(int(radii[n]), params.neuron_shape_irregularity)
        cell_shapes.append(profile)

    # Create xarray with proper coordinates
    footprints = xr.DataArray(
        np.zeros((params.num_neurons, params.height, params.width)),
        dims=["component", "height", "width"],
        coords={
            "id_": ("component", ids),
            "type_": ("component", types),
        },
    )

    # Place profiles in the full frame
    for idx, (radius, shape, (y_pos, x_pos)) in enumerate(zip(radii, cell_shapes, positions)):
        radius = int(radius)
        y_slice = slice(y_pos - radius, y_pos + radius + 1)
        x_slice = slice(x_pos - radius, x_pos + radius + 1)
        footprints[idx, y_slice, x_slice] = shape

    return footprints


@pytest.fixture(scope="session")
def spikes(
    params: CalciumVideoParams, ids: list[str], types: list[str], frame_coords: dict
) -> xr.DataArray:
    """Generate spike times for neurons."""
    firing_rates = np.random.uniform(*params.firing_rate_range, params.num_neurons)
    spikes = np.random.rand(params.num_neurons, params.frames) < firing_rates[:, None]

    return xr.DataArray(
        spikes,
        dims=["component", "frame"],
        coords={
            "id_": ("component", ids),
            "type_": ("component", types),
        },
    ).assign_coords(frame_coords)


@pytest.fixture(scope="session")
def traces(params: CalciumVideoParams, spikes: np.ndarray) -> np.ndarray:
    """Generate calcium traces from spikes."""
    decay_times = np.random.uniform(*params.decay_time_range, params.num_neurons)
    amplitudes = np.random.uniform(*params.amplitude_range, params.num_neurons)

    traces_data = np.zeros((params.num_neurons, params.frames))

    for n in range(params.num_neurons):
        spike_times = np.where(spikes[n])[0]
        for t in spike_times:
            traces_data[n, t:] += amplitudes[n] * np.exp(
                -(np.arange(params.frames - t)) / decay_times[n]
            )

    return xr.DataArray(traces_data, dims=["component", "frame"], coords=spikes.coords)


@pytest.fixture(scope="session")
def camera_motion(params: CalciumVideoParams) -> xr.DataArray:
    """Generate camera motion vectors."""
    # High frequency component for shake
    high_freq = np.random.normal(0, 1, (params.frames, 2))

    # Low frequency component for drift
    t = np.linspace(0, 2 * np.pi * params.motion_frequency, params.frames)
    low_freq_y = 0.3 * params.motion_amplitude[0] * np.sin(t + np.random.random() * np.pi)
    low_freq_x = 0.3 * params.motion_amplitude[1] * np.cos(t + np.random.random() * np.pi)

    # Combine and smooth
    motion_y = gaussian_filter(params.motion_amplitude[0] * high_freq[:, 0] + low_freq_y, sigma=1.0)
    motion_x = gaussian_filter(params.motion_amplitude[1] * high_freq[:, 1] + low_freq_x, sigma=1.0)

    return xr.DataArray(
        np.stack([motion_y, motion_x], axis=1),
        dims=["frame", "direction"],
        coords={"frame": range(params.frames), "direction": ["y", "x"]},
    )


@pytest.fixture(scope="session")
def motion_operator(camera_motion: xr.DataArray) -> np.ndarray:
    """Generate motion operator from camera motion."""
    return np.array(
        [[[1, 0, -motion_x], [0, 1, -motion_y]] for motion_y, motion_x in camera_motion.values],
        dtype=np.float32,
    )


@pytest.fixture(scope="session")
def scope_noise(
    params: CalciumVideoParams,
    glow: xr.DataArray,
    hot_pixels: xr.DataArray,
    dead_pixels: xr.DataArray,
    noise: xr.DataArray,
) -> xr.DataArray:
    """Generate noise and artifact patterns."""
    # Base
    residuals = xr.DataArray(
        np.zeros((params.frames, params.height, params.width)),
        dims=["frame", "height", "width"],
    )

    # Add artifacts
    residuals += glow
    residuals += hot_pixels
    residuals += dead_pixels
    residuals += noise

    return residuals


@pytest.fixture(scope="session")
def raw_calcium_video(
    params: CalciumVideoParams,
    footprints: xr.DataArray,
    traces: xr.DataArray,
    motion_operator: np.ndarray,
    scope_noise: xr.DataArray,
    photobleaching: np.ndarray,
) -> xr.DataArray:
    """Combine all components into final video."""

    # Start with residuals
    video = scope_noise.copy()

    # Add neurons with calcium activity
    video += (footprints @ traces).transpose("frame", "height", "width")

    # Add photobleaching
    video *= photobleaching

    # Apply motion
    # Apply transformations frame by frame (can't fully vectorize due to cv2.warpAffine)
    motion_video = np.stack(
        [
            cv2.warpAffine(
                frame.values,
                transform_matrix,
                (params.width, params.height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0.0,),
            )
            for frame, transform_matrix in zip(video, motion_operator)
        ]
    )

    return xr.DataArray(motion_video, dims=video.dims)


@pytest.fixture(scope="session")
def preprocessed_video(
    params: CalciumVideoParams,
    footprints: xr.DataArray,
    traces: xr.DataArray,
    motion_operator: np.ndarray,
    photobleaching: np.ndarray,
) -> xr.DataArray:
    """Calcium imaging video with artifacts removed except photobleaching."""
    # Add neurons with calcium activity
    video = (footprints @ traces).transpose("frame", "height", "width")

    # Add photobleaching
    video *= photobleaching

    # Apply motion
    # Apply transformations frame by frame (can't fully vectorize due to cv2.warpAffine)
    movie = np.stack(
        [
            cv2.warpAffine(
                frame.values,
                transform_matrix,
                (params.width, params.height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0.0,),
            )
            for frame, transform_matrix in zip(video, motion_operator)
        ]
    )

    return xr.DataArray(movie, dims=video.dims, coords=video.coords)


@pytest.fixture(scope="session")
def stabilized_video(
    footprints: xr.DataArray, traces: xr.DataArray, photobleaching: np.ndarray
) -> xr.DataArray:
    """Motion-corrected calcium imaging video."""

    stabilized = (footprints @ traces).transpose("frame", "height", "width")

    # add photobleaching
    stabilized *= photobleaching

    return stabilized


def create_irregular_neuron(radius: int, irregularity: float) -> np.ndarray:
    """Create an irregular neuron shape using pure Gaussian falloff."""
    # Create grid in polar coordinates
    y, x = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    distance = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)

    # Create base intensity using pure Gaussian
    sigma = radius * 0.5
    intensity = np.exp(-(distance**2) / (2 * sigma**2))

    # Add irregularity through angular modulation
    num_angles = int(3 + irregularity * 5)  # 3 < angular components
    for i in range(num_angles):
        # Create random angular frequency and phase
        freq = i + 1  # increasing frequencies
        phase = 2 * np.pi * np.random.random()
        amp = irregularity * 0.3 * (0.5**i)  # decreasing amplitude for higher frequencies

        # Modulate the radius based on angle
        modulation = 1 + amp * np.cos(freq * theta + phase)
        # Apply modulation with smooth falloff
        intensity *= 1 + 0.2 * modulation * np.exp(-(distance**2) / (2 * (radius * 0.8) ** 2))

    # Normalize
    intensity = intensity / np.max(intensity)

    return intensity


@pytest.fixture(scope="session")
def glow(params: CalciumVideoParams) -> xr.DataArray:
    """Generate moving glow artifact."""
    frames, height, width = params.frames, params.height, params.width
    video = np.zeros((frames, height, width))

    y, x = np.ogrid[0:height, 0:width]
    glow_center_y = height // 2 + np.sin(np.linspace(0, 2 * np.pi, frames)) * (height // 4)
    glow_center_x = width // 2 + np.cos(np.linspace(0, 2 * np.pi, frames)) * (width // 4)

    # Reshape centers to broadcast with coordinate grids
    glow_center_y = glow_center_y.reshape(-1, 1, 1)
    glow_center_x = glow_center_x.reshape(-1, 1, 1)

    # Calculate distances for all frames at once
    dist_sq = (y - glow_center_y) ** 2 + (x - glow_center_x) ** 2

    # Calculate glow
    glow = params.glow_intensity * np.exp(-dist_sq / (2 * (width * params.glow_sigma) ** 2))

    # Add temporal variation (vectorized)
    temporal_mod = 1 + 0.2 * np.sin(2 * np.pi * np.arange(frames) / (frames / 2))
    glow *= temporal_mod.reshape(-1, 1, 1)

    return xr.DataArray(video, dims=["frame", "height", "width"])


@pytest.fixture(scope="session")
def photobleaching(params: CalciumVideoParams) -> np.ndarray:
    """Generate photobleaching decay."""
    if params.photobleaching_decay > 0:
        decay = np.exp(-np.arange(params.frames) * params.photobleaching_decay / params.frames)
        return decay[:, np.newaxis, np.newaxis]
    return np.ones((params.frames, 1, 1))


@pytest.fixture(scope="session")
def dead_pixels(params: CalciumVideoParams) -> xr.DataArray:
    """Generate dead (black) pixels mask."""
    video = np.ones((params.frames, params.height, params.width))

    num_dead = int(params.height * params.width * params.dead_pixel_fraction)
    dead_y = np.random.randint(0, params.height, num_dead)
    dead_x = np.random.randint(0, params.width, num_dead)
    video[:, dead_y, dead_x] = 0

    return xr.DataArray(video, dims=["frame", "height", "width"])


@pytest.fixture(scope="session")
def hot_pixels(params: CalciumVideoParams) -> xr.DataArray:
    """Generate hot (bright) pixels."""
    video = np.zeros((params.frames, params.height, params.width))

    num_hot = int(params.height * params.width * params.hot_pixel_fraction)
    hot_y = np.random.randint(0, params.height, num_hot)
    hot_x = np.random.randint(0, params.width, num_hot)
    hot_values = params.hot_pixel_intensity * (1 + 0.2 * np.random.randn(num_hot))
    video[:, hot_y, hot_x] = hot_values[np.newaxis, :]

    return xr.DataArray(video, dims=["frame", "height", "width"])


@pytest.fixture(scope="session")
def noise(params: CalciumVideoParams) -> xr.DataArray:
    """Generate random noise with baseline."""
    noise = np.random.normal(
        params.baseline,
        params.noise_level,
        (params.frames, params.height, params.width),
    )

    # Add baseline drift
    tau = np.linspace(0, 4 * np.pi, params.frames)
    drift = params.drift_magnitude * np.sin(tau)
    noise += drift[:, np.newaxis, np.newaxis]

    return xr.DataArray(noise, dims=["frame", "height", "width"])
