from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
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
    neuron_size_range: Tuple[int, int] = (40, 60)  # min/max radius in pixels
    neuron_shape_irregularity: float = 1.8  # 0=perfect circles, higher = more irregular
    margin: int = 50  # border margin for neuron placement

    # Calcium dynamics
    decay_time_range: Tuple[float, float] = (10, 20)  # frames
    firing_rate_range: Tuple[float, float] = (0.05, 0.15)  # probability per frame
    amplitude_range: Tuple[float, float] = (0.5, 1.0)

    # Motion
    motion_amplitude: Tuple[float, float] = (7, 7)  # pixels in y, x
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


@pytest.fixture
def params():
    """Return default parameters for video generation."""
    return CalciumVideoParams()


@pytest.fixture
def footprints(params):
    """Generate spatial footprints for neurons."""
    footprints_data = []
    positions = []
    radii = np.random.uniform(*params.neuron_size_range, params.num_neurons)

    # Generate random positions
    for _ in range(params.num_neurons):
        while True:
            # Ensure position + radius stays within frame bounds
            pos = np.random.randint(
                params.margin
                + int(radii[len(positions)]),  # Min position must include radius
                np.array([params.height, params.width])
                - params.margin
                - int(radii[len(positions)]),  # Max position must include radius
                size=2,
            )
            # Check if position is far enough from existing neurons
            if not positions or all(
                np.linalg.norm(pos - p) > r1 + r2
                for p, r1, r2 in zip(
                    positions, radii[: len(positions)], [radii[len(positions)]]
                )
            ):
                positions.append(pos)
                break

    positions = np.array(positions)

    # Generate spatial profiles
    for n in range(params.num_neurons):
        profile = create_irregular_neuron(
            int(radii[n]), params.neuron_shape_irregularity
        )
        footprints_data.append(profile)

    # Create xarray with proper coordinates
    footprints_xr = xr.DataArray(
        np.zeros((params.num_neurons, params.height, params.width)),
        dims=["components", "height", "width"],
        coords={
            "components": range(params.num_neurons),
            "id_": ("components", [f"comp_{i}" for i in range(params.num_neurons)]),
            "type_": ("components", ["neuron"] * params.num_neurons),
        },
    )

    # Place profiles in the full frame
    for n in range(params.num_neurons):
        y_pos, x_pos = positions[n]
        radius = int(radii[n])
        y_slice = slice(y_pos - radius, y_pos + radius + 1)
        x_slice = slice(x_pos - radius, x_pos + radius + 1)
        footprints_xr[n, y_slice, x_slice] = footprints_data[n]

    return footprints_xr, positions, radii


@pytest.fixture
def spikes(params):
    """Generate spike times for neurons."""
    firing_rates = np.random.uniform(*params.firing_rate_range, params.num_neurons)
    spikes = (
        np.random.random((params.num_neurons, params.frames)) < firing_rates[:, None]
    )

    return xr.DataArray(
        spikes,
        dims=["components", "frames"],
        # coords={
        #     "components": range(params.num_neurons),
        #     "frames": range(params.frames),
        # },
    )


@pytest.fixture
def traces(params, spikes):
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

    return xr.DataArray(
        traces_data, dims=["components", "frames"], coords=spikes.coords
    )


@pytest.fixture
def camera_motion(params):
    """Generate camera motion vectors."""
    # High frequency component for shake
    high_freq = np.random.normal(0, 1, (params.frames, 2))

    # Low frequency component for drift
    t = np.linspace(0, 2 * np.pi * params.motion_frequency, params.frames)
    low_freq_y = (
        0.3 * params.motion_amplitude[0] * np.sin(t + np.random.random() * np.pi)
    )
    low_freq_x = (
        0.3 * params.motion_amplitude[1] * np.cos(t + np.random.random() * np.pi)
    )

    # Combine and smooth
    motion_y = gaussian_filter(
        params.motion_amplitude[0] * high_freq[:, 0] + low_freq_y, sigma=1.0
    )
    motion_x = gaussian_filter(
        params.motion_amplitude[1] * high_freq[:, 1] + low_freq_x, sigma=1.0
    )

    return xr.DataArray(
        np.stack([motion_y, motion_x], axis=1),
        dims=["frames", "direction"],
        coords={"frames": range(params.frames), "direction": ["y", "x"]},
    )


@pytest.fixture
def residuals(params):
    """Generate noise and artifact patterns."""
    # Base noise
    residuals = np.random.normal(
        params.baseline,
        params.noise_level,
        (params.frames, params.height, params.width),
    )

    # Add baseline drift
    tau = np.linspace(0, 4 * np.pi, params.frames)
    drift = params.drift_magnitude * np.sin(tau)
    residuals += drift[:, np.newaxis, np.newaxis]

    # Add artifacts
    residuals = add_artifacts(residuals, params)

    return xr.DataArray(residuals, dims=["frames", "height", "width"])


@pytest.fixture
def raw_calcium_video(params, footprints, traces, camera_motion, residuals):
    """Combine all components into final video."""

    # Start with residuals
    video = residuals.copy()

    # Add neurons with calcium activity
    for n in range(params.num_neurons):
        video += footprints[0][n] * traces[n]

    video = video.reset_coords(["components", "id_", "type_"], drop=True)
    # Apply blur
    video = xr.apply_ufunc(
        lambda x: gaussian_filter(x, params.blur_sigma),
        video,
        input_core_dims=[["height", "width"]],
        output_core_dims=[["height", "width"]],
        vectorize=True,
    )

    # Apply motion
    motion_video = np.zeros_like(video)
    for f in range(params.frames):
        transform_matrix = np.array(
            [[1, 0, -camera_motion[f, 1].item()], [0, 1, -camera_motion[f, 0].item()]],
            dtype=np.float32,
        )

        motion_video[f] = cv2.warpAffine(
            video[f].values,
            transform_matrix,
            (params.width, params.height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0.0,),
        )

    return xr.DataArray(motion_video, dims=video.dims, coords=video.coords)


@pytest.fixture
def preprocessed_video(raw_calcium_video, params, footprints, traces, camera_motion):
    """Calcium imaging video with artifacts removed except photobleaching."""
    video = raw_calcium_video
    frames, height, width = video.shape

    clean = np.zeros(
        (params.frames, params.height, params.width),
    )

    # Unpack footprints
    footprints_xr, positions, radii = footprints

    # Add neurons to video without motion
    for f in range(frames):
        for n in range(params.num_neurons):
            y_pos, x_pos = positions[n]
            radius = int(radii[n])

            y_slice = slice(y_pos - radius, y_pos + radius + 1)
            x_slice = slice(x_pos - radius, x_pos + radius + 1)

            if (
                0 <= y_pos - radius
                and y_pos + radius + 1 <= params.height
                and 0 <= x_pos - radius
                and x_pos + radius + 1 <= params.width
            ):
                clean[f, y_slice, x_slice] += (
                    traces[n, f] * footprints_xr[n, y_slice, x_slice]
                )

    # Apply Gaussian blur
    for f in range(frames):
        clean[f] = gaussian_filter(clean[f], sigma=params.blur_sigma)

    # Apply motion using subpixel interpolation
    motion_video = np.zeros_like(clean)
    for f in range(frames):
        # Create transformation matrix for translation
        transform_matrix = np.array(
            [[1, 0, camera_motion[f, 1].item()], [0, 1, camera_motion[f, 0].item()]],
            dtype=np.float32,
        )

        # Apply translation using warpAffine with bilinear interpolation
        frame = clean[f].astype(np.float32)
        motion_video[f] = cv2.warpAffine(
            frame,
            transform_matrix,
            (params.width, params.height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0.0,),
        )

    # Keep photobleaching artifact
    if params.photobleaching_decay > 0:
        decay = np.exp(-np.arange(frames) * params.photobleaching_decay / frames)
        motion_video *= decay[:, np.newaxis, np.newaxis]

    clean_xr = xr.DataArray(motion_video, dims=video.dims, coords=video.coords)

    return clean_xr


@pytest.fixture
def stabilized_video(preprocessed_video, camera_motion, params: CalciumVideoParams):
    """Motion-corrected calcium imaging video."""
    video = preprocessed_video

    # Remove the artificial motion using subpixel interpolation
    stabilized = np.zeros_like(video)

    for f in range(params.frames):
        # Create transformation matrix for reverse translation
        transform_matrix = np.array(
            [[1, 0, -camera_motion[f, 1].item()], [0, 1, -camera_motion[f, 0].item()]],
            dtype=np.float32,
        )

        # Apply reverse translation using warpAffine with bilinear interpolation
        frame = video[f].astype(np.float32)
        stabilized[f] = cv2.warpAffine(
            frame.values,
            transform_matrix,
            (params.width, params.height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0.0,),
        )

    # Convert to xarray with same coordinates
    stabilized_xr = xr.DataArray(stabilized, dims=video.dims, coords=video.coords)

    return stabilized_xr


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
        amp = (
            irregularity * 0.3 * (0.5**i)
        )  # decreasing amplitude for higher frequencies

        # Modulate the radius based on angle
        modulation = 1 + amp * np.cos(freq * theta + phase)
        # Apply modulation with smooth falloff
        intensity *= 1 + 0.2 * modulation * np.exp(
            -(distance**2) / (2 * (radius * 0.8) ** 2)
        )

    # Normalize
    intensity = intensity / np.max(intensity)

    return intensity


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
