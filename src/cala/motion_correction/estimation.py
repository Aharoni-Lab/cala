import numpy as np
import xarray as xr
from skimage.registration import phase_cross_correlation
from typing import Tuple, Optional


def select_base_frame(
    frames: xr.DataArray, base_index: Optional[int] = 0
) -> xr.DataArray:
    """
    Select a base frame from the video frames.

    Parameters:
    - frames: xarray.DataArray containing video frames.
    - base_index: Index of the base frame.

    Returns:
    - xarray.DataArray of the base frame.
    """
    if base_index < 0 or base_index >= frames.sizes["frame"]:
        raise IndexError("base_index is out of bounds.")

    base_frame = frames.isel(frame=base_index).compute().astype(np.uint8)
    return xr.DataArray(
        base_frame, dims=("y", "x"), coords={"y": frames.y, "x": frames.x}
    )


def compute_shift_vectorized(
    current_frame: np.ndarray, base_frame: np.ndarray
) -> Tuple[float, float]:
    """
    Compute the shift between a current frame and the base frame using phase cross-correlation.

    Parameters:
    - current_frame: Grayscale current frame as a 2D NumPy array.
    - base_frame: Grayscale base frame as a 2D NumPy array.

    Returns:
    - shift_y: Shift in the y-direction (pixels).
    - shift_x: Shift in the x-direction (pixels).
    """
    shift, error, diffphase = phase_cross_correlation(
        base_frame, current_frame, upsample_factor=10
    )
    return shift  # (shift_y, shift_x)


def estimate_motion(
    frames: xr.DataArray,
    base_frame: xr.DataArray,
) -> xr.Dataset:
    """
    Estimate x and y translations for each frame relative to the base frame using apply_ufunc.

    Parameters:
    - frames: xarray.DataArray containing video frames.
    - base_frame: xarray.DataArray of the base frame.

    Returns:
    - xarray.Dataset containing 'shift_x' and 'shift_y' for each frame.
    """
    # Ensure base_frame is a NumPy array
    base_frame_np = base_frame.data

    # Define the vectorized function using apply_ufunc
    shifts = xr.apply_ufunc(
        compute_shift_vectorized,  # Function to apply
        frames,  # Input frames
        xr.full_like(
            frames.isel(frame=0), base_frame_np
        ),  # Broadcast base_frame across all frames
        input_core_dims=[["y", "x"], ["y", "x"]],  # Core dimensions of inputs
        output_core_dims=[["shift"]],  # Core dimensions of outputs
        vectorize=True,  # Vectorize over non-core dimensions (frame)
        dask="parallelized",  # Enable Dask parallelization
        output_dtypes=[float],  # Specify output data types
    )

    # The shifts DataArray has a new 'shift' dimension with size 2 (shift_y, shift_x)
    # Split it into two separate DataArrays
    shift_y = shifts.sel(shift=0).rename("shift_y")
    shift_x = shifts.sel(shift=1).rename("shift_x")

    # Create a Dataset to hold both shifts
    shifts_ds = xr.Dataset(
        {
            "shift_y": shift_y,
            "shift_x": shift_x,
        },
        coords={"frame": frames.frame},
        attrs={
            "description": "Estimated translations relative to the base frame using apply_ufunc"
        },
    )

    return shifts_ds
