import xarray as xr


def assert_scalar_multiple_arrays(a: xr.DataArray, b: xr.DataArray, /, rtol: float = 1e-5) -> None:
    """Using the Pythagorean Theorem"""

    if not 0 <= rtol <= 1:
        raise ValueError(f"rtol must be between 0 and 1, got {rtol}.")

    abab = (a @ b) ** 2
    aabb = a.dot(a) * b.dot(b)

    assert abab > aabb * (1 - rtol)
