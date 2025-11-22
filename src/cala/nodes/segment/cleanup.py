from typing import Annotated as A

import numpy as np
from noob import Name

from cala.assets import Buffer, CompStats, Footprints, Overlaps, PixStats, Traces
from cala.models import AXIS


def clear_overestimates(
    footprints: Footprints, residuals: Buffer, nmf_error: float
) -> A[Footprints, Name("footprints")]:
    """
    Remove all sections of the footprints that cause negative residuals.

    This occurs by:
    1. find "significant" negative residual spots that is more than a noise level, and thus
    cannot be clipped to zero. !!!! (only of the latest frame, and then go back to trace update..?)
    2. all footprint values at these spots go to zero.
    """
    if residuals.array is None:
        return footprints
    R_min = residuals.array.isel({AXIS.frames_dim: -1}).reset_coords(
        [AXIS.frame_coord, AXIS.timestamp_coord], drop=True
    )
    tuned_fp = footprints.array.where(R_min > -nmf_error, 0, drop=False)

    return tuned_fp


def deprecate_components(
    footprints: Footprints,
    traces: Traces,
    pix_stats: PixStats,
    comp_stats: CompStats,
    overlaps: Overlaps,
    remove_ids: list[str],
) -> tuple[
    A[Footprints, Name("footprints")],
    A[Traces, Name("traces")],
    A[PixStats, Name("pix_stats")],
    A[CompStats, Name("comp_stats")],
    A[Overlaps, Name("overlaps")],
]:
    """
    Deprecate a list of components from all assets involved in omf.

    """
    keep_mask = ~np.isin(traces.array[AXIS.id_coord].values, remove_ids)

    traces.keep(keep_mask)
    # the line below compiles numba. gotta do it like in footprints.ingest_component
    # but then i need to redundantly convert COO -> csr -> COO -> csr -> COO
    footprints.array = footprints.array[keep_mask]
    pix_stats.array = pix_stats.array[keep_mask]
    comp_stats.array = comp_stats.array[keep_mask].T[keep_mask]
    overlaps.array = overlaps.array[keep_mask].T[keep_mask]

    return footprints, traces, pix_stats, comp_stats, overlaps


def find_inactive() -> list[str]:
    """
    Deprecate inactive components
    Component is deemed inactive if its own brightness contribution across
    all of its footprint is below threshold.

    1. has been some time since discovery
    2. within its own footprint, its brightness contribution is lower than
        some % of the minimum of the total brightness contributions from all components?
        - but what if the component is completely occluded sometimes?
    """
