from typing import Annotated as A

from noob import Name

from cala.assets import CompStats, Footprints, Movie, Overlaps, PixStats, Traces
from cala.nodes.component_stats import ingest_component as update_component_stats
from cala.nodes.footprints import ingest_component as update_footprints
from cala.nodes.overlap import ingest_component as update_overlap
from cala.nodes.pixel_stats import ingest_component as update_pixel_stats
from cala.nodes.traces import ingest_component as update_traces


def update_assets(
    new_footprints: Footprints,
    new_traces: Traces,
    footprints: Footprints,
    traces: Traces,
    pixel_stats: PixStats,
    component_stats: CompStats,
    overlaps: Overlaps,
    buffer: Movie,
) -> tuple[
    A[Traces, Name("traces")],
    A[Footprints, Name("footprints")],
    A[PixStats, Name("pixel_stats")],
    A[CompStats, Name("component_stats")],
    A[Overlaps, Name("overlaps")],
]:
    # Overlap must be done before Footprint to prevent the new footprints going into
    # known ones
    updated_overlaps = update_overlap(
        overlaps=overlaps, footprints=footprints, new_footprints=new_footprints
    )
    updated_shapes = update_footprints(footprints=footprints, new_footprints=new_footprints)

    updated_traces = update_traces(traces=traces, new_traces=new_traces)
    updated_pixel_stats = update_pixel_stats(
        pixel_stats=pixel_stats, frames=buffer, traces=traces, new_traces=new_traces
    )
    updated_component_stats = update_component_stats(
        component_stats=component_stats, traces=traces, new_traces=new_traces
    )

    return (
        updated_traces,
        updated_shapes,
        updated_pixel_stats,
        updated_component_stats,
        updated_overlaps,
    )
