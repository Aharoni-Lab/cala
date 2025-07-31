from cala.models import Footprints, Traces, PixStat, CompStat, Overlap, Residual, Movie


def footprints() -> Footprints:
    return Footprints()


def traces() -> Traces:
    return Traces()


def pixel_stats() -> PixStat:
    return PixStat()


def comp_stats() -> CompStat:
    return CompStat()


def overlaps() -> Overlap:
    return Overlap()


def residuals() -> Residual:
    return Residual()


def buffer() -> Movie:
    return Movie()
