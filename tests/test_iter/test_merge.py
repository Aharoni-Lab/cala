import numpy as np
import pytest
from scipy.sparse.csgraph import connected_components

from cala.assets import Overlaps
from cala.models import AXIS
from cala.nodes.merge import _filter_targets, _merge_matrix, merge_existing
from cala.testing.toy import FrameDims, Position, Toy


@pytest.fixture(scope="module")
def n_frames() -> int:
    return 1000


@pytest.fixture(scope="module")
def positions() -> list[Position]:
    return [
        Position(width=10, height=8),
        Position(width=15, height=14),
        Position(width=25, height=6),
        Position(width=40, height=20),
        Position(width=34, height=44),
        Position(width=30, height=40),
    ]


@pytest.fixture(scope="module")
def traces(n_frames) -> list[np.ndarray]:
    return [
        np.array(range(n_frames)),
        np.array(range(n_frames)) + 1,
        np.random.randint(n_frames, size=n_frames),
        np.array(range(n_frames, 0, -1)) + 2,
        np.array(range(n_frames, 0, -1)),
        np.abs(np.sin(np.arange(n_frames)) * n_frames),
    ]


@pytest.fixture(scope="module")
def toy(n_frames: int, positions: list[Position], traces: list[np.ndarray]) -> Toy:
    return Toy(
        n_frames=n_frames,
        frame_dims=FrameDims(width=50, height=50),
        cell_radii=5,
        cell_positions=positions,
        cell_traces=traces,
        detected_ons=list(np.arange(len(traces)) * 100),
    )


@pytest.fixture(scope="module")
def overlaps(toy: Toy) -> Overlaps:
    return Overlaps.from_array(
        toy.footprints.array @ toy.footprints.array.rename(AXIS.component_rename) > 0
    )


def test_merge_matrix(toy: Toy, overlaps: Overlaps) -> None:
    mm = _merge_matrix(
        traces=toy.traces.array, overlaps=overlaps.array, smooth_kwargs={"sigma": 2}, threshold=0.9
    ).as_numpy()
    # expect 1 and 2 to merge, rest stand alone
    assert np.array_equal(connected_components(mm)[1], [0, 0, 1, 2, 3, 4])


def test_age_limit(toy: Toy, overlaps: Overlaps):
    age_limit = 700

    traces = toy.traces
    targets = traces.array[AXIS.detect_coord] <= (traces.array[AXIS.frame_coord].max() - age_limit)

    target_ids = targets.where(targets, drop=True)[AXIS.id_coord].values

    fp, tr, ov = _filter_targets(
        target_ids=target_ids,
        shapes=toy.footprints,
        traces=traces,
        overlaps=overlaps,
        n_frames=age_limit,
    )

    assert np.array_equal(fp[AXIS.id_coord], ["cell_0", "cell_1", "cell_2"])
    assert np.array_equal(tr[AXIS.id_coord], ["cell_0", "cell_1", "cell_2"])
    assert np.array_equal(ov[AXIS.id_coord], ["cell_0", "cell_1", "cell_2"])


def test_merge_existing(toy: Toy, overlaps: Overlaps) -> None:
    interval = 333
    fp, tr = merge_existing(
        shapes=toy.footprints,
        traces=toy.traces,
        overlaps=overlaps,
        merge_interval=interval,
        merge_threshold=0.9,
        smooth_kwargs={"sigma": 2},
    )

    assert fp.array.attrs["replaces"] == ["cell_0", "cell_1"]
    assert tr.array.attrs["replaces"] == ["cell_0", "cell_1"]
    assert fp.array.sizes[AXIS.component_dim] == 1
    assert tr.array.sizes == {AXIS.component_dim: 1, AXIS.frames_dim: interval}
