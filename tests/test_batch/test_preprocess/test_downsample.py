import pytest

from cala.batch.preprocess.downsample import Downsampler


@pytest.mark.parametrize(
    "method,strides",
    [
        ("mean", [2, 2, 2]),
        ("mean", [3, 3, 3]),
        ("subset", [2, 2, 2]),
        ("subset", [3, 3, 3]),
    ],
)
def test_downsampler_methods(method, strides, raw_calcium_video):
    video, _, _ = raw_calcium_video

    downsampler = Downsampler(
        method=method, dimensions=["frames", "height", "width"], strides=strides
    )
    result = downsampler.transform(video)

    # Calculate expected shape based on method
    if method == "mean":
        expected_shape = tuple(s // stride for s, stride in zip(video.shape, strides))
    else:  # subset
        expected_shape = tuple(
            len(range(0, s, stride)) for s, stride in zip(video.shape, strides)
        )

    assert result.shape == expected_shape
    assert isinstance(result, type(video))
    assert result.notnull().all()
