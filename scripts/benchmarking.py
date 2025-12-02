from datetime import datetime

import cv2
import matplotlib
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from noob import SynchronousRunner, Tube

from cala.nodes.io import stream
from cala.nodes.prep import Anchor, blur, butter, package_frame, remove_mean
from cala.testing.util import total_gradient_magnitude

sns.set_style("whitegrid")
font = {"family": "normal", "weight": "regular", "size": 15}

matplotlib.rc("font", **font)


VIDEOS = [
    "minian/msCam1.avi",
    "minian/msCam2.avi",
    "minian/msCam3.avi",
    "minian/msCam4.avi",
    "minian/msCam5.avi",
    "minian/msCam6.avi",
    "minian/msCam7.avi",
    "minian/msCam8.avi",
    "minian/msCam9.avi",
    "minian/msCam10.avi",
    # "long_recording/0.avi",
    # "long_recording/1.avi",
    # "long_recording/2.avi",
    # "long_recording/3.avi",
    # "long_recording/4.avi",
]


def preprocess(arr, idx):
    frame = package_frame(arr, idx)
    frame = blur(frame, method="median", kwargs={"ksize": 3})
    frame = butter(frame, {})
    return remove_mean(frame, orient="both")


def test_encode():
    gen = stream(VIDEOS)

    fourcc = cv2.VideoWriter_fourcc(*"FFV1")
    out = cv2.VideoWriter("encode_test.avi", fourcc, 60.0, (600, 600))

    for arr in gen:
        frame_bgr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        out.write(frame_bgr)

    out.release()


def test_motion_movie():
    """
    For testing how well the motion correction performs with real movie

    """
    gen = stream(VIDEOS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("motion_test.avi", fourcc, 60.0, (600, 1200))

    stab = Anchor()

    for idx, arr in enumerate(gen):
        frame = preprocess(arr, idx)
        matched = stab.stabilize(frame)
        combined = np.concat([frame.array.values, matched.array.values], axis=0)

        frame_bgr = cv2.cvtColor(combined.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        out.write(frame_bgr)

    out.release()


def test_motion_crisp_pics():
    """
    For generating a mean summary frame across raw vs. motion-corrected video.
    The motion-corrected video should have a much crisper summary picture.

    """

    gen = stream(VIDEOS)

    stab = Anchor()
    raws = []
    stabs = []

    for idx, arr in enumerate(gen):
        frame = preprocess(arr, idx)
        matched = stab.stabilize(frame)
        raws.append(frame.array.values)
        stabs.append(matched.array.values)

    raw = np.stack(raws)
    stab = np.stack(stabs)

    raw_mean = np.mean(raw, axis=0)
    stab_mean = np.mean(stab, axis=0)

    crisp_raw = total_gradient_magnitude(raw_mean)
    crisp_stab = total_gradient_magnitude(stab_mean)

    print(f"{crisp_raw = }, {crisp_stab = }")

    mean = np.concatenate((raw_mean, stab_mean), axis=0)
    plt.imsave("motion_crisp_pics.png", mean, cmap="gray")


def test_motion_mean_corr():
    """
    For testing how well the motion correction performs with real movie
    """
    gen = stream(VIDEOS)

    stab = Anchor()
    raws = []
    stabs = []

    for idx, arr in enumerate(gen):
        frame = preprocess(arr, idx)
        matched = stab.stabilize(frame)
        raws.append(frame.array.values)
        stabs.append(matched.array.values)

    raw = np.stack(raws)
    stab = np.stack(stabs)

    raw_mean = np.mean(raw[:, 20:-20, 20:-20], axis=0)
    stab_mean = np.mean(stab[:, 20:-20, 20:-20], axis=0)

    raw_cms = [np.corrcoef(r[20:-20, 20:-20].flatten(), raw_mean.flatten())[0, 1] for r in raws]
    stab_cms = [np.corrcoef(s[20:-20, 20:-20].flatten(), stab_mean.flatten())[0, 1] for s in stabs]

    fig, ax = plt.subplots(figsize=(24, 10))
    plt.plot(raw_cms)
    plt.plot(stab_cms)
    plt.legend(["raw", "stabilized"], loc="upper right")
    plt.title("Mean Correlation")
    plt.xlabel("frame")
    plt.ylabel("correlation")
    plt.tight_layout()
    plt.savefig("mc.png")

    assert False


def test_with_movie():
    tube = Tube.from_specification("with-minian")
    runner = SynchronousRunner(tube=tube)
    processed_vid = runner.run()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("motion_test.avi", fourcc, 20.0, (100, 100))

    for arr in processed_vid:
        frame_bgr = cv2.cvtColor(arr.array.values.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        out.write(frame_bgr)
    out.release()


def test_processing_speed():
    tube = Tube.from_specification("with-minian")
    runner = SynchronousRunner(tube=tube)
    gen = runner.iter()
    frame_speed = []
    i = 0
    while True:
        try:
            start = datetime.now()
            next(gen)
            duration = datetime.now() - start
            frame_speed.append(round(duration.total_seconds(), 2))
            i += 1
        except RuntimeError:
            break
    fig, ax = plt.subplots(figsize=(20, 4))
    ax.set_yscale("log")
    plt.plot(frame_speed)
    plt.xlabel("frame", fontsize=20)
    plt.ylabel("time taken (s)", fontsize=20)
    plt.tight_layout()
    plt.savefig("frame_speed.png")


def test_deglow():

    gen = stream(VIDEOS[:3])

    stab = Anchor()
    stabs = []

    for idx, arr in enumerate(gen):
        frame = preprocess(arr, idx)
        matched = stab.stabilize(frame)
        stabs.append(matched.array.values)

    deglowed = stabs[-1] - np.min(stabs, axis=0)
    plt.imsave("deglowed.png", deglowed, cmap="gray")
