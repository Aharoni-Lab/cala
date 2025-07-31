from datetime import datetime

import numpy as np

from cala.assets import Frame
from cala.util import package_frame


def test_package_frame():
    # Create a sample 2D numpy array
    frame = np.random.randint(0, 256, size=(100, 200)).astype(np.float64)
    index = 5
    timestamp = datetime(2023, 4, 8, 12, 0, 0)

    # Transform the frame
    dataarray = package_frame(frame, index, timestamp)

    assert Frame.from_array(dataarray)
