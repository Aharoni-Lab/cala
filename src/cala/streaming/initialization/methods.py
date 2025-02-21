import cv2
import numpy as np
from skimage.segmentation import watershed


class Initializer:
    """Base class for initialization methods
    To start off realtime CNMF, I need the temporal trace matrix (C). To do that, I need to estimate A, b, and f.
    1. Estimate A based on LoG?
    2. Y - A is b.
    3. C can be traced out from A.
    3. f can be traced out from b.

    we could start off with user input of cell size and number of components --> users are wrong about these very often.
    we don't really need num_components to begin with
    multiple estimates
    estimate error rate
    confidence score for each

    Exogenous Information:

    calcium indicator rise/decay time constant --> this is not constant
    refractory period
    "probably no cell" areas - faster, sparse matrix compute
    out of focus cells - what to do.
    """

    def watershed_components(self, frame):
        # Convert frame to uint8 before thresholding
        frame_norm = (frame - frame.min()) * (255.0 / (frame.max() - frame.min()))
        frame_uint8 = frame_norm.astype(np.uint8)
        _, binary = cv2.threshold(
            frame_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Sure background area (by dilating the foreground)
        kernel = np.ones((3, 3), np.uint8)
        sure_background = cv2.dilate(binary, kernel, iterations=1)

        # 5. Compute distance transform of the foreground
        distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

        # 6. Threshold the distance transform to get sure foreground
        #    The factor 0.7 can be adjusted depending on how much footprints overlap
        _, sure_foreground = cv2.threshold(distance, 0.2 * distance.max(), 255, 0)
        sure_foreground = sure_foreground.astype(np.uint8)

        # 7. Identify unknown region (not sure foreground, not sure background)
        unknown = cv2.subtract(
            sure_background.astype(np.float32), sure_foreground.astype(np.float32)
        ).astype(np.uint8)

        # 8. Label the sure foreground with connected components
        num_markers, markers = cv2.connectedComponents(sure_foreground.astype(np.uint8))

        # Important: increment all labels so background is not 0 but 1
        markers = markers + 1
        # Mark the unknown region as 0
        markers[unknown == 255] = 0

        # 9. Call watershed
        markers = watershed(frame_uint8, markers)

        blobs = []
        for blob_idx in range(2, num_markers + 1):
            blobs.append(markers == blob_idx)

        return np.array(blobs)
