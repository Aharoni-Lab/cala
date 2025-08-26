import numpy as np
from skimage.metrics import structural_similarity

from cala.nodes.prep.hlines import remove
from cala.testing.util import generate_text_image
from cala.util import package_frame


def test_remove_lines():
    img = generate_text_image(
        "8", frame_dims=(256, 256), org=(25, 230), thickness=20, font_scale=10
    )

    noise_amp = 40
    noise = np.tile(np.random.randint(0, noise_amp, img.shape[0]), (img.shape[1], 1)).T

    noisy_img = img // 1.5 + noise

    frame = package_frame(noisy_img, 0)

    result = remove(frame)

    assert structural_similarity(img.astype(int), result.array.values.astype(int)) == 1
