import cv2
import numpy as np
import time


def create_shifted_image(image, dx, dy):
    """
    Shifts the input image by (dx, dy) pixels.
    """
    rows, cols = image.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(image, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)
    return shifted


def align_ecc(
    template,
    image,
    warp_mode=cv2.MOTION_TRANSLATION,
    number_of_iterations=500,
    termination_eps=1e-6,
):
    """
    Aligns `image` to `template` using ECC maximization.
    Returns the warp matrix and the time taken.
    """
    warp_matrix = np.eye(2, 3, dtype=np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        number_of_iterations,
        termination_eps,
    )

    start_time = time.time()
    try:
        cc, warp_matrix = cv2.findTransformECC(
            template, image, warp_matrix, warp_mode, criteria
        )
    except cv2.error as e:
        print("Error during findTransformECC:", e)
        return None, None
    end_time = time.time()

    return warp_matrix, end_time - start_time


def align_phase_correlation(template, image):
    """
    Aligns `image` to `template` using phase correlation.
    Returns the translation vector and the time taken.
    """
    # Convert images to float32
    template_f = np.float32(template)
    image_f = np.float32(image)

    start_time = time.time()
    (shift, response) = cv2.phaseCorrelate(template_f, image_f)
    end_time = time.time()

    return shift, end_time - start_time


def main():
    template = cv2.imread("./sample_image.jpg", cv2.IMREAD_GRAYSCALE)
    if template is None:
        print(
            "Error: Image not found. Please ensure 'sample_image.png' exists in the working directory."
        )
        return

    # Apply a known shift
    dx, dy = 30, 15  # Shift along x and y axes
    shifted = create_shifted_image(template, dx, dy)

    # Align using findTransformECC
    warp_matrix, ecc_time = align_ecc(template, shifted)
    if warp_matrix is not None:
        aligned_ecc = cv2.warpAffine(
            shifted,
            warp_matrix,
            (template.shape[1], template.shape[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        )
        # Calculate the difference
        difference_ecc = cv2.absdiff(template, aligned_ecc)
    else:
        print("ECC alignment failed.")
        difference_ecc = None

    # Align using Phase Correlation
    shift, phase_time = align_phase_correlation(template, shifted)
    # Compute the translation matrix for phase correlation result
    M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
    aligned_phase = cv2.warpAffine(
        shifted,
        M,
        (template.shape[1], template.shape[0]),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
    )
    difference_phase = cv2.absdiff(template, aligned_phase)

    print(f"findTransformECC time: {ecc_time:.6f} seconds")
    print(f"Phase Correlation time: {phase_time:.6f} seconds")

    cv2.imshow("Original (Template)", template)
    cv2.imshow("Shifted Image", shifted)
    if difference_ecc is not None:
        cv2.imshow("Aligned ECC", aligned_ecc)
        cv2.imshow("Difference ECC", difference_ecc)
    cv2.imshow("Aligned Phase Correlation", aligned_phase)
    cv2.imshow("Difference Phase Correlation", difference_phase)

    print(f"Known shift: dx={dx}, dy={dy}")
    if warp_matrix is not None:
        print(
            f"ECC estimated shift: dx={warp_matrix[0,2]:.2f}, dy={warp_matrix[1,2]:.2f}"
        )
    print(f"Phase Correlation estimated shift: dx={shift[0]:.2f}, dy={shift[1]:.2f}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
