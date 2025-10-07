# # These tests are applicable when there is real data.
# # Naturally, this is harder to implement on a CI/CD environment,
# # and thus have been left as comments for possible future uses.
#
#
# from collections.abc import Generator
#
# import cv2
# import numpy as np
# import pytest
# import xarray as xr
# from skimage.filters import difference_of_gaussians
#
# from cala.config import config
# from cala.nodes.io import stream
# from cala.nodes.prep import blur, butter, remove_mean
# from cala.nodes.prep.motion import Anchor, Shift, apply_shift, register_shift
# from cala.util import package_frame
#
#
# @pytest.fixture
# def real() -> Generator[np.ndarray]:
#     return stream(
#         [
#             "cala/msCam1.avi",
#             "cala/msCam2.avi",
#             "cala/msCam3.avi",
#             "cala/msCam4.avi",
#             "cala/msCam5.avi",
#             "cala/msCam6.avi",
#             "cala/msCam7.avi",
#             "cala/msCam8.avi",
#             "cala/msCam9.avi",
#             "cala/msCam10.avi",
#         ]
#     )
#
#
# def median(img: xr.DataArray) -> xr.DataArray:
#     tmp = difference_of_gaussians(img, low_sigma=3)  # nothing: 1.3 min
#     tmp = cv2.normalize(tmp, None, alpha=0, beta=255,
#     norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
#     res = cv2.medianBlur(tmp, 11)  # 2 mins
#
#     return xr.DataArray(res, dims=img.dims, coords=img.coords)
#
#
# def nlm(img: xr.DataArray) -> xr.DataArray:
#     tmp = difference_of_gaussians(img, low_sigma=3)  # nothing: 1.3 min
#     tmp = cv2.normalize(tmp, None, alpha=0, beta=255, n
#     orm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
#     res = cv2.fastNlMeansDenoising(tmp, None, 7, 7, 21)  # 3 mins
#
#     return xr.DataArray(res, dims=img.dims, coords=img.coords)
#
#
# def gauss(img: xr.DataArray) -> xr.DataArray:
#     # tmp = img[100:300, 500:700]
#     tmp = difference_of_gaussians(img, low_sigma=3)  # nothing: 1.3 min
#     tmp = cv2.normalize(tmp, None, alpha=0, beta=255,
#     norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
#     res = cv2.GaussianBlur(tmp.astype(float), (11, 11), 20)  # 1.5 mins
#
#     return xr.DataArray(res, dims=img.dims, coords=img.coords)
#
#
# def test_real(real):
#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     out = cv2.VideoWriter(config.video_dir / "mc_prev.avi", fourcc, 24.0, (1504, 960))
#     prev = None
#
#     for i, arr in enumerate(real):
#         frame = package_frame(arr, i)
#         frame = blur(frame, method="median", kwargs={"ksize": 3})
#         frame = butter(frame, kwargs={})
#         frame = remove_mean(frame, orient="both")  # all of these takes 30 ish seconds
#
#         subplots = []
#         for func in [median, gauss, nlm]:
#             if prev is None:
#                 prev = func(frame.array)
#                 tmpl = prev
#                 break
#
#             prepped = func(frame.array)
#             drift, _, _ = register_shift(
#                 prev.values.astype(float), prepped.values.astype(float), upsample_factor=10
#             )
#             drift = Shift(height=drift[0], width=drift[1])
#
#             corrected = apply_shift(frame.array, drift)
#             prev = apply_shift(prepped, drift)
#
#             if i % 1 == 0:
#                 slow_drift, _, _ = register_shift(  # THIS WAS CORRECTED< NOT PREV
#                     tmpl.values.astype(float), prev.values.astype(float), upsample_factor=10
#                 )
#                 slow_drift = Shift(height=slow_drift[0], width=slow_drift[1])
#                 corrected = apply_shift(corrected, slow_drift)
#                 prev = apply_shift(prev, slow_drift)
#
#             subplots.append(corrected)
#
#         if subplots:
#             left_row = np.concat([frame.array, subplots[0]])
#             right_row = np.concat([subplots[1], subplots[2]])
#             view = np.concat([left_row, right_row], axis=1)
#             frame_bgr = cv2.cvtColor(view.astype(np.uint8), cv2.COLOR_GRAY2BGR)
#             out.write(frame_bgr)
#
#
# def test_motion2(real):
#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     out = cv2.VideoWriter(config.video_dir / "mc_lockon.avi", fourcc, 24.0, (752, 960))
#     anchor = Anchor()
#
#     for i, arr in enumerate(real):
#         frame = package_frame(arr, i)
#         frame = blur(frame, method="median", kwargs={"ksize": 3})
#         frame = butter(frame, kwargs={})
#         pre_mc = remove_mean(frame, orient="both")
#         frame = anchor.stabilize(pre_mc)
#
#         frame_bgr = cv2.cvtColor(
#             np.concat([pre_mc.array.astype(np.uint8), frame.array.astype(np.uint8)]),
#             cv2.COLOR_GRAY2BGR,
#         )
#         out.write(frame_bgr)
