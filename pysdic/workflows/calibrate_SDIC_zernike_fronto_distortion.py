# Copyright 2026 Artezaru
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
from numbers import Integral
from typing import Optional, Dict, Tuple

import numpy
import pycvcam

import tqdm
import multiprocessing

from ..objects.point_cloud import PointCloud
from ..objects.camera import Camera
from ..objects.mesh import Mesh
from ..objects.view import View
from ..objects.image import Image
from ..objects.integration_points import IntegrationPoints

from ..core.optical_flow import compute_optical_flow


def calibrate_SDIC_zernike_fronto_distortion(
    image: numpy.ndarray,
    reference_image: numpy.ndarray,
    zernike_order: Integral,
    channel: Integral = 0,
    mask: Optional[numpy.ndarray] = None,
    disflow_parameters: Optional[Dict] = None,
    projection_parameters: Optional[Dict] = None,
    sdic_parameters: Optional[Dict] = None,
    verbose_level: Integral = 0,
) -> pycvcam.ZernikeDistortion:
    r"""
    TO WRITE
    """
    if not isinstance(image, numpy.ndarray):
        raise TypeError(
            f"Expected image to be an instance of numpy.ndarray, got {type(image)}"
        )
    if not numpy.issubdtype(image.dtype, numpy.unsignedinteger):
        raise TypeError(
            f"Expected image to have an unsigned integer dtype, got {image.dtype}"
        )
    if not image.ndim in (2, 3):
        raise ValueError(
            f"Expected image to be a 2D or 3D array, got an array with shape {image.shape}"
        )

    if not isinstance(reference_image, numpy.ndarray):
        raise TypeError(
            f"Expected reference_image to be an instance of numpy.ndarray, got {type(reference_image)}"
        )
    if not numpy.issubdtype(reference_image.dtype, numpy.unsignedinteger):
        raise TypeError(
            f"Expected reference_image to have an unsigned integer dtype, got {reference_image.dtype}"
        )
    if not reference_image.ndim == 2:
        raise ValueError(
            f"Expected reference_image to be a 2D array, got an array with shape {reference_image.shape}"
        )

    if not image.shape == reference_image.shape:
        raise ValueError(
            f"Expected image and reference_image to have the same shape, got {image.shape} and {reference_image.shape}"
        )
    if not image.dtype == reference_image.dtype:
        raise ValueError(
            f"Expected image and reference_image to have the same dtype, got {image.dtype} and {reference_image.dtype}"
        )

    if mask is None:
        mask = numpy.ones_like(image, dtype=bool)
    if not isinstance(mask, numpy.ndarray):
        raise TypeError(
            f"Expected mask to be an instance of numpy.ndarray, got {type(mask)}"
        )
    if not mask.dtype == bool:
        raise TypeError(f"Expected mask to have a boolean dtype, got {mask.dtype}")
    if not mask.shape == image.shape:
        raise ValueError(
            f"Expected mask to have the same shape as image, got {mask.shape} and {image.shape}"
        )

    if not isinstance(zernike_order, Integral):
        raise TypeError(
            f"Expected zernike_order to be an integer, got {type(zernike_order)}"
        )

    if not isinstance(channel, Integral):
        raise TypeError(f"Expected channel to be an integer, got {type(channel)}")

    if disflow_parameters is not None and not isinstance(disflow_parameters, dict):
        raise TypeError(
            f"Expected disflow_parameters to be a dictionary or None, got {type(disflow_parameters)}"
        )
    if disflow_parameters is None:
        disflow_parameters = {}
    if projection_parameters is not None and not isinstance(
        projection_parameters, dict
    ):
        raise TypeError(
            f"Expected projection_parameters to be a dictionary or None, got {type(projection_parameters)}"
        )
    if projection_parameters is None:
        projection_parameters = {}
    if sdic_parameters is not None and not isinstance(sdic_parameters, dict):
        raise TypeError(
            f"Expected sdic_parameters to be a dictionary or None, got {type(sdic_parameters)}"
        )
    if sdic_parameters is None:
        sdic_parameters = {}

    if not isinstance(verbose_level, Integral):
        raise TypeError(
            f"Expected verbose_level to be an integer, got {type(verbose_level)}"
        )

    # ----------------------
    # 1. DISFLOW COMPUTATION
    # ----------------------
    flow_x, flow_y = compute_optical_flow(
        image, reference_image, channel=channel, disflow_params=disflow_parameters
    )  # shape (H, W)

    # ----------------------
    # 2. PYCVCAM PROJECTION
    # ----------------------
    shape = image.shape
    width, height = shape[1], shape[0]
    center = (width - 1) / 2, (height - 1) / 2
    radius = numpy.sqrt((width - 1) / 2) ** 2 + (height - 1) / 2**2

    distortion = pycvcam.ZernikeDistortion(Nzer=zernike_order)
    distortion.center = center
    distortion.radius = radius

    pixel_points = numpy.indices(
        (height, width), dtype=numpy.float64
    )  # shape (2, H, W)
    image_points = pixel_points.reshape(
        2, -1
    ).T  # shape (H*W, 2) WARNING: [H, W -> Y, X]
    image_points = pixel_points[:, [1, 0]]  # Swap to [X, Y] format

    flow_x = flow_x.reshape(-1)  # shape (H*W,)
    flow_y = flow_y.reshape(-1)  # shape (H*W,)
    flow = numpy.stack([flow_x, flow_y], axis=1)  # shape (H*W, 2)

    distorted_points = image_points + flow  # shape (H*W, 2)

    masked_image_points = image_points[mask.flatten()]
    masked_flow_distorted_points = distorted_points[mask.flatten()]

    max_iter = projection_parameters.get("max_iter", 5)
    max_time = projection_parameters.get("max_time", None)
    cond_cutoff = projection_parameters.get("cond_cutoff", None)
    eps_threshold = projection_parameters.get("eps_threshold", None)
    delta_p_threshold = projection_parameters.get("delta_p_threshold", None)
    eps_sum_treshold = projection_parameters.get("eps_sum_threshold", None)
    gradient_threshold = projection_parameters.get("gradient_threshold", None)

    optimized_parameters = pycvcam.optimize.optimize_parameters(
        transform=distortion,
        input_points=masked_image_points,
        output_points=masked_flow_distorted_points,
        max_iter=max_iter,
        guess=numpy.zeros_like(distortion.parameters),
        verbose_level=verbose_level,
        max_time=max_time,
        cond_cutoff=cond_cutoff,
        eps_threshold=eps_threshold,
        delta_p_threshold=delta_p_threshold,
        eps_sum_threshold=eps_sum_treshold,
        gradient_threshold=gradient_threshold,
    )

    distortion.parameters = optimized_parameters

    # ----------------------
    # 3. SDIC CALIBRATION
    # ----------------------
    reference_image = Image.from_array(reference_image)
    image = Image.from_array(image)
    points = masked_image_points

    max_iter = sdic_parameters.get("max_iter", 5)
    delta_p_threshold = sdic_parameters.get("delta_p_threshold", None)
    relative_continuation_threshold = sdic_parameters.get(
        "relative_continuation_threshold", None
    )

    gl_reference = reference_image.evaluate_image_at_image_points(
        points
    )  # shape (H*W, C)

    for iter in tqdm.tqdm(
        range(max_iter),
        desc="Calibrate SDIC Zernike Fronto Distortion",
        disable=verbose_level < 1,
    ):

        distorted_points, _, jddis = distortion._transform(
            points, dx=False, dp=True
        )  # shape (H*W, 2), (H*W, 2, N_parameters)
        distorted_points_mask = numpy.all(
            (distorted_points >= 0) & (distorted_points < numpy.array(shape)[::-1]),
            axis=1,
        )  # shape (H*W,)
        distorted_points = distorted_points[distorted_points_mask]  # shape (F, 2)
        jddis = jddis[distorted_points_mask]  # shape (F, 2, N_parameters)

        gl_distorted = image.evaluate_image_at_image_points(
            distorted_points
        )  # shape (F, C)
        gl_distorted = gl_distorted.ravel()  # shape (F*C,)
        jdx_distorted = image.evaluate_image_jacobian_dx_at_image_points(
            distorted_points
        )  # shape (F, C)
        jdx_distorted = jdx_distorted.ravel()  # shape (F*C, )
        jdy_distorted = image.evaluate_image_jacobian_dy_at_image_points(
            distorted_points
        )  # shape (F, C)
        jdy_distorted = jdy_distorted.ravel()  # shape (F*C, )

        residual = (
            gl_distorted - gl_reference[distorted_points_mask].ravel()
        )  # shape (F*C,)
