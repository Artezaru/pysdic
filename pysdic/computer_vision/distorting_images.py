"""
pysdic - Python Stereo-Digital Image Correlation (Computer Vision
Cameras Models, Image Processing, Meshes and tools for SDIC analysis)
Copyright (C) 2026 Artezaru, artezaru.github@proton.me

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from __future__ import annotations
from typing import Any, Dict, Optional

import numpy
import cv2
import scipy.interpolate
import scipy.spatial

from ..transformations.transformation import Transformation
from .distorting_points import distort_points, undistort_points
from .._optionals import import_optional
from .._validation import _validate_choice, _validate_kwargs


# Interpolations available to sample the source image on a regular grid (remap).
# The values are the names of the cv2 flags, read when cv2 is imported (cv2 is optional).
_REMAP_INTERPOLATIONS = {
    "linear": "INTER_LINEAR",
    "nearest": "INTER_NEAREST",
    "cubic": "INTER_CUBIC",
    "lanczos4": "INTER_LANCZOS4",
    "spline3": None,  # scipy.interpolate.RectBivariateSpline (no cv2 needed)
}

# Interpolations available to sample a cloud of points (scatter)
_SCATTER_INTERPOLATIONS = ("linear", "nearest", "clough")

# Image types supported by cv2.remap
_CV2_DTYPES = (numpy.uint8, numpy.uint16, numpy.int16, numpy.float32, numpy.float64)


# ======================================================================
# Internal helpers
# ======================================================================
def _validate_image(src: numpy.ndarray) -> numpy.ndarray:
    # Check the image with shape (H, W, [C], [D])
    if not isinstance(src, numpy.ndarray):
        raise TypeError(f"Input 'src' must be a numpy array. Got type {type(src)}.")
    if src.ndim < 2 or src.ndim > 4:
        raise ValueError(f"Input 'src' must have the shape (H, W, [C], [D]). Got shape {src.shape}.")
    if not (numpy.issubdtype(src.dtype, numpy.integer) or numpy.issubdtype(src.dtype, numpy.floating)):
        raise TypeError(f"Input 'src' must contain integer or floating values. Got dtype {src.dtype}.")
    return src


def _pixel_grid(height: int, width: int) -> numpy.ndarray:
    # Image points (x, y) of all the pixels, shape (H * W, 2). Pixel [i, j] is at (x, y) = (j, i).
    y, x = numpy.indices((height, width), dtype=float)
    return numpy.stack((x.ravel(), y.ravel()), axis=-1)


def _remap(src: numpy.ndarray, map_points: numpy.ndarray, interpolation: str) -> numpy.ndarray:
    # dst[i, j] = src at the image point map_points[i, j] = (x, y), fill value 0.
    height, width = src.shape[:2]
    values = src.reshape(height, width, -1)  # (H, W, C * D)

    if interpolation != "spline3":
        if src.dtype.type not in _CV2_DTYPES:
            raise TypeError(
                f"src must have one of the types {[d.__name__ for d in _CV2_DTYPES]} for the "
                f"interpolation '{interpolation}' (cv2.remap), got {src.dtype}."
            )
        map_x = map_points[..., 0].astype(numpy.float32)
        map_y = map_points[..., 1].astype(numpy.float32)
        channels = []
        for start in range(0, values.shape[-1], 4):  # cv2.remap handles at most 4 channels
            channels.append(
                cv2.remap(
                    numpy.ascontiguousarray(values[..., start : start + 4]),
                    map_x,
                    map_y,
                    interpolation=getattr(cv2, _REMAP_INTERPOLATIONS[interpolation]),
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                ).reshape(height, width, -1)
            )
        return numpy.concatenate(channels, axis=-1).reshape(src.shape)

    # Cubic spline on the regular grid of pixels
    output = numpy.zeros(values.shape, dtype=float)
    x, y = map_points[..., 0], map_points[..., 1]
    mask = (
        numpy.isfinite(x)
        & numpy.isfinite(y)
        & (0.0 <= x) & (x <= width - 1.0)
        & (0.0 <= y) & (y <= height - 1.0)
    )
    for c in range(values.shape[-1]):
        spline = scipy.interpolate.RectBivariateSpline(
            numpy.arange(height), numpy.arange(width), values[..., c].astype(float), kx=3, ky=3, s=0
        )
        output[mask, c] = spline.ev(y[mask], x[mask])
    return output.reshape(src.shape)


def _scatter(src: numpy.ndarray, points: numpy.ndarray, interpolation: str) -> numpy.ndarray:
    # Each pixel of src is moved to the image point points[k] = (x, y); the output image is
    # interpolated on the regular grid of pixels from this cloud of points, fill value 0.
    height, width = src.shape[:2]
    values = src.reshape(height * width, -1).astype(float)  # (H * W, C * D)
    mask = numpy.all(numpy.isfinite(points), axis=-1)
    triangulation = scipy.spatial.Delaunay(points[mask])
    grid = _pixel_grid(height, width)

    if interpolation == "linear":
        interpolator = scipy.interpolate.LinearNDInterpolator(triangulation, values[mask], fill_value=0.0)
        output = interpolator(grid)
    elif interpolation == "clough":
        interpolator = scipy.interpolate.CloughTocher2DInterpolator(triangulation, values[mask], fill_value=0.0)
        output = interpolator(grid)
    else:  # nearest, with 0 outside the cloud of points as for the other methods
        interpolator = scipy.interpolate.NearestNDInterpolator(points[mask], values[mask])
        output = interpolator(grid).reshape(height * width, -1)
        output[triangulation.find_simplex(grid) < 0] = 0.0
    return output.reshape(src.shape)


# ======================================================================
# Public functions
# ======================================================================
def undistort_image(
    src: numpy.ndarray,
    intrinsic: Optional[Transformation],
    distortion: Optional[Transformation],
    interpolation: str = "linear",
    *,
    intrinsic_kwargs: Optional[Dict[str, Any]] = None,
    inverse_intrinsic_kwargs: Optional[Dict[str, Any]] = None,
    distortion_kwargs: Optional[Dict[str, Any]] = None,
) -> numpy.ndarray:
    r"""
    Undistort an image with the camera intrinsic and distortion transformations.

    This function follows the same architecture as ``cv2.undistort``, but works with any
    intrinsic and distortion transformations. For each pixel :math:`\vec{x}_i` of the
    **undistorted image**, the position in the **source image** is:

    .. math::

        \vec{x}_{src} = \text{Intrinsic}(\text{Distortion}(\text{Intrinsic}^{-1}(\vec{x}_i)))

    computed with :func:`pysdic.distort_points` (``P=intrinsic``), and the source image is sampled
    at these positions. Only direct transformations are used, so no iterative inversion is needed.

    The image ``src`` has the shape :math:`(H, W, [C], [D])`. The pixel :math:`[i, j]` is at the
    image point :math:`(x, y) = (j, i)`. The pixels sampled outside the source image are set to 0.

    The available interpolations are:

    +----------------+---------------------------------------------------------------------+
    | Interpolation  | Description                                                         |
    +================+=====================================================================+
    | ``"linear"``   | Bilinear interpolation (default), ``cv2.INTER_LINEAR``.             |
    +----------------+---------------------------------------------------------------------+
    | ``"nearest"``  | Nearest neighbor interpolation, ``cv2.INTER_NEAREST``.              |
    +----------------+---------------------------------------------------------------------+
    | ``"cubic"``    | Bicubic interpolation, ``cv2.INTER_CUBIC``.                         |
    +----------------+---------------------------------------------------------------------+
    | ``"lanczos4"`` | Lanczos interpolation over 8x8 pixels, ``cv2.INTER_LANCZOS4``.      |
    +----------------+---------------------------------------------------------------------+
    | ``"spline3"``  | Cubic spline, ``scipy.interpolate.RectBivariateSpline``.            |
    +----------------+---------------------------------------------------------------------+

    .. note::

        - The ``cv2`` interpolations keep the type of the image, which must be ``uint8``, ``uint16``,
          ``int16``, ``float32`` or ``float64``.
        - The ``"spline3"`` interpolation accepts any numeric type and returns a ``float64`` image.

    .. seealso::

        - Function :func:`pysdic.distort_image` for the opposite operation.
        - Function :func:`pysdic.distort_points` to compute the positions of the pixels.


    Parameters
    ----------
    src : numpy.ndarray
        The image to undistort, with shape :math:`(H, W, [C], [D])`.

    intrinsic : Optional[Transformation]
        The intrinsic transformation (2D to 2D). If None, no intrinsic is applied and the
        distortion is defined in the image coordinate system.

    distortion : Optional[Transformation]
        The distortion transformation (2D to 2D). If None, no distortion is applied.

    interpolation : str, optional (default: ``"linear"``)
        The interpolation method (see the table above).

    intrinsic_kwargs : Optional[Dict[str, Any]], optional (default: ``None``)
        Additional keyword arguments passed to the ``apply`` method of the intrinsic transformation.

    inverse_intrinsic_kwargs : Optional[Dict[str, Any]], optional (default: ``None``)
        Additional keyword arguments passed to the ``invert`` method of the intrinsic transformation.

    distortion_kwargs : Optional[Dict[str, Any]], optional (default: ``None``)
        Additional keyword arguments passed to the ``apply`` method of the distortion transformation.


    Returns
    -------
    numpy.ndarray
        The undistorted image with shape :math:`(H, W, [C], [D])`.


    Examples
    --------
    .. code-block:: python

        import cv2
        from pysdic import undistort_image, PinholeIntrinsic, BrownDistortion

        intrinsic = PinholeIntrinsic.from_matrix([[1000.0, 0.0, 320.0],
                                                  [0.0, 1000.0, 240.0],
                                                  [0.0, 0.0, 1.0]])
        distortion = BrownDistortion([0.1, -0.05, 0.001, 0.002, 0.01])

        src = cv2.imread("image.jpg")
        undistorted_image = undistort_image(src, intrinsic, distortion)

    """
    src = _validate_image(src)
    interpolation = _validate_choice(interpolation, "interpolation", tuple(_REMAP_INTERPOLATIONS))
    intrinsic_kwargs = _validate_kwargs(intrinsic_kwargs, "intrinsic_kwargs")
    inverse_intrinsic_kwargs = _validate_kwargs(inverse_intrinsic_kwargs, "inverse_intrinsic_kwargs")
    distortion_kwargs = _validate_kwargs(distortion_kwargs, "distortion_kwargs")

    height, width = src.shape[:2]
    map_points = distort_points(
        _pixel_grid(height, width),
        intrinsic,
        distortion,
        P=intrinsic,
        intrinsic_kwargs=inverse_intrinsic_kwargs,
        distortion_kwargs=distortion_kwargs,
        P_kwargs=intrinsic_kwargs,
    ).x.reshape(height, width, 2)

    return _remap(src, map_points, interpolation)


def distort_image(
    src: numpy.ndarray,
    intrinsic: Optional[Transformation],
    distortion: Optional[Transformation],
    method: str = "undistort",
    interpolation: str = "linear",
    *,
    intrinsic_kwargs: Optional[Dict[str, Any]] = None,
    inverse_intrinsic_kwargs: Optional[Dict[str, Any]] = None,
    distortion_kwargs: Optional[Dict[str, Any]] = None,
    inverse_distortion_kwargs: Optional[Dict[str, Any]] = None,
) -> numpy.ndarray:
    r"""
    Distort an image with the camera intrinsic and distortion transformations.

    This is the opposite of :func:`pysdic.undistort_image`: the source image is a perfect (undistorted)
    image and the output is the image seen through the distortion. Two methods are available.

    **Method** ``"undistort"`` **(default)**: for each pixel :math:`\vec{x}_i` of the **distorted image**,
    the position in the **source image** is:

    .. math::

        \vec{x}_{src} = \text{Intrinsic}(\text{Distortion}^{-1}(\text{Intrinsic}^{-1}(\vec{x}_i)))

    computed with :func:`pysdic.undistort_points` (``P=intrinsic``), and the source image is sampled
    on its regular grid of pixels (fast). The inverse of the distortion is usually computed with an
    iterative method, whose options can be given with ``inverse_distortion_kwargs``.
    The interpolations are the same as for :func:`pysdic.undistort_image`
    (``"linear"``, ``"nearest"``, ``"cubic"``, ``"lanczos4"``, ``"spline3"``).

    **Method** ``"distort"``: each pixel of the **source image** is moved to its distorted position:

    .. math::

        \vec{x}_{dst} = \text{Intrinsic}(\text{Distortion}(\text{Intrinsic}^{-1}(\vec{x}_{src})))

    computed with :func:`pysdic.distort_points` (``P=intrinsic``), and the distorted image is interpolated
    on its grid of pixels from this irregular cloud of points (slow, but no inverse of the distortion is needed).
    The available interpolations are:

    +----------------+----------------------------------------------------------------------------+
    | Interpolation  | Description                                                                |
    +================+============================================================================+
    | ``"linear"``   | Linear interpolation (default), ``scipy.interpolate.LinearNDInterpolator``.|
    +----------------+----------------------------------------------------------------------------+
    | ``"nearest"``  | Nearest neighbor, ``scipy.interpolate.NearestNDInterpolator``.             |
    +----------------+----------------------------------------------------------------------------+
    | ``"clough"``   | Piecewise cubic, ``scipy.interpolate.CloughTocher2DInterpolator``.         |
    +----------------+----------------------------------------------------------------------------+

    The image ``src`` has the shape :math:`(H, W, [C], [D])`. The pixel :math:`[i, j]` is at the
    image point :math:`(x, y) = (j, i)`. The pixels outside the source image (or outside the cloud of
    points) are set to 0.

    .. note::

        - The ``cv2`` interpolations keep the type of the image, which must be ``uint8``, ``uint16``,
          ``int16``, ``float32`` or ``float64``.
        - The ``"spline3"`` interpolation and the method ``"distort"`` accept any numeric type and
          return a ``float64`` image.

    .. seealso::

        - Function :func:`pysdic.undistort_image` for the opposite operation.


    Parameters
    ----------
    src : numpy.ndarray
        The image to distort, with shape :math:`(H, W, [C], [D])`.

    intrinsic : Optional[Transformation]
        The intrinsic transformation (2D to 2D). If None, no intrinsic is applied and the
        distortion is defined in the image coordinate system.

    distortion : Optional[Transformation]
        The distortion transformation (2D to 2D). If None, no distortion is applied.

    method : str, optional (default: ``"undistort"``)
        The method, ``"undistort"`` or ``"distort"``.

    interpolation : str, optional (default: ``"linear"``)
        The interpolation method, depending on the method (see above).

    intrinsic_kwargs : Optional[Dict[str, Any]], optional (default: ``None``)
        Additional keyword arguments passed to the ``apply`` method of the intrinsic transformation.

    inverse_intrinsic_kwargs : Optional[Dict[str, Any]], optional (default: ``None``)
        Additional keyword arguments passed to the ``invert`` method of the intrinsic transformation.

    distortion_kwargs : Optional[Dict[str, Any]], optional (default: ``None``)
        Additional keyword arguments passed to the ``apply`` method of the distortion transformation
        (method ``"distort"``).

    inverse_distortion_kwargs : Optional[Dict[str, Any]], optional (default: ``None``)
        Additional keyword arguments passed to the ``invert`` method of the distortion transformation
        (method ``"undistort"``).


    Returns
    -------
    numpy.ndarray
        The distorted image with shape :math:`(H, W, [C], [D])`.


    Examples
    --------
    .. code-block:: python

        distorted_image = distort_image(src, intrinsic, distortion)
        distorted_image = distort_image(src.astype(float), intrinsic, distortion, method="distort")

    """
    src = _validate_image(src)
    method = _validate_choice(method, "method", ("undistort", "distort"))
    available = tuple(_REMAP_INTERPOLATIONS) if method == "undistort" else _SCATTER_INTERPOLATIONS
    interpolation = _validate_choice(interpolation, "interpolation", available)
    intrinsic_kwargs = _validate_kwargs(intrinsic_kwargs, "intrinsic_kwargs")
    inverse_intrinsic_kwargs = _validate_kwargs(inverse_intrinsic_kwargs, "inverse_intrinsic_kwargs")
    distortion_kwargs = _validate_kwargs(distortion_kwargs, "distortion_kwargs")
    inverse_distortion_kwargs = _validate_kwargs(inverse_distortion_kwargs, "inverse_distortion_kwargs")

    height, width = src.shape[:2]
    function = undistort_points if method == "undistort" else distort_points
    points = function(
        _pixel_grid(height, width),
        intrinsic,
        distortion,
        P=intrinsic,
        intrinsic_kwargs=inverse_intrinsic_kwargs,
        distortion_kwargs=inverse_distortion_kwargs if method == "undistort" else distortion_kwargs,
        P_kwargs=intrinsic_kwargs,
    ).x  # (H * W, 2)

    if method == "undistort":
        return _remap(src, points.reshape(height, width, 2), interpolation)
    return _scatter(src, points, interpolation)