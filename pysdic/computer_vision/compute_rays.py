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

from numpy.typing import ArrayLike

from ..transformations.rays import Rays
from ..transformations.transformation import Transformation
from ..transformations.no_transformation import NoExtrinsic

from .distorting_points import undistort_points
from .._validation import _validate_kwargs


def compute_rays(
    image_points: ArrayLike,
    intrinsic: Optional[Transformation],
    distortion: Optional[Transformation],
    extrinsic: Optional[Transformation],
    *,
    intrinsic_kwargs: Optional[Dict[str, Any]] = None,
    distortion_kwargs: Optional[Dict[str, Any]] = None,
    extrinsic_kwargs: Optional[Dict[str, Any]] = None,
) -> Rays:
    r"""
    Compute the rays emitted from the camera to the scene through the given **image points**.

    .. math::

        \vec{x}_d &= \text{Intrinsic}^{-1}(\vec{x}_i) \\
        \vec{x}_n &= \text{Distortion}^{-1}(\vec{x}_d) \\
        \text{rays} &= \text{Extrinsic.compute\_rays}(\vec{x}_n)

    Where:

    - :math:`\vec{x}_i` are the 2D **image points** in the image coordinate system.
    - :math:`\vec{x}_d` are the 2D **distorted points** in the normalized coordinate system.
    - :math:`\vec{x}_n` are the 2D **normalized points** in the normalized coordinate system.

    The normalized points are computed with :func:`pysdic.undistort_points`, then the extrinsic
    transformation builds the rays in the world coordinate system with its ``compute_rays`` method
    (the origin and the unit direction of each ray, see :class:`pysdic.Rays`).

    .. important::

        The extrinsic transformation must implement a ``compute_rays(normalized_points, **kwargs)`` method
        returning a :class:`pysdic.Rays` object, as :class:`pysdic.PerspectiveExtrinsic`,
        :class:`pysdic.OrthographicExtrinsic` and :class:`pysdic.NoExtrinsic`.

    .. note::

        The **image points** can be obtained from the **pixel points** :math:`(u, v)` by swapping the axes.

    .. seealso::

        - Class :class:`pysdic.Rays` for the structure of the rays.
        - Function :func:`pysdic.undistort_points` for the computation of the normalized points.


    Parameters
    ----------
    image_points : ArrayLike
        The 2D image points in the image coordinate system. Shape :math:`(\dots, 2)`.

    intrinsic : Optional[Transformation]
        The intrinsic transformation (2D to 2D), inverted on the image points.
        If None, no intrinsic is applied (identity).

    distortion : Optional[Transformation]
        The distortion transformation (2D to 2D), inverted on the distorted points.
        If None, no distortion is applied (identity).

    extrinsic : Optional[Transformation]
        The extrinsic transformation with a ``compute_rays`` method.
        If None, :class:`pysdic.NoExtrinsic` is used (rays parallel to the :math:`z` axis).

    intrinsic_kwargs : Optional[Dict[str, Any]], optional (default: ``None``)
        Additional keyword arguments passed to the ``invert`` method of the intrinsic transformation.

    distortion_kwargs : Optional[Dict[str, Any]], optional (default: ``None``)
        Additional keyword arguments passed to the ``invert`` method of the distortion transformation.

    extrinsic_kwargs : Optional[Dict[str, Any]], optional (default: ``None``)
        Additional keyword arguments passed to the ``compute_rays`` method of the extrinsic transformation.


    Returns
    -------
    :class:`pysdic.Rays`
        The rays in the world coordinate system, with shape :math:`(\dots, 6)`.


    Examples
    --------
    Throw a ray through each pixel of an image:

    .. code-block:: python

        import numpy
        from pysdic import compute_rays, PinholeIntrinsic, BrownDistortion, PerspectiveExtrinsic

        height, width = 480, 640
        intrinsic = PinholeIntrinsic.from_matrix([[1000.0, 0.0, width / 2],
                                                  [0.0, 1000.0, height / 2],
                                                  [0.0, 0.0, 1.0]])
        distortion = BrownDistortion([0.1, -0.05, 0.0, 0.0, 0.0])
        extrinsic = PerspectiveExtrinsic.from_rt(rvec=[0.1, 0.2, 0.3], tvec=[0.0, 0.0, 5.0])

        y, x = numpy.indices((height, width))
        image_points = numpy.stack((x, y), axis=-1)  # (H, W, 2) with (x, y) = (column, row)

        rays = compute_rays(image_points, intrinsic, distortion, extrinsic)
        rays.origins     # (H, W, 3)
        rays.directions  # (H, W, 3)

    """
    extrinsic_kwargs = _validate_kwargs(extrinsic_kwargs, "extrinsic_kwargs")
    if extrinsic is None:
        extrinsic = NoExtrinsic(input_dim=3)
    if not isinstance(extrinsic, Transformation):
        raise TypeError(
            f"Input 'extrinsic' must be an instance of Transformation or None. Got type {type(extrinsic)}."
        )
    if not callable(getattr(extrinsic, "compute_rays", None)):
        raise TypeError(
            f"The extrinsic transformation ({extrinsic.__class__.__name__}) must implement "
            f"a 'compute_rays' method."
        )
    if not extrinsic.is_setup:
        raise ValueError(
            f"The parameters of the extrinsic transformation ({extrinsic.__class__.__name__}) are not set."
        )

    normalized_points = undistort_points(
        image_points,
        intrinsic,
        distortion,
        intrinsic_kwargs=intrinsic_kwargs,
        distortion_kwargs=distortion_kwargs,
    ).x  # (..., 2)

    rays = extrinsic.compute_rays(normalized_points, **extrinsic_kwargs)
    if not isinstance(rays, Rays):
        raise TypeError(
            f"The 'compute_rays' method of {extrinsic.__class__.__name__} must return a Rays object, "
            f"got {type(rays)}."
        )
    return Rays(rays.rays.reshape(*normalized_points.shape[:-1], 6))  # Batch shape of the image points