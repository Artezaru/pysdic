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
from typing import Sequence

import numpy

from ..transformations.rays import Rays


def intersect_rays(
    seq_rays: Sequence[Rays],
) -> numpy.ndarray:
    r"""
    Compute the point closest to several rays in the least squares sense (for example to triangulate
    points seen by several cameras, see :func:`pysdic.compute_rays`).

    Lets consider :math:`N` rays with origins :math:`\vec{O}_i` and unit directions :math:`\vec{d}_i`.
    The distance between a point :math:`\vec{X}` and the ray :math:`i` is the norm of the component of
    :math:`\vec{X} - \vec{O}_i` orthogonal to :math:`\vec{d}_i`, so the searched point is:

    .. math::

        \vec{X} = \arg\min_{\vec{X}} \sum_{i=1}^{N} \left\| M_i (\vec{X} - \vec{O}_i) \right\|^2
        \quad \text{with} \quad
        M_i = I - \vec{d}_i \vec{d}_i^T

    Since :math:`M_i` is a projection (:math:`M_i^T M_i = M_i`), the solution is given by the
    :math:`3 \times 3` linear system:

    .. math::

        \left( \sum_i M_i \right) \vec{X} = \sum_i M_i \vec{O}_i

    With two rays, :math:`\vec{X}` is the middle of their common perpendicular.

    The computation is done independently for each point of the batch: ``seq_rays[i]`` contains the
    ray :math:`i` of all the points, with the same batch shape for all the elements of ``seq_rays``.

    .. note::

        If the rays of a point are parallel (or contain NaN values), the system is singular and the
        point is set to NaN.

    .. seealso::

        - Class :class:`pysdic.Rays` for the structure of the rays.
        - Function :func:`pysdic.compute_rays` to compute the rays of a camera.
        - Function :func:`pysdic.optimize_chains_transformation_input_points` to refine the points by
          minimizing the reprojection errors in the images.


    Parameters
    ----------
    seq_rays : Sequence[:class:`pysdic.Rays`]
        At least two :class:`pysdic.Rays` objects with the same batch shape :math:`(\dots)`.


    Returns
    -------
    numpy.ndarray
        The intersection points in world coordinates with shape :math:`(\dots, 3)`.


    Examples
    --------
    Triangulate the points seen by two cameras:

    .. code-block:: python

        rays_0 = compute_rays(image_points_0, intrinsic_0, distortion_0, extrinsic_0)
        rays_1 = compute_rays(image_points_1, intrinsic_1, distortion_1, extrinsic_1)

        world_points = intersect_rays([rays_0, rays_1])  # (..., 3)

    """
    if not isinstance(seq_rays, Sequence):
        raise TypeError(
            f"Input 'seq_rays' must be a sequence of Rays objects. Got type {type(seq_rays)}."
        )
    if len(seq_rays) < 2:
        raise ValueError(
            f"Input 'seq_rays' must contain at least two Rays objects. Got {len(seq_rays)}."
        )
    for i, rays in enumerate(seq_rays):
        if not isinstance(rays, Rays):
            raise TypeError(f"Input 'seq_rays[{i}]' must be a Rays object. Got type {type(rays)}.")
    batch_shape = seq_rays[0].batch_shape
    if not all(rays.batch_shape == batch_shape for rays in seq_rays):
        raise ValueError(
            f"All the Rays objects must have the same batch shape, got {[rays.batch_shape for rays in seq_rays]}."
        )

    A = numpy.zeros((*batch_shape, 3, 3), dtype=float)  # sum of M_i
    b = numpy.zeros((*batch_shape, 3), dtype=float)  # sum of M_i O_i

    for rays in seq_rays:
        d = rays.directions / numpy.linalg.norm(rays.directions, axis=-1, keepdims=True)
        M = numpy.eye(3) - d[..., :, None] * d[..., None, :]  # (..., 3, 3)
        A += M
        b += (M @ rays.origins[..., None])[..., 0]

    # Singular systems (parallel rays) or invalid values are set to NaN
    A = A.reshape(-1, 3, 3)
    b = b.reshape(-1, 3)
    points = numpy.full(b.shape, numpy.nan)
    valid = numpy.all(numpy.isfinite(A), axis=(1, 2)) & numpy.all(numpy.isfinite(b), axis=1)
    valid[valid] = numpy.linalg.cond(A[valid]) < 1.0 / numpy.finfo(float).eps
    points[valid] = numpy.linalg.solve(A[valid], b[valid, :, None])[..., 0]

    return points.reshape(*batch_shape, 3)