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
from dataclasses import dataclass
from typing import Tuple

import numpy
from numpy.typing import ArrayLike

from .._validation import _as_batch_array


@dataclass(slots=True, frozen=True)
class Rays:
    r"""
    A collection of rays in 3D space, expressed in world coordinates.

    This class is returned by the ``compute_rays`` method of the extrinsic transformations
    (for example :class:`pysdic.PerspectiveExtrinsic`) and by the function :func:`pysdic.compute_rays`.
    Each ray passes through a normalized point of the camera.

    The rays are stored in an array with shape :math:`(\dots, 6)`, where the last dimension
    encodes the **origin** and the unit **direction** of each ray:

    .. code-block:: text

        rays[..., 0:3]    # origin (x, y, z)
        rays[..., 3:6]    # direction (dx, dy, dz)

    The points of a ray are :math:`\vec{X}(t) = \vec{O} + t \, \vec{d}` with :math:`t \geq 0`.

    .. note::

        The array is copied and read-only, so the rays cannot be modified after their creation.

    Parameters
    ----------
    rays : ArrayLike
        The rays in world coordinates. Shape :math:`(\dots, 6)`.

    """

    rays: numpy.ndarray

    def __post_init__(self) -> None:
        rays = _as_batch_array(
            self.rays, "rays", 6, numpy.float64, finite=False, min_ndim=1, copy=True
        )
        rays.flags.writeable = False
        object.__setattr__(self, "rays", rays)  # Frozen dataclass

    @property
    def batch_shape(self) -> Tuple[int, ...]:
        r"""
        [Get] the batch shape of the rays, i.e. the shape of the array without the last dimension.

        Returns
        -------
        Tuple[int, ...]
            The batch shape :math:`(\dots)`.
        """
        return self.rays.shape[:-1]

    @property
    def origins(self) -> numpy.ndarray:
        r"""
        [Get] the origins :math:`\vec{O}` of the rays in world coordinates (read-only view).

        Returns
        -------
        numpy.ndarray
            The origins of the rays. Shape :math:`(\dots, 3)`.
        """
        return self.rays[..., :3]

    @property
    def directions(self) -> numpy.ndarray:
        r"""
        [Get] the unit directions :math:`\vec{d}` of the rays in world coordinates (read-only view).

        Returns
        -------
        numpy.ndarray
            The directions of the rays. Shape :math:`(\dots, 3)`.
        """
        return self.rays[..., 3:6]

    def points_at(self, t: ArrayLike) -> numpy.ndarray:
        r"""
        Compute the points of the rays at the abscissa :math:`t`:

        .. math::

            \vec{X}(t) = \vec{O} + t \, \vec{d}

        Since the directions are unit vectors, :math:`t` is the distance to the origin.

        Parameters
        ----------
        t : ArrayLike
            The abscissa, a scalar or an array broadcastable to the batch shape of the rays.


        Returns
        -------
        numpy.ndarray
            The points in world coordinates. Shape :math:`(\dots, 3)`.
        """
        t = numpy.asarray(t, dtype=float)
        return self.origins + t[..., None] * self.directions