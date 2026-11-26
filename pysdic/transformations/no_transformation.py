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
from typing import Any, Dict, Optional, Tuple
from numbers import Integral
from numpy.typing import ArrayLike

import numpy

from .identity_transformation import IdentityTransformation
from .rays import Rays
from .._validation import _validate_integer, _as_scalar_or_1d_array


class NoIntrinsic(IdentityTransformation):
    r"""
    Subclass of :class:`pysdic.IdentityTransformation` that represents a
    no-intrinsic transformation (identity transformation of the 2D points).

    .. math::

        \vec{x}_i = \vec{x}_d

    This can be used as a default or placeholder intrinsic transformation when no
    intrinsic is present in the camera model. The image points are then equal to the
    distorted points.

    .. seealso::

        - Class :class:`pysdic.IdentityTransformation` for the identity transformation.

    """

    def __init__(self) -> None:
        super().__init__(input_dim=2, output_dim=2)

    def _to_dict(self) -> Dict[str, Any]:
        return {}

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> NoIntrinsic:
        return cls()

    def _copy(self) -> NoIntrinsic:
        return self.__class__()


class NoDistortion(IdentityTransformation):
    r"""
    Subclass of :class:`pysdic.IdentityTransformation` that represents a
    no-distortion transformation (identity transformation of the 2D points).

    .. math::

        \vec{x}_d = \vec{x}_n

    This can be used as a default or placeholder distortion transformation when no
    distortion is present in the camera model. The distorted points are then equal to the
    normalized points.

    .. seealso::

        - Class :class:`pysdic.IdentityTransformation` for the identity transformation.

    """

    def __init__(self) -> None:
        super().__init__(input_dim=2, output_dim=2)

    def _to_dict(self) -> Dict[str, Any]:
        return {}

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> NoDistortion:
        return cls()

    def _copy(self) -> NoDistortion:
        return self.__class__()


class NoExtrinsic(IdentityTransformation):
    r"""
    Subclass of :class:`pysdic.IdentityTransformation` that represents a
    no-extrinsic transformation (no rotation and no translation).

    This can be used as a default or placeholder extrinsic transformation when no
    rotation or translation is present in the camera model. The world points are already
    expressed in the camera coordinate system:

    - With 3D world points (``input_dim=3``, default), the normalized points are the first two
      coordinates of the world points (orthographic projection with an identity pose):

      .. math::

          \vec{x}_n = (X_w, Y_w)

      As for :class:`pysdic.OrthographicExtrinsic`, the inverse transformation needs the depth
      of the points: :math:`\vec{X}_w = (x_n, y_n, Z_c)`, given by the ``depth`` argument of
      ``invert`` (default 1.0). It is a right inverse but not a left inverse.

    - With 2D world points (``input_dim=2``), the transformation is the identity of the plane:

      .. math::

          \vec{x}_n = \vec{x}_w

    .. seealso::

        - Class :class:`pysdic.IdentityTransformation` for the identity transformation.
        - Class :class:`pysdic.OrthographicExtrinsic` for the orthographic extrinsic with any pose.


    Parameters
    ----------
    input_dim : Integral, optional (default: ``3``)
        The dimension of the world points, 3 (3D world points) or 2 (planar world points).


    Examples
    --------
    .. code-block:: python

        extrinsic = NoExtrinsic()
        extrinsic.apply([[1.0, 2.0, 5.0]]).x                 # array([[1., 2.]])
        extrinsic.invert([[1.0, 2.0]], depth=5.0).x          # array([[1., 2., 5.]])

        planar = NoExtrinsic(input_dim=2)
        planar.apply([[1.0, 2.0]]).x                         # array([[1., 2.]])

    """

    def __init__(self, input_dim: Integral = 3) -> None:
        input_dim = _validate_integer(input_dim, "input_dim")
        if input_dim not in (2, 3):
            raise ValueError(f"Input 'input_dim' must be 2 or 3. Got {input_dim}.")
        super().__init__(input_dim=input_dim, output_dim=2)

    def _to_dict(self) -> Dict[str, Any]:
        return {"input_dim": self.input_dim}

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> NoExtrinsic:
        return cls(input_dim=data.get("input_dim", 3))

    def _copy(self) -> NoExtrinsic:
        return self.__class__(input_dim=self.input_dim)

    def _invert(
        self,
        normalized_points: numpy.ndarray,
        *,
        dx: bool = False,
        dp: bool = False,
        depth: Optional[ArrayLike] = None,
    ) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        r"""
        Compute the inverse transformation from 2D **normalized points** to **world points**.

        - With 3D world points, the normalized points are placed at the depth :math:`Z_c`:
          :math:`\vec{X}_w = (x_n, y_n, Z_c)`.
        - With 2D world points, the transformation is the identity.

        Parameters
        ----------
        normalized_points : numpy.ndarray
            The normalized points. Shape ``(n_points, 2)``.

        dx : bool, optional (default: ``False``)
            If True, compute the Jacobian with respect to the normalized points.

        dp : bool, optional (default: ``False``)
            If True, compute the (empty) Jacobian with respect to the parameters.

        depth : Optional[ArrayLike], optional (default: ``None``)
            Depth :math:`Z_c` of the points (only for 3D world points). It can be a scalar or an array
            with one value per point (with the same batch shape as the points passed to ``invert``).
            If None, all depths are equal to 1.0.


        Returns
        -------
        world_points : numpy.ndarray
            The world points. Shape ``(n_points, input_dim)``.

        jacobian_dx : Optional[numpy.ndarray]
            The Jacobian with respect to the normalized points.
            Shape ``(n_points, input_dim, 2)`` if ``dx=True``, otherwise ``None``.

        jacobian_dp : Optional[numpy.ndarray]
            The Jacobian with respect to the parameters.
            Shape ``(n_points, input_dim, 0)`` if ``dp=True``, otherwise ``None``.
        """
        world_points, jacobian_dx, jacobian_dp = super()._invert(
            normalized_points, dx=dx, dp=dp
        )
        if self.input_dim == 3:
            world_points[:, 2] = _as_scalar_or_1d_array(
                1.0 if depth is None else depth,
                "depth",
                normalized_points.shape[0],
                numpy.float64,
                finite=False,
            )  # Depth 1 by default
        elif depth is not None:
            raise ValueError("depth is only available with 3D world points (input_dim=3).")
        return world_points, jacobian_dx, jacobian_dp

    def compute_rays(self, normalized_points: ArrayLike) -> Rays:
        r"""
        Compute the rays of the camera passing through the given normalized points
        (only for 3D world points).

        As for an orthographic camera with an identity pose, all the rays are parallel to the
        :math:`z` axis. The origin of each ray is the normalized point on the plane :math:`Z = 0`
        and the direction is :math:`(0, 0, 1)`.

        Parameters
        ----------
        normalized_points : ArrayLike
            Normalized points. Shape ``(..., 2)``.


        Returns
        -------
        :class:`pysdic.Rays`
            Rays in world coordinates with shape ``(..., 6)`` (same batch shape as the input points),
            where the last dimension encodes the ray origin ``(x, y, z)`` and the unit direction
            ``(dx, dy, dz)``.

        Raises
        ------
        ValueError
            If the world points are 2D (``input_dim=2``).
        """
        if self.input_dim != 3:
            raise ValueError("The rays are only available with 3D world points (input_dim=3).")
        points, batch_shape = self._validate_points(
            normalized_points, self.output_dim, "normalized_points"
        )
        rays = numpy.zeros((points.shape[0], 6), dtype=float)
        rays[:, :2] = points
        rays[:, 5] = 1.0
        return Rays(rays.reshape(*batch_shape, 6))