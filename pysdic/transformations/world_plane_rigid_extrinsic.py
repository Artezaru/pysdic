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
from typing import Optional, Tuple
from numpy.typing import ArrayLike

import numpy
import cv2

from .rigid_extrinsic import RigidExtrinsic
from .rays import Rays
from .._optionals import import_optional


class WorldPlaneRigidExtrinsic(RigidExtrinsic):
    r"""
    Abstract base class of the extrinsic camera models observing the **world plane**
    :math:`Z_w = 0` with a 3D rigid motion followed by a projection onto the normalized
    camera plane.

    The world points are 2D points :math:`\vec{x}_w = (x_w, y_w)^T` of the plane :math:`Z_w = 0`.
    This allows to work with planar displacement fields (no :math:`z` component) while the camera
    is free to observe the plane with any 3D pose:

    .. code-block:: text

        world_points → normalized_points → distorted_points → image_points

    The world point is embedded in 3D with :math:`Z_w = 0` and transformed into the camera
    coordinate system by the rigid motion:

    .. math::

        \vec{X}_c = R \cdot \begin{bmatrix} x_w \\ y_w \\ 0 \end{bmatrix} + T
        = A \vec{x}_w + T
        \quad \text{with} \quad
        A = \begin{bmatrix} \vec{r}_1 & \vec{r}_2 \end{bmatrix}

    where :math:`\vec{r}_1` and :math:`\vec{r}_2` are the first two columns of :math:`R`.

    The transformation is parameterized by **6 parameters** :math:`(r_x, r_y, r_z, t_x, t_y, t_z)`,
    with the same conventions as :class:`pysdic.RigidExtrinsic` (rotation vector in Rodrigues form).
    The projection of the camera points onto the normalized points is defined by the subclasses:

    - :class:`pysdic.WorldPlanePerspectiveExtrinsic`: perspective division :math:`\vec{x}_n = (X_c / Z_c, Y_c / Z_c)`.
    - :class:`pysdic.WorldPlaneOrthographicExtrinsic`: orthographic projection :math:`\vec{x}_n = (X_c, Y_c)`.

    Unlike the extrinsics with 3D world points, no depth is lost: the inverse transformation is the
    intersection of the camera ray with the plane :math:`Z_w = 0`, so no ``depth`` argument is
    needed and the transformations are left and right invertible.

    .. seealso::

        - Class :class:`pysdic.RigidExtrinsic` for the rigid motion, its parameters and accessors
          (``rvec``, ``tvec``, ``rmat``, ``tmatrix``, ``frame``) and the construction methods.
        - Class :class:`pysdic.PlanarRigidExtrinsic` for the 2D rigid motion of the plane, without
          3D pose nor projection.
        - Class :class:`pysdic.PlanarHomographicExtrinsic` for the general projective transformations
          of the plane, without 3D pose.

    """

    @property
    def input_dim(self) -> int:
        r"""
        [Get] the input dimension of the extrinsic transformation (2D world points of the plane
        :math:`Z_w = 0`).

        Returns
        -------
        int
            The input dimension of the extrinsic transformation.

        """
        return 2

    def _plane_rotation_and_derivatives(self) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        # Return R (3, 3), A = R[:, :2] (3, 2) and dA/dr (3, 2, 3) with [i, j, k] = dA[i, j] / dr[k]
        rmat, rmat_dr = self._rotation_and_derivatives()
        return rmat, rmat[:, :2], rmat_dr[:, :2, :]


class WorldPlanePerspectiveExtrinsic(WorldPlaneRigidExtrinsic):
    r"""
    Subclass of :class:`pysdic.WorldPlaneRigidExtrinsic` implementing a perspective camera
    observing the **world plane** :math:`Z_w = 0`.

    The 2D world points are transformed into the camera coordinate system by
    :math:`\vec{X}_c = A \vec{x}_w + T` (see :class:`pysdic.WorldPlaneRigidExtrinsic`), and the
    **normalized point** is obtained by perspective division:

    .. math::

        \vec{x}_n = \frac{1}{Z_c} \begin{bmatrix} X_c \\ Y_c \end{bmatrix}

    In homogeneous coordinates, it is the plane-induced homography:

    .. math::

        \begin{bmatrix} \vec{x}_n \\ 1 \end{bmatrix} \sim H \begin{bmatrix} \vec{x}_w \\ 1 \end{bmatrix}
        \quad \text{with} \quad
        H = \begin{bmatrix} \vec{r}_1 & \vec{r}_2 & T \end{bmatrix}

    The transformation is a bijection between the part of the plane **in front of the camera**
    (:math:`Z_c > 0`) and the normalized points whose ray hits the plane in front of the camera.
    The points behind the camera are rejected by ``apply`` and ``invert``.

    .. seealso::

        - Class :class:`pysdic.PerspectiveExtrinsic` for the perspective camera with 3D world points.
        - Class :class:`pysdic.PlanarHomographicExtrinsic` for the general projective transformations of the plane.


    Parameters
    ----------
    parameters : Optional[ArrayLike], optional (default: ``None``)
        The extrinsic parameters as a 1D array of shape (6,).
        The first 3 elements represent the **rotation vector** :math:`\vec{r} = (r_x, r_y, r_z)`.
        The last 3 elements represent the **translation vector** :math:`T = (t_x, t_y, t_z)`.


    Examples
    --------
    A camera located 5 units above the plane and looking at it perpendicularly:

    .. code-block:: python

        extrinsic = WorldPlanePerspectiveExtrinsic.from_rt(
            rvec=[0.0, 0.0, 0.0],
            tvec=[0.0, 0.0, 5.0],
        )

        extrinsic.apply([[1.0, 2.0]]).x          # array([[0.2, 0.4]])
        extrinsic.invert([[0.2, 0.4]]).x         # array([[1., 2.]])

    """

    _has_apply = True
    _has_apply_dx = True
    _has_apply_dp = True
    _has_invert = True
    _has_invert_dx = True
    _has_invert_dp = True
    _is_left_invertible = True  # The depth is recovered by the intersection with the plane
    _is_right_invertible = True

    # =============================================
    # Homography
    # =============================================
    @property
    def homography_matrix(self) -> Optional[numpy.ndarray]:
        r"""
        [Get] the 3x3 plane-induced homography :math:`H = [\vec{r}_1, \vec{r}_2, T]` between the
        world plane and the normalized plane.

        .. math::

            \begin{bmatrix} \vec{x}_n \\ 1 \end{bmatrix} \sim H \begin{bmatrix} \vec{x}_w \\ 1 \end{bmatrix}

        .. note::

            An alias for ``homography_matrix`` is ``hmatrix``.

        Returns
        -------
        Optional[numpy.ndarray]
            The homography matrix with shape ``(3, 3)``. (or None if the parameters are not set)
        """
        rmat = self.rotation_matrix
        if rmat is None:
            return None
        return numpy.column_stack([rmat[:, 0], rmat[:, 1], self._parameters[3:]])

    @property
    def hmatrix(self) -> Optional[numpy.ndarray]:
        return self.homography_matrix

    # =============================================
    # Transform methods
    # =============================================
    def _apply(
        self,
        world_points: numpy.ndarray,
        *,
        dx: bool = False,
        dp: bool = False,
    ) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        r"""
        Compute the transformation from 2D **world points** of the plane :math:`Z_w = 0` to
        2D **normalized points**.

        .. math::

            \vec{X}_c = A \vec{x}_w + T,
            \qquad
            \vec{x}_n = \frac{1}{Z_c} \begin{bmatrix} X_c \\ Y_c \end{bmatrix}

        Lets :math:`D` be the Jacobian of the perspective division with respect to the camera points:

        .. math::

            D = \frac{\partial \vec{x}_n}{\partial \vec{X}_c} = \frac{1}{Z_c}
            \begin{bmatrix}
            1 & 0 & -x_n \\
            0 & 1 & -y_n
            \end{bmatrix}

        The Jacobian with respect to the world points is:

        .. math::

            \frac{\partial \vec{x}_n}{\partial \vec{x}_w} = D A

        The Jacobian with respect to the parameters :math:`(\vec{r}, T)` is:

        .. math::

            \frac{\partial \vec{x}_n}{\partial r_k} = D \frac{\partial A}{\partial r_k} \vec{x}_w,
            \qquad
            \frac{\partial \vec{x}_n}{\partial T} = D

        where :math:`\frac{\partial A}{\partial r_k}` are the first two columns of
        :math:`\frac{\partial R}{\partial r_k}` given by ``cv2.Rodrigues``.


        Parameters
        ----------
        world_points : numpy.ndarray
            Input 2D points of the world plane. Shape ``(n_points, 2)``.

        dx : bool, optional (default: ``False``)
            If True, compute the Jacobian with respect to the input world points.

        dp : bool, optional (default: ``False``)
            If True, compute the Jacobian with respect to the extrinsic parameters.


        Returns
        -------
        normalized_points : numpy.ndarray
            The normalized points in camera coordinates. Shape ``(n_points, 2)``.

        jacobian_dx : Optional[numpy.ndarray]
            Jacobian of normalized points with respect to world points.
            Shape ``(n_points, 2, 2)`` if ``dx=True``, otherwise ``None``.

        jacobian_dp : Optional[numpy.ndarray]
            Jacobian of normalized points with respect to extrinsic parameters.
            Shape ``(n_points, 2, 6)`` if ``dp=True``, otherwise ``None``.

        Raises
        ------
        ValueError
            If at least one point is behind the camera or too close to the camera plane
            (:math:`Z_c \leq 10^{-6}`).

        """
        n_points = world_points.shape[0]
        _, amat, amat_dr = self._plane_rotation_and_derivatives()
        tvec = self._parameters[3:]

        camera_points = world_points @ amat.T + tvec  # (n_points, 3)
        z_c = camera_points[:, 2]
        if numpy.any(z_c < 1e-6):
            raise ValueError(
                "At least one point is behind the camera or too close to the camera plane "
                "(Z_c <= 1e-6 in the camera coordinate system): the perspective projection is undefined."
            )
        normalized_points = camera_points[:, :2] / z_c[:, None]  # (n_points, 2)

        if not (dx or dp):
            return normalized_points, None, None

        # Jacobian of the perspective division D = d x_n / d X_c
        division_dc = numpy.zeros((n_points, 2, 3), dtype=float)  # (n_points, 2, 3)
        division_dc[:, 0, 0] = 1.0
        division_dc[:, 1, 1] = 1.0
        division_dc[:, :, 2] = -normalized_points
        division_dc /= z_c[:, None, None]

        jacobian_dx = None
        if dx:
            jacobian_dx = division_dc @ amat  # (n_points, 2, 2)

        jacobian_dp = None
        if dp:
            camera_dp = numpy.empty((n_points, 3, 6), dtype=float)  # d X_c / d p
            camera_dp[:, :, :3] = numpy.einsum("ijk,nj->nik", amat_dr, world_points)
            camera_dp[:, :, 3:] = numpy.eye(3)
            jacobian_dp = division_dc @ camera_dp  # (n_points, 2, 6)

        return normalized_points, jacobian_dx, jacobian_dp

    def _invert(
        self,
        normalized_points: numpy.ndarray,
        *,
        dx: bool = False,
        dp: bool = False,
    ) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        r"""
        Compute the inverse transformation from 2D **normalized points** to 2D **world points**
        of the plane :math:`Z_w = 0` (intersection of the camera rays with the plane).

        With the homography :math:`H = [\vec{r}_1, \vec{r}_2, T]` and
        :math:`\vec{m} = (x_n, y_n, 1)^T`:

        .. math::

            \vec{u} = H^{-1} \vec{m} = \frac{1}{Z_c} \begin{bmatrix} \vec{x}_w \\ 1 \end{bmatrix},
            \qquad
            \vec{x}_w = \frac{1}{u_3} \begin{bmatrix} u_1 \\ u_2 \end{bmatrix}

        The intersection is in front of the camera if and only if :math:`u_3 = 1 / Z_c > 0`.
        Lets :math:`D_w` be the Jacobian of the division:

        .. math::

            D_w = \frac{\partial \vec{x}_w}{\partial \vec{u}} = \frac{1}{u_3}
            \begin{bmatrix}
            1 & 0 & -x_w \\
            0 & 1 & -y_w
            \end{bmatrix}

        The Jacobian with respect to the normalized points is:

        .. math::

            \frac{\partial \vec{x}_w}{\partial \vec{x}_n} = D_w H^{-1}_{:, :2}

        The Jacobian with respect to the parameters :math:`p` uses
        :math:`\frac{\partial H^{-1}}{\partial p} = - H^{-1} \frac{\partial H}{\partial p} H^{-1}`:

        .. math::

            \frac{\partial \vec{x}_w}{\partial p} = - D_w H^{-1} \frac{\partial H}{\partial p} \vec{u}
            \quad \text{with} \quad
            \frac{\partial H}{\partial r_k} \vec{u} = \frac{\partial A}{\partial r_k} \begin{bmatrix} u_1 \\ u_2 \end{bmatrix},
            \qquad
            \frac{\partial H}{\partial T} \vec{u} = u_3 I_3


        Parameters
        ----------
        normalized_points : numpy.ndarray
            Input 2D normalized points in camera coordinates. Shape ``(n_points, 2)``.

        dx : bool, optional (default: ``False``)
            If True, compute the Jacobian with respect to normalized points.

        dp : bool, optional (default: ``False``)
            If True, compute the Jacobian with respect to extrinsic parameters.


        Returns
        -------
        world_points : numpy.ndarray
            2D points of the world plane. Shape ``(n_points, 2)``.

        jacobian_dx : Optional[numpy.ndarray]
            Jacobian of world points with respect to normalized points.
            Shape ``(n_points, 2, 2)`` if ``dx=True``, otherwise ``None``.

        jacobian_dp : Optional[numpy.ndarray]
            Jacobian of world points with respect to extrinsic parameters.
            Shape ``(n_points, 2, 6)`` if ``dp=True``, otherwise ``None``.

        Raises
        ------
        ValueError
            If the camera center lies in the plane :math:`Z_w = 0` (singular homography), or if
            the ray of at least one point does not hit the plane in front of the camera (ray
            parallel to the plane or intersection behind the camera).

        """
        n_points = normalized_points.shape[0]
        rmat, _, amat_dr = self._plane_rotation_and_derivatives()
        tvec = self._parameters[3:]

        hmat = numpy.column_stack([rmat[:, 0], rmat[:, 1], tvec])  # (3, 3)
        # det(H) = r3 . T = - (signed distance of the camera center to the plane)
        if abs(numpy.linalg.det(hmat)) < 1e-12:
            raise ValueError(
                "The camera center lies in the world plane Z_w = 0: the plane-induced "
                "homography is singular and the transformation cannot be inverted."
            )
        hmat_inv = numpy.linalg.inv(hmat)  # (3, 3)

        homogeneous = numpy.column_stack([normalized_points, numpy.ones(n_points)])  # (n_points, 3)
        u = homogeneous @ hmat_inv.T  # (n_points, 3), u = [x_w, y_w, 1] / Z_c
        u_3 = u[:, 2]
        if numpy.any(u_3 < 1e-12):
            raise ValueError(
                "The ray of at least one point does not hit the world plane Z_w = 0 in front of "
                "the camera (ray parallel to the plane or intersection behind the camera)."
            )
        world_points = u[:, :2] / u_3[:, None]  # (n_points, 2)

        if not (dx or dp):
            return world_points, None, None

        # Jacobian of the division D_w = d x_w / d u
        division_du = numpy.zeros((n_points, 2, 3), dtype=float)  # (n_points, 2, 3)
        division_du[:, 0, 0] = 1.0
        division_du[:, 1, 1] = 1.0
        division_du[:, :, 2] = -world_points
        division_du /= u_3[:, None, None]

        jacobian_dx = None
        if dx:
            jacobian_dx = division_du @ hmat_inv[:, :2]  # (n_points, 2, 2)

        jacobian_dp = None
        if dp:
            h_dp_u = numpy.zeros((n_points, 3, 6), dtype=float)  # dH/dp u
            h_dp_u[:, :, :3] = numpy.einsum("ijk,nj->nik", amat_dr, u[:, :2])
            h_dp_u[:, :, 3:] = u_3[:, None, None] * numpy.eye(3)
            jacobian_dp = -division_du @ (hmat_inv @ h_dp_u)  # (n_points, 2, 6)

        return world_points, jacobian_dx, jacobian_dp

    # ==================================================================================
    # Rays computation
    # ==================================================================================
    def compute_rays(
        self,
        normalized_points: ArrayLike,
    ) -> Rays:
        r"""
        Compute the rays of the camera passing through the given normalized points.

        The rays are the same as for :class:`pysdic.PerspectiveExtrinsic`: they start from the
        **camera center** and are expressed in the **3D** world coordinate system (the world plane
        being :math:`Z_w = 0`):

        .. math::

            \vec{O}_w = - R^T \cdot T,
            \qquad
            \vec{d}_w = \frac{R^T \vec{X}_n}{\| R^T \vec{X}_n \|}
            \quad \text{with} \quad
            \vec{X}_n = \begin{bmatrix} x_n \\ y_n \\ 1 \end{bmatrix}

        Parameters
        ----------
        normalized_points : ArrayLike
            Normalized points in the camera coordinate system. Shape ``(..., 2)``.


        Returns
        -------
        :class:`pysdic.Rays`
            Rays in 3D world coordinates with shape ``(..., 6)`` (same batch shape as the input
            points), where the last dimension encodes the ray origin ``(x, y, z)`` and the unit
            direction ``(dx, dy, dz)``.

        """
        self._check_setup_for_rays()
        points, batch_shape = self._validate_points(
            normalized_points, self.output_dim, "normalized_points"
        )  # (n_points, 2)
        n_points = points.shape[0]

        rmat = cv2.Rodrigues(self._parameters[:3])[0]
        tvec = self._parameters[3:]

        origin = -rmat.T @ tvec  # (3,)
        directions = numpy.column_stack([points, numpy.ones(n_points)]) @ rmat  # R^T X_n
        directions /= numpy.linalg.norm(directions, axis=1, keepdims=True)

        rays = numpy.empty((n_points, 6), dtype=float)
        rays[:, :3] = origin
        rays[:, 3:] = directions
        return Rays(rays.reshape(*batch_shape, 6))


class WorldPlaneOrthographicExtrinsic(WorldPlaneRigidExtrinsic):
    r"""
    Subclass of :class:`pysdic.WorldPlaneRigidExtrinsic` implementing an orthographic camera
    observing the **world plane** :math:`Z_w = 0`.

    The 2D world points are transformed into the camera coordinate system by
    :math:`\vec{X}_c = A \vec{x}_w + T` (see :class:`pysdic.WorldPlaneRigidExtrinsic`), and the
    **normalized point** is obtained by orthographic projection, i.e. by dropping the depth:

    .. math::

        \vec{x}_n = \begin{bmatrix} X_c \\ Y_c \end{bmatrix} = M \vec{x}_w + \vec{t}
        \quad \text{with} \quad
        M = R_{:2, :2},
        \qquad
        \vec{t} = (t_x, t_y)^T

    The transformation is an **affine** map of the plane, invertible as long as the plane is not
    seen edge-on: :math:`\det(M) = R_{22} \neq 0` (cosine of the angle between the optical axis and
    the normal of the plane).

    .. note::

        As for :class:`pysdic.OrthographicExtrinsic`, the parameter :math:`t_z` has no effect on the
        normalized points (its Jacobian is zero). It must be fixed in an optimization.

    .. seealso::

        - Class :class:`pysdic.OrthographicExtrinsic` for the orthographic camera with 3D world points.


    Parameters
    ----------
    parameters : Optional[ArrayLike], optional (default: ``None``)
        The extrinsic parameters as a 1D array of shape (6,).
        The first 3 elements represent the **rotation vector** :math:`\vec{r} = (r_x, r_y, r_z)`.
        The last 3 elements represent the **translation vector** :math:`T = (t_x, t_y, t_z)`.


    Examples
    --------
    A camera rotated by 90° around the :math:`z` axis and translated along :math:`x`:

    .. code-block:: python

        extrinsic = WorldPlaneOrthographicExtrinsic.from_rt(
            rvec=[0.0, 0.0, numpy.pi / 2],
            tvec=[1.0, 0.0, 5.0],
        )

        extrinsic.apply([[1.0, 0.0]]).x          # array([[1., 1.]])
        extrinsic.invert([[1.0, 1.0]]).x         # array([[1., 0.]])

    """

    _has_apply = True
    _has_apply_dx = True
    _has_apply_dp = True
    _has_invert = True
    _has_invert_dx = True
    _has_invert_dp = True
    _is_left_invertible = True  # The depth is recovered by the intersection with the plane
    _is_right_invertible = True

    # =============================================
    # Affine matrix
    # =============================================
    @property
    def affine_matrix(self) -> Optional[numpy.ndarray]:
        r"""
        [Get] the 3x3 homogeneous affine matrix between the world plane and the normalized plane.

        .. math::

            \begin{bmatrix} \vec{x}_n \\ 1 \end{bmatrix} =
            \begin{bmatrix} M & \vec{t} \\ 0 & 1 \end{bmatrix}
            \begin{bmatrix} \vec{x}_w \\ 1 \end{bmatrix}

        .. note::

            An alias for ``affine_matrix`` is ``amatrix``.

        Returns
        -------
        Optional[numpy.ndarray]
            The affine matrix with shape ``(3, 3)``. (or None if the parameters are not set)
        """
        rmat = self.rotation_matrix
        if rmat is None:
            return None
        amatrix = numpy.eye(3, dtype=float)
        amatrix[:2, :2] = rmat[:2, :2]
        amatrix[:2, 2] = self._parameters[3:5]
        return amatrix

    @property
    def amatrix(self) -> Optional[numpy.ndarray]:
        return self.affine_matrix

    # =============================================
    # Transform methods
    # =============================================
    def _apply(
        self,
        world_points: numpy.ndarray,
        *,
        dx: bool = False,
        dp: bool = False,
    ) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        r"""
        Compute the transformation from 2D **world points** of the plane :math:`Z_w = 0` to
        2D **normalized points**.

        .. math::

            \vec{x}_n = M \vec{x}_w + \vec{t}
            \quad \text{with} \quad
            M = R_{:2, :2}

        The Jacobian with respect to the world points is constant:

        .. math::

            \frac{\partial \vec{x}_n}{\partial \vec{x}_w} = M

        The Jacobian with respect to the parameters :math:`(\vec{r}, T)` is:

        .. math::

            \frac{\partial \vec{x}_n}{\partial r_k} = \frac{\partial M}{\partial r_k} \vec{x}_w,
            \qquad
            \frac{\partial \vec{x}_n}{\partial (t_x, t_y)} = I_2,
            \qquad
            \frac{\partial \vec{x}_n}{\partial t_z} = 0

        where :math:`\frac{\partial M}{\partial r_k}` is extracted from ``cv2.Rodrigues``.


        Parameters
        ----------
        world_points : numpy.ndarray
            Input 2D points of the world plane. Shape ``(n_points, 2)``.

        dx : bool, optional (default: ``False``)
            If True, compute the Jacobian with respect to the input world points.

        dp : bool, optional (default: ``False``)
            If True, compute the Jacobian with respect to the extrinsic parameters.


        Returns
        -------
        normalized_points : numpy.ndarray
            The normalized points in camera coordinates. Shape ``(n_points, 2)``.

        jacobian_dx : Optional[numpy.ndarray]
            Jacobian of normalized points with respect to world points.
            Shape ``(n_points, 2, 2)`` if ``dx=True``, otherwise ``None``.

        jacobian_dp : Optional[numpy.ndarray]
            Jacobian of normalized points with respect to extrinsic parameters.
            Shape ``(n_points, 2, 6)`` if ``dp=True``, otherwise ``None``.

        """
        n_points = world_points.shape[0]
        _, amat, amat_dr = self._plane_rotation_and_derivatives()
        mmat = amat[:2]  # (2, 2)
        mmat_dr = amat_dr[:2]  # (2, 2, 3)

        normalized_points = world_points @ mmat.T + self._parameters[3:5]  # (n_points, 2)

        jacobian_dx = None
        if dx:
            jacobian_dx = numpy.tile(mmat, (n_points, 1, 1))  # (n_points, 2, 2)

        jacobian_dp = None
        if dp:
            jacobian_dp = numpy.zeros((n_points, 2, 6), dtype=float)  # (n_points, 2, 6)
            jacobian_dp[:, :, :3] = numpy.einsum("ijk,nj->nik", mmat_dr, world_points)  # d x_n / d r
            jacobian_dp[:, 0, 3] = 1.0  # d x_n / d tx
            jacobian_dp[:, 1, 4] = 1.0  # d y_n / d ty

        return normalized_points, jacobian_dx, jacobian_dp

    def _invert(
        self,
        normalized_points: numpy.ndarray,
        *,
        dx: bool = False,
        dp: bool = False,
    ) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        r"""
        Compute the inverse transformation from 2D **normalized points** to 2D **world points**
        of the plane :math:`Z_w = 0` (intersection of the camera rays with the plane).

        .. math::

            \vec{x}_w = M^{-1} (\vec{x}_n - \vec{t})

        The Jacobian with respect to the normalized points is constant:

        .. math::

            \frac{\partial \vec{x}_w}{\partial \vec{x}_n} = M^{-1}

        The Jacobian with respect to the parameters :math:`(\vec{r}, T)` is:

        .. math::

            \frac{\partial \vec{x}_w}{\partial r_k} = - M^{-1} \frac{\partial M}{\partial r_k} \vec{x}_w,
            \qquad
            \frac{\partial \vec{x}_w}{\partial (t_x, t_y)} = - M^{-1},
            \qquad
            \frac{\partial \vec{x}_w}{\partial t_z} = 0


        Parameters
        ----------
        normalized_points : numpy.ndarray
            Input 2D normalized points in camera coordinates. Shape ``(n_points, 2)``.

        dx : bool, optional (default: ``False``)
            If True, compute the Jacobian with respect to normalized points.

        dp : bool, optional (default: ``False``)
            If True, compute the Jacobian with respect to extrinsic parameters.


        Returns
        -------
        world_points : numpy.ndarray
            2D points of the world plane. Shape ``(n_points, 2)``.

        jacobian_dx : Optional[numpy.ndarray]
            Jacobian of world points with respect to normalized points.
            Shape ``(n_points, 2, 2)`` if ``dx=True``, otherwise ``None``.

        jacobian_dp : Optional[numpy.ndarray]
            Jacobian of world points with respect to extrinsic parameters.
            Shape ``(n_points, 2, 6)`` if ``dp=True``, otherwise ``None``.

        Raises
        ------
        ValueError
            If the plane is seen edge-on (the optical axis is parallel to the plane).

        """
        n_points = normalized_points.shape[0]
        _, amat, amat_dr = self._plane_rotation_and_derivatives()
        mmat = amat[:2]  # (2, 2)
        mmat_dr = amat_dr[:2]  # (2, 2, 3)

        # det(M) = R[2, 2] for a rotation matrix
        if abs(numpy.linalg.det(mmat)) < 1e-12:
            raise ValueError(
                "The world plane Z_w = 0 is seen edge-on (optical axis parallel to the plane): "
                "the orthographic projection of the plane cannot be inverted."
            )
        mmat_inv = numpy.linalg.inv(mmat)  # (2, 2)

        world_points = (normalized_points - self._parameters[3:5]) @ mmat_inv.T  # (n_points, 2)

        jacobian_dx = None
        if dx:
            jacobian_dx = numpy.tile(mmat_inv, (n_points, 1, 1))  # (n_points, 2, 2)

        jacobian_dp = None
        if dp:
            jacobian_dp = numpy.zeros((n_points, 2, 6), dtype=float)  # (n_points, 2, 6)
            jacobian_dp[:, :, :3] = -mmat_inv @ numpy.einsum(
                "ijk,nj->nik", mmat_dr, world_points
            )  # - M^-1 dM/dr x_w
            jacobian_dp[:, :, 3:5] = -mmat_inv  # d x_w / d (tx, ty)

        return world_points, jacobian_dx, jacobian_dp

    # ==================================================================================
    # Rays computation
    # ==================================================================================
    def compute_rays(
        self,
        normalized_points: ArrayLike,
    ) -> Rays:
        r"""
        Compute the rays of the camera passing through the given normalized points.

        The rays are the same as for :class:`pysdic.OrthographicExtrinsic`: they are parallel to the
        :math:`z` axis of the camera and are expressed in the **3D** world coordinate system (the world
        plane being :math:`Z_w = 0`):

        .. math::

            \vec{O}_w = R^T \cdot \left( \begin{bmatrix} x_n \\ y_n \\ 0 \end{bmatrix} - T \right),
            \qquad
            \vec{d}_w = R^T \cdot \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}

        Parameters
        ----------
        normalized_points : ArrayLike
            Normalized points in the camera coordinate system. Shape ``(..., 2)``.


        Returns
        -------
        :class:`pysdic.Rays`
            Rays in 3D world coordinates with shape ``(..., 6)`` (same batch shape as the input
            points), where the last dimension encodes the ray origin ``(x, y, z)`` and the unit
            direction ``(dx, dy, dz)``.

        """
        self._check_setup_for_rays()
        points, batch_shape = self._validate_points(
            normalized_points, self.output_dim, "normalized_points"
        )  # (n_points, 2)
        n_points = points.shape[0]

        rmat = cv2.Rodrigues(self._parameters[:3])[0]
        tvec = self._parameters[3:]

        camera_points = numpy.column_stack([points, numpy.zeros(n_points)])  # (n_points, 3), Z_c = 0
        origins = (camera_points - tvec) @ rmat  # R^T (X_c - T)
        direction = rmat.T[:, 2] / numpy.linalg.norm(rmat.T[:, 2])  # (3,)

        rays = numpy.empty((n_points, 6), dtype=float)
        rays[:, :3] = origins
        rays[:, 3:] = direction
        return Rays(rays.reshape(*batch_shape, 6))