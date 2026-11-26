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
from typing import Any, Dict, List, Optional, Tuple
from numbers import Real
from numpy.typing import ArrayLike

import numpy

from .transformation import Transformation
from .._validation import _validate_real, _as_array


_SET_HINT = "with the 'parameters' attribute or the 'from_rt' class method"


class PlanarRigidExtrinsic(Transformation):
    r"""
    Subclass of :class:`pysdic.Transformation` implementing a **planar rigid motion**
    (rotation and translation in 2D).

    This class represents an **extrinsic transformation in 2D**, used to run the camera
    algorithms on planar problems. It transforms 2D world points into 2D normalized points,
    without projection:

    .. code-block:: text

        world_points → normalized_points → distorted_points → image_points

    Given a **world point** :math:`\vec{x}_w = (x_w, y_w)^T`, the corresponding point
    in the normalized coordinate system is:

    .. math::

        \vec{x}_n = R(\theta) \cdot \vec{x}_w + \vec{t}

    where:

    .. math::

        R(\theta) =
        \begin{bmatrix}
        \cos(\theta) & -\sin(\theta) \\
        \sin(\theta) &  \cos(\theta)
        \end{bmatrix},
        \qquad
        \vec{t} =
        \begin{bmatrix}
        t_x \\
        t_y
        \end{bmatrix}

    The transformation is parameterized by **3 parameters** :math:`(\theta, t_x, t_y)`, with
    :math:`\theta` the rotation angle in radians. It is a bijection of the plane, and the inverse
    transformation is :math:`\vec{x}_w = R(\theta)^T \cdot (\vec{x}_n - \vec{t})`.

    .. seealso::

        - Class :class:`pysdic.RigidExtrinsic` for the rigid motions of 3D world points with a projection.
        - Class :class:`pysdic.WorldPlaneRigidExtrinsic` for the plane :math:`Z_w = 0` observed by a camera
          with a 3D pose.
        - Class :class:`pysdic.PlanarHomographicExtrinsic` for the projective transformations of the plane.

    .. note::

        The class can be instantiated in three ways:

        - providing the parameters directly as a numpy array of shape ``(3,)``.
        - using the :meth:`from_rt` class method with a rotation angle and a translation vector.
        - using the :meth:`from_tmatrix` class method with a 3x3 homogeneous transformation matrix.


    Parameters
    ----------
    parameters : Optional[ArrayLike], optional (default: ``None``)
        The extrinsic parameters as a 1D array of shape ``(3,)`` ordered as
        :math:`(\theta, t_x, t_y)`, where :math:`\theta` is the rotation angle in radians
        and :math:`(t_x, t_y)` is the translation vector.


    Examples
    --------
    A rotation of 90° followed by a translation along :math:`x`:

    .. code-block:: python

        extrinsic = PlanarRigidExtrinsic.from_rt(theta=numpy.pi / 2, tvec=[1.0, 0.0])

        extrinsic.apply([[1.0, 0.0]]).x      # array([[1., 1.]])
        extrinsic.invert([[1.0, 1.0]]).x     # array([[1., 0.]])

    """

    _has_apply = True
    _has_apply_dx = True
    _has_apply_dp = True
    _has_invert = True
    _has_invert_dx = True
    _has_invert_dp = True
    _is_left_invertible = True
    _is_right_invertible = True

    def __init__(
        self,
        parameters: Optional[ArrayLike] = None,
    ) -> None:
        super().__init__(parameters=parameters)

    # =============================================
    # Properties inherited from Transformation
    # =============================================
    @property
    def n_parameters(self) -> int:
        r"""
        [Get] the number of parameters of the planar rigid motion (always 3).

        Returns
        -------
        int
            The number of parameters of the planar rigid motion.

        """
        return 3

    @property
    def input_dim(self) -> int:
        r"""
        [Get] the input dimension of the planar rigid motion (2D world points).

        Returns
        -------
        int
            The input dimension of the planar rigid motion.

        """
        return 2

    @property
    def output_dim(self) -> int:
        r"""
        [Get] the output dimension of the planar rigid motion (2D normalized points).

        Returns
        -------
        int
            The output dimension of the planar rigid motion.

        """
        return 2

    # =============================================
    # I/O Methods
    # =============================================
    def _to_dict(self) -> Dict[str, Any]:
        return {}  # No constants, the parameters are serialized by the base class

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> PlanarRigidExtrinsic:
        return cls(parameters=data.get("parameters", None))

    def _copy(self) -> PlanarRigidExtrinsic:
        return self.__class__(parameters=self.parameters)

    # =============================================
    # Parameters
    # =============================================
    @property
    def parameter_names(self) -> List[str]:
        r"""
        [Get] the names of the parameters of the planar rigid motion : ``["theta", "tx", "ty"]``.

        Returns
        -------
        List[str]
            The names of the parameters of the planar rigid motion.
        """
        return ["theta", "tx", "ty"]

    # =============================================
    # Rotation angle
    # =============================================
    @property
    def rotation_angle(self) -> Optional[float]:
        r"""
        [Get/Set] the rotation angle :math:`\theta` in radians.

        .. note::

            An alias for ``rotation_angle`` is ``theta``.

        .. note::

            The rotation angle can only be set if all the parameters are already set.

        Returns
        -------
        Optional[float]
            The rotation angle in radians. (or None if the parameters are not set)
        """
        parameters = self.parameters
        return None if parameters is None else float(parameters[0])

    @rotation_angle.setter
    def rotation_angle(self, value: Real) -> None:
        self._set_parameters_at(0, value, "rotation_angle", hint=_SET_HINT)

    @property
    def theta(self) -> Optional[float]:
        return self.rotation_angle

    @theta.setter
    def theta(self, value: Real) -> None:
        self.rotation_angle = value

    # =============================================
    # Translation vector
    # =============================================
    @property
    def translation_vector(self) -> Optional[numpy.ndarray]:
        r"""
        [Get/Set] the translation vector :math:`\vec{t} = (t_x, t_y)`.

        .. note::

            An alias for ``translation_vector`` is ``tvec``.

        .. note::

            The translation vector can only be set if all the parameters are already set.

        Parameters
        ----------
        tvec : ArrayLike
            The translation vector with shape ``(2,)``.

        Returns
        -------
        Optional[numpy.ndarray]
            The translation vector with shape ``(2,)``. (or None if the parameters are not set)
        """
        parameters = self.parameters
        return None if parameters is None else parameters[1:].copy()

    @translation_vector.setter
    def translation_vector(self, value: ArrayLike) -> None:
        # ravel: the vectors (2, 1) are accepted
        self._set_parameters_at(slice(1, 3), numpy.ravel(value), "translation_vector", hint=_SET_HINT)

    @property
    def tvec(self) -> Optional[numpy.ndarray]:
        return self.translation_vector

    @tvec.setter
    def tvec(self, value: ArrayLike) -> None:
        self.translation_vector = value

    # =============================================
    # Rotation and transformation matrices
    # =============================================
    @staticmethod
    def _rotation(theta: float) -> Tuple[numpy.ndarray, numpy.ndarray]:
        # Return R(theta) (2, 2) and dR/dtheta (2, 2)
        c, s = numpy.cos(theta), numpy.sin(theta)
        rmat = numpy.array([[c, -s], [s, c]], dtype=float)
        rmat_dtheta = numpy.array([[-s, -c], [c, -s]], dtype=float)
        return rmat, rmat_dtheta

    @staticmethod
    def _angle_from_rotation_matrix(rmat: numpy.ndarray) -> float:
        # Check that the matrix (2, 2), already validated, is a 2D rotation matrix
        # and return its angle
        if not numpy.allclose(rmat.T @ rmat, numpy.eye(2), atol=1e-6) or not numpy.isclose(
            numpy.linalg.det(rmat), 1.0, atol=1e-6
        ):
            raise ValueError(
                "Rotation matrix must be orthonormal with a determinant equal to +1."
            )
        return float(numpy.arctan2(rmat[1, 0], rmat[0, 0]))

    @property
    def rotation_matrix(self) -> Optional[numpy.ndarray]:
        r"""
        [Get/Set] the 2D rotation matrix :math:`R(\theta)`.

        .. math::

            R(\theta) =
            \begin{bmatrix}
            \cos(\theta) & -\sin(\theta) \\
            \sin(\theta) &  \cos(\theta)
            \end{bmatrix}

        When the matrix is set, the angle is computed with :math:`\theta = \text{arctan2}(R_{10}, R_{00})`
        in :math:`(-\pi, \pi]`.

        .. note::

            An alias for ``rotation_matrix`` is ``rmat``.

        .. note::

            The rotation matrix can only be set if all the parameters are already set.

        Parameters
        ----------
        rmat : ArrayLike
            The rotation matrix with shape ``(2, 2)``. It must be orthonormal with a
            determinant equal to +1.

        Returns
        -------
        Optional[numpy.ndarray]
            The rotation matrix with shape ``(2, 2)``. (or None if the parameters are not set)
        """
        theta = self.rotation_angle
        return None if theta is None else self._rotation(theta)[0]

    @rotation_matrix.setter
    def rotation_matrix(self, rmat: ArrayLike) -> None:
        rmat = _as_array(rmat, "rotation_matrix", numpy.float64, shape=(2, 2), finite=True)
        theta = self._angle_from_rotation_matrix(rmat)
        self.rotation_angle = theta

    @property
    def rmat(self) -> Optional[numpy.ndarray]:
        return self.rotation_matrix

    @rmat.setter
    def rmat(self, value: ArrayLike) -> None:
        self.rotation_matrix = value

    @property
    def transformation_matrix(self) -> Optional[numpy.ndarray]:
        r"""
        [Get/Set] the 3x3 homogeneous transformation matrix.

        .. math::

            \begin{bmatrix}
            R(\theta) & \vec{t} \\
            0 & 1
            \end{bmatrix}

        .. note::

            An alias for ``transformation_matrix`` is ``tmatrix``.

        Parameters
        ----------
        tmatrix : Optional[ArrayLike]
            The homogeneous transformation matrix with shape ``(3, 3)``, or None to unset the parameters.

        Returns
        -------
        Optional[numpy.ndarray]
            The homogeneous transformation matrix with shape ``(3, 3)``. (or None if the parameters are not set)
        """
        parameters = self.parameters
        if parameters is None:
            return None
        tmatrix = numpy.eye(3, dtype=float)
        tmatrix[:2, :2] = self._rotation(parameters[0])[0]
        tmatrix[:2, 2] = parameters[1:]
        return tmatrix

    @transformation_matrix.setter
    def transformation_matrix(self, tmatrix: Optional[ArrayLike]) -> None:
        if tmatrix is None:
            self.parameters = None
            return
        tmatrix = _as_array(tmatrix, "transformation_matrix", numpy.float64, shape=(3, 3), finite=True)
        if not numpy.allclose(tmatrix[2], [0.0, 0.0, 1.0], atol=1e-6):
            raise ValueError("The last row of the transformation matrix must be [0, 0, 1].")
        theta = self._angle_from_rotation_matrix(tmatrix[:2, :2])
        self.parameters = [theta, tmatrix[0, 2], tmatrix[1, 2]]

    @property
    def tmatrix(self) -> Optional[numpy.ndarray]:
        return self.transformation_matrix

    @tmatrix.setter
    def tmatrix(self, value: Optional[ArrayLike]) -> None:
        self.transformation_matrix = value

    # =============================================
    # Instantiation methods
    # =============================================
    @classmethod
    def from_rt(cls, theta: Real, tvec: ArrayLike) -> PlanarRigidExtrinsic:
        r"""
        Create a :class:`pysdic.PlanarRigidExtrinsic` from a rotation angle and a
        translation vector.

        Parameters
        ----------
        theta : Real
            The rotation angle in radians.

        tvec : ArrayLike
            The translation vector with shape ``(2,)``.


        Returns
        -------
        :class:`pysdic.PlanarRigidExtrinsic`
            A new planar rigid motion.
        """
        theta = _validate_real(theta, "theta", finite=True)
        # ravel: the vectors (2, 1) are accepted
        tvec = _as_array(numpy.ravel(tvec), "tvec", numpy.float64, shape=(2,), finite=True)
        return cls(parameters=[theta, tvec[0], tvec[1]])

    @classmethod
    def from_tmatrix(cls, tmatrix: ArrayLike) -> PlanarRigidExtrinsic:
        r"""
        Create a :class:`pysdic.PlanarRigidExtrinsic` from a 3x3 homogeneous
        transformation matrix.

        Parameters
        ----------
        tmatrix : ArrayLike
            The homogeneous transformation matrix with shape ``(3, 3)``.


        Returns
        -------
        :class:`pysdic.PlanarRigidExtrinsic`
            A new planar rigid motion.
        """
        extrinsic = cls()
        extrinsic.transformation_matrix = tmatrix
        return extrinsic

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
        Compute the transformation from 2D **world points** to 2D **normalized points**.

        .. math::

            \vec{x}_n = R(\theta) \cdot \vec{x}_w + \vec{t}

        The Jacobian with respect to the world points is constant:

        .. math::

            \frac{\partial \vec{x}_n}{\partial \vec{x}_w} = R(\theta)

        The Jacobian with respect to the parameters :math:`(\theta, t_x, t_y)` is:

        .. math::

            \frac{\partial \vec{x}_n}{\partial \theta} = \frac{\partial R}{\partial \theta} \vec{x}_w,
            \qquad
            \frac{\partial \vec{x}_n}{\partial \vec{t}} = I_2
            \qquad \text{with} \qquad
            \frac{\partial R}{\partial \theta} =
            \begin{bmatrix}
            -\sin(\theta) & -\cos(\theta) \\
            \cos(\theta) & -\sin(\theta)
            \end{bmatrix}


        Parameters
        ----------
        world_points : numpy.ndarray
            Input 2D points in world coordinates. Shape ``(n_points, 2)``.

        dx : bool, optional (default: ``False``)
            If True, compute the Jacobian with respect to the input world points.

        dp : bool, optional (default: ``False``)
            If True, compute the Jacobian with respect to the parameters :math:`(\theta, t_x, t_y)`.


        Returns
        -------
        normalized_points : numpy.ndarray
            Transformed 2D points in the normalized coordinate system. Shape ``(n_points, 2)``.

        jacobian_dx : Optional[numpy.ndarray]
            Jacobian with respect to world points. Shape ``(n_points, 2, 2)`` if ``dx=True``,
            otherwise ``None``.

        jacobian_dp : Optional[numpy.ndarray]
            Jacobian with respect to the parameters. Shape ``(n_points, 2, 3)`` if ``dp=True``,
            otherwise ``None``.
        """
        n_points = world_points.shape[0]
        rmat, rmat_dtheta = self._rotation(self._parameters[0])
        tvec = self._parameters[1:]

        normalized_points = world_points @ rmat.T + tvec  # (n_points, 2)

        jacobian_dx = None
        if dx:
            jacobian_dx = numpy.tile(rmat, (n_points, 1, 1))  # (n_points, 2, 2)

        jacobian_dp = None
        if dp:
            jacobian_dp = numpy.empty((n_points, 2, 3), dtype=float)  # (n_points, 2, 3)
            jacobian_dp[:, :, 0] = world_points @ rmat_dtheta.T  # d x_n / d theta
            jacobian_dp[:, :, 1:] = numpy.eye(2)  # d x_n / d t

        return normalized_points, jacobian_dx, jacobian_dp

    def _invert(
        self,
        normalized_points: numpy.ndarray,
        *,
        dx: bool = False,
        dp: bool = False,
    ) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        r"""
        Compute the inverse transformation from 2D **normalized points** to 2D **world points**.

        .. math::

            \vec{x}_w = R(\theta)^T \cdot (\vec{x}_n - \vec{t})

        The Jacobian with respect to the normalized points is constant:

        .. math::

            \frac{\partial \vec{x}_w}{\partial \vec{x}_n} = R(\theta)^T

        The Jacobian with respect to the parameters :math:`(\theta, t_x, t_y)` is:

        .. math::

            \frac{\partial \vec{x}_w}{\partial \theta} = \frac{\partial R}{\partial \theta}^T (\vec{x}_n - \vec{t}),
            \qquad
            \frac{\partial \vec{x}_w}{\partial \vec{t}} = - R(\theta)^T


        Parameters
        ----------
        normalized_points : numpy.ndarray
            Input 2D points in the normalized coordinate system. Shape ``(n_points, 2)``.

        dx : bool, optional (default: ``False``)
            If True, compute the Jacobian with respect to the input normalized points.

        dp : bool, optional (default: ``False``)
            If True, compute the Jacobian with respect to the parameters :math:`(\theta, t_x, t_y)`.


        Returns
        -------
        world_points : numpy.ndarray
            Transformed 2D points in world coordinates. Shape ``(n_points, 2)``.

        jacobian_dx : Optional[numpy.ndarray]
            Jacobian with respect to normalized points. Shape ``(n_points, 2, 2)`` if ``dx=True``,
            otherwise ``None``.

        jacobian_dp : Optional[numpy.ndarray]
            Jacobian with respect to the parameters. Shape ``(n_points, 2, 3)`` if ``dp=True``,
            otherwise ``None``.
        """
        n_points = normalized_points.shape[0]
        rmat, rmat_dtheta = self._rotation(self._parameters[0])
        tvec = self._parameters[1:]

        centered = normalized_points - tvec  # (n_points, 2)
        world_points = centered @ rmat  # R^T (x_n - t) for each point, (n_points, 2)

        jacobian_dx = None
        if dx:
            jacobian_dx = numpy.tile(rmat.T, (n_points, 1, 1))  # (n_points, 2, 2)

        jacobian_dp = None
        if dp:
            jacobian_dp = numpy.empty((n_points, 2, 3), dtype=float)  # (n_points, 2, 3)
            jacobian_dp[:, :, 0] = centered @ rmat_dtheta  # dR^T/dtheta (x_n - t)
            jacobian_dp[:, :, 1:] = -rmat.T  # d x_w / d t

        return world_points, jacobian_dx, jacobian_dp