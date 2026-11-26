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
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
from numpy.typing import ArrayLike

import numpy
import cv2

from .transformation import Transformation
from .rays import Rays
from .._optionals import import_optional
from .._validation import _as_array, _as_scalar_or_1d_array

if TYPE_CHECKING:
    import py3dframe


_SET_HINT = "with the 'parameters' attribute or the 'from_rt' class method"


class RigidExtrinsic(Transformation):
    r"""
    Abstract base class of the extrinsic camera models based on a **rigid motion** of the
    world points into the camera coordinate system, followed by a projection onto the
    normalized camera plane.

    This class represents an **extrinsic transformation**, which is typically the
    first step in the camera projection pipeline:

    .. code-block:: text

        world_points → normalized_points → distorted_points → image_points

    Given a **world point** :math:`\vec{X}_w = (X_w, Y_w, Z_w)^T`, the corresponding point
    in the camera coordinate system is:

    .. math::

        \vec{X}_c = R \cdot \vec{X}_w + T

    where:

    - :math:`R` is a 3×3 rotation matrix, parameterized by a rotation vector :math:`\vec{r}`
      (Rodrigues form).
    - :math:`T` is a 3×1 translation vector.

    The transformation is parameterized by **6 parameters** :math:`(r_x, r_y, r_z, t_x, t_y, t_z)`.
    The projection of the camera points :math:`\vec{X}_c` onto the normalized points
    :math:`\vec{x}_n` is defined by the subclasses:

    - :class:`pysdic.PerspectiveExtrinsic`: perspective division :math:`\vec{x}_n = (X_c / Z_c, Y_c / Z_c)`.
    - :class:`pysdic.OrthographicExtrinsic`: orthographic projection :math:`\vec{x}_n = (X_c, Y_c)`.

    Since the depth is lost by the projection, the inverse transformation needs the depth
    :math:`Z_c` of the points in the camera coordinate system, given with the ``depth`` argument
    of ``invert``.

    .. note::

        To compute the rotation matrix or vector:

        - Use ``cv2.Rodrigues()`` to convert between rotation vectors and matrices.
        - Or use ``py3dframe.Frame`` (convention 4) for consistent 3D frame handling.

    .. seealso::

        `py3dframe <https://github.com/Artezaru/py3dframe>`_ for frame and rotation vector utilities.

    .. note::

        The subclasses can be instantiated in four ways:

        - providing the parameters directly as a numpy array of shape ``(6,)``.
        - using the :meth:`from_rt` class method with a rotation vector and a translation vector.
        - using the :meth:`from_tmatrix` class method with a 4x4 transformation matrix.
        - using the :meth:`from_frame` class method with a ``py3dframe.Frame``.


    Parameters
    ----------
    parameters : Optional[ArrayLike], optional (default: ``None``)
        The extrinsic parameters as a 1D array of shape (6,).
        The first 3 elements represent the **rotation vector** :math:`\vec{r} = (r_x, r_y, r_z)`.
        The last 3 elements represent the **translation vector** :math:`T = (t_x, t_y, t_z)`.


    Implementation details
    ----------------------

    The subclasses must set the capability flags and implement the projection with the methods
    ``_apply`` and ``_invert`` (with a ``depth`` keyword argument) and the method ``compute_rays``.
    The helper ``_rotation_and_derivatives`` can be used to get the rotation matrix with its
    derivatives, and ``pysdic._validation._as_scalar_or_1d_array`` to validate the depth.

    .. note::

        The rigid extrinsics require OpenCV (``cv2``, for ``cv2.Rodrigues``): an ``ImportError``
        is raised at the instantiation if it is missing. The ``frame`` accessors also require
        ``py3dframe``, which is only imported when they are used.

    """

    def __init__(
        self,
        parameters: Optional[ArrayLike] = None,
    ) -> None:
        super().__init__(parameters=parameters)

    # =============================================
    # Properties inherited from Transformation Class
    # =============================================
    @property
    def n_parameters(self) -> int:
        r"""
        [Get] the number of parameters of the extrinsic transformation (always 6).

        Returns
        -------
        int
            The number of parameters of the extrinsic transformation.

        """
        return 6

    @property
    def input_dim(self) -> int:
        r"""
        [Get] the input dimension of the extrinsic transformation (3D world points).

        Returns
        -------
        int
            The input dimension of the extrinsic transformation.

        """
        return 3

    @property
    def output_dim(self) -> int:
        r"""
        [Get] the output dimension of the extrinsic transformation (2D normalized points).

        Returns
        -------
        int
            The output dimension of the extrinsic transformation.

        """
        return 2

    # =============================================
    # I/O Methods for Transformation Class
    # =============================================
    def _to_dict(self) -> Dict[str, Any]:
        return {}  # No constants, the parameters are serialized by the base class

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> RigidExtrinsic:
        return cls(parameters=data.get("parameters", None))

    def _copy(self) -> RigidExtrinsic:
        return self.__class__(parameters=self.parameters)

    # =============================================
    # Parameters
    # =============================================
    @property
    def parameter_names(self) -> List[str]:
        r"""
        [Get] the names of the parameters of the extrinsic transformation :
        ``["rx", "ry", "rz", "tx", "ty", "tz"]``.

        Returns
        -------
        List[str]
            The names of the parameters of the extrinsic transformation.
        """
        return ["rx", "ry", "rz", "tx", "ty", "tz"]

    # =============================================
    # Rotation and translation accessors
    # =============================================
    @property
    def rotation_vector(self) -> Optional[numpy.ndarray]:
        r"""
        [Get/Set] the rotation vector :math:`\vec{r}` of the camera.

        The rotation vector is a numpy array of shape (3,) representing the
        rotation of the camera in Rodrigues form. It corresponds to the first
        three parameters of the extrinsic transformation.

        .. note::

            An alias for ``rotation_vector`` is ``rvec`` to match OpenCV conventions.

        .. note::

            The rotation vector can only be set if all the parameters are already set.

        .. seealso::

            - :meth:`translation_vector` or ``tvec`` to get the translation vector of the camera.

        Parameters
        ----------
        rvec : ArrayLike
            The rotation vector of the camera in Rodrigues form with shape (3,).

        Returns
        -------
        Optional[numpy.ndarray]
            The rotation vector of the camera with shape (3,). (or None if the parameters are not set)
        """
        parameters = self.parameters
        return None if parameters is None else parameters[:3].copy()

    @rotation_vector.setter
    def rotation_vector(self, value: ArrayLike) -> None:
        # ravel: the vectors (3, 1) returned by OpenCV are accepted
        self._set_parameters_at(slice(0, 3), numpy.ravel(value), "rotation_vector", hint=_SET_HINT)

    @property
    def rvec(self) -> Optional[numpy.ndarray]:
        return self.rotation_vector

    @rvec.setter
    def rvec(self, value: ArrayLike) -> None:
        self.rotation_vector = value

    @property
    def translation_vector(self) -> Optional[numpy.ndarray]:
        r"""
        [Get/Set] the translation vector :math:`T` of the camera.

        The translation vector is a numpy array of shape (3,). It corresponds to the last
        three parameters of the extrinsic transformation.

        .. note::

            An alias for ``translation_vector`` is ``tvec`` to match OpenCV conventions.

        .. note::

            The translation vector can only be set if all the parameters are already set.

        .. seealso::

            - :meth:`rotation_vector` or ``rvec`` to get the rotation vector of the camera.

        Parameters
        ----------
        tvec : ArrayLike
            The translation vector of the camera with shape (3,).

        Returns
        -------
        Optional[numpy.ndarray]
            The translation vector of the camera with shape (3,). (or None if the parameters are not set)
        """
        parameters = self.parameters
        return None if parameters is None else parameters[3:].copy()

    @translation_vector.setter
    def translation_vector(self, value: ArrayLike) -> None:
        # ravel: the vectors (3, 1) returned by OpenCV are accepted
        self._set_parameters_at(slice(3, 6), numpy.ravel(value), "translation_vector", hint=_SET_HINT)

    @property
    def tvec(self) -> Optional[numpy.ndarray]:
        return self.translation_vector

    @tvec.setter
    def tvec(self, value: ArrayLike) -> None:
        self.translation_vector = value

    @staticmethod
    def _check_rotation_matrix(rmat: numpy.ndarray) -> None:
        # Check that the matrix (3, 3), already validated, is a rotation matrix
        # (orthonormal with determinant +1)
        if not numpy.allclose(rmat.T @ rmat, numpy.eye(3), atol=1e-6) or not numpy.isclose(
            numpy.linalg.det(rmat), 1.0, atol=1e-6
        ):
            raise ValueError(
                "Rotation matrix must be orthonormal with a determinant equal to +1."
            )

    @property
    def rotation_matrix(self) -> Optional[numpy.ndarray]:
        r"""
        [Get/Set] the rotation matrix :math:`R` of the extrinsic transformation.

        The rotation matrix is a numpy array of shape (3, 3) computed from the rotation
        vector with ``cv2.Rodrigues``.

        .. note::

            An alias for ``rotation_matrix`` is ``rmat`` to match OpenCV conventions.

        .. note::

            The rotation matrix can only be set if all the parameters are already set.

        Parameters
        ----------
        rmat : ArrayLike
            The rotation matrix of the camera with shape (3, 3). It must be orthonormal
            with a determinant equal to +1.

        Returns
        -------
        Optional[numpy.ndarray]
            The rotation matrix of the camera with shape (3, 3). (or None if the parameters are not set)
        """
        rvec = self.rotation_vector
        if rvec is None:
            return None
        return cv2.Rodrigues(rvec)[0]

    @rotation_matrix.setter
    def rotation_matrix(self, rmat: ArrayLike) -> None:
        rmat = _as_array(rmat, "rotation_matrix", numpy.float64, shape=(3, 3), finite=True)
        self._check_rotation_matrix(rmat)
        self.rotation_vector = cv2.Rodrigues(rmat)[0].ravel()

    @property
    def rmat(self) -> Optional[numpy.ndarray]:
        return self.rotation_matrix

    @rmat.setter
    def rmat(self, rmat: ArrayLike) -> None:
        self.rotation_matrix = rmat

    @property
    def transformation_matrix(self) -> Optional[numpy.ndarray]:
        r"""
        [Get/Set] the 4x4 transformation matrix of the extrinsic transformation.

        .. math::

            \begin{bmatrix}
            R & T \\
            0 & 1
            \end{bmatrix}

        .. note::

            An alias for ``transformation_matrix`` is ``tmatrix``.

        Parameters
        ----------
        tmatrix : Optional[ArrayLike]
            The 4x4 transformation matrix of the camera, or None to unset the parameters.

        Returns
        -------
        Optional[numpy.ndarray]
            The 4x4 transformation matrix of the camera. (or None if the parameters are not set)
        """
        parameters = self.parameters
        if parameters is None:
            return None
        tmatrix = numpy.eye(4, dtype=float)
        tmatrix[:3, :3] = cv2.Rodrigues(parameters[:3])[0]
        tmatrix[:3, 3] = parameters[3:]
        return tmatrix

    @transformation_matrix.setter
    def transformation_matrix(self, tmatrix: Optional[ArrayLike]) -> None:
        if tmatrix is None:
            self.parameters = None
            return
        tmatrix = _as_array(tmatrix, "transformation_matrix", numpy.float64, shape=(4, 4), finite=True)
        if not numpy.allclose(tmatrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
            raise ValueError("The last row of the transformation matrix must be [0, 0, 0, 1].")
        self._check_rotation_matrix(tmatrix[:3, :3])
        rvec = cv2.Rodrigues(tmatrix[:3, :3])[0].ravel()
        self.parameters = numpy.concatenate([rvec, tmatrix[:3, 3]])

    @property
    def tmatrix(self) -> Optional[numpy.ndarray]:
        return self.transformation_matrix

    @tmatrix.setter
    def tmatrix(self, tmatrix: Optional[ArrayLike]) -> None:
        self.transformation_matrix = tmatrix

    @property
    def frame(self) -> Optional[py3dframe.Frame]:
        r"""
        [Get/Set] the 3D frame of the extrinsic transformation.

        The frame is a ``py3dframe.Frame`` object representing the 3D frame of the camera
        in the world coordinate system (convention 4).

        .. seealso::

            https://github.com/Artezaru/py3dframe for more information about the Frame class.

        Parameters
        ----------
        frame : Optional[py3dframe.Frame]
            The 3D frame of the camera in the world coordinate system, or None to unset
            the parameters.

        Returns
        -------
        Optional[py3dframe.Frame]
            The 3D frame of the camera in the world coordinate system. (or None if the parameters are not set)
        """
        py3dframe = import_optional("py3dframe", "RigidExtrinsic.frame")
        parameters = self.parameters
        if parameters is None:
            return None
        return py3dframe.Frame.from_rotation_vector(
            translation=parameters[3:], rotation_vector=parameters[:3], convention=4
        )

    @frame.setter
    def frame(self, frame: Optional[py3dframe.Frame]) -> None:
        py3dframe = import_optional("py3dframe", "RigidExtrinsic.frame")
        if frame is None:
            self.parameters = None
            return
        if not isinstance(frame, py3dframe.Frame):
            raise TypeError(f"frame must be a py3dframe.Frame object, got {type(frame)}.")
        rvec = numpy.asarray(frame.get_global_rotation_vector(convention=4)).ravel()
        tvec = numpy.asarray(frame.get_global_translation(convention=4)).ravel()
        self.parameters = numpy.concatenate([rvec, tvec])

    # =============================================
    # Instantiation methods
    # =============================================
    @classmethod
    def from_rt(cls, rvec: ArrayLike, tvec: ArrayLike) -> RigidExtrinsic:
        r"""
        Create an extrinsic transformation from a rotation vector and a translation vector.

        The returned object is an instance of the class on which the method is called.

        Parameters
        ----------
        rvec : ArrayLike
            The rotation vector of the camera in Rodrigues form with shape (3,).

        tvec : ArrayLike
            The translation vector of the camera with shape (3,).


        Returns
        -------
        :class:`pysdic.RigidExtrinsic`
            A new instance of the class with the specified rotation and translation vectors.

        """
        # ravel: the vectors (3, 1) returned by OpenCV are accepted
        rvec = _as_array(numpy.ravel(rvec), "rvec", numpy.float64, shape=(3,), finite=True)
        tvec = _as_array(numpy.ravel(tvec), "tvec", numpy.float64, shape=(3,), finite=True)
        return cls(parameters=numpy.concatenate([rvec, tvec]))

    @classmethod
    def from_frame(cls, frame: py3dframe.Frame) -> RigidExtrinsic:
        r"""
        Create an extrinsic transformation from a 3D frame (convention 4).

        The returned object is an instance of the class on which the method is called.

        Parameters
        ----------
        frame : py3dframe.Frame
            The 3D frame of the camera in the world coordinate system.


        Returns
        -------
        :class:`pysdic.RigidExtrinsic`
            A new instance of the class with the specified 3D frame.

        """
        import_optional("py3dframe", "RigidExtrinsic.from_frame")
        extrinsic = cls()
        extrinsic.frame = frame
        return extrinsic

    @classmethod
    def from_tmatrix(cls, tmatrix: ArrayLike) -> RigidExtrinsic:
        r"""
        Create an extrinsic transformation from a 4x4 transformation matrix.

        The returned object is an instance of the class on which the method is called.

        Parameters
        ----------
        tmatrix : ArrayLike
            The 4x4 transformation matrix of the camera with shape (4, 4).


        Returns
        -------
        :class:`pysdic.RigidExtrinsic`
            A new instance of the class with the specified transformation matrix.

        """
        extrinsic = cls()
        extrinsic.transformation_matrix = tmatrix
        return extrinsic

    # =============================================
    # Helpers for the subclasses
    # =============================================
    def _rotation_and_derivatives(self) -> Tuple[numpy.ndarray, numpy.ndarray]:
        # Return R (3, 3) and dR/dr (3, 3, 3) with [i, j, k] = dR[i, j] / dr[k]
        rmat, jacobian = cv2.Rodrigues(self._parameters[:3])  # (3, 3), (3, 9)
        rmat_dr = numpy.asarray(jacobian, dtype=float).reshape(3, 3, 3).transpose(1, 2, 0)
        return numpy.asarray(rmat, dtype=float), rmat_dr

    @abstractmethod
    def compute_rays(self, normalized_points: ArrayLike) -> Rays:
        r"""
        Compute the rays of the camera passing through the given normalized points.

        Parameters
        ----------
        normalized_points : ArrayLike
            Normalized points in the camera coordinate system. Shape ``(..., 2)``.


        Returns
        -------
        :class:`pysdic.Rays`
            Rays in world coordinates with shape ``(..., 6)`` (same batch shape as the input points),
            where the last dimension encodes the ray origin ``(x, y, z)`` and the unit direction
            ``(dx, dy, dz)``.

        """
        raise NotImplementedError(
            f"Subclass {self.__class__.__name__} must implement the method 'compute_rays'."
        )

    def _check_setup_for_rays(self) -> None:
        if not self.is_setup:
            raise ValueError(
                "Transformation parameters are not set. Please set the parameters before computing rays."
            )


class PerspectiveExtrinsic(RigidExtrinsic):
    r"""
    Subclass of :class:`pysdic.RigidExtrinsic` implementing the OpenCV perspective extrinsic camera model.

    The world points are first transformed into the camera coordinate system by the rigid motion
    :math:`\vec{X}_c = R \cdot \vec{X}_w + T` (see :class:`pysdic.RigidExtrinsic`).

    The **normalized point** is then obtained by perspective division:

    .. math::

        \vec{x}_n = \frac{1}{Z_c} \begin{bmatrix} X_c \\ Y_c \end{bmatrix}

    where :math:`Z_c` is the depth of the point in the camera coordinate system.

    The transformation is parameterized by **6 parameters** :math:`(r_x, r_y, r_z, t_x, t_y, t_z)`.

    Since the depth is lost by the projection, the inverse transformation needs the depth
    :math:`Z_c` of the points in the camera coordinate system (see :meth:`_invert`). It is a right
    inverse (:math:`T(T^{-1}(\vec{x}_n)) = \vec{x}_n` for any positive depth) but not a left inverse.
    The points behind the camera (:math:`Z_c \leq 0`) are rejected by ``apply`` and ``invert``.

    .. seealso::

        - Class :class:`pysdic.RigidExtrinsic` for the rigid motion, its parameters and accessors
          (``rvec``, ``tvec``, ``rmat``, ``tmatrix``, ``frame``) and the construction methods.


    Parameters
    ----------
    parameters : Optional[ArrayLike], optional (default: ``None``)
        The extrinsic parameters as a 1D array of shape (6,).
        The first 3 elements represent the **rotation vector** :math:`\vec{r} = (r_x, r_y, r_z)`.
        The last 3 elements represent the **translation vector** :math:`T = (t_x, t_y, t_z)`.


    Examples
    --------
    A camera located 5 units away from the world origin along its optical axis:

    .. code-block:: python

        extrinsic = PerspectiveExtrinsic.from_rt(
            rvec=[0.0, 0.0, 0.0],
            tvec=[0.0, 0.0, 5.0],
        )

        extrinsic.apply([[1.0, 2.0, 0.0]]).x                 # array([[0.2, 0.4]])
        extrinsic.invert([[0.2, 0.4]], depth=5.0).x          # array([[1., 2., 0.]])

    """

    _has_apply = True
    _has_apply_dx = True
    _has_apply_dp = True
    _has_invert = True
    _has_invert_dx = True
    _has_invert_dp = True
    _is_left_invertible = False  # The depth is lost by the projection
    _is_right_invertible = True

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
        Compute the transformation from 3D **world points** to 2D **normalized points**.

        The transformation consists of a change of reference frame followed by the
        perspective division:

        .. math::

            \vec{X}_c = R \cdot \vec{X}_w + T,
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

            \frac{\partial \vec{x}_n}{\partial \vec{X}_w} = D R

        The Jacobian with respect to the parameters :math:`(\vec{r}, T)` is:

        .. math::

            \frac{\partial \vec{x}_n}{\partial r_k} = D \frac{\partial R}{\partial r_k} \vec{X}_w,
            \qquad
            \frac{\partial \vec{x}_n}{\partial T} = D

        where :math:`\frac{\partial R}{\partial r_k}` is given by ``cv2.Rodrigues``.


        Parameters
        ----------
        world_points : numpy.ndarray
            Input 3D points in world coordinates. Shape ``(n_points, 3)``.

        dx : bool, optional (default: ``False``)
            If True, compute the Jacobian with respect to the input 3D points.

        dp : bool, optional (default: ``False``)
            If True, compute the Jacobian with respect to the extrinsic parameters.


        Returns
        -------
        normalized_points : numpy.ndarray
            The normalized points in camera coordinates. Shape ``(n_points, 2)``.

        jacobian_dx : Optional[numpy.ndarray]
            Jacobian of normalized points with respect to world points.
            Shape ``(n_points, 2, 3)`` if ``dx=True``, otherwise ``None``.

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
        rmat, rmat_dr = self._rotation_and_derivatives()
        tvec = self._parameters[3:]

        camera_points = world_points @ rmat.T + tvec  # (n_points, 3)
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
            jacobian_dx = division_dc @ rmat  # (n_points, 2, 3)

        jacobian_dp = None
        if dp:
            camera_dp = numpy.empty((n_points, 3, 6), dtype=float)  # d X_c / d p
            camera_dp[:, :, :3] = numpy.einsum("ijk,nj->nik", rmat_dr, world_points)
            camera_dp[:, :, 3:] = numpy.eye(3)
            jacobian_dp = division_dc @ camera_dp  # (n_points, 2, 6)

        return normalized_points, jacobian_dx, jacobian_dp

    def _invert(
        self,
        normalized_points: numpy.ndarray,
        *,
        dx: bool = False,
        dp: bool = False,
        depth: Optional[ArrayLike] = None,
    ) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        r"""
        Compute the inverse transformation from 2D **normalized points** to 3D **world points**
        at a given depth.

        The normalized points are placed at the depth :math:`Z_c` in the camera coordinate system
        and transformed back to world coordinates:

        .. math::

            \vec{X}_c = Z_c \begin{bmatrix} x_n \\ y_n \\ 1 \end{bmatrix},
            \qquad
            \vec{X}_w = R^T \cdot (\vec{X}_c - T)

        The Jacobian with respect to the normalized points (the depth is fixed) is:

        .. math::

            \frac{\partial \vec{X}_w}{\partial \vec{x}_n} = Z_c R^T
            \begin{bmatrix}
            1 & 0 \\
            0 & 1 \\
            0 & 0
            \end{bmatrix}

        The Jacobian with respect to the parameters :math:`(\vec{r}, T)` is:

        .. math::

            \frac{\partial \vec{X}_w}{\partial r_k} = \frac{\partial R}{\partial r_k}^T (\vec{X}_c - T),
            \qquad
            \frac{\partial \vec{X}_w}{\partial T} = - R^T

        .. note::

            The depth is an additional argument of ``invert``: ``extrinsic.invert(points, depth=...)``.
            The inverse is a right inverse of ``apply`` only for positive depths: the depths
            :math:`Z_c \leq 0` (behind the camera) are rejected.


        Parameters
        ----------
        normalized_points : numpy.ndarray
            Input 2D normalized points in camera coordinates. Shape ``(n_points, 2)``.

        dx : bool, optional (default: ``False``)
            If True, compute the Jacobian with respect to normalized points.

        dp : bool, optional (default: ``False``)
            If True, compute the Jacobian with respect to extrinsic parameters.

        depth : Optional[ArrayLike], optional (default: ``None``)
            Depth :math:`Z_c` of the points in camera coordinates. It can be a scalar or an array
            with one value per point (with the same batch shape as the points passed to ``invert``).
            If None, all depths are equal to 1.0. The depths must be strictly positive.


        Returns
        -------
        world_points : numpy.ndarray
            3D points in world coordinates. Shape ``(n_points, 3)``.

        jacobian_dx : Optional[numpy.ndarray]
            Jacobian of world points with respect to normalized points.
            Shape ``(n_points, 3, 2)`` if ``dx=True``, otherwise ``None``.

        jacobian_dp : Optional[numpy.ndarray]
            Jacobian of world points with respect to extrinsic parameters.
            Shape ``(n_points, 3, 6)`` if ``dp=True``, otherwise ``None``.

        Raises
        ------
        ValueError
            If at least one depth is not strictly positive (point behind the camera).

        """
        n_points = normalized_points.shape[0]
        rmat, rmat_dr = self._rotation_and_derivatives()
        tvec = self._parameters[3:]

        depth = _as_scalar_or_1d_array(
            1.0 if depth is None else depth, "depth", n_points, numpy.float64, finite=False
        )  # (n_points,), depth 1 by default
        if numpy.any(~(depth > 0.0)):  # ~(>) also rejects NaN
            raise ValueError(
                "The depth Z_c of the points must be strictly positive (points in front of the camera)."
            )

        camera_points = (
            numpy.column_stack([normalized_points, numpy.ones(n_points)]) * depth[:, None]
        )  # (n_points, 3)
        centered = camera_points - tvec  # (n_points, 3)
        world_points = centered @ rmat  # R^T (X_c - T) for each point, (n_points, 3)

        jacobian_dx = None
        if dx:
            jacobian_dx = (
                rmat.T[None, :, :2] * depth[:, None, None]
            )  # (n_points, 3, 2)

        jacobian_dp = None
        if dp:
            jacobian_dp = numpy.empty((n_points, 3, 6), dtype=float)  # (n_points, 3, 6)
            jacobian_dp[:, :, :3] = numpy.einsum(
                "jik,nj->nik", rmat_dr, centered
            )  # dR^T/dr (X_c - T)
            jacobian_dp[:, :, 3:] = -rmat.T  # d X_w / d T

        return world_points, jacobian_dx, jacobian_dp

    # ==================================================================================
    # Rays computation for Extrinsic in 3D
    # ==================================================================================
    def compute_rays(
        self,
        normalized_points: ArrayLike,
    ) -> Rays:
        r"""
        Compute the rays of the camera passing through the given normalized points.

        For a perspective projection, all the rays start from the **camera center** and pass
        through the normalized points placed on the plane :math:`Z_c = 1`. Each ray is represented
        by its **origin** and its unit **direction** in world coordinates:

        - The origin is the camera center:

        .. math::

            \vec{O}_w = - R^T \cdot T

        - The direction is the normalized point on the plane :math:`Z_c = 1` expressed in world
          coordinates:

        .. math::

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
            Rays in world coordinates with shape ``(..., 6)`` (same batch shape as the input points),
            where the last dimension encodes the ray origin ``(x, y, z)`` and the unit direction
            ``(dx, dy, dz)``.

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


class OrthographicExtrinsic(RigidExtrinsic):
    r"""
    Subclass of :class:`pysdic.RigidExtrinsic` implementing the orthographic extrinsic camera model.

    The world points are first transformed into the camera coordinate system by the rigid motion
    :math:`\vec{X}_c = R \cdot \vec{X}_w + T` (see :class:`pysdic.RigidExtrinsic`).

    The **normalized point** is then obtained by orthographic projection, i.e. by dropping
    the depth :math:`Z_c`:

    .. math::

        \vec{x}_n = (X_c, Y_c)^T

    The transformation is parameterized by **6 parameters** :math:`(r_x, r_y, r_z, t_x, t_y, t_z)`.

    Since the depth is lost by the projection, the inverse transformation needs the depth
    :math:`Z_c` of the points in the camera coordinate system (see :meth:`_invert`). It is a right
    inverse (:math:`T(T^{-1}(\vec{x}_n)) = \vec{x}_n` for any depth) but not a left inverse.

    .. seealso::

        - Class :class:`pysdic.RigidExtrinsic` for the rigid motion, its parameters and accessors
          (``rvec``, ``tvec``, ``rmat``, ``tmatrix``, ``frame``) and the construction methods.


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

        extrinsic = OrthographicExtrinsic.from_rt(
            rvec=[0.0, 0.0, numpy.pi / 2],
            tvec=[1.0, 0.0, 5.0],
        )

        extrinsic.apply([[1.0, 0.0, 0.0]]).x                 # array([[1., 1.]])
        extrinsic.invert([[1.0, 1.0]], depth=5.0).x          # array([[1., 0., 0.]])

    """

    _has_apply = True
    _has_apply_dx = True
    _has_apply_dp = True
    _has_invert = True
    _has_invert_dx = True
    _has_invert_dp = True
    _is_left_invertible = False  # The depth is lost by the projection
    _is_right_invertible = True

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
        Compute the transformation from 3D **world points** to 2D **normalized points**.

        The transformation consists of a change of reference frame followed by the
        orthographic projection:

        .. math::

            \vec{X}_c = R \cdot \vec{X}_w + T,
            \qquad
            \vec{x}_n = \begin{bmatrix} X_c \\ Y_c \end{bmatrix} = P \vec{X}_c
            \quad \text{with} \quad
            P = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \end{bmatrix}

        The Jacobian with respect to the world points is constant:

        .. math::

            \frac{\partial \vec{x}_n}{\partial \vec{X}_w} = P R

        The Jacobian with respect to the parameters :math:`(\vec{r}, T)` is:

        .. math::

            \frac{\partial \vec{x}_n}{\partial r_k} = P \frac{\partial R}{\partial r_k} \vec{X}_w,
            \qquad
            \frac{\partial \vec{x}_n}{\partial T} = P

        where :math:`\frac{\partial R}{\partial r_k}` is given by ``cv2.Rodrigues``.


        Parameters
        ----------
        world_points : numpy.ndarray
            Input 3D points in world coordinates. Shape ``(n_points, 3)``.

        dx : bool, optional (default: ``False``)
            If True, compute the Jacobian with respect to the input 3D points.

        dp : bool, optional (default: ``False``)
            If True, compute the Jacobian with respect to the extrinsic parameters.


        Returns
        -------
        normalized_points : numpy.ndarray
            The normalized points in camera coordinates. Shape ``(n_points, 2)``.

        jacobian_dx : Optional[numpy.ndarray]
            Jacobian of normalized points with respect to world points.
            Shape ``(n_points, 2, 3)`` if ``dx=True``, otherwise ``None``.

        jacobian_dp : Optional[numpy.ndarray]
            Jacobian of normalized points with respect to extrinsic parameters.
            Shape ``(n_points, 2, 6)`` if ``dp=True``, otherwise ``None``.

        """
        n_points = world_points.shape[0]
        rmat, rmat_dr = self._rotation_and_derivatives()
        tvec = self._parameters[3:]

        camera_points = world_points @ rmat.T + tvec  # (n_points, 3)
        normalized_points = camera_points[:, :2].copy()  # (n_points, 2)

        jacobian_dx = None
        if dx:
            jacobian_dx = numpy.tile(rmat[:2, :], (n_points, 1, 1))  # (n_points, 2, 3)

        jacobian_dp = None
        if dp:
            jacobian_dp = numpy.zeros((n_points, 2, 6), dtype=float)  # (n_points, 2, 6)
            jacobian_dp[:, :, :3] = numpy.einsum(
                "ijk,nj->nik", rmat_dr[:2], world_points
            )  # d x_n / d r
            jacobian_dp[:, 0, 3] = 1.0  # d x_n / d tx
            jacobian_dp[:, 1, 4] = 1.0  # d y_n / d ty

        return normalized_points, jacobian_dx, jacobian_dp

    def _invert(
        self,
        normalized_points: numpy.ndarray,
        *,
        dx: bool = False,
        dp: bool = False,
        depth: Optional[ArrayLike] = None,
    ) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        r"""
        Compute the inverse transformation from 2D **normalized points** to 3D **world points**
        at a given depth.

        The normalized points are placed at the depth :math:`Z_c` in the camera coordinate system
        and transformed back to world coordinates:

        .. math::

            \vec{X}_c = \begin{bmatrix} x_n \\ y_n \\ Z_c \end{bmatrix},
            \qquad
            \vec{X}_w = R^T \cdot (\vec{X}_c - T)

        The Jacobian with respect to the normalized points is constant (the depth is fixed):

        .. math::

            \frac{\partial \vec{X}_w}{\partial \vec{x}_n} = R^T P^T

        The Jacobian with respect to the parameters :math:`(\vec{r}, T)` is:

        .. math::

            \frac{\partial \vec{X}_w}{\partial r_k} = \frac{\partial R}{\partial r_k}^T (\vec{X}_c - T),
            \qquad
            \frac{\partial \vec{X}_w}{\partial T} = - R^T

        .. note::

            The depth is an additional argument of ``invert``: ``extrinsic.invert(points, depth=...)``.


        Parameters
        ----------
        normalized_points : numpy.ndarray
            Input 2D normalized points in camera coordinates. Shape ``(n_points, 2)``.

        dx : bool, optional (default: ``False``)
            If True, compute the Jacobian with respect to normalized points.

        dp : bool, optional (default: ``False``)
            If True, compute the Jacobian with respect to extrinsic parameters.

        depth : Optional[ArrayLike], optional (default: ``None``)
            Depth :math:`Z_c` of the points in camera coordinates. It can be a scalar or an array
            with one value per point (with the same batch shape as the points passed to ``invert``).
            If None, all depths are equal to 1.0.


        Returns
        -------
        world_points : numpy.ndarray
            3D points in world coordinates. Shape ``(n_points, 3)``.

        jacobian_dx : Optional[numpy.ndarray]
            Jacobian of world points with respect to normalized points.
            Shape ``(n_points, 3, 2)`` if ``dx=True``, otherwise ``None``.

        jacobian_dp : Optional[numpy.ndarray]
            Jacobian of world points with respect to extrinsic parameters.
            Shape ``(n_points, 3, 6)`` if ``dp=True``, otherwise ``None``.

        """
        n_points = normalized_points.shape[0]
        rmat, rmat_dr = self._rotation_and_derivatives()
        tvec = self._parameters[3:]

        depth = _as_scalar_or_1d_array(
            1.0 if depth is None else depth, "depth", n_points, numpy.float64, finite=False
        )  # (n_points,), depth 1 by default

        camera_points = numpy.column_stack([normalized_points, depth])  # (n_points, 3)
        centered = camera_points - tvec  # (n_points, 3)
        world_points = centered @ rmat  # R^T (X_c - T) for each point, (n_points, 3)

        jacobian_dx = None
        if dx:
            jacobian_dx = numpy.tile(rmat.T[:, :2], (n_points, 1, 1))  # (n_points, 3, 2)

        jacobian_dp = None
        if dp:
            jacobian_dp = numpy.empty((n_points, 3, 6), dtype=float)  # (n_points, 3, 6)
            jacobian_dp[:, :, :3] = numpy.einsum(
                "jik,nj->nik", rmat_dr, centered
            )  # dR^T/dr (X_c - T)
            jacobian_dp[:, :, 3:] = -rmat.T  # d X_w / d T

        return world_points, jacobian_dx, jacobian_dp

    # ==================================================================================
    # Rays computation for Extrinsic in 3D
    # ==================================================================================
    def compute_rays(
        self,
        normalized_points: ArrayLike,
    ) -> Rays:
        r"""
        Compute the rays of the camera passing through the given normalized points.

        For an orthographic projection, all the rays are parallel to the :math:`z` axis of the
        camera coordinate system. Each ray is represented by its **origin** and its **direction**
        in world coordinates:

        - The origin is the normalized point placed on the camera plane (:math:`Z_c = 0`):

        .. math::

            \vec{O}_w = R^T \cdot \left( \begin{bmatrix} x_n \\ y_n \\ 0 \end{bmatrix} - T \right)

        - The direction is the :math:`z` axis of the camera expressed in world coordinates:

        .. math::

            \vec{d}_w = R^T \cdot \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}

        Parameters
        ----------
        normalized_points : ArrayLike
            Normalized points in the camera coordinate system. Shape ``(..., 2)``.


        Returns
        -------
        :class:`pysdic.Rays`
            Rays in world coordinates with shape ``(..., 6)`` (same batch shape as the input points),
            where the last dimension encodes the ray origin ``(x, y, z)`` and the unit direction
            ``(dx, dy, dz)``.

        """
        self._check_setup_for_rays()
        points, batch_shape = self._validate_points(
            normalized_points, self.output_dim, "normalized_points"
        )  # (n_points, 2)

        rmat = cv2.Rodrigues(self._parameters[:3])[0]
        origins = self._invert(points, depth=0.0)[0]  # (n_points, 3)
        direction = rmat.T[:, 2] / numpy.linalg.norm(rmat.T[:, 2])  # (3,)

        rays = numpy.empty((points.shape[0], 6), dtype=float)
        rays[:, :3] = origins
        rays[:, 3:] = direction
        return Rays(rays.reshape(*batch_shape, 6))