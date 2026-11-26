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
from .._validation import _as_array


def _coefficient_property(index: int, row: int, col: int) -> property:
    # Build the [Get/Set] property of one coefficient H[row, col] of the homography matrix
    name = f"h{row}{col}"

    def getter(self) -> Optional[float]:
        return self._get_component(index)

    def setter(self, value: Real) -> None:
        self._set_parameters_at(
            index, value, name, hint="with the 'parameters' attribute or the 'from_hmat' class method"
        )

    return property(
        getter,
        setter,
        doc=(
            f"[Get/Set] the coefficient :math:`h_{{{row}{col}}}` of the homography matrix "
            f"(component ``H[{row}, {col}]``), or None if the parameters are not set.\n\n"
            f"The coefficient can only be set if all the parameters are already set."
        ),
    )


class PlanarHomographicExtrinsic(Transformation):
    r"""
    Subclass of :class:`pysdic.Transformation` implementing a **planar homography**
    (projective transformation of the plane).

    This class represents an **extrinsic transformation in 2D**, used to run the camera
    algorithms on planar problems. It applies a projective transformation to transform
    2D world points into 2D normalized points:

    .. code-block:: text

        world_points → normalized_points → distorted_points → image_points

    Given a **world point** :math:`\vec{x}_w = (x_w, y_w)`, the corresponding point
    :math:`\vec{x}_n = (x_n, y_n)` in the normalized coordinate system is obtained
    using homogeneous coordinates:

    .. math::

        \begin{bmatrix}
        u \\
        v \\
        w
        \end{bmatrix}
        =
        H
        \begin{bmatrix}
        x_w \\
        y_w \\
        1
        \end{bmatrix}

    followed by the projective normalization:

    .. math::

        x_n = \frac{u}{w},
        \qquad
        y_n = \frac{v}{w}

    The homography matrix is parameterized with **8 parameters** by fixing
    its bottom-right element to one:

    .. math::

        H =
        \begin{bmatrix}
        h_{00} & h_{01} & h_{02} \\
        h_{10} & h_{11} & h_{12} \\
        h_{20} & h_{21} & 1
        \end{bmatrix}

    The homography matrix must be invertible. The inverse transformation is the homography
    of matrix :math:`H^{-1}`, so the transformation is both left and right invertible,
    and all the Jacobians are available for ``apply`` and ``invert``.

    .. seealso::

        - Class :class:`pysdic.PlanarRigidExtrinsic` for the rigid motions of the plane.
        - Class :class:`pysdic.WorldPlanePerspectiveExtrinsic` for the homographies induced by a
          perspective camera with a 3D pose observing the plane :math:`Z_w = 0`
          (6 parameters :math:`(\vec{r}, T)` instead of 8 free coefficients).

    .. note::

        A homography is defined only up to an arbitrary non-zero scale.
        The matrix is always normalized so that ``H[2, 2] = 1``, which excludes the
        homographies with a zero bottom-right element.

    .. note::

        The transformation is only defined for points for which the projective denominator
        :math:`w` is non-zero (points not sent to infinity). A ``ValueError`` is raised otherwise.

    .. note::

        The class can be instantiated in three ways:

        - providing the parameters directly as a numpy array of shape ``(8,)``.
        - using the :meth:`from_hmat` class method to initialize the transformation
          from a homography matrix (normalized so that ``H[2, 2] = 1``).
        - using the :meth:`from_eye` class method to initialize the identity transformation.


    Parameters
    ----------
    parameters : Optional[ArrayLike], optional (default: ``None``)
        The homography parameters as a 1D array of shape ``(8,)``.
        The parameters are ordered as :math:`(h_{00}, h_{01}, h_{02}, h_{10}, h_{11}, h_{12}, h_{20}, h_{21})`.


    Examples
    --------
    A homography combining a scaling, a translation and a small perspective effect:

    .. code-block:: python

        extrinsic = PlanarHomographicExtrinsic.from_hmat(
            [[2.0, 0.0, 1.0],
             [0.0, 2.0, 0.0],
             [0.1, 0.0, 1.0]]
        )

        extrinsic.apply([[1.0, 1.0]]).x     # array([[2.72727273, 1.81818182]])
        extrinsic.invert([[2.72727273, 1.81818182]]).x   # array([[1., 1.]])

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
        [Get] the number of parameters of the homography (always 8).

        Returns
        -------
        int
            The number of parameters of the homography.

        """
        return 8

    @property
    def input_dim(self) -> int:
        r"""
        [Get] the input dimension of the homography (2D world points).

        Returns
        -------
        int
            The input dimension of the homography.

        """
        return 2

    @property
    def output_dim(self) -> int:
        r"""
        [Get] the output dimension of the homography (2D normalized points).

        Returns
        -------
        int
            The output dimension of the homography.

        """
        return 2

    # =============================================
    # I/O Methods
    # =============================================
    def _to_dict(self) -> Dict[str, Any]:
        return {}  # No constants, the parameters are serialized by the base class

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> PlanarHomographicExtrinsic:
        return cls(parameters=data.get("parameters", None))

    def _copy(self) -> PlanarHomographicExtrinsic:
        return self.__class__(parameters=self.parameters)

    # =============================================
    # Parameters
    # =============================================
    def set_parameters(
        self, value: Optional[ArrayLike], *, override_lock: bool = False
    ) -> None:
        r"""
        Set the parameters of the homography, with an option to override the lock.

        In addition to the checks of :meth:`pysdic.Transformation.set_parameters`, the parameters
        must be finite and the homography matrix must be invertible.

        Parameters
        ----------
        value : Optional[ArrayLike]
            The parameters of shape ``(8,)`` ordered as
            :math:`(h_{00}, h_{01}, h_{02}, h_{10}, h_{11}, h_{12}, h_{20}, h_{21})`,
            or None to unset the parameters.

        override_lock : bool, optional (default: ``False``)
            Whether to override the lock and allow setting parameters even if they are locked.

        Raises
        ------
        ValueError
            If the parameters do not have the correct shape, are not finite or if the
            homography matrix is singular.

        RuntimeError
            If the parameters are locked and ``override_lock`` is False.

        """
        if value is not None:
            value = self._validate_parameters(value)  # Shape (8,) and finite
            if numpy.linalg.matrix_rank(self._matrix_from_parameters(value)) < 3:
                raise ValueError(
                    "The homography matrix is singular, it must be invertible."
                )
        super().set_parameters(value, override_lock=override_lock)

    @staticmethod
    def _matrix_from_parameters(parameters: numpy.ndarray) -> numpy.ndarray:
        # Build the 3x3 homography matrix with H[2, 2] = 1
        return numpy.append(parameters, 1.0).reshape(3, 3)

    def _get_component(self, index: int) -> Optional[float]:
        # Return one component of the parameters, or None if the parameters are not set
        parameters = self.parameters
        if parameters is None:
            return None
        return float(parameters[index])

    @property
    def parameter_names(self) -> List[str]:
        r"""
        [Get] the names of the parameters of the homography :
        ``["h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21"]``.

        Returns
        -------
        List[str]
            The names of the parameters of the homography.
        """
        return ["h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21"]

    # =============================================
    # Homography matrix
    # =============================================
    @property
    def homography_matrix(self) -> Optional[numpy.ndarray]:
        r"""
        [Get/Set] the 3x3 homogeneous homography matrix :math:`H`.

        .. math::

            H =
            \begin{bmatrix}
            h_{00} & h_{01} & h_{02} \\
            h_{10} & h_{11} & h_{12} \\
            h_{20} & h_{21} & 1
            \end{bmatrix}

        When the matrix is set, it is normalized so that ``H[2, 2] = 1``.

        .. note::

            An alias for ``homography_matrix`` is ``hmat``.


        Parameters
        ----------
        homography_matrix : Optional[ArrayLike]
            The homography matrix with shape ``(3, 3)`` and a non-zero bottom-right element,
            or None to unset the parameters.


        Returns
        -------
        Optional[numpy.ndarray]
            The homography matrix with shape ``(3, 3)``. (or None if the parameters are not set)
        """
        parameters = self.parameters
        if parameters is None:
            return None
        return self._matrix_from_parameters(parameters)

    @homography_matrix.setter
    def homography_matrix(self, hmat: Optional[ArrayLike]) -> None:
        if hmat is None:
            self.parameters = None
            return
        hmat = _as_array(hmat, "homography_matrix", numpy.float64, shape=(3, 3), finite=True)
        if hmat[2, 2] == 0.0:
            raise ValueError(
                "Homography matrix must have a non-zero bottom-right element."
            )
        self.parameters = (hmat / hmat[2, 2]).ravel()[:8]

    @property
    def hmat(self) -> Optional[numpy.ndarray]:
        return self.homography_matrix

    @hmat.setter
    def hmat(self, value: Optional[ArrayLike]) -> None:
        self.homography_matrix = value

    # =============================================
    # Individual matrix coefficients
    # =============================================
    h00 = _coefficient_property(0, 0, 0)
    h01 = _coefficient_property(1, 0, 1)
    h02 = _coefficient_property(2, 0, 2)
    h10 = _coefficient_property(3, 1, 0)
    h11 = _coefficient_property(4, 1, 1)
    h12 = _coefficient_property(5, 1, 2)
    h20 = _coefficient_property(6, 2, 0)
    h21 = _coefficient_property(7, 2, 1)

    # =============================================
    # Instantiation methods
    # =============================================
    @classmethod
    def from_hmat(cls, hmat: ArrayLike) -> PlanarHomographicExtrinsic:
        r"""
        Create a :class:`pysdic.PlanarHomographicExtrinsic` from a 3x3 homography matrix.

        The matrix is normalized so that ``H[2, 2] = 1``.

        Parameters
        ----------
        hmat : ArrayLike
            The homography matrix with shape ``(3, 3)``.


        Returns
        -------
        :class:`pysdic.PlanarHomographicExtrinsic`
            A new homography transformation.
        """
        extrinsic = cls()
        extrinsic.homography_matrix = hmat
        return extrinsic

    @classmethod
    def from_eye(cls) -> PlanarHomographicExtrinsic:
        r"""
        Create a :class:`pysdic.PlanarHomographicExtrinsic` representing the identity
        transformation (:math:`H = I_3`).

        Returns
        -------
        :class:`pysdic.PlanarHomographicExtrinsic`
            A new homography transformation with the identity matrix.
        """
        return cls.from_hmat(numpy.eye(3, dtype=float))

    # =============================================
    # Transform methods
    # =============================================
    @staticmethod
    def _projective_map(
        matrix: numpy.ndarray, points: numpy.ndarray
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        # Apply the projective map of a 3x3 matrix to (n_points, 2) points.
        # Return the projected points (n_points, 2) and the denominators (n_points,)
        homogeneous = points @ matrix[:, :2].T + matrix[:, 2]  # (n_points, 3)
        denominator = homogeneous[:, 2]
        if numpy.any(denominator == 0.0):
            raise ValueError(
                "Homography projection is undefined for at least one input point "
                "(zero projective denominator)."
            )
        return homogeneous[:, :2] / denominator[:, None], denominator

    @staticmethod
    def _projective_jacobian_dx(
        matrix: numpy.ndarray, projected: numpy.ndarray, denominator: numpy.ndarray
    ) -> numpy.ndarray:
        # Jacobian of the projective map with respect to the points: (M[:2, :2] - p M[2, :2]) / w
        jacobian = (
            matrix[None, :2, :2] - projected[:, :, None] * matrix[None, 2, None, :2]
        )  # (n_points, 2, 2)
        return jacobian / denominator[:, None, None]

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

            x_n = \frac{h_{00} x_w + h_{01} y_w + h_{02}}{w},
            \qquad
            y_n = \frac{h_{10} x_w + h_{11} y_w + h_{12}}{w},
            \qquad
            w = h_{20} x_w + h_{21} y_w + 1

        The Jacobian with respect to the world points is:

        .. math::

            \frac{\partial \vec{x}_n}{\partial \vec{x}_w} = \frac{1}{w}
            \begin{bmatrix}
            h_{00} - x_n h_{20} & h_{01} - x_n h_{21} \\
            h_{10} - y_n h_{20} & h_{11} - y_n h_{21}
            \end{bmatrix}

        The Jacobian with respect to the parameters
        :math:`(h_{00}, h_{01}, h_{02}, h_{10}, h_{11}, h_{12}, h_{20}, h_{21})` is:

        .. math::

            \frac{\partial \vec{x}_n}{\partial \vec{p}} = \frac{1}{w}
            \begin{bmatrix}
            x_w & y_w & 1 & 0 & 0 & 0 & -x_n x_w & -x_n y_w \\
            0 & 0 & 0 & x_w & y_w & 1 & -y_n x_w & -y_n y_w
            \end{bmatrix}


        Parameters
        ----------
        world_points : numpy.ndarray
            Input 2D points in world coordinates. Shape ``(n_points, 2)``.

        dx : bool, optional (default: ``False``)
            If True, compute the Jacobian with respect to the input world points.

        dp : bool, optional (default: ``False``)
            If True, compute the Jacobian with respect to the homography parameters.


        Returns
        -------
        normalized_points : numpy.ndarray
            Transformed 2D points in the normalized coordinate system.
            Shape ``(n_points, 2)``.

        jacobian_dx : Optional[numpy.ndarray]
            Jacobian with respect to world points. Shape
            ``(n_points, 2, 2)`` if ``dx=True``, otherwise ``None``.

        jacobian_dp : Optional[numpy.ndarray]
            Jacobian with respect to the homography parameters. Shape
            ``(n_points, 2, 8)`` if ``dp=True``, otherwise ``None``.

        Raises
        ------
        ValueError
            If the projective denominator :math:`w` is zero for at least one point.
        """
        hmat = self._matrix_from_parameters(self._parameters)
        normalized_points, w = self._projective_map(hmat, world_points)

        jacobian_dx = None
        if dx:
            jacobian_dx = self._projective_jacobian_dx(
                hmat, normalized_points, w
            )  # (n_points, 2, 2)

        jacobian_dp = None
        if dp:
            n_points = world_points.shape[0]
            x_w = world_points[:, 0]
            y_w = world_points[:, 1]
            x_n = normalized_points[:, 0]
            y_n = normalized_points[:, 1]
            jacobian_dp = numpy.zeros((n_points, 2, 8), dtype=float)  # (n_points, 2, 8)
            jacobian_dp[:, 0, 0] = x_w
            jacobian_dp[:, 0, 1] = y_w
            jacobian_dp[:, 0, 2] = 1.0
            jacobian_dp[:, 1, 3] = x_w
            jacobian_dp[:, 1, 4] = y_w
            jacobian_dp[:, 1, 5] = 1.0
            jacobian_dp[:, 0, 6] = -x_n * x_w
            jacobian_dp[:, 0, 7] = -x_n * y_w
            jacobian_dp[:, 1, 6] = -y_n * x_w
            jacobian_dp[:, 1, 7] = -y_n * y_w
            jacobian_dp /= w[:, None, None]

        return normalized_points, jacobian_dx, jacobian_dp

    def _invert(
        self,
        normalized_points: numpy.ndarray,
        *,
        dx: bool = False,
        dp: bool = False,
    ) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        r"""
        Compute the inverse transformation from 2D **normalized points**
        to 2D **world points**.

        The inverse transformation is the homography of matrix :math:`G = H^{-1}`:

        .. math::

            \begin{bmatrix}
            a \\
            b \\
            d
            \end{bmatrix}
            =
            G
            \begin{bmatrix}
            x_n \\
            y_n \\
            1
            \end{bmatrix},
            \qquad
            x_w = \frac{a}{d},
            \qquad
            y_w = \frac{b}{d}

        The Jacobian with respect to the normalized points has the same form as for ``apply``
        with the coefficients of :math:`G` and the denominator :math:`d`:

        .. math::

            \frac{\partial \vec{x}_w}{\partial \vec{x}_n} = \frac{1}{d}
            \begin{bmatrix}
            g_{00} - x_w g_{20} & g_{01} - x_w g_{21} \\
            g_{10} - y_w g_{20} & g_{11} - y_w g_{21}
            \end{bmatrix}

        The Jacobian with respect to the parameters is obtained by differentiating the identity
        :math:`T(T^{-1}(\vec{x}_n; \vec{p}); \vec{p}) = \vec{x}_n` with respect to :math:`\vec{p}`:

        .. math::

            \frac{\partial T^{-1}}{\partial \vec{p}}(\vec{x}_n) =
            - \frac{\partial \vec{x}_w}{\partial \vec{x}_n} \cdot \frac{\partial T}{\partial \vec{p}}(\vec{x}_w)

        where :math:`\frac{\partial T}{\partial \vec{p}}` is the Jacobian of ``apply`` with respect to
        the parameters evaluated at the world points :math:`\vec{x}_w`.


        Parameters
        ----------
        normalized_points : numpy.ndarray
            Input 2D points in normalized coordinates. Shape ``(n_points, 2)``.

        dx : bool, optional (default: ``False``)
            If True, compute the Jacobian with respect to the input normalized points.

        dp : bool, optional (default: ``False``)
            If True, compute the Jacobian with respect to the homography parameters.


        Returns
        -------
        world_points : numpy.ndarray
            Inverse-transformed points in world coordinates. Shape ``(n_points, 2)``.

        jacobian_dx : Optional[numpy.ndarray]
            Jacobian with respect to normalized points.
            Shape ``(n_points, 2, 2)`` if ``dx=True``, otherwise ``None``.

        jacobian_dp : Optional[numpy.ndarray]
            Jacobian with respect to the homography parameters.
            Shape ``(n_points, 2, 8)`` if ``dp=True``, otherwise ``None``.

        Raises
        ------
        ValueError
            If the projective denominator :math:`d` is zero for at least one point.
        """
        hmat = self._matrix_from_parameters(self._parameters)
        hmat_inv = numpy.linalg.inv(hmat)  # Invertible by construction (checked when set)
        world_points, d = self._projective_map(hmat_inv, normalized_points)

        jacobian_dx = None
        if dx or dp:
            jacobian_dx = self._projective_jacobian_dx(
                hmat_inv, world_points, d
            )  # (n_points, 2, 2)

        jacobian_dp = None
        if dp:
            _, _, jacobian_apply_dp = self._apply(world_points, dp=True)  # (n_points, 2, 8)
            jacobian_dp = -jacobian_dx @ jacobian_apply_dp  # (n_points, 2, 8)

        return world_points, (jacobian_dx if dx else None), jacobian_dp