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
from numbers import Integral, Real
from numpy.typing import ArrayLike

import warnings
import numpy
import cv2

from .transformation import Transformation
from .._optionals import import_optional
from .._validation import (
    _validate_bool,
    _validate_integer,
    _validate_real,
    _validate_kwargs,
    _as_array,
)


_PARAMETER_NAMES = [
    "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6",
    "s1", "s2", "s3", "s4", "tau_x", "tau_y",
]  # fmt: skip

_ALLOWED_N_PARAMETERS = (0, 4, 5, 8, 12, 14)

_SET_HINT = "or use 'set_zero' to initialize them with zeros"


def _coefficient_property(index: int, description: str) -> property:
    # Build the [Get/Set] property of one distortion coefficient
    name = _PARAMETER_NAMES[index]
    min_n = next(n for n in _ALLOWED_N_PARAMETERS if n > index)

    def getter(self) -> Optional[float]:
        self._check_coefficient_available(index, name)
        parameters = self.parameters
        return None if parameters is None else float(parameters[index])

    def setter(self, value: Real) -> None:
        self._check_coefficient_available(index, name)
        self._set_parameters_at(index, value, name, hint=_SET_HINT)

    return property(
        getter,
        setter,
        doc=(
            f"[Get/Set] the {description} ``{name}`` (component {index} of the parameters), "
            f"or None if the parameters are not set.\n\n"
            f"The coefficient is only available if the model has at least {min_n} parameters, "
            f"and can only be set if the parameters are already set (see :meth:`set_zero`)."
        ),
    )


class BrownDistortion(Transformation):
    r"""
    Subclass of :class:`pysdic.Transformation` that represents the Brown-Conrady
    distortion model, as implemented by OpenCV, with radial, tangential, thin prism,
    and tilt distortions.

    This class represents a **distortion transformation**, which is typically the
    middle step in the camera projection pipeline:

    .. code-block:: text

        world_points → normalized_points → distorted_points → image_points

    Let :math:`\vec{x}_n = (x_n, y_n)` be a **normalized point** and :math:`r^2 = x_n^2 + y_n^2`.
    The distortion without tilt :math:`\vec{x}_0 = (x_0, y_0)` is given by the OpenCV model:

    .. math::

        \begin{bmatrix}
        x_0 \\
        y_0
        \end{bmatrix}
        =
        \begin{bmatrix}
        x_n K(r^2) + 2p_1 x_n y_n + p_2 (r^2 + 2x_n^2) + s_1 r^2 + s_2 r^4
        \\
        y_n K(r^2) + p_1 (r^2 + 2y_n^2) + 2p_2 x_n y_n + s_3 r^2 + s_4 r^4
        \end{bmatrix}
        \qquad \text{with} \qquad
        K(r^2) = \frac{1+k_1 r^2 + k_2 r^4 + k_3 r^6}{1 + k_4 r^2 + k_5 r^4 + k_6 r^6}

    where

    - :math:`k_i` are the radial distortion coefficients,
    - :math:`p_i` are the tangential distortion coefficients,
    - :math:`s_i` are the thin prism distortion coefficients.

    The **distorted point** :math:`\vec{x}_d = (x_d, y_d)` is then obtained by the tilt
    projection of parameters :math:`(\tau_x, \tau_y)`:

    .. math::

        \begin{bmatrix}
        h_x \\
        h_y \\
        h_z
        \end{bmatrix}
        =
        M(\tau)
        \begin{bmatrix}
        x_0 \\
        y_0 \\
        1
        \end{bmatrix},
        \qquad
        x_d = \frac{h_x}{h_z},
        \qquad
        y_d = \frac{h_y}{h_z}

    where :math:`M(\tau) = R_Z[R(\tau)] \, R(\tau)` is the tilt matrix of OpenCV
    (see :meth:`_compute_tilt_matrix`). Without tilt parameters, :math:`M = I_3` and
    :math:`\vec{x}_d = \vec{x}_0`.

    OpenCV supports several distortion configurations depending on the number
    of parameters :math:`N`:

    - **N = 0** : no distortion (identity)
    - **N = 4** : :math:`(k_1, k_2, p_1, p_2)` — radial and tangential distortion
    - **N = 5** : :math:`(k_1, k_2, p_1, p_2, k_3)` — additional radial term
    - **N = 8** : :math:`(k_1, k_2, p_1, p_2, k_3, k_4, k_5, k_6)` — rational radial model
    - **N = 12** : :math:`(k_1, k_2, p_1, p_2, k_3, k_4, k_5, k_6, s_1, s_2, s_3, s_4)`
      — adds thin prism distortion
    - **N = 14** : previous parameters plus :math:`(\tau_x, \tau_y)` — tilt distortion

    The missing coefficients of a configuration are equal to zero.

    The inverse transformation (undistortion) has no closed form and is computed with
    Newton iterations (see :meth:`_invert`). It is available with its Jacobians, and it is
    both a left and a right inverse of the distortion in the domain where the distortion is
    bijective (i.e. for reasonable distortions and points in the field of view).

    .. seealso::

        OpenCV calibration documentation:
        https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html

    .. important::

        During instantiation:

        - If both ``parameters`` and ``n_parameters`` are provided, the length of
          ``parameters`` must match ``n_parameters``.
        - If only ``n_parameters`` is provided, the parameters are not set and must
          be set before use (see :meth:`set_zero`).
        - If only ``parameters`` is provided, the number of parameters is
          inferred from the length of the array.
        - If neither is provided, an error is raised.


    Parameters
    ----------
    parameters : Optional[ArrayLike], optional (default: ``None``)
        Distortion parameters of shape ``(N,)`` ordered according to
        the selected OpenCV model.

    n_parameters : Optional[Integral], optional (default: ``None``)
        Number of parameters :math:`N` of the distortion model, in ``[0, 4, 5, 8, 12, 14]``.
        If not specified, it is inferred from the length of ``parameters``.


    Examples
    --------
    A radial and tangential distortion (5 parameters, as returned by ``cv2.calibrateCamera``):

    .. code-block:: python

        distortion = BrownDistortion([-0.1, 0.01, 0.001, -0.002, 0.0])

        distorted = distortion.apply([[0.1, 0.2]]).x
        distortion.invert(distorted).x    # array([[0.1, 0.2]])

        distortion.k1                     # -0.1

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
        *,
        n_parameters: Optional[Integral] = None,
    ) -> None:
        if parameters is None and n_parameters is None:
            raise ValueError("Either 'parameters' or 'n_parameters' must be provided.")
        if n_parameters is None:
            # Only to infer N: the parameters are validated by set_parameters
            n_parameters = _as_array(
                parameters, "parameters", numpy.float64, shape=(None,), shape_hint="(N,)", finite=True
            ).shape[0]
        n_parameters = _validate_integer(n_parameters, "n_parameters")
        if n_parameters not in _ALLOWED_N_PARAMETERS:
            raise ValueError(
                f"Input 'n_parameters' must be in {list(_ALLOWED_N_PARAMETERS)}. Got {n_parameters}."
            )
        self._n_distortion_parameters = n_parameters

        super().__init__(parameters=parameters)

    # =============================================
    # Properties inherited from Transformation Class
    # =============================================
    @property
    def n_parameters(self) -> int:
        r"""
        [Get] the number of parameters :math:`N` of the distortion model, in ``[0, 4, 5, 8, 12, 14]``.

        Returns
        -------
        int
            The number of parameters of the distortion model.

        """
        return self._n_distortion_parameters

    @property
    def input_dim(self) -> int:
        r"""
        [Get] the input dimension of the distortion (2D normalized points).

        Returns
        -------
        int
            The input dimension of the distortion.

        """
        return 2

    @property
    def output_dim(self) -> int:
        r"""
        [Get] the output dimension of the distortion (2D distorted points).

        Returns
        -------
        int
            The output dimension of the distortion.

        """
        return 2

    # =============================================
    # I/O Methods for Transformation Class
    # =============================================
    def _to_dict(self) -> Dict[str, Any]:
        return {"n_parameters": self._n_distortion_parameters}

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> BrownDistortion:
        return cls(
            parameters=data.get("parameters", None),
            n_parameters=data.get("n_parameters", None),
        )

    def _copy(self) -> BrownDistortion:
        return self.__class__(
            parameters=self.parameters, n_parameters=self._n_distortion_parameters
        )

    # =============================================
    # Parameters
    # =============================================
    def _check_coefficient_available(self, index: int, name: str) -> None:
        # Check that the coefficient exists in the current configuration
        if index >= self.n_parameters:
            raise ValueError(
                f"The coefficient '{name}' is not available with {self.n_parameters} parameters."
            )

    @property
    def parameter_names(self) -> List[str]:
        r"""
        [Get] the names of the parameters of the distortion model, i.e. the first :math:`N` names of
        ``["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6", "s1", "s2", "s3", "s4", "tau_x", "tau_y"]``.

        Returns
        -------
        List[str]
            The names of the parameters of the distortion model.
        """
        return _PARAMETER_NAMES[: self.n_parameters]

    def set_zero(self) -> None:
        r"""
        Set all the parameters of the distortion model to zero (no distortion).
        """
        self.parameters = numpy.zeros(self.n_parameters, dtype=float)

    # =================================================================
    # Distortion Model Coefficients
    # =================================================================
    k1 = _coefficient_property(0, "first radial distortion coefficient")
    k2 = _coefficient_property(1, "second radial distortion coefficient")
    p1 = _coefficient_property(2, "first tangential distortion coefficient")
    p2 = _coefficient_property(3, "second tangential distortion coefficient")
    k3 = _coefficient_property(4, "third radial distortion coefficient")
    k4 = _coefficient_property(5, "fourth radial distortion coefficient")
    k5 = _coefficient_property(6, "fifth radial distortion coefficient")
    k6 = _coefficient_property(7, "sixth radial distortion coefficient")
    s1 = _coefficient_property(8, "first thin prism distortion coefficient")
    s2 = _coefficient_property(9, "second thin prism distortion coefficient")
    s3 = _coefficient_property(10, "third thin prism distortion coefficient")
    s4 = _coefficient_property(11, "fourth thin prism distortion coefficient")
    tau_x = _coefficient_property(12, "tilt parameter along the x-axis")
    tau_y = _coefficient_property(13, "tilt parameter along the y-axis")

    # =============================================
    # Tilt matrix
    # =============================================
    @staticmethod
    def _compute_tilt_matrix(
        tau_x: float, tau_y: float
    ) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        r"""
        Compute the tilt matrix of OpenCV and its derivatives with respect to the tilt parameters.

        The tilt matrix is:

        .. math::

            M(\tau) = R_Z[R_Y R_X] \, R_Y R_X

        where :math:`R_X` and :math:`R_Y` are the rotation matrices along X and Y:

        .. math::

            R_X = \begin{pmatrix}
                1 & 0 & 0 \\
                0 & \cos(\tau_x) & \sin(\tau_x) \\
                0 & -\sin(\tau_x) & \cos(\tau_x)
            \end{pmatrix},
            \qquad
            R_Y = \begin{pmatrix}
                \cos(\tau_y) & 0 & -\sin(\tau_y) \\
                0 & 1 & 0 \\
                \sin(\tau_y) & 0 & \cos(\tau_y)
            \end{pmatrix}

        and the projection matrix :math:`R_Z` is given by:

        .. math::

            R_Z[R] = \begin{pmatrix}
                R_{33} & 0 & -R_{13} \\
                0 & R_{33} & -R_{23} \\
                0 & 0 & 1
            \end{pmatrix}

        Parameters
        ----------
        tau_x : float
            The tilt parameter along the x-axis.

        tau_y : float
            The tilt parameter along the y-axis.


        Returns
        -------
        numpy.ndarray
            The tilt matrix :math:`M` with shape (3, 3).

        numpy.ndarray
            The derivative :math:`\frac{\partial M}{\partial \tau_x}` with shape (3, 3).

        numpy.ndarray
            The derivative :math:`\frac{\partial M}{\partial \tau_y}` with shape (3, 3).
        """
        ctx, stx = numpy.cos(tau_x), numpy.sin(tau_x)
        cty, sty = numpy.cos(tau_y), numpy.sin(tau_y)

        Rx = numpy.array([[1, 0, 0], [0, ctx, stx], [0, -stx, ctx]], dtype=float)
        Ry = numpy.array([[cty, 0, -sty], [0, 1, 0], [sty, 0, cty]], dtype=float)
        Rx_dtx = numpy.array([[0, 0, 0], [0, -stx, ctx], [0, -ctx, -stx]], dtype=float)
        Ry_dty = numpy.array([[-sty, 0, -cty], [0, 0, 0], [cty, 0, -sty]], dtype=float)

        def projection(R: numpy.ndarray, derivative: bool) -> numpy.ndarray:
            return numpy.array(
                [
                    [R[2, 2], 0.0, -R[0, 2]],
                    [0.0, R[2, 2], -R[1, 2]],
                    [0.0, 0.0, 0.0 if derivative else 1.0],
                ],
                dtype=float,
            )

        Rxy = Ry @ Rx
        Rxy_dtx = Ry @ Rx_dtx
        Rxy_dty = Ry_dty @ Rx

        Rz = projection(Rxy, derivative=False)
        M = Rz @ Rxy
        M_dtx = Rz @ Rxy_dtx + projection(Rxy_dtx, derivative=True) @ Rxy
        M_dty = Rz @ Rxy_dty + projection(Rxy_dty, derivative=True) @ Rxy
        return M, M_dtx, M_dty

    # =================================================================
    # Implementation of the transform method
    # =================================================================
    def _full_parameters(self) -> numpy.ndarray:
        # Parameters padded with zeros to the 14 coefficients of the full model
        full = numpy.zeros(14, dtype=float)
        full[: self.n_parameters] = self._parameters
        return full

    def _distort(
        self, points: numpy.ndarray, dx: bool, dp: bool
    ) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        # Internal implementation of the distortion (see _apply for the equations)
        k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, tau_x, tau_y = self._full_parameters()
        n_points = points.shape[0]
        x = points[:, 0]
        y = points[:, 1]

        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2
        k_up = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
        k_down = 1.0 + k4 * r2 + k5 * r4 + k6 * r6
        K = k_up / k_down

        # Distortion without tilt
        x0 = x * K + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x) + s1 * r2 + s2 * r4
        y0 = y * K + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y + s3 * r2 + s4 * r4
        distorted = numpy.stack([x0, y0], axis=-1)  # (n_points, 2)

        jacobian_dx = None
        if dx:
            dK_dr2 = (
                (k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4) * k_down
                - k_up * (k4 + 2.0 * k5 * r2 + 3.0 * k6 * r4)
            ) / (k_down * k_down)
            prism_x = s1 + 2.0 * s2 * r2  # d(prism_x)/d(r2)
            prism_y = s3 + 2.0 * s4 * r2  # d(prism_y)/d(r2)
            jacobian_dx = numpy.empty((n_points, 2, 2), dtype=float)
            jacobian_dx[:, 0, 0] = K + 2.0 * x * x * dK_dr2 + 2.0 * p1 * y + 6.0 * p2 * x + 2.0 * x * prism_x
            jacobian_dx[:, 0, 1] = 2.0 * x * y * dK_dr2 + 2.0 * p1 * x + 2.0 * p2 * y + 2.0 * y * prism_x
            jacobian_dx[:, 1, 0] = 2.0 * x * y * dK_dr2 + 2.0 * p1 * x + 2.0 * p2 * y + 2.0 * x * prism_y
            jacobian_dx[:, 1, 1] = K + 2.0 * y * y * dK_dr2 + 6.0 * p1 * y + 2.0 * p2 * x + 2.0 * y * prism_y

        jacobian_dp = None
        if dp:
            jacobian_dp = numpy.zeros((n_points, 2, 14), dtype=float)
            radial = numpy.stack([r2, r4, r6], axis=-1) / k_down[:, None]  # (n_points, 3)
            jacobian_dp[:, 0, [0, 1, 4]] = x[:, None] * radial  # k1, k2, k3
            jacobian_dp[:, 1, [0, 1, 4]] = y[:, None] * radial
            jacobian_dp[:, 0, 5:8] = -(x * K)[:, None] * radial  # k4, k5, k6
            jacobian_dp[:, 1, 5:8] = -(y * K)[:, None] * radial
            jacobian_dp[:, 0, 2] = 2.0 * x * y  # p1
            jacobian_dp[:, 1, 2] = r2 + 2.0 * y * y
            jacobian_dp[:, 0, 3] = r2 + 2.0 * x * x  # p2
            jacobian_dp[:, 1, 3] = 2.0 * x * y
            jacobian_dp[:, 0, 8] = r2  # s1
            jacobian_dp[:, 0, 9] = r4  # s2
            jacobian_dp[:, 1, 10] = r2  # s3
            jacobian_dp[:, 1, 11] = r4  # s4

        # Tilt projection (only for the model with 14 parameters)
        if self.n_parameters == 14:
            M, M_dtx, M_dty = self._compute_tilt_matrix(tau_x, tau_y)
            homogeneous = numpy.column_stack([distorted, numpy.ones(n_points)])  # (n_points, 3)
            h = homogeneous @ M.T  # (n_points, 3)
            distorted = h[:, :2] / h[:, 2:3]  # (n_points, 2)

            if dx or dp:
                # Jacobian of the projection with respect to (x0, y0)
                projection = (
                    M[None, :2, :2] - distorted[:, :, None] * M[None, 2, None, :2]
                ) / h[:, 2, None, None]  # (n_points, 2, 2)
                if dx:
                    jacobian_dx = projection @ jacobian_dx
                if dp:
                    jacobian_dp = projection @ jacobian_dp
                    for column, M_d in ((12, M_dtx), (13, M_dty)):
                        h_d = homogeneous @ M_d.T  # (n_points, 3)
                        jacobian_dp[:, :, column] = (
                            h_d[:, :2] - distorted * h_d[:, 2:3]
                        ) / h[:, 2:3]

        if dp:
            jacobian_dp = jacobian_dp[:, :, : self.n_parameters]

        return distorted, jacobian_dx, jacobian_dp

    def _apply(
        self,
        normalized_points: numpy.ndarray,
        *,
        dx: bool = False,
        dp: bool = False,
        opencv: bool = False,
    ) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        r"""
        Compute the transformation from the **normalized points** to the **distorted points**.

        The distortion without tilt :math:`\vec{x}_0` is given by the OpenCV model (see the class
        documentation). Its Jacobian with respect to the normalized points is, with
        :math:`K' = \frac{dK}{d(r^2)}`:

        .. math::

            \frac{\partial \vec{x}_0}{\partial \vec{x}_n} =
            \begin{bmatrix}
            K + 2 x_n^2 K' + 2 p_1 y_n + 6 p_2 x_n + 2 x_n (s_1 + 2 s_2 r^2) &
            2 x_n y_n K' + 2 p_1 x_n + 2 p_2 y_n + 2 y_n (s_1 + 2 s_2 r^2) \\
            2 x_n y_n K' + 2 p_1 x_n + 2 p_2 y_n + 2 x_n (s_3 + 2 s_4 r^2) &
            K + 2 y_n^2 K' + 6 p_1 y_n + 2 p_2 x_n + 2 y_n (s_3 + 2 s_4 r^2)
            \end{bmatrix}

        and its Jacobian with respect to the coefficients is obtained by direct differentiation,
        for example :math:`\frac{\partial x_0}{\partial k_1} = x_n \frac{r^2}{1 + k_4 r^2 + k_5 r^4 + k_6 r^6}`
        and :math:`\frac{\partial x_0}{\partial k_4} = - x_n K \frac{r^2}{1 + k_4 r^2 + k_5 r^4 + k_6 r^6}`.

        With the tilt parameters, the Jacobians are multiplied by the Jacobian of the
        projection :math:`\vec{x}_0 \mapsto \vec{x}_d`:

        .. math::

            \frac{\partial \vec{x}_d}{\partial \vec{x}_0} = \frac{1}{h_z}
            \begin{bmatrix}
            M_{00} - x_d M_{20} & M_{01} - x_d M_{21} \\
            M_{10} - y_d M_{20} & M_{11} - y_d M_{21}
            \end{bmatrix}

        and the columns of the tilt parameters are
        :math:`\frac{\partial \vec{x}_d}{\partial \tau} = \frac{1}{h_z} \left( \vec{h}'_{xy} - \vec{x}_d h'_z \right)`
        with :math:`\vec{h}' = \frac{\partial M}{\partial \tau} (x_0, y_0, 1)^T`.

        To compute the distortion with OpenCV (``cv2.projectPoints``), set ``opencv=True``.
        In this case, the Jacobian ``dx`` is not available.


        Parameters
        ----------
        normalized_points : numpy.ndarray
            The normalized points. Shape ``(n_points, 2)``.

        dx : bool, optional (default: ``False``)
            If True, the Jacobian with respect to the normalized points is computed.

        dp : bool, optional (default: ``False``)
            If True, the Jacobian with respect to the distortion parameters is computed.

        opencv : bool, optional (default: ``False``)
            If True, the distortion is computed with the OpenCV function ``projectPoints``.


        Returns
        -------
        distorted_points : numpy.ndarray
            The distorted points. Shape ``(n_points, 2)``.

        jacobian_dx : Optional[numpy.ndarray]
            The Jacobian of the distorted points with respect to the normalized points.
            Shape ``(n_points, 2, 2)`` if ``dx=True``, otherwise ``None``.

        jacobian_dp : Optional[numpy.ndarray]
            The Jacobian of the distorted points with respect to the distortion parameters.
            Shape ``(n_points, 2, N)`` if ``dp=True``, otherwise ``None``.
        """
        opencv = _validate_bool(opencv, "opencv")
        if opencv:
            return self._apply_opencv(normalized_points, dx=dx, dp=dp)
        return self._distort(normalized_points, dx, dp)

    def _apply_opencv(
        self, normalized_points: numpy.ndarray, *, dx: bool = False, dp: bool = False
    ) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        # Distortion with cv2.projectPoints (identity extrinsic and intrinsic)
        if dx:
            raise ValueError(
                "The Jacobian with respect to the normalized points cannot be computed with OpenCV."
            )
        n_points = normalized_points.shape[0]
        if self.n_parameters == 0:
            jacobian_dp = numpy.zeros((n_points, 2, 0)) if dp else None
            return normalized_points.copy(), None, jacobian_dp

        object_points = numpy.ascontiguousarray(
            numpy.column_stack([normalized_points, numpy.ones(n_points)]).reshape(-1, 1, 3),
            dtype=float,
        )
        image_points, jacobian = cv2.projectPoints(
            object_points,
            numpy.zeros(3),
            numpy.zeros(3),
            numpy.eye(3),
            self._parameters,
        )  # (n_points, 1, 2), (2 * n_points, 10 + N)
        distorted_points = numpy.asarray(image_points[:, 0, :], dtype=float)

        jacobian_dp = None
        if dp:
            jacobian = numpy.asarray(jacobian, dtype=float)[:, -self.n_parameters :]
            jacobian_dp = jacobian.reshape(n_points, 2, self.n_parameters)

        return distorted_points, None, jacobian_dp

    def _invert(
        self,
        distorted_points: numpy.ndarray,
        *,
        dx: bool = False,
        dp: bool = False,
        opencv: bool = False,
        max_iteration: int = 50,
        tolerance: float = 1e-14,
        pysolvegn: bool = False,
        solve_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        r"""
        Compute the inverse transformation from the **distorted points** to the **normalized points**.

        The normalized points :math:`\vec{x}_n` are the solutions of :math:`D(\vec{x}_n) = \vec{x}_d`,
        where :math:`D` is the distortion. They are computed with Newton iterations, starting from
        :math:`\vec{x}_n^{(0)} = 2 \vec{x}_d - D(\vec{x}_d)`:

        .. math::

            \vec{x}_n^{(k+1)} = \vec{x}_n^{(k)} - \left(\frac{\partial D}{\partial \vec{x}_n}(\vec{x}_n^{(k)})\right)^{-1}
            \left( D(\vec{x}_n^{(k)}) - \vec{x}_d \right)

        until the update of every point is below ``tolerance`` (relatively to the magnitude of the point).

        The Jacobians are obtained by differentiating the identity :math:`D(D^{-1}(\vec{x}_d; \vec{p}); \vec{p}) = \vec{x}_d`:

        .. math::

            \frac{\partial D^{-1}}{\partial \vec{x}_d} = \left( \frac{\partial D}{\partial \vec{x}_n}(\vec{x}_n) \right)^{-1},
            \qquad
            \frac{\partial D^{-1}}{\partial \vec{p}} = - \frac{\partial D^{-1}}{\partial \vec{x}_d} \cdot \frac{\partial D}{\partial \vec{p}}(\vec{x}_n)

        To compute the undistortion with OpenCV (``cv2.undistortPoints``), set ``opencv=True``.
        In this case, the Jacobians are not available.

        To solve the undistortion with ``pysolvegn`` (for example with a Levenberg-Marquardt damping),
        set ``pysolvegn=True`` and give the options of the solver with ``solve_kwargs``.
        In this case, the Jacobians are not available.

        .. note::

            A ``RuntimeWarning`` is emitted if some points do not converge within ``max_iteration``
            iterations, which usually means that the points are outside the domain where the
            distortion is bijective.


        Parameters
        ----------
        distorted_points : numpy.ndarray
            The distorted points. Shape ``(n_points, 2)``.

        dx : bool, optional (default: ``False``)
            If True, the Jacobian with respect to the distorted points is computed.

        dp : bool, optional (default: ``False``)
            If True, the Jacobian with respect to the distortion parameters is computed.

        opencv : bool, optional (default: ``False``)
            If True, the undistortion is computed with the OpenCV function ``undistortPoints``.

        max_iteration : int, optional (default: ``50``)
            The maximum number of Newton iterations.

        tolerance : float, optional (default: ``1e-14``)
            The relative tolerance on the update of the points to stop the iterations.

        pysolvegn : bool, optional (default: ``False``)
            If True, the undistortion is computed with :func:`pysdic.optimize_transformation_input_points`
            (``pysolvegn.solve_batch``) instead of the Newton iterations, starting from the same initial guess.
            In this case, the Jacobians are not available.

        solve_kwargs : Optional[Dict[str, Any]], optional (default: ``None``)
            Only with ``pysolvegn=True``: additional keyword arguments passed to ``pysolvegn.solve_batch``
            (``damping``, ``verbosity``, ...) or to :func:`pysdic.optimize_transformation_input_points`
            (``mode``, ``loss``, ``loss_scale``). By default, ``max_iteration=max_iteration`` and ``xtol=tolerance``.


        Returns
        -------
        normalized_points : numpy.ndarray
            The normalized points. Shape ``(n_points, 2)``.

        jacobian_dx : Optional[numpy.ndarray]
            The Jacobian of the normalized points with respect to the distorted points.
            Shape ``(n_points, 2, 2)`` if ``dx=True``, otherwise ``None``.

        jacobian_dp : Optional[numpy.ndarray]
            The Jacobian of the normalized points with respect to the distortion parameters.
            Shape ``(n_points, 2, N)`` if ``dp=True``, otherwise ``None``.
        """
        opencv = _validate_bool(opencv, "opencv")
        pysolvegn = _validate_bool(pysolvegn, "pysolvegn")
        max_iteration = _validate_integer(max_iteration, "max_iteration", min_value=0)
        tolerance = _validate_real(tolerance, "tolerance", non_negative=True, finite=True)
        if opencv and pysolvegn:
            raise ValueError("opencv and pysolvegn cannot be both True.")
        if opencv:
            if dx or dp:
                raise ValueError(
                    "The Jacobians of the undistortion cannot be computed with OpenCV."
                )
            return self._invert_opencv(distorted_points), None, None

        if pysolvegn:
            if dx or dp:
                raise ValueError(
                    "The Jacobians of the undistortion cannot be computed with pysolvegn."
                )
            guess = 2.0 * distorted_points - self._distort(distorted_points, False, False)[0]
            return (
                self._invert_pysolvegn(distorted_points, guess, max_iteration, tolerance, solve_kwargs),
                None,
                None,
            )

        normalized_points = 2.0 * distorted_points - self._distort(
            distorted_points, False, False
        )[0]  # Initial guess (n_points, 2)
        active = numpy.arange(distorted_points.shape[0])
        for _ in range(max_iteration):
            if active.size == 0:
                break
            values, jacobian, _ = self._distort(normalized_points[active], True, False)
            residuals = values - distorted_points[active]  # (n_active, 2)
            steps = numpy.linalg.solve(jacobian, residuals[:, :, None])[:, :, 0]
            normalized_points[active] -= steps
            converged = numpy.all(
                numpy.abs(steps) <= tolerance * (1.0 + numpy.abs(normalized_points[active])),
                axis=1,
            )
            active = active[~converged]

        if active.size > 0:
            warnings.warn(
                f"The undistortion did not converge for {active.size} point(s) within "
                f"{max_iteration} iterations. The points may be outside the domain where "
                f"the distortion is bijective.",
                RuntimeWarning,
            )

        jacobian_dx = None
        jacobian_dp = None
        if dx or dp:
            _, jacobian_apply_dx, jacobian_apply_dp = self._distort(normalized_points, True, dp)
            jacobian_inverse_dx = numpy.linalg.inv(jacobian_apply_dx)  # (n_points, 2, 2)
            if dx:
                jacobian_dx = jacobian_inverse_dx
            if dp:
                jacobian_dp = -jacobian_inverse_dx @ jacobian_apply_dp  # (n_points, 2, N)

        return normalized_points, jacobian_dx, jacobian_dp

    def _invert_pysolvegn(
        self,
        distorted_points: numpy.ndarray,
        guess: numpy.ndarray,
        max_iteration: int,
        tolerance: float,
        solve_kwargs: Optional[Dict[str, Any]],
    ) -> numpy.ndarray:
        # Undistortion with pysolvegn: D(x_n) = x_d solved as an optimization of the input points of D
        from ..transformation_optimizations.optimize_transformation_input_points import (
            optimize_transformation_input_points,
        )

        solve_kwargs = {
            "max_iteration": max_iteration,
            "xtol": tolerance,
            **_validate_kwargs(solve_kwargs, "solve_kwargs"),
        }

        normalized_points, result = optimize_transformation_input_points(
            self, distorted_points, guess, **solve_kwargs
        )
        n_failed = int(numpy.sum(~result.success))
        if n_failed > 0:
            warnings.warn(
                f"The undistortion with pysolvegn did not converge for {n_failed} point(s). The points may "
                f"be outside the domain where the distortion is bijective.",
                RuntimeWarning,
            )
        return normalized_points

    def _invert_opencv(self, distorted_points: numpy.ndarray) -> numpy.ndarray:
        # Undistortion with cv2.undistortPoints (identity intrinsic, rectification and projection)
        if self.n_parameters == 0:
            return distorted_points.copy()
        points = numpy.ascontiguousarray(distorted_points.reshape(-1, 1, 2), dtype=float)
        normalized_points = cv2.undistortPoints(
            points, numpy.eye(3), self._parameters, numpy.eye(3), numpy.eye(3)
        )  # (n_points, 1, 2)
        return numpy.asarray(normalized_points[:, 0, :], dtype=float)