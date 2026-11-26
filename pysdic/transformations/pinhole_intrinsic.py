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
from .._validation import _validate_bool, _validate_real, _as_array


_SET_HINT = "with the 'parameters' attribute or the 'from_components' class method"


class PinholeIntrinsic(Transformation):
    r"""
    A subclass of :class:`pysdic.Transformation` that represents the OpenCV pinhole
    intrinsic camera model using the standard intrinsic matrix. To enable the skew parameter
    of the intrinsic matrix, set ``use_skew`` to True when instantiating the class.

    This class represents an **intrinsic transformation**, which is typically the
    last step in the camera projection pipeline:

    .. code-block:: text

        world_points → normalized_points → distorted_points → image_points

    Let :math:`\vec{x}_d = (x_d, y_d)` be a **distorted normalized point**.
    The corresponding **image point** :math:`\vec{x}_i = (x_i, y_i)` is obtained using
    the intrinsic matrix :math:`K` in homogeneous coordinates:

    .. math::

        \begin{bmatrix} x_i \\ y_i \\ 1 \end{bmatrix} = K \begin{bmatrix} x_d \\ y_d \\ 1 \end{bmatrix}

    where the intrinsic matrix is defined as

    .. math::

        K =
        \begin{bmatrix}
        f_x & s & c_x \\
        0 & f_y & c_y \\
        0 & 0 & 1
        \end{bmatrix}

    with

    - :math:`f_x, f_y` the focal lengths expressed in pixels,
    - :math:`c_x, c_y` the principal point coordinates in pixels,
    - :math:`s` the skew parameter, only enabled when ``use_skew=True`` (otherwise :math:`s = 0`).

    The transformation is parameterized by **4 to 5 parameters**:

    +-----------------------------------------------------+--------------------------+----------------------------------------------+
    | Class arguments                                     | :math:`N` parameters     | parameters                                   |
    +=====================================================+==========================+==============================================+
    | Default configuration                               | 4                        | :math:`(f_x,f_y,c_x,c_y)`                    |
    +-----------------------------------------------------+--------------------------+----------------------------------------------+
    | ``use_skew == True``                                | 5                        | :math:`(f_x,f_y,c_x,c_y,s)`                  |
    +-----------------------------------------------------+--------------------------+----------------------------------------------+

    The transformation is invertible since the focal lengths are non-zero, and all the
    Jacobians are available for ``apply`` and ``invert``.

    .. note::

        The class can be instantiated in four ways:

        - providing the parameters directly as a numpy array of shape ``(4,)``
          containing :math:`(f_x, f_y, c_x, c_y)`, or ``(5,)`` if
          ``use_skew=True``.
        - using the :meth:`from_matrix` class method to initialize the
          transformation from an intrinsic matrix.
        - using the :meth:`from_vector` class method to initialize the
          transformation from an intrinsic vector.
        - setting the parameters individually using the :meth:`from_components` class method.

    .. note::

        The focal lengths must be non-zero. If ``allow_negative_focal=False`` (default),
        they must also be strictly positive.


    Parameters
    ----------
    parameters : Optional[ArrayLike], optional (default: ``None``)
        Intrinsic parameters of shape ``(4,)`` ordered as
        :math:`(f_x, f_y, c_x, c_y)`, or shape ``(5,)`` ordered as
        :math:`(f_x, f_y, c_x, c_y, s)` if ``use_skew=True``.

    use_skew : bool, optional (default: ``False``)
        If True, the skew parameter is enabled and the transformation is parameterized
        by 5 parameters. If ``False``, the skew parameter is disabled and the transformation
        is parameterized by 4 parameters.

    allow_negative_focal : bool, optional (default: ``False``)
        If True, focal lengths can be negative. If
        ``False``, the focal lengths are constrained to remain strictly positive.


    Examples
    --------
    Create an intrinsic transformation and project a distorted normalized point:

    .. code-block:: python

        intrinsic = PinholeIntrinsic.from_components(fx=1000.0, fy=1000.0, cx=320.0, cy=240.0)

        intrinsic.apply([[0.1, -0.2]]).x      # array([[420., 40.]])
        intrinsic.invert([[420.0, 40.0]]).x   # array([[ 0.1, -0.2]])
        intrinsic.K                           # 3x3 intrinsic matrix

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
        use_skew: bool = False,
        allow_negative_focal: bool = False,
    ) -> None:
        self._use_skew = _validate_bool(use_skew, "use_skew")
        self._allow_negative_focal = _validate_bool(allow_negative_focal, "allow_negative_focal")

        super().__init__(parameters=parameters)

    # =============================================
    # Properties inherited from Transformation Class
    # =============================================
    @property
    def n_parameters(self) -> int:
        r"""
        [Get] the number of parameters of the intrinsic transformation: 4, or 5 if ``use_skew=True``.

        Returns
        -------
        int
            The number of parameters of the intrinsic transformation.

        """
        return 5 if self._use_skew else 4

    @property
    def input_dim(self) -> int:
        r"""
        [Get] the input dimension of the intrinsic transformation (2D distorted normalized points).

        Returns
        -------
        int
            The input dimension of the intrinsic transformation.

        """
        return 2

    @property
    def output_dim(self) -> int:
        r"""
        [Get] the output dimension of the intrinsic transformation (2D image points).

        Returns
        -------
        int
            The output dimension of the intrinsic transformation.

        """
        return 2

    # =============================================
    # I/O Methods for Transformation Class
    # =============================================
    def _to_dict(self) -> Dict[str, Any]:
        return {
            "use_skew": self._use_skew,
            "allow_negative_focal": self._allow_negative_focal,
        }

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> PinholeIntrinsic:
        return cls(
            parameters=data.get("parameters", None),
            use_skew=data.get("use_skew", False),
            allow_negative_focal=data.get("allow_negative_focal", False),
        )

    def _copy(self) -> PinholeIntrinsic:
        return self.__class__(
            parameters=self.parameters,
            use_skew=self._use_skew,
            allow_negative_focal=self._allow_negative_focal,
        )

    # =============================================
    # Parameters
    # =============================================
    def set_parameters(
        self, value: Optional[ArrayLike], *, override_lock: bool = False
    ) -> None:
        r"""
        Set the parameters of the intrinsic transformation, with an option to override the lock.

        In addition to the checks of :meth:`pysdic.Transformation.set_parameters`, the parameters
        must be finite and the focal lengths must be non-zero (and strictly positive if
        ``allow_negative_focal=False``).

        Parameters
        ----------
        value : Optional[ArrayLike]
            The parameters of shape ``(4,)`` ordered as :math:`(f_x, f_y, c_x, c_y)`, or shape ``(5,)``
            ordered as :math:`(f_x, f_y, c_x, c_y, s)` if ``use_skew=True``, or None to unset the parameters.

        override_lock : bool, optional (default: ``False``)
            Whether to override the lock and allow setting parameters even if they are locked.

        Raises
        ------
        ValueError
            If the parameters do not have the correct shape, are not finite or if the focal
            lengths are not valid.

        RuntimeError
            If the parameters are locked and ``override_lock`` is False.

        """
        if value is not None:
            value = self._validate_parameters(value)  # Shape (n_p,) and finite
            self._check_focal(value[0], "x")
            self._check_focal(value[1], "y")
        super().set_parameters(value, override_lock=override_lock)

    def _check_focal(self, focal: float, direction: str) -> None:
        # The focal lengths must be non-zero for the transformation to be invertible
        if focal == 0.0:
            raise ValueError(
                f"Focal length in pixels in {direction} direction must be non-zero."
            )
        if not self._allow_negative_focal and focal < 0.0:
            raise ValueError(
                f"Focal length in pixels in {direction} direction must be greater than 0, "
                f"got {focal}. Use allow_negative_focal=True to allow negative focal lengths."
            )

    def _get_component(self, index: int) -> Optional[float]:
        # Return one component of the parameters, or None if the parameters are not set
        parameters = self.parameters
        if parameters is None:
            return None
        return float(parameters[index])

    @property
    def parameter_names(self) -> List[str]:
        r"""
        [Get] the names of the parameters of the intrinsic transformation :
        ``["fx", "fy", "cx", "cy"]`` or ``["fx", "fy", "cx", "cy", "s"]`` if ``use_skew=True``.

        Returns
        -------
        List[str]
            The names of the parameters of the intrinsic transformation.
        """
        names = ["fx", "fy", "cx", "cy"]
        if self._use_skew:
            names.append("s")
        return names

    # =============================================
    # Constants
    # =============================================
    @property
    def use_skew(self) -> bool:
        r"""
        [Get] whether the skew parameter is enabled in the intrinsic transformation.

        If True, the skew parameter is enabled and the transformation is parameterized
        by 5 parameters. If False, the skew parameter is disabled and the transformation
        is parameterized by 4 parameters.

        Returns
        -------
        bool
            Whether the skew parameter is enabled in the intrinsic transformation.
        """
        return self._use_skew

    @property
    def allow_negative_focal(self) -> bool:
        r"""
        [Get] whether the focal lengths can be negative in the model.

        Returns
        -------
        bool
            Whether the focal lengths can be negative in the model.
        """
        return self._allow_negative_focal

    # =============================================
    # Focal length
    # =============================================
    @property
    def focal_length_x(self) -> Optional[float]:
        r"""
        [Get/Set] the focal length :math:`f_x` of the intrinsic transformation.

        The focal length is a float representing the focal length of the camera in pixels in x direction.

        This parameter is the component ``K[0, 0]`` of the intrinsic matrix K of the camera.

        .. note::

            An alias for ``focal_length_x`` is ``fx``.

        .. note::

            The component can only be set if all the parameters are already set.

        .. seealso::

            - :meth:`focal_length_y` or ``fy`` to set the focal length in pixels in y direction.

        Returns
        -------
        Optional[float]
            The focal length of the camera in pixels in x direction. (or None if not set)
        """
        return self._get_component(0)

    @focal_length_x.setter
    def focal_length_x(self, fx: Real) -> None:
        self._set_parameters_at(0, fx, "focal_length_x", hint=_SET_HINT)

    @property
    def fx(self) -> Optional[float]:
        return self.focal_length_x

    @fx.setter
    def fx(self, fx: Real) -> None:
        self.focal_length_x = fx

    @property
    def focal_length_y(self) -> Optional[float]:
        r"""
        [Get/Set] the focal length :math:`f_y` of the intrinsic transformation.

        The focal length is a float representing the focal length of the camera in pixels in y direction.

        This parameter is the component ``K[1, 1]`` of the intrinsic matrix K of the camera.

        .. note::

            An alias for ``focal_length_y`` is ``fy``.

        .. note::

            The component can only be set if all the parameters are already set.

        .. seealso::

            - :meth:`focal_length_x` or ``fx`` to set the focal length in pixels in x direction.

        Returns
        -------
        Optional[float]
            The focal length of the camera in pixels in y direction. (or None if not set)
        """
        return self._get_component(1)

    @focal_length_y.setter
    def focal_length_y(self, fy: Real) -> None:
        self._set_parameters_at(1, fy, "focal_length_y", hint=_SET_HINT)

    @property
    def fy(self) -> Optional[float]:
        return self.focal_length_y

    @fy.setter
    def fy(self, fy: Real) -> None:
        self.focal_length_y = fy

    # =============================================
    # Principal point
    # =============================================
    @property
    def principal_point_x(self) -> Optional[float]:
        r"""
        [Get/Set] the principal point :math:`c_x` of the intrinsic transformation.

        The principal point is a float representing the principal point of the camera in pixels in x direction.

        This parameter is the component ``K[0, 2]`` of the intrinsic matrix K of the camera.

        .. note::

            An alias for ``principal_point_x`` is ``cx``.

        .. note::

            The component can only be set if all the parameters are already set.

        .. seealso::

            - :meth:`principal_point_y` or ``cy`` to set the principal point in pixels in y direction.

        Returns
        -------
        Optional[float]
            The principal point of the camera in pixels in x direction. (or None if not set)
        """
        return self._get_component(2)

    @principal_point_x.setter
    def principal_point_x(self, cx: Real) -> None:
        self._set_parameters_at(2, cx, "principal_point_x", hint=_SET_HINT)

    @property
    def cx(self) -> Optional[float]:
        return self.principal_point_x

    @cx.setter
    def cx(self, cx: Real) -> None:
        self.principal_point_x = cx

    @property
    def principal_point_y(self) -> Optional[float]:
        r"""
        [Get/Set] the principal point :math:`c_y` of the intrinsic transformation.

        The principal point is a float representing the principal point of the camera in pixels in y direction.

        This parameter is the component ``K[1, 2]`` of the intrinsic matrix K of the camera.

        .. note::

            An alias for ``principal_point_y`` is ``cy``.

        .. note::

            The component can only be set if all the parameters are already set.

        .. seealso::

            - :meth:`principal_point_x` or ``cx`` to set the principal point in pixels in x direction.

        Returns
        -------
        Optional[float]
            The principal point of the camera in pixels in y direction. (or None if not set)
        """
        return self._get_component(3)

    @principal_point_y.setter
    def principal_point_y(self, cy: Real) -> None:
        self._set_parameters_at(3, cy, "principal_point_y", hint=_SET_HINT)

    @property
    def cy(self) -> Optional[float]:
        return self.principal_point_y

    @cy.setter
    def cy(self, cy: Real) -> None:
        self.principal_point_y = cy

    # ============================================
    # Skew parameter
    # ============================================
    @property
    def skew(self) -> Optional[float]:
        r"""
        [Get/Set] the skew parameter :math:`s` of the intrinsic transformation.

        The skew parameter is a float representing the skew of the camera. It is only enabled if ``use_skew=True``.

        This parameter is the component ``K[0, 1]`` of the intrinsic matrix K of the camera.

        .. note::

            An alias for ``skew`` is ``s``.

        .. note::

            The component can only be set if all the parameters are already set.

        Returns
        -------
        Optional[float]
            The skew parameter of the camera. (or None if not set)

        Raises
        ------
        AttributeError
            If the skew parameter is not enabled (``use_skew=False``).
        """
        if not self._use_skew:
            raise AttributeError(
                "Skew parameter is not enabled. Set use_skew=True to enable it."
            )
        return self._get_component(4)

    @skew.setter
    def skew(self, s: Real) -> None:
        if not self._use_skew:
            raise AttributeError(
                "Skew parameter is not enabled. Set use_skew=True to enable it."
            )
        self._set_parameters_at(4, s, "skew", hint=_SET_HINT)

    @property
    def s(self) -> Optional[float]:
        return self.skew

    @s.setter
    def s(self, s: Real) -> None:
        self.skew = s

    # =============================================
    # Intrinsic matrix
    # =============================================
    @property
    def intrinsic_matrix(self) -> Optional[numpy.ndarray]:
        r"""
        [Get/Set] the intrinsic matrix :math:`K` of the intrinsic transformation.

        The intrinsic matrix is a 3x3 matrix representing the intrinsic parameters of the camera.

        .. math::

            K = \begin{bmatrix}
            f_x & s & c_x \\
            0 & f_y & c_y \\
            0 & 0 & 1
            \end{bmatrix}

        .. note::

            An alias for ``intrinsic_matrix`` is ``K``.

        .. seealso::

            - :meth:`intrinsic_vector` or ``k`` to get the intrinsic vector of the camera.


        Parameters
        ----------
        intrinsic_matrix : Optional[ArrayLike]
            The intrinsic matrix :math:`K` of the camera with shape (3, 3), or None to unset the parameters.
            If ``use_skew=False``, the component ``K[0, 1]`` must be zero.


        Returns
        -------
        Optional[numpy.ndarray]
            The intrinsic matrix :math:`K` of the camera with shape (3, 3). (or None if the parameters are not set)
        """
        parameters = self.parameters
        if parameters is None:
            return None
        fx, fy, cx, cy = parameters[:4]
        s = parameters[4] if self._use_skew else 0.0
        return numpy.array(
            [
                [fx, s, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

    @intrinsic_matrix.setter
    def intrinsic_matrix(self, intrinsic_matrix: Optional[ArrayLike]) -> None:
        if intrinsic_matrix is None:
            self.parameters = None
            return
        intrinsic_matrix = _as_array(
            intrinsic_matrix, "intrinsic_matrix", numpy.float64, shape=(3, 3), finite=True
        )
        if not self._use_skew and abs(intrinsic_matrix[0, 1]) > 1e-6:
            raise ValueError(
                "Skew value K[0, 1] is not supported by PinholeIntrinsic if use_skew is False."
            )
        if (
            abs(intrinsic_matrix[1, 0]) > 1e-6
            or abs(intrinsic_matrix[2, 0]) > 1e-6
            or abs(intrinsic_matrix[2, 1]) > 1e-6
            or abs(intrinsic_matrix[2, 2] - 1.0) > 1e-6
        ):
            raise ValueError(
                "The intrinsic matrix must have the form [[fx, s, cx], [0, fy, cy], [0, 0, 1]]."
            )
        parameters = [
            intrinsic_matrix[0, 0],
            intrinsic_matrix[1, 1],
            intrinsic_matrix[0, 2],
            intrinsic_matrix[1, 2],
        ]
        if self._use_skew:
            parameters.append(intrinsic_matrix[0, 1])
        self.parameters = parameters

    @property
    def K(self) -> Optional[numpy.ndarray]:
        return self.intrinsic_matrix

    @K.setter
    def K(self, intrinsic_matrix: Optional[ArrayLike]) -> None:
        self.intrinsic_matrix = intrinsic_matrix

    # =============================================
    # Intrinsic vector
    # =============================================
    @property
    def intrinsic_vector(self) -> Optional[numpy.ndarray]:
        r"""
        [Get/Set] the intrinsic vector :math:`k` of the intrinsic transformation.

        The intrinsic vector is the vector of the parameters of the transformation, of shape
        (4,) or (5,) if the skew is used:

        .. math::

            k = \begin{bmatrix}
            f_x & f_y & c_x & c_y & (s)
            \end{bmatrix}^T

        .. note::

            An alias for ``intrinsic_vector`` is ``k``. It is equivalent to ``parameters``.

        .. seealso::

            - :meth:`intrinsic_matrix` or ``K`` to set the intrinsic matrix of the camera.

        Parameters
        ----------
        intrinsic_vector : Optional[ArrayLike]
            The intrinsic vector of the camera with shape (4,) or (5,) ordered as
            :math:`(f_x, f_y, c_x, c_y)` or :math:`(f_x, f_y, c_x, c_y, s)` if ``use_skew=True``.


        Returns
        -------
        Optional[numpy.ndarray]
            The intrinsic vector of the camera. (or None if the parameters are not set)
        """
        return self.parameters

    @intrinsic_vector.setter
    def intrinsic_vector(self, intrinsic_vector: Optional[ArrayLike]) -> None:
        self.parameters = intrinsic_vector

    @property
    def k(self) -> Optional[numpy.ndarray]:
        return self.intrinsic_vector

    @k.setter
    def k(self, intrinsic_vector: Optional[ArrayLike]) -> None:
        self.intrinsic_vector = intrinsic_vector

    # =============================================
    # Instantiation methods
    # =============================================
    @classmethod
    def from_matrix(cls, matrix: ArrayLike, **kwargs) -> PinholeIntrinsic:
        r"""
        Class method to create a :class:`pysdic.PinholeIntrinsic` object from an intrinsic
        matrix :math:`K`.

        Parameters
        ----------
        matrix : ArrayLike
            The intrinsic matrix :math:`K` of the camera with shape (3, 3).

        **kwargs
            Additional keyword arguments for the class initialisation (``use_skew``, ``allow_negative_focal``).


        Returns
        -------
        :class:`pysdic.PinholeIntrinsic`
            A new instance of the :class:`pysdic.PinholeIntrinsic` class with the specified intrinsic matrix.

        """
        intrinsic = cls(parameters=None, **kwargs)
        intrinsic.intrinsic_matrix = matrix
        return intrinsic

    @classmethod
    def from_vector(cls, vector: ArrayLike, **kwargs) -> PinholeIntrinsic:
        r"""
        Class method to create a :class:`pysdic.PinholeIntrinsic` object from an intrinsic vector :math:`k`.

        Parameters
        ----------
        vector : ArrayLike
            The intrinsic vector of the camera with shape (4,) ordered as :math:`(f_x, f_y, c_x, c_y)`
            or shape (5,) ordered as :math:`(f_x, f_y, c_x, c_y, s)` if ``use_skew=True``.

        **kwargs
            Additional keyword arguments for the class initialisation (``use_skew``, ``allow_negative_focal``).


        Returns
        -------
        :class:`pysdic.PinholeIntrinsic`
            A new instance of the :class:`pysdic.PinholeIntrinsic` class with the specified intrinsic vector.

        """
        return cls(parameters=vector, **kwargs)

    @classmethod
    def from_components(
        cls,
        fx: Real,
        fy: Real,
        cx: Real,
        cy: Real,
        s: Optional[Real] = None,
        **kwargs,
    ) -> PinholeIntrinsic:
        r"""
        Class method to create a :class:`pysdic.PinholeIntrinsic` object from the individual
        components of the intrinsic parameters.

        Parameters
        ----------
        fx : Real
            The focal length of the camera in pixels in x direction.

        fy : Real
            The focal length of the camera in pixels in y direction.

        cx : Real
            The principal point of the camera in pixels in x direction.

        cy : Real
            The principal point of the camera in pixels in y direction.

        s : Optional[Real], optional (default: ``None``)
            The skew parameter of the camera. Only allowed if ``use_skew=True``,
            in which case None means a zero skew.

        **kwargs
            Additional keyword arguments for the class initialisation (``use_skew``, ``allow_negative_focal``).


        Returns
        -------
        :class:`pysdic.PinholeIntrinsic`
            A new instance of the :class:`pysdic.PinholeIntrinsic` class with the specified intrinsic parameters.

        """
        use_skew = _validate_bool(kwargs.get("use_skew", False), "use_skew")
        fx = _validate_real(fx, "fx", finite=True)
        fy = _validate_real(fy, "fy", finite=True)
        cx = _validate_real(cx, "cx", finite=True)
        cy = _validate_real(cy, "cy", finite=True)
        s = _validate_real(s, "s", finite=True, allow_none=True)
        if s is not None and not use_skew:
            raise ValueError(
                "Skew parameter is not supported by PinholeIntrinsic if use_skew is False."
            )
        parameters = [fx, fy, cx, cy]
        if use_skew:
            parameters.append(0.0 if s is None else s)
        return cls(parameters=parameters, **kwargs)

    # =============================================
    # Transform methods
    # =============================================
    def _linear_part(self) -> numpy.ndarray:
        # Upper-left 2x2 block A of K, such that x_i = A x_d + c
        fx, fy = self._parameters[0], self._parameters[1]
        s = self._parameters[4] if self._use_skew else 0.0
        return numpy.array([[fx, s], [0.0, fy]], dtype=float)

    def _apply(
        self,
        distorted_points: numpy.ndarray,
        *,
        dx: bool = False,
        dp: bool = False,
    ) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        r"""
        Apply the intrinsic projection from distorted normalized coordinates
        to image coordinates.

        Let :math:`\vec{x}_d = (x_d, y_d)` be a point in distorted normalized
        camera coordinates. The corresponding image point
        :math:`\vec{x}_i = (x_i, y_i)` is obtained using the intrinsic parameters:

        .. math::

            x_i = f_x \cdot x_d + s \cdot y_d + c_x

        .. math::

            y_i = f_y \cdot y_d + c_y

        The Jacobian with respect to the distorted points is constant:

        .. math::

            \frac{\partial \vec{x}_i}{\partial \vec{x}_d} =
            \begin{bmatrix}
            f_x & s \\
            0 & f_y
            \end{bmatrix}

        The Jacobian with respect to the intrinsic parameters :math:`(f_x, f_y, c_x, c_y, s)` is:

        .. math::

            \frac{\partial \vec{x}_i}{\partial (f_x, f_y, c_x, c_y, s)} =
            \begin{bmatrix}
            x_d & 0 & 1 & 0 & y_d \\
            0 & y_d & 0 & 1 & 0
            \end{bmatrix}

        where the last column only exists if ``use_skew=True``.


        Parameters
        ----------
        distorted_points : numpy.ndarray
            Distorted normalized camera coordinates.
            Shape ``(n_points, 2)``.

        dx : bool, optional (default: ``False``)
            If True, compute the Jacobian with respect to the input points.

        dp : bool, optional (default: ``False``)
            If True, compute the Jacobian with respect to the intrinsic
            parameters.


        Returns
        -------
        image_points : numpy.ndarray
            Image coordinates computed from the intrinsic projection.
            Shape ``(n_points, 2)``.

        jacobian_dx : Optional[numpy.ndarray]
            Jacobian of the image points with respect to the distorted points.
            Shape ``(n_points, 2, 2)`` if ``dx=True``, otherwise ``None``.

        jacobian_dp : Optional[numpy.ndarray]
            Jacobian of the image points with respect to the intrinsic
            parameters.
            Shape ``(n_points, 2, n_parameters)`` if ``dp=True``, otherwise ``None``.
        """
        n_points = distorted_points.shape[0]
        A = self._linear_part()  # (2, 2)
        c = self._parameters[2:4]  # (2,)

        image_points = distorted_points @ A.T + c  # (n_points, 2)

        jacobian_dx = None
        if dx:
            jacobian_dx = numpy.tile(A, (n_points, 1, 1))  # (n_points, 2, 2)

        jacobian_dp = None
        if dp:
            x_d = distorted_points[:, 0]
            y_d = distorted_points[:, 1]
            jacobian_dp = numpy.zeros(
                (n_points, 2, self.n_parameters), dtype=float
            )  # (n_points, 2, n_parameters)
            jacobian_dp[:, 0, 0] = x_d  # d x_i / d fx
            jacobian_dp[:, 1, 1] = y_d  # d y_i / d fy
            jacobian_dp[:, 0, 2] = 1.0  # d x_i / d cx
            jacobian_dp[:, 1, 3] = 1.0  # d y_i / d cy
            if self._use_skew:
                jacobian_dp[:, 0, 4] = y_d  # d x_i / d s

        return image_points, jacobian_dx, jacobian_dp

    def _invert(
        self,
        image_points: numpy.ndarray,
        *,
        dx: bool = False,
        dp: bool = False,
    ) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        r"""
        Apply the inverse of the intrinsic projection from image coordinates
        to distorted normalized coordinates.

        Let :math:`\vec{x}_i = (x_i, y_i)` be a point in image coordinates. The
        corresponding distorted normalized camera point :math:`\vec{x}_d` is
        obtained using the intrinsic parameters:

        .. math::

            y_d = \frac{y_i - c_y}{f_y}

        .. math::

            x_d = \frac{x_i - s \cdot y_d - c_x}{f_x}

        The Jacobian with respect to the image points is constant:

        .. math::

            \frac{\partial \vec{x}_d}{\partial \vec{x}_i} =
            \begin{bmatrix}
            \frac{1}{f_x} & -\frac{s}{f_x f_y} \\
            0 & \frac{1}{f_y}
            \end{bmatrix}

        The Jacobian with respect to the intrinsic parameters :math:`(f_x, f_y, c_x, c_y, s)` is:

        .. math::

            \frac{\partial \vec{x}_d}{\partial (f_x, f_y, c_x, c_y, s)} =
            \begin{bmatrix}
            -\frac{x_d}{f_x} & \frac{s \, y_d}{f_x f_y} & -\frac{1}{f_x} & \frac{s}{f_x f_y} & -\frac{y_d}{f_x} \\
            0 & -\frac{y_d}{f_y} & 0 & -\frac{1}{f_y} & 0
            \end{bmatrix}

        where the last column only exists if ``use_skew=True`` (and :math:`s = 0` otherwise).


        Parameters
        ----------
        image_points : numpy.ndarray
            Image coordinates.
            Shape ``(n_points, 2)``.

        dx : bool, optional (default: ``False``)
            If True, compute the Jacobian with respect to the image points.

        dp : bool, optional (default: ``False``)
            If True, compute the Jacobian with respect to the intrinsic
            parameters.


        Returns
        -------
        distorted_points : numpy.ndarray
            Distorted normalized camera coordinates.
            Shape ``(n_points, 2)``.

        jacobian_dx : Optional[numpy.ndarray]
            Jacobian of the distorted points with respect to the image points.
            Shape ``(n_points, 2, 2)`` if ``dx=True``, otherwise ``None``.

        jacobian_dp : Optional[numpy.ndarray]
            Jacobian of the distorted points with respect to the intrinsic
            parameters.
            Shape ``(n_points, 2, n_parameters)`` if ``dp=True``, otherwise ``None``.
        """
        n_points = image_points.shape[0]
        fx, fy, cx, cy = self._parameters[:4]
        s = self._parameters[4] if self._use_skew else 0.0

        y_d = (image_points[:, 1] - cy) / fy
        x_d = (image_points[:, 0] - s * y_d - cx) / fx
        distorted_points = numpy.stack([x_d, y_d], axis=-1)  # (n_points, 2)

        jacobian_dx = None
        if dx:
            A_inv = numpy.array(
                [[1.0 / fx, -s / (fx * fy)], [0.0, 1.0 / fy]], dtype=float
            )
            jacobian_dx = numpy.tile(A_inv, (n_points, 1, 1))  # (n_points, 2, 2)

        jacobian_dp = None
        if dp:
            jacobian_dp = numpy.zeros(
                (n_points, 2, self.n_parameters), dtype=float
            )  # (n_points, 2, n_parameters)
            jacobian_dp[:, 0, 0] = -x_d / fx  # d x_d / d fx
            jacobian_dp[:, 0, 1] = s * y_d / (fx * fy)  # d x_d / d fy
            jacobian_dp[:, 0, 2] = -1.0 / fx  # d x_d / d cx
            jacobian_dp[:, 0, 3] = s / (fx * fy)  # d x_d / d cy
            jacobian_dp[:, 1, 1] = -y_d / fy  # d y_d / d fy
            jacobian_dp[:, 1, 3] = -1.0 / fy  # d y_d / d cy
            if self._use_skew:
                jacobian_dp[:, 0, 4] = -y_d / fx  # d x_d / d s

        return distorted_points, jacobian_dx, jacobian_dp