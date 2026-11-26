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
from numpy.typing import ArrayLike
from numbers import Integral

import numpy

from .transformation import Transformation
from .._validation import _validate_integer


class IdentityTransformation(Transformation):
    r"""
    The identity transformation :math:`T` that maps input points to output points without any change.

    .. math::

        \vec{x_o} = T(\vec{x_i}) = \vec{x_i}

    This transformation has no parameters and serves as a neutral element in transformation compositions.

    When the input and output dimensions differ, the transformation is the canonical
    projection or embedding between the two spaces:

    - If :math:`n_o < n_i`, the last :math:`n_i - n_o` coordinates are dropped (projection).
    - If :math:`n_o > n_i`, the last :math:`n_o - n_i` coordinates are set to zero (embedding).

    .. math::

        \vec{x_o}[k] = \begin{cases}
        \vec{x_i}[k] & \text{if } k < \min(n_i, n_o) \\
        0 & \text{otherwise}
        \end{cases}

    The inverse map :math:`T^{-1}` is the identity transformation from :math:`\mathbb{R}^{n_o}`
    to :math:`\mathbb{R}^{n_i}`. It is a left inverse if :math:`n_i \leq n_o` and a right inverse
    if :math:`n_i \geq n_o`.

    The Jacobian with respect to the points is the constant matrix
    :math:`\mathbf{J}_x = \mathbf{I}_{n_o \times n_i}` (the identity matrix truncated or padded
    with zeros), and the Jacobian with respect to the parameters is empty with
    shape :math:`(n_o, 0)`.

    .. seealso::

        - Class :class:`pysdic.Transformation` for the base class of transformations.


    Parameters
    ----------
    input_dim : Integral
        The dimension :math:`n_i` of the input points (strictly positive).

    output_dim : Optional[Integral], optional (default: ``None``)
        The dimension :math:`n_o` of the output points (strictly positive).
        If None, the output dimension is equal to the input dimension.


    Examples
    --------
    The identity in 2D and the embedding of 2D points into 3D:

    .. code-block:: python

        identity = IdentityTransformation(input_dim=2)
        identity.apply([[1.0, 2.0]]).x     # array([[1., 2.]])

        embedding = IdentityTransformation(input_dim=2, output_dim=3)
        embedding.apply([[1.0, 2.0]]).x    # array([[1., 2., 0.]])
        embedding.invert([[1.0, 2.0, 3.0]]).x    # array([[1., 2.]])

    """

    _has_apply = True
    _has_apply_dx = True
    _has_apply_dp = True
    _has_invert = True
    _has_invert_dx = True
    _has_invert_dp = True

    def __init__(
        self,
        input_dim: Integral,
        output_dim: Optional[Integral] = None,
        *,
        parameters: Optional[ArrayLike] = None,
    ) -> None:
        input_dim = _validate_integer(input_dim, "input_dim", min_value=1)
        output_dim = _validate_integer(
            output_dim, "output_dim", min_value=1, allow_none=True
        )
        if output_dim is None:
            output_dim = input_dim

        self._input_dim = input_dim
        self._output_dim = output_dim

        # The inverse map is a left inverse if no coordinate is dropped by apply,
        # and a right inverse if no coordinate is dropped by invert
        self._is_left_invertible = self._input_dim <= self._output_dim
        self._is_right_invertible = self._input_dim >= self._output_dim

        super().__init__(parameters=parameters)

    # ------
    # Properties
    # ------
    @property
    def n_parameters(self) -> int:
        r"""
        [Get] the number of parameters :math:`n_p` of the transformation (always 0).

        Returns
        -------
        int
            The number of parameters of the transformation.

        """
        return 0

    @property
    def input_dim(self) -> int:
        r"""
        [Get] the input dimension :math:`n_i` of the transformation.

        Returns
        -------
        int
            The input dimension of the transformation.

        """
        return self._input_dim

    @property
    def output_dim(self) -> int:
        r"""
        [Get] the output dimension :math:`n_o` of the transformation.

        Returns
        -------
        int
            The output dimension of the transformation.

        """
        return self._output_dim

    # ------
    # Serialization and copy
    # ------
    def _to_dict(self) -> Dict[str, Any]:
        return {"input_dim": self._input_dim, "output_dim": self._output_dim}

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> IdentityTransformation:
        if not "input_dim" in data:
            raise ValueError("Input dictionary must contain an 'input_dim' key.")
        return cls(
            input_dim=data["input_dim"],
            output_dim=data.get("output_dim", None),
        )

    def _copy(self) -> IdentityTransformation:
        return self.__class__(input_dim=self._input_dim, output_dim=self._output_dim)

    # ------
    # Apply and invert
    # ------
    @staticmethod
    def _identity(
        x: numpy.ndarray, dim_in: int, dim_out: int, dx: bool, dp: bool
    ) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        # Identity (truncated or padded with zeros) from R^dim_in to R^dim_out
        n_points = x.shape[0]
        n_common = min(dim_in, dim_out)

        y = numpy.zeros((n_points, dim_out), dtype=float)
        y[:, :n_common] = x[:, :n_common]  # (n_points, dim_out)

        jacobian_dx = None
        if dx:
            jacobian_dx = numpy.zeros((n_points, dim_out, dim_in), dtype=float)
            jacobian_dx[:, numpy.arange(n_common), numpy.arange(n_common)] = 1.0  # (n_points, dim_out, dim_in)

        jacobian_dp = (
            numpy.zeros((n_points, dim_out, 0), dtype=float) if dp else None
        )  # (n_points, dim_out, 0)

        return y, jacobian_dx, jacobian_dp

    def _apply(
        self,
        x_i: numpy.ndarray,
        *,
        dx: bool = False,
        dp: bool = False,
    ) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        r"""
        Apply the identity transformation from :math:`\mathbb{R}^{n_i}` to :math:`\mathbb{R}^{n_o}`.

        .. math::

            \vec{x_o} = \mathbf{I}_{n_o \times n_i} \vec{x_i}

        Parameters
        ----------
        x_i : numpy.ndarray
            The input points with shape :math:`(n_\text{points}, n_i)`.

        dx : bool, optional (default: ``False``)
            If True, compute the Jacobian with respect to the input points.

        dp : bool, optional (default: ``False``)
            If True, compute the (empty) Jacobian with respect to the parameters.


        Returns
        -------
        Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]
            A tuple containing:
            - The output points with shape :math:`(n_\text{points}, n_o)`.
            - The Jacobian with respect to input points with shape :math:`(n_\text{points}, n_o, n_i)` if computed, otherwise None.
            - The Jacobian with respect to parameters with shape :math:`(n_\text{points}, n_o, 0)` if computed, otherwise None.

        """
        return self._identity(x_i, self._input_dim, self._output_dim, dx, dp)

    def _invert(
        self,
        x_o: numpy.ndarray,
        *,
        dx: bool = False,
        dp: bool = False,
    ) -> Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]:
        r"""
        Apply the identity transformation from :math:`\mathbb{R}^{n_o}` to :math:`\mathbb{R}^{n_i}`.

        .. math::

            \vec{x_i} = \mathbf{I}_{n_i \times n_o} \vec{x_o}

        Parameters
        ----------
        x_o : numpy.ndarray
            The output points with shape :math:`(n_\text{points}, n_o)`.

        dx : bool, optional (default: ``False``)
            If True, compute the Jacobian with respect to the output points.

        dp : bool, optional (default: ``False``)
            If True, compute the (empty) Jacobian with respect to the parameters.


        Returns
        -------
        Tuple[numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]]
            A tuple containing:
            - The input points with shape :math:`(n_\text{points}, n_i)`.
            - The Jacobian with respect to output points with shape :math:`(n_\text{points}, n_i, n_o)` if computed, otherwise None.
            - The Jacobian with respect to parameters with shape :math:`(n_\text{points}, n_i, 0)` if computed, otherwise None.

        """
        return self._identity(x_o, self._output_dim, self._input_dim, dx, dp)