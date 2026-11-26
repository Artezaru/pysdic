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

import numpy

from .parametrization import Parametrization
from .._validation import _as_array


class AffineParametrization(Parametrization):
    r"""
    A parametrization :math:`P` defined by an affine map from the input parametric
    space :math:`\mathbb{R}^{n_i}` to the output parametric space :math:`\mathbb{R}^{n_o}`.

    .. math::

        \vec{p_o} = P(\vec{p_i}) = \mathbf{M} \vec{p_i} + \vec{p_0}

    where :math:`\mathbf{M}` is the matrix of the modes with shape :math:`(n_o, n_i)`
    and :math:`\vec{p_0}` is the offset with shape :math:`(n_o,)`. Each column of
    :math:`\mathbf{M}` is a mode, i.e. the direction of the output space driven by
    one input parameter.

    The Jacobian of the parametrization is constant:

    .. math::

        \mathbf{J}_p = \frac{\partial P}{\partial \vec{p_i}} = \mathbf{M}

    If :math:`\mathbf{M}` is invertible (square with full rank), the parametrization
    can be decoded with:

    .. math::

        \vec{p_i} = P^{-1}(\vec{p_o}) = \mathbf{M}^{-1} (\vec{p_o} - \vec{p_0})

    and the parametrization is both left and right invertible. Otherwise, the
    ``decode`` method is not available.

    .. seealso::

        - Class :class:`pysdic.Parametrization` for the base class of parametrizations.

    .. note::

        The ``modes`` and ``offset`` are **constants** of the parametrization. They are
        stored as read-only copies and cannot be modified after initialization.


    Parameters
    ----------
    modes : ArrayLike
        The matrix of the modes :math:`\mathbf{M}` with shape :math:`(n_o, n_i)`.

    offset : Optional[ArrayLike], optional (default: ``None``)
        The offset :math:`\vec{p_0}` with shape :math:`(n_o,)`.
        If None, a zero offset is used.


    Examples
    --------
    A parametrization of two output parameters driven by a single input parameter:

    .. code-block:: python

        parametrization = AffineParametrization(
            modes=[[1.0], [2.0]],
            offset=[0.0, 1.0],
        )

        parametrization.encode([3.0]).p   # array([3., 7.])
        parametrization.has_decode        # False (M is not square)

    This defines:

    .. math::

        \begin{bmatrix}
        p_{o,1} \\
        p_{o,2}
        \end{bmatrix}
        =
        \begin{bmatrix}
        1 \\
        2
        \end{bmatrix}
        p_{i}
        +
        \begin{bmatrix}
        0 \\
        1
        \end{bmatrix}

    With an invertible matrix, the parametrization can also be decoded:

    .. code-block:: python

        parametrization = AffineParametrization(
            modes=[[2.0, 0.0], [1.0, 1.0]],
        )

        parametrization.encode([1.0, 1.0]).p   # array([2., 2.])
        parametrization.decode([2.0, 2.0]).p   # array([1., 1.])

    """

    _has_encode = True
    _has_encode_dp = True

    def __init__(
        self,
        modes: ArrayLike,
        offset: Optional[ArrayLike] = None,
    ) -> None:
        modes = _as_array(
            modes, "modes", numpy.float64,
            shape=(None, None), shape_hint="(n_o, n_i)", finite=True, copy=True,
        )
        if modes.shape[0] == 0 or modes.shape[1] == 0:
            raise ValueError(
                f"Input 'modes' must have a non-zero shape. Got shape {modes.shape}."
            )
        n_o = modes.shape[0]

        if offset is None:
            offset = numpy.zeros(n_o, dtype=numpy.float64)
        else:
            offset = _as_array(offset, "offset", numpy.float64, shape=(n_o,), finite=True, copy=True)

        # Store the constants as read-only arrays
        modes.flags.writeable = False
        offset.flags.writeable = False
        self._modes = modes
        self._offset = offset

        # Decoding is only available if the matrix of the modes is invertible
        self._inverse_modes = self._compute_inverse(modes)
        is_invertible = self._inverse_modes is not None

        self._has_decode = is_invertible
        self._has_decode_dp = is_invertible
        self._is_left_invertible = is_invertible
        self._is_right_invertible = is_invertible

        super().__init__()

    @staticmethod
    def _compute_inverse(modes: numpy.ndarray) -> Optional[numpy.ndarray]:
        # Return the inverse of the matrix if it is square with full rank, None otherwise
        n_o, n_i = modes.shape
        if n_o != n_i or numpy.linalg.matrix_rank(modes) < n_i:
            return None
        inverse_modes = numpy.linalg.inv(modes)
        inverse_modes.flags.writeable = False
        return inverse_modes

    # ------
    # Properties
    # ------
    @property
    def modes(self) -> numpy.ndarray:
        r"""
        [Get] the matrix of the modes :math:`\mathbf{M}` of the parametrization.

        .. note::

            The returned array is read-only.

        Returns
        -------
        numpy.ndarray
            The matrix of the modes with shape :math:`(n_o, n_i)`.

        """
        return self._modes

    @property
    def offset(self) -> numpy.ndarray:
        r"""
        [Get] the offset :math:`\vec{p_0}` of the parametrization.

        .. note::

            The returned array is read-only.

        Returns
        -------
        numpy.ndarray
            The offset with shape :math:`(n_o,)`.

        """
        return self._offset

    @property
    def input_dim(self) -> int:
        r"""
        [Get] the input dimension :math:`n_i` of the parametrization, i.e. the number
        of modes (columns of :math:`\mathbf{M}`).

        Returns
        -------
        int
            The input dimension of the parametrization.

        """
        return self._modes.shape[1]

    @property
    def output_dim(self) -> int:
        r"""
        [Get] the output dimension :math:`n_o` of the parametrization, i.e. the number
        of rows of :math:`\mathbf{M}`.

        Returns
        -------
        int
            The output dimension of the parametrization.

        """
        return self._modes.shape[0]

    # ------
    # Serialization and copy
    # ------
    def _to_dict(self) -> Dict[str, Any]:
        return {
            "modes": self._modes.tolist(),
            "offset": self._offset.tolist(),
        }

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> AffineParametrization:
        if not "modes" in data:
            raise ValueError("Input dictionary must contain a 'modes' key.")
        return cls(modes=data["modes"], offset=data.get("offset", None))

    def _copy(self) -> AffineParametrization:
        return self.__class__(modes=self._modes, offset=self._offset)

    # ------
    # Encode and decode
    # ------
    def _encode(
        self, p_i: numpy.ndarray, *, dp: bool = False
    ) -> Tuple[numpy.ndarray, Optional[numpy.ndarray]]:
        r"""
        Apply the affine parametrization to input parameters :math:`\vec{p_i}`.

        .. math::

            \vec{p_o} = \mathbf{M} \vec{p_i} + \vec{p_0}

        .. math::

            \mathbf{J}_p = \mathbf{M}

        Parameters
        ----------
        p_i : numpy.ndarray
            A 1D array of shape :math:`(n_i,)` containing the input parameters to encode.

        dp : bool, optional (default: ``False``)
            Whether to compute the Jacobian with respect to the input parameters.


        Returns
        -------
        Tuple[numpy.ndarray, Optional[numpy.ndarray]]
            A tuple containing:

            - The encoded output parameters as a 1D array of shape :math:`(n_o,)`.
            - The Jacobian matrix with respect to the input parameters as a 2D array of
              shape :math:`(n_o, n_i)`, or None if not computed.

        """
        p_o = self._modes @ p_i + self._offset  # (n_o,)
        jacobian_dp = self._modes.copy() if dp else None  # (n_o, n_i)
        return p_o, jacobian_dp

    def _decode(
        self, p_o: numpy.ndarray, *, dp: bool = False
    ) -> Tuple[numpy.ndarray, Optional[numpy.ndarray]]:
        r"""
        Apply the inverse affine map to output parameters :math:`\vec{p_o}`.

        .. math::

            \vec{p_i} = \mathbf{M}^{-1} (\vec{p_o} - \vec{p_0})

        .. math::

            \mathbf{J}_p = \mathbf{M}^{-1}

        .. note::

            This method is only available if :math:`\mathbf{M}` is invertible
            (i.e. ``has_decode`` is True).

        Parameters
        ----------
        p_o : numpy.ndarray
            A 1D array of shape :math:`(n_o,)` containing the output parameters to decode.

        dp : bool, optional (default: ``False``)
            Whether to compute the Jacobian with respect to the output parameters.


        Returns
        -------
        Tuple[numpy.ndarray, Optional[numpy.ndarray]]
            A tuple containing:

            - The decoded input parameters as a 1D array of shape :math:`(n_i,)`.
            - The Jacobian matrix with respect to the output parameters as a 2D array of
              shape :math:`(n_i, n_o)`, or None if not computed.

        """
        p_i = self._inverse_modes @ (p_o - self._offset)  # (n_i,)
        jacobian_dp = self._inverse_modes.copy() if dp else None  # (n_i, n_o)
        return p_i, jacobian_dp