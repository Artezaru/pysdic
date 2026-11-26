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
from .._validation import _as_array, _as_bool_mask


class SubsetParametrization(Parametrization):
    r"""
    A parametrization :math:`P` that selects a subset of the output parameters to be
    driven by the input parameters, the other output parameters being fixed.

    Lets :math:`\mathcal{S} \subset \{1, \dots, n_o\}` be the set of the selected indices
    defined by a boolean ``mask`` of shape :math:`(n_o,)`, with :math:`n_i = |\mathcal{S}|`
    the number of selected parameters. The parametrization maps the input parameters
    :math:`\vec{p_i} \in \mathbb{R}^{n_i}` to the output parameters
    :math:`\vec{p_o} \in \mathbb{R}^{n_o}` according to:

    .. math::

        \vec{p_o}[\mathcal{S}] = \vec{p_i}
        \quad \text{and} \quad
        \vec{p_o}[\overline{\mathcal{S}}] = \vec{v}[\overline{\mathcal{S}}]

    where :math:`\vec{v}` are the fixed values of the non-selected parameters.

    This is typically used to optimize only some parameters of a transformation while
    the others are kept fixed (see :class:`pysdic.ReparametrizedTransformation`).

    The Jacobian of the parametrization is the constant selection matrix
    :math:`\mathbf{E} \in \mathbb{R}^{n_o \times n_i}` whose columns are the columns
    of the identity matrix at the selected indices:

    .. math::

        \mathbf{J}_p = \frac{\partial P}{\partial \vec{p_i}} = \mathbf{E} = \mathbf{I}_{n_o}[:, \mathcal{S}]

    The parametrization can be decoded by extracting the selected parameters:

    .. math::

        \vec{p_i} = P^{-1}(\vec{p_o}) = \vec{p_o}[\mathcal{S}]

    The decoding map is a left inverse of :math:`P`. It is also a right inverse only if
    all the parameters are selected (i.e. :math:`P` is the identity).

    .. seealso::

        - Class :class:`pysdic.Parametrization` for the base class of parametrizations.
        - Class :class:`pysdic.AffineParametrization` for general affine parametrizations.

    .. note::

        :class:`pysdic.SubsetParametrization` is equivalent to an
        :class:`pysdic.AffineParametrization` with ``modes`` :math:`\mathbf{E}` and
        ``offset`` :math:`\vec{v}` (with zeros at the selected indices), but it avoids
        dense matrix products. In particular, for any matrix :math:`\mathbf{J}` with
        :math:`n_o` columns (e.g. the Jacobian of a transformation with respect to its
        parameters), the chain rule reduces to a column selection:

        .. math::

            \mathbf{J} \cdot \mathbf{E} = \mathbf{J}[:, \mathcal{S}]

        which can be computed with ``jacobian[..., parametrization.mask]``.

    .. note::

        The ``mask`` and ``fixed_values`` are **constants** of the parametrization. They are
        stored as read-only copies and cannot be modified after initialization.


    Parameters
    ----------
    mask : ArrayLike
        A boolean array of shape :math:`(n_o,)` indicating the selected output parameters
        (True) driven by the input parameters. At least one parameter must be selected.

    fixed_values : Optional[ArrayLike], optional (default: ``None``)
        The values :math:`\vec{v}` of shape :math:`(n_o,)` used for the non-selected
        output parameters. The values at the selected indices must be ``0.0`` (a
        ``ValueError`` is raised otherwise, to avoid ignoring values given by mistake).
        If None, the non-selected parameters are fixed to zero.


    Examples
    --------
    Optimize only the first and third parameters of a vector of four parameters,
    the second and fourth parameters being fixed:

    .. code-block:: python

        parametrization = SubsetParametrization(
            mask=[True, False, True, False],
            fixed_values=[0.0, 5.0, 0.0, 7.0],
        )

        parametrization.encode([1.0, 2.0]).p              # array([1., 5., 2., 7.])
        parametrization.decode([1.0, 5.0, 2.0, 7.0]).p    # array([1., 2.])

    A convenient way to freeze the current values of the parameters of a transformation
    is to pass them as ``fixed_values``, with zeros at the selected indices:

    .. code-block:: python

        parametrization = SubsetParametrization(
            mask=mask,
            fixed_values=numpy.where(mask, 0.0, transformation.parameters),
        )

    """

    _has_encode = True
    _has_encode_dp = True
    _has_decode = True
    _has_decode_dp = True
    _is_left_invertible = True

    def __init__(
        self,
        mask: ArrayLike,
        fixed_values: Optional[ArrayLike] = None,
    ) -> None:
        mask = _as_bool_mask(mask, None, "mask", copy=True)  # any size n_o
        if not numpy.any(mask):
            raise ValueError("Input 'mask' must select at least one parameter.")
        n_o = mask.shape[0]

        if fixed_values is None:
            fixed_values = numpy.zeros(n_o, dtype=numpy.float64)
        else:
            fixed_values = _as_array(
                fixed_values, "fixed_values", numpy.float64, shape=(n_o,), finite=True, copy=True
            )
            if numpy.any(fixed_values[mask] != 0.0):
                raise ValueError(
                    f"Input 'fixed_values' must be 0.0 at the selected indices "
                    f"{numpy.flatnonzero(mask).tolist()} (driven by the input parameters). "
                    f"Got {fixed_values[mask].tolist()}. Use numpy.where(mask, 0.0, values) to "
                    f"set them to zero."
                )

        indices = numpy.flatnonzero(mask)  # (n_i,)

        # Store the constants as read-only arrays
        mask.flags.writeable = False
        fixed_values.flags.writeable = False
        indices.flags.writeable = False
        self._mask = mask
        self._fixed_values = fixed_values
        self._indices = indices

        # The decoding map is a right inverse only if all the parameters are selected
        self._is_right_invertible = bool(numpy.all(mask))

        super().__init__()

    # ------
    # Properties
    # ------
    @property
    def mask(self) -> numpy.ndarray:
        r"""
        [Get] the boolean mask of the selected output parameters.

        .. note::

            The returned array is read-only.

        Returns
        -------
        numpy.ndarray
            The boolean mask with shape :math:`(n_o,)`.

        """
        return self._mask

    @property
    def indices(self) -> numpy.ndarray:
        r"""
        [Get] the indices :math:`\mathcal{S}` of the selected output parameters,
        in increasing order.

        .. note::

            The returned array is read-only.

        Returns
        -------
        numpy.ndarray
            The indices of the selected output parameters with shape :math:`(n_i,)`.

        """
        return self._indices

    @property
    def fixed_values(self) -> numpy.ndarray:
        r"""
        [Get] the values :math:`\vec{v}` of the non-selected output parameters.

        The values at the selected indices are zeros.

        .. note::

            The returned array is read-only.

        Returns
        -------
        numpy.ndarray
            The fixed values with shape :math:`(n_o,)`.

        """
        return self._fixed_values

    @property
    def input_dim(self) -> int:
        r"""
        [Get] the input dimension :math:`n_i` of the parametrization, i.e. the number
        of selected output parameters.

        Returns
        -------
        int
            The input dimension of the parametrization.

        """
        return self._indices.shape[0]

    @property
    def output_dim(self) -> int:
        r"""
        [Get] the output dimension :math:`n_o` of the parametrization, i.e. the size
        of the mask.

        Returns
        -------
        int
            The output dimension of the parametrization.

        """
        return self._mask.shape[0]

    # ------
    # Serialization and copy
    # ------
    def _to_dict(self) -> Dict[str, Any]:
        return {
            "mask": self._mask.tolist(),
            "fixed_values": self._fixed_values.tolist(),
        }

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> SubsetParametrization:
        if not "mask" in data:
            raise ValueError("Input dictionary must contain a 'mask' key.")
        return cls(mask=data["mask"], fixed_values=data.get("fixed_values", None))

    def _copy(self) -> SubsetParametrization:
        return self.__class__(mask=self._mask, fixed_values=self._fixed_values)

    # ------
    # Encode and decode
    # ------
    def _encode(
        self, p_i: numpy.ndarray, *, dp: bool = False
    ) -> Tuple[numpy.ndarray, Optional[numpy.ndarray]]:
        r"""
        Insert the input parameters :math:`\vec{p_i}` at the selected indices of the
        output parameters.

        .. math::

            \vec{p_o}[\mathcal{S}] = \vec{p_i}
            \quad \text{and} \quad
            \vec{p_o}[\overline{\mathcal{S}}] = \vec{v}[\overline{\mathcal{S}}]

        .. math::

            \mathbf{J}_p = \mathbf{E} = \mathbf{I}_{n_o}[:, \mathcal{S}]

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
        p_o = self._fixed_values.copy()  # (n_o,)
        p_o[self._indices] = p_i

        jacobian_dp = None
        if dp:
            jacobian_dp = numpy.zeros((self.output_dim, self.input_dim), dtype=float)
            jacobian_dp[self._indices, numpy.arange(self.input_dim)] = 1.0  # (n_o, n_i)

        return p_o, jacobian_dp

    def _decode(
        self, p_o: numpy.ndarray, *, dp: bool = False
    ) -> Tuple[numpy.ndarray, Optional[numpy.ndarray]]:
        r"""
        Extract the selected parameters from the output parameters :math:`\vec{p_o}`.

        .. math::

            \vec{p_i} = \vec{p_o}[\mathcal{S}]

        .. math::

            \mathbf{J}_p = \mathbf{E}^T = \mathbf{I}_{n_o}[\mathcal{S}, :]

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
        p_i = p_o[self._indices]  # (n_i,)

        jacobian_dp = None
        if dp:
            jacobian_dp = numpy.zeros((self.input_dim, self.output_dim), dtype=float)
            jacobian_dp[numpy.arange(self.input_dim), self._indices] = 1.0  # (n_i, n_o)

        return p_i, jacobian_dp