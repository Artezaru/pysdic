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

from collections.abc import Mapping
from typing import Optional, Tuple, Dict, Any, Union
from numpy.typing import ArrayLike
from numbers import Integral

import numpy

from ..core.property_container import PropertyContainer
from ..core.lockable import LockableMixin
from ..core.serialization import (
    JsonSerializerMixin,
    NpzSerializerMixin,
    Hdf5SerializerMixin,
)
from .._validation import (
    _validate_bool,
    _validate_integer,
    _as_array,
    _as_integer_array,
    _as_bool_mask,
    _as_index_array,
)
from ..fem._validation import _validate_element_type
from ..fem.gauss_points import get_gauss_points


class IntegrationPoints(LockableMixin, JsonSerializerMixin, NpzSerializerMixin, Hdf5SerializerMixin):
    r"""
    A set of :math:`N_p` integration points located in the elements of a mesh.

    Each point is defined by the index of its element and its natural coordinates
    :math:`\Xi = (\xi, \eta, ...)` in the reference element. There are :math:`K`
    natural coordinates, the topological dimension of the elements (1 for segments, 2
    for triangles and quadrangles). Each point also has a weight (e.g. a Gauss
    quadrature weight).

    Each point can carry additional properties stored in a
    :class:`pysdic.core.property_container.PropertyContainer`, accessible through the ``properties`` attribute
    or directly with ``integration_points[name]``.

    A point outside the mesh is **unlocated**: its element index is ``-1``. Only the
    element index defines the location: the natural coordinates of an unlocated point
    are ignored by the :mod:`pysdic.fem` functions (the methods of this class set them
    to ``numpy.nan``).

    .. note::

        The arrays are copied when they are set and stored as ``numpy.float64``
        (natural coordinates, weights) and ``numpy.int64`` (element indices) arrays.
        The attributes return read-only views: to change the values, assign a new
        array.

    .. important::

        The integration points do not know their mesh: use
        :meth:`pysdic.Mesh.validate_integration_points` to check that they are
        consistent with a mesh (topological dimension and element indices).

    .. important::

        The properties are stored in a :class:`pysdic.core.property_container.PropertyContainer` with
        ``force_2d=True``, i.e. with shape :math:`(N_p, P)`. A container given by the
        user with other settings is accepted, but some operations may behave
        unexpectedly.

    Parameters
    ----------
    natural_coordinates: ArrayLike
        An array of shape :math:`(N_p, K)` with the natural coordinates of the points.
        A 1D array of shape :math:`(N_p,)` is accepted if ``n_topological_dimensions=1``.

    element_indices: ArrayLike
        An integer array of shape :math:`(N_p,)` with the index of the element of each
        point, or ``-1`` for an unlocated point. Floats are accepted only if they are
        integral (no truncation).

    weights: Optional[ArrayLike], optional (default: ``None``)
        An array of shape :math:`(N_p,)` with the finite and non-negative weights of
        the points. By default, all the weights are 1.

    n_topological_dimensions: Optional[Integral], optional (default: ``None``)
        The topological dimension :math:`K`. By default, it is inferred from the shape
        of ``natural_coordinates``. If given, it must match this shape.

    properties: Optional[Union[Dict[str, ArrayLike], PropertyContainer]], optional
        The properties of the points:

        - a dictionary mapping names to arrays of shape :math:`(N_p, \dots)`, which
          are copied into a new :class:`pysdic.core.property_container.PropertyContainer`,
        - or a :class:`pysdic.core.property_container.PropertyContainer` with :math:`N = N_p`, which is
          **shared** (not copied).


    Notes
    -----
    ``len(integration_points)`` gives the number of points :math:`N_p`, while
    ``integration_points[name]``, ``name in integration_points`` and
    ``iter(integration_points)`` refer to the properties.

    The methods never modify the integration points in place: the methods that change
    the points (:meth:`filter_points`, :meth:`filter_elements`, :meth:`unlocate_points_by_index`,
    ...) return new integration points. Only the setters (same number of points) and the
    properties modify existing integration points.

    .. note::

        Integration points can be locked by an object of the package that uses them (for
        example a term built with :func:`pysdic.build_sdic_term`, which computes the shape
        functions of the points once, see :class:`pysdic.core.lockable.LockableMixin`): the
        setters ``natural_coordinates``, ``element_indices`` and ``weights`` then raise a
        :class:`RuntimeError` explaining why. The locked points can still be read and used
        everywhere, and their properties can be modified. A copy (``copy``) is always
        unlocked.

    """

    __slots__ = [
        "_natural_coordinates",
        "_element_indices",
        "_weights",
        "_properties",
    ]

    def __init__(
        self,
        natural_coordinates: ArrayLike,
        element_indices: ArrayLike,
        *,
        weights: Optional[ArrayLike] = None,
        n_topological_dimensions: Optional[Integral] = None,
        properties: Optional[Union[Dict[str, ArrayLike], PropertyContainer]] = None,
    ) -> None:
        n_topological_dimensions = _validate_integer(
            n_topological_dimensions, "n_topological_dimensions", min_value=1, allow_none=True
        )
        natural_coordinates = self._as_natural_coordinates(
            natural_coordinates, n_topological_dimensions
        )
        N = natural_coordinates.shape[0]
        element_indices = self._as_element_indices(element_indices, N)
        weights = self._as_weights(weights, N)

        if properties is None:
            props = PropertyContainer(N=N, force_2d=True)
        elif isinstance(properties, PropertyContainer):
            if properties.N != N:
                raise ValueError(
                    f"The number of entities in 'properties' must match the number of "
                    f"integration points. Got N={properties.N} for properties and N_p={N} "
                    f"for integration points."
                )
            props = properties
        elif isinstance(properties, Mapping):
            props = PropertyContainer(N=N, properties=properties, force_2d=True)
        else:
            raise TypeError(
                f"Input 'properties' must be None, a dictionary of arrays or a "
                f"PropertyContainer. Got type {type(properties)}."
            )

        self._natural_coordinates = natural_coordinates
        self._element_indices = element_indices
        self._weights = weights
        self._properties = props
        self.validate(recursive=True)

    @staticmethod
    def _as_natural_coordinates(value: ArrayLike, n_topological_dimensions: Optional[int]) -> numpy.ndarray:
        # Copy the input to a (N_p, K) float64 array (NaN allowed for unlocated points).
        # 1D input is only accepted for K = 1.
        natural_coordinates = _as_array(
            value, "natural_coordinates", numpy.float64,
            shape=(None, n_topological_dimensions),
            shape_hint="(N_p, K)" if n_topological_dimensions is None else f"(N_p, {n_topological_dimensions})",
            finite=False, accept_1d_as_column=(n_topological_dimensions == 1), copy=True,
        )
        if natural_coordinates.shape[1] == 0:
            raise ValueError(
                "Input 'natural_coordinates' must have at least one dimension (K >= 1)."
            )
        return natural_coordinates

    @staticmethod
    def _as_element_indices(value: ArrayLike, n_points: int) -> numpy.ndarray:
        # Copy the input to a (N_p,) int64 array of indices >= -1
        element_indices = _as_integer_array(
            value, "element_indices", shape=(n_points,), shape_hint=f"({n_points},)", copy=True
        )
        if element_indices.size > 0 and element_indices.min() < -1:
            raise ValueError(
                f"Input 'element_indices' must contain element indices >= 0, or -1 for "
                f"unlocated points. Got a minimum index of {element_indices.min()}."
            )
        return element_indices

    @staticmethod
    def _as_weights(value: Optional[ArrayLike], n_points: int) -> numpy.ndarray:
        # Copy the input to a (N_p,) float64 array of finite non-negative weights (1 by default)
        if value is None:
            return numpy.ones(n_points, dtype=numpy.float64)
        weights = _as_array(
            value, "weights", numpy.float64, shape=(n_points,), shape_hint=f"({n_points},)",
            finite=True, copy=True,
        )
        if weights.size > 0 and weights.min() < 0:
            raise ValueError(
                f"Input 'weights' must contain non-negative values. Got a minimum weight "
                f"of {weights.min()}."
            )
        return weights

    def validate(self, *, recursive: bool = True) -> None:
        r"""
        Validate the integrity of the integration points.

        The following conditions are checked:

        - The natural coordinates are a ``numpy.float64`` array of shape
          :math:`(N_p, K)` with :math:`K \geq 1`.
        - The element indices are a ``numpy.int64`` array of shape :math:`(N_p,)` with
          values :math:`\geq -1`.
        - The weights are a ``numpy.float64`` array of shape :math:`(N_p,)` with finite
          non-negative values.
        - The properties are a :class:`pysdic.core.property_container.PropertyContainer` with
          :math:`N = N_p`.

        Parameters
        ----------
        recursive: bool, optional (default: ``True``)
            If True, the property container is also validated (see
            :meth:`pysdic.core.property_container.PropertyContainer.validate`).

        Raises
        ------
        TypeError
            If an attribute has a wrong type.

        ValueError
            If an attribute has a wrong shape or value.

        """
        recursive = _validate_bool(recursive, "recursive")

        # --- natural coordinates: 2D float64 array of shape (Np, K) ---
        if not isinstance(self._natural_coordinates, numpy.ndarray):
            raise TypeError("[IntegrationPoints validate] natural_coordinates should be a numpy ndarray.")
        if self._natural_coordinates.ndim != 2 or self._natural_coordinates.shape[1] == 0:
            raise ValueError(
                f"[IntegrationPoints validate] natural_coordinates should be a 2D array of "
                f"shape (Np, K) with K >= 1. Got shape {self._natural_coordinates.shape}."
            )
        if self._natural_coordinates.dtype != numpy.float64:
            raise TypeError(
                f"[IntegrationPoints validate] natural_coordinates should have dtype "
                f"float64. Got {self._natural_coordinates.dtype}."
            )
        N = self._natural_coordinates.shape[0]

        # --- element indices: int64 array of shape (Np,) with values >= -1 ---
        if not isinstance(self._element_indices, numpy.ndarray):
            raise TypeError("[IntegrationPoints validate] element_indices should be a numpy ndarray.")
        if self._element_indices.shape != (N,):
            raise ValueError(
                f"[IntegrationPoints validate] element_indices should have shape ({N},). "
                f"Got shape {self._element_indices.shape}."
            )
        if self._element_indices.dtype != numpy.int64:
            raise TypeError(
                f"[IntegrationPoints validate] element_indices should have dtype int64. "
                f"Got {self._element_indices.dtype}."
            )
        if N > 0 and self._element_indices.min() < -1:
            raise ValueError(
                f"[IntegrationPoints validate] element_indices should be >= -1. "
                f"Minimum index found: {self._element_indices.min()}."
            )

        # --- weights: float64 array of shape (Np,) with finite non-negative values ---
        if not isinstance(self._weights, numpy.ndarray):
            raise TypeError("[IntegrationPoints validate] weights should be a numpy ndarray.")
        if self._weights.shape != (N,):
            raise ValueError(
                f"[IntegrationPoints validate] weights should have shape ({N},). "
                f"Got shape {self._weights.shape}."
            )
        if self._weights.dtype != numpy.float64:
            raise TypeError(
                f"[IntegrationPoints validate] weights should have dtype float64. "
                f"Got {self._weights.dtype}."
            )
        if not numpy.all(numpy.isfinite(self._weights)) or (N > 0 and self._weights.min() < 0):
            raise ValueError(
                "[IntegrationPoints validate] weights should be finite and non-negative."
            )

        # --- properties: PropertyContainer with the same number of points ---
        if not isinstance(self._properties, PropertyContainer):
            raise TypeError("[IntegrationPoints validate] properties should be a PropertyContainer.")
        if self._properties.N != N:
            raise ValueError(
                f"[IntegrationPoints validate] The number of entities in properties should "
                f"match the number of points. Got N={self._properties.N} for properties and "
                f"N_p={N} for points."
            )

        if recursive:
            self._properties.validate()

    # ==========================
    # API interface
    # ==========================
    @property
    def natural_coordinates(self) -> numpy.ndarray:
        r"""
        [Get/Set] The natural coordinates of the points, an array of shape
        :math:`(N_p, K)`.

        The getter returns a read-only view. The setter copies the given array, whose
        shape must be :math:`(N_p, K)`. The setter is refused (``RuntimeError``) if the
        integration points are locked.

        Parameters
        ----------
        value: ArrayLike
            An array of shape :math:`(N_p, K)`. A 1D array of shape :math:`(N_p,)` is
            accepted if :math:`K = 1`.

        Returns
        -------
        numpy.ndarray
            A read-only view of shape :math:`(N_p, K)` on the natural coordinates.

        """
        view = self._natural_coordinates.view()
        view.flags.writeable = False
        return view

    @natural_coordinates.setter
    def natural_coordinates(self, value: ArrayLike) -> None:
        self._check_unlocked("set the natural coordinates")
        natural_coordinates = self._as_natural_coordinates(value, self.n_topological_dimensions)
        if natural_coordinates.shape[0] != self.n_points:
            raise ValueError(
                f"Input 'natural_coordinates' must keep the number of points "
                f"N_p={self.n_points}. Got {natural_coordinates.shape[0]} points. Consider "
                f"using the filter/concatenate methods to change the number of points."
            )
        self._natural_coordinates = natural_coordinates

    @property
    def element_indices(self) -> numpy.ndarray:
        r"""
        [Get/Set] The element index of each point (``-1`` for an unlocated point), an
        array of shape :math:`(N_p,)`.

        The getter returns a read-only view. The setter copies the given array. The
        setter is refused (``RuntimeError``) if the integration points are locked.

        Parameters
        ----------
        value: ArrayLike
            An integer array of shape :math:`(N_p,)` with values :math:`\geq -1`.

        Returns
        -------
        numpy.ndarray
            A read-only view of shape :math:`(N_p,)` on the element indices.

        """
        view = self._element_indices.view()
        view.flags.writeable = False
        return view

    @element_indices.setter
    def element_indices(self, value: ArrayLike) -> None:
        self._check_unlocked("set the element indices")
        self._element_indices = self._as_element_indices(value, self.n_points)

    @property
    def weights(self) -> numpy.ndarray:
        r"""
        [Get/Set] The weights of the points, an array of shape :math:`(N_p,)`.

        The getter returns a read-only view. The setter copies the given array. The
        setter is refused (``RuntimeError``) if the integration points are locked.

        Parameters
        ----------
        value: Optional[ArrayLike]
            An array of shape :math:`(N_p,)` with finite non-negative values, or
            ``None`` to set all the weights to 1.

        Returns
        -------
        numpy.ndarray
            A read-only view of shape :math:`(N_p,)` on the weights.

        """
        view = self._weights.view()
        view.flags.writeable = False
        return view

    @weights.setter
    def weights(self, value: Optional[ArrayLike]) -> None:
        self._check_unlocked("set the weights")
        self._weights = self._as_weights(value, self.n_points)

    @property
    def n_topological_dimensions(self) -> int:
        r"""
        [Get] The topological dimension :math:`K` of the elements.

        Returns
        -------
        int
            The number of natural coordinates.

        """
        return self._natural_coordinates.shape[1]

    @property
    def n_points(self) -> int:
        r"""
        [Get] The number of points :math:`N_p` (located or not).

        Returns
        -------
        int
            The number of points.

        """
        return self._natural_coordinates.shape[0]

    def __len__(self) -> int:
        r"""Return the number of points :math:`N_p`."""
        return self.n_points

    def is_located(self) -> numpy.ndarray:
        r"""
        Check which points are located in an element (element index :math:`\neq -1`).

        Returns
        -------
        numpy.ndarray
            A boolean array of shape :math:`(N_p,)`, True for the located points.

        """
        return self._element_indices != -1

    @property
    def n_located(self) -> int:
        r"""
        [Get] The number of points located in an element (element index
        :math:`\neq -1`).

        Returns
        -------
        int
            The number of located points.

        """
        return int(numpy.count_nonzero(self.is_located()))

    @property
    def shape(self) -> Tuple[int, int]:
        r"""
        [Get] The shape :math:`(N_p, K)` of the natural coordinates.

        Returns
        -------
        Tuple[int, int]
            The number of points and the topological dimension.

        """
        return (self.n_points, self.n_topological_dimensions)

    @property
    def properties(self) -> PropertyContainer:
        r"""
        [Get] The properties of the points.

        Returns
        -------
        :class:`pysdic.core.property_container.PropertyContainer`
            The property container (not a copy).

        """
        return self._properties

    def __getitem__(self, key):
        r"""Shortcut for ``integration_points.properties[key]``."""
        return self._properties[key]

    def __setitem__(self, key, value):
        r"""Shortcut for ``integration_points.properties[key] = value``."""
        self._properties[key] = value

    def __delitem__(self, key):
        r"""Shortcut for ``del integration_points.properties[key]``."""
        del self._properties[key]

    def __contains__(self, key):
        r"""Shortcut for ``key in integration_points.properties``."""
        return key in self._properties

    def __iter__(self):
        r"""Shortcut for ``iter(integration_points.properties)`` (iterates over the names)."""
        return iter(self._properties)

    @property
    def n_properties(self) -> int:
        r"""
        [Get] The number of properties of the points.

        Returns
        -------
        int
            The number of properties.

        """
        return self._properties.n_properties

    def __repr__(self) -> str:
        string = (
            f"IntegrationPoints(n_points={self.n_points}, n_located={self.n_located}, "
            f"n_topological_dimensions={self.n_topological_dimensions})"
        )
        for key in self._properties.keys():
            string += f"\n  - Property: '{key}' with shape {self._properties.get_shape(key)}"
        return string

    # ==========================
    # Creation and I/O Methods
    # ==========================
    @classmethod
    def from_gauss_points(cls, element_type: str, n_elements: Integral) -> IntegrationPoints:
        r"""
        Create the Gauss quadrature points of all the elements of a mesh.

        The :math:`N_{gp}` Gauss points of the element type (see
        :func:`pysdic.fem.get_gauss_points`) are repeated for each element, with their
        quadrature weights. The points are sorted by element: the points
        :math:`e N_{gp}` to :math:`(e + 1) N_{gp} - 1` belong to the element :math:`e`.

        Parameters
        ----------
        element_type: str
            The type of the elements (case insensitive, see
            :class:`pysdic.Connectivity`).

        n_elements: Integral
            The number of elements :math:`N_e` of the mesh (non-negative).

        Returns
        -------
        :class:`pysdic.IntegrationPoints`
            :math:`N_p = N_e N_{gp}` integration points, all located.

        Examples
        --------
        .. code-block:: python
            :linenos:

            from pysdic import IntegrationPoints

            integration_points = IntegrationPoints.from_gauss_points("quadrangle_4", 2)
            print(integration_points.element_indices)  # [0 0 0 0 1 1 1 1]
            print(integration_points.weights)          # [1. 1. 1. 1. 1. 1. 1. 1.]

        """
        element_type = _validate_element_type(element_type)
        n_elements = _validate_integer(n_elements, "n_elements", min_value=0)

        gauss_points, gauss_weights = get_gauss_points(element_type, return_weights=True)
        n_gauss = gauss_points.shape[0]
        return cls(
            natural_coordinates=numpy.tile(gauss_points, (n_elements, 1)),
            element_indices=numpy.repeat(numpy.arange(n_elements), n_gauss),
            weights=numpy.tile(gauss_weights, n_elements),
            n_topological_dimensions=gauss_points.shape[1],
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any], *, load_properties: bool = True) -> IntegrationPoints:
        r"""
        Create integration points from a dictionary.

        The dictionary must contain the keys ``"natural_coordinates"`` (shape
        :math:`(N_p, K)`) and ``"element_indices"`` (shape :math:`(N_p,)`), and may
        contain the key ``"weights"`` (shape :math:`(N_p,)`, all 1 if missing). The
        properties are read from the keys ``"_property/{property_name}"``.

        .. note::

            The dictionary is flat (one level), as required by the NPZ format. See
            :class:`pysdic.core.serialization.NpzSerializerMixin`.

        Parameters
        ----------
        data: Dict[str, Any]
            The dictionary to read.

        load_properties: bool, optional (default: ``True``)
            If False, the properties are ignored.

        Returns
        -------
        :class:`pysdic.IntegrationPoints`
            The new integration points.

        """
        if not isinstance(data, Mapping):
            raise TypeError(f"Input 'data' must be a dictionary. Got type {type(data)}.")
        for key, shape in (("natural_coordinates", "(N_p, K)"), ("element_indices", "(N_p,)")):
            if key not in data:
                raise ValueError(f"Input dictionary must contain a key '{key}' with shape {shape}.")
        load_properties = _validate_bool(load_properties, "load_properties")

        properties = {}
        if load_properties:
            prefix = "_property/"
            for key in data:
                if isinstance(key, str) and key.startswith(prefix):
                    properties[key[len(prefix):]] = data[key]

        return cls(
            natural_coordinates=data["natural_coordinates"],
            element_indices=data["element_indices"],
            weights=data.get("weights"),
            properties=properties,
        )

    def to_dict(self, *, save_properties: bool = True) -> Dict[str, Any]:
        r"""
        Convert the integration points to a flat dictionary.

        The natural coordinates, element indices and weights are stored under the keys
        ``"natural_coordinates"``, ``"element_indices"`` and ``"weights"``, and the
        properties under the keys ``"_property/{property_name}"``. All the arrays are
        copies.

        .. note::

            The dictionary is flat (one level), as required by the NPZ format. See
            :class:`pysdic.core.serialization.NpzSerializerMixin`.

        Parameters
        ----------
        save_properties: bool, optional (default: ``True``)
            If False, the properties are not included.

        Returns
        -------
        Dict[str, Any]
            The dictionary representation of the integration points.

        """
        save_properties = _validate_bool(save_properties, "save_properties")

        data = {
            "natural_coordinates": self._natural_coordinates.copy(),
            "element_indices": self._element_indices.copy(),
            "weights": self._weights.copy(),
        }
        if save_properties:
            for key, values in self._properties.to_dict().items():
                data[f"_property/{key}"] = values
        return data

    # =======================
    # Structure Methods
    # =======================
    def _new(
        self,
        natural_coordinates: numpy.ndarray,
        element_indices: numpy.ndarray,
        weights: numpy.ndarray,
        properties: Optional[PropertyContainer],
    ) -> IntegrationPoints:
        # New instance with the same topological dimension (the arrays are copied by the constructor)
        return self.__class__(
            natural_coordinates=natural_coordinates,
            element_indices=element_indices,
            weights=weights,
            n_topological_dimensions=self.n_topological_dimensions,
            properties=properties,
        )

    def concatenate(
        self, other: IntegrationPoints, *, propagate_properties: bool = True
    ) -> IntegrationPoints:
        r"""
        Concatenate two sets of integration points (the points of ``other`` come after).

        .. note::

            The element indices are not shifted: both sets must refer to the same mesh.

        The properties are concatenated with :meth:`pysdic.core.property_container.PropertyContainer.concatenate`:
        a property present in only one set is completed with ``numpy.nan`` for the points
        of the other one (integer and boolean properties are then promoted to float).

        Parameters
        ----------
        other: :class:`pysdic.IntegrationPoints`
            The integration points to append, with the same :math:`K`.

        propagate_properties: bool, optional (default: ``True``)
            If False, the new integration points have no properties.

        Returns
        -------
        :class:`pysdic.IntegrationPoints`
            New integration points with :math:`N_p + N_p^{other}` points.

        """
        if not isinstance(other, IntegrationPoints):
            raise TypeError(f"Input 'other' must be an IntegrationPoints. Got type {type(other)}.")
        propagate_properties = _validate_bool(propagate_properties, "propagate_properties")
        if self.n_topological_dimensions != other.n_topological_dimensions:
            raise ValueError(
                f"IntegrationPoints instances must have the same topological dimension to "
                f"be concatenated. Got K={self.n_topological_dimensions} and "
                f"K={other.n_topological_dimensions}."
            )

        concatenated_properties = None
        if propagate_properties:
            concatenated_properties = self._properties.concatenate(other._properties, fill=numpy.nan)

        return self._new(
            numpy.vstack((self._natural_coordinates, other._natural_coordinates)),
            numpy.concatenate((self._element_indices, other._element_indices)),
            numpy.concatenate((self._weights, other._weights)),
            concatenated_properties,
        )

    def copy(self, *, copy_properties: bool = True) -> IntegrationPoints:
        r"""
        Create a copy of the integration points. The copy is always unlocked.

        Parameters
        ----------
        copy_properties: bool, optional (default: ``True``)
            If False, the copy has no properties.

        Returns
        -------
        :class:`pysdic.IntegrationPoints`
            New integration points with copies of the arrays and properties.

        """
        copy_properties = _validate_bool(copy_properties, "copy_properties")

        copied_properties = None
        if copy_properties:
            copied_properties = self._properties.copy()

        return self._new(
            self._natural_coordinates, self._element_indices, self._weights, copied_properties
        )

    # -----------------------
    # Points
    # -----------------------
    def filter_points(
        self,
        mask: ArrayLike,
        *,
        propagate_properties: bool = True,
    ) -> IntegrationPoints:
        r"""
        Keep only the points selected by a boolean mask.

        Parameters
        ----------
        mask: ArrayLike
            A boolean array of shape :math:`(N_p,)`. The points with a True value are
            kept.

        propagate_properties: bool, optional (default: ``True``)
            If False, the new integration points have no properties.

        Returns
        -------
        :class:`pysdic.IntegrationPoints`
            The filtered integration points.

        Raises
        ------
        TypeError
            If the mask is not a boolean array (use :meth:`keep_points_by_index` for
            indices).

        ValueError
            If the mask does not have the shape :math:`(N_p,)`.

        """
        mask = _as_bool_mask(mask, self.n_points)
        propagate_properties = _validate_bool(propagate_properties, "propagate_properties")

        # Advanced indexing: copies
        new_natural_coordinates = self._natural_coordinates[mask]
        new_element_indices = self._element_indices[mask]
        new_weights = self._weights[mask]

        new_properties = None
        if propagate_properties:
            new_properties = self._properties.filter(mask)

        return self._new(new_natural_coordinates, new_element_indices, new_weights, new_properties)

    def remove_points_by_index(
        self,
        indices: ArrayLike,
        *,
        propagate_properties: bool = True,
    ) -> IntegrationPoints:
        r"""
        Remove the points with the given indices.

        Parameters
        ----------
        indices: ArrayLike
            An integer array of shape :math:`(R,)` with the indices of the points to
            remove, in :math:`[0, N_p)`. Duplicates are allowed.

        propagate_properties: bool, optional (default: ``True``)
            If False, the new integration points have no properties.

        Returns
        -------
        :class:`pysdic.IntegrationPoints`
            The integration points without the given points.

        Raises
        ------
        TypeError
            If the indices are not integers.

        IndexError
            If an index is out of :math:`[0, N_p)`.

        """
        indices = _as_index_array(indices, self.n_points)
        mask = numpy.ones(self.n_points, dtype=bool)
        mask[indices] = False
        return self.filter_points(mask, propagate_properties=propagate_properties)

    def keep_points_by_index(
        self,
        indices: ArrayLike,
        *,
        propagate_properties: bool = True,
    ) -> IntegrationPoints:
        r"""
        Keep only the points with the given indices.

        .. note::

            The points keep their original order and duplicated indices are ignored.

        Parameters
        ----------
        indices: ArrayLike
            An integer array of shape :math:`(R,)` with the indices of the points to
            keep, in :math:`[0, N_p)`.

        propagate_properties: bool, optional (default: ``True``)
            If False, the new integration points have no properties.

        Returns
        -------
        :class:`pysdic.IntegrationPoints`
            The integration points with only the given points.

        Raises
        ------
        TypeError
            If the indices are not integers.

        IndexError
            If an index is out of :math:`[0, N_p)`.

        """
        indices = _as_index_array(indices, self.n_points)
        mask = numpy.zeros(self.n_points, dtype=bool)
        mask[indices] = True
        return self.filter_points(mask, propagate_properties=propagate_properties)

    def _points_in_elements(self, element_indices: ArrayLike) -> numpy.ndarray:
        # Mask (N_p,) of the points located in the given elements (-1 selects the unlocated points)
        element_indices = _as_integer_array(
            element_indices, "element_indices", shape=(None,), shape_hint="(R,)"
        )
        if element_indices.size > 0 and element_indices.min() < -1:
            raise ValueError(
                f"Input 'element_indices' must contain element indices >= 0, or -1 for "
                f"unlocated points. Got a minimum index of {element_indices.min()}."
            )
        return numpy.isin(self._element_indices, element_indices)

    def remove_points_by_element(
        self,
        element_indices: ArrayLike,
        *,
        propagate_properties: bool = True,
    ) -> IntegrationPoints:
        r"""
        Remove the points located in the given elements.

        Parameters
        ----------
        element_indices: ArrayLike
            An integer array of shape :math:`(R,)` with the indices of the elements
            (``-1`` selects the unlocated points). Indices of elements without points
            are allowed.

        propagate_properties: bool, optional (default: ``True``)
            If False, the new integration points have no properties.

        Returns
        -------
        :class:`pysdic.IntegrationPoints`
            The integration points without the points of the given elements.

        """
        mask = self._points_in_elements(element_indices)
        return self.filter_points(~mask, propagate_properties=propagate_properties)

    def keep_points_by_element(
        self,
        element_indices: ArrayLike,
        *,
        propagate_properties: bool = True,
    ) -> IntegrationPoints:
        r"""
        Keep only the points located in the given elements.

        Parameters
        ----------
        element_indices: ArrayLike
            An integer array of shape :math:`(R,)` with the indices of the elements
            (``-1`` selects the unlocated points). Indices of elements without points
            are allowed.

        propagate_properties: bool, optional (default: ``True``)
            If False, the new integration points have no properties.

        Returns
        -------
        :class:`pysdic.IntegrationPoints`
            The integration points with only the points of the given elements.

        """
        mask = self._points_in_elements(element_indices)
        return self.filter_points(mask, propagate_properties=propagate_properties)

    def remove_unlocated(
        self, *, propagate_properties: bool = True
    ) -> IntegrationPoints:
        r"""
        Remove the unlocated points (element index ``-1``).

        Parameters
        ----------
        propagate_properties: bool, optional (default: ``True``)
            If False, the new integration points have no properties.

        Returns
        -------
        :class:`pysdic.IntegrationPoints`
            The integration points without the unlocated points.

        """
        return self.filter_points(
            self.is_located(), propagate_properties=propagate_properties
        )

    def _unlocate(
        self, mask: numpy.ndarray, propagate_properties: bool
    ) -> IntegrationPoints:
        # Unlocate the points selected by a validated mask: element index -1 and NaN coordinates
        propagate_properties = _validate_bool(propagate_properties, "propagate_properties")

        new_natural_coordinates = self._natural_coordinates.copy()
        new_element_indices = self._element_indices.copy()
        new_natural_coordinates[mask, :] = numpy.nan
        new_element_indices[mask] = -1

        new_properties = None
        if propagate_properties:
            new_properties = self._properties.copy()
        return self._new(new_natural_coordinates, new_element_indices, self._weights, new_properties)

    def unlocate_points_by_index(
        self,
        indices: ArrayLike,
        *,
        propagate_properties: bool = True,
    ) -> IntegrationPoints:
        r"""
        Unlocate the points with the given indices without removing them: their element
        index is set to ``-1`` and their natural coordinates to ``numpy.nan``.

        Parameters
        ----------
        indices: ArrayLike
            An integer array of shape :math:`(R,)` with the indices of the points to
            unlocate, in :math:`[0, N_p)`. Duplicates are allowed.

        propagate_properties: bool, optional (default: ``True``)
            If False, the new integration points have no properties.

        Returns
        -------
        :class:`pysdic.IntegrationPoints`
            The integration points with the given points unlocated.

        Raises
        ------
        TypeError
            If the indices are not integers.

        IndexError
            If an index is out of :math:`[0, N_p)`.

        """
        indices = _as_index_array(indices, self.n_points)
        mask = numpy.zeros(self.n_points, dtype=bool)
        mask[indices] = True
        return self._unlocate(mask, propagate_properties)

    def unlocate_points_by_element(
        self,
        element_indices: ArrayLike,
        *,
        propagate_properties: bool = True,
    ) -> IntegrationPoints:
        r"""
        Unlocate the points located in the given elements without removing them: their
        element index is set to ``-1`` and their natural coordinates to ``numpy.nan``.

        Parameters
        ----------
        element_indices: ArrayLike
            An integer array of shape :math:`(R,)` with the indices of the elements.
            Indices of elements without points are allowed.

        propagate_properties: bool, optional (default: ``True``)
            If False, the new integration points have no properties.

        Returns
        -------
        :class:`pysdic.IntegrationPoints`
            The integration points with the points of the given elements unlocated.

        """
        mask = self._points_in_elements(element_indices)
        return self._unlocate(mask, propagate_properties)

    # -----------------------
    # Elements of the mesh
    # -----------------------
    def filter_elements(
        self,
        mask: ArrayLike,
        *,
        propagate_properties: bool = True,
    ) -> IntegrationPoints:
        r"""
        Update the integration points after the elements of their mesh are filtered
        with a boolean mask (see :meth:`pysdic.Mesh.filter_connectivity`).

        The points of the removed elements are unlocated (element index ``-1`` and
        natural coordinates ``numpy.nan``) but not removed, and the element indices of
        the other points are renumbered like the elements of the filtered mesh.

        Parameters
        ----------
        mask: ArrayLike
            A boolean array of shape :math:`(N_e,)`, with :math:`N_e` the number of
            elements of the mesh, True for the elements to keep.

        propagate_properties: bool, optional (default: ``True``)
            If False, the new integration points have no properties.

        Returns
        -------
        :class:`pysdic.IntegrationPoints`
            The integration points consistent with the filtered mesh.

        Raises
        ------
        TypeError
            If the mask is not a boolean array.

        IndexError
            If a point refers to an element :math:`\geq N_e`.

        Examples
        --------
        .. code-block:: python
            :linenos:

            from pysdic import IntegrationPoints

            integration_points = IntegrationPoints([[0.2], [0.5], [0.1]], [0, 1, 2])
            filtered = integration_points.filter_elements([True, False, True])
            print(filtered.element_indices)  # [ 0 -1  1]

        """
        mask = _as_bool_mask(mask, None, "mask")
        propagate_properties = _validate_bool(propagate_properties, "propagate_properties")
        n_elements = mask.shape[0]

        located = self.is_located()
        if numpy.any(self._element_indices[located] >= n_elements):
            raise IndexError(
                f"The integration points refer to elements out of the mask of {n_elements} "
                f"elements. Maximum element index found: {self._element_indices.max()}."
            )

        new_index = numpy.cumsum(mask) - 1  # old element index -> new element index (kept elements)
        new_element_indices = numpy.full(self.n_points, -1, dtype=numpy.int64)
        kept = located.copy()
        kept[located] = mask[self._element_indices[located]]
        new_element_indices[kept] = new_index[self._element_indices[kept]]

        new_natural_coordinates = self._natural_coordinates.copy()
        new_natural_coordinates[located & ~kept, :] = numpy.nan  # Points of the removed elements

        new_properties = None
        if propagate_properties:
            new_properties = self._properties.copy()
        return self._new(new_natural_coordinates, new_element_indices, self._weights, new_properties)

    def remove_elements(
        self,
        element_indices: ArrayLike,
        n_elements: Integral,
        *,
        propagate_properties: bool = True,
    ) -> IntegrationPoints:
        r"""
        Update the integration points after elements are removed from their mesh (see
        :meth:`pysdic.Mesh.remove_connectivity_by_index`).

        Equivalent to :meth:`filter_elements` with a mask of :math:`N_e` elements, False
        for the removed elements.

        Parameters
        ----------
        element_indices: ArrayLike
            An integer array of shape :math:`(R,)` with the indices of the removed
            elements, in :math:`[0, N_e)`. Duplicates are allowed.

        n_elements: Integral
            The number of elements :math:`N_e` of the mesh before the removal.

        propagate_properties: bool, optional (default: ``True``)
            If False, the new integration points have no properties.

        Returns
        -------
        :class:`pysdic.IntegrationPoints`
            The integration points consistent with the new mesh.

        Raises
        ------
        IndexError
            If an element index is out of :math:`[0, N_e)`, or if a point refers to an
            element :math:`\geq N_e`.

        """
        n_elements = _validate_integer(n_elements, "n_elements", min_value=0)
        element_indices = _as_index_array(element_indices, n_elements, "element_indices")
        mask = numpy.ones(n_elements, dtype=bool)
        mask[element_indices] = False
        return self.filter_elements(mask, propagate_properties=propagate_properties)

    def keep_elements(
        self,
        element_indices: ArrayLike,
        n_elements: Integral,
        *,
        propagate_properties: bool = True,
    ) -> IntegrationPoints:
        r"""
        Update the integration points after only some elements of their mesh are kept
        (see :meth:`pysdic.Mesh.keep_connectivity_by_index`).

        Equivalent to :meth:`filter_elements` with a mask of :math:`N_e` elements, True
        for the kept elements.

        Parameters
        ----------
        element_indices: ArrayLike
            An integer array of shape :math:`(R,)` with the indices of the kept
            elements, in :math:`[0, N_e)`. Duplicates are allowed.

        n_elements: Integral
            The number of elements :math:`N_e` of the mesh before the operation.

        propagate_properties: bool, optional (default: ``True``)
            If False, the new integration points have no properties.

        Returns
        -------
        :class:`pysdic.IntegrationPoints`
            The integration points consistent with the new mesh.

        Raises
        ------
        IndexError
            If an element index is out of :math:`[0, N_e)`, or if a point refers to an
            element :math:`\geq N_e`.

        """
        n_elements = _validate_integer(n_elements, "n_elements", min_value=0)
        element_indices = _as_index_array(element_indices, n_elements, "element_indices")
        mask = numpy.zeros(n_elements, dtype=bool)
        mask[element_indices] = True
        return self.filter_elements(mask, propagate_properties=propagate_properties)