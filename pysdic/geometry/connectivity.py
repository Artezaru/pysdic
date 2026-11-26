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
from typing import Optional, Dict, Any, Union, Tuple
from numpy.typing import ArrayLike
from numbers import Integral, Real

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
    _validate_real,
    _validate_integer,
    _validate_tolerances,
    _as_array,
    _as_integer_array,
    _as_bool_mask,
    _as_index_array,
)
from ..fem._implemented_element_type import _MAPPING_ELEMENT_TYPE_TO_PROPERTIES
from ..fem._validation import _validate_element_type


def _read_element_type(value: Any) -> str:
    r"""
    Read an element type stored by a serializer: a ``str``, ``bytes`` or a numpy array
    with one string (0-d or 1-element, as written by NPZ or HDF5).

    .. important::

        Internal use only.

    """
    if isinstance(value, numpy.ndarray):
        if value.size != 1 or value.dtype.kind not in {"U", "S", "O"}:
            raise ValueError(
                f"The 'element_type' entry must be a string or an array with one string. "
                f"Got an array with dtype {value.dtype} and shape {value.shape}."
            )
        value = value.reshape(-1)[0]
    if isinstance(value, (bytes, numpy.bytes_)):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        raise TypeError(
            f"The 'element_type' entry must be a string. Got type {type(value)}."
        )
    return value


class Connectivity(LockableMixin, JsonSerializerMixin, NpzSerializerMixin, Hdf5SerializerMixin):
    r"""
    The connectivity of a mesh: :math:`N_e` elements, each defined by the indices of
    its :math:`N_{vpe}` vertices.

    Each element can carry additional properties (material, labels, ...) stored in a
    :class:`pysdic.core.property_container.PropertyContainer`, accessible through the ``properties`` attribute
    or directly with ``connectivity[name]``.

    If ``element_type`` is given, the connectivity is associated with one of the
    element types implemented in :mod:`pysdic.fem` (shape functions, Gauss points,
    VTK export, ...):

    +---------------------+-------------------------------------------------------------------+
    | Element Type        | Description                                                       |
    +=====================+===================================================================+
    | "segment_2"         | 2-node line element                                               |
    +---------------------+-------------------------------------------------------------------+
    | "segment_3"         | 3-node line element                                               |
    +---------------------+-------------------------------------------------------------------+
    | "triangle_3"        | 3-node triangular element                                         |
    +---------------------+-------------------------------------------------------------------+
    | "triangle_6"        | 6-node triangular element                                         |
    +---------------------+-------------------------------------------------------------------+
    | "quadrangle_4"      | 4-node quadrilateral element                                      |
    +---------------------+-------------------------------------------------------------------+
    | "quadrangle_8"      | 8-node quadrilateral element                                      |
    +---------------------+-------------------------------------------------------------------+

    .. note::

        Only the number of vertices per element is checked against ``element_type``:
        the ordering of the vertices inside each element is the responsibility of the
        user.

    .. note::

        The vertex indices are copied when they are set and stored as a
        ``numpy.int64`` array. The ``elements`` attribute returns a read-only view: to
        change the elements, assign a new array.

    .. important::

        The properties are stored in a :class:`pysdic.core.property_container.PropertyContainer` with
        ``force_2d=True``, i.e. with shape :math:`(N_e, P)`. A container given by the
        user with other settings is accepted, but some operations (like VTK export) may
        behave unexpectedly.

    Parameters
    ----------
    elements: ArrayLike
        An integer array of shape :math:`(N_e, N_{vpe})` with the non-negative vertex
        indices of each element. Floats are accepted only if they are integral (no
        truncation). A 1D array of shape :math:`(N_e,)` is accepted if
        ``n_vertices_per_element=1``.

    n_vertices_per_element: Optional[Integral], optional (default: ``None``)
        The number of vertices per element :math:`N_{vpe}`. By default, it is inferred
        from the shape of ``elements``. If given, it must match this shape.

    properties: Optional[Union[Dict[str, ArrayLike], PropertyContainer]], optional
        The properties of the elements:

        - a dictionary mapping names to arrays of shape :math:`(N_e, \dots)`, which
          are copied into a new :class:`pysdic.core.property_container.PropertyContainer`,
        - or a :class:`pysdic.core.property_container.PropertyContainer` with :math:`N = N_e`, which is
          **shared** (not copied).

    element_type: Optional[str], optional (default: ``None``)
        The type of the elements (case insensitive, see the table above). Its number
        of vertices must match :math:`N_{vpe}`.


    Notes
    -----
    ``len(connectivity)`` gives the number of elements :math:`N_e`, while
    ``connectivity[name]``, ``name in connectivity`` and ``iter(connectivity)`` refer
    to the properties.

    The methods never modify the connectivity in place: the methods that change the
    elements (:meth:`filter_elements`, :meth:`keep_elements_by_index`, ...) return a new
    connectivity. Only the ``elements`` and ``element_type`` setters (same number of
    elements) and the properties modify an existing connectivity.

    .. note::

        A connectivity can be locked by an object of the package that owns it (for example the
        mesh of an SDIC analysis, see :class:`pysdic.core.lockable.LockableMixin`): the
        setters of ``elements`` and ``element_type`` then raise a :class:`RuntimeError` explaining why. The
        properties can always be modified. A copy (``copy``) is always unlocked.

    """

    __slots__ = ["_element_type", "_elements", "_properties"]

    def __init__(
        self,
        elements: ArrayLike,
        *,
        n_vertices_per_element: Optional[Integral] = None,
        properties: Optional[Union[Dict[str, ArrayLike], PropertyContainer]] = None,
        element_type: Optional[str] = None,
    ) -> None:
        n_vertices_per_element = _validate_integer(
            n_vertices_per_element, "n_vertices_per_element", min_value=1, allow_none=True
        )
        elements = self._as_elements(elements, n_vertices_per_element)
        N = elements.shape[0]

        if properties is None:
            props = PropertyContainer(N=N, force_2d=True)
        elif isinstance(properties, PropertyContainer):
            if properties.N != N:
                raise ValueError(
                    f"The number of entities in 'properties' must match the number of "
                    f"elements. Got N={properties.N} for properties and N_e={N} for elements."
                )
            props = properties
        elif isinstance(properties, Mapping):
            props = PropertyContainer(N=N, properties=properties, force_2d=True)
        else:
            raise TypeError(
                f"Input 'properties' must be None, a dictionary of arrays or a "
                f"PropertyContainer. Got type {type(properties)}."
            )

        self._elements = elements
        self._properties = props
        self._element_type = None
        self.element_type = element_type  # setter: validation and N_vpe consistency
        self.validate(recursive=True)

    @staticmethod
    def _as_elements(value: ArrayLike, n_vertices_per_element: Optional[int]) -> numpy.ndarray:
        # Copy the input to a (N_e, N_vpe) int64 array of non-negative indices.
        # 1D input is only accepted for N_vpe = 1.
        elements = _as_integer_array(
            value, "elements",
            shape=(None, n_vertices_per_element),
            shape_hint="(N_e, N_vpe)" if n_vertices_per_element is None else f"(N_e, {n_vertices_per_element})",
            accept_1d_as_column=(n_vertices_per_element == 1), copy=True,
        )
        if elements.shape[1] == 0:
            raise ValueError("Input 'elements' must have at least one vertex per element.")
        if elements.size > 0 and elements.min() < 0:
            raise ValueError(
                f"Input 'elements' must contain non-negative vertex indices. "
                f"Got a minimum index of {elements.min()}."
            )
        return elements

    def validate(self, *, recursive: bool = True) -> None:
        r"""
        Validate the integrity of the connectivity.

        The following conditions are checked:

        - The elements are a ``numpy.int64`` array of shape :math:`(N_e, N_{vpe})` with
          :math:`N_{vpe} \geq 1` and non-negative indices.
        - If ``element_type`` is set, it is an implemented type with :math:`N_{vpe}`
          vertices.
        - The properties are a :class:`pysdic.core.property_container.PropertyContainer` with
          :math:`N = N_e`.

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

        # --- elements must be a 2D int64 array of shape (Ne, Nvpe) with indices >= 0 ---
        if not isinstance(self._elements, numpy.ndarray):
            raise TypeError("[Connectivity validate] elements should be a numpy ndarray.")
        if self._elements.ndim != 2 or self._elements.shape[1] == 0:
            raise ValueError(
                f"[Connectivity validate] elements should be a 2D array of shape (Ne, Nvpe) "
                f"with Nvpe >= 1. Got shape {self._elements.shape}."
            )
        if self._elements.dtype != numpy.int64:
            raise TypeError(
                f"[Connectivity validate] elements should have dtype int64. "
                f"Got {self._elements.dtype}."
            )
        if self._elements.size > 0 and self._elements.min() < 0:
            raise ValueError(
                f"[Connectivity validate] elements contain negative vertex indices. "
                f"Minimum index found: {self._elements.min()}."
            )

        # --- element type consistency ---
        if self._element_type is not None:
            if self._element_type not in _MAPPING_ELEMENT_TYPE_TO_PROPERTIES:
                raise ValueError(
                    f"[Connectivity validate] Element type '{self._element_type}' is not "
                    f"implemented. Supported types are: "
                    f"{list(_MAPPING_ELEMENT_TYPE_TO_PROPERTIES.keys())}."
                )
            expected_N_vpe = _MAPPING_ELEMENT_TYPE_TO_PROPERTIES[self._element_type]["expected_N_vpe"]
            if self.n_vertices_per_element != expected_N_vpe:
                raise ValueError(
                    f"[Connectivity validate] Current elements have "
                    f"{self.n_vertices_per_element} vertices per element, which does not "
                    f"match the expected {expected_N_vpe} for element type "
                    f"'{self._element_type}'."
                )

        # --- properties must be a PropertyContainer with the same number of elements ---
        if not isinstance(self._properties, PropertyContainer):
            raise TypeError("[Connectivity validate] properties should be a PropertyContainer.")
        if self._properties.N != self.n_elements:
            raise ValueError(
                f"[Connectivity validate] The number of entities in properties should "
                f"match the number of elements. Got N={self._properties.N} for properties "
                f"and N_e={self.n_elements} for elements."
            )

        if recursive:
            self._properties.validate()

    # ==========================
    # API interface
    # ==========================
    @property
    def elements(self) -> numpy.ndarray:
        r"""
        [Get/Set] The vertex indices of the elements, an array of shape
        :math:`(N_e, N_{vpe})`.

        The getter returns a read-only view. The setter copies the given array, whose
        shape must be :math:`(N_e, N_{vpe})`: to change the number of elements, use
        :meth:`filter_elements` or :meth:`concatenate` (which return a new connectivity).
        The setter is refused if the connectivity is locked.

        Parameters
        ----------
        value: ArrayLike
            An integer array of shape :math:`(N_e, N_{vpe})` with non-negative
            indices. A 1D array of shape :math:`(N_e,)` is accepted if
            :math:`N_{vpe} = 1`.

        Returns
        -------
        numpy.ndarray
            A read-only view of shape :math:`(N_e, N_{vpe})` on the elements.

        Raises
        ------
        TypeError
            If the input does not contain integers.

        ValueError
            If the shape of the input is not :math:`(N_e, N_{vpe})` or if it contains
            negative indices.

        RuntimeError
            If the connectivity is locked (see ``locked_message``).

        """
        view = self._elements.view()
        view.flags.writeable = False
        return view

    @elements.setter
    def elements(self, value: ArrayLike) -> None:
        self._check_unlocked("set the elements")
        elements = self._as_elements(value, self.n_vertices_per_element)
        if elements.shape[0] != self.n_elements:
            raise ValueError(
                f"Input 'elements' must keep the number of elements N_e={self.n_elements}. "
                f"Got {elements.shape[0]} elements. Consider using the filter/concatenate "
                f"methods to change the number of elements."
            )
        self._elements = elements

    @property
    def n_elements(self) -> int:
        r"""
        [Get] The number of elements :math:`N_e`.

        Returns
        -------
        int
            The number of elements.

        """
        return self._elements.shape[0]

    def __len__(self) -> int:
        r"""Return the number of elements :math:`N_e`."""
        return self.n_elements

    @property
    def n_vertices_per_element(self) -> int:
        r"""
        [Get] The number of vertices per element :math:`N_{vpe}`.

        Returns
        -------
        int
            The number of vertices per element.

        """
        return self._elements.shape[1]

    @property
    def element_type(self) -> Optional[str]:
        r"""
        [Get/Set] The type of the elements (see the table in
        :class:`pysdic.Connectivity`), or ``None`` if no type is associated.

        Parameters
        ----------
        value: Optional[str]
            An implemented element type (case insensitive) with :math:`N_{vpe}`
            vertices, or ``None``.

        Returns
        -------
        Optional[str]
            The element type in lower case, or ``None``.

        Raises
        ------
        TypeError
            If the value is not a string or ``None``.

        ValueError
            If the type is not implemented or if its number of vertices does not match
            :math:`N_{vpe}`.

        RuntimeError
            If the connectivity is locked (see ``locked_message``).

        """
        return self._element_type

    @element_type.setter
    def element_type(self, value: Optional[str]) -> None:
        self._check_unlocked("set the element type")
        if value is not None:
            value = _validate_element_type(value)
            expected_N_vpe = _MAPPING_ELEMENT_TYPE_TO_PROPERTIES[value]["expected_N_vpe"]
            if self.n_vertices_per_element != expected_N_vpe:
                raise ValueError(
                    f"Current elements have {self.n_vertices_per_element} vertices per "
                    f"element, which does not match the expected {expected_N_vpe} for "
                    f"element type '{value}'."
                )
        self._element_type = value

    def _element_type_property(self, name: str) -> Optional[Any]:
        # Property of the element type in the fem mapping, or None if no type is set
        if self._element_type is None:
            return None
        return _MAPPING_ELEMENT_TYPE_TO_PROPERTIES[self._element_type][name]

    @property
    def n_topological_dimensions(self) -> Optional[int]:
        r"""
        [Get] The topological dimension :math:`K` of the elements (1 for segments, 2
        for triangles and quadrangles), or ``None`` if ``element_type`` is not set.

        Returns
        -------
        Optional[int]
            The topological dimension of the elements.

        """
        return self._element_type_property("expected_K")

    @property
    def shape(self) -> Tuple[int, int]:
        r"""
        [Get] The shape :math:`(N_e, N_{vpe})` of the elements.

        Returns
        -------
        Tuple[int, int]
            The number of elements and the number of vertices per element.

        """
        return (self.n_elements, self.n_vertices_per_element)

    @property
    def properties(self) -> PropertyContainer:
        r"""
        [Get] The properties of the elements.

        Returns
        -------
        :class:`pysdic.core.property_container.PropertyContainer`
            The property container (not a copy).

        """
        return self._properties

    def __getitem__(self, key):
        r"""Shortcut for ``connectivity.properties[key]``."""
        return self._properties[key]

    def __setitem__(self, key, value):
        r"""Shortcut for ``connectivity.properties[key] = value``."""
        self._properties[key] = value

    def __delitem__(self, key):
        r"""Shortcut for ``del connectivity.properties[key]``."""
        del self._properties[key]

    def __contains__(self, key):
        r"""Shortcut for ``key in connectivity.properties``."""
        return key in self._properties

    def __iter__(self):
        r"""Shortcut for ``iter(connectivity.properties)`` (iterates over the names)."""
        return iter(self._properties)

    @property
    def n_properties(self) -> int:
        r"""
        [Get] The number of properties of the elements.

        Returns
        -------
        int
            The number of properties.

        """
        return self._properties.n_properties

    @property
    def _meshio_cell_type(self) -> Optional[str]:
        # meshio cell type of the elements, or None if element_type is not set
        return self._element_type_property("meshio_cell_type")

    @property
    def _vtk_cell_type(self) -> Optional[int]:
        # VTK cell type of the elements, or None if element_type is not set
        return self._element_type_property("vtk_cell_type")

    @property
    def _expected_N_vpe(self) -> Optional[int]:
        # Number of vertices of the element type, or None if element_type is not set
        return self._element_type_property("expected_N_vpe")

    @property
    def _expected_K(self) -> Optional[int]:
        # Same as n_topological_dimensions (kept for internal callers)
        return self._element_type_property("expected_K")

    def __repr__(self) -> str:
        string = (
            f"Connectivity(n_elements={self.n_elements}, "
            f"n_vertices_per_element={self.n_vertices_per_element})"
        )
        if self._element_type is not None:
            string += f"\n  - Element Type: '{self._element_type}'"
        for key in self._properties.keys():
            string += f"\n  - Property: '{key}' with shape {self._properties.get_shape(key)}"
        return string

    # ==========================
    # I/O Methods
    # ==========================
    @classmethod
    def from_array(
        cls,
        elements: ArrayLike,
        *,
        properties: Optional[Union[Dict[str, ArrayLike], PropertyContainer]] = None,
        element_type: Optional[str] = None,
    ) -> Connectivity:
        r"""
        Create a connectivity from an array of elements.

        Equivalent to ``Connectivity(elements, properties=properties,
        element_type=element_type)``.

        Parameters
        ----------
        elements: ArrayLike
            An integer array of shape :math:`(N_e, N_{vpe})` with the vertex indices.

        properties: Optional[Union[Dict[str, ArrayLike], PropertyContainer]], optional
            The properties of the elements (see :class:`pysdic.Connectivity`).

        element_type: Optional[str], optional (default: ``None``)
            The type of the elements.

        Returns
        -------
        :class:`pysdic.Connectivity`
            The new connectivity.

        """
        return cls(elements=elements, properties=properties, element_type=element_type)

    def to_array(self) -> numpy.ndarray:
        r"""
        Return a copy of the elements.

        Returns
        -------
        numpy.ndarray
            A writable array of shape :math:`(N_e, N_{vpe})`.

        """
        return self._elements.copy()

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        *,
        load_properties: bool = True,
        element_type: Optional[str] = None,
    ) -> Connectivity:
        r"""
        Create a connectivity from a dictionary.

        The dictionary must contain the key ``"elements"`` with an array of shape
        :math:`(N_e, N_{vpe})`. The properties are read from the keys
        ``"_property/{property_name}"``, and the element type from the optional key
        ``"element_type"`` (a string, or an array with one string as written by the NPZ
        and HDF5 serializers).

        .. note::

            The dictionary is flat (one level), as required by the NPZ format. See
            :class:`pysdic.core.serialization.NpzSerializerMixin`.

        Parameters
        ----------
        data: Dict[str, Any]
            The dictionary to read.

        load_properties: bool, optional (default: ``True``)
            If False, the properties are ignored.

        element_type: Optional[str], optional (default: ``None``)
            The expected element type. If the dictionary also defines one, both must
            be the same.

        Returns
        -------
        :class:`pysdic.Connectivity`
            The new connectivity.

        """
        if not isinstance(data, Mapping):
            raise TypeError(f"Input 'data' must be a dictionary. Got type {type(data)}.")
        if "elements" not in data:
            raise ValueError(
                "Input dictionary must contain a key 'elements' with shape (N_e, N_vpe)."
            )
        load_properties = _validate_bool(load_properties, "load_properties")

        properties = {}
        if load_properties:
            prefix = "_property/"
            for key in data:
                if isinstance(key, str) and key.startswith(prefix):
                    properties[key[len(prefix):]] = data[key]

        if element_type is not None:
            element_type = _validate_element_type(element_type)
        if "element_type" in data:
            read_element_type = _validate_element_type(_read_element_type(data["element_type"]))
            if element_type is not None and read_element_type != element_type:
                raise ValueError(
                    f"The provided element_type '{element_type}' does not match the "
                    f"element_type '{read_element_type}' in the input dictionary."
                )
            element_type = read_element_type

        return cls(elements=data["elements"], properties=properties, element_type=element_type)

    def to_dict(self, *, save_properties: bool = True) -> Dict[str, Any]:
        r"""
        Convert the connectivity to a flat dictionary.

        The elements are stored under the key ``"elements"``, the properties under the
        keys ``"_property/{property_name}"`` and the element type (if set) under the key
        ``"element_type"``. All the arrays are copies.

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
            The dictionary representation of the connectivity.

        """
        save_properties = _validate_bool(save_properties, "save_properties")

        data = {"elements": self._elements.copy()}
        if save_properties:
            for key, values in self._properties.to_dict().items():
                data[f"_property/{key}"] = values
        if self._element_type is not None:
            data["element_type"] = self._element_type
        return data

    # =======================
    # Structure Methods
    # =======================
    def concatenate(
        self, other: Connectivity, *, propagate_properties: bool = True
    ) -> Connectivity:
        r"""
        Concatenate two connectivities (the elements of ``other`` come after).

        .. note::

            The vertex indices are not shifted: both connectivities must refer to the
            same set of vertices.

        The element type of the result is the common element type of both
        connectivities, or ``None`` if one of them has no type.

        The properties are concatenated with :meth:`pysdic.core.property_container.PropertyContainer.concatenate`:
        a property present in only one connectivity is completed with ``numpy.nan``
        for the elements of the other one (integer and boolean properties are then
        promoted to float).

        Parameters
        ----------
        other: :class:`pysdic.Connectivity`
            The connectivity to append, with the same :math:`N_{vpe}`.

        propagate_properties: bool, optional (default: ``True``)
            If False, the new connectivity has no properties.

        Returns
        -------
        :class:`pysdic.Connectivity`
            A new connectivity with :math:`N_e + N_e^{other}` elements.

        Raises
        ------
        ValueError
            If the numbers of vertices per element differ, or if both connectivities
            have different element types.

        """
        if not isinstance(other, Connectivity):
            raise TypeError(f"Input 'other' must be a Connectivity. Got type {type(other)}.")
        propagate_properties = _validate_bool(propagate_properties, "propagate_properties")
        if self.n_vertices_per_element != other.n_vertices_per_element:
            raise ValueError(
                f"Connectivity instances must have the same number of vertices per "
                f"element to be concatenated. Got {self.n_vertices_per_element} and "
                f"{other.n_vertices_per_element}."
            )
        if (
            self._element_type is not None
            and other._element_type is not None
            and self._element_type != other._element_type
        ):
            raise ValueError(
                f"Connectivity instances with different element types cannot be "
                f"concatenated. Got '{self._element_type}' and '{other._element_type}'."
            )
        element_type = self._element_type if self._element_type == other._element_type else None

        concatenated_properties = None
        if propagate_properties:
            concatenated_properties = self._properties.concatenate(
                other._properties, fill=numpy.nan
            )

        return self.__class__(
            elements=numpy.vstack((self._elements, other._elements)),
            n_vertices_per_element=self.n_vertices_per_element,
            properties=concatenated_properties,
            element_type=element_type,
        )

    def copy(self, *, copy_properties: bool = True) -> Connectivity:
        r"""
        Create a copy of the connectivity (with the same element type). The copy is
        always unlocked.

        Parameters
        ----------
        copy_properties: bool, optional (default: ``True``)
            If False, the copy has no properties.

        Returns
        -------
        :class:`pysdic.Connectivity`
            A new connectivity with copies of the elements and properties.

        """
        copy_properties = _validate_bool(copy_properties, "copy_properties")

        copied_properties = None
        if copy_properties:
            copied_properties = self._properties.copy()

        return self.__class__(
            elements=self._elements,  # copied by the constructor
            n_vertices_per_element=self.n_vertices_per_element,
            properties=copied_properties,
            element_type=self._element_type,
        )

    def filter_elements(
        self,
        mask: ArrayLike,
        *,
        propagate_properties: bool = True,
    ) -> Connectivity:
        r"""
        Keep only the elements selected by a boolean mask (the element type is kept).

        Parameters
        ----------
        mask: ArrayLike
            A boolean array of shape :math:`(N_e,)`. The elements with a True value
            are kept. A mask with only False values gives an empty connectivity.

        propagate_properties: bool, optional (default: ``True``)
            If False, the new connectivity has no properties.

        Returns
        -------
        :class:`pysdic.Connectivity`
            The filtered connectivity.

        Raises
        ------
        TypeError
            If the mask is not a boolean array (an integer array is rejected: use
            :meth:`keep_elements_by_index` for indices).

        ValueError
            If the mask does not have the shape :math:`(N_e,)`.

        """
        mask = _as_bool_mask(mask, self.n_elements)
        propagate_properties = _validate_bool(propagate_properties, "propagate_properties")

        new_elements = self._elements[mask]  # advanced indexing: copy

        new_properties = None
        if propagate_properties:
            new_properties = self._properties.filter(mask)

        return self.__class__(
            elements=new_elements,
            n_vertices_per_element=self.n_vertices_per_element,
            properties=new_properties,
            element_type=self._element_type,
        )

    def remove_elements_by_index(
        self,
        indices: ArrayLike,
        *,
        propagate_properties: bool = True,
    ) -> Connectivity:
        r"""
        Remove the elements with the given indices.

        Parameters
        ----------
        indices: ArrayLike
            An integer array of shape :math:`(R,)` with the indices of the elements to
            remove, in :math:`[0, N_e)`. Duplicates are allowed.

        propagate_properties: bool, optional (default: ``True``)
            If False, the new connectivity has no properties.

        Returns
        -------
        :class:`pysdic.Connectivity`
            The connectivity without the given elements.

        Raises
        ------
        TypeError
            If the indices are not integers.

        IndexError
            If an index is out of :math:`[0, N_e)`.

        """
        indices = _as_index_array(indices, self.n_elements)
        mask = numpy.ones(self.n_elements, dtype=bool)
        mask[indices] = False
        return self.filter_elements(
            mask, propagate_properties=propagate_properties
        )

    def keep_elements_by_index(
        self,
        indices: ArrayLike,
        *,
        propagate_properties: bool = True,
    ) -> Connectivity:
        r"""
        Keep only the elements with the given indices.

        .. note::

            The elements keep their original order and duplicated indices are ignored:
            ``keep_elements_by_index([2, 0])`` returns the elements 0 and 2, in this
            order.

        Parameters
        ----------
        indices: ArrayLike
            An integer array of shape :math:`(R,)` with the indices of the elements to
            keep, in :math:`[0, N_e)`.

        propagate_properties: bool, optional (default: ``True``)
            If False, the new connectivity has no properties.

        Returns
        -------
        :class:`pysdic.Connectivity`
            The connectivity with only the given elements.

        Raises
        ------
        TypeError
            If the indices are not integers.

        IndexError
            If an index is out of :math:`[0, N_e)`.

        """
        indices = _as_index_array(indices, self.n_elements)
        mask = numpy.zeros(self.n_elements, dtype=bool)
        mask[indices] = True
        return self.filter_elements(
            mask, propagate_properties=propagate_properties
        )

    # ==========================
    # Comparison and checks
    # ==========================
    def all_close(
        self,
        other: Union[Connectivity, ArrayLike],
        *,
        rtol: Real = 1e-05,
        atol: Real = 1e-08,
        equal_nan: bool = True,
        ordered: bool = True,
    ) -> bool:
        r"""
        Check if the elements are equal to the elements of another connectivity (see
        ``numpy.allclose``).

        Connectivities with different shapes are never close. The element types and
        the properties are not compared.

        .. note::

            With ``ordered=False``, the elements (rows) are sorted before the
            comparison, but the order of the vertices inside each element still
            matters: ``[0, 1, 2]`` and ``[1, 2, 0]`` are different elements.

        Parameters
        ----------
        other: Union[:class:`pysdic.Connectivity`, ArrayLike]
            Another connectivity or an array of shape :math:`(N_e, N_{vpe})`.

        rtol: Real, optional (default: ``1e-05``)
            The relative tolerance (non-negative and finite).

        atol: Real, optional (default: ``1e-08``)
            The absolute tolerance (non-negative and finite).

        equal_nan: bool, optional (default: ``True``)
            If True, NaN values are considered equal (only relevant for a float array
            ``other``).

        ordered: bool, optional (default: ``True``)
            If False, the order of the elements is ignored (see the note above).

        Returns
        -------
        bool
            True if all the elements are close, False otherwise.

        Examples
        --------
        .. code-block:: python
            :linenos:

            import numpy
            from pysdic import Connectivity

            connectivity = Connectivity([[0, 1, 2], [1, 3, 2]])

            print(connectivity.all_close([[1, 3, 2], [0, 1, 2]]))                 # False
            print(connectivity.all_close([[1, 3, 2], [0, 1, 2]], ordered=False))  # True

        """
        rtol, atol = _validate_tolerances(rtol, atol)
        equal_nan = _validate_bool(equal_nan, "equal_nan")
        ordered = _validate_bool(ordered, "ordered")

        if isinstance(other, Connectivity):
            other_elements = other._elements
        else:
            other_elements = _as_array(
                other, "other", numpy.float64, shape=(None, None), shape_hint="(N_e, N_vpe)", finite=False
            )

        if self._elements.shape != other_elements.shape:
            return False

        self_elements = self._elements
        if not ordered:
            # lexsort uses the last key as the primary one
            self_elements = self_elements[numpy.lexsort(self_elements.T[::-1])]
            other_elements = other_elements[numpy.lexsort(other_elements.T[::-1])]

        return bool(
            numpy.allclose(
                self_elements, other_elements, rtol=rtol, atol=atol, equal_nan=equal_nan
            )
        )

    def all_positive(self) -> bool:
        r"""
        Check if all the vertex indices are non-negative (:math:`\geq 0`).

        .. note::

            Always True for a valid connectivity, since negative indices are rejected.

        Returns
        -------
        bool
            True if all the indices are non-negative.

        """
        return bool((self._elements >= 0).all())

    def all_lower_than(self, value: Real) -> bool:
        r"""
        Check if all the vertex indices are strictly lower than ``value``.

        For example, ``connectivity.all_lower_than(n_vertices)`` checks that all the
        indices refer to existing vertices.

        Parameters
        ----------
        value: Real
            The threshold.

        Returns
        -------
        bool
            True if all the indices are :math:`<` ``value``.

        """
        value = _validate_real(value, "value")
        return bool((self._elements < value).all())

    def is_positive(self) -> numpy.ndarray:
        r"""
        Check which elements have only non-negative vertex indices (:math:`\geq 0`).

        .. note::

            Always True for a valid connectivity, since negative indices are rejected.

        Returns
        -------
        numpy.ndarray
            A boolean array of shape :math:`(N_e,)`.

        """
        return (self._elements >= 0).all(axis=1)

    def has_negative(self) -> numpy.ndarray:
        r"""
        Check which elements have at least one negative vertex index (:math:`< 0`).

        .. note::

            Always False for a valid connectivity, since negative indices are rejected.

        Returns
        -------
        numpy.ndarray
            A boolean array of shape :math:`(N_e,)`.

        """
        return (self._elements < 0).any(axis=1)

    def has_higher_than(self, value: Real) -> numpy.ndarray:
        r"""
        Check which elements have at least one vertex index strictly higher than
        ``value``.

        .. note::

            To find the elements referring to missing vertices, use
            ``connectivity.has_higher_than(n_vertices - 1)``: an index equal to
            ``n_vertices`` is already invalid.

        Parameters
        ----------
        value: Real
            The threshold.

        Returns
        -------
        numpy.ndarray
            A boolean array of shape :math:`(N_e,)`, True for the elements with an
            index :math:`>` ``value``.

        """
        value = _validate_real(value, "value")
        return (self._elements > value).any(axis=1)