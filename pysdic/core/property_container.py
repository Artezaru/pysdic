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
from numpy.typing import ArrayLike
from typing import Dict, List, Optional, Tuple, Union
from numbers import Integral, Real

import numpy

from .serialization import JsonSerializerMixin, Hdf5SerializerMixin, NpzSerializerMixin
from .._validation import (
    _validate_bool,
    _validate_integer,
    _validate_str,
    _validate_choice,
    _as_bool_mask,
)


class PropertyContainer(JsonSerializerMixin, Hdf5SerializerMixin, NpzSerializerMixin):
    r"""
    A container for properties associated with points or other entities.

    Given :math:`N` entities, each property is a named NumPy array of shape
    :math:`(N, \dots)`, where :math:`\dots` represents the attributes of one entity.
    For example:

    - a scalar property has shape :math:`(N,)` or :math:`(N, 1)`,
    - a 3D vector property has shape :math:`(N, 3)`,
    - a tensor property may have additional dimensions, e.g. :math:`(N, 3, 3)`.

    The container exposes a dictionary-like interface (``[]``, ``in``, ``keys()``, ...)
    but it is not a ``dict``: all the properties must share the same first
    dimension :math:`N`, and their shapes are validated when they are set.

    .. note::

        The arrays are copied when they are set, and returned as read-only views.
        Modifying the input array after setting it, or trying to write into a returned
        array, cannot change the content of the container.

    Parameters
    ----------
    N : Integral
        The number of entities :math:`N`, i.e. the first dimension of all the property
        arrays.

    properties : Dict[str, ArrayLike], optional
        Initial properties, mapping property names to arrays of shape
        :math:`(N, \dots)`. By default the container is empty.

    force_2d : bool, optional (default: ``False``)
        If True, all properties are stored as 2D arrays of shape :math:`(N, P)`.
        1D arrays of shape :math:`(N,)` are reshaped to :math:`(N, 1)` and arrays with
        more than two dimensions are rejected.


    Notes
    -----
    The number of entities :math:`N` is fixed at the creation of the container: the
    methods that change it (:meth:`filter`, :meth:`concatenate`) return a new container.
    A container shared by several objects (for example two point clouds built on the
    same container) can therefore never become inconsistent with them. The properties
    can still be added, modified or deleted: all the references observe these changes.

    Use :meth:`validate` to check the integrity of the container.

    """

    __slots__ = ["_N", "_properties", "_force_2d"]

    def __init__(
        self,
        N: Integral,
        properties: Optional[Dict[str, ArrayLike]] = None,
        *,
        force_2d: bool = False,
    ):
        if properties is not None and not isinstance(properties, Mapping):
            raise TypeError(
                f"Input 'properties' must be a dictionary or None. Got type {type(properties)}."
            )

        self._N = _validate_integer(N, "N", min_value=0)
        self._properties: Dict[str, numpy.ndarray] = {}
        self._force_2d = _validate_bool(force_2d, "force_2d")

        if properties is not None:
            for key, value in properties.items():
                self.set(key, value)

        self.validate()

    # ==========================
    # Internal helpers
    # ==========================
    @staticmethod
    def _check_key_type(key: str) -> str:
        # The key must be a string (the empty string is accepted)
        return _validate_str(key, "key", allow_empty=True)

    def _check_key(self, key: str) -> str:
        # The key must be a string and an existing property
        key = self._check_key_type(key)
        if key not in self._properties:
            raise KeyError(f"Property '{key}' not found in the container.")
        return key

    # ==========================
    # Validation
    # ==========================
    def validate(self) -> None:
        r"""
        Validate the integrity of the container.

        The following conditions are checked:

        - :math:`N` is a non-negative python integer.
        - All the keys are strings and all the values are NumPy arrays.
        - All the properties have the same first dimension :math:`N` and at least one
          attribute (the product of the other dimensions is not zero).
        - If ``force_2d`` is enabled, all the properties are 2D arrays.

        Raises
        ------
        TypeError
            If a key, a value or :math:`N` has a wrong type.

        ValueError
            If a value or :math:`N` has a wrong shape or value.

        """
        # --- N must be a non-negative integer ---
        if not isinstance(self._N, int) or isinstance(self._N, bool):
            raise TypeError(
                f"[PropertyContainer validate] N must be an integer, but got {type(self._N)}."
            )
        if self._N < 0:
            raise ValueError(
                f"[PropertyContainer validate] N must be a non-negative integer, but got {self._N}."
            )

        # --- _properties must be a dictionary mapping strings to numpy arrays ---
        if not isinstance(self._properties, dict):
            raise TypeError(
                f"[PropertyContainer validate] Internal _properties must be a dictionary, "
                f"but got {type(self._properties)}."
            )
        for key, value in self._properties.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"[PropertyContainer validate] Property keys must be strings, "
                    f"but got key of type {type(key)}."
                )
            if not isinstance(value, numpy.ndarray):
                raise TypeError(
                    f"[PropertyContainer validate] Property values must be numpy arrays, "
                    f"but got value of type {type(value)} for key '{key}'."
                )
            if value.ndim == 0:
                raise ValueError(
                    f"[PropertyContainer validate] Property '{key}' must have at least "
                    f"one dimension, but got a scalar."
                )
            if value.shape[0] != self._N:
                raise ValueError(
                    f"[PropertyContainer validate] Property '{key}' must have first "
                    f"dimension of size {self._N}, but got shape {value.shape}."
                )
            if numpy.prod(value.shape[1:]) == 0:
                raise ValueError(
                    f"[PropertyContainer validate] Property '{key}' must have at least "
                    f"one attribute, but got shape {value.shape}."
                )
            if self._force_2d and value.ndim != 2:
                raise ValueError(
                    f"[PropertyContainer validate] Property '{key}' must have exactly 2 "
                    f"dimensions when force_2d is enabled, but got shape {value.shape}."
                )

    # ==========================
    # Attributes
    # ==========================
    @property
    def N(self) -> int:
        r"""
        [Get] The number of entities :math:`N`, i.e. the first dimension of all the
        property arrays.

        :math:`N` is fixed at the creation of the container, even if the container is
        empty: the objects using the container (point clouds, connectivities, ...) rely
        on it. Use :meth:`filter` or :meth:`concatenate` to get a new container with
        another number of entities.

        Returns
        -------
        int
            The number of entities.

        """
        return self._N

    @property
    def force_2d(self) -> bool:
        r"""
        [Get/Set] Whether all the properties are stored as 2D arrays of shape
        :math:`(N, P)`.

        Enabling ``force_2d`` reshapes the existing 1D properties to :math:`(N, 1)`.
        If an existing property has more than two dimensions, a ``ValueError`` is raised
        and the container is left unchanged: use :meth:`flatten` on that property first.

        Parameters
        ----------
        value : bool
            True to enforce 2D properties.

        Returns
        -------
        bool
            True if ``force_2d`` is enabled, False otherwise.

        """
        return self._force_2d

    @force_2d.setter
    def force_2d(self, value: bool):
        value = _validate_bool(value, "force_2d")
        if value:
            too_large = [key for key, array in self._properties.items() if array.ndim > 2]
            if too_large:
                raise ValueError(
                    f"Cannot enable force_2d: the properties {too_large} have more than 2 "
                    f"dimensions. Consider using the 'flatten' method on them first."
                )
            for key, array in self._properties.items():
                if array.ndim == 1:
                    self._properties[key] = array.reshape(self._N, 1)
        self._force_2d = value

    @property
    def n_properties(self) -> int:
        r"""
        [Get] The number of properties stored in the container.

        .. note::

            You can also use ``len(container)``.

        Returns
        -------
        int
            The number of properties.

        """
        return len(self._properties)

    def __len__(self) -> int:
        r"""Return the number of properties (an empty container is falsy)."""
        return self.n_properties

    def __repr__(self) -> str:
        return f"PropertyContainer(N={self._N}, properties={list(self._properties.keys())})"

    # ==========================
    # Dictionary-like interface
    # ==========================
    def get(self, key: str) -> numpy.ndarray:
        r"""
        Get the values of a property.

        Parameters
        ----------
        key : str
            The name of the property.

        Returns
        -------
        numpy.ndarray
            A read-only view of shape :math:`(N, \dots)` on the property values.

        Raises
        ------
        TypeError
            If the key is not a string.

        KeyError
            If the property does not exist.

        """
        self._check_key(key)
        view = self._properties[key].view()
        view.flags.writeable = False
        return view

    def __getitem__(self, key: str) -> numpy.ndarray:
        r"""Alias of :meth:`get`."""
        return self.get(key)

    def set(
        self,
        key: str,
        value: ArrayLike,
        *,
        dtype: Optional[numpy.dtype] = None,
        force_2d: Optional[bool] = None,
    ) -> None:
        r"""
        Set the values of a property (the input is copied).

        Parameters
        ----------
        key : str
            The name of the property.

        value : ArrayLike
            An array of shape :math:`(N, \dots)` with the values of the property.

        dtype : numpy.dtype, optional
            The data type of the stored array. By default, it is inferred from
            ``value``.

        force_2d : bool, optional
            If True, the property is stored as a 2D array even if the container does
            not enforce ``force_2d``. If the container enforces ``force_2d``, this
            parameter has no effect.

        Raises
        ------
        TypeError
            If the key is not a string.

        ValueError
            If the shape of ``value`` is not :math:`(N, \dots)` with at least one
            attribute, or if the property cannot be stored as a 2D array.

        """
        key = self._check_key_type(key)
        if force_2d is not None:
            force_2d = _validate_bool(force_2d, "force_2d")  # validated even if unused
        force_2d = self._force_2d or bool(force_2d)

        value_array = numpy.array(value, dtype=dtype)  # always a copy
        if value_array.ndim == 0:
            raise ValueError(
                f"Invalid shape for property '{key}'. Expected an array with at least "
                f"one dimension, but got a scalar."
            )
        if value_array.shape[0] != self._N:
            raise ValueError(
                f"Invalid shape for property '{key}'. Expected first dimension to be "
                f"{self._N}, but got {value_array.shape[0]}."
            )
        if numpy.prod(value_array.shape[1:]) == 0:
            raise ValueError(
                f"Invalid shape for property '{key}'. The property must have at least one "
                f"attribute, but got shape {value_array.shape}."
            )

        if force_2d:
            if value_array.ndim == 1:
                value_array = value_array.reshape(self._N, 1)
            elif value_array.ndim > 2:
                raise ValueError(
                    f"Invalid shape for property '{key}' with force_2d enabled. Expected "
                    f"an array with at most 2 dimensions, but got {value_array.ndim} dimensions."
                )

        self._properties[key] = value_array

    def __setitem__(self, key: str, value: ArrayLike):
        r"""Alias of :meth:`set`."""
        self.set(key, value)

    def delete(self, key: str) -> None:
        r"""
        Delete a property.

        Parameters
        ----------
        key : str
            The name of the property to delete.

        Raises
        ------
        TypeError
            If the key is not a string.

        KeyError
            If the property does not exist.

        """
        self._check_key(key)
        del self._properties[key]

    def __delitem__(self, key: str):
        r"""Alias of :meth:`delete`."""
        self.delete(key)

    def clear(self) -> None:
        r"""
        Remove all the properties. :math:`N` is unchanged.
        """
        self._properties.clear()

    def has(self, key: str) -> bool:
        r"""
        Check if a property exists.

        Parameters
        ----------
        key : str
            The name of the property.

        Returns
        -------
        bool
            True if the property exists, False otherwise (including when ``key`` is
            not a string).

        """
        return isinstance(key, str) and key in self._properties

    def __contains__(self, key: str) -> bool:
        r"""Alias of :meth:`has`."""
        return self.has(key)

    def keys(self):
        r"""
        Get the names of the properties.

        Returns
        -------
        dict_keys
            A dynamic view on the property names.

        """
        return self._properties.keys()

    def items(self) -> List[Tuple[str, numpy.ndarray]]:
        r"""
        Get the properties as ``(name, values)`` pairs.

        Returns
        -------
        List[Tuple[str, numpy.ndarray]]
            The list of the properties, with read-only views on the values.

        """
        return [(key, self.get(key)) for key in self._properties]

    def values(self) -> List[numpy.ndarray]:
        r"""
        Get the values of the properties.

        Returns
        -------
        List[numpy.ndarray]
            The list of read-only views on the property values.

        """
        return [self.get(key) for key in self._properties]

    def __iter__(self):
        r"""Iterate over the property names."""
        return iter(self._properties.keys())

    # ==========================
    # Inspection
    # ==========================
    def get_dtype(self, key: str) -> numpy.dtype:
        r"""
        Get the data type of a property.

        Parameters
        ----------
        key : str
            The name of the property.

        Returns
        -------
        numpy.dtype
            The data type of the property.

        """
        self._check_key(key)
        return self._properties[key].dtype

    def get_shape(self, key: str) -> Tuple[int, ...]:
        r"""
        Get the shape :math:`(N, \dots)` of a property.

        Parameters
        ----------
        key : str
            The name of the property.

        Returns
        -------
        Tuple[int, ...]
            The shape of the property.

        """
        self._check_key(key)
        return self._properties[key].shape

    def get_ndim(self, key: str) -> int:
        r"""
        Get the number of dimensions of a property.

        Parameters
        ----------
        key : str
            The name of the property.

        Returns
        -------
        int
            The number of dimensions of the property.

        """
        self._check_key(key)
        return self._properties[key].ndim

    def check(
        self,
        key: str,
        *,
        shape: Optional[Tuple[int, ...]] = None,
        dtype: Optional[numpy.dtype] = None,
    ) -> bool:
        r"""
        Check if a property exists and has the expected shape and data type.

        Parameters
        ----------
        key : str
            The name of the property.

        shape : Tuple[int, ...], optional
            The expected shape of the property (including :math:`N`).

        dtype : numpy.dtype, optional
            The expected data type of the property.

        Returns
        -------
        bool
            True if the property exists and matches the given ``shape`` and ``dtype``,
            False otherwise.

        """
        if not self.has(key):
            return False

        prop = self._properties[key]
        if shape is not None and prop.shape != tuple(shape):
            return False
        if dtype is not None and prop.dtype != numpy.dtype(dtype):
            return False
        return True

    # ==========================
    # Reshaping
    # ==========================
    def flatten(self, key: str, *, order: str = "C") -> None:
        r"""
        Flatten a property from shape :math:`(N, \dots)` to shape :math:`(N, P)`,
        where :math:`P` is the product of the attribute dimensions.

        Parameters
        ----------
        key : str
            The name of the property.

        order : {'C', 'F'}, optional (default: ``'C'``)
            Read the attributes in row-major (``'C'``) or column-major (``'F'``) order.

        """
        self._check_key(key)
        order = _validate_choice(order, "order", ("C", "F"))
        array = self._properties[key]
        self._properties[key] = array.reshape(
            self._N, int(numpy.prod(array.shape[1:])), order=order
        )

    def reshape(self, key: str, new_shape: Union[Integral, Tuple[int, ...]]) -> None:
        r"""
        Reshape the attributes of a property, keeping the first dimension :math:`N`.

        For example, a property of shape :math:`(N, 9)` can be reshaped to
        :math:`(N, 3, 3)` with ``new_shape=(3, 3)``.

        Parameters
        ----------
        key : str
            The name of the property.

        new_shape : Union[Integral, Tuple[int, ...]]
            The new shape of the attributes (without :math:`N`). It must contain the
            same number of attributes as the current shape.

        Raises
        ------
        ValueError
            If the number of attributes changes, or if the new shape is not 2D while
            ``force_2d`` is enabled.

        """
        self._check_key(key)
        if isinstance(new_shape, Integral):
            new_shape = (new_shape,)
        if not isinstance(new_shape, (tuple, list, numpy.ndarray)):
            raise TypeError(
                f"Input 'new_shape' must be an integer or a tuple of integers. "
                f"Got type {type(new_shape)}."
            )
        new_shape = tuple(_validate_integer(n, "new_shape", min_value=1) for n in new_shape)

        original_shape = self._properties[key].shape[1:]
        if numpy.prod(new_shape) != numpy.prod(original_shape):
            raise ValueError(
                f"New shape {new_shape} is not compatible with the original shape "
                f"{original_shape} for property '{key}'."
            )
        if self._force_2d and len(new_shape) != 1:
            raise ValueError(
                f"Cannot reshape property '{key}' to {(self._N, *new_shape)} while "
                f"force_2d is enabled."
            )

        self._properties[key] = self._properties[key].reshape(self._N, *new_shape)

    # ==========================
    # Structure methods
    # ==========================
    def copy(self) -> PropertyContainer:
        r"""
        Create a copy of the container (the arrays are copied).

        Returns
        -------
        :class:`pysdic.core.property_container.PropertyContainer`
            A new container with the same :math:`N`, properties and ``force_2d`` flag.

        """
        # The constructor copies the arrays
        return self.__class__(
            N=self._N,
            properties=self._properties,
            force_2d=self._force_2d,
        )

    def concatenate(
        self, other: PropertyContainer, *, fill: Real = numpy.nan
    ) -> PropertyContainer:
        r"""
        Concatenate two containers along the first dimension.

        The resulting container has :math:`N_{self} + N_{other}` entities:

        - a property present in both containers is concatenated (the attribute shapes
          must match),
        - a property present in only one container is completed with ``fill`` for the
          entities of the other container.

        The data type of each property is promoted if needed to hold both the values
        and ``fill``. For example, an integer property completed with ``numpy.nan``
        becomes a float property.

        The ``force_2d`` flag of the result is True only if it is True for both
        containers.

        Parameters
        ----------
        other : :class:`pysdic.core.property_container.PropertyContainer`
            The container to append after this one.

        fill : Real, optional (default: ``numpy.nan``)
            The value used to complete the properties that exist in only one container.

        Returns
        -------
        :class:`pysdic.core.property_container.PropertyContainer`
            A new container with the concatenated properties.

        Raises
        ------
        ValueError
            If a property present in both containers has different attribute shapes.

        """
        if not isinstance(other, PropertyContainer):
            raise TypeError(
                f"Input 'other' must be a PropertyContainer. Got type {type(other)}."
            )
        if numpy.ndim(fill) != 0:
            raise TypeError(f"Input 'fill' must be a scalar. Got {fill!r}.")
        fill = numpy.asarray(fill)

        new_properties = {}
        for key in list(self.keys()) + [k for k in other.keys() if k not in self]:
            a = self._properties.get(key)
            b = other._properties.get(key)
            if a is not None and b is not None and a.shape[1:] != b.shape[1:]:
                raise ValueError(
                    f"Cannot concatenate property '{key}': attribute shapes differ "
                    f"({a.shape[1:]} and {b.shape[1:]})."
                )

            if a is None:
                a = numpy.full((self._N,) + b.shape[1:], fill, dtype=numpy.result_type(b, fill))
            elif b is None:
                b = numpy.full((other._N,) + a.shape[1:], fill, dtype=numpy.result_type(a, fill))

            dtype = numpy.result_type(a, b)
            new_properties[key] = numpy.concatenate(
                [a.astype(dtype, copy=False), b.astype(dtype, copy=False)], axis=0
            )

        return self.__class__(
            N=self._N + other._N,
            properties=new_properties,
            force_2d=self._force_2d and other._force_2d,
        )

    def update(
        self,
        other: PropertyContainer,
        *,
        conflict: bool = True,
    ) -> PropertyContainer:
        r"""
        Combine the properties of two containers with the same :math:`N`.

        .. note::

            Unlike ``dict.update``, this container is not modified: a new container is
            returned.

        The ``force_2d`` flag of the result is the one of this container.

        Parameters
        ----------
        other : :class:`pysdic.core.property_container.PropertyContainer`
            The container whose properties are added.

        conflict : bool, optional (default: ``True``)
            If True, a ``ValueError`` is raised when a property exists in both
            containers. If False, the values of ``other`` overwrite the values of this
            container.

        Returns
        -------
        :class:`pysdic.core.property_container.PropertyContainer`
            A new container with the properties of both containers.

        """
        if not isinstance(other, PropertyContainer):
            raise TypeError(
                f"Input 'other' must be a PropertyContainer. Got type {type(other)}."
            )
        conflict = _validate_bool(conflict, "conflict")
        if self._N != other._N:
            raise ValueError(
                f"Cannot merge PropertyContainers with different N values. "
                f"Current container has N={self._N}, other container has N={other._N}."
            )

        if conflict:
            conflicting_keys = set(self.keys()) & set(other.keys())
            if conflicting_keys:
                raise ValueError(
                    f"Cannot merge PropertyContainers because of conflicting keys: "
                    f"{conflicting_keys}. Set conflict=False to overwrite the values with "
                    f"the ones of the other container."
                )

        new_container = self.copy()
        for key in other.keys():
            new_container.set(key, other._properties[key])
        return new_container

    def filter(self, mask: ArrayLike) -> PropertyContainer:
        r"""
        Keep only the entities selected by a boolean mask, in a new container (this
        container is unchanged).

        The ``force_2d`` flag of the result is the one of this container.

        Parameters
        ----------
        mask : ArrayLike
            A boolean array of shape :math:`(N,)`. The entities with a True value are
            kept. A mask with only False values gives an empty container
            (:math:`N = 0`).

        Returns
        -------
        :class:`pysdic.core.property_container.PropertyContainer`
            The filtered container.

        Raises
        ------
        TypeError
            If the mask is not a boolean array (an integer array is rejected).

        ValueError
            If the mask does not have the shape :math:`(N,)`.

        """
        mask = _as_bool_mask(mask, self._N)

        # Advanced indexing copies the arrays
        new_properties = {key: array[mask] for key, array in self._properties.items()}
        new_N = int(numpy.count_nonzero(mask))

        return self.__class__(
            N=new_N,
            properties=new_properties,
            force_2d=self._force_2d,
        )

    # ==========================
    # Conversion
    # ==========================
    def to_dict(self) -> Dict[str, numpy.ndarray]:
        r"""
        Convert the container to a dictionary.

        Returns
        -------
        Dict[str, numpy.ndarray]
            A dictionary mapping the property names to copies of the property arrays.

        """
        return {key: array.copy() for key, array in self._properties.items()}

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, ArrayLike],
        *,
        N: Optional[Integral] = None,
        force_2d: bool = False,
    ) -> PropertyContainer:
        r"""
        Create a container from a dictionary of properties.

        Parameters
        ----------
        data : Dict[str, ArrayLike]
            A dictionary mapping the property names to arrays of shape
            :math:`(N, \dots)`.

        N : Integral, optional
            The number of entities. By default, it is inferred from the first property
            of ``data``, which is then required to be non-empty.

        force_2d : bool, optional (default: ``False``)
            If True, all properties are stored as 2D arrays of shape :math:`(N, P)`.

        Returns
        -------
        :class:`pysdic.core.property_container.PropertyContainer`
            A new container with the properties of ``data``.

        """
        if not isinstance(data, Mapping):
            raise TypeError(f"Input 'data' must be a dictionary. Got type {type(data)}.")
        if len(data) == 0 and N is None:
            raise ValueError("Input 'data' cannot be empty if N is not provided.")

        if N is None:
            first_key = next(iter(data))
            first_data = numpy.asarray(data[first_key])
            if first_data.ndim == 0:
                raise ValueError(
                    f"Invalid shape for property '{first_key}'. Expected an array with at "
                    f"least one dimension, but got a scalar."
                )
            N = first_data.shape[0]

        return cls(N=N, properties=data, force_2d=force_2d)