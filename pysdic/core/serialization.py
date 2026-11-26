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
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List

import json
import os
import numpy

from .._optionals import import_optional
from .._validation import _validate_bool, _validate_str, _validate_path

if TYPE_CHECKING:
    import h5py


def _make_parent_dirs(file_path: Any) -> None:
    r"""
    Create the parent directories of ``file_path`` if they do not exist.

    ``os.path.dirname`` returns an empty string for a bare file name (e.g. ``"data.json"``),
    and ``os.makedirs("")`` raises a ``FileNotFoundError``, so the empty case is skipped.
    """
    directory = os.path.dirname(os.fspath(file_path))
    if directory:
        os.makedirs(directory, exist_ok=True)


class SerializerMixin(ABC):
    r"""
    Base mixin class for serialization and deserialization capabilities.
    This class serves as a common interface for different serialization formats (e.g., JSON, HDF5, NPZ, ...).
    It defines the abstract methods that subclasses must implement to provide specific serialization logic.

    Each subclass should implement the `to_dict` and `from_dict` methods to define
    how the object is represented as a dictionary and how to create an instance from a
    dictionary, respectively.

    .. important::

        The `to_dict` method should return a dictionary representation of the object,
        while the `from_dict` method should take a dictionary as input and return an instance of the class.
        These methods are essential for enabling the serialization and deserialization processes in the subclasses.

    .. note::

        The helper methods of each mixin have format-specific names (``_json_*``, ``_hdf5_*``,
        ``_npz_*``) so that a class can inherit from several mixins without one helper
        shadowing another in the MRO.

    """

    __slots__ = ()

    @abstractmethod
    def to_dict(self, **kwargs) -> Dict[str, Any]:
        r"""
        Convert the object to a dictionary representation.

        .. important::

            This method should be implemented by subclasses to
            define how the object is represented as a dictionary.

        """
        raise NotImplementedError("Subclasses must implement the to_dict method.")

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any], **kwargs) -> Any:
        r"""
        Create an instance of the class from a dictionary representation.

        .. important::

            This method should be implemented by subclasses to define how to
            create an instance from a dictionary.

        """
        raise NotImplementedError("Subclasses must implement the from_dict method.")


class JsonSerializerMixin(SerializerMixin):
    r"""
    Mixin class to provide JSON serialization and deserialization capabilities.
    Classes that inherit from this mixin can be easily converted to and from JSON format.

    .. important::

        Numpy arrays, numpy scalars and sets are not directly serializable to JSON.
        They are converted to lists and Python scalars when serializing.

        When deserializing from JSON using the ``from_dict`` method, ensure that you
        convert lists back to numpy arrays if needed.

    """

    __slots__ = ()

    @staticmethod
    def _json_default(obj: Any) -> Any:
        r"""
        Fallback used by ``json.dumps`` (``default=``) for objects that are not natively
        JSON serializable.

        ``json`` already handles ``dict``, ``list``, ``tuple``, ``str``, ``int``, ``float``,
        ``bool`` and ``None`` (including subclasses such as ``numpy.float64``), so this
        method is only called for the remaining types. The returned value is then encoded
        recursively by ``json``.

        Raises
        ------
        TypeError
            If the object type is not supported.
        """
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        if isinstance(obj, numpy.generic):
            return obj.item()  # numpy.int64, numpy.bool_, numpy.float32, ...
        if isinstance(obj, (set, frozenset)):
            return list(obj)
        raise TypeError(
            f"Object of type {type(obj).__name__} is not JSON serializable: {obj!r}"
        )

    def to_json_string(self, **kwargs) -> str:
        r"""
        Serialize the object to a JSON string.

        This method uses the ``to_dict`` method to convert the object to a dictionary,
        and then converts that dictionary to a JSON string.

        Returns
        -------
        str
            A JSON string representation of the object, with indentation for readability.

        Raises
        ------
        ValueError
            If the dictionary returned by ``to_dict`` cannot be encoded to JSON.

        """
        data = self.to_dict(**kwargs)  # Errors of to_dict are not relabeled
        try:
            return json.dumps(data, indent=4, default=self._json_default)
        except (TypeError, ValueError) as e:  # ValueError: circular reference
            raise ValueError(f"Failed to serialize object to JSON: {e}") from e

    @classmethod
    def from_json_string(cls, json_string: str, **kwargs) -> Any:
        r"""
        Deserialize a JSON string to an object of the class.

        This method parses the input JSON string, converts it to a dictionary, and then
        uses the ``from_dict`` method to create an instance of the class.

        Parameters
        ----------
        json_string : str
            A JSON string representation of the object.

        Returns
        -------
        Any
            An instance of the class, populated with the data from the JSON string.

        Raises
        ------
        ValueError
            If the input JSON string is not valid or cannot be parsed.
        """
        try:
            data = json.loads(json_string)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON string: {e}") from e
        return cls.from_dict(data, **kwargs)  # Errors of from_dict are not relabeled

    def to_json_file(
        self, file_path: Any, *, make_dirs: bool = False, **kwargs
    ) -> None:
        r"""
        Serialize the object to a JSON file.

        Parameters
        ----------
        file_path : Any
            The path to the file where the JSON representation of the class
            will be saved (str, Path, ...).

        make_dirs : bool, optional (default: ``False``)
            If True, the method will create any missing directories
            in the file path before writing the JSON.

        """
        file_path = _validate_path(file_path, "file_path")
        make_dirs = _validate_bool(make_dirs, "make_dirs")

        json_string = self.to_json_string(**kwargs)  # Fails before touching the file

        if make_dirs:
            _make_parent_dirs(file_path)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(json_string)

    @classmethod
    def from_json_file(cls, file_path: Any, **kwargs) -> Any:
        r"""
        Deserialize a JSON file to an object of the class.

        Parameters
        ----------
        file_path : Any
            The path to the JSON file containing the serialized class data (str, Path, ...).

        """
        file_path = _validate_path(file_path, "file_path", must_exist=True)

        with open(file_path, "r", encoding="utf-8") as f:
            json_str = f.read()

        return cls.from_json_string(json_str, **kwargs)


class Hdf5SerializerMixin(SerializerMixin):
    r"""
    Mixin class to provide HDF5 serialization and deserialization capabilities.
    Classes that inherit from this mixin can be easily converted to and from HDF5 format.

    The dictionary returned by ``to_dict`` is mapped to HDF5 as follows:

    - ``dict`` -> group (written recursively).
    - non-empty ``list`` / ``tuple`` of ``dict`` -> group with subgroups ``"0"``, ``"1"``, ...
      (read back as a ``list`` of ``dict``).
    - ``numpy.ndarray``, ``list``, ``tuple``, ``set`` -> dataset (read back as a ``numpy.ndarray``).
      Arrays of strings are stored as UTF-8 strings and read back as arrays of ``str``.
    - ``int``, ``float``, ``bool``, ``str``, ``bytes`` and numpy scalars -> attribute
      (read back as Python scalars).
    - ``None`` -> attribute with a sentinel value (read back as ``None``).

    .. important::

        Lists, tuples and sets are read back as numpy arrays. When deserializing using the
        ``from_dict`` method, convert them back to lists or tuples if needed.

    .. important::

        Keys must be non-empty strings without ``"/"`` (``"/"`` is the HDF5 path separator).

    """

    __slots__ = ()

    _HDF5_NONE: str = "__None__"
    _HDF5_LIST_OF_DICTS: str = "__list_of_dicts__"

    @classmethod
    def _hdf5_check_key(cls, key: Any) -> None:
        key = _validate_str(key, "key")  # TypeError if not a string, ValueError if empty
        if key == "." or "/" in key:
            raise ValueError(
                f"Invalid HDF5 key {key!r}: keys must be non-empty and must not contain '/'."
            )
        if key == cls._HDF5_LIST_OF_DICTS:
            raise ValueError(f"The key {key!r} is reserved by the HDF5 serializer.")

    @staticmethod
    def _hdf5_to_array(key: str, value: Any) -> numpy.ndarray:
        if isinstance(value, (set, frozenset)):
            value = list(value)
        try:
            array = numpy.asarray(value)
        except ValueError as e:  # Ragged nested sequences
            raise TypeError(
                f"Value for key '{key}' cannot be converted to a regular array: {e}"
            ) from e
        if array.dtype.kind == "O":
            raise TypeError(
                f"Unsupported object array for key '{key}' (mixed or non-numeric elements)."
            )
        return array

    @classmethod
    def _hdf5_write(cls, h5group: h5py.Group, data: Dict[str, Any]) -> None:
        r"""
        Write a dictionary of data to an HDF5 group (see the class docstring for the mapping).

        Parameters
        ----------
        h5group : h5py.Group
            The HDF5 group where the data will be written.

        data : Dict[str, Any]
            The dictionary to write.

        """
        h5py = import_optional("h5py", "Hdf5SerializerMixin._hdf5_write")

        if not isinstance(data, dict):
            raise TypeError(f"Data to write must be a dictionary, got {type(data)}.")

        for key, value in data.items():
            cls._hdf5_check_key(key)

            if isinstance(value, dict):
                cls._hdf5_write(h5group.create_group(key), value)

            elif (
                isinstance(value, (list, tuple))
                and len(value) > 0
                and all(isinstance(item, dict) for item in value)
            ):
                subgroup = h5group.create_group(key)
                subgroup.attrs[cls._HDF5_LIST_OF_DICTS] = True
                for index, item in enumerate(value):
                    cls._hdf5_write(subgroup.create_group(str(index)), item)

            elif isinstance(value, (numpy.ndarray, list, tuple, set, frozenset)):
                array = cls._hdf5_to_array(key, value)
                if array.dtype.kind == "U":
                    # h5py cannot store numpy unicode arrays: use variable-length UTF-8
                    h5group.create_dataset(
                        key, data=array.astype(object), dtype=h5py.string_dtype()
                    )
                else:
                    h5group.create_dataset(key, data=array)

            elif isinstance(value, (bool, int, float, str, bytes, numpy.generic)):
                h5group.attrs[key] = value

            elif value is None:
                h5group.attrs[key] = cls._HDF5_NONE

            else:
                raise TypeError(f"Unsupported data type for key '{key}': {type(value)}")

    @classmethod
    def _hdf5_read(cls, h5group: h5py.Group) -> Any:
        r"""
        Read an HDF5 group written by ``_hdf5_write``.

        Parameters
        ----------
        h5group : h5py.Group
            The HDF5 group to read.

        Returns
        -------
        Dict[str, Any] or List[Dict[str, Any]]
            A dictionary, or a list of dictionaries if the group stores a list of dictionaries.

        """
        h5py = import_optional("h5py", "Hdf5SerializerMixin._hdf5_read")

        if h5group.attrs.get(cls._HDF5_LIST_OF_DICTS, False):
            return [cls._hdf5_read(h5group[str(index)]) for index in range(len(h5group))]

        result: Dict[str, Any] = {}

        # Groups and datasets
        for key, item in h5group.items():
            if isinstance(item, h5py.Group):
                result[key] = cls._hdf5_read(item)

            elif isinstance(item, h5py.Dataset):
                string_info = h5py.check_string_dtype(item.dtype)
                if string_info is not None and string_info.encoding == "utf-8":
                    result[key] = item.asstr()[()]  # str or array of str (not bytes)
                else:
                    result[key] = item[()]

            else:
                raise TypeError(
                    f"Unsupported HDF5 item type for key '{key}': {type(item)}"
                )

        # Attributes
        for key, value in h5group.attrs.items():
            if key == cls._HDF5_LIST_OF_DICTS:
                continue
            if isinstance(value, str) and value == cls._HDF5_NONE:
                result[key] = None
            elif isinstance(value, numpy.generic):
                result[key] = value.item()  # numpy.int64 -> int, numpy.bool_ -> bool, ...
            else:
                result[key] = value

        return result

    def to_hdf5_file(
        self, file_path: Any, *, make_dirs: bool = False, **kwargs
    ) -> None:
        r"""
        Serialize the object to an HDF5 file.

        Parameters
        ----------
        file_path : Any
            The path to the file where the HDF5 representation of the class
            will be saved (str, Path, ...).

        make_dirs : bool, optional (default: ``False``)
            If True, the method will create any missing directories
            in the file path before writing the HDF5.

        """
        h5py = import_optional("h5py", "Hdf5SerializerMixin.to_hdf5_file")
        file_path = _validate_path(file_path, "file_path")
        make_dirs = _validate_bool(make_dirs, "make_dirs")

        data = self.to_dict(**kwargs)  # Fails before truncating an existing file

        if make_dirs:
            _make_parent_dirs(file_path)

        with h5py.File(file_path, "w") as h5file:
            self._hdf5_write(h5file, data)

    @classmethod
    def from_hdf5_file(cls, file_path: Any, **kwargs) -> Any:
        r"""
        Deserialize an object from an HDF5 file.

        Parameters
        ----------
        file_path : Any
            The path to the HDF5 file to be read (str, Path, ...).

        Returns
        -------
        Any
            An instance of the class, populated with the data read from the HDF5 file.

        """
        h5py = import_optional("h5py", "Hdf5SerializerMixin.from_hdf5_file")
        file_path = _validate_path(file_path, "file_path", must_exist=True)

        with h5py.File(file_path, "r") as h5file:
            data = cls._hdf5_read(h5file)
        return cls.from_dict(data, **kwargs)


class NpzSerializerMixin(SerializerMixin):
    r"""
    Mixin class to provide NPZ serialization and deserialization capabilities.
    Classes that inherit from this mixin can be easily converted to and from NPZ format.

    The file is written and read **without pickle** (``allow_pickle=False``): only
    numeric, boolean and string data are supported.

    - ``numpy.ndarray``, ``list``, ``tuple``, ``set`` -> array (read back as a ``numpy.ndarray``).
    - ``int``, ``float``, ``bool``, ``str``, ``bytes`` and numpy scalars -> 0-d array
      (read back as Python scalars).
    - ``None`` -> listed in a reserved entry (read back as ``None``).

    .. warning::

        The dictionary returned by the ``to_dict`` method must have only
        one level of nesting, as NPZ format does not support nested structures.

    .. warning::

        Files written by previous versions of this mixin stored scalars as pickled
        object arrays and cannot be read with ``allow_pickle=False``.

    """

    __slots__ = ()

    _NPZ_NONE_KEYS: str = "__none_keys__"
    # Names that collide with the parameters of numpy.savez(file, *args, allow_pickle=..., **kwds)
    _NPZ_FORBIDDEN_KEYS = frozenset({"file", "allow_pickle"})

    @classmethod
    def _npz_serialize(cls, data: Dict[str, Any]) -> Dict[str, numpy.ndarray]:
        r"""
        Convert a flat dictionary to a dictionary of numpy arrays ready for ``numpy.savez``.

        Parameters
        ----------
        data : Dict[str, Any]
            A flat dictionary of data to serialize.

        Returns
        -------
        Dict[str, numpy.ndarray]
            The arrays to save, without any object (pickled) array.
        """
        if not isinstance(data, dict):
            raise TypeError(f"Data to serialize must be a dictionary, got {type(data)}.")

        arrays: Dict[str, numpy.ndarray] = {}
        none_keys: List[str] = []

        for key, value in data.items():
            key = _validate_str(key, "key")  # TypeError if not a string, ValueError if empty
            if key == cls._NPZ_NONE_KEYS or key in cls._NPZ_FORBIDDEN_KEYS:
                raise ValueError(f"The key {key!r} is reserved by the NPZ serializer.")

            if value is None:
                none_keys.append(key)
                continue

            if isinstance(value, dict):
                raise ValueError(
                    f"Nested dictionaries are not supported in NPZ serialization for key '{key}'."
                )

            if isinstance(value, (set, frozenset)):
                value = list(value)

            if not isinstance(
                value,
                (numpy.ndarray, list, tuple, bool, int, float, str, bytes, numpy.generic),
            ):
                raise TypeError(f"Unsupported data type for key '{key}': {type(value)}")

            try:
                array = numpy.asarray(value)
            except ValueError as e:  # Ragged nested sequences
                raise TypeError(
                    f"Value for key '{key}' cannot be converted to a regular array: {e}"
                ) from e
            if array.dtype.kind == "O":
                raise TypeError(
                    f"Unsupported object array for key '{key}' (would require pickle)."
                )
            arrays[key] = array

        if none_keys:
            arrays[cls._NPZ_NONE_KEYS] = numpy.array(none_keys, dtype=str)

        return arrays

    @classmethod
    def _npz_deserialize(cls, npz: Any) -> Dict[str, Any]:
        r"""
        Convert an opened ``numpy.lib.npyio.NpzFile`` to a dictionary.

        0-d arrays are converted back to Python scalars, and the keys listed in the
        reserved entry are set to ``None``.
        """
        result: Dict[str, Any] = {}
        for key in npz.files:
            if key == cls._NPZ_NONE_KEYS:
                continue
            array = npz[key]
            result[key] = array.item() if array.ndim == 0 else array

        if cls._NPZ_NONE_KEYS in npz.files:
            for key in npz[cls._NPZ_NONE_KEYS].tolist():
                result[key] = None

        return result

    def to_npz_file(self, file_path: Any, *, make_dirs: bool = False, **kwargs) -> None:
        r"""
        Serialize the object to an NPZ file.

        .. note::

            The file is written at exactly ``file_path``: no ``.npz`` extension is appended
            (unlike ``numpy.savez`` called with a path).

        Parameters
        ----------
        file_path : Any
            The path to the file where the NPZ representation of the class
            will be saved (str, Path, ...).

        make_dirs : bool, optional (default: ``False``)
            If True, the method will create any missing directories
            in the file path before writing the NPZ.

        """
        file_path = _validate_path(file_path, "file_path")
        make_dirs = _validate_bool(make_dirs, "make_dirs")

        arrays = self._npz_serialize(self.to_dict(**kwargs))

        if make_dirs:
            _make_parent_dirs(file_path)

        # Opening the file ourselves prevents numpy from appending '.npz' to the path
        with open(file_path, "wb") as f:
            numpy.savez(f, **arrays)

    @classmethod
    def from_npz_file(cls, file_path: Any, **kwargs) -> Any:
        r"""
        Deserialize an object from an NPZ file.

        Parameters
        ----------
        file_path : Any
            The path to the NPZ file to be read (str, Path, ...).

        Returns
        -------
        Any
            An instance of the class, populated with the data read from the NPZ file.

        """
        file_path = _validate_path(file_path, "file_path", must_exist=True)

        with numpy.load(file_path, allow_pickle=False) as npz:
            data = cls._npz_deserialize(npz)
        return cls.from_dict(data, **kwargs)