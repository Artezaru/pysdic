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
from typing import TYPE_CHECKING, Optional, Tuple, Union, Dict, Any
from numbers import Integral, Real
import numpy
from numpy.typing import ArrayLike
import os

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
    _validate_tolerances,
    _validate_path,
    _validate_choice,
    _as_array,
    _as_scalar_or_1d_array,
    _as_bool_mask,
    _as_index_array,
)
from .._optionals import import_optional
from ._vtk_properties import (
    _check_vtk_property_shapes,
    _dict_to_vtk_attributes,
    _vtk_attributes_to_dict,
)

if TYPE_CHECKING:
    import matplotlib.colors
    import meshio
    import py3dframe
    import pyvista
    from vtkmodules.vtkCommonDataModel import vtkPointSet, vtkPolyData


class PointCloud(LockableMixin, JsonSerializerMixin, NpzSerializerMixin, Hdf5SerializerMixin):
    r"""
    A cloud of :math:`N_p` points in a :math:`E`-dimensional space.

    Each point can carry additional properties (colors, normals, displacements, ...)
    stored in a :class:`pysdic.core.property_container.PropertyContainer`, accessible through the
    ``properties`` attribute or directly with ``point_cloud[name]``.

    .. note::

        The coordinates are copied when they are set and stored as a ``numpy.float64``
        array. The ``coordinates`` attribute returns a read-only view: to change the
        coordinates, assign a new array.

    .. important::

        The properties are stored in a :class:`pysdic.core.property_container.PropertyContainer` with
        ``force_2d=True``, i.e. with shape :math:`(N_p, P)`. A container given by the
        user with other settings is accepted, but some operations (like VTK export) may
        behave unexpectedly.

    Parameters
    ----------
    coordinates: ArrayLike
        An array of shape :math:`(N_p, E)` with the coordinates of the points.
        A 1D array of shape :math:`(N_p,)` is accepted if ``n_dimensions=1``.

    n_dimensions: Optional[Integral], optional (default: ``None``)
        The dimension :math:`E` of the space. By default, it is inferred from the
        shape of ``coordinates``. If given, it must match this shape.

    properties: Optional[Union[Dict[str, ArrayLike], PropertyContainer]], optional
        The properties of the points:

        - a dictionary mapping names to arrays of shape :math:`(N_p, \dots)`, which
          are copied into a new :class:`pysdic.core.property_container.PropertyContainer`,
        - or a :class:`pysdic.core.property_container.PropertyContainer` with :math:`N = N_p`, which is
          **shared** (not copied).


    Notes
    -----
    ``len(point_cloud)`` gives the number of points :math:`N_p`, while
    ``point_cloud[name]``, ``name in point_cloud`` and ``iter(point_cloud)`` refer to
    the properties.

    The methods never modify the point cloud in place: the methods that change the
    points (:meth:`filter_points`, :meth:`frame_transform`, :meth:`remove_dimension`,
    ...) return a new point cloud. Only the ``coordinates`` setter (same number of
    points) and the properties modify an existing point cloud.

    .. note::

        A point cloud can be locked by an object of the package that owns it (for example the
        mesh of an SDIC analysis, see :class:`pysdic.core.lockable.LockableMixin`): the
        ``coordinates`` setter then raises a :class:`RuntimeError` explaining why. The
        properties can always be modified. A copy (``copy``) is always unlocked.

    """

    __slots__ = [
        "_coordinates",
        "_properties",
    ]

    def __init__(
        self,
        coordinates: ArrayLike,
        *,
        n_dimensions: Optional[Integral] = None,
        properties: Optional[Union[Dict[str, ArrayLike], PropertyContainer]] = None,
    ) -> None:
        n_dimensions = _validate_integer(
            n_dimensions, "n_dimensions", min_value=1, allow_none=True
        )
        coordinates = self._as_coordinates(coordinates, n_dimensions)
        N = coordinates.shape[0]

        if properties is None:
            props = PropertyContainer(N=N, force_2d=True)
        elif isinstance(properties, PropertyContainer):
            if properties.N != N:
                raise ValueError(
                    f"The number of entities in 'properties' must match the number of "
                    f"points. Got N={properties.N} for properties and N_p={N} for points."
                )
            props = properties
        elif isinstance(properties, Mapping):
            props = PropertyContainer(N=N, properties=properties, force_2d=True)
        else:
            raise TypeError(
                f"Input 'properties' must be None, a dictionary of arrays or a "
                f"PropertyContainer. Got type {type(properties)}."
            )

        self._coordinates = coordinates
        self._properties = props
        self.validate(recursive=True)

    @staticmethod
    def _as_coordinates(value: ArrayLike, n_dimensions: Optional[int]) -> numpy.ndarray:
        # Copy the input to a (N_p, E) float64 array. 1D input is only accepted for E = 1.
        coordinates = _as_array(
            value, "coordinates", numpy.float64,
            shape=(None, n_dimensions),
            shape_hint="(N_p, E)" if n_dimensions is None else f"(N_p, {n_dimensions})",
            finite=False, accept_1d_as_column=(n_dimensions == 1), copy=True,
        )
        if coordinates.shape[1] == 0:
            raise ValueError("Input 'coordinates' must have at least one dimension (E >= 1).")
        return coordinates

    def validate(self, *, recursive: bool = True) -> None:
        r"""
        Validate the integrity of the point cloud.

        The following conditions are checked:

        - The coordinates are a ``numpy.float64`` array of shape :math:`(N_p, E)` with
          :math:`E \geq 1`.
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
            If an attribute has a wrong shape.

        """
        recursive = _validate_bool(recursive, "recursive")

        # --- coordinates must be a 2D float64 array of shape (Np, E) ---
        if not isinstance(self._coordinates, numpy.ndarray):
            raise TypeError("[PointCloud validate] coordinates should be a numpy ndarray.")
        if self._coordinates.ndim != 2 or self._coordinates.shape[1] == 0:
            raise ValueError(
                f"[PointCloud validate] coordinates should be a 2D array of shape (Np, E) "
                f"with E >= 1. Got shape {self._coordinates.shape}."
            )
        if self._coordinates.dtype != numpy.float64:
            raise TypeError(
                f"[PointCloud validate] coordinates should have dtype float64. "
                f"Got {self._coordinates.dtype}."
            )

        # --- properties must be a PropertyContainer with the same number of points ---
        if not isinstance(self._properties, PropertyContainer):
            raise TypeError("[PointCloud validate] properties should be a PropertyContainer.")
        if self._properties.N != self.n_points:
            raise ValueError(
                f"[PointCloud validate] The number of entities in properties should match "
                f"the number of points. Got N={self._properties.N} for properties and "
                f"N_p={self.n_points} for points."
            )

        if recursive:
            self._properties.validate()

    # ==========================
    # API interface
    # ==========================
    @property
    def coordinates(self) -> numpy.ndarray:
        r"""
        [Get/Set] The coordinates of the points, an array of shape :math:`(N_p, E)`.

        The getter returns a read-only view. The setter copies the given array, whose
        shape must be :math:`(N_p, E)`: to change the number of points, use
        :meth:`filter_points` or :meth:`concatenate` (which return a new point cloud).
        The setter is refused if the point cloud is locked.

        Parameters
        ----------
        value: ArrayLike
            An array of shape :math:`(N_p, E)`. A 1D array of shape :math:`(N_p,)` is
            accepted if :math:`E = 1`.

        Returns
        -------
        numpy.ndarray
            A read-only view of shape :math:`(N_p, E)` on the coordinates.

        Raises
        ------
        ValueError
            If the shape of the input is not :math:`(N_p, E)`.

        RuntimeError
            If the point cloud is locked (see ``locked_message``).

        """
        view = self._coordinates.view()
        view.flags.writeable = False
        return view

    @coordinates.setter
    def coordinates(self, value: ArrayLike) -> None:
        self._check_unlocked("set the coordinates")
        coordinates = self._as_coordinates(value, self.n_dimensions)
        if coordinates.shape[0] != self.n_points:
            raise ValueError(
                f"Input 'coordinates' must keep the number of points N_p={self.n_points}. "
                f"Got {coordinates.shape[0]} points. Consider using the filter/concatenate "
                f"methods to change the number of points."
            )
        self._coordinates = coordinates

    @property
    def n_points(self) -> int:
        r"""
        [Get] The number of points :math:`N_p`.

        Returns
        -------
        int
            The number of points.

        """
        return self._coordinates.shape[0]

    def __len__(self) -> int:
        r"""Return the number of points :math:`N_p`."""
        return self.n_points

    @property
    def n_dimensions(self) -> int:
        r"""
        [Get] The dimension :math:`E` of the space.

        Returns
        -------
        int
            The dimension of the space.

        """
        return self._coordinates.shape[1]

    @property
    def shape(self) -> Tuple[int, int]:
        r"""
        [Get] The shape :math:`(N_p, E)` of the coordinates.

        Returns
        -------
        Tuple[int, int]
            The number of points and the dimension of the space.

        """
        return (self.n_points, self.n_dimensions)

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
        r"""Shortcut for ``point_cloud.properties[key]``."""
        return self._properties[key]

    def __setitem__(self, key, value):
        r"""Shortcut for ``point_cloud.properties[key] = value``."""
        self._properties[key] = value

    def __delitem__(self, key):
        r"""Shortcut for ``del point_cloud.properties[key]``."""
        del self._properties[key]

    def __contains__(self, key):
        r"""Shortcut for ``key in point_cloud.properties``."""
        return key in self._properties

    def __iter__(self):
        r"""Shortcut for ``iter(point_cloud.properties)`` (iterates over the names)."""
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
        string = f"PointCloud(n_points={self.n_points}, n_dimensions={self.n_dimensions})"
        for key in self._properties.keys():
            string += f"\n  - Property: '{key}' with shape {self._properties.get_shape(key)}"
        return string

    # ==========================
    # I/O Methods
    # ==========================
    @classmethod
    def from_array(
        cls,
        coordinates: ArrayLike,
        *,
        properties: Optional[Union[Dict[str, ArrayLike], PropertyContainer]] = None,
    ) -> PointCloud:
        r"""
        Create a point cloud from an array of coordinates.

        Equivalent to ``PointCloud(coordinates, properties=properties)``.

        Parameters
        ----------
        coordinates: ArrayLike
            An array of shape :math:`(N_p, E)` with the coordinates of the points.

        properties: Optional[Union[Dict[str, ArrayLike], PropertyContainer]], optional
            The properties of the points (see :class:`pysdic.PointCloud`).

        Returns
        -------
        :class:`pysdic.PointCloud`
            The new point cloud.

        """
        return cls(coordinates=coordinates, properties=properties)

    def to_array(self) -> numpy.ndarray:
        r"""
        Return a copy of the coordinates.

        Returns
        -------
        numpy.ndarray
            A writable array of shape :math:`(N_p, E)`.

        """
        return self._coordinates.copy()

    @classmethod
    def from_dict(cls, data: Dict[str, Any], *, load_properties: bool = True) -> PointCloud:
        r"""
        Create a point cloud from a dictionary.

        The dictionary must contain the key ``"coordinates"`` with an array of shape
        :math:`(N_p, E)`. The properties are read from the keys
        ``"_property/{property_name}"``.

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
        :class:`pysdic.PointCloud`
            The new point cloud.

        """
        if not isinstance(data, Mapping):
            raise TypeError(f"Input 'data' must be a dictionary. Got type {type(data)}.")
        if "coordinates" not in data:
            raise ValueError(
                "Input dictionary must contain a key 'coordinates' with shape (N_p, E)."
            )
        load_properties = _validate_bool(load_properties, "load_properties")

        properties = {}
        if load_properties:
            prefix = "_property/"
            for key in data:
                if isinstance(key, str) and key.startswith(prefix):
                    properties[key[len(prefix):]] = data[key]

        return cls(coordinates=data["coordinates"], properties=properties)

    def to_dict(self, *, save_properties: bool = True) -> Dict[str, Any]:
        r"""
        Convert the point cloud to a flat dictionary.

        The coordinates are stored under the key ``"coordinates"`` and the properties
        under the keys ``"_property/{property_name}"``. All the arrays are copies.

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
            The dictionary representation of the point cloud.

        """
        save_properties = _validate_bool(save_properties, "save_properties")

        data = {"coordinates": self._coordinates.copy()}
        if save_properties:
            for key, values in self._properties.to_dict().items():
                data[f"_property/{key}"] = values
        return data

    @classmethod
    def from_meshio(cls, mesh: meshio.Mesh, *, load_properties: bool = True) -> PointCloud:
        r"""
        Create a point cloud from the vertices of a :class:`meshio.Mesh`.

        The cells are ignored and the properties are read from the point data (their
        data types are kept).

        Parameters
        ----------
        mesh: :class:`meshio.Mesh`
            The mesh to read.

        load_properties: bool, optional (default: ``True``)
            If False, the point data are ignored.

        Returns
        -------
        :class:`pysdic.PointCloud`
            The new point cloud.

        """
        meshio = import_optional("meshio", "PointCloud.from_meshio")

        if not isinstance(mesh, meshio.Mesh):
            raise TypeError(f"Input 'mesh' must be a meshio.Mesh. Got type {type(mesh)}.")
        load_properties = _validate_bool(load_properties, "load_properties")
        if mesh.points is None:
            raise ValueError("The mesh does not contain any points.")

        properties = {}
        if load_properties and mesh.point_data is not None:
            properties = dict(mesh.point_data)  # copied by the PropertyContainer

        return cls(coordinates=mesh.points, properties=properties)

    def to_meshio(self, *, save_properties: bool = True) -> meshio.Mesh:
        r"""
        Convert the point cloud to a :class:`meshio.Mesh` without cells.

        The points are the vertices of the mesh and the properties are saved as point
        data.

        Parameters
        ----------
        save_properties: bool, optional (default: ``True``)
            If False, the properties are not saved.

        Returns
        -------
        :class:`meshio.Mesh`
            The mesh representation of the point cloud.

        """
        meshio = import_optional("meshio", "PointCloud.to_meshio")

        save_properties = _validate_bool(save_properties, "save_properties")

        point_data = {}
        if save_properties:
            point_data = self._properties.to_dict()

        return meshio.Mesh(points=self._coordinates.copy(), cells=[], point_data=point_data)

    @classmethod
    def from_pyvista(cls, mesh: pyvista.DataSet, *, load_properties: bool = True) -> PointCloud:
        r"""
        Create a point cloud from the points of a :class:`pyvista.DataSet`
        (:class:`pyvista.PolyData`, :class:`pyvista.UnstructuredGrid`, ...).

        The cells are ignored and the properties are read from the point data (their
        data types are kept). The point cloud is always 3D (:math:`E = 3`).

        Parameters
        ----------
        mesh: :class:`pyvista.DataSet`
            The dataset to read.

        load_properties: bool, optional (default: ``True``)
            If False, the point data are ignored.

        Returns
        -------
        :class:`pysdic.PointCloud`
            The new point cloud.

        """
        pyvista = import_optional("pyvista", "PointCloud.from_pyvista")

        if not isinstance(mesh, pyvista.DataSet):
            raise TypeError(f"Input 'mesh' must be a pyvista.DataSet. Got type {type(mesh)}.")
        load_properties = _validate_bool(load_properties, "load_properties")

        properties = {}
        if load_properties:
            # numpy.asarray drops the pyvista_ndarray subclass, the arrays are copied by the PropertyContainer
            properties = {key: numpy.asarray(mesh.point_data[key]) for key in mesh.point_data.keys()}

        return cls(coordinates=numpy.asarray(mesh.points), n_dimensions=3, properties=properties)

    def to_pyvista(self, *, only_finite: bool = False, save_properties: bool = True) -> pyvista.PolyData:
        r"""
        Convert the point cloud to a :class:`pyvista.PolyData` (only for 3D point clouds).

        The points are stored with one ``vertex`` cell per point, and the properties as
        point data.

        Parameters
        ----------
        only_finite: bool, optional (default: ``False``)
            If True, the points with non-finite coordinates are skipped. Otherwise, a
            ``ValueError`` is raised if such points exist.

        save_properties: bool, optional (default: ``True``)
            If False, the properties are not saved.

        Returns
        -------
        :class:`pyvista.PolyData`
            The PyVista representation of the point cloud (with copies of the arrays).

        """
        pyvista = import_optional("pyvista", "PointCloud.to_pyvista")
        points, point_data = self._vtk_export_data(only_finite, save_properties)

        mesh = pyvista.PolyData(points)  # creates one vertex cell per point
        for key, values in point_data.items():
            mesh.point_data[key] = values
        return mesh

    @classmethod
    def from_vtk(cls, dataset: vtkPointSet, *, load_properties: bool = True) -> PointCloud:
        r"""
        Create a point cloud from the points of a VTK dataset (``vtkPolyData``,
        ``vtkUnstructuredGrid``, ``vtkStructuredGrid``, ..., i.e. any ``vtkPointSet``).

        The cells are ignored and the properties are read from the point data (their
        data types are kept). The point cloud is always 3D (:math:`E = 3`).

        Parameters
        ----------
        dataset: ``vtkPointSet``
            The VTK dataset to read.

        load_properties: bool, optional (default: ``True``)
            If False, the point data are ignored.

        Returns
        -------
        :class:`pysdic.PointCloud`
            The new point cloud.

        Raises
        ------
        TypeError
            If ``dataset`` is not a ``vtkPointSet`` (e.g. a ``vtkImageData``, whose
            points are implicit), or if a point data array is not numeric (e.g. a
            ``vtkStringArray``; use ``load_properties=False`` to ignore the point data).

        ValueError
            If the dataset has no points array or if a point data array has no name.

        """
        import_optional("vtk", "PointCloud.from_vtk")
        from vtkmodules.vtkCommonDataModel import vtkPointSet
        from vtkmodules.util.numpy_support import vtk_to_numpy

        if not isinstance(dataset, vtkPointSet):
            raise TypeError(f"Input 'dataset' must be a vtkPointSet. Got type {type(dataset)}.")
        load_properties = _validate_bool(load_properties, "load_properties")
        if dataset.GetPoints() is None:
            raise ValueError("The dataset does not contain any points.")

        properties = {}
        if load_properties:
            properties = _vtk_attributes_to_dict(dataset.GetPointData(), "point data")

        coordinates = vtk_to_numpy(dataset.GetPoints().GetData())  # copied by the constructor
        return cls(coordinates=coordinates, n_dimensions=3, properties=properties)

    def to_vtk(self, *, only_finite: bool = False, save_properties: bool = True) -> vtkPolyData:
        r"""
        Convert the point cloud to a ``vtkPolyData`` (only for 3D point clouds).

        The points are stored with one ``vertex`` cell per point, and the properties as
        point data arrays (named after the properties).

        .. note::

            VTK arrays have at most 2 dimensions: a property of shape
            :math:`(N_p, A, B)` raises a ``ValueError``. Boolean properties are stored
            as ``uint8`` arrays.

        Parameters
        ----------
        only_finite: bool, optional (default: ``False``)
            If True, the points with non-finite coordinates are skipped. Otherwise, a
            ``ValueError`` is raised if such points exist.

        save_properties: bool, optional (default: ``True``)
            If False, the properties are not saved.

        Returns
        -------
        ``vtkPolyData``
            The VTK representation of the point cloud (with copies of the arrays).

        """
        import_optional("vtk", "PointCloud.to_vtk")
        from vtkmodules.vtkCommonCore import vtkPoints
        from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
        from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, ID_TYPE_CODE

        points, point_data = self._vtk_export_data(only_finite, save_properties)
        _check_vtk_property_shapes(point_data)

        vtk_points = vtkPoints()
        vtk_points.SetData(numpy_to_vtk(points, deep=True))

        # One vertex cell per point, legacy layout [1, i, 1, i, ...]
        n_points = points.shape[0]
        connectivity = numpy.empty((n_points, 2), dtype=ID_TYPE_CODE)
        connectivity[:, 0] = 1
        connectivity[:, 1] = numpy.arange(n_points)
        vertices = vtkCellArray()
        vertices.SetCells(n_points, numpy_to_vtkIdTypeArray(connectivity.ravel(), deep=True))

        polydata = vtkPolyData()
        polydata.SetPoints(vtk_points)
        polydata.SetVerts(vertices)
        _dict_to_vtk_attributes(point_data, polydata.GetPointData())
        return polydata

    def _vtk_export_data(
        self, only_finite: bool, save_properties: bool
    ) -> Tuple[numpy.ndarray, Dict[str, numpy.ndarray]]:
        # Validate the export options and return copies of the points (N_p', 3) and of the
        # properties to save, without the non-finite points if only_finite is True.
        only_finite = _validate_bool(only_finite, "only_finite")
        save_properties = _validate_bool(save_properties, "save_properties")

        if self.n_dimensions != 3:
            raise ValueError("VTK format only supports 3D point clouds.")

        finite_mask = self.is_finite()
        if not only_finite and not numpy.all(finite_mask):
            raise ValueError(
                "Point cloud contains NaN or infinite values. Consider using "
                "'only_finite=True' to skip them."
            )
        if not numpy.any(finite_mask):
            raise ValueError("No finite points to save in the point cloud.")

        points = self._coordinates[finite_mask]  # advanced indexing: copy
        point_data = {}
        if save_properties:
            point_data = {key: values[finite_mask] for key, values in self._properties.items()}
        return points, point_data

    @classmethod
    def from_vtk_file(
        cls,
        file_path: Union[str, os.PathLike],
        *,
        load_properties: bool = True,
        use_module: str = "meshio",
    ) -> PointCloud:
        r"""
        Create a point cloud from a legacy VTK file (``.vtk``, only 3D point clouds).

        The vertices are read, the cells are ignored and the properties are read from the
        point data.

        Parameters
        ----------
        file_path : Union[str, os.PathLike]
            The path to an existing VTK file, with the extension ``.vtk``.

        load_properties: bool, optional (default: ``True``)
            If False, the point data are ignored.

        use_module: str, optional (default: ``"meshio"``)
            The package used to read the file: ``"meshio"``, ``"pyvista"`` or ``"vtk"``
            (see :meth:`from_meshio`, :meth:`from_pyvista` and :meth:`from_vtk`).

        Returns
        -------
        :class:`pysdic.PointCloud`
            The new point cloud.

        """
        use_module = _validate_choice(use_module, "use_module", ("meshio", "pyvista", "vtk"))
        module = import_optional(use_module, f"PointCloud.from_vtk_file(use_module='{use_module}')")
        file_path = _validate_path(file_path, "file_path", must_exist=True, extensions=(".vtk",))
        load_properties = _validate_bool(load_properties, "load_properties")

        if use_module == "meshio":
            mesh = module.read(file_path, file_format="vtk")
            point_cloud = cls.from_meshio(mesh, load_properties=load_properties)
        elif use_module == "pyvista":
            mesh = module.read(file_path)
            point_cloud = cls.from_pyvista(mesh, load_properties=load_properties)
        else:
            from vtkmodules.vtkIOLegacy import vtkDataSetReader
            reader = vtkDataSetReader()
            reader.SetFileName(os.fspath(file_path))
            # By default, the legacy readers only read the first array of each attribute type
            reader.ReadAllScalarsOn()
            reader.ReadAllVectorsOn()
            reader.ReadAllNormalsOn()
            reader.ReadAllTensorsOn()
            reader.ReadAllColorScalarsOn()
            reader.ReadAllTCoordsOn()
            reader.ReadAllFieldsOn()
            reader.Update()
            dataset = reader.GetOutput()
            if dataset is None:
                raise ValueError(f"Cannot read the VTK file '{file_path}'.")
            point_cloud = cls.from_vtk(dataset, load_properties=load_properties)

        if point_cloud.n_dimensions != 3:
            raise ValueError("VTK format only supports 3D point clouds.")
        return point_cloud

    def to_vtk_file(
        self,
        file_path: Union[str, os.PathLike],
        *,
        make_dirs: bool = False,
        binary: bool = False,
        only_finite: bool = False,
        save_properties: bool = True,
        use_module: str = "meshio",
    ) -> None:
        r"""
        Save the point cloud to a legacy VTK file (``.vtk``, only for 3D point clouds).

        The points are saved as ``vertex`` cells, and the properties as point data.

        .. warning::

            VTK cannot handle NaN or infinite coordinates: use ``only_finite=True`` to
            skip these points. Some property shapes may also not be supported by VTK.

        Parameters
        ----------
        file_path : Union[str, os.PathLike]
            The path to the output file, with the extension ``.vtk``.

        make_dirs: bool, optional (default: ``False``)
            If True, the missing parent directories are created.

        binary: bool, optional (default: ``False``)
            If True, the file is written in binary format.

        only_finite: bool, optional (default: ``False``)
            If True, the points with non-finite coordinates are skipped. Otherwise, a
            ``ValueError`` is raised if such points exist.

        save_properties: bool, optional (default: ``True``)
            If False, the properties are not saved.

        use_module: str, optional (default: ``"meshio"``)
            The package used to write the file: ``"meshio"``, ``"pyvista"`` or ``"vtk"``
            (see :meth:`to_meshio`, :meth:`to_pyvista` and :meth:`to_vtk`).

        """
        use_module = _validate_choice(use_module, "use_module", ("meshio", "pyvista", "vtk"))
        module = import_optional(use_module, f"PointCloud.to_vtk_file(use_module='{use_module}')")
        file_path = _validate_path(file_path, "file_path", extensions=(".vtk",))
        make_dirs = _validate_bool(make_dirs, "make_dirs")
        binary = _validate_bool(binary, "binary")

        # Build the output object before touching the file system
        if use_module == "meshio":
            points, point_data = self._vtk_export_data(only_finite, save_properties)
            output = module.Mesh(
                points=points,
                cells=[("vertex", numpy.arange(points.shape[0]).reshape((-1, 1)))],
                point_data=point_data,
            )
        elif use_module == "pyvista":
            output = self.to_pyvista(only_finite=only_finite, save_properties=save_properties)
        else:
            output = self.to_vtk(only_finite=only_finite, save_properties=save_properties)

        if make_dirs:
            directory = os.path.dirname(file_path)
            if directory:  # empty for a file in the current directory
                os.makedirs(directory, exist_ok=True)

        if use_module == "meshio":
            module.write(file_path, output, file_format="vtk", binary=binary)
        elif use_module == "pyvista":
            output.save(file_path, binary=binary)
        else:
            from vtkmodules.vtkIOLegacy import vtkPolyDataWriter
            writer = vtkPolyDataWriter()
            writer.SetFileName(os.fspath(file_path))
            writer.SetInputData(output)
            if binary:
                writer.SetFileTypeToBinary()
            else:
                writer.SetFileTypeToASCII()
            if writer.Write() != 1:
                raise OSError(f"Cannot write the VTK file '{file_path}'.")

    # =======================
    # Structure Methods
    # =======================
    def concatenate(self, other: PointCloud, *, propagate_properties: bool = True) -> PointCloud:
        r"""
        Concatenate two point clouds (the points of ``other`` come after).

        The properties are concatenated with :meth:`pysdic.core.property_container.PropertyContainer.concatenate`:
        a property present in only one point cloud is completed with ``numpy.nan`` for
        the points of the other one (integer and boolean properties are then
        promoted to float).

        Parameters
        ----------
        other: :class:`pysdic.PointCloud`
            The point cloud to append, with the same dimension :math:`E`.

        propagate_properties: bool, optional (default: ``True``)
            If False, the new point cloud has no properties.

        Returns
        -------
        :class:`pysdic.PointCloud`
            A new point cloud with :math:`N_p + N_p^{other}` points.

        """
        if not isinstance(other, PointCloud):
            raise TypeError(f"Input 'other' must be a PointCloud. Got type {type(other)}.")
        propagate_properties = _validate_bool(propagate_properties, "propagate_properties")
        if self.n_dimensions != other.n_dimensions:
            raise ValueError(
                f"PointCloud instances must have the same dimension to be concatenated. "
                f"Got E={self.n_dimensions} and E={other.n_dimensions}."
            )

        concatenated_properties = None
        if propagate_properties:
            concatenated_properties = self._properties.concatenate(
                other._properties, fill=numpy.nan
            )

        return self.__class__(
            coordinates=numpy.vstack((self._coordinates, other._coordinates)),
            n_dimensions=self.n_dimensions,
            properties=concatenated_properties,
        )

    def copy(self, *, copy_properties: bool = True) -> PointCloud:
        r"""
        Create a copy of the point cloud. The copy is always unlocked.

        Parameters
        ----------
        copy_properties: bool, optional (default: ``True``)
            If False, the copy has no properties.

        Returns
        -------
        :class:`pysdic.PointCloud`
            A new point cloud with copies of the coordinates and properties.

        """
        copy_properties = _validate_bool(copy_properties, "copy_properties")

        copied_properties = None
        if copy_properties:
            copied_properties = self._properties.copy()

        return self.__class__(
            coordinates=self._coordinates,  # copied by the constructor
            n_dimensions=self.n_dimensions,
            properties=copied_properties,
        )

    def filter_points(
        self,
        mask: ArrayLike,
        *,
        propagate_properties: bool = True,
    ) -> PointCloud:
        r"""
        Keep only the points selected by a boolean mask.

        Parameters
        ----------
        mask: ArrayLike
            A boolean array of shape :math:`(N_p,)`. The points with a True value are
            kept. A mask with only False values gives an empty point cloud.

        propagate_properties: bool, optional (default: ``True``)
            If False, the new point cloud has no properties.

        Returns
        -------
        :class:`pysdic.PointCloud`
            The filtered point cloud.

        Raises
        ------
        TypeError
            If the mask is not a boolean array (an integer array is rejected: use
            :meth:`keep_points_by_index` for indices).

        ValueError
            If the mask does not have the shape :math:`(N_p,)`.

        """
        mask = _as_bool_mask(mask, self.n_points)
        propagate_properties = _validate_bool(propagate_properties, "propagate_properties")

        new_coordinates = self._coordinates[mask]  # advanced indexing: copy

        new_properties = None
        if propagate_properties:
            new_properties = self._properties.filter(mask)

        return self.__class__(
            coordinates=new_coordinates,
            n_dimensions=self.n_dimensions,
            properties=new_properties,
        )

    def remove_points_by_index(
        self,
        indices: ArrayLike,
        *,
        propagate_properties: bool = True,
    ) -> PointCloud:
        r"""
        Remove the points with the given indices.

        Parameters
        ----------
        indices: ArrayLike
            An integer array of shape :math:`(R,)` with the indices of the points to
            remove, in :math:`[0, N_p)`. Duplicates are allowed.

        propagate_properties: bool, optional (default: ``True``)
            If False, the new point cloud has no properties.

        Returns
        -------
        :class:`pysdic.PointCloud`
            The point cloud without the given points.

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
        return self.filter_points(
            mask, propagate_properties=propagate_properties
        )

    def keep_points_by_index(
        self,
        indices: ArrayLike,
        *,
        propagate_properties: bool = True,
    ) -> PointCloud:
        r"""
        Keep only the points with the given indices.

        .. note::

            The points keep their original order and duplicated indices are ignored:
            ``keep_points_by_index([2, 0])`` returns the points 0 and 2, in this order.

        Parameters
        ----------
        indices: ArrayLike
            An integer array of shape :math:`(R,)` with the indices of the points to
            keep, in :math:`[0, N_p)`.

        propagate_properties: bool, optional (default: ``True``)
            If False, the new point cloud has no properties.

        Returns
        -------
        :class:`pysdic.PointCloud`
            The point cloud with only the given points.

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
        return self.filter_points(
            mask, propagate_properties=propagate_properties
        )

    # ==========================
    # Methods Geometry Manipulation
    # ==========================
    def all_close(
        self,
        other: Union[PointCloud, ArrayLike],
        *,
        rtol: Real = 1e-05,
        atol: Real = 1e-08,
        equal_nan: bool = True,
        ordered: bool = True,
    ) -> bool:
        r"""
        Check if the points are approximately equal to the points of another point
        cloud (see ``numpy.allclose``).

        Point clouds with different shapes are never close.

        .. note::

            With ``ordered=False``, both point clouds are sorted lexicographically
            before the comparison. Two points whose first coordinates are equal up to
            the tolerance may then be sorted differently, so a False result is not a
            proof that the point clouds differ.

        Parameters
        ----------
        other: Union[:class:`pysdic.PointCloud`, ArrayLike]
            Another point cloud or an array of shape :math:`(N_p, E)`.

        rtol: Real, optional (default: ``1e-05``)
            The relative tolerance (non-negative and finite).

        atol: Real, optional (default: ``1e-08``)
            The absolute tolerance (non-negative and finite).

        equal_nan: bool, optional (default: ``True``)
            If True, NaN values are considered equal.

        ordered: bool, optional (default: ``True``)
            If False, the order of the points is ignored (see the note above).

        Returns
        -------
        bool
            True if all the points are close, False otherwise.

        Examples
        --------
        Compare a point cloud with a slightly noisy copy and with a shifted copy.

        .. code-block:: python
            :linenos:

            import numpy
            from pysdic import PointCloud

            points = numpy.random.rand(100, 3)
            point_cloud = PointCloud(points)

            noise = numpy.random.normal(scale=1e-9, size=points.shape)
            print(point_cloud.all_close(PointCloud(points + noise)))  # True
            print(point_cloud.all_close(PointCloud(points + 1.0)))    # False

        """
        rtol, atol = _validate_tolerances(rtol, atol)
        equal_nan = _validate_bool(equal_nan, "equal_nan")
        ordered = _validate_bool(ordered, "ordered")

        if isinstance(other, PointCloud):
            other_coordinates = other._coordinates
        else:
            other_coordinates = _as_array(
                other, "other", numpy.float64, shape=(None, None), shape_hint="(N_p, E)", finite=False
            )

        if self._coordinates.shape != other_coordinates.shape:
            return False

        self_coordinates = self._coordinates
        if not ordered:
            # lexsort uses the last key as the primary one
            self_coordinates = self_coordinates[numpy.lexsort(self_coordinates.T[::-1])]
            other_coordinates = other_coordinates[numpy.lexsort(other_coordinates.T[::-1])]

        return bool(
            numpy.allclose(
                self_coordinates, other_coordinates, rtol=rtol, atol=atol, equal_nan=equal_nan
            )
        )

    def all_finite(self) -> bool:
        r"""
        Check if all the coordinates are finite (no NaN or infinite values).

        Returns
        -------
        bool
            True if all the coordinates are finite, False otherwise.

        """
        return bool(numpy.isfinite(self._coordinates).all())

    def is_finite(self) -> numpy.ndarray:
        r"""
        Check which points have only finite coordinates.

        Returns
        -------
        numpy.ndarray
            A boolean array of shape :math:`(N_p,)`, True for the finite points.

        """
        return numpy.isfinite(self._coordinates).all(axis=1)

    def has_nan(self) -> numpy.ndarray:
        r"""
        Check which points have at least one NaN coordinate.

        Returns
        -------
        numpy.ndarray
            A boolean array of shape :math:`(N_p,)`, True for the points with NaN.

        """
        return numpy.isnan(self._coordinates).any(axis=1)

    def has_inf(self) -> numpy.ndarray:
        r"""
        Check which points have at least one infinite coordinate.

        Returns
        -------
        numpy.ndarray
            A boolean array of shape :math:`(N_p,)`, True for the points with an
            infinite value.

        """
        return numpy.isinf(self._coordinates).any(axis=1)

    def frame_transform(
        self,
        input_frame: Optional[py3dframe.Frame] = None,
        output_frame: Optional[py3dframe.Frame] = None,
        *,
        propagate_properties: bool = True,
    ) -> PointCloud:
        r"""
        Express the points, given in ``input_frame``, in ``output_frame`` (only for 3D
        point clouds).

        .. note::

            Only the coordinates are transformed. Vector properties (normals,
            displacements, ...) are copied as they are and are not rotated.

        .. seealso::

            Package `py3dframe <https://pypi.org/project/py3dframe/>`_ for
            :class:`py3dframe.Frame` and :class:`py3dframe.FrameTransform`.

        Parameters
        ----------
        input_frame: Optional[:class:`py3dframe.Frame`], optional
            The frame in which the coordinates are currently expressed. By default, the
            canonical frame.

        output_frame: Optional[:class:`py3dframe.Frame`], optional
            The frame in which the coordinates are expressed after the transformation.
            By default, the canonical frame.

        propagate_properties: bool, optional (default: ``True``)
            If False, the new point cloud has no properties.

        Returns
        -------
        :class:`pysdic.PointCloud`
            The point cloud expressed in ``output_frame``.

        Examples
        --------
        Express a point cloud given in the canonical frame in a local frame with origin
        :math:`(1, 1, 1)` and axes :math:`x = (0, 1, 0)`, :math:`y = (-1, 0, 0)` and
        :math:`z = (0, 0, 1)`.

        .. code-block:: python
            :linenos:

            import numpy
            import py3dframe
            from pysdic import PointCloud

            point_cloud = PointCloud(numpy.random.rand(100, 3))

            output_frame = py3dframe.Frame(origin=[1, 1, 1], x_axis=[0, 1, 0], y_axis=[-1, 0, 0], z_axis=[0, 0, 1])
            local_point_cloud = point_cloud.frame_transform(output_frame=output_frame)

        """
        py3dframe = import_optional("py3dframe", "PointCloud.frame_transform")
        if self.n_dimensions != 3:
            raise ValueError("py3dframe.Frame transformations are only supported for 3D point clouds.")
        if input_frame is not None and not isinstance(input_frame, py3dframe.Frame):
            raise TypeError(
                f"Input 'input_frame' must be a py3dframe.Frame or None. Got type {type(input_frame)}."
            )
        if output_frame is not None and not isinstance(output_frame, py3dframe.Frame):
            raise TypeError(
                f"Input 'output_frame' must be a py3dframe.Frame or None. Got type {type(output_frame)}."
            )
        propagate_properties = _validate_bool(propagate_properties, "propagate_properties")

        if input_frame is None:
            input_frame = py3dframe.Frame.canonical()
        if output_frame is None:
            output_frame = py3dframe.Frame.canonical()

        transform = py3dframe.FrameTransform(input_frame=input_frame, output_frame=output_frame)
        transformed_points = numpy.array(
            transform.transform(point=self._coordinates.T).T, dtype=numpy.float64
        )

        new_properties = None
        if propagate_properties:
            new_properties = self._properties.copy()

        return self.__class__(
            coordinates=transformed_points,
            n_dimensions=self.n_dimensions,
            properties=new_properties,
        )

    # ==============
    # Geometric Computations
    # ==============
    def _finite_coordinates(self) -> numpy.ndarray:
        # Finite points, with an explicit error if there are none
        if self.n_points == 0:
            raise ValueError("The point cloud is empty.")
        finite_points = self._coordinates[self.is_finite()]
        if finite_points.shape[0] == 0:
            raise ValueError("The point cloud contains only non-finite points.")
        return finite_points

    def bounding_box(self) -> Tuple[numpy.ndarray, numpy.ndarray]:
        r"""
        Compute the axis-aligned bounding box of the point cloud.

        The non-finite points (NaN, Inf) are ignored.

        Returns
        -------
        numpy.ndarray
            The minimum coordinates, an array of shape :math:`(E,)`.

        numpy.ndarray
            The maximum coordinates, an array of shape :math:`(E,)`.

        Raises
        ------
        ValueError
            If the point cloud is empty or contains only non-finite points.

        Examples
        --------
        .. code-block:: python
            :linenos:

            import numpy
            from pysdic import PointCloud

            point_cloud = PointCloud([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]])
            min_coords, max_coords = point_cloud.bounding_box()
            print(min_coords, max_coords)  # [0. 0. 0.] [1. 2. 3.]

        """
        finite_points = self._finite_coordinates()
        return numpy.min(finite_points, axis=0), numpy.max(finite_points, axis=0)

    def bounding_sphere(self) -> Tuple[numpy.ndarray, float]:
        r"""
        Compute a bounding sphere of the point cloud.

        The center is the centroid :math:`\bar{x}` of the points and the radius is the
        largest distance to the centroid:

        .. math::

            r = \max_i \| x_i - \bar{x} \|

        .. note::

            This sphere contains all the points but is not the smallest one.

        The non-finite points (NaN, Inf) are ignored.

        Returns
        -------
        numpy.ndarray
            The center of the sphere, an array of shape :math:`(E,)`.

        float
            The radius of the sphere.

        Raises
        ------
        ValueError
            If the point cloud is empty or contains only non-finite points.

        Examples
        --------
        .. code-block:: python
            :linenos:

            import numpy
            from pysdic import PointCloud

            point_cloud = PointCloud([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]])
            center, radius = point_cloud.bounding_sphere()
            print(center, radius)  # [0.25 0.5  0.75] 2.318...

        """
        finite_points = self._finite_coordinates()
        center = numpy.mean(finite_points, axis=0)  # (E,)
        radius = float(numpy.max(numpy.linalg.norm(finite_points - center, axis=1)))
        return center, radius

    def remove_dimension(
        self, dim: Integral, *, propagate_properties: bool = True
    ) -> PointCloud:
        r"""
        Remove one coordinate axis, giving a point cloud of shape :math:`(N_p, E-1)`.

        For example, with :math:`E=3`, ``dim=1`` removes the :math:`y` axis.

        Parameters
        ----------
        dim: Integral
            The index of the axis to remove, in :math:`[0, E)`.

        propagate_properties: bool, optional (default: ``True``)
            If False, the new point cloud has no properties.

        Returns
        -------
        :class:`pysdic.PointCloud`
            The point cloud without the given axis.

        Raises
        ------
        ValueError
            If ``dim`` is out of range or if :math:`E = 1`.

        """
        if self.n_dimensions <= 1:
            raise ValueError("Cannot remove a dimension from a point cloud with E = 1.")
        dim = _validate_integer(dim, "dim", min_value=0)
        if dim >= self.n_dimensions:
            raise ValueError(f"Input 'dim' must be in [0, {self.n_dimensions - 1}]. Got {dim}.")
        propagate_properties = _validate_bool(propagate_properties, "propagate_properties")

        new_coordinates = numpy.delete(self._coordinates, dim, axis=1)

        new_properties = None
        if propagate_properties:
            new_properties = self._properties.copy()

        return self.__class__(
            coordinates=new_coordinates,
            n_dimensions=self.n_dimensions - 1,
            properties=new_properties,
        )

    def insert_dimension(
        self,
        dim: Integral,
        value: Union[Real, ArrayLike] = 0.0,
        *,
        propagate_properties: bool = True,
    ) -> PointCloud:
        r"""
        Insert a new coordinate axis, giving a point cloud of shape
        :math:`(N_p, E+1)`.

        For example, with :math:`E=2`, ``dim=2`` adds a :math:`z` axis after
        :math:`(x, y)`.

        Parameters
        ----------
        dim: Integral
            The index of the new axis, in :math:`[0, E]`.

        value: Union[Real, ArrayLike], optional (default: ``0.0``)
            The coordinates along the new axis: one value for all the points, or an
            array of shape :math:`(N_p,)`.

        propagate_properties: bool, optional (default: ``True``)
            If False, the new point cloud has no properties.

        Returns
        -------
        :class:`pysdic.PointCloud`
            The point cloud with the new axis.

        """
        dim = _validate_integer(dim, "dim", min_value=0)
        if dim > self.n_dimensions:
            raise ValueError(f"Input 'dim' must be in [0, {self.n_dimensions}]. Got {dim}.")
        propagate_properties = _validate_bool(propagate_properties, "propagate_properties")

        value = _as_scalar_or_1d_array(value, "value", self.n_points, numpy.float64, finite=False)

        new_coordinates = numpy.insert(self._coordinates, dim, values=value, axis=1)

        new_properties = None
        if propagate_properties:
            new_properties = self._properties.copy()

        return self.__class__(
            coordinates=new_coordinates,
            n_dimensions=self.n_dimensions + 1,
            properties=new_properties,
        )

    # ==============
    # Visualization
    # ==============
    def quick_visualize(
        self,
        *,
        warp_by: Optional[Union[str, ArrayLike]] = None,
        color: Optional[Union[str, Tuple[Real, Real, Real]]] = None,
        size: Optional[Real] = None,
        opacity: Optional[Real] = None,
        mask: Optional[ArrayLike] = None,
        property: Optional[Union[str, ArrayLike]] = None,
        property_axis: Optional[int] = None,
        property_clim: Optional[Tuple[Real, Real]] = None,
        property_cmap: Optional[Union[str, matplotlib.colors.Colormap]] = None,
        property_below_color: Optional[Union[str, Tuple[Real, Real, Real]]] = None,
        property_above_color: Optional[Union[str, Tuple[Real, Real, Real]]] = None,
        title: Optional[str] = None,
        show_bar: bool = True,
        show_bounds: bool = True,
        show_points_index: bool = False,
        index_font_size: Real = 10,
        screenshot: Optional[str] = None,
    ) -> None:
        r"""
        Visualize the point cloud with PyVista (only for :math:`E \leq 3`).

        Only the finite points are displayed.

        .. seealso::

            - :class:`pysdic.Plotter3D` for more control over the visualization.
            - `PyVista Documentation <https://docs.pyvista.org>`_.

        Parameters
        ----------
        warp_by: Optional[Union[str, ArrayLike]], optional
            The property used to warp the points: the name of a property of the point
            cloud, or an array of shape :math:`(N_p,)` or :math:`(N_p, A)`.

        color: Union[str, Tuple[Real, Real, Real]], optional
            The color of the points: a color name (e.g. ``"red"``), an RGB tuple or a
            hex string. Ignored if ``property`` is given.

        size: Real, optional
            The size of the points (positive). PyVista default: ``5.0``.

        opacity: Real, optional
            The opacity of the points, between ``0.0`` (transparent) and ``1.0``
            (opaque). PyVista default: ``1.0``.

        mask: Optional[ArrayLike], optional
            A boolean array of shape :math:`(N_p,)`, True for the points to display.
            By default, all the finite points are displayed.

        property: Union[str, ArrayLike], optional
            The property used to color the points: the name of a property of the point
            cloud, or an array of shape :math:`(N_p,)` or :math:`(N_p, A)`.

        property_axis: Optional[Integral], optional
            The component used to color the points when the property has :math:`A > 1`
            components. By default, the norm of the property vector is used.

        property_clim: Optional[Tuple[Real, Real]], optional
            The color limits (min, max). By default, they are set automatically.

        property_cmap: Union[str, :class:`matplotlib.colors.Colormap`], optional
            The colormap (name or object). Matplotlib default: ``"viridis"``.

        property_below_color: Union[str, Tuple[Real, Real, Real]], optional
            The color of the values below ``property_clim``. By default, the first
            color of the colormap.

        property_above_color: Union[str, Tuple[Real, Real, Real]], optional
            The color of the values above ``property_clim``. By default, the last
            color of the colormap.

        title: Optional[str], optional
            The title of the window.

        show_bar: bool, optional (default: ``True``)
            Whether to show the color bar when ``property`` is given.

        show_bounds: bool, optional (default: ``True``)
            Whether to show the axes bounds.

        show_points_index: bool, optional (default: ``False``)
            Whether to write the index of each displayed point next to it.

        index_font_size: Real, optional (default: ``10``)
            The font size of the indices, in pixels of the screen (constant when zooming).

        screenshot: Optional[str], optional
            The path of a screenshot to save. By default, no screenshot is saved.

        Examples
        --------
        .. code-block:: python
            :linenos:

            import numpy
            from pysdic import PointCloud

            point_cloud = PointCloud(numpy.random.rand(100, 3))
            point_cloud.quick_visualize(color='red', size=10)
            point_cloud.quick_visualize(show_points_index=True)  # Numbering of the points

        """
        from .plotter3d import Plotter3D

        plotter = Plotter3D()
        plotter.add_point_cloud(
            point_cloud=self,
            warp_by=warp_by,
            color=color,
            size=size,
            opacity=opacity,
            mask=mask,
            property=property,
            property_axis=property_axis,
            property_clim=property_clim,
            property_cmap=property_cmap,
            property_below_color=property_below_color,
            property_above_color=property_above_color,
            show_bar=show_bar,
            show_points_index=show_points_index,
            index_font_size=index_font_size,
        )
        if show_bounds:
            plotter.set_bounds()
        plotter.set_title(title=title)
        plotter.show(screenshot=screenshot)