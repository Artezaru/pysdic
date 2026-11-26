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
from numpy.typing import ArrayLike

import numpy
import scipy.sparse
import os

from .point_cloud import PointCloud
from ._vtk_properties import (
    _check_vtk_property_shapes,
    _dict_to_vtk_attributes,
    _vtk_attributes_to_dict,
)
from .connectivity import Connectivity, _read_element_type
from .integration_points import IntegrationPoints
from ..transformations.rays import Rays
from ..core.lockable import LockableMixin
from ..core.serialization import (
    JsonSerializerMixin,
    NpzSerializerMixin,
    Hdf5SerializerMixin,
)
from .._validation import (
    _validate_bool,
    _validate_choice,
    _validate_path,
    _as_array,
    _as_bool_mask,
    _as_index_array,
)
from .._optionals import import_optional
from .. import fem
from ..fem._implemented_element_type import _MAPPING_ELEMENT_TYPE_TO_PROPERTIES
from ..fem._validation import _validate_element_type

if TYPE_CHECKING:
    import matplotlib.colors
    import meshio
    import pyvista
    from vtkmodules.vtkCommonDataModel import vtkPointSet, vtkUnstructuredGrid


class Mesh(LockableMixin, JsonSerializerMixin, NpzSerializerMixin, Hdf5SerializerMixin):
    r"""
    A mesh: :math:`N_v` vertices (a :class:`pysdic.PointCloud` of shape
    :math:`(N_v, E)`) and :math:`N_e` elements (a :class:`pysdic.Connectivity` of shape
    :math:`(N_e, N_{vpe})`) whose indices refer to these vertices.

    A point inside an element is located by its natural coordinates
    :math:`\Xi = (\xi, \eta, \zeta, ...)` in the reference element. There are :math:`K`
    natural coordinates, the topological dimension of the element
    (``n_topological_dimensions``). The global coordinates of the point are given by
    the shape functions :math:`N_i` of the element:

    .. math::

        X = \sum_{i=1}^{N_{vpe}} N_i(\Xi) X_i

    where :math:`X_i` are the coordinates of the vertices of the element.

    .. seealso::

        :func:`pysdic.fem.compute_shape_functions` for the shape functions.

    The element types implemented in :mod:`pysdic.fem` are listed in
    :class:`pysdic.Connectivity`. An element type is required for the operations that
    use shape functions and for the meshio and VTK exports.

    .. important::

        The mesh always satisfies :math:`0 \leq` index :math:`< N_v` for all the vertex
        indices of the elements. The setters of :class:`pysdic.Mesh` (``vertices``,
        ``connectivity``, ``coordinates``, ``elements``, ...) check it.

        ``mesh.vertices`` and ``mesh.connectivity`` return the internal objects, so
        that properties can be read and set directly (``mesh.vertices["u"] = ...``).
        Their number of vertices and elements cannot be changed through them (the
        methods that change it return new objects). Their coordinates and elements can
        be set through them (``mesh.connectivity.elements = ...``), but this bypasses the
        check of the vertex indices: prefer the setters of :class:`pysdic.Mesh`, or call
        :meth:`validate`.

    .. note::

        A mesh can be locked by an object of the package that owns it (for example an
        SDIC analysis, see :class:`pysdic.core.lockable.LockableMixin`). Its vertices and
        its connectivity are locked with it: the setters ``vertices``, ``connectivity``,
        ``coordinates``, ``elements`` and ``element_type`` of the mesh, and the setters of
        ``mesh.vertices`` and ``mesh.connectivity``, then raise a :class:`RuntimeError`
        explaining why. The properties (and the ``uvmap``) can always be modified. A copy
        (``copy``) is always unlocked.

    Parameters
    ----------
    vertices: Union[:class:`pysdic.PointCloud`, ArrayLike]
        The vertices of the mesh, as a :class:`pysdic.PointCloud` or an array of shape
        :math:`(N_v, E)`. A given point cloud is copied (with its properties).

    connectivity: Union[:class:`pysdic.Connectivity`, ArrayLike]
        The elements of the mesh, as a :class:`pysdic.Connectivity` or an integer array
        of shape :math:`(N_e, N_{vpe})`. A given connectivity is copied (with its
        properties and element type).

    element_type: Optional[str], optional (default: ``None``)
        The type of the elements. If the connectivity already has an element type,
        both must be the same.

    Notes
    -----
    The methods never modify the mesh in place: the methods that change the vertices or
    the elements (:meth:`filter_connectivity`, :meth:`filter_vertices`,
    :meth:`remove_unused_vertices`, ...) return a new mesh, and the current one is
    unchanged. The new mesh never shares its point cloud, connectivity or properties
    with the current one.

    """

    __slots__ = ["_vertices", "_connectivity"]

    def __init__(
        self,
        vertices: Union[PointCloud, ArrayLike],
        connectivity: Union[Connectivity, ArrayLike],
        *,
        element_type: Optional[str] = None,
    ) -> None:
        vertices = (
            vertices.copy() if isinstance(vertices, PointCloud) else PointCloud(vertices)
        )
        connectivity = (
            connectivity.copy()
            if isinstance(connectivity, Connectivity)
            else Connectivity(connectivity)
        )

        if element_type is not None:
            element_type = _validate_element_type(element_type)
            if connectivity.element_type not in (None, element_type):
                raise ValueError(
                    f"Input 'element_type' ('{element_type}') does not match the element "
                    f"type of the connectivity ('{connectivity.element_type}')."
                )
            connectivity.element_type = element_type  # on the copy only

        self._vertices = vertices
        self._connectivity = connectivity
        self.validate(recursive=True)

    @classmethod
    def _from_owned(cls, vertices: PointCloud, connectivity: Connectivity) -> Mesh:
        # Build a mesh from objects that are not referenced anywhere else (no copy)
        mesh = cls.__new__(cls)
        mesh._vertices = vertices
        mesh._connectivity = connectivity
        mesh.validate(recursive=False)
        return mesh

    # =======================
    # Internals
    # =======================
    def _get_expected_N_vpe(self) -> Optional[int]:
        # Number of vertices of the element type, or None if element_type is not set
        return self._connectivity._expected_N_vpe

    def _get_expected_K(self) -> Optional[int]:
        # Topological dimension of the element type, or None if element_type is not set
        return self._connectivity._expected_K

    def _get_meshio_cell_type(self) -> Optional[str]:
        # meshio cell type of the elements, or None if element_type is not set
        return self._connectivity._meshio_cell_type

    def _get_vtk_cell_type(self) -> Optional[int]:
        # VTK cell type of the elements, or None if element_type is not set
        return self._connectivity._vtk_cell_type

    def _check_surface_mesh(
        self, feature: str, *, require_3d: bool = False, require_element_type: bool = True
    ) -> Optional[str]:
        # Check that the mesh is a surface mesh (K = 2), in 3D if required, and return its
        # element type. Without element type, raise an error if required, otherwise return None.
        element_type = self.element_type
        if element_type is None:
            if require_element_type:
                raise ValueError(
                    f"'{feature}' requires the element type of the mesh. "
                    f"Set 'mesh.element_type' first."
                )
            return None
        expected_K = self._get_expected_K()
        if expected_K != 2:
            raise ValueError(
                f"'{feature}' is only available for surface meshes (K = 2). Got element "
                f"type '{element_type}' with K={expected_K}."
            )
        if require_3d and self.n_dimensions != 3:
            raise ValueError(
                f"'{feature}' is only available for surface meshes in 3D space (E = 3). "
                f"Got E={self.n_dimensions}."
            )
        return element_type

    def _check_indices(self, elements: numpy.ndarray, n_vertices: int) -> None:
        # All the vertex indices must refer to existing vertices
        if elements.size > 0 and elements.max() >= n_vertices:
            raise IndexError(
                f"The connectivity contains vertex indices out of bounds. Maximum index "
                f"found: {elements.max()}, number of vertices: {n_vertices}."
            )

    def validate(self, *, recursive: bool = True) -> None:
        r"""
        Validate the integrity of the mesh.

        The following conditions are checked:

        - The vertices are a :class:`pysdic.PointCloud` and the connectivity a
          :class:`pysdic.Connectivity`.
        - All the vertex indices of the elements are lower than :math:`N_v`.

        Parameters
        ----------
        recursive: bool, optional (default: ``True``)
            If True, the point cloud and the connectivity are also validated (see
            :meth:`pysdic.PointCloud.validate` and
            :meth:`pysdic.Connectivity.validate`).

        Raises
        ------
        TypeError
            If an attribute has a wrong type.

        ValueError
            If an attribute has a wrong shape or value.

        IndexError
            If a vertex index is out of :math:`[0, N_v)`.

        """
        recursive = _validate_bool(recursive, "recursive")

        if not isinstance(self._vertices, PointCloud):
            raise TypeError("[Mesh validate] vertices should be a PointCloud.")
        if not isinstance(self._connectivity, Connectivity):
            raise TypeError("[Mesh validate] connectivity should be a Connectivity.")
        if recursive:
            self._vertices.validate(recursive=True)
            self._connectivity.validate(recursive=True)

        self._check_indices(self._connectivity._elements, self._vertices.n_points)

    def validate_integration_points(self, integration_points: IntegrationPoints) -> None:
        r"""
        Check that integration points are consistent with the mesh.

        The following conditions are checked:

        - ``integration_points`` is a :class:`pysdic.IntegrationPoints`.
        - The element type of the mesh is set, and its topological dimension :math:`K`
          is the number of natural coordinates of the points.
        - The element indices of the located points are lower than :math:`N_e` (the
          unlocated points, with element index ``-1``, are accepted).

        The natural coordinates are not checked against the reference element: the
        :mod:`pysdic.fem` functions handle the points outside it.

        Parameters
        ----------
        integration_points: :class:`pysdic.IntegrationPoints`
            The integration points to check.

        Raises
        ------
        TypeError
            If ``integration_points`` is not a :class:`pysdic.IntegrationPoints`.

        ValueError
            If the mesh has no element type, or if the topological dimensions differ.

        IndexError
            If a point refers to an element :math:`\geq N_e`.

        Examples
        --------
        .. code-block:: python
            :linenos:

            from pysdic import Mesh, IntegrationPoints

            mesh = Mesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]], element_type="triangle_3")

            mesh.validate_integration_points(IntegrationPoints.from_gauss_points("triangle_3", 1))  # OK
            mesh.validate_integration_points(IntegrationPoints([[0.2, 0.2]], [1]))  # IndexError

        """
        if not isinstance(integration_points, IntegrationPoints):
            raise TypeError(
                f"Input 'integration_points' must be an IntegrationPoints. "
                f"Got type {type(integration_points)}."
            )
        expected_K = self._get_expected_K()
        if expected_K is None:
            raise ValueError(
                "Cannot check integration points against a mesh without element type. "
                "Set 'mesh.element_type' first."
            )
        if integration_points.n_topological_dimensions != expected_K:
            raise ValueError(
                f"The integration points have K={integration_points.n_topological_dimensions} "
                f"natural coordinates, but the elements '{self.element_type}' of the mesh "
                f"have K={expected_K}."
            )
        element_indices = integration_points._element_indices
        if element_indices.size > 0 and element_indices.max() >= self.n_elements:
            raise IndexError(
                f"The integration points refer to elements out of bounds. Maximum element "
                f"index found: {element_indices.max()}, number of elements: {self.n_elements}."
            )

    def __repr__(self) -> str:
        string = (
            f"Mesh(n_vertices={self.n_vertices}, n_dimensions={self.n_dimensions}, "
            f"n_elements={self.n_elements}, n_vertices_per_element={self.n_vertices_per_element})"
        )
        if self.element_type is not None:
            string += f"\n  - Element Type: '{self.element_type}'"
        for key in self._vertices.properties.keys():
            shape = self._vertices.properties.get_shape(key)
            string += f"\n  - Vertex property: '{key}' with shape {shape}"
        for key in self._connectivity.properties.keys():
            shape = self._connectivity.properties.get_shape(key)
            string += f"\n  - Element property: '{key}' with shape {shape}"
        return string

    # =======================
    # I/O Methods
    # =======================
    @classmethod
    def from_meshio(
        cls,
        mesh: meshio.Mesh,
        *,
        element_type: Optional[str] = None,
        load_properties: bool = True,
    ) -> Mesh:
        r"""
        Create a mesh from a :class:`meshio.Mesh` with a single cell block.

        The following fields are read:

        - ``mesh.points`` → vertices,
        - ``mesh.cells[0].data`` → connectivity,
        - ``mesh.point_data`` → vertex properties,
        - ``mesh.cell_data`` → element properties.

        .. seealso::

            - :meth:`to_meshio` for the reverse operation.
            - `meshio documentation <https://pypi.org/project/meshio/>`_.

        Parameters
        ----------
        mesh: :class:`meshio.Mesh`
            The mesh to read. It must contain exactly one cell block.

        element_type: Optional[str], optional
            The type of the elements. By default, it is inferred from the meshio cell
            type when this type is implemented (otherwise the mesh has no element type).

        load_properties: bool, optional (default: ``True``)
            If False, the point and cell data are ignored.

        Returns
        -------
        :class:`pysdic.Mesh`
            The new mesh.

        Raises
        ------
        TypeError
            If ``mesh`` is not a :class:`meshio.Mesh`.

        ValueError
            If the mesh does not contain exactly one cell block, or if ``element_type``
            does not match the cells.

        Examples
        --------
        .. code-block:: python
            :linenos:

            import numpy
            import meshio
            from pysdic import Mesh

            points = numpy.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            cells = [("triangle", numpy.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]))]

            mesh = Mesh.from_meshio(meshio.Mesh(points=points, cells=cells))
            print(mesh.element_type)  # triangle_3

        """
        meshio = import_optional("meshio", "Mesh.from_meshio")

        if not isinstance(mesh, meshio.Mesh):
            raise TypeError(f"Input 'mesh' must be a meshio.Mesh. Got type {type(mesh)}.")
        load_properties = _validate_bool(load_properties, "load_properties")
        if len(mesh.cells) != 1:
            raise ValueError(
                f"Input 'mesh' must contain exactly one cell block. Got {len(mesh.cells)}."
            )
        cell_block = mesh.cells[0]

        vertex_properties = {}
        element_properties = {}
        if load_properties:
            vertex_properties = dict(mesh.point_data)
            element_properties = {key: value[0] for key, value in mesh.cell_data.items()}

        if element_type is None:
            for name, properties in _MAPPING_ELEMENT_TYPE_TO_PROPERTIES.items():
                if properties["meshio_cell_type"] == cell_block.type:
                    element_type = name
                    break

        vertices = PointCloud(mesh.points, properties=vertex_properties)
        connectivity = Connectivity(
            cell_block.data, properties=element_properties, element_type=element_type
        )
        return cls._from_owned(vertices, connectivity)

    def to_meshio(self, *, save_properties: bool = True) -> meshio.Mesh:
        r"""
        Convert the mesh to a :class:`meshio.Mesh` with a single cell block.

        The element type must be set and the mesh must not be empty.

        The following fields are written:

        - vertices → ``mesh.points``,
        - connectivity → ``mesh.cells[0]``,
        - vertex properties → ``mesh.point_data``,
        - element properties → ``mesh.cell_data``.

        .. seealso::

            - :meth:`from_meshio` for the reverse operation.
            - `meshio documentation <https://pypi.org/project/meshio/>`_.

        Parameters
        ----------
        save_properties: bool, optional (default: ``True``)
            If False, the properties are not saved.

        Returns
        -------
        :class:`meshio.Mesh`
            The meshio representation of the mesh.

        Raises
        ------
        ValueError
            If the mesh is empty or has no element type.

        """
        meshio = import_optional("meshio", "Mesh.to_meshio")

        save_properties = _validate_bool(save_properties, "save_properties")
        if self.n_vertices == 0 or self.n_elements == 0:
            raise ValueError("Cannot convert an empty mesh to a meshio.Mesh.")
        meshio_cell_type = self._get_meshio_cell_type()
        if meshio_cell_type is None:
            raise ValueError(
                "Cannot convert a mesh without element type to a meshio.Mesh. "
                "Set 'mesh.element_type' first."
            )

        point_data = {}
        cell_data = {}
        if save_properties:
            point_data = self._vertices.properties.to_dict()
            cell_data = {
                key: [value] for key, value in self._connectivity.properties.to_dict().items()
            }

        return meshio.Mesh(
            points=self._vertices.to_array(),
            cells=[meshio.CellBlock(meshio_cell_type, data=self._connectivity.to_array())],
            point_data=point_data,
            cell_data=cell_data,
        )

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        *,
        load_properties: bool = True,
        element_type: Optional[str] = None,
    ) -> Mesh:
        r"""
        Create a mesh from a flat dictionary.

        The dictionary must contain the keys ``"coordinates"`` (shape :math:`(N_v, E)`)
        and ``"elements"`` (shape :math:`(N_e, N_{vpe})`). The properties are read from
        the keys ``"_vertex_property/{name}"`` and ``"_element_property/{name}"``, and
        the element type from the optional key ``"element_type"`` (a string, or an
        array with one string as written by the NPZ and HDF5 serializers).

        .. note::

            The dictionary is flat (one level), as required by the NPZ format. See
            :class:`pysdic.NpzSerializerMixin`.

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
        :class:`pysdic.Mesh`
            The new mesh.

        """
        if not isinstance(data, Mapping):
            raise TypeError(f"Input 'data' must be a dictionary. Got type {type(data)}.")
        for key, shape in (("coordinates", "(N_v, E)"), ("elements", "(N_e, N_vpe)")):
            if key not in data:
                raise ValueError(
                    f"Input dictionary must contain a key '{key}' with shape {shape}."
                )
        load_properties = _validate_bool(load_properties, "load_properties")

        vertex_properties = {}
        element_properties = {}
        if load_properties:
            for key in data:
                if not isinstance(key, str):
                    continue
                if key.startswith("_vertex_property/"):
                    vertex_properties[key[len("_vertex_property/"):]] = data[key]
                elif key.startswith("_element_property/"):
                    element_properties[key[len("_element_property/"):]] = data[key]

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

        vertices = PointCloud(data["coordinates"], properties=vertex_properties)
        connectivity = Connectivity(
            data["elements"], properties=element_properties, element_type=element_type
        )
        return cls._from_owned(vertices, connectivity)

    def to_dict(self, *, save_properties: bool = True) -> Dict[str, Any]:
        r"""
        Convert the mesh to a flat dictionary.

        The coordinates are stored under the key ``"coordinates"``, the elements under
        ``"elements"``, the element type (if set) under ``"element_type"`` and the
        properties under ``"_vertex_property/{name}"`` and
        ``"_element_property/{name}"``. All the arrays are copies.

        .. note::

            The dictionary is flat (one level), as required by the NPZ format. See
            :class:`pysdic.NpzSerializerMixin`.

        Parameters
        ----------
        save_properties: bool, optional (default: ``True``)
            If False, the properties are not included.

        Returns
        -------
        Dict[str, Any]
            The dictionary representation of the mesh.

        """
        save_properties = _validate_bool(save_properties, "save_properties")

        data = {
            "coordinates": self._vertices.to_array(),
            "elements": self._connectivity.to_array(),
        }
        if self.element_type is not None:
            data["element_type"] = self.element_type
        if save_properties:
            for key, values in self._vertices.properties.to_dict().items():
                data[f"_vertex_property/{key}"] = values
            for key, values in self._connectivity.properties.to_dict().items():
                data[f"_element_property/{key}"] = values
        return data

    def _check_vtk_export(self) -> int:
        # The mesh must be 3D, non-empty and have an element type: return its VTK cell type
        if self.n_dimensions != 3:
            raise ValueError("VTK format only supports meshes with E = 3.")
        if self.n_vertices == 0 or self.n_elements == 0:
            raise ValueError("Cannot convert an empty mesh to VTK.")
        vtk_cell_type = self._get_vtk_cell_type()
        if vtk_cell_type is None:
            raise ValueError(
                "Cannot convert a mesh without element type to VTK. "
                "Set 'mesh.element_type' first."
            )
        return vtk_cell_type

    @staticmethod
    def _element_type_from_vtk_cell_type(vtk_cell_type: int) -> Optional[str]:
        # Implemented element type with this VTK cell type, or None
        for name, properties in _MAPPING_ELEMENT_TYPE_TO_PROPERTIES.items():
            if properties["vtk_cell_type"] == vtk_cell_type:
                return name
        return None

    @classmethod
    def from_pyvista(
        cls,
        mesh: pyvista.DataSet,
        *,
        element_type: Optional[str] = None,
        load_properties: bool = True,
    ) -> Mesh:
        r"""
        Create a mesh from a :class:`pyvista.DataSet` with a single cell type
        (:class:`pyvista.UnstructuredGrid`, :class:`pyvista.PolyData`, ...).

        A dataset which is not a :class:`pyvista.UnstructuredGrid` is first converted
        with ``cast_to_unstructured_grid``. The following fields are read:

        - ``mesh.points`` → vertices (always 3D, :math:`E = 3`),
        - the cells → connectivity,
        - ``mesh.point_data`` → vertex properties,
        - ``mesh.cell_data`` → element properties.

        .. seealso::

            - :meth:`to_pyvista` for the reverse operation.
            - `PyVista Documentation <https://docs.pyvista.org>`_.

        Parameters
        ----------
        mesh: :class:`pyvista.DataSet`
            The dataset to read. All its cells must have the same type.

        element_type: Optional[str], optional
            The type of the elements. By default, it is inferred from the VTK cell type
            when this type is implemented (otherwise the mesh has no element type).

        load_properties: bool, optional (default: ``True``)
            If False, the point and cell data are ignored.

        Returns
        -------
        :class:`pysdic.Mesh`
            The new mesh.

        Raises
        ------
        TypeError
            If ``mesh`` is not a :class:`pyvista.DataSet`.

        ValueError
            If the cells do not have exactly one type, or if ``element_type`` does not
            match the cells.

        """
        pyvista = import_optional("pyvista", "Mesh.from_pyvista")

        if not isinstance(mesh, pyvista.DataSet):
            raise TypeError(f"Input 'mesh' must be a pyvista.DataSet. Got type {type(mesh)}.")
        load_properties = _validate_bool(load_properties, "load_properties")
        if not isinstance(mesh, pyvista.UnstructuredGrid):
            mesh = mesh.cast_to_unstructured_grid()

        cells_dict = mesh.cells_dict  # {vtk cell type: (N_e, N_vpe) array}
        if len(cells_dict) != 1:
            raise ValueError(
                f"Input 'mesh' must contain exactly one cell type. Got {len(cells_dict)}."
            )
        vtk_cell_type, elements = next(iter(cells_dict.items()))

        vertex_properties = {}
        element_properties = {}
        if load_properties:
            # numpy.asarray drops the pyvista_ndarray subclass, the arrays are copied by the PropertyContainer
            vertex_properties = {key: numpy.asarray(mesh.point_data[key]) for key in mesh.point_data.keys()}
            element_properties = {key: numpy.asarray(mesh.cell_data[key]) for key in mesh.cell_data.keys()}

        if element_type is None:
            element_type = cls._element_type_from_vtk_cell_type(int(vtk_cell_type))

        vertices = PointCloud(numpy.asarray(mesh.points), n_dimensions=3, properties=vertex_properties)
        connectivity = Connectivity(elements, properties=element_properties, element_type=element_type)
        return cls._from_owned(vertices, connectivity)

    def to_pyvista(self, *, save_properties: bool = True) -> pyvista.UnstructuredGrid:
        r"""
        Convert the mesh to a :class:`pyvista.UnstructuredGrid` (only for :math:`E = 3`).

        The element type must be set and the mesh must not be empty. The vertex
        properties are saved as point data and the element properties as cell data.

        .. seealso::

            - :meth:`from_pyvista` for the reverse operation.
            - `PyVista Documentation <https://docs.pyvista.org>`_.

        Parameters
        ----------
        save_properties: bool, optional (default: ``True``)
            If False, the properties are not saved.

        Returns
        -------
        :class:`pyvista.UnstructuredGrid`
            The PyVista representation of the mesh (with copies of the arrays).

        Raises
        ------
        ValueError
            If :math:`E \neq 3`, if the mesh is empty or has no element type.

        """
        pyvista = import_optional("pyvista", "Mesh.to_pyvista")
        save_properties = _validate_bool(save_properties, "save_properties")
        vtk_cell_type = self._check_vtk_export()

        elements = self._connectivity._elements
        cells = numpy.empty((self.n_elements, self.n_vertices_per_element + 1), dtype=numpy.int64)
        cells[:, 0] = self.n_vertices_per_element
        cells[:, 1:] = elements
        cell_types = numpy.full(self.n_elements, vtk_cell_type, dtype=numpy.uint8)

        grid = pyvista.UnstructuredGrid(cells.ravel(), cell_types, self._vertices.to_array())
        if save_properties:
            for key, values in self._vertices.properties.to_dict().items():
                grid.point_data[key] = values
            for key, values in self._connectivity.properties.to_dict().items():
                grid.cell_data[key] = values
        return grid

    @classmethod
    def from_vtk(
        cls,
        dataset: vtkPointSet,
        *,
        element_type: Optional[str] = None,
        load_properties: bool = True,
    ) -> Mesh:
        r"""
        Create a mesh from a VTK dataset with a single cell type (``vtkUnstructuredGrid``,
        ``vtkPolyData``, ..., i.e. any ``vtkPointSet``).

        A dataset which is not a ``vtkUnstructuredGrid`` is first converted with
        ``vtkAppendFilter``. The following fields are read:

        - the points → vertices (always 3D, :math:`E = 3`),
        - the cells → connectivity,
        - the point data → vertex properties,
        - the cell data → element properties.

        .. note::

            Requires VTK 9 or later (new layout of ``vtkCellArray``).

        Parameters
        ----------
        dataset: ``vtkPointSet``
            The VTK dataset to read. All its cells must have the same type.

        element_type: Optional[str], optional
            The type of the elements. By default, it is inferred from the VTK cell type
            when this type is implemented (otherwise the mesh has no element type).

        load_properties: bool, optional (default: ``True``)
            If False, the point and cell data are ignored.

        Returns
        -------
        :class:`pysdic.Mesh`
            The new mesh.

        Raises
        ------
        TypeError
            If ``dataset`` is not a ``vtkPointSet``, or if a data array is not numeric
            (use ``load_properties=False`` to ignore the data).

        ValueError
            If the dataset has no points, if the cells do not have exactly one type,
            if a data array has no name, or if ``element_type`` does not match the
            cells.

        """
        import_optional("vtk", "Mesh.from_vtk")
        from vtkmodules.vtkCommonDataModel import vtkPointSet, vtkUnstructuredGrid
        from vtkmodules.vtkFiltersCore import vtkAppendFilter
        from vtkmodules.util.numpy_support import vtk_to_numpy

        if not isinstance(dataset, vtkPointSet):
            raise TypeError(f"Input 'dataset' must be a vtkPointSet. Got type {type(dataset)}.")
        load_properties = _validate_bool(load_properties, "load_properties")
        if dataset.GetPoints() is None:
            raise ValueError("The dataset does not contain any points.")
        if not isinstance(dataset, vtkUnstructuredGrid):
            append = vtkAppendFilter()
            append.AddInputData(dataset)
            append.Update()
            dataset = append.GetOutput()

        n_elements = dataset.GetNumberOfCells()
        cell_types = (
            numpy.unique(vtk_to_numpy(dataset.GetCellTypesArray())) if n_elements > 0 else numpy.empty(0)
        )
        if cell_types.size != 1:
            raise ValueError(f"Input 'dataset' must contain exactly one cell type. Got {cell_types.size}.")
        cells = dataset.GetCells()
        offsets = vtk_to_numpy(cells.GetOffsetsArray())  # (N_e + 1,)
        n_vertices_per_element = numpy.unique(numpy.diff(offsets))
        if n_vertices_per_element.size != 1:
            raise ValueError("All the cells of 'dataset' must have the same number of vertices.")
        elements = vtk_to_numpy(cells.GetConnectivityArray()).reshape((n_elements, -1))

        vertex_properties = {}
        element_properties = {}
        if load_properties:
            vertex_properties = _vtk_attributes_to_dict(dataset.GetPointData(), "point data")
            element_properties = _vtk_attributes_to_dict(dataset.GetCellData(), "cell data")

        if element_type is None:
            element_type = cls._element_type_from_vtk_cell_type(int(cell_types[0]))

        coordinates = vtk_to_numpy(dataset.GetPoints().GetData())  # copied by PointCloud
        vertices = PointCloud(coordinates, n_dimensions=3, properties=vertex_properties)
        connectivity = Connectivity(elements, properties=element_properties, element_type=element_type)
        return cls._from_owned(vertices, connectivity)

    def to_vtk(self, *, save_properties: bool = True) -> vtkUnstructuredGrid:
        r"""
        Convert the mesh to a ``vtkUnstructuredGrid`` (only for :math:`E = 3`).

        The element type must be set and the mesh must not be empty. The vertex
        properties are saved as point data arrays and the element properties as cell
        data arrays (named after the properties).

        .. note::

            VTK arrays have at most 2 dimensions: a property of shape
            :math:`(N, A, B)` raises a ``ValueError``. Boolean properties are stored as
            ``uint8`` arrays.

        Parameters
        ----------
        save_properties: bool, optional (default: ``True``)
            If False, the properties are not saved.

        Returns
        -------
        ``vtkUnstructuredGrid``
            The VTK representation of the mesh (with copies of the arrays).

        Raises
        ------
        ValueError
            If :math:`E \neq 3`, if the mesh is empty or has no element type, or if a
            property has more than 2 dimensions.

        """
        import_optional("vtk", "Mesh.to_vtk")
        from vtkmodules.vtkCommonCore import vtkPoints
        from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkUnstructuredGrid
        from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, ID_TYPE_CODE

        save_properties = _validate_bool(save_properties, "save_properties")
        vtk_cell_type = self._check_vtk_export()

        point_data = {}
        cell_data = {}
        if save_properties:
            point_data = self._vertices.properties.to_dict()
            cell_data = self._connectivity.properties.to_dict()
            _check_vtk_property_shapes(point_data)
            _check_vtk_property_shapes(cell_data)

        vtk_points = vtkPoints()
        vtk_points.SetData(numpy_to_vtk(numpy.ascontiguousarray(self._vertices._coordinates), deep=True))

        # Legacy layout [N_vpe, i_1, ..., i_N_vpe, N_vpe, ...]
        cells = numpy.empty((self.n_elements, self.n_vertices_per_element + 1), dtype=ID_TYPE_CODE)
        cells[:, 0] = self.n_vertices_per_element
        cells[:, 1:] = self._connectivity._elements
        vtk_cells = vtkCellArray()
        vtk_cells.SetCells(self.n_elements, numpy_to_vtkIdTypeArray(cells.ravel(), deep=True))

        grid = vtkUnstructuredGrid()
        grid.SetPoints(vtk_points)
        grid.SetCells(vtk_cell_type, vtk_cells)
        _dict_to_vtk_attributes(point_data, grid.GetPointData())
        _dict_to_vtk_attributes(cell_data, grid.GetCellData())
        return grid

    @classmethod
    def from_vtk_file(
        cls,
        file_path: Union[str, os.PathLike],
        *,
        element_type: Optional[str] = None,
        load_properties: bool = True,
        use_module: str = "meshio",
    ) -> Mesh:
        r"""
        Create a mesh from a legacy VTK file (``.vtk``) with a single cell type.

        The points are always 3D (:math:`E = 3`).

        .. seealso::

            - :meth:`to_vtk_file` for the reverse operation.

        Parameters
        ----------
        file_path : Union[str, os.PathLike]
            The path to an existing VTK file, with the extension ``.vtk``.

        element_type: Optional[str], optional
            The type of the elements. By default, it is inferred from the cells.

        load_properties: bool, optional (default: ``True``)
            If False, the point and cell data are ignored.

        use_module: str, optional (default: ``"meshio"``)
            The package used to read the file: ``"meshio"``, ``"pyvista"`` or ``"vtk"``
            (see :meth:`from_meshio`, :meth:`from_pyvista` and :meth:`from_vtk`).

        Returns
        -------
        :class:`pysdic.Mesh`
            The new mesh.

        """
        use_module = _validate_choice(use_module, "use_module", ("meshio", "pyvista", "vtk"))
        module = import_optional(use_module, f"Mesh.from_vtk_file(use_module='{use_module}')")
        file_path = _validate_path(file_path, "file_path", must_exist=True, extensions=(".vtk",))

        if use_module == "meshio":
            mesh = module.read(file_path, file_format="vtk")
            return cls.from_meshio(mesh, element_type=element_type, load_properties=load_properties)
        if use_module == "pyvista":
            mesh = module.read(file_path)
            return cls.from_pyvista(mesh, element_type=element_type, load_properties=load_properties)

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
        return cls.from_vtk(dataset, element_type=element_type, load_properties=load_properties)

    def to_vtk_file(
        self,
        file_path: Union[str, os.PathLike],
        *,
        make_dirs: bool = False,
        save_properties: bool = True,
        binary: bool = False,
        use_module: str = "meshio",
    ) -> None:
        r"""
        Save the mesh to a legacy VTK file (``.vtk``, only for :math:`E = 3`).

        The mesh must have an element type and must not be empty.

        .. warning::

            Some property shapes may not be supported by the VTK format.

        Parameters
        ----------
        file_path : Union[str, os.PathLike]
            The path to the output file, with the extension ``.vtk``.

        make_dirs: bool, optional (default: ``False``)
            If True, the missing parent directories are created.

        save_properties: bool, optional (default: ``True``)
            If False, the properties are not saved.

        binary: bool, optional (default: ``False``)
            If True, the file is written in binary format.

        use_module: str, optional (default: ``"meshio"``)
            The package used to write the file: ``"meshio"``, ``"pyvista"`` or ``"vtk"``
            (see :meth:`to_meshio`, :meth:`to_pyvista` and :meth:`to_vtk`).

        Raises
        ------
        ValueError
            If :math:`E \neq 3`, or if the mesh cannot be converted (see
            :meth:`to_meshio`, :meth:`to_pyvista` or :meth:`to_vtk`).

        """
        use_module = _validate_choice(use_module, "use_module", ("meshio", "pyvista", "vtk"))
        module = import_optional(use_module, f"Mesh.to_vtk_file(use_module='{use_module}')")
        file_path = _validate_path(file_path, "file_path", extensions=(".vtk",))
        make_dirs = _validate_bool(make_dirs, "make_dirs")
        binary = _validate_bool(binary, "binary")
        if self.n_dimensions != 3:
            raise ValueError("VTK format only supports meshes with E = 3.")

        # Build the output object before touching the file system
        if use_module == "meshio":
            output = self.to_meshio(save_properties=save_properties)
        elif use_module == "pyvista":
            output = self.to_pyvista(save_properties=save_properties)
        else:
            output = self.to_vtk(save_properties=save_properties)

        if make_dirs:
            directory = os.path.dirname(file_path)
            if directory:  # empty for a file in the current directory
                os.makedirs(directory, exist_ok=True)

        if use_module == "meshio":
            module.write(file_path, output, file_format="vtk", binary=binary)
        elif use_module == "pyvista":
            output.save(file_path, binary=binary)
        else:
            from vtkmodules.vtkIOLegacy import vtkUnstructuredGridWriter
            writer = vtkUnstructuredGridWriter()
            writer.SetFileName(os.fspath(file_path))
            writer.SetInputData(output)
            if binary:
                writer.SetFileTypeToBinary()
            else:
                writer.SetFileTypeToASCII()
            if writer.Write() != 1:
                raise OSError(f"Cannot write the VTK file '{file_path}'.")

    # =======================
    # Components
    # =======================
    @property
    def vertices(self) -> PointCloud:
        r"""
        [Get/Set] The vertices of the mesh, a :class:`pysdic.PointCloud` of shape
        :math:`(N_v, E)`.

        The getter returns the internal point cloud (see the class documentation). The
        setter copies the given point cloud, which must keep :math:`N_v` and
        :math:`E`. The setter is refused if the mesh is locked.

        Parameters
        ----------
        value: Union[:class:`pysdic.PointCloud`, ArrayLike]
            The new vertices, of shape :math:`(N_v, E)`.

        Returns
        -------
        :class:`pysdic.PointCloud`
            The vertices of the mesh.

        """
        return self._vertices

    @vertices.setter
    def vertices(self, value: Union[PointCloud, ArrayLike]) -> None:
        self._check_unlocked("replace the vertices of the mesh")
        value = value.copy() if isinstance(value, PointCloud) else PointCloud(value)
        if value.shape != self._vertices.shape:
            raise ValueError(
                f"Input 'vertices' must have shape {self._vertices.shape}. "
                f"Got {value.shape}."
            )
        self._vertices = value

    @property
    def connectivity(self) -> Connectivity:
        r"""
        [Get/Set] The elements of the mesh, a :class:`pysdic.Connectivity` of shape
        :math:`(N_e, N_{vpe})`.

        The getter returns the internal connectivity (see the class documentation). The
        setter copies the given connectivity, which must keep :math:`N_e` and
        :math:`N_{vpe}` and refer to existing vertices. A connectivity given as an
        array keeps the current element type. The setter is refused if the mesh is
        locked.

        Parameters
        ----------
        value: Union[:class:`pysdic.Connectivity`, ArrayLike]
            The new connectivity, of shape :math:`(N_e, N_{vpe})`.

        Returns
        -------
        :class:`pysdic.Connectivity`
            The connectivity of the mesh.

        """
        return self._connectivity

    @connectivity.setter
    def connectivity(self, value: Union[Connectivity, ArrayLike]) -> None:
        self._check_unlocked("replace the connectivity of the mesh")
        if isinstance(value, Connectivity):
            value = value.copy()
        else:
            value = Connectivity(value, element_type=self.element_type)
        if value.shape != self._connectivity.shape:
            raise ValueError(
                f"Input 'connectivity' must have shape {self._connectivity.shape}. "
                f"Got {value.shape}."
            )
        self._check_indices(value._elements, self.n_vertices)
        self._connectivity = value

    # =======================
    # Fast access properties
    # =======================
    @property
    def coordinates(self) -> numpy.ndarray:
        r"""
        [Get/Set] The coordinates of the vertices, a read-only array of shape
        :math:`(N_v, E)` (shortcut for ``mesh.vertices.coordinates``).
        """
        return self._vertices.coordinates

    @coordinates.setter
    def coordinates(self, value: ArrayLike) -> None:
        self._check_unlocked("set the coordinates of the mesh")
        self._vertices.coordinates = value

    @property
    def elements(self) -> numpy.ndarray:
        r"""
        [Get/Set] The vertex indices of the elements, a read-only array of shape
        :math:`(N_e, N_{vpe})` (shortcut for ``mesh.connectivity.elements``).

        The setter checks that all the indices refer to existing vertices.
        """
        return self._connectivity.elements

    @elements.setter
    def elements(self, value: ArrayLike) -> None:
        self._check_unlocked("set the elements of the mesh")
        elements = Connectivity._as_elements(value, self.n_vertices_per_element)
        self._check_indices(elements, self.n_vertices)
        self._connectivity.elements = elements

    @property
    def n_vertices(self) -> int:
        r"""
        [Get] The number of vertices :math:`N_v`.
        """
        return self._vertices.n_points

    @property
    def n_elements(self) -> int:
        r"""
        [Get] The number of elements :math:`N_e`.
        """
        return self._connectivity.n_elements

    @property
    def n_dimensions(self) -> int:
        r"""
        [Get] The dimension :math:`E` of the space.
        """
        return self._vertices.n_dimensions

    @property
    def n_vertices_per_element(self) -> int:
        r"""
        [Get] The number of vertices per element :math:`N_{vpe}`.
        """
        return self._connectivity.n_vertices_per_element

    @property
    def n_topological_dimensions(self) -> Optional[int]:
        r"""
        [Get] The topological dimension :math:`K` of the elements, or ``None`` if the
        element type is not set.
        """
        return self._connectivity.n_topological_dimensions

    @property
    def element_type(self) -> Optional[str]:
        r"""
        [Get/Set] The type of the elements (shortcut for
        ``mesh.connectivity.element_type``), e.g. ``"triangle_3"`` or
        ``"quadrangle_4"``, or ``None`` if not set.

        See :class:`pysdic.Connectivity` for the implemented types.

        """
        return self._connectivity.element_type

    @element_type.setter
    def element_type(self, value: Optional[str]) -> None:
        self._check_unlocked("set the element type of the mesh")
        self._connectivity.element_type = value

    # =======================
    # Lock
    # =======================
    def _set_locked(self, locked: bool = True, *, message: Optional[str] = None) -> None:
        r"""
        Lock or unlock the mesh, its vertices and its connectivity.

        The lock is propagated to ``vertices`` and ``connectivity`` with the same
        ``message``, so that they cannot be modified through ``mesh.vertices`` or
        ``mesh.connectivity`` either.

        .. important::

            Internal use only (see :class:`pysdic.core.lockable.LockableMixin`).

        Parameters
        ----------
        locked : bool, optional (default: ``True``)
            True to lock the mesh and its components, False to unlock them.

        message : Optional[str], optional (default: ``None``)
            The reason of the lock, added to the errors raised by the locked objects.
            Ignored when ``locked`` is False.

        """
        super()._set_locked(locked, message=message)  # validates the inputs first
        self._vertices._set_locked(locked, message=message)
        self._connectivity._set_locked(locked, message=message)

    def _check_unlocked(self, action: str, *, override_lock: bool = False) -> None:
        # The mesh first (the error names the mesh), then its components
        super()._check_unlocked(action, override_lock=override_lock)
        self._vertices._check_unlocked(action, override_lock=override_lock)
        self._connectivity._check_unlocked(action, override_lock=override_lock)

    # =======================
    # Structure methods
    # =======================
    def copy(self, *, copy_properties: bool = True) -> Mesh:
        r"""
        Create a copy of the mesh (with the same element type). The copy, its vertices
        and its connectivity are always unlocked.

        Parameters
        ----------
        copy_properties: bool, optional (default: ``True``)
            If False, the copy has no vertex and element properties.

        Returns
        -------
        :class:`pysdic.Mesh`
            A new mesh with copies of the vertices, elements and properties.

        """
        return self._from_owned(
            self._vertices.copy(copy_properties=copy_properties),
            self._connectivity.copy(copy_properties=copy_properties),
        )

    def add_connectivity(
        self,
        new_connectivity: Union[Connectivity, ArrayLike],
        *,
        propagate_properties: bool = True,
    ) -> Mesh:
        r"""
        Return a new mesh with additional elements (appended after the current ones).

        The new elements must refer to the existing vertices. See
        :meth:`pysdic.Connectivity.concatenate` for the element type and the
        properties.

        Parameters
        ----------
        new_connectivity: Union[:class:`pysdic.Connectivity`, ArrayLike]
            The new elements, as a :class:`pysdic.Connectivity` or an integer array of
            shape :math:`(R, N_{vpe})`. An array takes the element type of the mesh.

        propagate_properties: bool, optional (default: ``True``)
            If False, the new mesh has no element properties. The vertex properties are
            always copied.

        Returns
        -------
        :class:`pysdic.Mesh`
            A new mesh with :math:`N_e + R` elements.

        Examples
        --------
        .. code-block:: python
            :linenos:

            import numpy
            from pysdic import Mesh

            mesh = Mesh([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], [[0, 1, 2]], element_type="triangle_3")
            mesh.connectivity["material"] = [10.0]

            mesh = mesh.add_connectivity([[1, 3, 2]])
            print(mesh.connectivity["material"])  # [[10.] [nan]]

        """
        if not isinstance(new_connectivity, Connectivity):
            new_connectivity = Connectivity(new_connectivity, element_type=self.element_type)
        combined = self._connectivity.concatenate(
            new_connectivity, propagate_properties=propagate_properties
        )
        return self._from_owned(self._vertices.copy(), combined)

    def add_vertices(
        self,
        new_vertices: Union[PointCloud, ArrayLike],
        *,
        propagate_properties: bool = True,
    ) -> Mesh:
        r"""
        Return a new mesh with additional vertices (appended after the current ones,
        so the current indices are unchanged).

        See :meth:`pysdic.PointCloud.concatenate` for the properties.

        Parameters
        ----------
        new_vertices: Union[:class:`pysdic.PointCloud`, ArrayLike]
            The new vertices, as a :class:`pysdic.PointCloud` or an array of shape
            :math:`(Q, E)`.

        propagate_properties: bool, optional (default: ``True``)
            If False, the new mesh has no vertex properties. The element properties are
            always copied.

        Returns
        -------
        :class:`pysdic.Mesh`
            A new mesh with :math:`N_v + Q` vertices.

        Examples
        --------
        .. code-block:: python
            :linenos:

            import numpy
            from pysdic import Mesh

            mesh = Mesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]], element_type="triangle_3")
            mesh.vertices["u"] = [1.0, 2.0, 3.0]

            mesh = mesh.add_vertices([[1, 1, 0]])
            print(mesh.vertices["u"])  # [[ 1.] [ 2.] [ 3.] [nan]]

        """
        if not isinstance(new_vertices, PointCloud):
            new_vertices = PointCloud(new_vertices)
        combined = self._vertices.concatenate(
            new_vertices, propagate_properties=propagate_properties
        )
        return self._from_owned(combined, self._connectivity.copy())

    # -----------------------
    # Elements
    # -----------------------
    def filter_connectivity(
        self,
        mask: ArrayLike,
        *,
        propagate_properties: bool = True,
    ) -> Mesh:
        r"""
        Keep only the elements selected by a boolean mask. The vertices are unchanged
        (see :meth:`remove_unused_vertices`).

        Parameters
        ----------
        mask: ArrayLike
            A boolean array of shape :math:`(N_e,)`, True for the elements to keep.

        propagate_properties: bool, optional (default: ``True``)
            If False, the new mesh has no element properties.

        Returns
        -------
        :class:`pysdic.Mesh`
            The mesh with the selected elements.

        Examples
        --------
        .. code-block:: python
            :linenos:

            import numpy
            from pysdic import Mesh

            points = numpy.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            elements = numpy.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh = Mesh(points, elements, element_type="triangle_3")

            filtered = mesh.filter_connectivity([True, False, True, False])
            print(filtered.elements)  # [[0 1 2] [0 2 3]]

        """
        connectivity = self._connectivity.filter_elements(
            mask, propagate_properties=propagate_properties
        )
        return self._from_owned(self._vertices.copy(), connectivity)

    def keep_connectivity_by_index(
        self,
        indices: Union[Integral, ArrayLike],
        *,
        propagate_properties: bool = True,
    ) -> Mesh:
        r"""
        Keep only the elements with the given indices (in their original order). The
        vertices are unchanged.

        Parameters
        ----------
        indices: Union[Integral, ArrayLike]
            An index or an integer array of shape :math:`(R,)` with indices in
            :math:`[0, N_e)`.

        propagate_properties: bool, optional (default: ``True``)
            If False, the new mesh has no element properties.

        Returns
        -------
        :class:`pysdic.Mesh`
            The mesh with the selected elements.

        """
        indices = _as_index_array(numpy.atleast_1d(indices), self.n_elements)
        mask = numpy.zeros(self.n_elements, dtype=bool)
        mask[indices] = True
        return self.filter_connectivity(
            mask, propagate_properties=propagate_properties
        )

    def remove_connectivity_by_index(
        self,
        indices: Union[Integral, ArrayLike],
        *,
        propagate_properties: bool = True,
    ) -> Mesh:
        r"""
        Remove the elements with the given indices. The vertices are unchanged.

        Parameters
        ----------
        indices: Union[Integral, ArrayLike]
            An index or an integer array of shape :math:`(R,)` with indices in
            :math:`[0, N_e)`.

        propagate_properties: bool, optional (default: ``True``)
            If False, the new mesh has no element properties.

        Returns
        -------
        :class:`pysdic.Mesh`
            The mesh without the given elements.

        """
        indices = _as_index_array(numpy.atleast_1d(indices), self.n_elements)
        mask = numpy.ones(self.n_elements, dtype=bool)
        mask[indices] = False
        return self.filter_connectivity(
            mask, propagate_properties=propagate_properties
        )

    # -----------------------
    # Vertices
    # -----------------------
    def filter_vertices(
        self,
        mask: ArrayLike,
        *,
        propagate_properties: bool = True,
    ) -> Mesh:
        r"""
        Keep only the vertices selected by a boolean mask.

        The elements using a removed vertex are removed too, and the indices of the
        remaining elements are renumbered.

        Parameters
        ----------
        mask: ArrayLike
            A boolean array of shape :math:`(N_v,)`, True for the vertices to keep.

        propagate_properties: bool, optional (default: ``True``)
            If False, the new mesh has no vertex and element properties.

        Returns
        -------
        :class:`pysdic.Mesh`
            The mesh with the selected vertices.

        Examples
        --------
        .. code-block:: python
            :linenos:

            import numpy
            from pysdic import Mesh

            points = numpy.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            elements = numpy.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh = Mesh(points, elements, element_type="triangle_3")

            filtered = mesh.filter_vertices([True, True, False, True])
            print(filtered.n_vertices)  # 3
            print(filtered.elements)    # [[0 1 2]] (the old element [0, 1, 3])

        """
        keep = _as_bool_mask(mask, self.n_vertices)
        propagate_properties = _validate_bool(propagate_properties, "propagate_properties")

        # Elements using only kept vertices, renumbered with the new vertex indices
        elements = self._connectivity._elements
        keep_elements = keep[elements].all(axis=1)  # (N_e,)
        new_index = numpy.cumsum(keep) - 1  # old index -> new index (for kept vertices)
        new_elements = new_index[elements[keep_elements]]

        # All the inputs are validated: the following operations cannot fail
        connectivity = self._connectivity.filter_elements(
            keep_elements, propagate_properties=propagate_properties
        )
        connectivity.elements = new_elements
        vertices = self._vertices.filter_points(keep, propagate_properties=propagate_properties)
        return self._from_owned(vertices, connectivity)

    def keep_vertices_by_index(
        self,
        indices: Union[Integral, ArrayLike],
        *,
        propagate_properties: bool = True,
    ) -> Mesh:
        r"""
        Keep only the vertices with the given indices (in their original order).

        The elements using a removed vertex are removed too, and the indices of the
        remaining elements are renumbered.

        Parameters
        ----------
        indices: Union[Integral, ArrayLike]
            An index or an integer array of shape :math:`(R,)` with indices in
            :math:`[0, N_v)`.

        propagate_properties: bool, optional (default: ``True``)
            If False, the new mesh has no vertex and element properties.

        Returns
        -------
        :class:`pysdic.Mesh`
            The mesh with the selected vertices.

        """
        indices = _as_index_array(numpy.atleast_1d(indices), self.n_vertices)
        mask = numpy.zeros(self.n_vertices, dtype=bool)
        mask[indices] = True
        return self.filter_vertices(
            mask, propagate_properties=propagate_properties
        )

    def remove_vertices_by_index(
        self,
        indices: Union[Integral, ArrayLike],
        *,
        propagate_properties: bool = True,
    ) -> Mesh:
        r"""
        Remove the vertices with the given indices.

        The elements using a removed vertex are removed too, and the indices of the
        remaining elements are renumbered.

        Parameters
        ----------
        indices: Union[Integral, ArrayLike]
            An index or an integer array of shape :math:`(R,)` with indices in
            :math:`[0, N_v)`. Duplicates are allowed.

        propagate_properties: bool, optional (default: ``True``)
            If False, the new mesh has no vertex and element properties.

        Returns
        -------
        :class:`pysdic.Mesh`
            The mesh without the given vertices.

        Examples
        --------
        .. code-block:: python
            :linenos:

            import numpy
            from pysdic import Mesh

            points = numpy.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            elements = numpy.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh = Mesh(points, elements, element_type="triangle_3")

            filtered = mesh.remove_vertices_by_index(2)
            print(filtered.n_vertices)  # 3
            print(filtered.elements)    # [[0 1 2]] (the old element [0, 1, 3])

        """
        indices = _as_index_array(numpy.atleast_1d(indices), self.n_vertices)
        mask = numpy.ones(self.n_vertices, dtype=bool)
        mask[indices] = False
        return self.filter_vertices(
            mask, propagate_properties=propagate_properties
        )

    def get_used_vertices_mask(self) -> numpy.ndarray:
        r"""
        Get the vertices used by at least one element.

        Returns
        -------
        numpy.ndarray
            A boolean array of shape :math:`(N_v,)`, True for the used vertices.

        Examples
        --------
        .. code-block:: python
            :linenos:

            import numpy
            from pysdic import Mesh

            points = numpy.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
            elements = numpy.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh = Mesh(points, elements, element_type="triangle_3")

            print(mesh.get_used_vertices_mask())  # [ True  True  True  True False]

        """
        used_vertices_mask = numpy.zeros(self.n_vertices, dtype=bool)
        used_vertices_mask[self._connectivity._elements.ravel()] = True
        return used_vertices_mask

    def remove_unused_vertices(
        self, *, propagate_properties: bool = True
    ) -> Mesh:
        r"""
        Remove the vertices that are not used by any element (the indices of the
        elements are renumbered).

        Parameters
        ----------
        propagate_properties: bool, optional (default: ``True``)
            If False, the new mesh has no vertex and element properties.

        Returns
        -------
        :class:`pysdic.Mesh`
            The mesh without unused vertices.

        """
        return self.filter_vertices(
            self.get_used_vertices_mask(),
            propagate_properties=propagate_properties,
        )

    def get_degenerated_elements_mask(self) -> numpy.ndarray:
        r"""
        Get the degenerated elements, i.e. the elements using the same vertex at least
        twice.

        .. note::

            Only repeated indices are detected: an element with distinct but aligned
            (or coincident) vertices is not considered degenerated.

        Returns
        -------
        numpy.ndarray
            A boolean array of shape :math:`(N_e,)`, True for the degenerated elements.

        Examples
        --------
        .. code-block:: python
            :linenos:

            import numpy
            from pysdic import Mesh

            points = numpy.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            elements = numpy.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 2]])
            mesh = Mesh(points, elements, element_type="triangle_3")

            print(mesh.get_degenerated_elements_mask())  # [False False False  True]

        """
        sorted_indices = numpy.sort(self._connectivity._elements, axis=1)
        return numpy.any(sorted_indices[:, 1:] == sorted_indices[:, :-1], axis=1)

    def remove_degenerated_elements(
        self,
        *,
        propagate_properties: bool = True,
    ) -> Mesh:
        r"""
        Remove the degenerated elements (see :meth:`get_degenerated_elements_mask`).
        The vertices are unchanged.

        Parameters
        ----------
        propagate_properties: bool, optional (default: ``True``)
            If False, the new mesh has no element properties.

        Returns
        -------
        :class:`pysdic.Mesh`
            The mesh without degenerated elements.

        """
        return self.filter_connectivity(
            ~self.get_degenerated_elements_mask(),
            propagate_properties=propagate_properties,
        )

    # ===================================
    # Surface meshes
    # ===================================
    @property
    def uvmap(self) -> Optional[numpy.ndarray]:
        r"""
        [Get/Set] The UV coordinates of the vertices of each element (only for surface
        meshes, :math:`K = 2`).

        The UV map is an array of shape :math:`(N_e, 2 N_{vpe})` storing, for each
        element, :math:`(u_1, v_1, u_2, v_2, ..., u_{N_{vpe}}, v_{N_{vpe}})`. It is
        stored as the element property ``"uvmap"``.

        Parameters
        ----------
        value: Optional[ArrayLike]
            An array of shape :math:`(N_e, 2 N_{vpe})` with finite values, or ``None``
            to remove the UV map.

        Returns
        -------
        Optional[numpy.ndarray]
            A read-only array of shape :math:`(N_e, 2 N_{vpe})`, or ``None`` if no UV
            map is set.

        Raises
        ------
        ValueError
            If the element type is set and :math:`K \neq 2`.

        """
        self._check_surface_mesh("Mesh.uvmap", require_element_type=False)
        if "uvmap" not in self._connectivity:
            return None
        uvmap = self._connectivity["uvmap"]
        expected_shape = (self.n_elements, 2 * self.n_vertices_per_element)
        if uvmap.shape != expected_shape:
            raise ValueError(
                f"The 'uvmap' element property must have shape {expected_shape}. "
                f"Got {uvmap.shape}."
            )
        return uvmap

    @uvmap.setter
    def uvmap(self, value: Optional[ArrayLike]) -> None:
        self._check_surface_mesh("Mesh.uvmap", require_element_type=False)
        if value is None:
            if "uvmap" in self._connectivity:
                del self._connectivity["uvmap"]
            return

        value = _as_array(
            value, "uvmap", numpy.float64,
            shape=(self.n_elements, 2 * self.n_vertices_per_element),
            finite=True,
        )
        self._connectivity["uvmap"] = value

    # =======================
    # FEM operations
    # =======================
    def _resolve_values(
        self, values: Union[str, ArrayLike], container: Any, n_rows: int, name: str, location: str
    ) -> numpy.ndarray:
        # Values given by the name of a property of 'container' or as an array of shape (n_rows, ...)
        if isinstance(values, str):
            if values not in container:
                raise ValueError(f"Property '{values}' not found in the {location}.")
            return container[values]
        values = numpy.asarray(values)
        if values.ndim == 0 or values.shape[0] != n_rows:
            raise ValueError(
                f"Input '{name}' must be the name of a property of the {location} or an "
                f"array of shape ({n_rows}, ...). Got shape {values.shape}."
            )
        return values

    def _fem_inputs(self, integration_points: IntegrationPoints) -> Tuple[str, numpy.ndarray, numpy.ndarray]:
        # Check the integration points and return (element_type, natural_coordinates, element_indices)
        self.validate_integration_points(integration_points)
        return (
            self.element_type,
            integration_points._natural_coordinates,
            integration_points._element_indices,
        )

    def compute_shape_functions(
        self,
        integration_points: IntegrationPoints,
        *,
        return_derivatives: bool = False,
        default: Optional[Real] = 0.0,
        tolerance: Real = 0.0,
    ) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
        r"""
        Compute the shape functions of the elements of the mesh at integration points.

        Shortcut for :func:`pysdic.fem.compute_shape_functions` with the element type of
        the mesh and the natural coordinates of ``integration_points`` (see this function
        for the conventions on ``default`` and ``tolerance``).

        Parameters
        ----------
        integration_points: :class:`pysdic.IntegrationPoints`
            The integration points, consistent with the mesh (see
            :meth:`validate_integration_points`).

        return_derivatives: bool, optional (default: ``False``)
            If True, the derivatives of the shape functions with respect to the natural
            coordinates are also returned.

        default: Optional[Real], optional (default: ``0.0``)
            The value of the shape functions at the points outside the reference element.

        tolerance: Real, optional (default: ``0.0``)
            The tolerance to decide if a point is inside the reference element.

        Returns
        -------
        shape_functions: numpy.ndarray
            An array of shape :math:`(N_p, N_{vpe})`.

        shape_function_derivatives: numpy.ndarray
            Only returned if ``return_derivatives`` is True. An array of shape
            :math:`(N_p, N_{vpe}, K)`.

        Examples
        --------
        .. code-block:: python
            :linenos:

            from pysdic import Mesh, IntegrationPoints

            mesh = Mesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]], element_type="triangle_3")
            integration_points = IntegrationPoints.from_gauss_points("triangle_3", mesh.n_elements)

            print(mesh.compute_shape_functions(integration_points))  # [[0.33333333 0.33333333 0.33333333]]

        """
        element_type, natural_coordinates, _ = self._fem_inputs(integration_points)
        return fem.compute_shape_functions(
            natural_coordinates,
            element_type,
            return_derivatives=return_derivatives,
            default=default,
            tolerance=tolerance,
        )

    def compute_property_interpolation(
        self,
        values: Union[str, ArrayLike],
        integration_points: IntegrationPoints,
        *,
        allow_unlocated_points: bool = False,
        tolerance: Real = 0.0,
        default: Union[Real, ArrayLike] = numpy.nan,
    ) -> numpy.ndarray:
        r"""
        Interpolate a property defined at the vertices of the mesh at integration points.

        Shortcut for :func:`pysdic.fem.compute_property_interpolation` with the
        connectivity and element type of the mesh (see this function for the
        interpolation formula).

        Parameters
        ----------
        values: Union[str, ArrayLike]
            The property at the vertices: the name of a vertex property (see
            ``mesh.vertices``) or an array of shape :math:`(N_v, \dots)`.

        integration_points: :class:`pysdic.IntegrationPoints`
            The integration points, consistent with the mesh (see
            :meth:`validate_integration_points`).

        allow_unlocated_points: bool, optional (default: ``False``)
            If True, the unlocated points (element index ``-1``) get the value
            ``default``. Otherwise, they raise an error.

        tolerance: Real, optional (default: ``0.0``)
            The tolerance to decide if a point is inside the reference element.

        default: Union[Real, ArrayLike], optional (default: ``numpy.nan``)
            The value of the unlocated points: a scalar or an array of :math:`P` values.

        Returns
        -------
        numpy.ndarray
            An array of shape :math:`(N_p, P)` with the interpolated property.

        Examples
        --------
        Interpolate the coordinates of the vertices to get the global coordinates of
        the Gauss points.

        .. code-block:: python
            :linenos:

            from pysdic import Mesh, IntegrationPoints

            mesh = Mesh([[0, 0, 0], [3, 0, 0], [0, 3, 0]], [[0, 1, 2]], element_type="triangle_3")
            integration_points = IntegrationPoints.from_gauss_points("triangle_3", mesh.n_elements)

            print(mesh.compute_property_interpolation(mesh.coordinates, integration_points))  # [[1. 1. 0.]]

        """
        element_type, natural_coordinates, element_indices = self._fem_inputs(integration_points)
        values = self._resolve_values(values, self._vertices, self.n_vertices, "values", "vertices")
        return fem.compute_property_interpolation(
            values,
            self._connectivity._elements,
            element_type,
            natural_coordinates,
            element_indices,
            allow_unlocated_points=allow_unlocated_points,
            tolerance=tolerance,
            default=default,
        )

    def compute_property_derivative(
        self,
        values: Union[str, ArrayLike],
        integration_points: IntegrationPoints,
        *,
        allow_unlocated_points: bool = False,
        tolerance: Real = 0.0,
        default: Real = numpy.nan,
    ) -> numpy.ndarray:
        r"""
        Compute the derivatives of a property defined at the vertices of the mesh with
        respect to the global coordinates, at integration points.

        Shortcut for :func:`pysdic.fem.compute_property_derivative` with the coordinates,
        connectivity and element type of the mesh (see this function for the shape of
        the output and the treatment of degenerated elements).

        Parameters
        ----------
        values: Union[str, ArrayLike]
            The property at the vertices: the name of a vertex property (see
            ``mesh.vertices``) or an array of shape :math:`(N_v, \dots)`.

        integration_points: :class:`pysdic.IntegrationPoints`
            The integration points, consistent with the mesh (see
            :meth:`validate_integration_points`).

        allow_unlocated_points: bool, optional (default: ``False``)
            If True, the unlocated points (element index ``-1``) get the value
            ``default``. Otherwise, they raise an error.

        tolerance: Real, optional (default: ``0.0``)
            The tolerance to decide if a point is inside the reference element.

        default: Real, optional (default: ``numpy.nan``)
            The value of the unlocated points.

        Returns
        -------
        numpy.ndarray
            The derivatives of the property at the integration points (see
            :func:`pysdic.fem.compute_property_derivative`).

        """
        element_type, natural_coordinates, element_indices = self._fem_inputs(integration_points)
        values = self._resolve_values(values, self._vertices, self.n_vertices, "values", "vertices")
        return fem.compute_property_derivative(
            values,
            self._vertices._coordinates,
            self._connectivity._elements,
            element_type,
            natural_coordinates,
            element_indices,
            allow_unlocated_points=allow_unlocated_points,
            tolerance=tolerance,
            default=default,
        )

    def compute_property_projection(
        self,
        values: Union[str, ArrayLike],
        integration_points: IntegrationPoints,
        *,
        sparse: bool = False,
        allow_unlocated_points: bool = False,
        tolerance: Real = 0.0,
        default: Real = numpy.nan,
        return_unaffected: bool = False,
    ) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
        r"""
        Project a property defined at integration points to the vertices of the mesh.

        Shortcut for :func:`pysdic.fem.compute_property_projection` with the connectivity
        and element type of the mesh (see this function for the projection formula).

        .. important::

            The weights of the integration points (``integration_points.weights``) are
            used as the weights of the projection. Set ``integration_points.weights =
            None`` for a projection with equal weights.

        Parameters
        ----------
        values: Union[str, ArrayLike]
            The property at the integration points: the name of a property of
            ``integration_points`` or an array of shape :math:`(N_p, \dots)`.

        integration_points: :class:`pysdic.IntegrationPoints`
            The integration points, consistent with the mesh (see
            :meth:`validate_integration_points`).

        sparse: bool, optional (default: ``False``)
            If True, the linear system of the projection is solved with sparse matrices.

        allow_unlocated_points: bool, optional (default: ``False``)
            If True, the unlocated points (element index ``-1``) are ignored. Otherwise,
            they raise an error.

        tolerance: Real, optional (default: ``0.0``)
            The tolerance to decide if a point is inside the reference element.

        default: Real, optional (default: ``numpy.nan``)
            The value of the vertices that are not affected by any point.

        return_unaffected: bool, optional (default: ``False``)
            If True, the mask of the unaffected vertices is also returned.

        Returns
        -------
        projected_values: numpy.ndarray
            An array of shape :math:`(N_v, P)` with the projected property.

        unaffected_mask: numpy.ndarray
            Only returned if ``return_unaffected`` is True. A boolean array of shape
            :math:`(N_v,)`, True for the vertices set to ``default``.

        Examples
        --------
        Interpolate a property at the 4 Gauss points of a quadrangle and project it back
        to the vertices (the property is recovered).

        .. code-block:: python
            :linenos:

            import numpy
            from pysdic import Mesh, IntegrationPoints

            mesh = Mesh([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], [[0, 1, 2, 3]], element_type="quadrangle_4")
            mesh.vertices["u"] = [0.0, 1.0, 2.0, 3.0]

            integration_points = IntegrationPoints.from_gauss_points("quadrangle_4", mesh.n_elements)
            integration_points["u"] = mesh.compute_property_interpolation("u", integration_points)

            projected = mesh.compute_property_projection("u", integration_points)
            print(numpy.round(projected.ravel(), 6))  # [0. 1. 2. 3.]

        """
        _, natural_coordinates, element_indices = self._fem_inputs(integration_points)
        values = self._resolve_values(
            values, integration_points, integration_points.n_points, "values", "integration points"
        )
        return fem.compute_property_projection(
            values,
            self._connectivity._elements,
            self.element_type,
            natural_coordinates,
            element_indices,
            integration_points._weights,
            n_vertices=self.n_vertices,
            sparse=sparse,
            allow_unlocated_points=allow_unlocated_points,
            tolerance=tolerance,
            default=default,
            return_unaffected=return_unaffected,
        )

    def compute_integration_points_coordinates(
        self,
        integration_points: IntegrationPoints,
        *,
        allow_unlocated_points: bool = False,
        tolerance: Real = 0.0,
        propagate_properties: bool = True,
    ) -> PointCloud:
        r"""
        Compute the global coordinates of integration points.

        The coordinates of the vertices are interpolated at the integration points (see
        :meth:`compute_property_interpolation`):

        .. math::

            X = \sum_{i=1}^{N_{vpe}} N_i(\Xi) X_i

        Parameters
        ----------
        integration_points: :class:`pysdic.IntegrationPoints`
            The integration points, consistent with the mesh (see
            :meth:`validate_integration_points`).

        allow_unlocated_points: bool, optional (default: ``False``)
            If True, the unlocated points (element index ``-1``) get ``numpy.nan``
            coordinates. Otherwise, they raise an error.

        tolerance: Real, optional (default: ``0.0``)
            The tolerance to decide if a point is inside the reference element.

        propagate_properties: bool, optional (default: ``True``)
            If True, the properties of the integration points are copied to the point
            cloud.

        Returns
        -------
        :class:`pysdic.PointCloud`
            A point cloud of shape :math:`(N_p, E)`, with the points in the same order
            as ``integration_points``.

        Examples
        --------
        .. code-block:: python
            :linenos:

            from pysdic import Mesh, IntegrationPoints

            mesh = Mesh([[0, 0, 0], [3, 0, 0], [0, 3, 0]], [[0, 1, 2]], element_type="triangle_3")
            integration_points = IntegrationPoints.from_gauss_points("triangle_3", mesh.n_elements)

            point_cloud = mesh.compute_integration_points_coordinates(integration_points)
            print(point_cloud.coordinates)  # [[1. 1. 0.]]

        """
        propagate_properties = _validate_bool(propagate_properties, "propagate_properties")
        coordinates = self.compute_property_interpolation(
            self._vertices._coordinates,
            integration_points,
            allow_unlocated_points=allow_unlocated_points,
            tolerance=tolerance,
            default=numpy.nan,
        )
        properties = integration_points.properties.copy() if propagate_properties else None
        return PointCloud(coordinates, n_dimensions=self.n_dimensions, properties=properties)

    def compute_shape_function_matrix(
        self,
        integration_points: IntegrationPoints,
        *,
        sparse: bool = False,
        sparse_format: str = "csr",
        allow_unlocated_points: bool = False,
        tolerance: Real = 0.0,
    ) -> Union[numpy.ndarray, scipy.sparse.csr_matrix, scipy.sparse.coo_matrix]:
        r"""
        Compute the shape function matrix :math:`\mathcal{N}` of the mesh at integration
        points, such that the values of a property at the points are
        :math:`\mathcal{N} U`, with :math:`U` the values at the vertices.

        Shortcut for :func:`pysdic.fem.compute_shape_function_matrix` with the
        connectivity, element type and number of vertices of the mesh.

        Parameters
        ----------
        integration_points: :class:`pysdic.IntegrationPoints`
            The integration points, consistent with the mesh (see
            :meth:`validate_integration_points`).

        sparse: bool, optional (default: ``False``)
            If True, a sparse matrix is returned.

        sparse_format: str, optional (default: ``"csr"``)
            The format of the sparse matrix: ``"csr"`` or ``"coo"``.

        allow_unlocated_points: bool, optional (default: ``False``)
            If True, the rows of the unlocated points (element index ``-1``) are zero.
            Otherwise, they raise an error.

        tolerance: Real, optional (default: ``0.0``)
            The tolerance to decide if a point is inside the reference element.

        Returns
        -------
        Union[numpy.ndarray, scipy.sparse.csr_matrix, scipy.sparse.coo_matrix]
            A matrix of shape :math:`(N_p, N_v)`.

        Examples
        --------
        .. code-block:: python
            :linenos:

            from pysdic import Mesh, IntegrationPoints

            mesh = Mesh([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], [[0, 1, 2], [1, 3, 2]], element_type="triangle_3")
            integration_points = IntegrationPoints.from_gauss_points("triangle_3", mesh.n_elements)

            print(mesh.compute_shape_function_matrix(integration_points))
            # [[0.33333333 0.33333333 0.33333333 0.        ]
            #  [0.         0.33333333 0.33333333 0.33333333]]

        """
        element_type, natural_coordinates, element_indices = self._fem_inputs(integration_points)
        return fem.compute_shape_function_matrix(
            self._connectivity._elements,
            element_type,
            natural_coordinates,
            element_indices,
            n_vertices=self.n_vertices,
            sparse=sparse,
            sparse_format=sparse_format,
            allow_unlocated_points=allow_unlocated_points,
            tolerance=tolerance,
        )

    def compute_jacobian_matrix(
        self,
        integration_points: IntegrationPoints,
        *,
        allow_unlocated_points: bool = False,
        tolerance: Real = 0.0,
        default: Real = numpy.nan,
    ) -> numpy.ndarray:
        r"""
        Compute the Jacobian matrices :math:`\frac{\partial X}{\partial \Xi}` of the
        mapping from the reference element to the mesh, at integration points.

        Shortcut for :func:`pysdic.fem.compute_jacobian_matrix` with the coordinates,
        connectivity and element type of the mesh.

        Parameters
        ----------
        integration_points: :class:`pysdic.IntegrationPoints`
            The integration points, consistent with the mesh (see
            :meth:`validate_integration_points`).

        allow_unlocated_points: bool, optional (default: ``False``)
            If True, the unlocated points (element index ``-1``) get the value
            ``default``. Otherwise, they raise an error.

        tolerance: Real, optional (default: ``0.0``)
            The tolerance to decide if a point is inside the reference element.

        default: Real, optional (default: ``numpy.nan``)
            The value of the unlocated points.

        Returns
        -------
        numpy.ndarray
            An array of shape :math:`(N_p, E, K)` with the Jacobian matrix of each point.

        Examples
        --------
        .. code-block:: python
            :linenos:

            from pysdic import Mesh, IntegrationPoints

            mesh = Mesh([[0, 0, 0], [2, 0, 0], [0, 3, 0]], [[0, 1, 2]], element_type="triangle_3")
            integration_points = IntegrationPoints.from_gauss_points("triangle_3", mesh.n_elements)

            print(mesh.compute_jacobian_matrix(integration_points)[0])
            # [[2. 0.]
            #  [0. 3.]
            #  [0. 0.]]

        """
        element_type, natural_coordinates, element_indices = self._fem_inputs(integration_points)
        return fem.compute_jacobian_matrix(
            self._vertices._coordinates,
            self._connectivity._elements,
            element_type,
            natural_coordinates,
            element_indices,
            allow_unlocated_points=allow_unlocated_points,
            tolerance=tolerance,
            default=default,
        )

    def assemble_vertices_adjacency_graph(self) -> scipy.sparse.csr_matrix:
        r"""
        Assemble the vertices adjacency graph of the mesh: two vertices are neighbors if
        they belong to the same element.

        Shortcut for :func:`pysdic.fem.assemble_vertices_adjacency_graph` with the
        connectivity and the number of vertices of the mesh (the vertices used by no
        element have no neighbors).

        Returns
        -------
        scipy.sparse.csr_matrix
            A symmetric boolean sparse matrix of shape :math:`(N_v, N_v)` with an empty
            diagonal. The neighbors of vertex :math:`i` are ``graph[i].indices``.

        """
        return fem.assemble_vertices_adjacency_graph(
            self._connectivity._elements, n_vertices=self.n_vertices
        )

    def assemble_elements_adjacency_graph(self) -> scipy.sparse.csr_matrix:
        r"""
        Assemble the elements adjacency graph of the mesh: two elements are neighbors if
        they share at least one vertex.

        Shortcut for :func:`pysdic.fem.assemble_elements_adjacency_graph` with the
        connectivity of the mesh.

        Returns
        -------
        scipy.sparse.csr_matrix
            A symmetric boolean sparse matrix of shape :math:`(N_e, N_e)` with an empty
            diagonal. The neighbors of element :math:`i` are ``graph[i].indices``.

        """
        return fem.assemble_elements_adjacency_graph(self._connectivity._elements)

    def compute_elements_centers(self) -> numpy.ndarray:
        r"""
        Compute the center of each element of the mesh, the image of the center of the
        reference element.

        Shortcut for :func:`pysdic.fem.compute_elements_centers` with the coordinates,
        connectivity and element type of the mesh (any element type).

        Returns
        -------
        numpy.ndarray
            An array of shape :math:`(N_e, E)` with the center of each element.

        Examples
        --------
        .. code-block:: python
            :linenos:

            from pysdic import Mesh

            mesh = Mesh([[0, 0, 0], [3, 0, 0], [0, 3, 0]], [[0, 1, 2]], element_type="triangle_3")
            print(mesh.compute_elements_centers())  # [[1. 1. 0.]]

        """
        if self.element_type is None:
            raise ValueError(
                "'Mesh.compute_elements_centers' requires the element type of the mesh. "
                "Set 'mesh.element_type' first."
            )
        return fem.compute_elements_centers(
            self._vertices._coordinates, self._connectivity._elements, self.element_type
        )

    def compute_elements_areas(self) -> numpy.ndarray:
        r"""
        Compute the area of each element of a surface mesh (:math:`K = 2`).

        Shortcut for :func:`pysdic.fem.compute_elements_areas` with the coordinates,
        connectivity and element type of the mesh (see this function for the
        quadrature).

        Returns
        -------
        numpy.ndarray
            An array of shape :math:`(N_e,)` with the area of each element.

        Examples
        --------
        .. code-block:: python
            :linenos:

            from pysdic import Mesh

            mesh = Mesh([[0, 0, 0], [2, 0, 0], [2, 3, 0], [0, 3, 0]], [[0, 1, 2, 3]], element_type="quadrangle_4")
            print(mesh.compute_elements_areas())  # [6.]

        """
        element_type = self._check_surface_mesh("Mesh.compute_elements_areas")
        return fem.compute_elements_areas(
            self._vertices._coordinates, self._connectivity._elements, element_type
        )

    def compute_elements_normals(self) -> numpy.ndarray:
        r"""
        Compute the unit normal of each element of a 3D surface mesh (:math:`K = 2`,
        :math:`E = 3`), at the center of the element.

        Shortcut for :func:`pysdic.fem.compute_elements_normals` with the coordinates,
        connectivity and element type of the mesh.

        Returns
        -------
        numpy.ndarray
            An array of shape :math:`(N_e, 3)` with the unit normal of each element (NaN
            for the degenerated elements).

        """
        element_type = self._check_surface_mesh("Mesh.compute_elements_normals", require_3d=True)
        return fem.compute_elements_normals(
            self._vertices._coordinates, self._connectivity._elements, element_type
        )

    def compute_vertices_normals(self) -> numpy.ndarray:
        r"""
        Compute the unit normal at each vertex of a 3D surface mesh (:math:`K = 2`,
        :math:`E = 3`), as the mean of the normals of the elements weighted by their
        areas.

        Shortcut for :func:`pysdic.fem.compute_vertices_normals` with the coordinates,
        connectivity and element type of the mesh.

        Returns
        -------
        numpy.ndarray
            An array of shape :math:`(N_v, 3)` with the unit normal at each vertex (NaN
            for the vertices used by no valid element).

        """
        element_type = self._check_surface_mesh("Mesh.compute_vertices_normals", require_3d=True)
        return fem.compute_vertices_normals(
            self._vertices._coordinates, self._connectivity._elements, element_type
        )

    def compute_integration_points_normals(
        self,
        integration_points: IntegrationPoints,
        *,
        allow_unlocated_points: bool = False,
        tolerance: Real = 0.0,
    ) -> numpy.ndarray:
        r"""
        Compute the unit normal of a 3D surface mesh (:math:`K = 2`, :math:`E = 3`) at
        integration points.

        Shortcut for :func:`pysdic.fem.compute_integration_points_normals` with the
        coordinates, connectivity and element type of the mesh.

        Parameters
        ----------
        integration_points: :class:`pysdic.IntegrationPoints`
            The integration points, consistent with the mesh (see
            :meth:`validate_integration_points`).

        allow_unlocated_points: bool, optional (default: ``False``)
            If True, the unlocated points (element index ``-1``) get NaN normals.
            Otherwise, they raise an error.

        tolerance: Real, optional (default: ``0.0``)
            The tolerance to decide if a point is inside the reference element.

        Returns
        -------
        numpy.ndarray
            An array of shape :math:`(N_p, 3)` with the unit normals.

        """
        self._check_surface_mesh("Mesh.compute_integration_points_normals", require_3d=True)
        element_type, natural_coordinates, element_indices = self._fem_inputs(integration_points)
        return fem.compute_integration_points_normals(
            self._vertices._coordinates,
            self._connectivity._elements,
            element_type,
            natural_coordinates,
            element_indices,
            allow_unlocated_points=allow_unlocated_points,
            tolerance=tolerance,
        )

    def compute_strains(
        self,
        displacements: Union[str, ArrayLike],
        integration_points: IntegrationPoints,
        *,
        allow_unlocated_points: bool = False,
        tolerance: Real = 0.0,
        default: Real = numpy.nan,
    ) -> numpy.ndarray:
        r"""
        Compute the linear strain tensor of a displacement field at integration points.

        Shortcut for :func:`pysdic.fem.compute_strains` with the coordinates,
        connectivity and element type of the mesh.

        Parameters
        ----------
        displacements: Union[str, ArrayLike]
            The displacements of the vertices: the name of a vertex property (see
            ``mesh.vertices``) or an array of shape :math:`(N_v, E)`.

        integration_points: :class:`pysdic.IntegrationPoints`
            The integration points, consistent with the mesh (see
            :meth:`validate_integration_points`).

        allow_unlocated_points: bool, optional (default: ``False``)
            If True, the unlocated points (element index ``-1``) get the value
            ``default``. Otherwise, they raise an error.

        tolerance: Real, optional (default: ``0.0``)
            The tolerance to decide if a point is inside the reference element.

        default: Real, optional (default: ``numpy.nan``)
            The value of the unlocated points.

        Returns
        -------
        numpy.ndarray
            An array of shape :math:`(N_p, E, E)` with the strain tensor at each point.

        """
        element_type, natural_coordinates, element_indices = self._fem_inputs(integration_points)
        displacements = self._resolve_values(
            displacements, self._vertices, self.n_vertices, "displacements", "vertices"
        )
        return fem.compute_strains(
            displacements,
            self._vertices._coordinates,
            self._connectivity._elements,
            element_type,
            natural_coordinates,
            element_indices,
            allow_unlocated_points=allow_unlocated_points,
            tolerance=tolerance,
            default=default,
        )

    def cast_rays(
        self,
        rays: Rays,
        *,
        n_subdivisions: Optional[Integral] = None,
        max_iteration: Integral = 20,
        tolerance: Real = 1e-10,
        retry_neighbors: bool = False,
    ) -> IntegrationPoints:
        r"""
        Compute the first intersection of rays with a 3D surface mesh (:math:`K = 2`,
        :math:`E = 3`).

        Shortcut for :func:`pysdic.fem.cast_rays` with the coordinates, connectivity and
        element type of the mesh (see this function for the method and the parameters).

        The rays are flattened in C order: the ray ``rays.rays[i, j]`` of a batch of shape
        :math:`(H, W)` gives the point ``i * W + j``. Reshape the outputs with
        ``rays.batch_shape``. The rays with non-finite components (e.g. invalid pixels)
        are unlocated.

        .. important::

            Requires the optional package ``open3d`` (``pip install open3d``).

        Parameters
        ----------
        rays: :class:`pysdic.Rays`
            The rays, with any batch shape :math:`(\dots)` (see
            :func:`pysdic.compute_rays`).

        n_subdivisions: Optional[Integral], optional (default: ``None``)
            The number of segments per edge of the reference element for the first
            approximation (ignored for ``triangle_3``).

        max_iteration: Integral, optional (default: ``20``)
            The maximum number of Newton iterations (ignored for ``triangle_3``).

        tolerance: Real, optional (default: ``1e-10``)
            The convergence tolerance on the natural coordinates (ignored for
            ``triangle_3``).

        retry_neighbors: bool, optional (default: ``False``)
            If True, the rays rejected by the Newton iterations are searched again in the
            elements at a distance :math:`\leq 1` of their first element (ignored for
            ``triangle_3``).

        Returns
        -------
        :class:`pysdic.IntegrationPoints`
            One point per ray, :math:`N_p = \prod (\dots)`. The rays without intersection
            are unlocated (element index ``-1``).

        Examples
        --------
        .. code-block:: python
            :linenos:

            from pysdic import Mesh, Rays

            mesh = Mesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]], element_type="triangle_3")
            rays = Rays([[0.2, 0.3, 1.0, 0.0, 0.0, -1.0]])

            integration_points = mesh.cast_rays(rays)
            print(integration_points.natural_coordinates, integration_points.element_indices)  # [[0.2 0.3]] [0]

        """
        element_type = self._check_surface_mesh("Mesh.cast_rays", require_3d=True)
        if not isinstance(rays, Rays):
            raise TypeError(f"Input 'rays' must be a Rays. Got type {type(rays)}.")
        flat_rays = rays.rays.reshape((-1, 6))  # (N_r, 6)
        finite = numpy.all(numpy.isfinite(flat_rays), axis=1)

        natural_coordinates = numpy.full((flat_rays.shape[0], 2), numpy.nan, dtype=numpy.float64)
        element_indices = numpy.full(flat_rays.shape[0], -1, dtype=numpy.int64)
        natural_coordinates[finite], element_indices[finite] = fem.cast_rays(
            self._vertices._coordinates,
            self._connectivity._elements,
            element_type,
            flat_rays[finite, 0:3],
            flat_rays[finite, 3:6],
            n_subdivisions=n_subdivisions,
            max_iteration=max_iteration,
            tolerance=tolerance,
            retry_neighbors=retry_neighbors,
        )
        return IntegrationPoints(natural_coordinates, element_indices, n_topological_dimensions=2)

    def locate_points(
        self,
        points: ArrayLike,
        *,
        n_subdivisions: Optional[Integral] = None,
        max_iteration: Integral = 20,
        tolerance: Real = 1e-10,
    ) -> IntegrationPoints:
        r"""
        Locate points in a 2D surface mesh (:math:`K = 2`, :math:`E = 2`): compute the
        element containing each point and its natural coordinates.

        Shortcut for :func:`pysdic.fem.locate_points` with the coordinates, connectivity and
        element type of the mesh (see this function for the method and the parameters).

        The points are flattened in C order: the point ``points[i, j]`` of a batch of shape
        :math:`(H, W, 2)` gives the integration point ``i * W + j``. Points outside the mesh
        or not finite are unlocated.

        .. seealso::

            - :meth:`cast_rays` to locate the intersections of rays with a 3D surface mesh.

        Parameters
        ----------
        points: ArrayLike
            The coordinates of the points, an array of shape :math:`(\dots, 2)`.

        n_subdivisions: Optional[Integral], optional (default: ``None``)
            The number of segments per edge of the reference element for the first
            approximation (ignored for ``triangle_3``).

        max_iteration: Integral, optional (default: ``20``)
            The maximum number of Newton iterations (ignored for ``triangle_3``).

        tolerance: Real, optional (default: ``1e-10``)
            The tolerance on the natural coordinates (convergence of the Newton iterations
            and membership of the elements).

        Returns
        -------
        :class:`pysdic.IntegrationPoints`
            One point per input point, :math:`N_p = \prod (\dots)`. The points outside the
            mesh are unlocated (element index ``-1``).

        Raises
        ------
        ValueError
            If the mesh is not a 2D surface mesh with an element type, or ``points`` does
            not have a last dimension of size 2.

        Examples
        --------
        .. code-block:: python
            :linenos:

            from pysdic import Mesh

            mesh = Mesh([[0, 0], [1, 0], [0, 1]], [[0, 1, 2]], element_type="triangle_3")

            integration_points = mesh.locate_points([[0.2, 0.3], [1.0, 1.0]])
            print(integration_points.natural_coordinates, integration_points.element_indices)
            # [[0.2 0.3] [nan nan]] [ 0 -1]

        """
        element_type = self._check_surface_mesh("Mesh.locate_points")
        if self.n_dimensions != 2:
            raise ValueError(
                f"'Mesh.locate_points' is only available for surface meshes in 2D space (E = 2). "
                f"Got E={self.n_dimensions}. Use 'Mesh.cast_rays' for a 3D surface mesh."
            )
        points = numpy.asarray(points, dtype=numpy.float64)
        if points.ndim == 0 or points.shape[-1] != 2:
            raise ValueError(f"Input 'points' must have shape (..., 2). Got shape {points.shape}.")
        natural_coordinates, element_indices = fem.locate_points(
            self._vertices._coordinates,
            self._connectivity._elements,
            element_type,
            points.reshape((-1, 2)),
            n_subdivisions=n_subdivisions,
            max_iteration=max_iteration,
            tolerance=tolerance,
        )
        return IntegrationPoints(natural_coordinates, element_indices, n_topological_dimensions=2)

    def compute_vertices_neighborhood(
        self, *, max_distance: Integral = 1, self_neighborhood: bool = True
    ) -> scipy.sparse.csr_matrix:
        r"""
        Compute the neighborhood of each vertex of the mesh, i.e. the vertices within a
        given distance (two vertices are at distance 1 if they belong to the same
        element).

        Shortcut for :func:`pysdic.fem.compute_vertices_neighborhood` with the
        connectivity and the number of vertices of the mesh (the vertices used by no
        element have an empty neighborhood, or only themselves).

        Parameters
        ----------
        max_distance: Integral, optional (default: ``1``)
            The maximum distance of the vertices in the neighborhood.

        self_neighborhood: bool, optional (default: ``True``)
            If True, the neighborhood of each vertex includes the vertex itself.

        Returns
        -------
        scipy.sparse.csr_matrix
            A symmetric boolean sparse matrix of shape :math:`(N_v, N_v)`, where the
            neighborhood of vertex :math:`i` is ``neighborhood[i].indices`` (sorted).

        """
        return fem.compute_vertices_neighborhood(
            self._connectivity._elements,
            n_vertices=self.n_vertices,
            max_distance=max_distance,
            self_neighborhood=self_neighborhood,
        )

    def compute_elements_neighborhood(
        self, *, max_distance: Integral = 1, self_neighborhood: bool = True
    ) -> scipy.sparse.csr_matrix:
        r"""
        Compute the neighborhood of each element of the mesh, i.e. the elements within a
        given distance (two elements are at distance 1 if they share a vertex).

        Shortcut for :func:`pysdic.fem.compute_elements_neighborhood` with the
        connectivity of the mesh.

        Parameters
        ----------
        max_distance: Integral, optional (default: ``1``)
            The maximum distance of the elements in the neighborhood.

        self_neighborhood: bool, optional (default: ``True``)
            If True, the neighborhood of each element includes the element itself.

        Returns
        -------
        scipy.sparse.csr_matrix
            A symmetric boolean sparse matrix of shape :math:`(N_e, N_e)`, where the
            neighborhood of element :math:`i` is ``neighborhood[i].indices`` (sorted).

        """
        return fem.compute_elements_neighborhood(
            self._connectivity._elements,
            max_distance=max_distance,
            self_neighborhood=self_neighborhood,
        )

    def compute_integration_points_neighborhood(
        self,
        integration_points: IntegrationPoints,
        *,
        max_distance: Integral = 1,
        self_neighborhood: bool = True,
        skip_m1: bool = True,
    ) -> scipy.sparse.csr_matrix:
        r"""
        Compute, for each element of the mesh, the integration points located in the
        elements of its neighborhood.

        Shortcut for :func:`pysdic.fem.compute_integration_points_neighborhood` with the
        connectivity of the mesh.

        Parameters
        ----------
        integration_points: :class:`pysdic.IntegrationPoints`
            The integration points, consistent with the mesh (see
            :meth:`validate_integration_points`).

        max_distance: Integral, optional (default: ``1``)
            The maximum distance of the elements in the neighborhood.

        self_neighborhood: bool, optional (default: ``True``)
            If True, the neighborhood of each element includes the element itself (and
            so its own integration points).

        skip_m1: bool, optional (default: ``True``)
            If True, the unlocated points (element index ``-1``) belong to no
            neighborhood. Otherwise, they raise an ``IndexError``.

        Returns
        -------
        scipy.sparse.csr_matrix
            A boolean sparse matrix of shape :math:`(N_e, N_p)`, where the integration
            points in the neighborhood of element :math:`i` are
            ``points_neighborhood[i].indices`` (sorted).

        """
        _, _, element_indices = self._fem_inputs(integration_points)
        return fem.compute_integration_points_neighborhood(
            self._connectivity._elements,
            element_indices,
            max_distance=max_distance,
            self_neighborhood=self_neighborhood,
            skip_m1=skip_m1,
        )

    # =======================
    # Visualization Methods
    # =======================
    def quick_visualize(
        self,
        *,
        warp_by: Optional[Union[str, ArrayLike]] = None,
        vertex_color: Optional[Union[str, Tuple[Real, Real, Real]]] = None,
        vertex_size: Optional[Real] = None,
        vertex_opacity: Optional[Real] = None,
        vertex_mask: Optional[ArrayLike] = None,
        show_vertices: bool = True,
        edge_color: Optional[Union[str, Tuple[Real, Real, Real]]] = None,
        edge_width: Optional[Real] = None,
        edge_opacity: Optional[Real] = None,
        show_edges: bool = True,
        face_color: Optional[Union[str, Tuple[Real, Real, Real]]] = None,
        face_opacity: Optional[Real] = None,
        face_mask: Optional[ArrayLike] = None,
        show_faces: bool = True,
        vertex_property: Optional[Union[str, ArrayLike]] = None,
        element_property: Optional[Union[str, ArrayLike]] = None,
        property_axis: Optional[Integral] = None,
        property_clim: Optional[Tuple[Real, Real]] = None,
        property_cmap: Optional[Union[str, matplotlib.colors.Colormap]] = None,
        property_below_color: Optional[Union[str, Tuple[Real, Real, Real]]] = None,
        property_above_color: Optional[Union[str, Tuple[Real, Real, Real]]] = None,
        texture: Optional[ArrayLike] = None,
        texture_rgb: Optional[bool] = True,
        title: Optional[str] = None,
        show_bar: bool = True,
        show_bounds: bool = True,
        show_vertices_index: bool = False,
        show_elements_index: bool = False,
        index_font_size: Real = 10,
        screenshot: Optional[str] = None,
    ) -> None:
        r"""
        Visualize the mesh with PyVista (only for :math:`E \leq 3`).

        Only the finite vertices are displayed.

        .. seealso::

            - :class:`pysdic.Plotter3D` for more control over the visualization.
            - `PyVista Documentation <https://docs.pyvista.org>`_.

        Parameters
        ----------
        warp_by: Optional[Union[str, ArrayLike]], optional
            The vertex property used to warp the mesh: the name of a vertex property, or
            an array of shape :math:`(N_v,)` or :math:`(N_v, A)`.

        vertex_color: Union[str, Tuple[Real, Real, Real]], optional
            The color of the vertices: a color name (e.g. ``"red"``), an RGB tuple or a
            hex string. Ignored if ``vertex_property`` is given.

        vertex_size: Real, optional
            The size of the vertices (positive). PyVista default: ``5.0``.

        vertex_opacity: Real, optional
            The opacity of the vertices, between ``0.0`` and ``1.0``. PyVista default:
            ``1.0``.

        vertex_mask: Optional[ArrayLike], optional
            A boolean array of shape :math:`(N_v,)`, True for the vertices to display.
            By default, all the finite vertices are displayed.

        show_vertices: bool, optional (default: ``True``)
            Whether to show the vertices.

        edge_color: Union[str, Tuple[Real, Real, Real]], optional
            The color of the edges.

        edge_width: Real, optional
            The width of the edges (positive). PyVista default: ``1.0``.

        edge_opacity: Real, optional
            The opacity of the edges, between ``0.0`` and ``1.0``. PyVista default:
            ``1.0``.

        show_edges: bool, optional (default: ``True``)
            Whether to show the edges.

        face_color: Union[str, Tuple[Real, Real, Real]], optional
            The color of the faces (elements). Ignored if ``element_property`` is
            given.

        face_opacity: Real, optional
            The opacity of the faces, between ``0.0`` and ``1.0``. PyVista default:
            ``1.0``.

        face_mask: Optional[ArrayLike], optional
            A boolean array of shape :math:`(N_e,)`, True for the faces to display.
            By default, all the faces are displayed.

        show_faces: bool, optional (default: ``True``)
            Whether to show the faces.

        vertex_property: Union[str, ArrayLike], optional
            The property used to color the vertices: the name of a vertex property, or
            an array of shape :math:`(N_v,)` or :math:`(N_v, A)`.

        element_property: Union[str, ArrayLike], optional
            The property used to color the faces: the name of an element property, or
            an array of shape :math:`(N_e,)` or :math:`(N_e, A)`.

        property_axis: Optional[Integral], optional
            The component used for coloring when the property has :math:`A > 1`
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

        texture: Optional[ArrayLike], optional
            An image used as texture, of shape :math:`(H, W, 3)` (RGB) or
            :math:`(H, W)` (gray levels). Only for surface meshes with a UV map (see
            ``uvmap``).

        texture_rgb: bool, optional (default: ``True``)
            True if ``texture`` is an RGB image, False if it is a gray level image
            mapped with ``property_cmap``.

        title: Optional[str], optional
            The title of the window.

        show_bar: bool, optional (default: ``True``)
            Whether to show the color bar when a property is displayed.

        show_bounds: bool, optional (default: ``True``)
            Whether to show the axes bounds.

        show_vertices_index: bool, optional (default: ``False``)
            Whether to write the index of each displayed vertex next to it.

        show_elements_index: bool, optional (default: ``False``)
            Whether to write the index of each displayed element at its center.

        index_font_size: Real, optional (default: ``10``)
            The font size of the indices, in pixels of the screen (constant when zooming).

        screenshot: Optional[str], optional
            The path of a screenshot to save. By default, no screenshot is saved.

        Examples
        --------
        .. code-block:: python
            :linenos:

            import numpy
            from pysdic import Mesh

            points = numpy.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            elements = numpy.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh = Mesh(points, elements, element_type="triangle_3")

            mesh.quick_visualize(face_color="lightgray", edge_color="black")
            mesh.quick_visualize(show_vertices_index=True, show_elements_index=True)  # Numbering

        """
        from .plotter3d import Plotter3D

        plotter = Plotter3D()
        plotter.add_mesh(
            mesh=self,
            warp_by=warp_by,
            vertex_color=vertex_color,
            vertex_size=vertex_size,
            vertex_opacity=vertex_opacity,
            vertex_mask=vertex_mask,
            show_vertices=show_vertices,
            edge_color=edge_color,
            edge_width=edge_width,
            edge_opacity=edge_opacity,
            show_edges=show_edges,
            face_color=face_color,
            face_opacity=face_opacity,
            face_mask=face_mask,
            show_faces=show_faces,
            vertex_property=vertex_property,
            element_property=element_property,
            property_axis=property_axis,
            property_clim=property_clim,
            property_cmap=property_cmap,
            property_below_color=property_below_color,
            property_above_color=property_above_color,
            texture=texture,
            texture_rgb=texture_rgb,
            show_bar=show_bar,
            show_vertices_index=show_vertices_index,
            show_elements_index=show_elements_index,
            index_font_size=index_font_size,
        )
        if show_bounds:
            plotter.set_bounds()
        plotter.set_title(title=title if title is not None else "")
        plotter.show(screenshot=screenshot)