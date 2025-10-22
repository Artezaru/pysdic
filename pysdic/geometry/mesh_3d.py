# Copyright 2025 Artezaru
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
from abc import ABC, abstractmethod

from typing import Optional, Union, Dict, Tuple, Callable

import numpy
import meshio
import os

from .point_cloud_3d import PointCloud3D
from .integrated_points import IntegratedPoints

class Mesh3D(ABC):
    r"""
    A Mesh is a collection of vertices (PointCloud3D) and connectivity information.
    
    This is an abstract base class for 3D meshes.

    The vertices are represented as a PointCloud3D instance with shape (N, 3),
    The connectivity is represented as a numpy ndarray with shape (M, K),
    where N is the number of vertices (``n_vertices``), M is the number of elements (``n_elements``), and K is
    the number of vertices per element (``n_nodes_per_element``). 

    The coordinates of a point into the mesh can be accessed by the natural coordinates
    in the reference element. The number of natural coordinates :math:`(\xi, \eta, \zeta, ...)` depends on the type of element and
    is noted as d (the topological dimension of the element) accessible through the property ``n_dimensions``.

    Lets consider a mesh with K vertices per element, and d natural coordinates.
    Lets :math:`X` be the coordinates of a point in the mesh. The transformation from natural coordinates to global coordinates is given by:

    .. math::

        X = \sum_{i=1}^{K} N_i(\xi, \eta, \zeta, ...) X_i

    where :math:`N_i` are the shape functions associated with each vertex, and :math:`X_i` are the coordinates of the vertices of the element.
        
    The subclasses must implement the following attributes and methods:

    - (class property) ``_n_nodes_per_element``: int, the number of nodes K per element.
    - (class property) ``_n_dimensions``: int, the topological dimension d of the elements.
    - (class property) ``_meshio_cell_type``: str, the cell type used by meshio for this type of element.
    - (method) ``shape_functions``: Callable[[numpy.ndarray], [numpy.ndarray, numpy.ndarray]], a method to compute the shape functions at given natural coordinates (and optional Jacobians).

    The mesh can also store additional properties for the elements and vertices through the dictionaries

    - ``_vertices_properties``: dict, a dictionary to store properties of the vertices, each property is a numpy ndarray of shape (N, A) where N is the number of vertices and A is the number of attributes for that property.
    - ``_elements_properties``: dict, a dictionary to store properties of the elements, each property is a numpy ndarray of shape (M, B) where M is the number of elements and B is the number of attributes for that property.
        
    Parameters
    ----------
    vertices : PointCloud3D
        The vertices of the mesh as a PointCloud3D instance with shape (N, 3).

    connectivity : numpy.ndarray
        The connectivity of the mesh as a numpy ndarray with shape (M, K),
        where M is the number of elements and K is the number of vertices per element.

    vertices_properties : Optional[dict], optional
        A dictionary to store properties of the vertices, each property should be a numpy ndarray of shape (N, A) where N is the number of vertices and A is the number of attributes for that property, by default None.

    elements_properties : Optional[dict], optional
        A dictionary to store properties of the elements, each property should be a numpy ndarray of shape (M, B) where M is the number of elements and B is the number of attributes for that property, by default None.

    internal_bypass : bool, optional
        If True, internal checks are skipped for better performance, by default False.
    """

    _n_nodes_per_element: int = None
    _n_dimensions: int = None
    _meshio_cell_type: str = None

    __slots__ = [
        '_internal_bypass', 
        '_vertices', 
        '_connectivity',
        '_vertices_properties',
        '_elements_properties',
        '_vertices_predefined_metadata',
        '_elements_predefined_metadata',
    ]

    def __init__(self, vertices: PointCloud3D, connectivity: numpy.ndarray, vertices_properties: Optional[Dict] = None, elements_properties: Optional[Dict] = None, internal_bypass: bool = False) -> None:
        # Define expected properties informations
        if not hasattr(self, "_vertices_predefined_metadata"):
            self._vertices_predefined_metadata = {}
        if not hasattr(self, "_elements_predefined_metadata"):
            self._elements_predefined_metadata = {}
        
        # Initialize attributes
        self._internal_bypass = True
        self.vertices = vertices
        self.connectivity = connectivity
        self._vertices_properties = vertices_properties if vertices_properties is not None else {}
        self._elements_properties = elements_properties if elements_properties is not None else {}
        self._internal_bypass = internal_bypass
        self.validate()

    # =======================
    # Internals
    # =======================
    @property
    def internal_bypass(self) -> bool:
        r"""
        When enabled, internal checks are skipped for better performance.

        This is useful for testing purposes, but should not be used in production code.
        Please ensure that all necessary checks are performed before using this mode.

        .. note::

            This property is settable, but it is recommended to set it only when necessary.

        Parameters
        ----------
        value : bool
            If True, internal checks are bypassed. If False, internal checks are performed.

        Returns
        -------
        bool
            True if internal checks are bypassed, False otherwise.

        Raises
        --------
        TypeError
            If the value is not a boolean.

        """
        return self._internal_bypass
    
    @internal_bypass.setter
    def internal_bypass(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError(f"Bypass mode must be a boolean, got {type(value)}.")
        self._internal_bypass = value

    def _internal_check_vertices(self) -> None:
        r"""
        Internal method to check the validity of the vertices.
        
        Raises
        ------
        TypeError
            If vertices is not a PointCloud3D instance.
        ValueError
            If vertices do not have 3 dimensions.
        """
        if self.internal_bypass:
            return
        if not isinstance(self.vertices, PointCloud3D):
            raise TypeError(f"Vertices must be a PointCloud3D instance, got {type(self.vertices)}.")
        if self.vertices.n_points < self.n_nodes_per_element:
            raise ValueError(f"Vertices must have at least {self.n_nodes_per_element} points, got {self.vertices.n_points}.")

    def _internal_check_connectivity(self) -> None:
        r"""
        Internal method to check the validity of the connectivity.
        
        Raises
        ------
        TypeError
            If connectivity is not a numpy ndarray.
        ValueError
            If connectivity does not have the correct shape or contains invalid indices.
        """
        if self.internal_bypass:
            return
        if not isinstance(self.connectivity, numpy.ndarray):
            raise TypeError(f"Connectivity must be a numpy ndarray, got {type(self.connectivity)}.")
        if self.connectivity.ndim != 2:
            raise ValueError(f"Connectivity must be a 2D array, got {self.connectivity.ndim}D array.")
        if self.connectivity.size == 0:
            raise ValueError("Connectivity array cannot be empty.")
        if self.connectivity.shape[1] != self.n_nodes_per_element:
            raise ValueError("Connectivity must have at least two columns (for edges).")
        if numpy.any(self.connectivity < 0) or numpy.any(self.connectivity >= len(self.vertices)):
            raise ValueError("Connectivity contains invalid vertex indices.")
        
    def _internal_check_vertices_property(self, key: str) -> None:
        r"""
        Internal method to check the validity of a specific vertices property.
        
        Parameters
        ----------
        key : str
            The key of the vertices property to check.

        Raises
        ------
        TypeError
            If the vertices property is not a numpy ndarray or has invalid type.
        ValueError
            If the vertices property has invalid shape or values.
        """
        if self.internal_bypass:
            return
        if key not in self._vertices_properties:
            return
        
        # Global checks
        value = self._vertices_properties[key]
        if not isinstance(value, numpy.ndarray):
            raise TypeError(f"Vertices property '{key}' must be a numpy ndarray, got {type(value)}.")
        if value.ndim != 2:
            raise ValueError(f"Vertices property '{key}' must be a 2D array, got {value.ndim}D array.")
        if value.shape[0] != len(self.vertices):
            raise ValueError(f"Vertices property '{key}' must have shape ({len(self.vertices)}, A), got {value.shape}.")
        
        # Specific checks
        if key in self._vertices_predefined_metadata:
            expected_dim = self._vertices_predefined_metadata[key]["dim"]
            expected_type = self._vertices_predefined_metadata[key]["type"]
            check_method = self._vertices_predefined_metadata[key].get("check_method", None)
            if value.shape[1] != expected_dim:
                raise ValueError(f"Vertices property '{key}' must have {expected_dim} columns, got {value.shape[1]}.")
            if value.dtype != expected_type:
                raise TypeError(f"Vertices property '{key}' must have type {expected_type}, got {value.dtype}.")
            if check_method is not None:
                check_method(value)
        
    def _internal_check_vertices_properties(self) -> None:
        r"""
        Internal method to check the validity of the vertices properties.
        
        Raises
        ------
        TypeError
            If vertices properties is not a dictionary or contains invalid types.
        ValueError
            If vertices properties contains invalid shapes.
        """
        if self.internal_bypass:
            return
        for key in self._vertices_properties:
            self._internal_check_vertices_property(key)

    def _internal_check_elements_property(self, key: str) -> None:
        r"""
        Internal method to check the validity of a specific elements property.
        
        Parameters
        ----------
        key : str
            The key of the elements property to check.

        Raises
        ------
        TypeError
            If the elements property is not a numpy ndarray or has invalid type.
        ValueError
            If the elements property has invalid shape or values.
        """
        if self.internal_bypass:
            return
        if key not in self._elements_properties:
            return
        
        # Global checks
        value = self._elements_properties[key]
        if not isinstance(value, numpy.ndarray):
            raise TypeError(f"Elements property '{key}' must be a numpy ndarray, got {type(value)}.")
        if value.ndim != 2:
            raise ValueError(f"Elements property '{key}' must be a 2D array, got {value.ndim}D array.")
        if value.shape[0] != self.n_elements:
            raise ValueError(f"Elements property '{key}' must have shape ({self.n_elements}, B), got {value.shape}.")
        
        # Specific checks
        if key in self._elements_predefined_metadata:
            expected_dim = self._elements_predefined_metadata[key]["dim"]
            expected_type = self._elements_predefined_metadata[key]["type"]
            check_method = self._elements_predefined_metadata[key].get("check_method", None)
            if value.shape[1] != expected_dim:
                raise ValueError(f"Elements property '{key}' must have {expected_dim} columns, got {value.shape[1]}.")
            if value.dtype != expected_type:
                raise TypeError(f"Elements property '{key}' must have type {expected_type}, got {value.dtype}.")
            if check_method is not None:
                check_method(value)
            
    def _internal_check_elements_properties(self) -> None:
        r"""
        Internal method to check the validity of the elements properties.
        
        Raises
        ------
        TypeError
            If elements properties is not a dictionary or contains invalid types.
        ValueError
            If elements properties contains invalid shapes.
        """
        if self.internal_bypass:
            return
        for key in self._elements_properties:
            self._internal_check_elements_property(key)

    def _check_integrated_points(self, integrated_points: IntegratedPoints) -> None:
        r"""
        Internal method to check the validity of an IntegratedPoints instance for this mesh.
        
        Parameters
        ----------
        
        integrated_points : IntegratedPoints
            The IntegratedPoints instance to check.
            
        Raises
        ------
        TypeError
            If integrated_points is not an IntegratedPoints instance.
        ValueError
            If integrated_points has invalid dimensions or contains invalid element indices.
            
        """
        if not isinstance(integrated_points, IntegratedPoints):
            raise TypeError(f"Input must be an IntegratedPoints instance, got {type(integrated_points)}.")
        if not integrated_points.n_dimensions == self.n_dimensions:
            raise ValueError(f"IntegratedPoints dimensions ({integrated_points.dimensions}) do not match mesh dimensions ({self.n_dimensions}).")
        if integrated_points.n_points == 0:
            return PointCloud3D(numpy.empty((0, 3), dtype=numpy.float64))
        if numpy.max(integrated_points.element_indices) >= self.n_elements:
            raise ValueError("IntegratedPoints contains element indices out of bounds for this mesh.")

    def _get_vertices_property(self, key: Optional[None], default: Optional[numpy.ndarray] = None, raise_error: bool = False) -> Optional[numpy.ndarray]:
        r"""
        Internal method to get a vertices property or return a default value if the property does not exist.

        Parameters
        ----------
        key : Optional[str]
            The key of the vertices property to retrieve. If None, returns the default value.

        default : Optional[numpy.ndarray], optional
            The default value to return if the property does not exist, by default None.

        raise_error : bool, optional
            If True, raises a KeyError if the property does not exist, by default False.

        Returns
        -------
        Optional[numpy.ndarray]
            The vertices property associated with the key, or the default value if the property does not exist.
        """
        # Overwrite default if key is provided
        if key is not None:
            default = self._vertices_properties.get(key, None)

        if default is None and raise_error:
            raise KeyError(f"Vertices property '{key}' does not exist in the mesh.")
        if default is None:
            return None
        
        default = numpy.asarray(default)
        if not isinstance(default, numpy.ndarray):
            raise TypeError(f"Vertices property must be a numpy ndarray, got {type(default)}.")
        if not default.ndim == 2 or not default.shape[0] == len(self.vertices) or not default.shape[1] >= 1:
            raise ValueError(f"Vertices property must have shape ({len(self.vertices)}, A), got {default.shape}.")
        
        return default
    
    def _get_elements_property(self, key: Optional[None], default: Optional[numpy.ndarray] = None, raise_error: bool = False) -> Optional[numpy.ndarray]:
        r"""
        Internal method to get an elements property or return a default value if the property does not exist.

        Parameters
        ----------
        key : Optional[str]
            The key of the elements property to retrieve. If None, returns the default value.

        default : Optional[numpy.ndarray], optional
            The default value to return if the property does not exist, by default None.

        raise_error : bool, optional
            If True, raises a KeyError if the property does not exist, by default False.

        Returns
        -------
        Optional[numpy.ndarray]
            The elements property associated with the key, or the default value if the property does not exist.
        """
        # Overwrite default if key is provided
        if key is not None:
            default = self._elements_properties.get(key, None)

        if default is None and raise_error:
            raise KeyError(f"Elements property '{key}' does not exist in the mesh.")
        if default is None:
            return None

        default = numpy.asarray(default)        
        if not isinstance(default, numpy.ndarray):
            raise TypeError(f"Elements property must be a numpy ndarray, got {type(default)}.")
        if not default.ndim == 2 or not default.shape[0] == self.n_elements or not default.shape[1] >= 1:
            raise ValueError(f"Elements property must have shape ({self.n_elements}, B), got {default.shape}.")
        
        return default
        
    # =======================
    # I/O Methods
    # =======================
    @classmethod
    def from_meshio(cls, mesh: meshio.Mesh, load_properties: bool = True) -> Mesh3D:
        r"""
        Create a Mesh3D instance from a meshio Mesh object.

        The following fields are extracted:

        - mesh.points → vertices
        - mesh.cells[0].data → triangles
        - mesh.point_data → _vertex_properties as arrays of shape (N, A)
        - mesh.cell_data → _element_properties as arrays of shape (M, B)

        .. seealso::

            - :meth:`Mesh3D.to_meshio` for the reverse operation.
            - `meshio documentation <https://pypi.org/project/meshio/>`_ for more information.

        Parameters
        ----------
        mesh : meshio.Mesh
            A meshio Mesh object.

        load_properties : bool, optional
            If True, properties are extracted from the meshio Mesh object, by default True.

        Returns
        -------
        Mesh3D
            A Mesh3D instance created from the meshio Mesh object.

        Raises
        ------
        TypeError
            If the input is not a meshio Mesh object.
        ValueError
            If the mesh structure is invalid.

        Examples
        --------
        Create a simple meshio Mesh object.

        .. code-block:: python

            import numpy as np
            import meshio
            from pysdic.geometry import Mesh3D

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            cells = [("triangle", np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]))]

            mesh = meshio.Mesh(points=points, cells=cells)

        Create a Mesh3D instance from the meshio Mesh object.

        .. code-block:: python

            mesh3d = Mesh3D.from_meshio(mesh)
            print(mesh3d.vertices)
            # Output: PointCloud3D with 4 points [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]

        """
        # Validations
        if not isinstance(mesh, meshio.Mesh):
            raise TypeError(f"Input must be a meshio Mesh object, got {type(mesh)}.")
        if not len(mesh.cells) == 1 or mesh.cells[0].data.ndim != 2 or mesh.cells[0].data.shape[1] != cls.n_nodes_per_element:
            raise ValueError("Invalid mesh structure.")
        if not isinstance(load_properties, bool):
            raise TypeError(f"load_properties must be a boolean, got {type(load_properties)}.")

        # Extract data
        vertices = PointCloud3D(mesh.points)
        connectivity = mesh.cells[0].data
        mesh_properties = {}
        vertices_properties = {}
        elements_properties = {}
        
        # Extract properties if requested
        if load_properties:
            for key, value in mesh.field_data.items():
                data = int(value[0])
                mesh_properties[key] = data
            
            for key, value in mesh.point_data.items():
                vertices_properties[key] = numpy.asarray(value[0])

            for key, value in mesh.cell_data.items():
                elements_properties[key] = numpy.asarray(value[0])

        # Create Mesh3D instance
        return cls(vertices, connectivity, mesh_properties=mesh_properties, vertices_properties=vertices_properties, elements_properties=elements_properties)


    def to_meshio(self, save_properties: bool = True) -> meshio.Mesh:
        r"""
        Convert the Mesh3D instance to a meshio Mesh object.

        The following fields are created:

        - vertices → mesh.points
        - connectivity → mesh.cells[0].data
        - _vertex_properties as arrays of shape (N, A) → mesh.point_data
        - _element_properties as arrays of shape (M, B) → mesh.cell_data

        .. seealso::

            - :meth:`Mesh3D.from_meshio` for the reverse operation.
            - `meshio documentation <https://pypi.org/project/meshio/>`_ for more information.

        Parameters
        ----------
        save_properties : bool, optional
            If True, properties are saved to the meshio Mesh object, by default True.

        Returns
        -------
        meshio.Mesh
            A meshio Mesh object created from the Mesh3D instance.

        Raises
        ------
        TypeError
            If save_properties is not a boolean.

        Examples
        --------
        Create a simple Mesh3D instance.

        .. code-block:: python

            import numpy as np
            from pysdic.geometry import Mesh3D, PointCloud3D

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])

            mesh3d = Mesh3D(PointCloud3D.from_array(points), connectivity)

        Convert the Mesh3D instance to a meshio Mesh object.

        .. code-block:: python

            mesh = mesh3d.to_meshio()
            print(mesh.points)
            # Output: [[0. 0. 0.] [1. 0. 0.] [0. 1. 0.] [0. 0. 1.]]
            
        """
        if not isinstance(save_properties, bool):
            raise TypeError(f"save_properties must be a boolean, got {type(save_properties)}.")
        
        cells = [meshio.CellBlock(type=self.meshio_cell_type(), data=self.connectivity)]
        
        if save_properties:
            point_data = {key: value for key, value in self._vertices_properties.items()}
            cell_data = {key: [value] for key, value in self._elements_properties.items()}
            field_data = {key: (numpy.array([value]), 1) for key, value in self._mesh_properties.items()}
        else:
            point_data = {}
            cell_data = {}
            field_data = {}
        
        return meshio.Mesh(points=self.vertices.coordinates, cells=cells, point_data=point_data, cell_data=cell_data, field_data=field_data)
    
    @classmethod
    def from_vtk(cls, filename: str, load_properties: bool = True) -> Mesh3D:
        r"""
        Create a Mesh3D instance from a VTK file.

        This method uses meshio to read the VTK file and then converts it to a Mesh3D instance.

        .. seealso::

            - :meth:`Mesh3D.to_vtk` for the reverse operation.
            - :meth:`Mesh3D.from_meshio` for more information on the conversion process.
            - `meshio documentation <https://pypi.org/project/meshio/>`_ for more information.

        Parameters
        ----------
        filename : str
            The path to the VTK file.

        load_properties : bool, optional
            If True, properties are extracted from the VTK file, by default True.

        Returns
        -------
        Mesh3D
            A Mesh3D instance created from the VTK file.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file format is not supported or the mesh structure is invalid.
        
        Examples
        --------
        Create a simple meshio Mesh object.

        .. code-block:: python

            import numpy as np
            import meshio
            from pysdic.geometry import Mesh3D

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            cells = [("triangle", np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]))]

            mesh = meshio.Mesh(points=points, cells=cells)

        Save the meshio Mesh object to a VTK file.

        .. code-block:: python

            mesh.write("simple_mesh.vtk", file_format="vtk")

        Create a Mesh3D instance from the VTK file.

        .. code-block:: python

            mesh3d = Mesh3D.from_vtk("simple_mesh.vtk")
            print(mesh3d.vertices)
            # Output: PointCloud3D with 4 points [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]      

        """
        path = os.path.abspath(os.path.expanduser(filename))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File '{filename}' does not exist.")
        
        mesh = meshio.read(filename, file_format="vtk")
        return cls.from_meshio(mesh, load_properties=load_properties)

    def to_vtk(self, filename: str, save_properties: bool = True) -> None:
        r"""
        Write the Mesh3D instance to a VTK file.

        This method uses meshio to write the Mesh3D instance to a VTK file.

        .. seealso::

            - :meth:`Mesh3D.from_vtk` for the reverse operation.
            - :meth:`Mesh3D.to_meshio` for more information on the conversion process.
            - `meshio documentation <https://pypi.org/project/meshio/>`_ for more information.

        Parameters
        ----------
        filename : str
            The path to the output VTK file.
        
        save_properties : bool, optional
            If True, properties are saved to the VTK file, by default True.

        Raises
        ------
        ValueError
            If the file format is not supported.

        Examples
        --------
        Create a simple Mesh3D instance.

        .. code-block:: python

            import numpy as np
            from pysdic.geometry import Mesh3D, PointCloud3D

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = Mesh3D(PointCloud3D.from_array(points), connectivity)

        Save the Mesh3D instance to a VTK file.

        .. code-block:: python

            mesh3d.to_vtk("simple_mesh.vtk")
            # This will create a file named 'simple_mesh.vtk' in the current directory.
            
        """
        path = os.path.abspath(os.path.expanduser(filename))
        os.makedirs(os.path.dirname(path), exist_ok=True)

        mesh = self.to_meshio(save_properties=save_properties)
        mesh.write(filename, file_format="vtk")

    # =======================
    # Properties
    # =======================
    @property
    def vertices(self) -> PointCloud3D:
        r"""
        The vertices of the mesh in an :class:`PointCloud3D` instance.

        The vertices are represented as a PointCloud3D instance with shape (N, 3) where N is the number of vertices.

        .. note::

            This property is settable.

        To change the number of vertices, it is recommended to create a new Mesh3D instance with the updated vertices and connectivity
        rather than modifying the vertices in place. For memory considerations, you can also modify the vertices in place, but please ensure that
        all necessary checks are performed before using this mode.

        You should set `internal_bypass` to True before modifying the vertices, and set it back to False afterwards.

        .. code-block:: python

            mesh.internal_bypass = True
            mesh.vertices = new_vertices
            mesh.connectivity = new_connectivity
            mesh.clear_properties()  # Optional: clear properties if they are no longer valid
            mesh.internal_bypass = False
            mesh.validate()  # Optional: ensure the mesh is still valid

        .. warning::

            If the vertices are changed, the connectivity and properties may become invalid. 
            Please ensure to recompute or update them accordingly.

        Parameters
        ----------
        value : Union[PointCloud3D, numpy.ndarray]
            The new vertices for the mesh with shape (N, 3).

        Returns
        -------
        PointCloud3D
            The vertices of the mesh as a PointCloud3D instance of shape (N, 3).

        Exemples
        --------
        Create a simple Mesh3D instance.

        .. code-block:: python

            import numpy as np
            from pysdic.geometry import Mesh3D, PointCloud3D

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = Mesh3D(PointCloud3D.from_array(points), connectivity)

        Access the vertices of the mesh.

        .. code-block:: python

            print(mesh3d.vertices)
            # Output: PointCloud3D with 4 points [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        
        """
        return self._vertices
    
    @vertices.setter
    def vertices(self, value: Union[PointCloud3D, numpy.ndarray]) -> None:
        if not isinstance(value, PointCloud3D):
            value = PointCloud3D(value)
        self._vertices = value
        self._internal_check_vertices()

    @property
    def connectivity(self) -> numpy.ndarray:
        r"""
        Get or set the connectivity of the mesh.

        The connectivity is represented as a numpy ndarray with shape (M, K),
        where M is the number of elements and K is the number of vertices per element.

        .. note::

            If the connectivity is changed, the properties should be updated accordingly.
            To avoid inconsistencies, it is recommended to create a new Mesh3D instance with the updated vertices and connectivity.

            Otherwise, you should set `internal_bypass` to True before modifying the connectivity, and set it back to False afterwards.

            .. code-block:: python

                mesh.internal_bypass = True
                mesh.connectivity = new_connectivity
                mesh.clear_elements_properties()  # Optional: clear properties if they are no longer valid
                mesh.elements_uvmap = new_uvmap  # Optional: set new uvmap if needed or other properties
                mesh.internal_bypass = False
                mesh.validate()  # Optional: ensure the mesh is still valid

        .. warning::

            If the connectivity is changed, the properties may become invalid. 
            Please ensure to recompute or update them accordingly.        

        Parameters
        ----------
        value : numpy.ndarray
            The new connectivity for the mesh as an array-like of shape (M, K).

        Returns
        -------
        numpy.ndarray
            The connectivity of the mesh.
        """
        return self._connectivity
    
    @connectivity.setter
    def connectivity(self, value: numpy.ndarray) -> None:
        value = numpy.asarray(value, dtype=int)
        self._connectivity = value
        self._internal_check_connectivity()

    @property
    def n_vertices(self) -> int:
        r"""
        Get the number of vertices in the mesh.

        .. note::

            Alias for `mesh.vertices.n_points`.
            You can also use `len(mesh.vertices)`.

        Returns
        -------
        int
            The number of vertices in the mesh.
        """
        return len(self.vertices)
    
    @property
    def n_elements(self) -> int:
        r"""
        Get the number of elements in the mesh.

        Returns
        -------
        int
            The number of elements in the mesh.
        """
        return self.connectivity.shape[0]
    
    @property
    def n_nodes_per_element(self) -> int:
        r"""
        Get the number of nodes per element in the mesh.

        Returns
        -------
        int
            The number of nodes per element.
        """
        if self._n_nodes_per_element is None:
            raise NotImplementedError("Subclasses must implement n_nodes_per_element property.")
        return self._n_nodes_per_element

    @property
    def n_dimensions(self) -> int:
        r"""
        Get the topological dimension of the elements in the mesh.

        Returns
        -------
        int
            The topological dimension of the elements.
        """
        if self._n_dimensions is None:
            raise NotImplementedError("Subclasses must implement n_dimensions property.")
        return self._n_dimensions

    @property
    def meshio_cell_type(self) -> str:
        r"""
        Get the cell type used by meshio for this type of element.

        Returns
        -------
        str
            The cell type used by meshio.
        """
        if self._meshio_cell_type is None:
            raise NotImplementedError("Subclasses must implement meshio_cell_type property.")
        return self._meshio_cell_type


    # =======================
    # Properties Methods
    # =======================
    def get_vertices_property(self, key: str) -> Optional[numpy.ndarray]:
        r"""
        Get a property associated with the vertices of the mesh.

        Parameters
        ----------
        key : str
            The key of the property to retrieve.

        Returns
        -------
        numpy.ndarray or None
            The property associated with the vertices, or None if the property does not exist.
        """
        return self._get_vertices_property(key, None, raise_error=False)
    
    def get_elements_property(self, key: str) -> Optional[numpy.ndarray]:
        r"""
        Get a property associated with the elements of the mesh.

        Parameters
        ----------
        key : str
            The key of the property to retrieve.

        Returns
        -------
        numpy.ndarray or None
            The property associated with the elements, or None if the property does not exist.
        """
        return self._get_elements_property(key, None, raise_error=False)
    
    def set_vertices_property(self, key: str, value: Optional[numpy.ndarray]) -> None:
        r"""
        Set a property associated with the vertices of the mesh.

        Parameters
        ----------
        key : str
            The key of the property to set.

        value : Optional[numpy.ndarray]
            The property to associate with the vertices as an array-like of shape (N, A),
            where N is the number of vertices and A is the number of attributes for that property.
            If None, the property is removed.

        Raises
        ------
        TypeError
            If value is not a numpy ndarray or None.
        ValueError
            If value does not have the correct shape.
        """
        if value is None:
            if key in self._vertices_properties:
                del self._vertices_properties[key]
            return
        
        if key in self._vertices_predefined_metadata:
            type_info = self._vertices_predefined_metadata[key]["type"]
        else:
            type_info = None

        value = numpy.asarray(value, dtype=type_info)
        self._vertices_properties[key] = value
        self._internal_check_vertices_property(key)

    def set_elements_property(self, key: str, value: Optional[numpy.ndarray]) -> None:
        r"""
        Set a property associated with the elements of the mesh.

        Parameters
        ----------
        key : str
            The key of the property to set.

        value : Optional[numpy.ndarray]
            The property to associate with the elements as an array-like of shape (M, B),
            where M is the number of elements and B is the number of attributes for that property.
            If None, the property is removed.

        Raises
        ------
        TypeError
            If value is not a numpy ndarray or None.
        ValueError
            If value does not have the correct shape.
        """
        if value is None:
            if key in self._elements_properties:
                del self._elements_properties[key]
            return
        
        if key in self._elements_predefined_metadata:
            type_info = self._elements_predefined_metadata[key]["type"]
        else:
            type_info = None

        value = numpy.asarray(value, dtype=type_info)
        self._elements_properties[key] = value
        self._internal_check_elements_property(key)

    def remove_vertices_property(self, key: str) -> None:
        r"""
        Remove a property associated with the vertices of the mesh.

        Parameters
        ----------
        key : str
            The key of the property to remove.

        Raises
        ------
        KeyError
            If the property does not exist.
        """
        if key in self._vertices_properties:
            del self._vertices_properties[key]
        else:
            raise KeyError(f"Vertices property '{key}' does not exist.")
    
    def remove_elements_property(self, key: str) -> None:
        r"""
        Remove a property associated with the elements of the mesh.

        Parameters
        ----------
        key : str
            The key of the property to remove.

        Raises
        ------
        KeyError
            If the property does not exist.
        """
        if key in self._elements_properties:
            del self._elements_properties[key]
        else:
            raise KeyError(f"Elements property '{key}' does not exist.")
        
    def list_vertices_properties(self) -> Tuple[str]:
        r"""
        List all keys of the properties associated with the vertices of the mesh.

        Returns
        -------
        Tuple[str]
            A tuple containing all keys of the vertices properties.
        """
        return tuple(self._vertices_properties.keys())
    
    def list_elements_properties(self) -> Tuple[str]:
        r"""
        List all keys of the properties associated with the elements of the mesh.

        Returns
        -------
        Tuple[str]
            A tuple containing all keys of the elements properties.
        """
        return tuple(self._elements_properties.keys())

    def clear_vertices_properties(self) -> None:
        r"""
        Clear all properties associated with the vertices of the mesh.

        After calling this method, the vertices properties dictionary will be empty.
        """
        self._vertices_properties.clear()

    def clear_elements_properties(self) -> None:
        r"""
        Clear all properties associated with the elements of the mesh.

        After calling this method, the elements properties dictionary will be empty.
        """
        self._elements_properties.clear()

    def clear_properties(self) -> None:
        r"""
        Clear all properties of the mesh, including mesh properties, vertices properties, and elements properties.

        After calling this method, the properties dictionaries will be empty.
        """
        self.clear_elements_properties()
        self.clear_vertices_properties()

    def validate(self) -> None:
        r"""
        Validate the mesh by performing internal checks on vertices and connectivity.

        Raises
        ------
        TypeError
            If vertices is not a PointCloud3D instance or connectivity is not a numpy ndarray.
        ValueError
            If vertices do not have 3 dimensions, if connectivity does not have the correct shape, or contains invalid indices.
        """
        self._internal_check_vertices()
        self._internal_check_connectivity()
        self._internal_check_vertices_properties()
        self._internal_check_elements_properties()
        
    # =======================
    # Public Methods
    # =======================    
    def copy(self) -> Mesh3D:
        r"""
        Create a deep copy of the Mesh3D instance.

        Returns
        -------
        Mesh3D
            A deep copy of the Mesh3D instance.
        """
        return type(self)(
            vertices=self.vertices.copy(),
            connectivity=self.connectivity.copy(),
            vertices_properties={key: value.copy() for key, value in self._vertices_properties.items()},
            elements_properties={key: value.copy() for key, value in self._elements_properties.items()},
            internal_bypass=self.internal_bypass
        ) 

    def get_vertices_coordinates(self, element_indices: numpy.ndarray) -> numpy.ndarray:
        r"""
        Get the coordinates of the vertices for the specified elements.

        If the element_indices is of shape (M,), the returned array will be of shape (M, K, 3),
        where K is the number of vertices per element.

        Parameters
        ----------
        element_indices : numpy.ndarray
            An array of shape (M,) containing the element indices. Must be a 1D array.

        Returns
        -------
        numpy.ndarray
            An array of shape (M, K, 3) containing the vertex coordinates for each element.
        """
        element_indices = numpy.asarray(element_indices, dtype=int)
        if element_indices.ndim != 1:
            raise ValueError(f"Element indices must be a 1D array, got {element_indices.ndim}D array.")
        return self.vertices.points[self.connectivity[element_indices, :], :]

    def natural_to_global(self, natural_coords: numpy.ndarray, element_indices: numpy.ndarray) -> numpy.ndarray:
        r"""
        Transform natural coordinates to global coordinates for specified elements.

        Lets consider a mesh with K vertices per element, and d natural coordinates.
        The given natural coordinates should be (Np, d) where Np is the number of points to evaluate.
        The element_indices should be (Np,) where each entry is the index of the element in which to evaluate the natural coordinates.
        The returned global coordinates will be of shape (Np, 3).

        The transformation from natural coordinates to global coordinates is given by:

        .. math::

            X = \sum_{i=1}^{K} N_i(\xi, \eta, \zeta, ...) X_i

        where :math:`N_i` are the shape functions associated with each vertex, and :math:`X_i` are the coordinates of the vertices of the element.

        .. note:

            For one point, the input must be (1, d) and not only (d,).

        .. seealso::

            - :meth:`shape_functions` for computing shape functions.
            - :meth:`global_to_natural` for the inverse transformation.

        Parameters
        ----------
        natural_coords : numpy.ndarray
            An array of shape (Np, d) containing the natural coordinates.

        element_indices : numpy.ndarray
            An array of shape (Np,) containing the element indices for each point.

        Returns
        -------
        numpy.ndarray
            An array of shape (Np, 3) containing the global coordinates.
        """
        # Validate input shapes
        natural_coords = numpy.asarray(natural_coords, dtype=numpy.float64)
        element_indices = numpy.asarray(element_indices, dtype=int)
        if natural_coords.ndim != 2 or natural_coords.shape[1] != self.n_dimensions:
            raise ValueError(f"Invalid natural_coords shape: {natural_coords.shape}")
        if element_indices.ndim != 1 or element_indices.shape[0] != natural_coords.shape[0]:
            raise ValueError(f"Invalid element_indices shape: {element_indices.shape}")

        # Get the shape functions and vertices coordinates
        shape_functions, _ = self.shape_functions(natural_coords)  # (Np, K)
        vertices_coords = self.get_vertices_coordinates(element_indices)  # (Np, K, 3)

        # Compute global coordinates
        global_coords = numpy.einsum('ij,ijk->ik', shape_functions, vertices_coords)  # (Np, 3)
        return global_coords


    def evaluate_vertices_property_at(self, natural_coords: numpy.ndarray, element_indices: numpy.ndarray, *, property_key: Optional[str] = None, property_array: Optional[numpy.ndarray] = None) -> numpy.ndarray:
        r"""
        Evaluate a vertices property at given natural coordinates and element indices.

        Lets consider a mesh with K vertices per element, and d natural coordinates.
        The given natural coordinates should be (Np, d) where Np is the number of points to evaluate.
        The element_indices should be (Np,) where each entry is the index of the element in which to evaluate the natural coordinates.
        The returned property values will be of shape (Np, A) where A is the number of attributes for that property.

        The evaluation of the property at the given natural coordinates is performed using the shape functions:

        .. math::

            P = \sum_{i=1}^{K} N_i(\xi, \eta, \zeta, ...) P_i

        where :math:`N_i` are the shape functions associated with each vertex, and :math:`P_i` are the property values at the vertices of the element.

        .. note:

            For one point, the input must be (1, d) and not only (d,).

        Parameters
        ----------
        natural_coords : numpy.ndarray
            An array of shape (Np, d) containing the natural coordinates.

        element_indices : numpy.ndarray
            An array of shape (Np,) containing the element indices for each point.

        property_key : Optional[str], optional
            The name of the vertices property to evaluate. If None, property_array must be provided., by default None.

        property_array : Optional[numpy.ndarray], optional
            An array of shape (N, A) containing the vertices property values. If None, property_key must be provided., by default None.

        Returns
        -------
        numpy.ndarray
            An array of shape (Np, A) containing the evaluated property values.
        
        Raises
        ------
        ValueError
            If neither or both property_key and property_array are provided.
            If property_key does not exist in vertices properties.
            If property_array does not have the correct shape.
        """
        # Validate input shapes
        natural_coords = numpy.asarray(natural_coords, dtype=numpy.float64)
        element_indices = numpy.asarray(element_indices, dtype=int)
        if natural_coords.ndim != 2 or natural_coords.shape[1] != self.n_dimensions:
            raise ValueError(f"Invalid natural_coords shape: {natural_coords.shape}")
        if element_indices.ndim != 1 or element_indices.shape[0] != natural_coords.shape[0]:
            raise ValueError(f"Invalid element_indices shape: {element_indices.shape}")
        
        # Get the property array
        if (property_key is None and property_array is None) or (property_key is not None and property_array is not None):
            raise ValueError("Either property_key or property_array must be provided, but not both.")
        property_array = self._get_vertices_property(property_key, property_array, raise_error=True)  # (N, A)
        
        # Get the shape functions and vertices coordinates
        shape_functions, _ = self.shape_functions(natural_coords)  # (Np, K)
        vertices_property = property_array[self.connectivity[element_indices, :], :]  # (Np, K, A)

        # Compute evaluated property
        evaluated_property = numpy.einsum('ij,ijk->ik', shape_functions, vertices_property)  # (Np, A)
        return evaluated_property
        

    # =======================
    # Convenience Methods
    # =======================
    def natural_to_global_points(self, integrated_points: IntegratedPoints) -> PointCloud3D:
        r"""
        Transform natural coordinates to global coordinates for an IntegratedPoints instance.

        This method is a convenience wrapper around :meth:`natural_to_global` that extracts the natural coordinates and element indices from the IntegratedPoints instance.

        The integrated points with "-1" element index are ignored and the corresponding global coordinates are setted to NaN.

        Parameters
        ----------
        integrated_points : IntegratedPoints
            An IntegratedPoints instance containing natural coordinates and element indices.

        Returns
        -------
        PointCloud3D
            A PointCloud3D instance containing the global coordinates.
        """
        self._check_integrated_points(integrated_points)
        
        global_coordinates = numpy.empty((len(integrated_points), 3), dtype=numpy.float64)
        valid_mask = integrated_points.element_indices != -1
        global_coordinates[valid_mask, :] = self.natural_to_global(integrated_points.natural_coordinates[valid_mask, :], integrated_points.element_indices[valid_mask])
        global_coordinates[~valid_mask, :] = numpy.nan
        return PointCloud3D(global_coordinates)
    
    def evaluate_vertices_property_at_points(self, integrated_points: IntegratedPoints, *, property_key: Optional[str] = None, property_array: Optional[numpy.ndarray] = None) -> numpy.ndarray:
        r"""
        Evaluate a vertices property at given natural coordinates and element indices from an IntegratedPoints instance.

        This method is a convenience wrapper around :meth:`evaluate_vertices_property_at` that extracts the natural coordinates and element indices from the IntegratedPoints instance.

        The integrated points with "-1" element index are ignored and the corresponding property values are setted to NaN.

        Parameters
        ----------
        integrated_points : IntegratedPoints
            An IntegratedPoints instance containing natural coordinates and element indices.

        property_key : Optional[str], optional
            The name of the vertices property to evaluate. If None, property_array must be provided., by default None.

        property_array : Optional[numpy.ndarray], optional
            An array of shape (N, A) containing the vertices property values. If None, property_key must be provided., by default None.

        Returns
        -------
        numpy.ndarray
            An array of shape (Np, A) containing the evaluated property values.
    
        """
        self._check_integrated_points(integrated_points)
        
        # Get the property array
        if (property_key is None and property_array is None) or (property_key is not None and property_array is not None):
            raise ValueError("Either property_key or property_array must be provided, but not both.")
        property_array = self._get_vertices_property(property_key, property_array, raise_error=True)  # (N, A)

        evaluated_property = numpy.empty((len(integrated_points), property_array.shape[1]), dtype=property_array.dtype)
        valid_mask = integrated_points.element_indices != -1
        evaluated_property[valid_mask, :] = self.evaluate_vertices_property_at(integrated_points.natural_coordinates[valid_mask, :], integrated_points.element_indices[valid_mask], property_array=property_array)
        evaluated_property[~valid_mask, :] = numpy.nan
        return evaluated_property


    # =======================
    # Abstract Methods
    # =======================
    @abstractmethod
    def shape_functions(self, natural_coords: numpy.ndarray, jacobian: bool = False) -> Tuple[numpy.ndarray, Optional[numpy.ndarray]]:
        r"""
        Compute the shape functions at given natural coordinates.

        Lets consider a mesh with K vertices per element, and d natural coordinates.
        The given natural coordinates should be (Np, d) where Np is the number of points to evaluate and d is the dimension of the natural coordinates.
        The returned shape functions will be of shape (Np, K) and each row will sum to 1 and contain the values of the shape functions associated with each vertex of the element.

        The shape fonctions :math:`N_i` are defined such that:

        .. math::

            X = \sum_{i=1}^{K} N_i(\xi, \eta, \zeta, ...) X_i

        where :math:`X` are the global coordinates of a point, and :math:`X_i` are the coordinates of the vertices of the element and :math:`(\xi, \eta, \zeta, ...)` are the natural coordinates.

        .. note:

            For one point, the input must be (1, d) and not only (d,).

        If ``jacobian`` is True, the method also returns the Jacobian of the shape functions with respect to the natural coordinates,
        The returned Jacobian will be of shape (Np, K, d) where each entry (i, j, k) is the derivative of the j-th shape function with respect to the k-th natural coordinate at the i-th point.

        .. math::

            \frac{\partial X}{\partial \xi_j} = \sum_{i=1}^{K} \frac{\partial N_i}{\partial \xi_j} X_i

        .. seealso::

            - :meth:`natural_to_global` for transforming natural coordinates to global coordinates.

        Parameters
        ----------
        natural_coords : numpy.ndarray
            An array-like of shape (Np, d) where Np is the number of points to evaluate and d is the number of natural coordinates.

        Returns
        -------
        numpy.ndarray
            An array of shape (Np, K) where K is the number of nodes per element.

        Optional[numpy.ndarray]
            If ``jacobian`` is True, an array of shape (Np, K, d) where K is the number of nodes per element and d is the number of natural coordinates. Otherwise, None.

        """
        raise NotImplementedError("Subclasses must implement shape_functions method.")
    

