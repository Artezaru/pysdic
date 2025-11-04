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
import scipy
import os

from .point_cloud_3d import PointCloud3D
from .integration_points import IntegrationPoints

class Mesh3D(ABC):
    r"""
    A Mesh is a collection of vertices (PointCloud3D) and connectivity information that defines the elements of the mesh.
    
    This is an abstract base class for 3D meshes.

    The vertices are represented as a PointCloud3D instance with shape (N, 3),
    The connectivity is represented as a numpy ndarray with shape (M, K),
    where N is the number of vertices (``n_vertices``), M is the number of elements (``n_elements``), and K is
    the number of vertices per element (``n_vertices_per_element``). 

    The coordinates of a point into the mesh can be accessed by the natural coordinates
    in the reference element. The number of natural coordinates :math:`(\xi, \eta, \zeta, ...)` depends on the type of element and
    is noted as d (the topological dimension of the element) accessible through the property ``n_dimensions``.

    Lets consider a mesh with K vertices per element, and d natural coordinates.
    Lets :math:`X` be the coordinates of a point in the mesh. The transformation from natural coordinates to global coordinates is given by:

    .. math::

        X = \sum_{i=1}^{K} N_i(\xi, \eta, \zeta, ...) X_i

    where :math:`N_i` are the shape functions associated with each vertex, and :math:`X_i` are the coordinates of the vertices of the element.
        
    The subclasses must implement the following attributes and methods:

    - (class property) ``_n_vertices_per_element``: int, the number of vertices K per element.
    - (class property) ``_n_dimensions``: int, the topological dimension d of the elements.
    - (class property) ``_meshio_cell_type``: str, the cell type used by meshio for this type of element.
    - (method) ``shape_functions``: Callable[[numpy.ndarray], [numpy.ndarray, numpy.ndarray]], a method to compute the shape functions at given natural coordinates (and optional Jacobians).

    The mesh can also store additional properties for the elements and vertices through the dictionaries:

    - ``_vertices_properties``: dict, a dictionary to store properties of the vertices, each property is a numpy ndarray with float64 dtype of shape (N, A) where N is the number of vertices and A is the number of attributes for that property.
    - ``_elements_properties``: dict, a dictionary to store properties of the elements, each property is a numpy ndarray with float64 dtype of shape (M, B) where M is the number of elements and B is the number of attributes for that property.

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

    _n_vertices_per_element: int = None
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
        self._vertices_properties = {}
        self._elements_properties = {}
        if vertices_properties is not None:
            for key, value in vertices_properties.items():
                self.set_vertices_property(key, value)
        if elements_properties is not None:
            for key, value in elements_properties.items():
                self.set_elements_property(key, value)
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
        if not numpy.isfinite(self.vertices.points).all():
            raise ValueError("Vertices contain NaN or infinite values.")

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
        if self.connectivity.shape[1] != self.n_vertices_per_element:
            raise ValueError("Connectivity must have at least two columns (for edges).")
        if numpy.any(self.connectivity < 0) or numpy.any(self.connectivity >= len(self.vertices)):
            raise ValueError("Connectivity contains invalid vertex indices.")
        if not numpy.issubdtype(self.connectivity.dtype, numpy.integer):
            raise TypeError(f"Connectivity must have integer type, got {self.connectivity.dtype}.")
        
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
            check_method = self._vertices_predefined_metadata[key].get("check_method", None)
            if value.shape[1] != expected_dim:
                raise ValueError(f"Vertices property '{key}' must have {expected_dim} columns, got {value.shape[1]}.")
            if value.dtype != numpy.float64:
                raise TypeError(f"Vertices property '{key}' must have type float64, got {value.dtype}.")
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
            check_method = self._elements_predefined_metadata[key].get("check_method", None)
            if value.shape[1] != expected_dim:
                raise ValueError(f"Elements property '{key}' must have {expected_dim} columns, got {value.shape[1]}.")
            if value.dtype != numpy.float64:
                raise TypeError(f"Elements property '{key}' must have type float64, got {value.dtype}.")
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

    def _check_integration_points(self, integration_points: IntegrationPoints) -> None:
        r"""
        Internal method to check the validity of an IntegrationPoints instance for this mesh.
        
        Parameters
        ----------
        
        integration_points : IntegrationPoints
            The IntegrationPoints instance to check.
            
        Raises
        ------
        TypeError
            If integration_points is not an IntegrationPoints instance.
        ValueError
            If integration_points has invalid dimensions or contains invalid element indices.
            
        """
        if not isinstance(integration_points, IntegrationPoints):
            raise TypeError(f"Input must be an IntegrationPoints instance, got {type(integration_points)}.")
        if not integration_points.n_dimensions == self.n_dimensions:
            raise ValueError(f"IntegrationPoints dimensions ({integration_points.dimensions}) do not match mesh dimensions ({self.n_dimensions}).")
        if numpy.max(integration_points.element_indices) >= self.n_elements:
            raise ValueError("IntegrationPoints contains element indices out of bounds for this mesh.")

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
        
        default = numpy.asarray(default, dtype=numpy.float64)
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

        default = numpy.asarray(default, dtype=numpy.float64)
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
        Lets consider the subclass :class:`pysdic.geometry.LinearTriangleMesh3D` of Mesh3D.
        
        Create a simple meshio Mesh object.

        .. code-block:: python

            import numpy as np
            import meshio
            from pysdic.geometry import LinearTriangleMesh3D

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            cells = [("triangle", np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]))]

            mesh = meshio.Mesh(points=points, cells=cells)

        Create a LinearTriangleMesh3D instance from the meshio Mesh object.

        .. code-block:: python

            mesh3d = LinearTriangleMesh3D.from_meshio(mesh)
            print(mesh3d.vertices)
            # Output: PointCloud3D with 4 points [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]

        """
        # Validations
        if not isinstance(mesh, meshio.Mesh):
            raise TypeError(f"Input must be a meshio Mesh object, got {type(mesh)}.")
        if not len(mesh.cells) == 1 or mesh.cells[0].data.ndim != 2 or mesh.cells[0].data.shape[1] != cls._n_vertices_per_element:
            raise ValueError("Invalid mesh structure.")
        if not isinstance(load_properties, bool):
            raise TypeError(f"load_properties must be a boolean, got {type(load_properties)}.")

        # Extract data
        vertices = PointCloud3D(mesh.points)
        connectivity = mesh.cells[0].data
        vertices_properties = {}
        elements_properties = {}
        
        # Extract properties if requested
        if load_properties:            
            for key, value in mesh.point_data.items():
                vertices_properties[key] = numpy.asarray(value).reshape(-1, 1) if value.ndim == 1 else numpy.asarray(value)

            for key, value in mesh.cell_data.items():
                elements_properties[key] = numpy.asarray(value[0]).reshape(-1, 1) if value[0].ndim == 1 else numpy.asarray(value[0])

        # Create Mesh3D instance
        return cls(vertices, connectivity, vertices_properties=vertices_properties, elements_properties=elements_properties)


    def to_meshio(self, save_properties: bool = True) -> meshio.Mesh:
        r"""
        Convert the Mesh3D instance to a meshio Mesh object.
        The mesh must not be empty.

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
        ValueError
            If the mesh is empty.

        Examples
        --------
        Lets consider the subclass :class:`pysdic.geometry.LinearTriangleMesh3D` of Mesh3D.

        Create a simple LinearTriangleMesh3D instance.

        .. code-block:: python

            import numpy as np
            from pysdic.geometry import LinearTriangleMesh3D, PointCloud3D

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])

            mesh3d = LinearTriangleMesh3D(PointCloud3D.from_array(points), connectivity)

        Convert the LinearTriangleMesh3D instance to a meshio Mesh object.

        .. code-block:: python

            mesh = mesh3d.to_meshio()
            print(mesh.points)
            # Output: [[0. 0. 0.] [1. 0. 0.] [0. 1. 0.] [0. 0. 1.]]
            
        """
        if self.n_vertices == 0 or self.n_elements == 0:
            raise ValueError("Cannot convert an empty mesh to meshio Mesh object.")
        if not isinstance(save_properties, bool):
            raise TypeError(f"save_properties must be a boolean, got {type(save_properties)}.")
        
        cells = [meshio.CellBlock(self.meshio_cell_type, data=self.connectivity)]
        
        if save_properties:
            point_data = {key: value for key, value in self._vertices_properties.items()}
            cell_data = {key: [value] for key, value in self._elements_properties.items()}
        else:
            point_data = {}
            cell_data = {}
        
        return meshio.Mesh(points=self.vertices.points, cells=cells, point_data=point_data, cell_data=cell_data)
    
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
        Lets consider the subclass :class:`pysdic.geometry.LinearTriangleMesh3D` of Mesh3D.

        Create a simple meshio Mesh object.

        .. code-block:: python

            import numpy as np
            import meshio
            from pysdic.geometry import LinearTriangleMesh3D

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            cells = [("triangle", np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]))]

            mesh = meshio.Mesh(points=points, cells=cells)

        Save the meshio Mesh object to a VTK file.

        .. code-block:: python

            mesh.write("simple_mesh.vtk", file_format="vtk")

        Create a LinearTriangleMesh3D instance from the VTK file.

        .. code-block:: python

            mesh3d = LinearTriangleMesh3D.from_vtk("simple_mesh.vtk")
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
        The mesh must not be empty.

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
            If the file format is not supported or the mesh is empty.

        Examples
        --------
        Lets consider the subclass :class:`pysdic.geometry.LinearTriangleMesh3D` of Mesh3D.

        Create a simple LinearTriangleMesh3D instance.

        .. code-block:: python

            import numpy as np
            from pysdic.geometry import LinearTriangleMesh3D, PointCloud3D

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = LinearTriangleMesh3D(PointCloud3D.from_array(points), connectivity)

        Save the LinearTriangleMesh3D instance to a VTK file.

        .. code-block:: python

            mesh3d.to_vtk("simple_mesh.vtk")
            # This will create a file named 'simple_mesh.vtk' in the current directory.
            
        """
        if self.n_vertices == 0 or self.n_elements == 0:
            raise ValueError("Cannot write an empty mesh to file.")
        if not isinstance(save_properties, bool):
            raise TypeError(f"save_properties must be a boolean, got {type(save_properties)}.")
        
        path = os.path.abspath(os.path.expanduser(filename))
        os.makedirs(os.path.dirname(path), exist_ok=True)

        mesh = self.to_meshio(save_properties=save_properties)
        mesh.write(filename, file_format="vtk")


    @classmethod
    def from_mesh(cls, other: Mesh3D, load_properties: bool = True) -> Mesh3D:
        r"""
        Create a Mesh3D instance from another Mesh3D instance.

        This method creates a new Mesh3D instance with the same vertices, connectivity, and properties as the input Mesh3D instance.

        .. seealso::

            - :meth:`copy` for creating a copy of the Mesh3D instance.

        Parameters
        ----------
        other : Mesh3D
            The input Mesh3D instance to copy.

        load_properties : bool, optional
            If True, properties are copied from the input Mesh3D instance, by default True.

        Returns
        -------
        Mesh3D
            A new Mesh3D instance created from the input Mesh3D instance.

        Raises
        ------
        TypeError
            If the input is not a Mesh3D instance.

        Examples
        --------
        Lets consider the subclass :class:`pysdic.geometry.LinearTriangleMesh3D` of Mesh3D.

        Create a simple LinearTriangleMesh3D instance.

        .. code-block:: python

            import numpy as np
            from pysdic.geometry import LinearTriangleMesh3D, PointCloud3D

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = LinearTriangleMesh3D(PointCloud3D.from_array(points), connectivity)

        Create a new LinearTriangleMesh3D instance from the existing one.

        .. code-block:: python

            new_mesh3d = LinearTriangleMesh3D.from_mesh(mesh3d)
            print(new_mesh3d.vertices)
            # Output: PointCloud3D with 4 points [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]      

        """
        if not isinstance(other, cls):
            raise TypeError(f"Input must be a Mesh3D instance, got {type(other)}.")
        if not isinstance(load_properties, bool):
            raise TypeError(f"load_properties must be a boolean, got {type(load_properties)}.")
        
        output = cls(
            vertices=other.vertices.copy(),
            connectivity=other.connectivity.copy(),
            vertices_properties={key: value.copy() for key, value in other._vertices_properties.items()} if load_properties else {},
            elements_properties={key: value.copy() for key, value in other._elements_properties.items()} if load_properties else {},
            internal_bypass=True  # Skip validation on copy
        )
        output.internal_bypass = other.internal_bypass
        return output
    

    @classmethod
    def from_empty(cls) -> Mesh3D:
        r"""
        Create an empty Mesh3D instance.

        This method creates a new Mesh3D instance with no vertices, no connectivity, and no properties.

        .. seealso::

            - :meth:`copy` for creating a copy of the Mesh3D instance.
            - :meth:`from_mesh` for creating a Mesh3D instance from another Mesh3D instance.

        Returns
        -------
        Mesh3D
            An empty Mesh3D instance.


        Examples
        --------
        Lets consider the subclass :class:`pysdic.geometry.LinearTriangleMesh3D` of Mesh3D.

        Create an empty LinearTriangleMesh3D instance.

        .. code-block:: python

            from pysdic.geometry import LinearTriangleMesh3D

            empty_mesh3d = LinearTriangleMesh3D.from_empty()

            print(empty_mesh3d.vertices)
            # Output: PointCloud3D with 0 points

        """
        return cls(
            vertices=PointCloud3D.from_empty(),
            connectivity=numpy.zeros((0, cls._n_vertices_per_element), dtype=int),
            vertices_properties={},
            elements_properties={}
        )


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


        Examples
        --------
        Lets consider the subclass :class:`pysdic.geometry.LinearTriangleMesh3D` of Mesh3D.

        Create a simple LinearTriangleMesh3D instance.

        .. code-block:: python

            import numpy as np
            from pysdic.geometry import Mesh3D, PointCloud3D

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = LinearTriangleMesh3D(PointCloud3D.from_array(points), connectivity)

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

        An alias for this property is `mesh.elements`.

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


        Examples
        --------
        Lets consider the subclass :class:`pysdic.geometry.LinearTriangleMesh3D` of Mesh3D.

        Create a simple LinearTriangleMesh3D instance.

        .. code-block:: python

            import numpy as np
            from pysdic.geometry import LinearTriangleMesh3D, PointCloud3D

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = LinearTriangleMesh3D(PointCloud3D.from_array(points), connectivity)

        Access the connectivity of the mesh.

        .. code-block:: python

            print(mesh3d.connectivity)
            # Output: [[0 1 2] [0 1 3] [0 2 3] [1 2 3]]

        """
        return self._connectivity
    
    @connectivity.setter
    def connectivity(self, value: numpy.ndarray) -> None:
        value = numpy.asarray(value, dtype=int)
        self._connectivity = value
        self._internal_check_connectivity()

    @property
    def elements(self) -> numpy.ndarray:
        r"""
        Alias for :attr:`connectivity` property.
        """
        return self.connectivity
    
    @elements.setter
    def elements(self, value: numpy.ndarray) -> None:
        self.connectivity = value

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

        
        Examples
        --------
        Lets consider the subclass :class:`pysdic.geometry.LinearTriangleMesh3D` of Mesh3D.

        Create a simple LinearTriangleMesh3D instance.

        .. code-block:: python

            import numpy as np
            from pysdic.geometry import LinearTriangleMesh3D, PointCloud3D

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = LinearTriangleMesh3D(PointCloud3D.from_array(points), connectivity)

        Get the number of vertices in the mesh.

        .. code-block:: python

            print(mesh3d.n_vertices)
            # Output: 4

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

        
        Examples
        --------
        Lets consider the subclass :class:`pysdic.geometry.LinearTriangleMesh3D` of Mesh3D.

        Create a simple LinearTriangleMesh3D instance.

        .. code-block:: python

            import numpy as np
            from pysdic.geometry import LinearTriangleMesh3D, PointCloud3D

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = LinearTriangleMesh3D(PointCloud3D.from_array(points), connectivity)

        Get the number of elements in the mesh.

        .. code-block:: python

            print(mesh3d.n_elements)
            # Output: 4

        """
        return self.connectivity.shape[0]
    
    @property
    def n_vertices_per_element(self) -> int:
        r"""
        Get the number of vertices per element in the mesh.

        Returns
        -------
        int
            The number of vertices per element.

        
        Examples
        --------
        Lets consider the subclass :class:`pysdic.geometry.LinearTriangleMesh3D` of Mesh3D.

        Create a simple LinearTriangleMesh3D instance.

        .. code-block:: python

            import numpy as np
            from pysdic.geometry import LinearTriangleMesh3D, PointCloud3D

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = LinearTriangleMesh3D(PointCloud3D.from_array(points), connectivity)

        Get the number of vertices per element in the mesh.

        .. code-block:: python

            print(mesh3d.n_vertices_per_element)
            # Output: 3
        
        """
        if self._n_vertices_per_element is None:
            raise NotImplementedError("Subclasses must implement n_vertices_per_element property.")
        return self._n_vertices_per_element

    @property
    def n_dimensions(self) -> int:
        r"""
        Get the topological dimension of the elements in the mesh.

        Returns
        -------
        int
            The topological dimension of the elements.


        Examples
        --------
        Lets consider the subclass :class:`pysdic.geometry.LinearTriangleMesh3D` of Mesh3D.

        Create a simple LinearTriangleMesh3D instance.

        .. code-block:: python

            import numpy as np
            from pysdic.geometry import LinearTriangleMesh3D, PointCloud3D

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = LinearTriangleMesh3D(PointCloud3D.from_array(points), connectivity)

        Get the topological dimension of the elements in the mesh.

        .. code-block:: python

            print(mesh3d.n_dimensions)
            # Output: 2
        
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
        Get a property associated with the vertices of the mesh with shape (N, A).

        ``N`` is the number of vertices and ``A`` is the size of the property.

        .. seealso::

            - :meth:`Mesh3D.set_vertices_property` to set a vertices property.

        Parameters
        ----------
        key : str
            The key of the property to retrieve.

        Returns
        -------
        numpy.ndarray or None
            The property associated with the vertices, or None if the property does not exist.

        
        Examples
        --------

        Lets consider the subclass :class:`pysdic.geometry.LinearTriangleMesh3D` of Mesh3D.

        Create a simple LinearTriangleMesh3D instance with a vertex property.

        .. code-block:: python

            import numpy as np
            from pysdic.geometry import LinearTriangleMesh3D, PointCloud3D

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])

            vertex_property = np.array([0.0, 1.0, 2.0, 3.0]).reshape(-1, 1) # Shape (4, 1)
            mesh3d.set_vertices_property("my_property", vertex_property)

        Extract the vertex property.

        .. code-block:: python

            prop = mesh3d.get_vertices_property("my_property")
            print(prop)
            # Output: [[0.] [1.] [2.] [3.]]    

        """
        return self._get_vertices_property(key, None, raise_error=False)
    
    def get_elements_property(self, key: str) -> Optional[numpy.ndarray]:
        r"""
        Get a property associated with the elements of the mesh with shape (M, B)

        ``M`` is the number of elements and ``B`` is the size of the property.

        .. seealso::

            - :meth:`Mesh3D.set_elements_property` to set an elements property.

        Parameters
        ----------
        key : str
            The key of the property to retrieve.

        Returns
        -------
        numpy.ndarray or None
            The property associated with the elements, or None if the property does not exist.

        Examples
        --------

        Lets consider the subclass :class:`pysdic.geometry.LinearTriangleMesh3D` of Mesh3D.

        Create a simple LinearTriangleMesh3D instance with an element property.

        .. code-block:: python

            import numpy as np
            from pysdic.geometry import LinearTriangleMesh3D, PointCloud3D

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])

            element_property = np.array([10.0, 20.0, 30.0, 40.0]).reshape(-1, 1) # Shape (4, 1)
            mesh3d.set_elements_property("my_element_property", element_property)

        Extract the element property.

        .. code-block:: python

            prop = mesh3d.get_elements_property("my_element_property")
            print(prop)
            # Output: [[10.] [20.] [30.] [40.]]

        """
        return self._get_elements_property(key, None, raise_error=False)
    
    def set_vertices_property(self, key: str, value: Optional[numpy.ndarray]) -> None:
        r"""
        Set a property associated with the vertices of the mesh with shape (N, A).

        ``N`` is the number of vertices and ``A`` is the size of the property.

        .. note::

            Even if the size of the property is 1, the property must be provided as a 2D array of shape (N, 1).

        .. seealso::

            - :meth:`Mesh3D.get_vertices_property` to get a vertices property.

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

        
        Examples
        --------

        Lets consider the subclass :class:`pysdic.geometry.LinearTriangleMesh3D` of Mesh3D.

        Create a simple LinearTriangleMesh3D instance.

        .. code-block:: python

            import numpy as np
            from pysdic.geometry import LinearTriangleMesh3D, PointCloud3D

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = LinearTriangleMesh3D(PointCloud3D.from_array(points), connectivity)

        Set a vertex property.

        .. code-block:: python

            vertex_property = np.array([0.0, 1.0, 2.0, 3.0]).reshape(-1, 1) # Shape (4, 1)
            mesh3d.set_vertices_property("my_property", vertex_property)

            prop = mesh3d.get_vertices_property("my_property")
            print(prop)
            # Output: [[0.] [1.] [2.] [3.]]
        
        """
        if value is None:
            if key in self._vertices_properties:
                del self._vertices_properties[key]
            return
        
        value = numpy.asarray(value, dtype=numpy.float64)
        self._vertices_properties[key] = value
        self._internal_check_vertices_property(key)


    def set_elements_property(self, key: str, value: Optional[numpy.ndarray]) -> None:
        r"""
        Set a property associated with the elements of the mesh with shape (M, B).

        ``M`` is the number of elements and ``B`` is the size of the property.

        .. note::

            Even if the size of the property is 1, the property must be provided as a 2D array of shape (M, 1).

        .. seealso::

            - :meth:`Mesh3D.get_elements_property` to get an elements property.

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


        Examples
        --------

        Lets consider the subclass :class:`pysdic.geometry.LinearTriangleMesh3D` of Mesh3D.

        Create a simple LinearTriangleMesh3D instance.

        .. code-block:: python

            import numpy as np
            from pysdic.geometry import LinearTriangleMesh3D, PointCloud3D

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = LinearTriangleMesh3D(PointCloud3D.from_array(points), connectivity)

        Set an element property.

        .. code-block:: python

            element_property = np.array([10.0, 20.0, 30.0, 40.0]).reshape(-1, 1) # Shape (4, 1)
            mesh3d.set_elements_property("my_element_property", element_property)

            prop = mesh3d.get_elements_property("my_element_property")
            print(prop)
            # Output: [[10.] [20.] [30.] [40.]]

        """
        if value is None:
            if key in self._elements_properties:
                del self._elements_properties[key]
            return
        
        value = numpy.asarray(value, dtype=numpy.float64)
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
    # Manipulate Mesh geometry
    # ======================= 
    def add_elements(self, new_connectivity: numpy.ndarray) -> None:
        r"""
        Add new elements to the mesh by appending new connectivity entries.

        .. note::

            The new elements will be added to the end of the existing connectivity array.
            The elements properties stored in the mesh are extended with default ``numpy.nan`` values for the new elements.

        Parameters
        ----------
        new_connectivity : numpy.ndarray
            An array of shape (P, K) containing the connectivity of the new elements to add,
            where P is the number of new elements and K is the number of vertices per element.

        Raises
        ------
        ValueError
            If new_connectivity does not have the correct shape or contains invalid indices.

            
        Examples
        --------

        Lets consider the subclass :class:`pysdic.geometry.LinearTriangleMesh3D` of Mesh3D.

        Create a simple LinearTriangleMesh3D instance.

        .. code-block:: python

            import numpy as np
            from pysdic.geometry import LinearTriangleMesh3D, PointCloud3D

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
            connectivity = np.array([[0, 1, 2]])
            mesh3d = LinearTriangleMesh3D(PointCloud3D.from_array(points), connectivity)

        Add some properties to the elements.

        .. code-block:: python

            element_property = np.array([10.0]).reshape(-1, 1) # Shape (1, 1)
            mesh3d.set_elements_property("my_element_property", element_property)

        Add new elements to the mesh.

        .. code-block:: python

            new_connectivity = np.array([[0, 1, 2], [1, 2, 0]])
            mesh3d.add_elements(new_connectivity)

            print(mesh3d.connectivity)
            # Output: [[0 1 2] [0 1 2] [1 2 0]]

            print(mesh3d.elements_properties)
            # Output: {'my_element_property': [[10.], [nan], [nan]]}

        """
        # Check new connectivity
        new_connectivity = numpy.asarray(new_connectivity, dtype=int)
        if new_connectivity.ndim != 2 or new_connectivity.shape[1] != self.n_vertices_per_element:
            raise ValueError(
                f"new_connectivity must be a 2D array with shape (P, {self.n_vertices_per_element}), "
                f"where P is the number of new elements. Got shape {new_connectivity.shape}."
            )

        # Bypass checks during addition
        current_internal_bypass = self.internal_bypass
        self.internal_bypass = True  # Bypass checks during addition

        # Combine connectivity
        combined_connectivity = numpy.vstack((self.connectivity, new_connectivity))
        self.connectivity = combined_connectivity
        
        # Extend elements properties
        n_new_elements = new_connectivity.shape[0]
        for key, value in self._elements_properties.items():
            n_attributes = value.shape[1]
            extension = numpy.full((n_new_elements, n_attributes), numpy.nan, dtype=numpy.float64)
            self._elements_properties[key] = numpy.vstack((value, extension))

        self.internal_bypass = current_internal_bypass  # Restore original bypass state
        self._internal_check_connectivity()
        self._internal_check_elements_properties()


    def add_vertices(self, new_vertices: Union[PointCloud3D, numpy.ndarray]) -> None:
        r"""
        Add new vertices to the mesh by appending new vertex coordinates.

        .. note::

            The new vertices will be added to the end of the existing vertex list.
            The vertices properties stored in the mesh are extended with default ``numpy.nan`` values for the new vertices.

        Parameters
        ----------
        new_vertices : Union[PointCloud3D, numpy.ndarray]
            An array of shape (Q, 3) containing the coordinates of the new vertices to add,
            where Q is the number of new vertices.

        Raises
        ------
        ValueError
            If new_vertices does not have the correct shape.

            
        Examples
        --------

        Lets consider the subclass :class:`pysdic.geometry.LinearTriangleMesh3D` of Mesh3D.

        Create a simple LinearTriangleMesh3D instance.

        .. code-block:: python

            import numpy as np
            from pysdic.geometry import LinearTriangleMesh3D, PointCloud3D

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
            connectivity = np.array([[0, 1, 2]])
            mesh3d = LinearTriangleMesh3D(PointCloud3D.from_array(points), connectivity)

        Add new vertices to the mesh.

        .. code-block:: python

            new_vertices = np.array([[0, 0, 1], [1, 1, 1]])
            mesh3d.add_vertices(new_vertices)

            print(mesh3d.vertices)
            # Output: PointCloud3D with 5 points [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]

        """
        if not isinstance(new_vertices, PointCloud3D):
            new_vertices = PointCloud3D.from_array(new_vertices)
        
        # Bypass checks during addition
        current_internal_bypass = self.internal_bypass
        self.internal_bypass = True  # Bypass checks during addition

        # Combine vertices
        self.vertices = self.vertices.concatenate(new_vertices)

        # Extend vertices properties
        n_new_vertices = new_vertices.n_points
        for key, value in self._vertices_properties.items():
            n_attributes = value.shape[1]
            extension = numpy.full((n_new_vertices, n_attributes), numpy.nan, dtype=numpy.float64)
            self._vertices_properties[key] = numpy.vstack((value, extension))

        self.internal_bypass = current_internal_bypass  # Restore original bypass state
        self._internal_check_vertices()
        self._internal_check_vertices_properties()


    def are_used_vertices(self, vertex_indices: numpy.ndarray) -> numpy.ndarray:
        r"""
        Check if multiple vertices are used in the connectivity of the mesh.

        Parameters
        ----------
        vertex_indices : numpy.ndarray
            An array of shape (R,) containing the indices of the vertices to check.

        Returns
        -------
        numpy.ndarray
            A boolean array of shape (R,) where each entry indicates whether the corresponding vertex is used in the connectivity.

        Raises
        ------
        ValueError
            If any vertex_index is out of bounds.

        
        Examples
        --------
        Lets consider the subclass :class:`pysdic.geometry.LinearTriangleMesh3D` of Mesh3D.

        Create a simple LinearTriangleMesh3D instance.

        .. code-block:: python

            import numpy as np
            from pysdic.geometry import LinearTriangleMesh3D, PointCloud3D

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = LinearTriangleMesh3D(PointCloud3D.from_array(points), connectivity)

        Check if vertices 2, 4, and 0 are used in the connectivity.

        .. code-block:: python

            vertex_indices = np.array([2, 4, 0])
            used_flags = mesh3d.are_used_vertices(vertex_indices)
            print(used_flags)
            # Output: [ True False  True]

        """
        vertex_indices = numpy.asarray(vertex_indices, dtype=int)
        if vertex_indices.ndim != 1:
            raise ValueError("vertex_indices must be a 1D array.")
        if numpy.any(vertex_indices < 0) or numpy.any(vertex_indices >= self.n_vertices):
            raise ValueError("One or more vertex_index is out of bounds.")

        used_flags = numpy.array([numpy.any(self.connectivity == idx) for idx in vertex_indices], dtype=bool)
        return used_flags


    def is_empty(self) -> bool:
        r"""
        Check if the mesh is empty (i.e., has no vertices or no elements).

        Returns
        -------
        bool
            True if the mesh is empty, False otherwise.

        
        Examples
        --------
        Lets consider the subclass :class:`pysdic.geometry.LinearTriangleMesh3D` of Mesh3D.

        Create an empty LinearTriangleMesh3D instance.

        .. code-block:: python

            from pysdic.geometry import LinearTriangleMesh3D

            empty_mesh3d = LinearTriangleMesh3D.from_empty()
            print(empty_mesh3d.is_empty())
            # Output: True

        Create a non-empty LinearTriangleMesh3D instance.

        .. code-block:: python

            import numpy as np
            from pysdic.geometry import LinearTriangleMesh3D, PointCloud3D

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = LinearTriangleMesh3D(PointCloud3D.from_array(points), connectivity)

            print(mesh3d.is_empty())
            # Output: False

        """
        return self.n_vertices == 0 or self.n_elements == 0
    

    def keep_elements(self, element_indices: numpy.ndarray) -> None:
        r"""
        Keep only the specified elements in the mesh by specifying their indices in the connectivity array.

        .. note::

            The elements properties stored in the mesh are updated accordingly to keep only the properties of the kept elements.

        .. seealso::

            - :meth:`remove_elements` to remove specified elements from the mesh.

        Parameters
        ----------
        element_indices : numpy.ndarray
            An array of shape (R,) containing the indices of the elements to keep,
            where R is the number of elements to keep.

        Raises
        ------
        ValueError
            If element_indices does not have the correct shape or contains invalid indices.

        
        Examples
        --------
        Lets consider the subclass :class:`pysdic.geometry.LinearTriangleMesh3D` of Mesh3D.

        Create a simple LinearTriangleMesh3D instance.

        .. code-block:: python

            import numpy as np
            from pysdic.geometry import LinearTriangleMesh3D, PointCloud3D
        
            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = LinearTriangleMesh3D(PointCloud3D.from_array(points), connectivity)

        Add some properties to the elements.

        .. code-block:: python

            element_property = np.array([10.0, 20.0, 30.0, 40.0]).reshape(-1, 1) # Shape (4, 1)
            mesh3d.set_elements_property("my_element_property", element_property)

        Keep only elements with indices 0 and 2 in the mesh.

        .. code-block:: python

            element_indices = np.array([0, 2])
            mesh3d.keep_elements(element_indices)

            print(mesh3d.connectivity)
            # Output: [[0 1 2] [0 2 3]]
            print(mesh3d.elements_properties)
            # Output: {'my_element_property': [[10.] [30.]]}

        """
        element_indices = numpy.asarray(element_indices, dtype=int)
        if element_indices.ndim != 1:
            raise ValueError(
                f"element_indices must be a 1D array with shape (R,), "
                f"where R is the number of elements to keep. Got shape {element_indices.shape}."
            )
        
        # Create the mask of elements to remove
        all_indices = numpy.arange(self.n_elements)
        mask_to_remove = numpy.ones(self.n_elements, dtype=bool)
        mask_to_remove[element_indices] = False
        remove_indices = all_indices[mask_to_remove]

        self.remove_elements(remove_indices)


    def remove_elements(self, element_indices: numpy.ndarray) -> None:
        r"""
        Remove elements from the mesh by specifying their indices in the connectivity array.

        .. note::

            The elements properties stored in the mesh are updated accordingly to remove the properties of the removed elements.

        Parameters
        ----------
        element_indices : numpy.ndarray
            An array of shape (R,) containing the indices of the elements to remove,
            where R is the number of elements to remove.

        Raises
        ------
        ValueError
            If element_indices does not have the correct shape or contains invalid indices.

            
        Examples
        --------
        Lets consider the subclass :class:`pysdic.geometry.LinearTriangleMesh3D` of Mesh3D.

        Create a simple LinearTriangleMesh3D instance.

        .. code-block:: python

            import numpy as np
            from pysdic.geometry import LinearTriangleMesh3D, PointCloud3D

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = LinearTriangleMesh3D(PointCloud3D.from_array(points), connectivity)

        Add some properties to the elements.

        .. code-block:: python

            element_property = np.array([10.0, 20.0, 30.0, 40.0]).reshape(-1, 1) # Shape (4, 1)
            mesh3d.set_elements_property("my_element_property", element_property)

        Remove elements with indices 1 and 3 from the mesh.

        .. code-block:: python

            element_indices = np.array([1, 3])
            mesh3d.remove_elements(element_indices)

            print(mesh3d.connectivity)
            # Output: [[0 1 2] [0 2 3]]

            print(mesh3d.elements_properties)
            # Output: {'my_element_property': [[10.] [30.]]}

        """
        element_indices = numpy.asarray(element_indices, dtype=int)
        if element_indices.ndim != 1:
            raise ValueError(
                f"element_indices must be a 1D array with shape (R,), "
                f"where R is the number of elements to remove. Got shape {element_indices.shape}."
            )
        if numpy.any(element_indices < 0) or numpy.any(element_indices >= self.n_elements):
            raise ValueError("element_indices contains invalid indices.")

        # Bypass checks during removal
        unique_indices = numpy.unique(element_indices)

        current_internal_bypass = self.internal_bypass
        self.internal_bypass = True  # Bypass checks during removal

        # Remove elements entries
        mask = numpy.ones(self.n_elements, dtype=bool)
        mask[unique_indices] = False
        self.connectivity = self.connectivity[mask, :]

        # Update elements properties
        for key, value in self._elements_properties.items():
            self._elements_properties[key] = value[mask, :]

        self.internal_bypass = current_internal_bypass  # Restore original bypass state
        self._internal_check_connectivity()
        self._internal_check_elements_properties()


    def remove_vertices(self, vertex_indices: numpy.ndarray) -> None:
        r"""
        Remove vertices from the mesh by specifying their indices.

        .. note::

            The vertices properties stored in the mesh are updated accordingly to remove references to the removed vertices.

        .. warning::

            Cannot remove vertices that are used in the connectivity.

        .. seealso::

            - :meth:`remove_unused_vertices` to remove all unused vertices.

        Parameters
        ----------
        vertex_indices : numpy.ndarray
            An array of shape (R,) containing the indices of the vertices to remove,
            where R is the number of vertices to remove.

        Raises
        ------
        ValueError
            If vertex_indices does not have the correct shape, contains invalid indices, or if any vertex is used in the connectivity.


        Examples
        --------
        Lets consider the subclass :class:`pysdic.geometry.LinearTriangleMesh3D` of Mesh3D.

        Create a simple LinearTriangleMesh3D instance.

        .. code-block:: python

            import numpy as np
            from pysdic.geometry import LinearTriangleMesh3D, PointCloud3D

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = LinearTriangleMesh3D(PointCloud3D.from_array(points), connectivity)

        Add some properties to the vertices.

        .. code-block:: python

            vertex_property = np.array([0.0, 1.0, 2.0, 3.0, 4.0]).reshape(-1, 1) # Shape (5, 1)
            mesh3d.set_vertices_property("my_vertex_property", vertex_property)

        Remove vertex with index 4 from the mesh.

        .. code-block:: python

            vertex_indices = np.array([4])
            mesh3d.remove_vertices(vertex_indices)

            print(mesh3d.vertices)
            # Output: PointCloud3D with 4 points [[0 0 0] [1 0 0] [0 1 0] [0 0 1]]

            print(mesh3d.vertices_properties)
            # Output: {'my_vertex_property': [[0.] [1.] [2.] [3.]]}

        """
        vertex_indices = numpy.asarray(vertex_indices, dtype=int)
        if vertex_indices.ndim != 1:
            raise ValueError(
                f"vertex_indices must be a 1D array with shape (R,), "
                f"where R is the number of vertices to remove. Got shape {vertex_indices.shape}."
            )
        if numpy.any(vertex_indices < 0) or numpy.any(vertex_indices >= self.n_vertices):
            raise ValueError("vertex_indices contains invalid indices.")
        
        # Unique indices
        unique_indices = numpy.unique(vertex_indices)
        sorted_unique_indices = numpy.sort(unique_indices)

        # Create a array "shift" that indicates how many vertices have been removed before each index
        shift = numpy.zeros(self.n_vertices, dtype=int)
        shift[sorted_unique_indices] = 1
        shift = numpy.cumsum(shift)

        # Check if any vertex is used in connectivity
        used_flag = self.are_used_vertices(sorted_unique_indices)
        if numpy.any(used_flag):
            raise ValueError("Cannot remove vertices that are used in the connectivity.")

        # Bypass checks during removal
        current_internal_bypass = self.internal_bypass
        self.internal_bypass = True  # Bypass checks during removal

        # Remove vertices
        mask = numpy.ones(self.n_vertices, dtype=bool)
        mask[sorted_unique_indices] = False

        self.vertices = self.vertices.remove_points_at(sorted_unique_indices)

        # Update vertices properties
        for key, value in self._vertices_properties.items():
            self._vertices_properties[key] = value[mask, :]

        # Update connectivity
        updated_connectivity = self.connectivity - shift[self.connectivity]
        self.connectivity = updated_connectivity

        self.internal_bypass = current_internal_bypass  # Restore original bypass state
        self._internal_check_vertices()
        self._internal_check_vertices_properties()
        self._internal_check_connectivity()
        self._internal_check_elements_properties()


    def remove_unused_vertices(self) -> None:
        r"""
        Remove all vertices that are not used in the connectivity of the mesh.

        .. note::

            The vertices properties stored in the mesh are updated accordingly to remove references to the removed vertices.

        .. seealso::

            - :meth:`remove_vertices` to remove specific vertices by their indices.
        
        Examples
        --------
        Lets consider the subclass :class:`pysdic.geometry.LinearTriangleMesh3D` of Mesh3D.

        Create a simple LinearTriangleMesh3D instance.

        .. code-block:: python

            import numpy as np
            from pysdic.geometry import LinearTriangleMesh3D, PointCloud3D

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = LinearTriangleMesh3D(PointCloud3D.from_array(points), connectivity)

        Add some properties to the vertices.

        .. code-block:: python

            vertex_property = np.array([0.0, 1.0, 2.0, 3.0, 4.0]).reshape(-1, 1) # Shape (5, 1)
            mesh3d.set_vertices_property("my_vertex_property", vertex_property)

        Remove all unused vertices from the mesh.

        .. code-block:: python

            mesh3d.remove_unused_vertices()

            print(mesh3d.vertices)
            # Output: PointCloud3D with 4 points [[0 0 0] [1 0 0] [0 1 0] [0 0 1]]

            print(mesh3d.vertices_properties)
            # Output: {'my_vertex_property': [[0.] [1.] [2.] [3.]]}

        """
        used_flags = self.are_used_vertices(numpy.arange(self.n_vertices))
        unused_indices = numpy.where(~used_flags)[0]
        if unused_indices.size > 0:
            self.remove_vertices(unused_indices)





    # =======================
    # Public Methods
    # =======================    
    def copy(self) -> Mesh3D:
        r"""
        Create a deep copy of the Mesh3D instance.

        .. note::

            This method creates a new Mesh3D instance with copies of the vertices, connectivity, and properties with the class method :func:`from_mesh`.

        Returns
        -------
        Mesh3D
            A deep copy of the Mesh3D instance.
        """
        return self.from_mesh(self)
    
    
    def interpolate_property_at_natural_coordinates(
            self, 
            natural_coords: numpy.ndarray, 
            element_indices: numpy.ndarray, 
            *, 
            property_key: Optional[str] = None, 
            property_array: Optional[numpy.ndarray] = None,
            vertices_coordinates: bool = False,
            use_elements_property: bool = False,
        ) -> numpy.ndarray:
        r"""
        Interpolate a vertices property at given natural coordinates and element indices.

        This method is a convenience wrapper around :meth:`interpolate_property_at_integration_points`
        that auto constructs the :class:`IntegrationPoints` instance from the provided natural coordinates and element indices.
        For mathematical details and explanations, see :meth:`interpolate_property_at_integration_points`.

        The property to interpolate can be provided by one of the three options:

        - property_key : The name of the vertices property (or elements property if ``use_elements_property`` is True) to evaluate.
        - property_array : An array of shape (N, A) containing the vertices property values (or elements property values if ``use_elements_property`` is True).
        - vertices_coordinates : If True, the vertices coordinates are used as the property to interpolate.

        .. note::

            Even if the property is 1-dimensional (A=1), the input and output arrays must have 2 dimensions (Np, 1) and not (Np,).

        .. note::

            The integration points with "-1" element index are ignored and the corresponding property values are setted to NaN.

        .. note::

            If ``use_elements_property`` is True, the property array must have shape (M, B) where M is the number of elements.
            No interpolation is performed in this case, and the property values are mapped directly to the integration points based on their element indices.    

        Parameters
        ----------
        natural_coords : numpy.ndarray
            An array of shape (Np, d) containing the natural coordinates.

        element_indices : numpy.ndarray
            An array of shape (Np,) containing the element indices for each point.

        property_key : Optional[str], optional
            The name of the vertices property to evaluate. If None, property_array must be provided or vertices_coordinates must be True, by default None.

        property_array : Optional[numpy.ndarray], optional
            An array of shape (N, A) containing the vertices property values. If None, property_key must be provided or vertices_coordinates must be True, by default None.

        vertices_coordinates : bool, optional
            If True, the vertices coordinates are used as the property to interpolate. If False, property_key or property_array must be provided, by default False.

        use_elements_property : bool, optional
            If True, the property is retrieved from the elements properties instead of vertices properties. Cannot be used with vertices_coordinates=True.
            In this case no interpolation is performed and the element property values are mapped directly to the integration points based on their element indices. by default False.

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
        
        
        Examples
        --------

        Lets consider the subclass :class:`pysdic.geometry.LinearTriangleMesh3D` of Mesh3D.

        Create a simple LinearTriangleMesh3D instance with a vertices property.

        .. code-block:: python

            import numpy as np
            from pysdic.geometry import LinearTriangleMesh3D, PointCloud3D

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = LinearTriangleMesh3D(PointCloud3D.from_array(points), connectivity)

            temperature = np.array([100, 200, 150, 250]).reshape(-1, 1)  # Shape (4, 1)
            mesh3d.set_vertices_property("temperature", temperature)

        Evaluate the "temperature" property at given natural coordinates in elements 0 and 3.

        .. code-block:: python

            natural_coords = np.array([[0.3, 0.3], [0.2, 0.5]])
            element_indices = np.array([0, 3])

            temperatures = mesh3d.interpolate_property_at_natural_coordinates(natural_coords, element_indices, property_key="temperature")
            print(temperatures)
            # Output: [[145.]
            #          [215.]]

        """
        integration_points = IntegrationPoints(natural_coords, element_indices)
        return self.interpolate_property_at_integration_points(
            integration_points, 
            property_key=property_key, 
            property_array=property_array,
            vertices_coordinates=vertices_coordinates,
            use_elements_property=use_elements_property,
        )
        

    def interpolate_property_at_integration_points(
            self, 
            integration_points: IntegrationPoints, 
            *, 
            property_key: Optional[str] = None, 
            property_array: Optional[numpy.ndarray] = None,
            vertices_coordinates: bool = False,
            use_elements_property: bool = False,
        ) -> numpy.ndarray:
        r"""
        Interpolate a vertices property at given natural coordinates and element indices from an :class:`IntegrationPoints` instance.

        Lets consider a mesh with K vertices per element, and d natural coordinates.
        The given natural coordinates should be (Np, d) where Np is the number of points to evaluate.
        The element_indices should be (Np,) where each entry is the index of the element in which to evaluate the natural coordinates.
        The returned property values will be of shape (Np, A) where A is the number of attributes for that property.

        The evaluation of the property at the given natural coordinates is performed using the shape functions:

        .. math::

            P = \sum_{i=1}^{K} N_i(\xi, \eta, \zeta, ...) P_i

        where :math:`N_i` are the shape functions associated with each vertex, and :math:`P_i` are the property values at the vertices of the element.

        With matrix notation, this can be expressed as:

        .. math::

            P_{p} = N_{f} P_{n}

        where :math:`P_{p}` is the property at the points to evaluate with shape (Np, A), 
        :math:`N_{f}` is the matrix of shape functions evaluated at the natural coordinates with shape (Np, N), 
        and :math:`P_{n}` is the property at the vertices of the elements with shape (N, A).

        The property to interpolate can be provided by one of the three options:

        - property_key : The name of the vertices property (or elements property if ``use_elements_property`` is True) to evaluate.
        - property_array : An array of shape (N, A) containing the vertices property values (or elements property values if ``use_elements_property`` is True).
        - vertices_coordinates : If True, the vertices coordinates are used as the property to interpolate.

        .. note::

            Even if the property is 1-dimensional (A=1), the input and output arrays must have 2 dimensions (Np, 1) and not (Np,).

        .. note::

            The integration points with "-1" element index are ignored and the corresponding property values are setted to NaN.

        .. note::

            If ``use_elements_property`` is True, the property array must have shape (M, B) where M is the number of elements.
            No interpolation is performed in this case, and the property values are mapped directly to the integration points based on their element indices.

        .. seealso::

            - :meth:`Mesh3D.shape_functions` to compute the shape functions at given natural coordinates.
            - :meth:`Mesh3D.project_integration_property_to_vertices` to project an integration property to vertices (inverse operation).
            - :meth:`Mesh3D.interpolate_property_at_natural_coordinates` to interpolate property at given natural coordinates and element indices.

        Parameters
        ----------
        integration_points : IntegrationPoints
            An IntegrationPoints instance containing natural coordinates and element indices.

        property_key : Optional[str], optional
            The name of the vertices property to evaluate. If None, property_array must be provided or vertices_coordinates must be True, by default None.

        property_array : Optional[numpy.ndarray], optional
            An array of shape (N, A) containing the vertices property values. If None, property_key must be provided or vertices_coordinates must be True, by default None.

        vertices_coordinates : bool, optional
            If True, the vertices coordinates are used as the property to interpolate. If False, property_key or property_array must be provided, by default False.

        use_elements_property : bool, optional
            If True, the property is retrieved from the elements properties instead of vertices properties. Cannot be used with vertices_coordinates=True.
            In this case no interpolation is performed and the element property values are mapped directly to the integration points based on their element indices. by default False.

        Returns
        -------
        numpy.ndarray
            An array of shape (Np, A) containing the evaluated property values.

        Raises
        ------
        TypeError
            If integration_points is not an IntegrationPoints instance.

        ValueError
            If neither or both property_key and property_array are provided.
            If property_key does not exist in vertices properties.
            If property_array does not have the correct shape.
            If the integration_points contain invalid element indices.

        
        Examples
        --------

        Lets consider the subclass :class:`pysdic.geometry.LinearTriangleMesh3D` of Mesh3D.

        Create a simple LinearTriangleMesh3D instance with a vertices property.

        .. code-block:: python

            import numpy as np
            from pysdic.geometry import LinearTriangleMesh3D, PointCloud3D, IntegrationPoints

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = LinearTriangleMesh3D(PointCloud3D.from_array(points), connectivity)

            temperature = np.array([100, 200, 150, 250]).reshape(-1, 1)  # Shape (4, 1)
            mesh3d.set_vertices_property("temperature", temperature)

            natural_coords = np.array([[0.3, 0.3], [0.2, 0.5], [0.1, 0.1]])
            element_indices = np.array([0, 3, -1])  # The last point is invalid
            integration_points = IntegrationPoints(natural_coords, element_indices)

        Evaluate the "temperature" property at given natural coordinates.

        .. code-block:: python

            temperatures = mesh3d.interpolate_property_at_integration_points(integration_points, property_key="temperature")
            print(temperatures)
            # Output: [[145.]
            #          [215.]
            #          [ nan]]
    
        """
        self._check_integration_points(integration_points)
        
        # Get the property array
        count = sum([property_key is not None, property_array is not None, vertices_coordinates])
        if count == 0:
            raise ValueError("One of property_key, property_array or vertices_coordinates must be provided.")
        if count > 1:
            raise ValueError("Either property_key, property_array or vertices_coordinates must be provided, but not several.")
        if not isinstance(use_elements_property, bool):
            raise ValueError("use_elements_property must be a boolean value.")
        if use_elements_property and vertices_coordinates:
            raise ValueError("Cannot use vertices_coordinates=True with use_elements_property=True as vertices coordinates are not defined per element.")
        
        # Extract property array
        if use_elements_property:
            property_array = self._get_element_property(property_key, property_array, raise_error=True)  # (M, B)
        elif not vertices_coordinates:
            property_array = self._get_vertices_property(property_key, property_array, raise_error=True)  # (N, A)
        else:
            property_array = self.vertices.points  # (N, 3)

        # Evaluate property at integration points
        evaluated_property: numpy.ndarray = numpy.full((len(integration_points), property_array.shape[1]), numpy.nan, dtype=numpy.float64)
        valid_mask = integration_points.element_indices != -1

        # Case of vertices property interpolation
        if not use_elements_property:
            # Get the shape functions and vertices coordinates
            shape_functions, _ = self.shape_functions(integration_points.natural_coordinates[valid_mask, :])  # (Np, K)
            vertices_property = property_array[self.connectivity[integration_points.element_indices[valid_mask], :], :]  # (Np, K, A)

            # Compute evaluated property
            property = numpy.einsum('ij,ijk->ik', shape_functions, vertices_property)  # (Np, A)

        # Case of elements property mapping
        else:
            property = property_array[integration_points.element_indices[valid_mask], :].copy()  # (Np, B)

        evaluated_property[valid_mask, :] = property
        return evaluated_property

    
    def get_vertices_coordinates(self, element_indices: numpy.ndarray) -> numpy.ndarray:
        r"""
        Get the coordinates of the vertices for the specified elements.

        If the element_indices is of shape (M,), the returned array will be of shape (M, K, 3),
        where K is the number of vertices per element.

        .. note::

            If an element index is "-1", NaN values will be returned for the corresponding vertices coordinates.

        Parameters
        ----------
        element_indices : numpy.ndarray
            An array of shape (M,) containing the element indices. Must be a 1D array.

        Returns
        -------
        numpy.ndarray
            An array of shape (M, K, 3) containing the vertex coordinates for each element.

        Raises
        ------
        ValueError
            If element_indices is not a 1D array or contains invalid element indices.


        Examples
        --------
        Lets consider the subclass :class:`pysdic.geometry.LinearTriangleMesh3D` of Mesh3D.

        Create a simple LinearTriangleMesh3D instance.

        .. code-block:: python

            import numpy as np
            from pysdic.geometry import LinearTriangleMesh3D, PointCloud3D
            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = LinearTriangleMesh3D(PointCloud3D.from_array(points), connectivity)

        Get the coordinates of the vertices for elements 0 and 2.

        .. code-block:: python

            element_indices = np.array([0, 2])
            vertices_coords = mesh3d.get_vertices_coordinates(element_indices)
            print(vertices_coords)
            # Output: [[[0 0 0]
            #           [1 0 0]
            #           [0 1 0]]
            #          [[0 0 0]
            #           [0 1 0]
            #           [0 0 1]]]

        """
        element_indices = numpy.asarray(element_indices, dtype=int)
        if element_indices.ndim != 1:
            raise ValueError(f"Element indices must be a 1D array, got {element_indices.ndim}D array.")
        if numpy.any(element_indices < -1) or numpy.any(element_indices >= self.n_elements):
            raise ValueError("element_indices contain invalid element indices.")
        return self.vertices.points[self.connectivity[element_indices, :], :]
    

    def integration_points_to_global_coordinates(self, integration_points: IntegrationPoints) -> numpy.ndarray:
        r"""
        Transform natural coordinates to global coordinates from an :class:`IntegrationPoints` instance.

        This method is a convenience wrapper around :meth:`interpolate_property_at_integration_points` with ``vertices_coordinates=True``.

        The transformation from natural coordinates to global coordinates is given by:

        .. math::

            X = \sum_{i=1}^{K} N_i(\xi, \eta, \zeta, ...) X_i

        where :math:`N_i` are the shape functions associated with each vertex, and :math:`X_i` are the coordinates of the vertices of the element.

        .. note::

            The integration points with "-1" element index are ignored and the corresponding coordinates are set to NaN.

        .. seealso::

            - :meth:`Mesh3D.shape_functions` to compute the shape functions at given natural coordinates.
            - :meth:`Mesh3D.interpolate_property_at_integration_points` to interpolate a vertices property at given integration points.
            - :meth:`natural_to_global_coordinates` to transform natural coordinates to global coordinates from natural coordinates and element indices.

        Parameters
        ----------
        integration_points : IntegrationPoints
            An IntegrationPoints instance containing natural coordinates and element indices.

        Returns
        -------
        numpy.ndarray
            An array of shape (Np, 3) containing the global coordinates.


        Examples
        --------
        Lets consider the subclass :class:`pysdic.geometry.LinearTriangleMesh3D` of Mesh3D.

        Create a simple LinearTriangleMesh3D instance.

        .. code-block:: python

            import numpy as np
            from pysdic.geometry import LinearTriangleMesh3D, PointCloud3D, IntegrationPoints

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = LinearTriangleMesh3D(PointCloud3D.from_array(points), connectivity)

            natural_coords = np.array([[0.3, 0.3], [0.2, 0.5], [0.1, 0.1]])
            element_indices = np.array([0, 3, -1])  # The last point is invalid
            integration_points = IntegrationPoints(natural_coords, element_indices)

        Transform natural coordinates to global coordinates.

        .. code-block:: python

            global_points = mesh3d.natural_to_global_coordinates_points(integration_points)
            print(global_points.points)
            # Output: [[0.3 0.3 0. ]
            #          [0.3 0.2 0.5]
            #          [ nan  nan  nan]]
        """
        return self.interpolate_property_at_integration_points(
            integration_points, 
            vertices_coordinates=True
        )


    def natural_to_global_coordinates(self, natural_coords: numpy.ndarray, element_indices: numpy.ndarray) -> numpy.ndarray:
        r"""
        Transform natural coordinates to global coordinates for specified natural coordinates and element indices.

        This method is a convenience wrapper around :meth:`interpolate_property_at_integration_points`
        that auto constructs the :class:`IntegrationPoints` instance from the provided natural coordinates and element indices with ``vertices_coordinates=True``.

        The transformation from natural coordinates to global coordinates is given by:

        .. math::

            X = \sum_{i=1}^{K} N_i(\xi, \eta, \zeta, ...) X_i

        where :math:`N_i` are the shape functions associated with each vertex, and :math:`X_i` are the coordinates of the vertices of the element.

        .. note::

            The integration points with "-1" element index are ignored and the corresponding coordinates are set to NaN.

        .. seealso::

            - :meth:`Mesh3D.shape_functions` to compute the shape functions at given natural coordinates.
            - :meth:`Mesh3D.interpolate_property_at_integration_points` to interpolate a vertices property at given integration points.
            - :meth:`integration_points_to_global_coordinates` to transform natural coordinates to global coordinates from an :class:`IntegrationPoints` instance.
            
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

            
        Examples
        --------
        Lets consider the subclass :class:`pysdic.geometry.LinearTriangleMesh3D` of Mesh3D.

        Create a simple LinearTriangleMesh3D instance.

        .. code-block:: python

            import numpy as np
            from pysdic.geometry import LinearTriangleMesh3D, PointCloud3D

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = LinearTriangleMesh3D(PointCloud3D.from_array(points), connectivity)

        Transform natural coordinates to global coordinates for points in elements 0 and 3.

        .. code-block:: python

            natural_coords = np.array([[0.3, 0.3], [0.2, 0.5]])
            element_indices = np.array([0, 3])
            global_coords = mesh3d.natural_to_global_coordinates(natural_coords, element_indices)
            print(global_coords)
            # Output: [[0.3 0.3 0. ]
            #          [0.3 0.2 0.5]]

        """
        integration_points = IntegrationPoints(natural_coords, element_indices)
        return self.interpolate_property_at_integration_points(
            integration_points,
            vertices_coordinates=True
        )


    def project_integration_property_to_vertices(
        self,
        integration_points: IntegrationPoints,
        property_array: numpy.ndarray,
    ) -> numpy.ndarray:
        r"""
        Project an integration points property to the vertices of the mesh.

        Lets consider a mesh with K vertices per element, and d natural coordinates.
        The given natural coordinates should be (Np, d) where Np is the number of integration points.
        The element_indices should be (Np,) where each entry is the index of the element in which to evaluate the natural coordinates.
        The weights should be (Np,) where each entry is the weight associated with each integration point.
    
        The evaluation of a property at the given natural coordinates is performed using the shape functions:

        .. math::

            P_{p} = N_{f} P_{n}

        where :math:`P_{p}` is the property at the integration points to evaluate with shape (Np, A),
        :math:`N_{f}` is the matrix of shape functions evaluated at the natural coordinates with shape (Np, N), 
        and :math:`P_{n}` is the property at the vertices of the elements with shape (N, A).

        To project the property from the integration points to the vertices, we use the following formula:

        .. math::

            P_{n} = (N_f^T W N_f)^{-1} N_f^T W P_{p}

        where :math:`W` is a diagonal matrix of weights with shape (Np, Np).

        .. note:

            - For a property with one attribute, the input property_array must be of shape (Np, 1) and not (Np,).

        .. seealso::

            - :meth:`Mesh3D.shape_functions` to compute the shape functions at given natural coordinates.
            - :meth:`Mesh3D.interpolate_property_at_integration_points` to interpolate a vertices property at given integration points (inverse operation).

        .. warning::

            Ensure that at least A integration points are associated with each element in the mesh to avoid singularities during the projection.

        Parameters
        ----------
        integration_points : IntegrationPoints
            An IntegrationPoints instance containing natural coordinates and element indices.

        property_array : numpy.ndarray
            An array of shape (Np, A) containing the property values at the integration points.

        Returns
        -------
        numpy.ndarray
            An array of shape (N, A) containing the projected property values.
        
        Raises
        ------
        TypeError
            If integration_points is not an IntegrationPoints instance.
        ValueError
            If property_array does not have the correct shape.
            If not enough integration points are associated with each element in the mesh to avoid singularities during the projection.


        Demonstration
        ---------------

        Lets consider :

        - :math:`N` : the number of vertices in the mesh.
        - :math:`N_p` : the number of integration points.
        - :math:`A` : the number of attributes for the property :math:`P`.
        - :`P_p` : the property at the integration points with shape (Np, A).
        - :`P_n` : the property at the vertices of the mesh with shape (N, A).
        - :`N_f` : the shape functions matrix evaluated at the natural coordinates with shape (Np, N).
        - :`W` : the diagonal weight matrix with shape (Np, Np).

        We know that the property at the integration points is given by:

        .. math::

            P_{p} = N_{f} P_{n}

        To project the property from the integration points to the vertices, we want to minimize the following weighted least squares problem:

        .. math::

            J(P_n) = \frac{1}{2}\sum_{i=1}^{N_p} w_i \| P_{p,i} - N_{f,i} P_n \|^2 = \frac{1}{2} (P_p - N_f P_n)^T W (P_p - N_f P_n)

        The gradient of the cost function is given by:

        .. math::

            \nabla J(P_n) = -N_f^T W (P_p - N_f P_n)

        To find the optimal property at the vertices, we set the gradient to zero:

        .. math::

            \nabla J(P_n) = 0 \implies N_f^T W P_p = N_f^T W N_f P_n

        So if we denote :math:`N_f^T W N_f` is invertible, we can solve for :math:`P_n`:

        .. math::

            P_{n} = (N_f^T W N_f)^{-1} N_f^T W P_{p}

        Examples
        --------

        Lets consider the subclass :class:`pysdic.geometry.LinearTriangleMesh3D` of Mesh3D.

        Create a simple LinearTriangleMesh3D instance.

        .. code-block:: python

            import numpy as np
            from pysdic.geometry import LinearTriangleMesh3D, PointCloud3D, IntegrationPoints

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = LinearTriangleMesh3D(PointCloud3D.from_array(points), connectivity)

        Set a property to vertices.

        .. code-block:: python

            vertex_property = np.array([10.0, 20.0, 15.0, 25.0]).reshape(-1, 1)  # Shape (4, 1)
            mesh3d.set_vertices_property("my_vertex_property", vertex_property)

        Create integration points in each element.

        .. code-block:: python

            natural_coords = np.array([[0.3, 0.3], [0.2, 0.5], [0.5, 0.2], [0.1, 0.1]])
            natural_coords = np.vstack([natural_coords] * mesh3d.n_elements)  # Shape (Np=16, 2)
            element_indices = np.repeat(np.arange(mesh3d.n_elements), 4)  # Shape (Np=16,)
            weights = np.full((natural_coords.shape[0],), 1.0)  # Shape (Np=16,)
            integration_points = IntegrationPoints(natural_coords, element_indices, weights)

        Interpolate the property at the integration points.

        .. code-block:: python

            property_at_integration_points = mesh3d.interpolate_property_at_integration_points(
                integration_points, 
                property_key="my_vertex_property"
            )

        Project the property from the integration points back to the vertices.

        .. code-block:: python

            projected_property = mesh3d.project_integration_property_to_vertices(
                integration_points, 
                property_at_integration_points
            )

            print(projected_property)
            # Output: [[10.]
            #          [20.]
            #          [15.]
            #          [25.]]

        """
        self._check_integration_points(integration_points)
        
        property_array = numpy.asarray(property_array, dtype=numpy.float64)
        if not (property_array.ndim == 2 and property_array.shape[0] == len(integration_points)):
            raise ValueError(f"Invalid property_array shape: {property_array.shape}")
        
        # Extract valid integration points
        valid_mask = integration_points.element_indices != -1
        natural_coords = integration_points.natural_coordinates[valid_mask, :]  # (Np_valid, d)
        element_indices = integration_points.element_indices[valid_mask]  # (Np_valid,)
        property_array = property_array[valid_mask, :]  # (Np_valid, A)
        weights = integration_points.weights[valid_mask]  # (Np_valid,)
        
        # Ensure enough integration points per element
        unique_elements, counts = numpy.unique(element_indices, return_counts=True)
        if unique_elements.size < self.n_elements:
            raise ValueError("At least one integration point must be associated with each element in the mesh to avoid singularities during the projection.")
        if numpy.any(counts < property_array.shape[1]):
            raise ValueError("Some elements have fewer integration points than the number of attributes in the property, which may lead to singularities during the projection.")

        # Get the shape functions matrix
        shape_functions_matrix, _ = self.shape_functions_matrix(natural_coords, element_indices, jacobian=False, sparse=True)  # (Np, N)

        # Build the weight matrix
        W = scipy.sparse.diags(weights, format='csr')  # (Np, Np)

        # Compute the matrices for the normal equations
        A = shape_functions_matrix.T @ W @ shape_functions_matrix  # (N, N)
        b = shape_functions_matrix.T @ W @ property_array  # (N, A)

        # Solve the normal equations
        projected_property = numpy.array(scipy.sparse.linalg.spsolve(A, b), dtype=numpy.float64)  # (N, A)
        return projected_property.reshape(self.n_vertices, -1)


    def shape_functions_matrix(self, natural_coords: numpy.ndarray, element_indices: numpy.ndarray, jacobian: bool = False, sparse: bool = False) -> Tuple[Union[numpy.ndarray, scipy.sparse.csr_matrix], Optional[Union[numpy.ndarray, scipy.sparse.csr_matrix]]]:
        r"""
        Compute the shape functions matrix at given natural coordinates.

        This method is a convenience wrapper around :meth:`shape_functions` that returns the shape functions in matrix form.

        Lets consider a mesh with K vertices per element, and d natural coordinates.
        The given natural coordinates should be (Np, d) where Np is the number of points to evaluate and d is the dimension of the natural coordinates.
        
        The :meth:`shape_functions` method returns the shape functions of shape (Np, K) and optionally the Jacobian of shape (Np, K, d).
        This method returns the shape functions in matrix form of shape (Np, N) where N is the total number of vertices in the mesh and optionally the Jacobian in matrix form of shape (Np, N, d).

        .. note:

            For one point, the input must be (1, d) and not only (d,).

        .. seealso::

            - :meth:`Mesh3D.shape_functions` to compute the shape functions at given natural coordinates.

        .. warning::

            For sparse=True and jacobian=True, the Jacobian is returned as a stack of sparse matrices of shape (Np, N) for each natural coordinate.

            .. code-block:: python

                _, array = mesh3d.shape_functions_matrix(natural_coords, element_indices, jacobian=True, sparse=True)
                _, sparse = mesh3d.shape_functions_matrix(natural_coords, element_indices, jacobian=True, sparse=False)

                # To reorganise wih the correct shape and order
                array = array.reshape(Np, -1, order='F')  # (Np, N, d) -> (Np, N * d) 


        Parameters
        ----------
        natural_coords : numpy.ndarray
            An array-like of shape (Np, d) where Np is the number of points to evaluate and d is the number of natural coordinates.

        element_indices : numpy.ndarray
            An array-like of shape (Np,) where Np is the number of points to evaluate.

        jacobian : bool, optional
            Whether to compute the Jacobian of the shape functions with respect to the natural coordinates, by default False.

        sparse : bool, optional
            Whether to return the shape functions matrix and Jacobian as sparse matrices, by default False.

        Returns
        -------
        Union[numpy.ndarray, scipy.sparse.csr_matrix]
            An array or sparse matrix of shape (Np, N) where N is the total number of vertices in the mesh.

        Optional[Union[numpy.ndarray, scipy.sparse.csr_matrix]]
            If ``jacobian`` is True, an array of shape (Np, N, d) or a sparse matrix of shape (Np, N * d) as (Np, N[d1] + N[d2] + ...) where N is the total number of vertices in the mesh and d is the number of natural coordinates. Otherwise, None.

        """
        natural_coords = numpy.asarray(natural_coords, dtype=numpy.float64)
        element_indices = numpy.asarray(element_indices, dtype=int)
        if natural_coords.ndim != 2 or natural_coords.shape[1] != self._n_dimensions:
            raise ValueError(f"natural_coords must be of shape (Np, {self._n_dimensions})")
        if element_indices.ndim != 1 or element_indices.shape[0] != natural_coords.shape[0]:
            raise ValueError("element_indices must be of shape (Np,)")
        
        # Get the shape functions and jacobian
        shape_functions, jacobian_array = self.shape_functions(natural_coords, jacobian=jacobian)  # (Np, K), (Np, K, d) or None

        # Get sizes
        Np = natural_coords.shape[0]
        K = self.n_vertices_per_element
        N = self.n_vertices
        d = self.n_dimensions

        # Extract the vertices indices for each point
        vertex_indices = self.connectivity[element_indices, :]  # (Np, K)

        # Build the numpy arrays
        if not sparse:
            shape_functions_matrix = numpy.zeros((Np, N), dtype=shape_functions.dtype)
            # Indexation vectorisée
            shape_functions_matrix[numpy.arange(Np)[:, None], vertex_indices] = shape_functions

            jacobian_matrix = None
            if jacobian:
                jacobian_matrix = numpy.zeros((Np, N, d), dtype=jacobian_array.dtype)
                jacobian_matrix[numpy.arange(Np)[:, None], vertex_indices, :] = jacobian_array

        # Build the scipy sparse matrices
        else:
            row_idx = numpy.repeat(numpy.arange(Np), K)
            col_idx = vertex_indices.ravel()
            data = shape_functions.ravel()
            shape_functions_matrix = scipy.sparse.csr_matrix((data, (row_idx, col_idx)), shape=(Np, N))

            jacobian_matrix = None
            if jacobian:
                jacobian_matrix = [scipy.sparse.csr_matrix((jacobian_array[..., j].ravel(), (row_idx, col_idx)), shape=(Np, N)) for j in range(d)]
                jacobian_matrix = scipy.sparse.hstack(jacobian_matrix, format='csr')  # (Np, N * d)

        return shape_functions_matrix, jacobian_matrix


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

            - :meth:`natural_to_global_coordinates` for transforming natural coordinates to global coordinates.

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
    

