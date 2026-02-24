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

from typing import Optional, Tuple, Union, List, Dict, Any
from numbers import Integral, Real
from numpy.typing import ArrayLike

import numpy
import pyvista
import meshio
import os
import matplotlib
from collections import defaultdict

from .point_cloud import PointCloud
from .connectivity import Connectivity
from .integration_points import IntegrationPoints

from ..core.shape_functions import compute_shape_functions
from ..core.adjacency_operations import compute_vertices_neighborhood, compute_elements_neighborhood, compute_elements_adjacency_matrix, compute_vertices_adjacency_matrix, compute_neighborhood_statistics
from ..core.integration_points_operations import assemble_property_projection, assemble_property_interpolation, assemble_property_derivative, assemble_shape_function_matrix, assemble_jacobian_matrix, remap_vertices_coordinates


class Mesh(ABC):
    r"""
    A Mesh is a collection of vertices (:class:`PointCloud`) associated with connectivity information (:class:`Connectivity`) that defines the elements of the mesh.

    The vertices are represented as a :class:`PointCloud` instance with shape :math:`(N_v, E)`, where :math:`N_v` is the number of vertices (``n_vertices``) and :math:`E` is the embedding dimension (``n_dimensions``).
    The connectivity is represented as a :class:`Connectivity` instance with shape :math:`(N_e, N_{vpe})`, where :math:`N_e` is the number of elements (``n_elements``), and :math:`N_{vpe}` is the number of vertices per element (``n_vertices_per_element``).

    The coordinates of a point into the mesh can be accessed by the natural coordinates
    in the reference element. The number of natural coordinates :math:`(\xi, \eta, \zeta, ...)` depends on the type of element and
    is noted as :math:`K` (the topological dimension of the element) accessible through the property ``n_topological_dimensions``.

    Lets consider a mesh with :math:`N_{vpe}` vertices per element, and :math:`K` natural coordinates.
    Lets :math:`X` be the coordinates of a point in the mesh. The transformation from natural coordinates to global coordinates is given by:

    .. math::

        X = \sum_{i=1}^{N_{vpe}} N_i(\xi, \eta, \zeta, ...) X_i

    where :math:`N_i` are the shape functions associated with each vertex, and :math:`X_i` are the coordinates of the vertices of the element.

    .. seealso::

        - :func:`pysdic.compute_shape_functions` for more information on shape functions.

    If :obj:`element_type` is provided, the mesh will enforce that all elements are of the specified type. 
    This allow to use the predefined shape functions and properties associated with that element type.
    The implemented types are:

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
    
    .. warning::
    
        Once the mesh is created with a specific :obj:`element_type`, number of vertices per element and topological dimensions. Please avoid adding or removing vertices or elements that do not conform to the expected structure of the mesh, as this may lead to unintended consequences and errors in the mesh structure.
        Consider create a copy of the mesh before performing such operations to avoid modifying the original mesh in place, which may lead to unintended consequences if the original mesh is used elsewhere.



    Parameters
    ----------
    vertices: Union[:class:`PointCloud`, ArrayLike]
        The vertices of the mesh as a :class:`PointCloud` instance with shape :math:`(N_v, E)`, where :math:`N_v` is the number of vertices and :math:`E` is the embedding dimension.

    connectivity: Union[:class:`Connectivity`, ArrayLike]
        The connectivity of the mesh as a :class:`Connectivity` instance with shape (:math:`N_e`, :math:`N_{vpe}`), where :math:`N_e` is the number of elements and :math:`N_{vpe}` is the number of vertices per element.

    element_type: Optional[:class:`str`], optional
        The expected type of elements in the mesh, by default None. The element type will be set to the connectivity. If provided, must be similar to the connectivity element type.
    
    """

    __slots__ = [
        '_internal_bypass',
        '_vertices', 
        '_connectivity',
        '_precomputed'
    ]

    def __init__(self, 
        vertices: Union[PointCloud, ArrayLike],
        connectivity: Union[Connectivity, ArrayLike],
        element_type: Optional[str] = None,
    ) -> None:
        # Convert the inputs to the correct types
        if isinstance(vertices, numpy.ndarray):
            vertices = PointCloud.from_array(vertices)
        if isinstance(connectivity, numpy.ndarray):
            connectivity = Connectivity.from_array(connectivity, element_type=element_type)
            
        # Ensure the types are correct
        if not isinstance(vertices, PointCloud):
            raise TypeError(f"Vertices must be a PointCloud instance, got {type(vertices)}.")
        if not isinstance(connectivity, Connectivity):
            raise TypeError(f"Connectivity must be a Connectivity instance, got {type(connectivity)}.")
        
        # Set the element type if provided
        if element_type is not None:
            connectivity.element_type = element_type
        
        # Assign attributes
        self._vertices = vertices
        self._connectivity = connectivity
        self._precomputed = {}
        
        # Initialize attributes
        self._internal_bypass = False
        self.validate()


    # =======================
    # Internals
    # =======================
    def _get_expected_N_vpe(self) -> Optional[int]:
        r"""
        Internal method to get the expected number of vertices per element for the mesh.

        Returns
        -------
        Optional[:class:`int`]
            The expected number of vertices per element, or None if not set.
        """
        return self.connectivity._expected_N_vpe
        
    def _get_expected_K(self) -> Optional[int]:
        r"""
        Internal method to get the expected number of natural coordinates (topological dimensions) for the mesh.

        Returns
        -------
        Optional[:class:`int`]
            The expected number of natural coordinates (topological dimensions), or None if not set.
        """
        return self.connectivity._expected_K
        
    def _get_meshio_cell_type(self) -> Optional[str]:
        r"""
        Internal method to get the expected meshio cell type for the mesh.

        Returns
        -------
        Optional[:class:`str`]
            The expected meshio cell type, or None if not set.
        """
        return self.connectivity._meshio_cell_type
    
    def _get_vtk_cell_type(self) -> Optional[int]:
        r"""
        Internal method to get the expected VTK cell type for the mesh.

        Returns
        -------
        Optional[:class:`int`]
            The expected VTK cell type, or None if not set.
        """
        return self.connectivity._vtk_cell_type

    def _internal_check_vertices(self) -> None:
        r"""
        Internal method to check the validity of the vertices.
        
        Raises
        ------
        TypeError
            If :obj:`vertices` is not a :class:`PointCloud` instance.
            
        ValueError
            If :obj:`vertices` do not have the correct embedding dimension or contain invalid values.
        """
        if self._internal_bypass:
            return
        
        expected_K = self._get_expected_K()

        if not isinstance(self._vertices, PointCloud):
            raise TypeError(f"Vertices must be a PointCloud instance, got {type(self._vertices)}.")
        if expected_K is not None and not self._vertices.n_dimensions >= expected_K:
            raise ValueError(f"Vertices must have embedding dimension greater or equal to {expected_K}, got {self._vertices.n_dimensions}.")
        if not self._vertices.all_finite():
            raise ValueError("Vertices contain NaN or infinite values.")

    def _internal_check_connectivity(self) -> None:
        r"""
        Internal method to check the validity of the connectivity.
        
        Raises
        ------
        TypeError
            If :obj:`connectivity` is not a :class:`numpy.ndarray`.
            
        ValueError
            If :obj:`connectivity` does not have the correct shape or contains invalid indices.
        """
        if self._internal_bypass:
            return
        
        expected_N_vpe = self._get_expected_N_vpe()

        if not isinstance(self._connectivity, Connectivity):
            raise TypeError(f"Connectivity must be a Connectivity instance, got {type(self._connectivity)}.")
        if expected_N_vpe is not None and not self._connectivity.n_vertices_per_element == expected_N_vpe:
            raise ValueError(f"Connectivity must have {expected_N_vpe} vertices per element, got {self._connectivity.n_vertices_per_element}.")
        if not numpy.all((self._connectivity.elements >= 0) & (self._connectivity.elements < self.n_vertices)):
            raise ValueError("Connectivity contains invalid vertex indices.")

    def _get_vertex_property(self, key: Optional[str] = None, default: Optional[numpy.ndarray] = None) -> Optional[numpy.ndarray]:
        r"""
        Internal method to get a vertices property or return a default value if the property does not exist.

        Parameters
        ----------
        key : Optional[:class:`str`]
            The key of the vertices property to retrieve. If None, returns the default value.

        default : Optional[:class:`numpy.ndarray`], optional
            The default value to return if the property does not exist, by default None.
            
        Returns
        -------
        Optional[:class:`numpy.ndarray`]
            The vertices property associated with the key, or the default value if the property does not exist.
        """
        if key is not None and not isinstance(key, str):
            raise TypeError(f"Key must be a string, got {type(key)}.")
        
        if key is not None and key in self._vertices:
            value = self._vertices[key]
            
        else:
            value = numpy.asarray(default, dtype=numpy.float64)
            if not value.ndim == 2 or not value.shape[0] == self.n_vertices or not value.shape[1] >= 1:
                raise ValueError(f"Default vertices property must have shape ({self.n_vertices}, A), got {value.shape}.")
        
        return value
    
    def _get_element_property(self, key: Optional[str] = None, default: Optional[numpy.ndarray] = None) -> Optional[numpy.ndarray]:
        r"""
        Internal method to get an elements property or return a default value if the property does not exist.
        
        Parameters
        ----------
        key : Optional[:class:`str`]
            The key of the elements property to retrieve. If None, returns the default value.
            
        default : Optional[:class:`numpy.ndarray`], optional
            The default value to return if the property does not exist, by default None.    
            
        Returns
        -------
        Optional[:class:`numpy.ndarray`]
            The elements property associated with the key, or the default value if the property does not exist.
            
        """
        if key is not None and not isinstance(key, str):
            raise TypeError(f"Key must be a string, got {type(key)}.")
        
        if key is not None and key in self._connectivity:   
            value = self._connectivity[key]
            
        else:
            value = numpy.asarray(default, dtype=numpy.float64)
            if not value.ndim == 2 or not value.shape[0] == self.n_elements or not value.shape[1] >= 1:
                raise ValueError(f"Default elements property must have shape ({self.n_elements}, B), got {value.shape}.")
            
        return value
    
    def _check_integration_points(self, integration_points: IntegrationPoints) -> None:
        r"""
        Internal method to check the validity of the integration points for the mesh.

        Parameters
        ----------
        integration_points : :class:`IntegrationPoints`
            An instance of the :class:`IntegrationPoints` class containing the natural coordinates of the points where the shape functions are to be evaluated.

        Raises
        ------
        TypeError
            If :obj:`integration_points` is not an instance of :class:`IntegrationPoints`.
            
        ValueError
            If the number of natural coordinates in :obj:`integration_points` does not match the expected number for the mesh.
            If the mesh does not have a defined expected number of natural coordinates (topological dimensions).
            
        """
        if not isinstance(integration_points, IntegrationPoints):
            raise TypeError(f"Integration points must be an instance of IntegrationPoints, got {type(integration_points)}.")
        
        expected_K = self._get_expected_K()
        if expected_K is None:
            raise ValueError("Mesh does not have a defined expected number of natural coordinates (topological dimensions). Cannot check integration points. Please set the `element_type` of the mesh connectivity to define the expected number of natural coordinates.")
        if not integration_points.n_topological_dimensions == expected_K:
            raise ValueError(f"Integration points must have {expected_K} natural coordinates, got {integration_points.n_topological_dimensions}.")
        
        
    def _set_precomputed(self, key: str, value: Any) -> None:
        r"""
        Internal method to set a precomputed value in the mesh.

        Parameters
        ----------
        key : :class:`str`
            The key to associate with the precomputed value.

        value : Any
            The value to store as precomputed.
        """
        if not isinstance(key, str):
            raise TypeError(f"Key must be a string, got {type(key)}.")
        
        self._precomputed[key] = value
        
    def _get_precomputed(self, key: str) -> Optional[Any]:
        r"""
        Internal method to get a precomputed value from the mesh.

        Parameters
        ----------
        key : :class:`str`
            The key associated with the precomputed value to retrieve.

        Returns
        -------
        Optional[Any]
            The precomputed value associated with the key, or None if the key does not exist.
        """
        if not isinstance(key, str):
            raise TypeError(f"Key must be a string, got {type(key)}.")
        
        return self._precomputed.get(key, None)
    
    def _has_precomputed(self, key: str) -> bool:
        r"""
        Internal method to check if a precomputed value exists in the mesh.

        Parameters
        ----------
        key : :class:`str`
            The key associated with the precomputed value to check.

        Returns
        -------
        bool
            True if a precomputed value associated with the key exists, False otherwise.
        """
        if not isinstance(key, str):
            raise TypeError(f"Key must be a string, got {type(key)}.")
        
        return key in self._precomputed
    
    def _clear_precomputed(self) -> None:
        r"""
        Internal method to clear all precomputed values from the mesh.
        """
        self._precomputed.clear()


    # =======================
    # I/O Methods
    # =======================
    @classmethod
    def from_meshio(cls, mesh: meshio.Mesh, element_type: Optional[str] = None, load_properties: bool = True) -> Mesh:
        r"""
        Create a Mesh instance from a :class:`meshio.Mesh` object.

        The following fields are extracted:

        - mesh.points → vertices
        - mesh.cells[0].data → connectivity
        - mesh.point_data → vertices properties as arrays of shape (N, A)
        - mesh.cell_data → elements properties as arrays of shape (M, B)

        .. seealso::

            - :meth:`Mesh.to_meshio` for the reverse operation.
            - `meshio documentation <https://pypi.org/project/meshio/>`_ for more information.
            - :class:`Mesh` for more information on the Mesh class and element types.

        Parameters
        ----------
        mesh: :class:`meshio.Mesh`
            A meshio Mesh object to extract the first cell block and create the Mesh instance.

        element_type: Optional[:class:`str`], optional
            The expected type of elements in the mesh, by default None.

        load_properties: :class:`bool`, optional
            If :obj:`True`, properties are extracted from the :class:`meshio.Mesh` object, by default :obj:`True`.


        Returns
        -------
        :class:`Mesh`
            A Mesh instance created from the :class:`meshio.Mesh` object.


        Raises
        ------
        TypeError
            If the input is not a :class:`meshio.Mesh` object.
            If element_type is not a string.
            If load_properties is not a boolean.
            
        ValueError
            If the mesh structure is invalid.
            If element_type is invalid or does not match the mesh structure.


        Examples
        --------
        Lets create a mesh using :class:`meshio.Mesh` and convert it to a :class:`Mesh` instance.

        .. code-block:: python
            :linenos:

            import numpy as np
            import meshio
            from pysdic import Mesh

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            cells = [("triangle", np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]))]

            mesh = meshio.Mesh(points=points, cells=cells)

        Create a :class:`Mesh` instance from the :class:`meshio.Mesh` object.

        .. code-block:: python
            :linenos:

            mesh3d = Mesh.from_meshio(mesh, element_type="triangle_3")
            print(mesh3d.vertices)
            # Output: PointCloud with 4 points [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]

        """
        # Validate the mesh structure
        if not isinstance(mesh, meshio.Mesh):
            raise TypeError(f"Input must be a meshio Mesh object, got {type(mesh)}.")
        if mesh.points.ndim != 2 or mesh.points.shape[1] < 1:
            raise ValueError("mesh.points must be a 2D array with at least one coordinate dimension.")
        if not len(mesh.cells) == 1 or mesh.cells[0].data.ndim != 2 or mesh.cells[0].data.shape[1] < 1:
            raise ValueError("Invalid mesh structure.")

        if not isinstance(load_properties, bool):
            raise TypeError(f"load_properties must be a boolean, got {type(load_properties)}.")

        # Extract data
        vertices = mesh.points
        connectivity = mesh.cells[0].data
        vertices_properties = {}
        elements_properties = {}
        
        # Extract properties if requested
        if load_properties:            
            for key, value in mesh.point_data.items():
                vertices_properties[key] = numpy.asarray(value).reshape(-1, 1) if value.ndim == 1 else numpy.asarray(value)

            for key, value in mesh.cell_data.items():
                elements_properties[key] = numpy.asarray(value[0]).reshape(-1, 1) if value[0].ndim == 1 else numpy.asarray(value[0])

        # Create Mesh instance
        vertices = PointCloud.from_array(vertices, properties=vertices_properties)
        connectivity = Connectivity.from_array(connectivity, properties=elements_properties, element_type=element_type)
        return cls(vertices, connectivity, element_type=element_type)
        

    def to_meshio(self, save_properties: bool = True) -> meshio.Mesh:
        r"""
        Convert the :class:`Mesh` instance to a :class:`meshio.Mesh` object (:obj:`element_type` must be defined).
        The mesh must not be empty. 

        .. warning::

            If the mesh does not have a defined element type, this method will raise a ValueError.
            See :func:`Mesh.element_type` to define the element type before conversion.

        The following fields are created:

        - vertices → mesh.points
        - connectivity → mesh.cells[0].data
        - vertices properties as arrays of shape (N, A) → mesh.point_data
        - elements properties as arrays of shape (M, B) → mesh.cell_data

        .. seealso::

            - :meth:`Mesh.from_meshio` for the reverse operation.
            - `meshio documentation <https://pypi.org/project/meshio/>`_ for more information.

        Parameters
        ----------
        save_properties: :class:`bool`, optional
            If :obj:`True`, properties are saved to the :class:`meshio.Mesh` object, by default :obj:`True`.


        Returns
        -------
        :class:`meshio.Mesh`
            A meshio Mesh object created from the Mesh instance.


        Raises
        ------
        TypeError
            If save_properties is not a boolean.
        ValueError
            If the mesh is empty.


        Examples
        --------
        Create a simple :class:`Mesh` instance.

        .. code-block:: python
            :linenos:

            import numpy as np
            from pysdic import Mesh, PointCloud

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])

            mesh3d = Mesh(points, connectivity, element_type="triangle_3")

        Convert the :class:`Mesh` instance to a :class:`meshio.Mesh` object.

        .. code-block:: python
            :linenos:

            mesh = mesh3d.to_meshio()
            print(mesh.points)
            # Output: [[0. 0. 0.] [1. 0. 0.] [0. 1. 0.] [0. 0. 1.]]
            
        """
        if self.n_vertices == 0 or self.n_elements == 0:
            raise ValueError("Cannot convert an empty mesh to meshio Mesh object.")
        if not isinstance(save_properties, bool):
            raise TypeError(f"save_properties must be a boolean, got {type(save_properties)}.")
        
        meshio_cell_type = self._get_meshio_cell_type()
        if meshio_cell_type is None:
            raise ValueError("Cannot convert to meshio Mesh object without a defined element type. See method 'Mesh.element_type' to define it.")
        
        cells = [meshio.CellBlock(meshio_cell_type, data=self._connectivity.to_array())]
        
        point_data = {}
        cell_data = {}
        
        if save_properties:
            point_data = self._vertices.copy_properties()
            cell_data = self._connectivity.copy_properties()
            
        return meshio.Mesh(points=self._vertices.to_array(), cells=cells, point_data=point_data, cell_data=cell_data)
    

    @classmethod
    def from_npz(cls, filename: str, element_type: Optional[str] = None, load_properties: bool = True) -> Mesh:
        r"""
        Create a Mesh instance from a NPZ file.

        This method uses numpy to read the NPZ file and then converts it to a :class:`Mesh` instance.

        .. seealso::

            - :meth:`Mesh.to_npz` for the reverse operation.
            - `numpy documentation <https://numpy.org/doc/stable/reference/generated/numpy.load.html>`_ for more information.
            - :class:`Mesh` for more information on the Mesh class and element types.

        Parameters
        ----------
        filename: :class:`str`
            The path to the NPZ file.

        element_type: Optional[:class:`str`], optional
            The expected type of elements in the mesh, by default None.

        load_properties: :class:`bool`, optional
            If :obj:`True`, properties are extracted from the NPZ file, by default :obj:`True`.


        Returns
        -------
        :class:`Mesh`
            A :class:`Mesh` instance created from the NPZ file.


        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file format is not supported or the mesh structure is invalid.

        
        Examples
        --------
        Create a simple :class:`meshio.Mesh` object.

        .. code-block:: python
            :linenos:

            import numpy as np
            from pysdic import Mesh

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            cells = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh = Mesh(vertices=points, connectivity=cells, element_type="triangle_3")
            mesh.save_npz("simple_mesh.npz")

        Create a :class:`Mesh` instance from the NPZ file.

        .. code-block:: python
            :linenos:

            mesh3d = Mesh.from_npz("simple_mesh.npz")
            print(mesh3d.vertices)
            # Output: PointCloud with 4 points [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
            
        """
        if not isinstance(load_properties, bool):
            raise TypeError(f"load_properties must be a boolean, got {type(load_properties)}.")
        
        path = os.path.abspath(os.path.expanduser(filename))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File '{filename}' does not exist.")
        
        data = numpy.load(path, allow_pickle=True)
        points = data["vertices"]
        connectivity = data["connectivity"]
        vertices_properties = {}
        elements_properties = {}

        if load_properties:
            vertices_properties = data.get("vertices_properties", {}).item()
            elements_properties = data.get("elements_properties", {}).item()

        # Create Mesh instance
        vertices = PointCloud.from_array(points, properties=vertices_properties)
        connectivity = Connectivity.from_array(connectivity, properties=elements_properties, element_type=element_type)
        return cls(vertices, connectivity, element_type=element_type)
    

    def to_npz(self, filename: str, save_properties: bool = True) -> None:
        r"""
        Write the :class:`Mesh` instance to a NPZ file.
        
        The mesh must not be empty.

        This method uses numpy to write the Mesh instance to a NPZ file.

        .. seealso::

            - :meth:`Mesh.from_npz` for the reverse operation.
            - `numpy documentation <https://numpy.org/doc/stable/reference/generated/numpy.savez.html>`_ for more information.

        Parameters
        ----------
        filename: :class:`str`
            The path to the output NPZ file.

        save_properties: :class:`bool`, optional
            If :obj:`True`, properties are saved to the NPZ file, by default :obj:`True`.


        Raises
        ------
        ValueError
            If the file format is not supported or the mesh is empty.

            
        Examples
        --------
        Create a simple :class:`Mesh` instance.

        .. code-block:: python
            :linenos:

            import numpy as np
            from pysdic import Mesh, PointCloud
            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = Mesh(points, connectivity, element_type="triangle_3")

        Save the :class:`Mesh` instance to a NPZ file.

        .. code-block:: python
            :linenos:

            mesh3d.to_npz("simple_mesh.npz")
            # This will create a file named 'simple_mesh.npz' in the current directory.

        """
        if self.n_vertices == 0 or self.n_elements == 0:
            raise ValueError("Cannot save an empty mesh to NPZ file.")
        if not isinstance(save_properties, bool):
            raise TypeError(f"save_properties must be a boolean, got {type(save_properties)}.")
        
        path = os.path.abspath(os.path.expanduser(filename))
        if save_properties:
            numpy.savez(
                path,
                vertices=self._vertices.to_array(),
                connectivity=self._connectivity.to_array(),
                vertices_properties=self._vertices.copy_properties(),
                elements_properties=self._connectivity.copy_properties()
            )
        else:
            numpy.savez(
                path,
                vertices=self._vertices.to_array(),
                connectivity=self._connectivity.to_array()
            )
    

    @classmethod
    def from_vtk(cls, filename: str, element_type: Optional[str] = None, load_properties: bool = True) -> Mesh:
        r"""
        Create a Mesh instance from a VTK file (Only for 3D embedding dimension meshes :math:`E=3`).

        This method uses meshio to read the VTK file and then converts it to a :class:`Mesh` instance.

        .. seealso::

            - :meth:`Mesh.to_vtk` for the reverse operation.
            - :meth:`Mesh.from_meshio` for more information on the conversion process.
            - `meshio documentation <https://pypi.org/project/meshio/>`_ for more information.
            - :class:`Mesh` for more information on the Mesh class and element types.

        .. warning::

            This method is only compatible with meshes having an embedding dimension of 3.

        Parameters
        ----------
        filename: :class:`str`
            The path to the VTK file.

        element_type: Optional[:class:`str`], optional
            The expected type of elements in the mesh, by default None.

        load_properties: :class:`bool`, optional
            If :obj:`True`, properties are extracted from the VTK file, by default :obj:`True`.
            

        Returns
        -------
        :class:`Mesh`
            A :class:`Mesh` instance created from the VTK file.


        Raises
        ------
        FileNotFoundError
            If the file does not exist.
            
        ValueError
            If the file format is not supported or the mesh structure is invalid.
        
        
        Examples
        --------
        Create a simple :class:`meshio.Mesh` object.

        .. code-block:: python
            :linenos:

            import numpy as np
            import meshio
            from pysdic import Mesh

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            cells = [("triangle", np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]))]

            mesh = meshio.Mesh(points=points, cells=cells)

        Save the meshio Mesh object to a VTK file.

        .. code-block:: python
            :linenos:

            mesh.write("simple_mesh.vtk", file_format="vtk")

        Create a :class:`Mesh` instance from the VTK file.

        .. code-block:: python
            :linenos:

            mesh3d = Mesh.from_vtk("simple_mesh.vtk")
            print(mesh3d.vertices)
            # Output: PointCloud with 4 points [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]      

        """
        path = os.path.abspath(os.path.expanduser(filename))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File '{filename}' does not exist.")
        
        mesh = meshio.read(filename, file_format="vtk")
        return cls.from_meshio(mesh, element_type=element_type, load_properties=load_properties)


    def to_vtk(self, filename: str, save_properties: bool = True) -> None:
        r"""
        Write the :class:`Mesh` instance to a VTK file (Only for 3D embedding dimension meshes :math:`E=3`).
        
        The mesh must not be empty.

        This method uses meshio to write the Mesh instance to a VTK file.

        .. seealso::

            - :meth:`Mesh.from_vtk` for the reverse operation.
            - :meth:`Mesh.to_meshio` for more information on the conversion process.
            - `meshio documentation <https://pypi.org/project/meshio/>`_ for more information.

        Parameters
        ----------
        filename: :class:`str`
            The path to the output VTK file.
        
        save_properties: :class:`bool`, optional
            If :obj:`True`, properties are saved to the VTK file, by default :obj:`True`.


        Raises
        ------
        ValueError
            If the file format is not supported or the mesh is empty.

        Examples
        --------

        Create a simple :class:`Mesh` instance.

        .. code-block:: python
            :linenos:

            import numpy as np
            from pysdic import Mesh, PointCloud

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = Mesh(PointCloud.from_array(points), connectivity)

        Save the :class:`Mesh` instance to a VTK file.

        .. code-block:: python
            :linenos:

            mesh3d.to_vtk("simple_mesh.vtk")
            # This will create a file named 'simple_mesh.vtk' in the current directory.
            
        """
        if self.n_vertices == 0 or self.n_elements == 0:
            raise ValueError("Cannot write an empty mesh to file.")
        if not isinstance(save_properties, bool):
            raise TypeError(f"save_properties must be a boolean, got {type(save_properties)}.")
        if not self.n_dimensions == 3:
            raise ValueError("VTK file format is only supported for meshes with embedding dimension of 3.")
        
        path = os.path.abspath(os.path.expanduser(filename))
        os.makedirs(os.path.dirname(path), exist_ok=True)

        mesh = self.to_meshio(save_properties=save_properties)
        mesh.write(filename, file_format="vtk")


    # =======================
    # Properties
    # =======================
    @property
    def vertices(self) -> PointCloud:
        r"""
        [Get or Set] The vertices of the mesh in an :class:`PointCloud` instance.

        The vertices are represented as a PointCloud instance with shape (:math:`N_v`, :math:`E`) where :math:`N_v` is the number of vertices and :math:`E` is the embedding dimension.

        .. note::

            This property is settable.

        .. warning::

            If the vertices are changed, the connectivity and properties may become invalid. 
            Please ensure to recompute or update them accordingly.

            To change the number of vertices, it is recommended to create a new Mesh instance with the updated vertices and connectivity
            rather than modifying the vertices in place. For memory considerations, you can also modify the vertices in place, but please ensure that
            all necessary checks are performed before using this mode.

            
        Parameters
        ----------
        value: Union[:class:`PointCloud`, ArrayLike]
            The new vertices for the mesh with shape (:math:`N_v`, :math:`E`).
            

        Returns
        -------
        :class:`PointCloud`
            The vertices of the mesh as a PointCloud instance of shape (:math:`N_v`, :math:`E`).


        Examples
        --------
        Create a simple :class:`Mesh` instance.

        .. code-block:: python
            :linenos:

            import numpy as np
            from pysdic import Mesh, PointCloud

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = Mesh(points, connectivity, element_type="triangle_3")

        Access the vertices of the mesh.

        .. code-block:: python

            print(mesh3d.vertices)
            # Output: PointCloud with 4 points [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        
        """
        return self._vertices
    
    @vertices.setter
    def vertices(self, value: Union[PointCloud, ArrayLike]) -> None:
        if not isinstance(value, PointCloud):
            value = PointCloud.from_array(value)
        if not isinstance(value, PointCloud):
            raise TypeError(f"Vertices must be a PointCloud instance, got {type(value)}.")
        if not value.n_dimensions >= self.n_dimensions:
            raise ValueError(f"Vertices must have embedding dimension greater or equal to {self.n_dimensions}, got {value.n_dimensions}.")
        if not value.n_points == self.n_vertices:
            raise ValueError(f"Vertices must have {self.n_vertices} points, got {value.n_points}.")
        self._vertices = value
        self._internal_check_vertices()


    @property
    def connectivity(self) -> Connectivity:
        r"""
        [Get or Set] The connectivity of the mesh in a :class:`Connectivity` instance.

        The connectivity is represented as a Connectivity instance with shape (:math:`N_e`, :math:`N_{vpe}`) where :math:`N_e` is the number of elements and :math:`N_{vpe}` is the number of vertices per element.

        .. note::

            This property is settable.

        .. warning::

            If the connectivity is changed, the vertices and properties may become invalid. 
            Please ensure to recompute or update them accordingly.

            To change the number of elements, it is recommended to create a new Mesh instance with the updated vertices and connectivity
            rather than modifying the connectivity in place. For memory considerations, you can also modify the connectivity in place, but please ensure that
            all necessary checks are performed before using this mode.
            
        Parameters
        ----------
        value: Union[:class:`Connectivity`, ArrayLike]
            The new connectivity for the mesh with shape (:math:`N_e`, :math:`N_{vpe}`).
            
            
        Returns
        -------
        :class:`Connectivity`
            The connectivity of the mesh as a Connectivity instance of shape (:math:`N_e`, :math:`N_{vpe}`).
            
            
        Examples
        --------
        Create a simple :class:`Mesh` instance.
        
        .. code-block:: python
            :linenos:
        
            import numpy as np
            from pysdic import Mesh, PointCloud
            
            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = Mesh(points, connectivity, element_type="triangle_3")
            
        Access the connectivity of the mesh.
        
        .. code-block:: python
            :linenos:
        
            print(mesh3d.connectivity)
            # Output: Connectivity with 4 elements [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
            
        """
        return self._connectivity
    
    @connectivity.setter
    def connectivity(self, value: Union[Connectivity, ArrayLike]) -> None:
        if not isinstance(value, Connectivity):
            value = Connectivity.from_array(value, element_type=self.element_type)
        if not isinstance(value, Connectivity):
            raise TypeError(f"Connectivity must be a Connectivity instance, got {type(value)}.")
        if not value.n_vertices_per_element == self.n_vertices_per_element:
            raise ValueError(f"Connectivity must have {self.n_vertices_per_element} vertices per element, got {value.n_vertices_per_element}.")
        if not value.n_elements == self.n_elements:
            raise ValueError(f"Connectivity must have {self.n_elements} elements, got {value.n_elements}.")
        self._connectivity = value
        self._internal_check_connectivity()
    
    
    # =======================
    # Fast access properties
    # =======================
    @property
    def points(self) -> numpy.ndarray:
        r"""
        [Get or Set] The coordinates of the mesh vertices - alias for :obj:`self.vertices.points` property.
        
        .. seealso::
        
            - :class:`PointCloud` for more information on the points property.
            
        """
        return self._vertices.points
    
    @points.setter
    def points(self, value: ArrayLike) -> None:
        self._vertices.points = value
        
    @property
    def coordinates(self) -> numpy.ndarray:
        r"""
        [Get or Set] Alias for :obj:`self.points` property, returns the coordinates of the mesh vertices as a numpy array of shape (:math:`N_v`, :math:`E`).
        
        .. seealso::
        
            - :class:`PointCloud` for more information on the points property.
                     
        """
        return self._vertices.points
    
    @coordinates.setter
    def coordinates(self, value: ArrayLike) -> None:
        self._vertices.points = value
        
    @property
    def elements(self) -> numpy.ndarray:
        r"""
        [Get or Set] The vertex indices of the mesh elements - alias for :obj:`self.connectivity.elements` property.
        
        .. seealso::
        
            - :class:`Connectivity` for more information on the elements property.
            
        """
        return self._connectivity.elements
    
    @elements.setter
    def elements(self, value: ArrayLike) -> None:
        self._connectivity.elements = value


    @property
    def n_vertices(self) -> int:
        r"""
        [Get] The number of vertices :math:`N_v` in the mesh - alias for :obj:`self.vertices.n_points` property.
        This method is equivalent to calling :obj:`self.n_points` property of the vertices, but is provided for convenience and readability when working with meshes.
        
        Returns
        -------
        :class:`int`
            The number of vertices in the mesh.


        Examples
        --------
        Create a simple :class:`Mesh` instance.

        .. code-block:: python
            :linenos:

            import numpy as np
            from pysdic import Mesh, PointCloud

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = Mesh(points, connectivity, element_type="triangle_3")

        Get the number of vertices in the mesh.

        .. code-block:: python
            :linenos:

            print(mesh3d.n_vertices)
            # Output: 4

        """
        return self._vertices.n_points
    
    @property
    def n_points(self) -> int:
        r"""
        [Get] Alias for :obj:`self.n_vertices` property, returns the number of vertices :math:`N_v` in the mesh.
        """
        return self.n_vertices

    @property
    def n_elements(self) -> int:
        r"""
        [Get] The number of elements :math:`N_e` in the mesh - alias for :obj:`self.connectivity.n_elements` property.
        
        Returns
        -------
        :class:`int`
            The number of elements in the mesh.
            
            
        Examples
        --------
        Create a simple :class:`Mesh` instance.
        
        .. code-block:: python
            :linenos:

            import numpy as np
            from pysdic import Mesh, PointCloud

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = Mesh(points, connectivity, element_type="triangle_3")
            
        Get the number of elements in the mesh.
        
        .. code-block:: python
            :linenos:

            print(mesh3d.n_elements)
            # Output: 4
            
        """
        return self._connectivity.n_elements
    
    @property
    def n_dimensions(self) -> int:
        r"""
        [Get] The embedding dimension :math:`E` of the mesh - alias for :obj:`self.vertices.n_dimensions` property.

        Returns
        -------
        :class:`int`
            The embedding dimension of the mesh.


        Examples
        --------
        Create a simple :class:`Mesh` instance.

        .. code-block:: python
            :linenos:
            
            import numpy as np
            from pysdic import Mesh, PointCloud

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = Mesh(points, connectivity, element_type="triangle_3")
            
        Get the embedding dimension of the mesh.

        .. code-block:: python
            :linenos:

            print(mesh3d.n_dimensions)
            # Output: 3

        """
        return self._vertices.n_dimensions

    @property
    def n_vertices_per_element(self) -> int:
        r"""
        [Get] The number of vertices per element :math:`N_{vpe}` in the mesh - alias for :obj:`self.connectivity.n_vertices_per_element` property.
        
        Returns
        -------
        :class:`int`
            The number of vertices per element.

        
        Examples
        --------
        Create a simple :class:`Mesh` instance.

        .. code-block:: python
            :linenos:

            import numpy as np
            from pysdic import Mesh, PointCloud

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = Mesh(points, connectivity, element_type="triangle_3")

        Get the number of vertices per element in the mesh.

        .. code-block:: python
            :linenos:

            print(mesh3d.n_vertices_per_element)
            # Output: 3
        
        """
        return self._connectivity.n_vertices_per_element
    

    @property
    def n_topological_dimensions(self) -> int:
        r"""
        [Get] The topological dimension :math:`K` of the elements in the mesh - alias for :obj:`self.connectivity.n_topological_dimensions` property.

        Returns
        -------
        :class:`int`
            The topological dimension of the elements.


        Examples
        --------
        Create a simple :class:`Mesh` instance.

        .. code-block:: python
            :linenos:

            import numpy as np
            from pysdic import Mesh, PointCloud

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = Mesh(points, connectivity, element_type="triangle_3")

        Get the topological dimension of the elements in the mesh.

        .. code-block:: python
            :linenos:

            print(mesh3d.n_topological_dimensions)
            # Output: 2
        
        """
        return self._connectivity.n_topological_dimensions
    

    @property
    def element_type(self) -> Optional[str]:
        r"""
        [Get or Set] The element type of the mesh - alias for :obj:`self.connectivity.element_type` property.
        

        Returns
        -------
        :class:`str` or None
            The element type of the mesh, or None if not specified.

        
        Examples
        --------
        Create a simple :class:`Mesh` instance with a specified element type.

        .. code-block:: python
            :linenos:

            import numpy as np
            from pysdic import Mesh, PointCloud

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = Mesh(PointCloud.from_array(points), connectivity, element_type="triangle_3")

        Get the element type of the mesh.

        .. code-block:: python
            :linenos:

            print(mesh3d.element_type)
            # Output: triangle_3

        """
        return self._connectivity.element_type
    
    @element_type.setter
    def element_type(self, value: Optional[str]) -> None:
        self._connectivity.element_type = value
        self._internal_check_connectivity()

    def validate(self) -> None:
        r"""
        Validate the mesh by performing internal checks on vertices and connectivity.
        """
        self._internal_check_vertices()
        self._internal_check_connectivity()
        

    def copy(self) -> Mesh:
        r"""
        Create a deep copy of the Mesh instance.

        Returns
        -------
        :class:`Mesh`
            A new Mesh instance that is a deep copy of the original.
            
        """
        vertices_copy = self._vertices.copy()
        connectivity_copy = self._connectivity.copy()
        return Mesh(vertices_copy, connectivity_copy, element_type=self.element_type)


    # =======================
    # Manipulate Mesh geometry
    # ======================= 
    def add_connectivity(self, new_connectivity: Union[Connectivity, ArrayLike], force_inplace: bool = False) -> None:
        r"""
        Add new elements to the mesh by appending new connectivity entries. 
        
        This method is an extension of the :obj:`connectivity.concatenate` method that allows adding new elements to the mesh while ensuring that the connectivity structure is maintained and validated.
        This method will ensure that the new connectivity is compatible with the existing mesh structure and will update the mesh's connectivity accordingly.

        .. note::

            The new elements will be added to the end of the existing connectivity object and store in a new connectivity object (Copy of the original connectivity object with the new connectivity concatenated).
            The properties of the new elements will be initialized with NaN values and stored in the new connectivity object.
            
        .. seealso::
        
            - :meth:`add_vertices` to add new vertices to the mesh.
            - :class:`Connectivity` for more information on the connectivity structure and element types.


        Parameters
        ----------
        new_connectivity: Union[:class:`Connectivity`, ArrayLike]
            An array of shape (:math:`P`, :math:`N_{vpe}`) containing the connectivity of the new elements to add,
            where :math:`P` is the number of new elements and :math:`N_{vpe}` is the number of vertices per element.
            
        force_inplace: :class:`bool`, optional
            If :obj:`True`, the new connectivity will be concatenated to the existing connectivity in place, modifying the original connectivity object. 
            If :obj:`False`, a new connectivity object will be created to store the combined connectivity, leaving the original connectivity object unchanged. 
            By default :obj:`False`.


        Raises
        ------
        TypeError
            If :obj:`new_connectivity` is not a :class:`Connectivity` instance or a numpy array.
            If :obj:`force_inplace` is not a boolean.
            
        ValueError
            If :obj:`new_connectivity` does not have the correct shape or contains invalid indices.
            
            
        Examples
        --------
        Create a simple :class:`Mesh` instance.

        .. code-block:: python
            :linenos:

            import numpy as np
            from pysdic import Mesh, PointCloud

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
            elements = np.array([[0, 1, 2]])
            mesh3d = Mesh(points, elements, element_type="triangle_3")

        Add some properties to the elements.

        .. code-block:: python
            :linenos:

            element_property = np.array([10.0]).reshape(-1, 1) # Shape (1, 1)
            mesh3d.connectivity["my_element_property"] = element_property

        Add new elements to the mesh.

        .. code-block:: python
            :linenos:

            new_elements = np.array([[0, 1, 2], [1, 2, 0]])
            mesh3d.add_elements(new_elements)

            print(mesh3d.connectivity)
            # Output: Connectivity with 3 elements [[0, 1, 2], [0, 1, 2], [1, 2, 0]]

            print(mesh3d.connectivity["my_element_property"])
            # Output: [[10.], [nan], [nan]]

        """
        # Check new connectivity
        if not isinstance(new_connectivity, Connectivity):
            new_connectivity = Connectivity.from_array(new_connectivity, element_type=self.element_type)
            
        if not isinstance(new_connectivity, Connectivity):
            raise TypeError(f"new_connectivity must be a Connectivity instance or a numpy array, got {type(new_connectivity)}.")
        if not new_connectivity.n_vertices_per_element == self.n_vertices_per_element:
            raise ValueError(f"new_connectivity must have {self.n_vertices_per_element} vertices per element, got {new_connectivity.n_vertices_per_element}.")
        if not new_connectivity.n_elements > 0:
            raise ValueError("new_connectivity must contain at least one element.")
        if new_connectivity.element_type is not None and new_connectivity.element_type != self.element_type:
            raise ValueError(f"new_connectivity must have the same element type as the mesh ({self.element_type}), got {new_connectivity.element_type}.")
        
        if not isinstance(force_inplace, bool):
            raise TypeError(f"force_inplace must be a boolean, got {type(force_inplace)}.")
        
        # Concatenate the new connectivity with the existing one anfd store in a new connectivity object
        combined_connectivity = self._connectivity.concatenate(new_connectivity, inplace=force_inplace)

        # Set the combined connectivity to the mesh
        self._connectivity = combined_connectivity
        
        # Validate the updated connectivity
        self._internal_check_connectivity()


    def add_vertices(self, new_vertices: Union[PointCloud, ArrayLike], force_inplace: bool = False) -> None:
        r"""
        Add new vertices to the mesh by appending new vertex coordinates.
        
        This method is an extension of the :obj:`vertices.concatenate` method that allows adding new vertices to the mesh while ensuring that the vertices structure is maintained and validated.
        This method will ensure that the new vertices are compatible with the existing mesh structure and will update the mesh's vertices accordingly.

        .. note::
        
            The new vertices will be added to the end of the existing vertices object and store in a new vertices object (Copy of the original vertices object with the new vertices concatenated).
            The properties of the new vertices will be initialized with NaN values and stored in the new vertices object.
            
        .. seealso::
        
            - :meth:`add_connectivity` to add new elements to the mesh.
            - :class:`PointCloud` for more information on the vertices structure.


        Parameters
        ----------
        new_vertices: Union[:class:`PointCloud`, ArrayLike]
            An array of shape (:math:`Q`, :math:`E`) containing the coordinates of the new vertices to add,
            where :math:`Q` is the number of new vertices and :math:`E` is the embedding dimension.
            
        force_inplace: :class:`bool`, optional
            If :obj:`True`, the new vertices will be concatenated to the existing vertices in place, modifying the original vertices object. 
            If :obj:`False`, a new vertices object will be created to store the combined vertices, leaving the original vertices object unchanged. 
            By default :obj:`False`.


        Raises
        ------
        TypeError
            If :obj:`new_vertices` is not a :class:`PointCloud` instance or an array-like object.
            If :obj:`force_inplace` is not a boolean.
            
        ValueError
            If :obj:`new_vertices` does not have the correct shape.

            
        Examples
        --------

        Create a simple :class:`Mesh` instance.

        .. code-block:: python
            :linenos:

            import numpy as np
            from pysdic import Mesh, PointCloud

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
            connectivity = np.array([[0, 1, 2]])
            mesh3d = Mesh(points, connectivity, element_type="triangle_3")

        Add new vertices to the mesh.

        .. code-block:: python
            :linenos:

            new_vertices = np.array([[0, 0, 1], [1, 1, 1]])
            mesh3d.add_vertices(new_vertices)

            print(mesh3d.vertices)
            # Output: PointCloud with 5 points [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]

        """
        if not isinstance(new_vertices, PointCloud):
            new_vertices = PointCloud.from_array(new_vertices)
        
        if not isinstance(new_vertices, PointCloud):
            raise TypeError(f"new_vertices must be a PointCloud instance or a numpy array, got {type(new_vertices)}.")
        if not new_vertices.n_dimensions == self.n_dimensions:
            raise ValueError(f"new_vertices must have the same embedding dimension as the mesh ({self.n_dimensions}), got {new_vertices.n_dimensions}.")
        if not new_vertices.n_points > 0:
            raise ValueError("new_vertices must contain at least one vertex.")
        
        if not isinstance(force_inplace, bool):
            raise TypeError(f"force_inplace must be a boolean, got {type(force_inplace)}.")
        
        # Concatenate the new vertices with the existing one and store in a new vertices object
        combined_vertices = self._vertices.concatenate(new_vertices, inplace=force_inplace)
        
        # Set the combined vertices to the mesh
        self._vertices = combined_vertices
        
        # Validate the updated vertices
        self._internal_check_vertices()
        
        
    def filter_connectivity(self, mask: ArrayLike, force_inplace: bool = False) -> None:
        r"""
        Filter the connectivity of the mesh by keeping only the elements corresponding to the True values in the mask.
        
        This metod is an extension of the :obj:`connectivity.filter_elements` method that allows filtering the connectivity of the mesh while ensuring that the connectivity structure is maintained and validated.
        This method will ensure that the mask is compatible with the existing mesh structure and will update the mesh's connectivity accordingly.

        .. note::

            The  new connectivity will be stored in a new connectivity object (Copy of the original connectivity object with only the filtered elements).
            The properties of the filtered elements will be updated to keep only the properties of the kept elements and stored in the new connectivity object.
            
        .. seealso::
        
            - :meth:`remove_connectivity` to remove specified elements from the mesh.
            - :meth:`keep_connectivity` to keep only specified elements in the mesh.
            - :class:`Connectivity` for more information on the connectivity structure and element types.
            
        Parameters
        ----------
        mask: ArrayLike
            A boolean array of shape :math:`(N_e,)` where :math:`N_e` is the number of elements in the mesh, indicating which elements to keep (True) or remove (False) from the mesh.
            
        force_inplace: :class:`bool`, optional
            If :obj:`True`, the connectivity will be filtered in place, modifying the original connectivity object. 
            If :obj:`False`, a new connectivity object will be created to store the filtered connectivity, leaving the original connectivity object unchanged. 
            By default :obj:`False`.
            
        
        Raises
        ------
        TypeError
            If :obj:`mask` is not an array-like object or is not a boolean array.
            If :obj:`force_inplace` is not a boolean.
            
        ValueError
            If :obj:`mask` does not have the correct shape or is not a boolean array.
            
    
        Examples
        --------
        Create a simple :class:`Mesh` instance.
        
        .. code-block:: python
            :linenos:
            
            import numpy as np
            from pysdic import Mesh, PointCloud
            
            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = Mesh(points, connectivity, element_type="triangle_3")
            
        Filter the connectivity of the mesh to keep only the first and third elements.
        
        .. code-block:: python
            :linenos:
        
            mask = np.array([True, False, True, False])
            mesh3d.filter_connectivity(mask)
            
            print(mesh3d.connectivity)
            # Output: Connectivity with 2 elements [[0, 1, 2], [0, 2, 3]]
            
        """
        # Filter the connectivity and store in a new connectivity object
        filtered_connectivity = self._connectivity.filter_elements(mask, inplace=force_inplace)
        
        # Set the filtered connectivity to the mesh
        self._connectivity = filtered_connectivity
        
        # Validate the updated connectivity
        self._internal_check_connectivity()
        
        
    def keep_connectivity(self, element_indices: ArrayLike, force_inplace: bool = False) -> None:
        r"""
        Keep only the specified elements in the mesh by specifying their indices in the connectivity array.
        
        This method is an extension of the :obj:`connectivity.keep_elements` method that allows keeping only specified elements in the mesh while ensuring that the connectivity structure is maintained and validated.
        This method will ensure that the specified element indices are compatible with the existing mesh structure and will update the mesh's connectivity accordingly.

        .. note::

            The new connectivity will be stored in a new connectivity object (Copy of the original connectivity object with only the kept elements).
            The properties of the kept elements will be updated to keep only the properties of the kept elements and stored in the new connectivity object.
            
        .. seealso::
        
            - :meth:`remove_connectivity` to remove specified elements from the mesh.
            - :meth:`filter_connectivity` to filter the connectivity of the mesh using a boolean mask.
            - :class:`Connectivity` for more information on the connectivity structure and element types.
            
        Parameters
        ----------
        element_indices: ArrayLike
            An array of shape :math:`(R,)` containing the indices of the elements to keep in the mesh, where :math:`R` is the number of elements to keep.
            
        force_inplace: :class:`bool`, optional
            If :obj:`True`, the connectivity will be updated in place, modifying the original connectivity object. 
            If :obj:`False`, a new connectivity object will be created to store the updated connectivity, leaving the original connectivity object unchanged. 
            By default :obj:`False`.
            
        
        Raises
        ------
        TypeError
            If :obj:`element_indices` is not an array-like object or is not an integer array.
            If :obj:`force_inplace` is not a boolean.
            
        ValueError
            If :obj:`element_indices` does not have the correct shape or contains invalid indices.
            
        
        Examples
        --------
        Create a simple :class:`Mesh` instance.
        
        .. code-block:: python
            :linenos:
        
            import numpy as np
            from pysdic import Mesh, PointCloud
            
            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = Mesh(points, connectivity, element_type="triangle_3")
            
        Keep only the first and third elements in the mesh.
        
        .. code-block:: python
            :linenos:
            
            element_indices = np.array([0, 2])
            mesh3d.keep_connectivity(element_indices)
            
            print(mesh3d.connectivity)
            # Output: Connectivity with 2 elements [[0, 1, 2], [0, 2, 3]]
            
        """
        # Filter the connectivity to keep only the specified elements and store in a new connectivity object
        kept_connectivity = self._connectivity.keep_elements(element_indices, inplace=force_inplace)
        
        # Set the kept connectivity to the mesh
        self._connectivity = kept_connectivity
        
        # Validate the updated connectivity
        self._internal_check_connectivity()
        
    
    def remove_connectivity(self, element_indices: ArrayLike, force_inplace: bool = False) -> None:
        r"""
        Remove specified elements from the mesh by specifying their indices in the connectivity array.
        
        This method is an extension of the :obj:`connectivity.remove_elements` method that allows removing specified elements from the mesh while ensuring that the connectivity structure is maintained and validated.
        This method will ensure that the specified element indices are compatible with the existing mesh structure and will update the mesh's connectivity accordingly.

        .. note::

            The new connectivity will be stored in a new connectivity object (Copy of the original connectivity object with only the remaining elements).
            The properties of the remaining elements will be updated to keep only the properties of the remaining elements and stored in the new connectivity object.
            
        .. seealso::
        
            - :meth:`keep_connectivity` to keep only specified elements in the mesh.
            - :meth:`filter_connectivity` to filter the connectivity of the mesh using a boolean mask.
            - :class:`Connectivity` for more information on the connectivity structure and element types.
            
        Parameters
        ----------
        element_indices: ArrayLike
            An array of shape :math:`(R,)` containing the indices of the elements to remove from the mesh, where :math:`R` is the number of elements to remove.
            
        force_inplace: :class:`bool`, optional
            If :obj:`True`, the connectivity will be updated in place, modifying the original connectivity object. 
            If :obj:`False`, a new connectivity object will be created to store the updated connectivity, leaving the original connectivity object unchanged. 
            By default :obj:`False`.
            
        
        Raises
        ------
        TypeError
            If :obj:`element_indices` is not an array-like object or is not an integer array.
            If :obj:`force_inplace` is not a boolean.
            
        ValueError
            If :obj:`element_indices` does not have the correct shape or contains invalid indices.
            
        
        Examples
        --------
        Create a simple :class:`Mesh` instance.
        
        .. code-block:: python
            :linenos:
            
            import numpy as np
            from pysdic import Mesh, PointCloud
            
            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = Mesh(points, connectivity, element_type="triangle_3")
            
        Remove the first and third elements from the mesh.
        
        .. code-block:: python
            :linenos:
            
            element_indices = np.array([0, 2])
            mesh3d.remove_connectivity(element_indices)
            
            print(mesh3d.connectivity)
            # Output: Connectivity with 2 elements [[0, 1, 3], [1, 2, 3]]
            
        """
        # Filter the connectivity to remove the specified elements and store in a new connectivity object
        remaining_connectivity = self._connectivity.remove_elements(element_indices, inplace=force_inplace)
        
        # Set the remaining connectivity to the mesh
        self._connectivity = remaining_connectivity
        
        # Validate the updated connectivity
        self._internal_check_connectivity()
        
    
    def filter_vertices(self, mask: ArrayLike, force_inplace: bool = False) -> None:
        r"""
        Filter the vertices of the mesh by keeping only the vertices corresponding to the True values in the mask, remove all the elements that are connected to the removed vertices and remap the vertices indices accordingly.
        
        This method must be used instead of the :obj:`vertices.filter_points` method because it allows filtering the vertices of the mesh while ensuring that the connectivity structure is maintained and validated.
        This method will ensure that the mask is compatible with the existing mesh structure and will update the mesh's vertices accordingly.

        .. note::

            The new vertices will be stored in a new vertices object (Copy of the original vertices object with only the filtered vertices).
            The new connectivity will be stored in a new connectivity object (Copy of the original connectivity object with only the remaining elements that are not connected to the removed vertices and with remapped vertex indices).
            The properties of the filtered vertices will be updated to keep only the properties of the kept vertices and stored in the new vertices object.
            
        .. seealso::
        
            - :meth:`keep_vertices` to keep only specified vertices in the mesh.
            - :meth:`remove_vertices` to remove specified vertices from the mesh.
            - :class:`PointCloud` for more information on the vertices structure.
            - :class:`Connectivity` for more information on the connectivity structure and element types.
            
            
        Parameters
        ----------
        mask: ArrayLike
            A boolean array of shape (:math:`N_v`,) where :math:`N_v` is the number of vertices in the mesh, indicating which vertices to keep (True) or remove (False) from the mesh.
            
        force_inplace: :class:`bool`, optional
            If :obj:`True`, the vertices will be updated in place, modifying the original vertices object and the connectivity object will be updated in place as well.
            If :obj:`False`, new vertices and connectivity objects will be created to store the updated vertices and connectivity, leaving the original vertices and connectivity objects unchanged.
            By default :obj:`False`.
            
    
        Raises
        ------
        TypeError
            If :obj:`mask` is not an array-like object or is not a boolean array.
            If :obj:`force_inplace` is not a boolean.
            
        ValueError
            If :obj:`mask` does not have the correct shape or is not a boolean array.
            
        
        Examples
        --------
        Create a simple :class:`Mesh` instance.
        
        .. code-block:: python
            :linenos:
        
            import numpy as np
            from pysdic import Mesh, PointCloud
            
            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = Mesh(points, connectivity, element_type="triangle_3")
            
        Filter the vertices of the mesh to remove the first vertex and all the elements connected to it.
        
        .. code-block:: python
            :linenos:
        
            mask = np.array([False, True, True, True])
            mesh3d.filter_vertices(mask)
            
            print(mesh3d.vertices)
            # Output: PointCloud with 3 points [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            print(mesh3d.connectivity)
            # Output: Connectivity with 1 element [[0, 1, 2]]
            
        """
        mask = numpy.asarray(mask, dtype=bool)
        if mask.shape != (self.n_vertices,):
            raise ValueError(f"mask must have shape ({self.n_vertices},), got {mask.shape}.")
        
        # Create the indices array of vertices to remove
        # Computation will be ensure by the remove_vertices method
        remove_indices = numpy.where(~mask)[0]
        
        # Remove the vertices corresponding to the False values in the mask and update the connectivity accordingly
        self.remove_vertices(remove_indices, force_inplace=force_inplace)
        
        
    def keep_vertices(self, vertex_indices: ArrayLike, force_inplace: bool = False) -> None:
        r"""
        Keep only the specified vertices in the mesh by specifying their indices in the vertices array, remove all the elements that are connected to the removed vertices and remap the vertices indices accordingly.
        
        This method must be used instead of the :obj:`vertices.keep_points` method because it allows keeping only specified vertices in the mesh while ensuring that the connectivity structure is maintained and validated.
        This method will ensure that the specified vertex indices are compatible with the existing mesh structure and will update the mesh's vertices accordingly.

        .. note::

            The new vertices will be stored in a new vertices object (Copy of the original vertices object with only the kept vertices).
            The new connectivity will be stored in a new connectivity object (Copy of the original connectivity object with only the remaining elements that are not connected to the removed vertices and with remapped vertex indices).
            The properties of the kept vertices will be updated to keep only the properties of the kept vertices and stored in the new vertices object.
            
        .. seealso::
        
            - :meth:`filter_vertices` to filter the vertices of the mesh using a boolean mask.
            - :meth:`remove_vertices` to remove specified vertices from the mesh.
            - :class:`PointCloud` for more information on the vertices structure.
            - :class:`Connectivity` for more information on the connectivity structure and element types.
            
        
        Parameters
        ----------
        vertex_indices: ArrayLike
            An array of shape (:math:`R`,) containing the indices of the vertices to keep in the mesh, where :math:`R` is the number of vertices to keep.

        force_inplace: :class:`bool`, optional
            If :obj:`True`, the vertices will be updated in place, modifying the original vertices object and the connectivity object will be updated in place as well.
            If :obj:`False`, new vertices and connectivity objects will be created to store the updated vertices and connectivity, leaving the original vertices and connectivity objects unchanged.
            By default :obj:`False`.
            
        
        Raises
        ------
        TypeError
            If :obj:`vertex_indices` is not an array-like object or is not an integer array.
            If :obj:`force_inplace` is not a boolean. 
            
        ValueError
            If :obj:`vertex_indices` does not have the correct shape or contains invalid indices.
            
        
        Examples
        --------
        Create a simple :class:`Mesh` instance.
        
        .. code-block:: python
            :linenos:
        
            import numpy as np
            from pysdic import Mesh, PointCloud
            
            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = Mesh(points, connectivity, element_type="triangle_3")
            
        Remove the first vertex from the mesh and all the elements connected to it, keeping only the remaining vertices and elements.
        
        .. code-block:: python
            :linenos:
        
            vertex_indices = np.array([1, 2, 3])
            mesh3d.keep_vertices(vertex_indices)
            
            print(mesh3d.vertices)
            # Output: PointCloud with 3 points [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            
            print(mesh3d.connectivity)
            # Output: Connectivity with 1 element [[0, 1, 2]]
        
        """
        vertex_indices = numpy.asarray(vertex_indices, dtype=numpy.int64)
        if vertex_indices.ndim != 1:
            raise ValueError(f"vertex_indices must be a 1D array with shape (R,), where R is the number of vertices to keep. Got shape {vertex_indices.shape}.")
        if numpy.any(vertex_indices < 0) or numpy.any(vertex_indices >= self.n_vertices):
            raise ValueError("vertex_indices contains invalid indices.")
        
        # Create the indices array of vertices to remove
        # Computation will be ensure by the remove_vertices method
        all_indices = numpy.arange(self.n_vertices)
        remove_indices = numpy.setdiff1d(all_indices, vertex_indices)
        
        # Remove the vertices corresponding to the False values in the mask and update the connectivity accordingly
        self.remove_vertices(remove_indices, force_inplace=force_inplace)
        
        
    def remove_vertices(self, vertex_indices: ArrayLike, force_inplace: bool = False) -> None:
        r"""
        Remove specified vertices from the mesh by specifying their indices in the vertices array, remove all the elements that are connected to the removed vertices and remap the vertices indices accordingly.
        
        This method must be used instead of the :obj:`vertices.remove_points` method because it allows removing specified vertices from the mesh while ensuring that the connectivity structure is maintained and validated.
        This method will ensure that the specified vertex indices are compatible with the existing mesh structure and will update the mesh's vertices accordingly.

        .. note::

            The new vertices will be stored in a new vertices object (Copy of the original vertices object with only the remaining vertices).
            The new connectivity will be stored in a new connectivity object (Copy of the original connectivity object with only the remaining elements that are not connected to the removed vertices and with remapped vertex indices).
            The properties of the remaining vertices will be updated to keep only the properties of the remaining vertices and stored in the new vertices object.
            
        .. seealso::
        
            - :meth:`filter_vertices` to filter the vertices of the mesh using a boolean mask.
            - :meth:`keep_vertices` to keep only specified vertices in the mesh.
            - :class:`PointCloud` for more information on the vertices structure.
            - :class:`Connectivity` for more information on the connectivity structure and element types.
            
        
        Parameters
        ----------
        vertex_indices: ArrayLike
            An array of shape (:math:`R`,) containing the indices of the vertices to remove from the mesh, where :math:`R` is the number of vertices to remove.

        force_inplace: :class:`bool`, optional
            If :obj:`True`, the vertices will be updated in place, modifying the original vertices object and the connectivity object will be updated in place as well.
            If :obj:`False`, new vertices and connectivity objects will be created to store the updated vertices and connectivity, leaving the original vertices and connectivity objects unchanged.
            By default :obj:`False`.
            
        
        Raises
        ------
        TypeError
            If :obj:`vertex_indices` is not an array-like object or is not an integer array.
            If :obj:`force_inplace` is not a boolean. 
            
        ValueError
            If :obj:`vertex_indices` does not have the correct shape or contains invalid indices.
            
        
        Examples
        --------
        Create a simple :class:`Mesh` instance.
        
        .. code-block:: python
            :linenos:
        
            import numpy as np
            from pysdic import Mesh, PointCloud
            
            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = Mesh(points, connectivity, element_type="triangle_3")
            
        Remove the first vertex from the mesh and all the elements connected to it, keeping only the remaining vertices and elements.
        
        .. code-block:: python
            :linenos:
        
            vertex_indices = np.array([0])
            mesh3d.remove_vertices(vertex_indices)
            
            print(mesh3d.vertices)
            # Output: PointCloud with 3 points [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            
            print(mesh3d.connectivity)
            # Output: Connectivity with 1 element [[0, 1, 2]]
            
        """
        vertex_indices = numpy.asarray(vertex_indices, dtype=numpy.int64)
        if vertex_indices.ndim != 1:
            raise ValueError(f"vertex_indices must be a 1D array with shape (R,), where R is the number of vertices to remove. Got shape {vertex_indices.shape}.")
        if numpy.any(vertex_indices < 0) or numpy.any(vertex_indices >= self.n_vertices):
            raise ValueError("vertex_indices contains invalid indices.")
        
        # Unique indices
        unique_indices = numpy.unique(vertex_indices)
        sorted_unique_indices = numpy.sort(unique_indices)
        
        # 1. Remove the elements that are connected to the removed vertices
        # --------------------------
        # Create a mask of elements to keep (Elements that are not connected to any of the removed vertices)
        keep_elements = numpy.where(numpy.isin(self.connectivity.elements, unique_indices).any(axis=1))[0]
        
        # Filter the connectivity to keep only the remaining elements and store in a new connectivity object
        remaining_connectivity = self._connectivity.keep_elements(keep_elements, inplace=force_inplace)
        
        # 2. Remove the vertices corresponding to the vertex_indices and update the connectivity accordingly
        # --------------------------
        # Create a array "shift" that indicates how many vertices have been removed before each index
        shift = numpy.zeros(self.n_vertices, dtype=int)
        shift[sorted_unique_indices] = 1
        shift = numpy.cumsum(shift)
        
        # Remove the vertices corresponding to the vertex_indices and store in a new vertices object
        remaining_vertices = self._vertices.remove_points(unique_indices, inplace=force_inplace)
        
        # Set the remaining vertices to the mesh
        self._vertices = remaining_vertices
        
        # Update the connectivity by remapping the vertex indices according to the shift array and store in a new connectivity object
        remaining_elements = remaining_connectivity.elements - shift[remaining_connectivity.elements]
        if force_inplace:
            self._connectivity.elements = remaining_elements
        else:
            self._connectivity = Connectivity(remaining_elements, element_type=self.element_type)
            
        # Ensure the updated vertices and connectivity are valid
        self._internal_check_vertices()
        self._internal_check_connectivity()
        
    
    def get_used_vertices_mask(self) -> numpy.ndarray:
        r"""
        Get a boolean mask indicating which vertices are used in the connectivity of the mesh.

        Returns
        -------
        :class:`numpy.ndarray`
            A boolean array of shape (:math:`N_v`,) where :math:`N_v` is the number of vertices in the mesh, indicating which vertices are used (True) or unused (False) in the connectivity.

        
        Examples
        --------
        Create a simple :class:`Mesh` instance.

        .. code-block:: python
            :linenos:

            import numpy as np
            from pysdic import Mesh, PointCloud

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = Mesh(points, connectivity, element_type="triangle_3")

        Get the mask of used vertices in the connectivity.

        .. code-block:: python
            :linenos:

            used_vertices_mask = mesh3d.get_used_vertices_mask()
            print(used_vertices_mask)
            # Output: [ True  True  True  True False]

        """
        used_vertices_mask = numpy.zeros(self.n_vertices, dtype=bool)
        used_vertices_mask[numpy.unique(self.connectivity.elements)] = True
        return used_vertices_mask
    
    
    def remove_unused_vertices(self, force_inplace: bool = False) -> None:
        r"""
        Remove all unused vertices from the mesh, where unused vertices are defined as vertices that are not referenced in the connectivity of the mesh.

        .. note::

            The new vertices will be stored in a new vertices object (Copy of the original vertices object with only the used vertices).
            The new connectivity will be stored in a new connectivity object (Copy of the original connectivity object with only the remaining elements that are not connected to the removed vertices and with remapped vertex indices).
            The properties of the remaining vertices will be updated to keep only the properties of the remaining vertices and stored in the new vertices object.
            
        .. seealso::
        
            - :meth:`get_used_vertices_mask` to get a boolean mask of the used vertices in the connectivity.
            - :meth:`remove_vertices` to remove specified vertices from the mesh.
            - :class:`PointCloud` for more information on the vertices structure.
            
            
        Parameters
        ----------
        force_inplace: :class:`bool`, optional
            If :obj:`True`, the vertices will be updated in place, modifying the original vertices object and the connectivity object will be updated in place as well.
            If :obj:`False`, new vertices and connectivity objects will be created to store the updated vertices and connectivity, leaving the original vertices and connectivity objects unchanged.
            By default :obj:`False`.
            
            
        Examples
        --------
        Create a simple :class:`Mesh` instance.

        .. code-block:: python
        
            import numpy as np
            from pysdic import Mesh, PointCloud
            
            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = Mesh(points, connectivity, element_type="triangle_3")
            
        Remove all unused vertices from the mesh.
        
        .. code-block:: python
            :linenos:
        
            mesh3d.remove_unused_vertices()
            
            print(mesh3d.vertices)
            # Output: PointCloud with 4 points [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
            
            print(mesh3d.connectivity)
            # Output: Connectivity with 4 elements [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
            
        """
        used_vertices_mask = self.get_used_vertices_mask()
        self.filter_vertices(used_vertices_mask, force_inplace=force_inplace)
        

    # =======================
    # Precompute properties
    # =======================
    def get_precomputed(self, key: str) -> Any:
        r"""
        Get the value of a precomputed property stored in the mesh by its key.

        Parameters
        ----------
        key: :class:`str`
            The key of the precomputed property to retrieve.

        Returns
        -------
        Any
            The value of the precomputed property associated with the specified key, or :obj:`None` if the key does not exist in the precomputed properties of the mesh.
        
        
        Examples
        --------
        Create a simple :class:`Mesh` instance.

        .. code-block:: python
            :linenos:

            import numpy as np
            from pysdic import Mesh, PointCloud

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3], [1, 2, 4]])
            mesh3d = Mesh(points, connectivity)

        Precompute the vertices path distance matrix from the connectivity and store it in the precomputed properties of the mesh with the key 'vertices_adjacency_matrix'.

        .. code-block:: python
            :linenos:

            distance_matrix = mesh3d.precompute_vertices_adjacency_matrix()

        Get the value of the precomputed vertices adjacency matrix using its key.

        .. code-block:: python
            :linenos:

            retrieved_distance_matrix = mesh3d.get_precomputed('vertices_adjacency_matrix')
            print(retrieved_distance_matrix)

        """
        return self._get_precomputed(key)
    

    def list_precomputed(self) -> List[str]:
        r"""
        List all the keys of the precomputed properties that are currently stored in the mesh.

        Returns
        -------
        List[:class:`str`]
            A list of strings representing the keys of the precomputed properties stored in the mesh.
    
    
        Examples
        --------
        Create a simple :class:`Mesh` instance.

        .. code-block:: python
            :linenos:

            import numpy as np
            from pysdic import Mesh, PointCloud

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3], [1, 2, 4]])
            mesh3d = Mesh(points, connectivity)

        List all the keys of the precomputed properties stored in the mesh.

        .. code-block:: python
            :linenos:

            precomputed_keys = mesh3d.list_precomputed()
            print(precomputed_keys)

        """
        return list(self._precomputed.keys())
    
    
    def clear_precomputed(self) -> None:
        r"""
        Clear all the precomputed properties stored in the mesh, removing all key-value pairs from the precomputed properties dictionary.
        """
        self._clear_precomputed()
        
        
    def compute_vertices_adjacency_matrix(self, max_distance: Optional[Integral] = None, precompute: bool = False) -> numpy.ndarray:
        r"""
        Compute the vertices path distance matrix (adjacency matrix) from the connectivity of the mesh.

        The distance between two vertices is defined as the minimum number of elements that must be traversed to go from one vertex to another.
        The output is a symmetric square matrix of shape (:math:`N_v`, :math:`N_v`), where :math:`N_v` is the number of vertices.
        
        .. note::
        
            The method is a convenience wrapper around the :func:`pysdic.compute_vertices_adjacency_matrix` function, which computes the vertices adjacency matrix using a breadth-first search (BFS) algorithm for each vertex in the mesh.
        
        .. warning::
        
            The same object returned by this method is stored in the precomputed properties of the mesh with the key 'vertices_adjacency_matrix' if :obj:`precompute` is set to :obj:`True`.
            Please be cautious when modifying the returned adjacency matrix, as it will affect the value stored in the precomputed properties of the mesh.
            
        Parameters
        ----------
        max_distance: Optional[Integral], optional
            The maximum distance to consider between vertices. If the distance between two vertices exceeds this value,
            the corresponding entry in the adjacency matrix will be set to -1. If :obj:`None`, there is no maximum distance and all distances will be computed. By default :obj:`None`.
            
        precompute: :class:`bool`, optional
            If :obj:`True`, the computed vertices adjacency matrix will be stored in the precomputed properties of the mesh with the key 'vertices_adjacency_matrix'. If :obj:`False`, the computed vertices adjacency matrix will not be stored in the precomputed properties of the mesh. By default :obj:`False`.
    
    
        Returns
        -------
        :class:`numpy.ndarray`
            A square array of shape (:math:`N_v`, :math:`N_v`) representing the path distance matrix between vertices.
            
            
        See Also
        --------
        :func:`pysdic.compute_vertices_adjacency_matrix` 
            For more information on the algorithm used to compute the vertices adjacency matrix and an example of the output format.

        """
        if not isinstance(precompute, bool):
            raise ValueError(f"precompute must be a boolean value, got {type(precompute)}.")
        
        value = compute_vertices_adjacency_matrix(self.connectivity.elements, self.n_vertices, max_distance=max_distance)
        if precompute:
            self._set_precomputed('vertices_adjacency_matrix', value)
        return value
    
    
    def compute_vertices_neighborhood(self, max_distance: Integral = 1, self_neighborhood: bool = True, precompute: bool = False) -> List[List[int]]:
        r"""
        Compute the neighborhood of each vertex in the mesh, where the neighborhood of a vertex is defined as the set of vertices that are within a specified distance from the vertex in the connectivity of the mesh.
        
        .. note::
        
            The function is a convenience wrapper around the :func:`pysdic.compute_vertices_neighborhood` function, which computes the neighborhood of each vertex using a breadth-first search (BFS) algorithm for each vertex in the mesh, where the BFS is limited to a maximum distance specified by :obj:`max_distance`.

        .. note::
        
            The neighborhood of a vertex is computed using a breadth-first search (BFS) algorithm for each vertex in the mesh, where the BFS is limited to a maximum distance specified by :obj:`max_distance`.
            
        .. warning::
        
            The same object returned by this method is stored in the precomputed properties of the mesh with the key 'vertices_neighborhood' if :obj:`precompute` is set to :obj:`True`.
            Please be cautious when modifying the returned neighborhood list, as it will affect the value stored in the precomputed properties of the mesh.
            
            
        Parameters
        ----------
        max_distance: Integral, optional
            The maximum distance to consider for the neighborhood of each vertex. The neighborhood will include all vertices that are within this distance from the vertex in the connectivity of the mesh. By default 1,
            which means that only directly adjacent vertices will be included in the neighborhood.
            
        self_neighborhood: :class:`bool`, optional
            If :obj:`True`, the neighborhood of each vertex will include the vertex itself (i.e., the vertex will be considered as part of its own neighborhood). If :obj:`False`, the neighborhood of each vertex will not include the vertex itself. By default :obj:`True`.
            
        precompute: :class:`bool`, optional
            If :obj:`True`, the computed vertices neighborhood will be stored in the precomputed properties of the mesh with the key 'vertices_neighborhood'. If :obj:`False`, the computed vertices neighborhood will not be stored in the precomputed properties of the mesh. By default :obj:`Falses`.
                        
            
        Returns
        -------       
        List[List[:class:`int`]]
            A list where each element is a list containing the indices of the neighboring vertices for the corresponding vertex in the mesh.
            
        
        See Also
        --------
        :func:`pysdic.compute_vertices_neighborhood`
            For more information on the algorithm used to compute the vertices neighborhood and an example of the output format.
            
        """
        if not isinstance(precompute, bool):
            raise ValueError(f"precompute must be a boolean value, got {type(precompute)}.")
        
        value = compute_vertices_neighborhood(self.connectivity.elements, self.n_vertices, max_distance=max_distance, self_neighborhood=self_neighborhood)
        if precompute:
            self._set_precomputed('vertices_neighborhood', value)
        return value            
            
            
    def compute_elements_adjacency_matrix(self, max_distance: Optional[int] = None, precompute: bool = False) -> numpy.ndarray:
        r"""
        Compute the elements adjacency matrix from the connectivity of the mesh.

        The adjacency between two elements is defined as 1 if they share at least one vertex, and 0 otherwise.
        The output is a symmetric square matrix of shape (:math:`N_e`, :math:`N_e`), where :math:`N_e` is the number of elements.
        
        .. note::
        
            The method is a convenience wrapper around the :func:`pysdic.compute_elements_adjacency_matrix` function, which computes the elements adjacency matrix using a breadth-first search (BFS) algorithm for each element in the mesh.
            
        .. warning::
        
            The same object returned by this method is stored in the precomputed properties of the mesh with the key 'elements_adjacency_matrix' if :obj:`precompute` is set to :obj:`True`.
            Please be cautious when modifying the returned adjacency matrix, as it will affect the value stored in the precomputed properties of the mesh.
            
            
        Parameters
        ----------
        max_distance: Integral, optional
            The maximum distance to consider between elements. If the distance between two elements exceeds this value,
            the corresponding entry in the adjacency matrix will be set to -1. If :obj:`None`, there is no maximum distance and all adjacencies will be computed. By default :obj:`None`.
            
        precompute: :class:`bool`, optional
            If :obj:`True`, the computed elements adjacency matrix will be stored in the precomputed properties of the mesh with the key 'elements_adjacency_matrix'. If :obj:`False`, the computed elements adjacency matrix will not be stored in the precomputed properties of the mesh. By default :obj:`False`.
            
            
        Returns
        -------
        :class:`numpy.ndarray`
            A square array of shape (:math:`N_e`, :math:`N_e`) representing the adjacency matrix between elements.
            
            
        See Also
        --------
        :func:`pysdic.compute_elements_adjacency_matrix`
            For more information on the algorithm used to compute the elements adjacency matrix and an example of the output format.
             
        """
        if not isinstance(precompute, bool):
            raise ValueError(f"precompute must be a boolean value, got {type(precompute)}.")
        
        value = compute_elements_adjacency_matrix(self.connectivity.elements, max_distance=max_distance)
        if precompute:      
            self._set_precomputed('elements_adjacency_matrix', value)
        return value
    

    def compute_elements_neighborhood(self, max_distance: Integral = 1, self_neighborhood: bool = True, precompute: bool = False) -> List[List[int]]:
        r"""
        Compute the neighborhood of each element in the mesh, where the neighborhood of an element is defined as the set of elements that are within a specified distance from the element in the connectivity of the mesh.

        .. note::
        
            The function is a convenience wrapper around the :func:`pysdic.compute_elements_neighborhood` function, which computes the neighborhood of each element using a breadth-first search (BFS) algorithm for each element in the mesh, where the BFS is limited to a maximum distance specified by :obj:`max_distance`.
            
        .. warning::
        
            The same object returned by this method is stored in the precomputed properties of the mesh with the key 'elements_neighborhood' if :obj:`precompute` is set to :obj:`True`.
            Please be cautious when modifying the returned neighborhood list, as it will affect the value stored in the precomputed properties of the mesh.
            
            
        Parameters
        ----------
        max_distance: Integral, optional
            The maximum distance to consider for the neighborhood of each element. The neighborhood will include all elements that are within this distance from the element in the connectivity of the mesh. By default 1,
            which means that only directly adjacent elements will be included in the neighborhood.
            
        self_neighborhood: :class:`bool`, optional
            If :obj:`True`, the neighborhood of each element will include the element itself (i.e., the element will be considered as part of its own neighborhood). If :obj:`False`, the neighborhood of each element will not include the element itself. By default :obj:`True`.
            
        precompute: :class:`bool`, optional
            If :obj:`True`, the computed elements neighborhood will be stored in the precomputed properties of the mesh with the key 'elements_neighborhood'. If :obj:`False`, the computed elements neighborhood will not be stored in the precomputed properties of the mesh. By default :obj:`False`.
            
            
        Returns
        -------
        List[List[:class:`int`]]
            A list where each element is a list containing the indices of the neighboring elements for the corresponding element in the mesh.
            
            
        See Also
        --------
        :func:`pysdic.compute_elements_neighborhood`
            For more information on the algorithm used to compute the elements neighborhood and an example of the output format.
            
        """
        if not isinstance(precompute, bool):
            raise ValueError(f"precompute must be a boolean value, got {type(precompute)}.")
        
        value = compute_elements_neighborhood(self.connectivity.elements, max_distance=max_distance, self_neighborhood=self_neighborhood)
        if precompute:
            self._set_precomputed('elements_neighborhood', value)
        return value
    
    
    def compute_vertices_neighborhood_statistics(
        self,
        property: Union[str, ArrayLike],
        statistics: Union[str, List[str]],
    ) -> Dict[str, numpy.ndarray]:
        r"""
        Compute the specified statistics for the given property array defined at the vertices of a mesh based on the neighborhood information for each vertex.
        
        .. note::
        
            The function is a convenience wrapper around the :func:`pysdic.compute_neighborhood_statistics` function, which computes the neighborhood statistics for each vertex of a graph.
            
        .. important::
        
            - ``vertices_neighborhood`` must be precomputed before calling this function.
            
        Implemented statistics include:
     
        +----------------+--------------------------------------------------------------------------------------------------------------+
        | Statistic      | Description                                                                                                  |
        +================+==============================================================================================================+
        | ``mean``       | Compute the mean of the property values from the neighbors.                                                  |
        +----------------+--------------------------------------------------------------------------------------------------------------+
        | ``median``     | Compute the median of the property values from the neighbors.                                                |
        +----------------+--------------------------------------------------------------------------------------------------------------+
        | ``min``        | Compute the minimum of the property values from the neighbors.                                               |
        +----------------+--------------------------------------------------------------------------------------------------------------+
        | ``max``        | Compute the maximum of the property values from the neighbors.                                               |
        +----------------+--------------------------------------------------------------------------------------------------------------+
        | ``std``        | Compute the standard deviation of the property values from the neighbors.                                    |
        +----------------+--------------------------------------------------------------------------------------------------------------+
        | ``var``        | Compute the variance of the property values from the neighbors.                                              |
        +----------------+--------------------------------------------------------------------------------------------------------------+
        | ``Q1``         | Compute the first quartile (25th percentile) of the property values from the neighbors.                      |
        +----------------+--------------------------------------------------------------------------------------------------------------+
        | ``Q3``         | Compute the third quartile (75th percentile) of the property values from the neighbors.                      |
        +----------------+--------------------------------------------------------------------------------------------------------------+
        | ``IQR``        | Compute the interquartile range (IQR) of the property values from the neighbors, defined as Q3 - Q1.         |
        +----------------+--------------------------------------------------------------------------------------------------------------+
            
        Parameters
        ----------
        property: Union[:class:`str`, ArrayLike]
            An array of shape :math:`(N_v, P)` containing the property values defined at the vertices of the mesh. If 1D-array is provided, it will be treated as a single-component property of shape :math:`(N, 1)`. If a string, the property will be retrieved from the properties of the vertices using the string as a key.
            
        statistics : Union[:class:`str`, Sequence[:class:`str`]]
            A string or a sequence of strings specifying the statistic(s) to compute for each vertex/element based on its neighbors.
            
            
        Returns
        -------
        Dict[:class:`str`, :class:`numpy.ndarray`]
            A dictionary where the keys are the names of the computed statistics and the values are arrays of shape :math:`(N_v, P)` containing the computed statistic values for each vertex and each property.
            
            
        Raises
        ------
        TypeError
            If the input property array is not 1D or 2D.
            If the statistic parameter is not a string or a sequence of strings.
        
        ValueError
            If the statistic parameter contains an unsupported statistic name.
            
            
        See Also
        --------
        :func:`pysdic.compute_neighborhood_statistics`
            For more information on the algorithm used to compute the neighborhood statistics and an example of how to use it.
        
        """
        if isinstance(property, str):
            if property in self.vertices:
                property = self.vertices.get_property(property)
            else:
                raise ValueError(f"The property '{property}' is not found in the properties of the vertices. Please provide a valid property name or an array of shape (N_v, P) containing the property values defined at the vertices.")
        property_array = numpy.asarray(property, dtype=numpy.float64)
            
        if not self._has_precomputed('vertices_neighborhood'):
            raise ValueError("The vertices neighborhood must be precomputed to compute the neighborhood statistics. Please precompute the vertices neighborhood using the compute_vertices_neighborhood method before calling this method.")
        
        neighborhood = self.get_precomputed('vertices_neighborhood')
        return compute_neighborhood_statistics(property_array, neighborhood, statistics)
    
    
    def compute_elements_neighborhood_statistics(
        self,
        property: Union[str, ArrayLike],
        statistics: Union[str, List[str]],
    ) -> Dict[str, numpy.ndarray]:
        r"""
        Compute the specified statistics for the given property array defined at the elements of a mesh based on the neighborhood information for each element.
        
        .. note::
        
            The function is a convenience wrapper around the :func:`pysdic.compute_neighborhood_statistics` function, which computes the neighborhood statistics for each vertex of a graph.
            
        .. important::
        
            - ``elements_neighborhood`` must be precomputed before calling this function.
            
        Implemented statistics include:
        
        +----------------+--------------------------------------------------------------------------------------------------------------+
        | Statistic      | Description                                                                                                  |
        +================+==============================================================================================================+
        | ``mean``       | Compute the mean of the property values from the neighbors.                                                  |
        +----------------+--------------------------------------------------------------------------------------------------------------+
        | ``median``     | Compute the median of the property values from the neighbors.                                                |
        +----------------+--------------------------------------------------------------------------------------------------------------+
        | ``min``        | Compute the minimum of the property values from the neighbors.                                               |
        +----------------+--------------------------------------------------------------------------------------------------------------+
        | ``max``        | Compute the maximum of the property values from the neighbors.                                               |
        +----------------+--------------------------------------------------------------------------------------------------------------+
        | ``std``        | Compute the standard deviation of the property values from the neighbors.                                    |
        +----------------+--------------------------------------------------------------------------------------------------------------+
        | ``var``        | Compute the variance of the property values from the neighbors.                                              |
        +----------------+--------------------------------------------------------------------------------------------------------------+
        | ``Q1``         | Compute the first quartile (25th percentile) of the property values from the neighbors.                      |
        +----------------+--------------------------------------------------------------------------------------------------------------+
        | ``Q3``         | Compute the third quartile (75th percentile) of the property values from the neighbors.                      |
        +----------------+--------------------------------------------------------------------------------------------------------------+
        | ``IQR``        | Compute the interquartile range (IQR) of the property values from the neighbors, defined as Q3 - Q1.         |
        +----------------+--------------------------------------------------------------------------------------------------------------+
        
        
        Parameters
        ----------
        property: Union[:class:`str`, ArrayLike]
            An array of shape :math:`(N_e, P)` containing the property values defined at the elements of the mesh. If 1D-array is provided, it will be treated as a single-component property of shape :math:`(N, 1)`. If a string, the property will be retrieved from the properties of the elements using the string as a key.
            
        statistics : Union[:class:`str`, Sequence[:class:`str`]]
            A string or a sequence of strings specifying the statistic(s) to compute for each vertex/element based on its neighbors.
            
        
        Returns
        -------
        Dict[:class:`str`, :class:`numpy.ndarray`]
            A dictionary where the keys are the names of the computed statistics and the values are arrays of shape :math:`(N_e, P)` containing the computed statistic values for each element and each property.
            
            
        Raises
        ------
        TypeError
            If the input property array is not 1D or 2D.
            If the statistic parameter is not a string or a sequence of strings.
            
        ValueError
            If the statistic parameter contains an unsupported statistic name.
            
        
        See Also
        --------
        :func:`pysdic.compute_neighborhood_statistics`
            For more information on the algorithm used to compute the neighborhood statistics and an example of how to use it.
            
        """
        if isinstance(property, str):
            if property in self.connectivity:
                property = self.connectivity.get_property(property)
            else:
                raise ValueError(f"The property '{property}' is not found in the properties of the connectivity. Please provide a valid property name or an array of shape (N_e, P) containing the property values defined at the elements.")
        property_array = numpy.asarray(property, dtype=numpy.float64)
            
        if not self._has_precomputed('elements_neighborhood'):
            raise ValueError("The elements neighborhood must be precomputed to compute the neighborhood statistics. Please precompute the elements neighborhood using the compute_elements_neighborhood method before calling this method.")
        
        neighborhood = self.get_precomputed('elements_neighborhood')
        return compute_neighborhood_statistics(property_array, neighborhood, statistics)
    
    
    def remap_integration_points_neighborhood(
        self,
        integration_points: IntegrationPoints,
        precompute: bool = False
    ) -> List[List[int]]:
        r"""
        Remap the neighborhood of integration points in the mesh based on the element neighborhood.

        This function organizes integration points into groups based on their element indices, 
        and for each element's neighborhood, returns the corresponding integration points that 
        belong to that neighborhood.
        
        .. important::

            - ``elements_neighborhood`` must be precomputed before calling this function.
            
        Parameters
        ----------
        integration_points: :class:`IntegrationPoints`
            An instance of the :class:`IntegrationPoints` class containing the natural coordinates and element indices of the integration points for which the neighborhood remapping is to be performed.
            
        precompute: :class:`bool`, optional
            If :obj:`True`, the computed remapped neighborhood of integration points will be stored in the precomputed quantities of the :class:`IntegrationPoints` object with the key 'integration_points_neighborhood'. Default is :obj:`False`.
            
        
        Returns
        -------
        List[List[:class:`int`]]
            A list where each element is a list containing the indices of the integration points contained in the neighborhood of the corresponding element in the mesh. The length of the output list is equal to the number of elements in the mesh.
            
        
        Raises
        ------
        ValueError
            If the elements neighborhood is not precomputed in the mesh.
            If the input integration points do not have the required attributes (natural coordinates and element indices).
            
            
        See Also
        --------
        :meth:`Mesh.compute_integration_points_neighborhood_statistics`
            For computing statistics for integration points based on their remapped neighborhood.
            
        
        Examples
        --------
        Example: Remapping the neighborhood of integration points in a quadrangle mesh.

        This example demonstrates how to create a mesh with 4 quadrangular elements and 8 integration 
        points, and how to remap the neighborhoods based on the elements' neighborhoods.
        
        .. figure:: ../_static/adjacency/graph_mesh.png
            :align: center
            :width: 50%
            
            ``Quadrangle 4`` mesh.
        
        .. code-block:: python
            :linenos:
            
            import numpy as np
            from pysdic import Mesh, IntegrationPoints
            
            vertices = numpy.random.rand(10, 3) # 10 vertices with 3 coordinates (x, y, z)
            connectivity = np.array([[0, 1, 4, 5], [1, 2, 5, 6], [2, 3, 6, 7], [6, 7, 8, 9]])
            mesh = Mesh(vertices, connectivity, element_type="quadrangle_4")
            
            natural_coordinates = numpy.zeros((8, 2)) # 8 integrated points with 2 natural coordinates (xi, eta)
            element_indices = np.array([0, 0, 1, 1, 2, 2, 3, 3]) # Two integrated points are located in each element of the mesh
            integration_points = IntegrationPoints(natural_coordinates, element_indices)
            
            # Precomputed the elements neighborhood to be used for remapping the neighborhood of the integrated points
            neighbors = mesh.compute_elements_neighborhood(precompute=True, max_distance=1, self_neighborhood=True)
            
            print(f"elements neighborhood: length {len(neighbors)}")
            print(neighbors)
            
            # Remap the neighborhood of the integration points based on the neighborhood of the elements in the mesh
            remapped_neighborhood = mesh.remap_integration_points_neighborhood(integration_points, precompute=False)
            print(f"remapped neighborhood of integration points: length {len(remapped_neighborhood)}")
            print(remapped_neighborhood)
            
        .. code-block:: console
        
            elements neighborhood: length 4
            [[0, 1], [0, 1, 2, 3], [1, 2, 3], [1, 2, 3]]
            remapped neighborhood of integration points: length 4
            [[0, 1, 2, 3], [0, 1, 2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 7]]
            
        """
        self._check_integration_points(integration_points)
        if not self._has_precomputed('elements_neighborhood'):
            raise ValueError("The elements neighborhood must be precomputed to remap the neighborhood of integration points. Please precompute the elements neighborhood using the compute_elements_neighborhood method before calling this method.")
        
        if not isinstance(precompute, bool):
            raise ValueError(f"precompute must be a boolean value, got {type(precompute)}.")
        
        neighborhood = self.get_precomputed('elements_neighborhood') # List[List[int]] of length Ne, where Ne is the number of elements in the mesh and each element is a list of neighboring element indices.
        element_indices = integration_points.element_indices # shape (Npoints, K)
        
        remapped_neighborhood = defaultdict(list)
        
        for point_index, element_index in enumerate(element_indices):
            for neighbor_element_index in neighborhood[element_index]:
                remapped_neighborhood[neighbor_element_index].append(point_index)
                
        remapped_neighborhood_list = [remapped_neighborhood[element_index] for element_index in range(self.n_elements)]
        
        if precompute:
            integration_points._set_precomputed('integration_points_neighborhood', remapped_neighborhood_list)
        
        return remapped_neighborhood_list
        
            
    def compute_integration_points_neighborhood_statistics(
        self,
        integration_points: IntegrationPoints,
        property: Union[str, ArrayLike],
        statistics: Union[str, List[str]],     
    ) -> Dict[str, numpy.ndarray]:
        r"""
        Compute the specified statistics for the given property array defined at the integration points of a mesh based on the neighborhood information for each integration point.

        .. note::
        
            The function is a convenience wrapper around the :func:`pysdic.compute_neighborhood_statistics` function, which computes the neighborhood statistics for each vertex of a graph.
            
        .. important::
        
            - ``elements_neighborhood`` must be precomputed in the mesh before calling this function.
            - If ``integration_points_neighborhood`` is precomputed in :class:`IntegrationPoints`, it will be used, otherwise it will be computed on the fly using the :meth:`Mesh.remap_integration_points_neighborhood` method.
            
        Implemented statistics include:

        +----------------+--------------------------------------------------------------------------------------------------------------+
        | Statistic      | Description                                                                                                  |
        +================+==============================================================================================================+
        | ``mean``       | Compute the mean of the property values from the neighbors.                                                  |
        +----------------+--------------------------------------------------------------------------------------------------------------+
        | ``median``     | Compute the median of the property values from the neighbors.                                                |
        +----------------+--------------------------------------------------------------------------------------------------------------+
        | ``min``        | Compute the minimum of the property values from the neighbors.                                               |
        +----------------+--------------------------------------------------------------------------------------------------------------+
        | ``max``        | Compute the maximum of the property values from the neighbors.                                               |
        +----------------+--------------------------------------------------------------------------------------------------------------+
        | ``std``        | Compute the standard deviation of the property values from the neighbors.                                    |
        +----------------+--------------------------------------------------------------------------------------------------------------+
        | ``var``        | Compute the variance of the property values from the neighbors.                                              |
        +----------------+--------------------------------------------------------------------------------------------------------------+
        | ``Q1``         | Compute the first quartile (25th percentile) of the property values from the neighbors.                      |
        +----------------+--------------------------------------------------------------------------------------------------------------+
        | ``Q3``         | Compute the third quartile (75th percentile) of the property values from the neighbors.                      |
        +----------------+--------------------------------------------------------------------------------------------------------------+
        | ``IQR``        | Compute the interquartile range (IQR) of the property values from the neighbors, defined as Q3 - Q1.         |
        +----------------+--------------------------------------------------------------------------------------------------------------+
        
        
        Parameters
        ----------
        integration_points: :class:`IntegrationPoints`
            An instance of the :class:`IntegrationPoints` class containing the natural coordinates and element indices of the integration points for which the neighborhood statistics are to be computed.
        
        property: Union[:class:`str`, ArrayLike]
            An array of shape :math:`(N_{ip}, P)` containing the property values defined at the integration points of the mesh. If 1D-array is provided, it will be treated as a single-component property of shape :math:`(N, 1)`. If a string, the property will be retrieved from the properties of the integration points using the string as a key.
            
        statistics : Union[:class:`str`, Sequence[:class:`str`]]
            A string or a sequence of strings specifying the statistic(s) to compute for each vertex/element based on its neighbors.
            
        
        Returns
        -------
        Dict[:class:`str`, :class:`numpy.ndarray`]
            A dictionary where the keys are the names of the computed statistics and the values are arrays of shape :math:`(N_e, P)` containing the computed statistic values for each element and each property.
            
            
        Raises
        ------
        TypeError
            If the input property array is not 1D or 2D.
            If the statistic parameter is not a string or a sequence of strings.
            
        ValueError
            If the statistic parameter contains an unsupported statistic name.
            
        
        See Also
        --------
        :func:`pysdic.compute_neighborhood_statistics`
            For more information on the algorithm used to compute the neighborhood statistics (with :obj:`integrated_property` set to :obj:`True`) and an example of how to use it.
            
        """
        self._check_integration_points(integration_points)
        if not self._has_precomputed('elements_neighborhood'):
            raise ValueError("The elements neighborhood must be precomputed to compute the neighborhood statistics for integration points. Please precompute the elements neighborhood using the compute_elements_neighborhood method before calling this method.")
        
        if isinstance(property, str):
            if property in integration_points:
                property = integration_points.get_property(property)
            else:
                raise ValueError(f"The property '{property}' is not found in the properties of the integration points. Please provide a valid property name or an array of shape (N_ip, P) containing the property values defined at the integration points.")
        property_array = numpy.asarray(property, dtype=numpy.float64)
            
        if integration_points._has_precomputed('integration_points_neighborhood'):
            neighborhood = integration_points.get_precomputed('integration_points_neighborhood')
        else:
            neighborhood = self.remap_integration_points_neighborhood(integration_points, precompute=False)
            
        return compute_neighborhood_statistics(property_array, neighborhood, statistics, integrated_property=True)

            
    # =======================
    # Methods for shape functions
    # =======================
    def compute_shape_functions(
        self,
        integration_points: IntegrationPoints,
        return_derivatives: bool = False,
        precompute: bool = False,
        default: Real = 0.0
    ) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
        r"""
        Compute the shape functions and their first derivatives for the mesh elements at given :class:`IntegrationPoints` (convenient method for :func:`pysdic.compute_shape_functions`).
        The :obj:`element_type` must be specified in the mesh connectivity to compute the shape functions for the elements of the mesh.

        In a space of dimension :math:`E`, we consider a :math:`K`-dimensional element (with :math:`K \leq E`) defined by :math:`N_{vpe}` nodes/vertices.

        For an :math:`K`-dimensional element defined by :math:`N_{vpe}` nodes, points inside the element are represented in a local coordinate system :math:`(\xi, \eta, \zeta, ...)` also named ``natural coordinates``.
        Shape functions are defined in this local coordinate system in order to interpolate values at any point within the element based on the values at the nodes.

        .. math::

            P(\xi, \eta, \zeta, ...) = \sum_{i=1}^{N_{vpe}} N_i(\xi, \eta, \zeta, ...) P_i

        where :math:`P` is the interpolated value at the point, :math:`N_i` are the shape functions, and :math:`P_i` are the nodal values.
        
        .. note::
        
            The function is a convenience wrapper around the :func:`pysdic.compute_shape_functions` function.
            
        .. warning::
        
            No test are performed to check if the provided integration points are consistent with the mesh (i.e., existing element indices).
            Only shape check is performed.
            The behavior of the function is undefined if the provided integration points are not consistent with the mesh.
    

        Parameters
        ----------
        integration_points: :class:`IntegrationPoints`
            An instance of the :class:`IntegrationPoints` class containing the natural coordinates of the points where the shape functions are to be evaluated.
            
        return_derivatives: :class:`bool`, optional
            If set to :obj:`True`, the method will also compute and return the derivatives of the shape functions with respect to the local coordinates. Default is :obj:`False`.
            
        precompute: :class:`bool`, optional
            If set to :obj:`True`, the computed shape functions (and their derivatives if :obj:`return_derivatives` is True) will be stored in the integration points object for future reuse, avoiding redundant computations if the shape functions are needed again at the same integration points under the keys ``shape_functions`` and ``shape_function_derivatives``. Default is :obj:`False`.

        default: Real, optional
            The default value to assign to shape functions for points outside the valid range. Default is :obj:`0.0`.


        Returns
        -------
        shape_functions: :class:`numpy.ndarray`
            An array of shape (:math:`M`, :math:`N_{vpe}`) containing the evaluated shape functions at the specified points for the :math:`N_{vpe}` nodes of the element.

        shape_function_derivatives: :class:`numpy.ndarray`, optional
            An array of shape (:math:`M`, :math:`N_{vpe}`, :math:`K`) containing the derivatives of the shape functions with respect to the local coordinates, if :obj:`return_derivatives` is True.


        See Also
        --------
        :func:`pysdic.compute_shape_functions`
            For more information on the algorithm used to compute the shape functions and their derivatives, and an example of how to use it.
            
        """
        if self.element_type is None:
            raise ValueError("Element type must be specified in the mesh connectivity to compute shape functions.")
        self._check_integration_points(integration_points)
        
        if not isinstance(precompute, bool):
            raise TypeError(f"precompute must be a boolean, got {type(precompute)}.")
        
        output = compute_shape_functions(
            natural_coordinates=integration_points.natural_coordinates,
            element_type=self.element_type,
            return_derivatives=return_derivatives,
            default=default
        )
        
        if precompute:
            if return_derivatives:
                shape_functions, shape_function_derivatives = output
                integration_points._set_precomputed("shape_functions", shape_functions)
                integration_points._set_precomputed("shape_function_derivatives", shape_function_derivatives)
            else:
                shape_functions = output
                integration_points._set_precomputed("shape_functions", shape_functions)
                
        return output


    def compute_property_interpolation(
        self, 
        property: Union[str, ArrayLike],
        integration_points: IntegrationPoints,
        *,
        skip_m1 : bool = True,
        default: Union[Real, ArrayLike] = numpy.nan,
    ) -> numpy.ndarray:
        r"""
        Compute the interpolation of a property at given natural coordinates for specified elements in the mesh (convenient method for :func:`pysdic.compute_property_interpolation`).
        The :obj:`element_type` must be specified in the mesh connectivity to compute the shape functions for the elements of the mesh, which are used to perform the interpolation of the property.

        In a space of dimension :math:`E`, we consider a :math:`K`-dimensional element (with :math:`K \leq E`) defined by :math:`N_{vpe}` nodes/vertices.

        For an :math:`K`-dimensional element defined by :math:`N_{vpe}` nodes, points inside the element are represented in a local coordinate system :math:`(\xi, \eta, \zeta, ...)` also named ``natural coordinates``.
        Shape functions are defined in this local coordinate system in order to interpolate values at any point within the element based on the values at the nodes.

        .. math::

            P(\xi, \eta, \zeta, ...) = \sum_{i=1}^{N_{vpe}} N_i(\xi, \eta, \zeta, ...) P_i
            
        where :math:`P` is the interpolated value at the point, :math:`N_i` are the shape functions, and :math:`P_i` are the nodal values.
        
        Method to compute the interpolation of a property at given natural coordinates for specified elements in the mesh.
        
        .. note::
        
            The function is a convenience wrapper around the :func:`pysdic.compute_property_interpolation` function.
            
        .. important::
        
            - If ``shape_functions`` are precomputed in :class:`IntegrationPoints`, they will be used, otherwise they will be computed on the fly using the :meth:`Mesh.compute_shape_functions` method.
            
        .. hint::
        
            To interpolate the position of the vertices use ``property=mesh.vertices.points``.
            
        .. warning::
        
            No test are performed to check if the provided integration points are consistent with the mesh (i.e., existing element indices).
            Only shape check is performed.
            The behavior of the function is undefined if the provided integration points are not consistent with the mesh.
            
        Parameters
        ----------
        property: Union[:class:`str`, ArrayLike]
            The property to interpolate.
            If a string is provided, it should correspond to a property name in the mesh (either a vertex property or an element property).
            If a numpy array is provided, it should have shape (:math:`N_v`, :math:`P`) for vertex properties or shape (:math:`N_e`, :math:`P`) for element properties, where :math:`N_v` is the number of vertices, :math:`N_e` is the number of elements, and :math:`P` is the number of property channels.
    
        integration_points: :class:`pysdic.objects.IntegrationPoints`
            The integration points at which to compute the property interpolation.
        
        skip_m1: :class:`bool`, optional
            If set to :obj:`True`, any element index of -1 in :obj:`element_indices` will result in the corresponding interpolated property being set to :obj:`default`.
            Default is :obj:`True`.
            
        default: Union[Real, ArrayLike], optional
            The default value to assign to interpolated properties for integration points associated with an element index of -1 when :obj:`skip_m1` is :obj:`True`.
            Default is :obj:`numpy.nan`. The input can also be a :class:`numpy.ndarray` of shape (P,) to assign different default values for each property component.
            
            
        Returns
        -------
        interpolated_properties: :class:`numpy.ndarray`
            An array of shape :math:`(N_{p}, P)` containing the interpolated property values at each of the :math:`N_{p}` integration points.
            
    
        See Also
        --------
        :func:`pysdic.compute_shape_functions`
            For more information on shape functions and their role in property interpolation.
            
        :func:`pysdic.compute_property_interpolation`
            For more information on the algorithm used to compute the property interpolation and an example of how to use it.

        """
        self._check_integration_points(integration_points)
        if self.element_type is None:
            raise ValueError("Element type must be specified in the mesh connectivity to compute property interpolation.")
        
        # Case where the property is given as a string corresponding to a property name in the vertices
        if isinstance(property, str):
            if property in self.vertices:
                property = self.vertices.get_property(property)
            else:
                raise ValueError(f"The property '{property}' is not found in the properties of the vertices. Please provide a valid property name or an array of shape (N_v, P) containing the property values defined at the vertices.")
        property_array = numpy.asarray(property, dtype=numpy.float64)
            
        # Get the shape functions for the integration points, either from precomputed values or by computing them on the fly
        if integration_points._has_precomputed("shape_functions"):
            shape_functions = integration_points.get_precomputed("shape_functions")
        else:
            shape_functions = self.compute_shape_functions(integration_points, return_derivatives=False, precompute=False)
        
        return assemble_property_interpolation(
            property_array=property_array,
            connectivity=self.connectivity.elements,
            shape_functions=shape_functions,
            element_indices=integration_points.element_indices,
            skip_m1=skip_m1,
            default=default
        )
    
    
    def compute_property_projection(
        self, 
        property_array: Union[str, ArrayLike],
        integration_points: IntegrationPoints,
        *,
        sparse: bool = False,
        skip_m1 : bool = True,
    ) -> numpy.ndarray:
        r"""
        Project a property defined at integration points within elements back to the vertices of a mesh using mesh and integration point information (convenient method for :func:`pysdic.compute_property_projection`).
        The :obj:`element_type` must be specified in the mesh connectivity to compute the shape functions for the elements of the mesh, which are used to perform the projection of the property.
        
        .. note::
        
            The function is a convenience wrapper around the :func:`pysdic.compute_property_projection` function.
            
        .. important::
        
            - If ``shape_functions`` are precomputed in :class:`IntegrationPoints`, they will be used, otherwise they will be computed on the fly using the :meth:`Mesh.compute_shape_functions` method.
        
        .. warning::
        
            No test are performed to check if the provided integration points are consistent with the mesh (i.e., existing element indices).
            Only shape check is performed.
            The behavior of the function is undefined if the provided integration points are not consistent with the mesh.
        
        Parameters
        ----------
        property_array: Union[:class:`str`, ArrayLike]
            An array of shape :math:`(N_{p}, P)` containing the property values defined at the :math:`N_{p}` integration points. If 1D-array is provided, it will be treated as a single-component property of shape :math:`(N_{p}, 1)`. If a string, the property will be retrieved from the properties of the integration points using the string as a key.
        
        integration_points: :class:`pysdic.objects.IntegrationPoints`
            The integration points at which the property values are defined and from which the projection to the vertices is to be computed.
            
        sparse: bool, optional
            If set to :obj:`True`, the shape functions matrix will be constructed as a sparse matrix to optimize memory usage for large meshes.
            Default is :obj:`False`.
            
        skip_m1: bool, optional
            If set to :obj:`True`, any element index of -1 in :obj:`element_indices` will result in the corresponding contribution to the vertex property being skipped (i.e., not added to the vertex property values).
            Default is :obj:`True`.
            
        
        Returns
        -------
        projected_properties: :class:`numpy.ndarray`
            An array of shape :math:`(N_{v}, P)` containing the projected property values at the nodes of the mesh.
            
        
        See Also
        --------
        :func:`pysdic.compute_shape_functions`
            For more information on shape functions and their role in property projection.
            
        :func:`pysdic.compute_property_projection`
            For more information on the algorithm used to compute the property projection and an example of how to use it.
            
        """
        self._check_integration_points(integration_points)
        if self.element_type is None:
            raise ValueError("Element type must be specified in the mesh connectivity to compute property projection.")
        
        # Case where the property is given as a string corresponding to a property name in the integration points
        if isinstance(property, str):
            if property in integration_points:
                property = integration_points.get_property(property)
            else:
                raise ValueError(f"The property '{property}' is not found in the properties of the integration points. Please provide a valid property name or an array of shape (N_p, P) containing the property values defined at the integration points.")
        property_array = numpy.asarray(property, dtype=numpy.float64)
            
        # Get the shape functions for the integration points, either from precomputed values or by computing them on the fly
        if integration_points._has_precomputed("shape_functions"):
            shape_functions = integration_points.get_precomputed("shape_functions")
        else:
            shape_functions = self.compute_shape_functions(integration_points, return_derivatives=False, precompute=False)
            
        shape_function_matrix = assemble_shape_function_matrix(
            shape_functions=shape_functions,
            connectivity=self.connectivity.elements,
            element_indices=integration_points.element_indices,
            n_vertices=self.n_vertices,
            sparse=sparse,
            skip_m1=skip_m1,
            default=0.0
        )
            
        return assemble_property_projection(
            property_array=property_array,
            shape_function_matrix=shape_function_matrix,
            point_weights=integration_points.weights,
            return_unaffected=False
        )
            
            
    def compute_property_derivative(
        self, 
        property_array: numpy.ndarray,
        integration_points: IntegrationPoints,
        *,
        skip_m1 : bool = True,
        default: Union[Real, numpy.ndarray] = numpy.nan,
    ) -> numpy.ndarray:
        r"""
        Compute the derivative of a property defined at the nodes of a mesh to given integration points within elements with respect to global coordinates :math:`(x,y,z,...)` from mesh and integration point information (convenient method for :func:`pysdic.compute_property_derivative`).
        The :obj:`element_type` must be specified in the mesh connectivity to compute the shape functions for the elements of the mesh, which are used to perform the computation of the property derivatives.
        
        .. note::
        
            The function is a convenience wrapper around the :func:`pysdic.compute_property_derivative` function.
            
        .. important::
        
            - If ``shape_function_derivatives`` are precomputed in :class:`IntegrationPoints`, they will be used, otherwise they will be computed on the fly using the :meth:`Mesh.compute_shape_functions` method with :obj:`return_derivatives` set to :obj:`True`.
            
        .. warning::
        
            No test are performed to check if the provided integration points are consistent with the mesh (i.e., existing element indices).
            Only shape check is performed.
            The behavior of the function is undefined if the provided integration points are not consistent with the mesh.
            
        
        Parameters
        ----------
        property_array: :class:`numpy.ndarray`
            An array of shape :math:`(N_{v}, P)` containing the property values defined at the :math:`N_{v}` vertices of the mesh. If 1D-array is provided, it will be treated as a single-component property of shape :math:`(N_{v}, 1)`.
            
        integration_points: :class:`pysdic.objects.IntegrationPoints`
            The integration points at which the property derivatives are to be computed.
            
        skip_m1: bool, optional
            If set to :obj:`True`, any element index of -1 in :obj:`element_indices` will result in the corresponding contribution to the property derivatives being skipped (i.e., not added to the property derivative values). Default is :obj:`True`.
            
        default: Union[:class:`Real`, :class:`numpy.ndarray`], optional
            The default value to assign to property derivatives for integration points associated with an element index of -1 when :obj:`skip_m1` is :obj:`True`. Default is :obj:`numpy.nan`. The input can also be a :class:`numpy.ndarray` of shape (P,) to assign different default values for each property component.
            
        
        Returns
        -------
        property_derivatives: :class:`numpy.ndarray`
            An array of shape :math:`(N_{p}, P, E)` containing the computed derivatives of the property with respect to global coordinates at each of the :math:`N_{p}` integration points, where :math:`E` is the dimension of the global coordinate system.
        
        
        See Also
        --------
        :func:`pysdic.compute_shape_functions`
            For more information on shape functions and their role in computing property derivatives.
            
        :func:`pysdic.compute_property_derivative`
            For more information on the algorithm used to compute the property derivatives and an example of how to use it.
            
        """
        self._check_integration_points(integration_points)
        if self.element_type is None:
            raise ValueError("Element type must be specified in the mesh connectivity to compute property derivatives.")
        
        # Case where the property is given as a string corresponding to a property name in the vertices
        if isinstance(property, str):
            if property in self.vertices:
                property = self.vertices.get_property(property)
            else:
                raise ValueError(f"The property '{property}' is not found in the properties of the vertices. Please provide a valid property name or an array of shape (N_v, P) containing the property values defined at the vertices.")
        property_array = numpy.asarray(property, dtype=numpy.float64)
            
        # Get the shape functions for the integration points, either from precomputed values or by computing them on the fly
        if integration_points._has_precomputed("shape_function_derivatives"):
            shape_function_derivatives = integration_points.get_precomputed("shape_function_derivatives")
        else:
            _, shape_function_derivatives = self.compute_shape_functions(integration_points, return_derivatives=True, precompute=False)
            
        # Compute the derivatives of the property at the integration points using the shape function derivatives
        remap_coordinates = remap_vertices_coordinates(
            vertices_coordinates=self.vertices.points,
            connectivity=self.connectivity.elements,
            element_indices=integration_points.element_indices,
        )
        
        jacobian_matrix = assemble_jacobian_matrix(
            shape_function_derivatives=shape_function_derivatives,
            remapped_coordinates=remap_coordinates
        )
        
        return assemble_property_derivative(
            property_array=property_array,
            connectivity=self.connectivity.elements,
            element_indices=integration_points.element_indices,
            shape_function_derivatives=shape_function_derivatives,
            jacobian_matrix=jacobian_matrix,
            skip_m1=skip_m1,
            default=default
        )
        
        
    # ===================================
    # New Properties for SurfaceMesh
    # ===================================
    @property
    def uvmap(self) -> Optional[numpy.ndarray]:
        r"""
        [Get or Set] The UV mapping of each element in the mesh (only for surfacique meshes :math:`K=2`).

        The UV mapping is stored as a numpy ndarray of shape (:math:`N_e`, 2 * :math:`N_{vpe}`), where :math:`N_e` is the number of elements and :math:`N_{vpe}` is the number of vertices per element.

        The values correspond to the UV coordinates of the :math:`N_{vpe}` vertices of the element: :math:`(u_1, v_1, u_2, v_2, u_3, v_3, ..., u_{N_{vpe}}, v_{N_{vpe}})`.

        .. note::

            The UV coordinates are stored as an element property of the mesh under the key "uvmap".

        Parameters
        ----------
        value: ArrayLike, optional
            A numpy ndarray of shape (:math:`N_e`, 2 * :math:`N_{vpe}`) to set as the UV mapping.

        Returns
        -------
        :class:`numpy.ndarray` or None
            An array of shape (:math:`N_e`, 2 * :math:`N_{vpe}`) where :math:`N_e` is the number of elements. Or None if not set.
            
        """
        expected_K = self._get_expected_K()
        if expected_K is not None and expected_K != 2:
            raise AttributeError("uvmap property is only available for surfacique meshes (K=2).")
        if not 'uvmap' in self._connectivity:
            return None
        uvmap = self._connectivity['uvmap']
        if not isinstance(uvmap, numpy.ndarray):
            raise TypeError(f"uvmap property must be a numpy array, got {type(uvmap)}.")
        if not uvmap.ndim == 2:
            raise ValueError(f"uvmap property must be a 2D array of shape (N_e, 2 * N_vpe), where N_e is the number of elements and N_vpe is the number of vertices per element. Got shape {uvmap.shape}.")
        N_e = self.n_elements
        N_vpe = self.n_vertices_per_element
        if uvmap.shape != (N_e, 2 * N_vpe):
            raise ValueError(f"uvmap property must have shape (N_e, 2 * N_vpe), where N_e is the number of elements ({N_e}) and N_vpe is the number of vertices per element ({N_vpe}). Got shape {uvmap.shape}.")
        return uvmap

    @uvmap.setter
    def uvmap(self, value: Optional[ArrayLike]) -> None:
        expected_K = self._get_expected_K()
        if expected_K is not None and expected_K != 2:
            raise AttributeError("uvmap property is only available for surfacique meshes (K=2).")
        if value is None and 'uvmap' in self._connectivity:
            self._connectivity.delete_property('uvmap')
        if value is not None:
            value = numpy.asarray(value, dtype=numpy.float64)
            if not value.ndim == 2:
                raise ValueError(f"uvmap must be a 2D array of shape (N_e, 2 * N_vpe), where N_e is the number of elements and N_vpe is the number of vertices per element. Got shape {value.shape}.")
            N_e = self.n_elements
            N_vpe = self.n_vertices_per_element
            if value.shape != (N_e, 2 * N_vpe):
                raise ValueError(f"uvmap must have shape (N_e, 2 * N_vpe), where N_e is the number of elements ({N_e}) and N_vpe is the number of vertices per element ({N_vpe}). Got shape {value.shape}.")
            self._connectivity['uvmap'] = value 


    # =======================
    # Visualization Methods
    # =======================
    def visualize(
        self, 
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
        
        additional_points: Optional[Union[ArrayLike, PointCloud, IntegrationPoints]] = None,
        additional_points_color: Optional[Union[str, Tuple[Real, Real, Real]]] = None,
        additional_points_size: Optional[Real] = None,
        additional_points_opacity: Optional[Real] = None,
        
        bar_title: Optional[str] = None,
        bar_width: Optional[Real] = None,
        bar_height: Optional[Real] = None,
        bar_position_x: Optional[Real] = None,
        bar_position_y: Optional[Real] = None,
        bar_title_font_family: Optional[str] = None,
        bar_title_font_size: Optional[Integral] = None,
        bar_label_font_size: Optional[Integral] = None,
        bar_title_color: Optional[Union[str, Tuple[Real, Real, Real]]] = None,
        bar_title_bold: bool = False,
        bar_vertical: bool = True,
        bar_interactive: bool = False,
        show_bar: bool = True,

        title: Optional[str] = None,
        title_color: Optional[Union[str, Tuple[Real, Real, Real]]] = None,
        title_font_family: Optional[str] = None,
        title_font_size: Optional[Integral] = None,
        show_title: bool = True,
        
        bounds: Optional[Tuple[Real, Real, Real, Real, Real, Real]] = None,
        bounds_grid: Optional[str] = None,
        bounds_ticks: Optional[str] = None,
        bounds_location: Optional[str] = None,
        bounds_color: Optional[Union[str, Tuple[Real, Real, Real]]] = None,
        bounds_font_size: Optional[Integral] = None,
        bounds_font_family: Optional[str] = None,
        bounds_bold: bool = False,
        bounds_fmt: Optional[str] = None,
        bounds_title_x: str = "X Axis",
        bounds_title_y: str = "Y Axis",
        bounds_title_z: str = "Z Axis",
        bounds_n_xlabels: Integral = 5,
        bounds_n_ylabels: Integral = 5,
        bounds_n_zlabels: Integral = 5,
        bounds_show_xaxis: bool = True,
        bounds_show_yaxis: bool = True,
        bounds_show_zaxis: bool = True,
        bounds_show_xlabels: bool = True,
        bounds_show_ylabels: bool = True,
        bounds_show_zlabels: bool = True,
        show_bounds: bool = True,
        
        camera_position: Optional[ArrayLike] = None,
        camera_focal_point: Optional[ArrayLike] = None,
        camera_view_up: Optional[ArrayLike] = None,
        
        screenshot_path: Optional[str] = None,
    ) -> None:
        r"""
        Visualize the mesh using PyVista (only for :math:`E \leq 3` embbeded dimension), the mesh must have :obj:`element_type` set.
        
        .. note::
        
            For 1D or 2D vertices clouds, the vertices are visualized in a 3D space by adding zero coordinates for the missing dimensions.

        .. seealso::

            - `PyVista Documentation <https://docs.pyvista.org>`_ for more details on visualization options.


        Parameters
        ----------
        vertex_color: Union[:class:`str`, Tuple[Real, Real, Real]], optional
            The color of the vertices in the visualization. Can be a string representing a color name (e.g., :obj:`"red"`), or an RGB tuple or hex string.
            
        vertex_size: Real, optional
            The size of the vertices in the visualization. Must be a positive value if provided. Default is set by PyVista to :obj:`5.0`.
            
        vertex_opacity: Real, optional
            The opacity of the vertices in the visualization. Must be between :obj:`0.0` (fully transparent) and :obj:`1.0` (fully opaque) if provided. Default is set by PyVista to :obj:`1.0`.
            
        vertex_mask: ArrayLike, optional
            A boolean array of shape :math:`(N_v,)` where :math:`N_v` is the number of vertices in the mesh, indicating which vertices to show (True) or hide (False) in the visualization.
        
        show_vertices: :class:`bool`, optional
            Whether to show the vertices in the visualization. Default is :obj:`True`.
            
        edge_color: Union[:class:`str`, Tuple[Real, Real, Real]], optional
            The color of the edges in the visualization. Can be a string representing a color name (e.g., :obj:`"black"`), or an RGB tuple or hex string.
            
        edge_width: Real, optional
            The width of the edges in the visualization. Must be a positive value if provided.
            
        edge_opacity: Real, optional
            The opacity of the edges in the visualization. Must be between :obj:`0.0` (fully transparent) and :obj:`1.0` (fully opaque) if provided.
            
        show_edges: :class:`bool`, optional
            Whether to show the edges in the visualization. Default is :obj:`True`.
            
        face_color: Union[:class:`str`, Tuple[Real, Real, Real]], optional
            The color of the faces in the visualization. Can be a string representing a color name (e.g., :obj:`"lightgray"`), or an RGB tuple or hex string.
            This parameter is ignored if :obj:`vertex_property` or :obj:`element_property` is provided.
        
        face_opacity: Real, optional
            The opacity of the faces in the visualization. Must be between :obj:`0.0` (fully transparent) and :obj:`1.0` (fully opaque) if provided.
        
        face_mask: ArrayLike, optional
            A boolean array of shape :math:`(N_e,)` where :math:`N_e` is the number of elements in the mesh, indicating which faces to show (True) or hide (False) in the visualization.
            
        show_faces: :class:`bool`, optional
            Whether to show the faces in the visualization. Default is :obj:`True`.
            
        vertex_property: Union[:class:`str`, ArrayLike], optional
            The property to use for coloring the mesh based on a vertex property. Can be a string representing the property key of a property stored in the point cloud, or a NumPy array of shape :math:`(N_p,)` or :math:`(N_p, A)` representing a property to use for coloring the vertices.
            Note that if a string key is provided, the corresponding property must exist in the point cloud.
        
        element_property: Union[:class:`str`, ArrayLike], optional
            The property to use for coloring the mesh based on an element property. Can be a string representing the property key of a property stored in the connectivity, or a NumPy array of shape :math:`(N_e,)` or :math:`(N_e, A)` representing a property to use for coloring the elements.
            Note that if a string key is provided, the corresponding property must exist in the connectivity.
            
        property_axis: Optional[Integral], optional
            The axis to use when :obj:`property` contains multiple attributes per vertex (shape :math:`(N_p, A)`). If :obj:`None`, the magnitude of the property vector is used for coloring.
            By default, :obj:`None`.
            
        property_clim: Optional[Tuple[Real, Real]], optional
            The color limits (min, max) for the property values when :obj:`property` is provided. If :obj:`None`, the color limits are determined automatically.
            Default is :obj:`None`.
            
        property_cmap: Union[:class:`str`, :class:`matplotlib.colors.Colormap`], optional
            The colormap to use for coloring the points based on the property values. Can be a string representing a Matplotlib colormap or a Matplotlib Colormap object.
            If :obj:`None`, the default colormap :obj:`"viridis"` is used by Matplotlib.
            
        property_below_color: Union[:class:`str`, Tuple[Real, Real, Real]], optional
            The color to use for points with property values below the minimum color limit when :obj:`property_clim` is provided.
            Can be a string representing a color name (e.g., :obj:`"blue"`), or an RGB tuple or hex string.
            If :obj:`None`, the default below color is set to the minimum color of the colormap.
            
        property_above_color: Union[:class:`str`, Tuple[Real, Real, Real]], optional
            The color to use for points with property values above the maximum color limit when :obj:`property_clim` is provided.
            Can be a string representing a color name (e.g., :obj:`"red"`), or an RGB tuple or hex string.
            If :obj:`None`, the default above color is set to the maximum color of the colormap.
            
        texture: Optional[ArrayLike], optional
            A numpy array representing a texture to apply to the mesh. The shape and format of the texture array should be compatible with PyVista's texture mapping requirements.
            The :obj:`uvmap` must be set for the mesh connectivity.
            
        texture_rgb: :class:`bool`, optional
            If :obj:`True`, the texture is interpreted as an RGB image. If :obj:`False`, the texture is interpreted as a single-channel scalar field. Default is :obj:`True`.
            
        additional_points: Optional[Union[ArrayLike, :class:`PointCloud`, :class:`IntegrationPoints`]], optional
            A numpy array of shape (:math:`N_{ap}`, 3) containing the coordinates of additional points to visualize alongside the mesh. The additional points are visualized as spheres in the visualization.
            If provides as a :class:`PointCloud` or :class:`IntegrationPoints`, the coordinates of the points will be extracted from the :obj:`points` attribute of the object or compute by interpolating the vertices coordinates at the integration points, respectively.
            
        additional_points_color: Union[:class:`str`, Tuple[Real, Real, Real]], optional
            The color of the additional points in the visualization. Can be a string representing a color name (e.g., :obj:`"green"`), or an RGB tuple or hex string.
            
        additional_points_size: Optional[Real], optional
            The size of the additional points in the visualization. Must be a positive value if provided.
            
        additional_points_opacity: Optional[Real], optional
            The opacity of the additional points in the visualization. Must be between :obj:`0.0` (fully transparent) and :obj:`1.0` (fully opaque) if provided.
            
        bar_title: Optional[:class:`str`], optional
            The title of the color bar when :obj:`property` is provided. If :obj:`None`, no title is displayed.
            
        bar_width: Optional[Real], optional
            The width of the color bar as a fraction of the window width. Must be between :obj:`0.0` and :obj:`1.0` if provided.
            
        bar_height: Optional[Real], optional
            The height of the color bar as a fraction of the window height. Must be between :obj:`0.0` and :obj:`1.0` if provided.
            
        bar_position_x: Optional[Real], optional
            The x position of the color bar as a fraction of the window width. Must be between :obj:`0.0` and :obj:`1.0` if provided.
            
        bar_position_y: Optional[Real], optional
            The y position of the color bar as a fraction of the window height. Must be between :obj:`0.0` and :obj:`1.0` if provided.
            
        bar_title_font_family: Optional[:class:`str`], optional
            The font type of the color bar title between :obj:`"arial"`, :obj:`"courier"`, and :obj:`"times"`.
            
        bar_title_font_size: Optional[Integral], optional
            The font size of the color bar title.
            
        bar_label_font_size: Optional[Integral], optional
            The font size of the color bar labels.
            
        bar_title_color: Union[:class:`str`, Tuple[Real, Real, Real]], optional
            The color of the color bar title text. Can be a string representing a color name (e.g., :obj:`"black"`), or an RGB tuple or hex string.
            
        bar_title_bold: :class:`bool`, optional
            Whether to display the color bar title in bold font. Default is :obj:`False`.
            
        bar_vertical: :class:`bool`, optional
            Whether to display the color bar vertically. If :obj:`False`, the color bar is displayed horizontally. Default is :obj:`True`.
            
        bar_interactive: :class:`bool`, optional
            If :obj:`True`, enables interactive mode for the color bar in PyVista. Default is :obj:`False`. 
            
        show_bar: :class:`bool`, optional
            Whether to show the color bar in the visualization. Default is :obj:`True`.           
                    
        title: Optional[:class:`str`], optional
            The title of the visualization window.
            
        title_color: Union[:class:`str`, Tuple[Real, Real, Real]], optional
            The color of the title text. Can be a string representing a color name (e.g., :obj:`"blue"`), or an RGB tuple or hex string.

        title_font_family: Optional[:class:`str`], optional
            The font type of the title between :obj:`"arial"`, :obj:`"courier"`, and :obj:`"times"`.
            
        title_font_size: Optional[Integral], optional
            The font size of the title.
            
        show_title: :class:`bool`, optional
            Whether to show the title in the visualization. Default is :obj:`True`.
            
        bounds: Optional[Tuple[Real, Real, Real, Real, Real, Real]], optional
            The bounds of the axes in the format (:obj:`min_x`, :obj:`max_x`, :obj:`min_y`, :obj:`max_y`, :obj:`min_z`, :obj:`max_z`). If :obj:`None`, the bounds are determined automatically.
            
        bounds_grid: Optional[:class:`str`], optional
            Add grid lines to the backface (True, 'back', or 'backface') or to the frontface ('front', 'frontface') of the axes actor.
            
        bounds_ticks: Optional[:class:`str`], optional
            Set how the ticks are drawn on the axes grid. Options include: 'inside', 'outside', 'both'.
            
        bounds_location: Optional[:class:`str`], optional
            Set how the axes are drawn: either static ('all'), closest triad ('front', 'closest', 'default'), furthest triad ('back', 'furthest'), static closest to the origin ('origin'), or outer edges ('outer') in relation to the camera position.
            
        bounds_color: Union[:class:`str`, Tuple[Real, Real, Real]], optional
            The color of the axes bounds. Can be a string representing a color name (e.g., :obj:`"black"`), or an RGB tuple or hex string.

        bounds_font_size: Optional[Integral], optional
            The font size of the axes bounds.
            
        bounds_font_family: Optional[:class:`str`], optional
            The font type of the axes bounds between :obj:`"arial"`, :obj:`"courier"`, and :obj:`"times"`.
            
        bounds_bold: :class:`bool`, optional
            Whether to display the axes bounds in bold font. Default is :obj:`False`.
            
        bounds_fmt: Optional[:class:`str`], optional
            The format string for the axes bounds labels (e.g., :obj:`"{:.2f}"` for 2 decimal places).
            
        bounds_title_x: :class:`str`, optional
            The title for the x-axis bounds. Default is :obj:`"X Axis"`.
            
        bounds_title_y: :class:`str`, optional
            The title for the y-axis bounds. Default is :obj:`"Y Axis"`.
            
        bounds_title_z: :class:`str`, optional
            The title for the z-axis bounds. Default is :obj:`"Z Axis"`.
            
        bounds_n_xlabels: Integral, optional
            The number of labels to display on the x-axis bounds. Default is :obj:`5`.
            
        bounds_n_ylabels: Integral, optional
            The number of labels to display on the y-axis bounds. Default is :obj:`5`.
            
        bounds_n_zlabels: Integral, optional
            The number of labels to display on the z-axis bounds. Default is :obj:`5`.
            
        bounds_show_xaxis: :class:`bool`, optional
            Whether to show the x-axis in the bounds. Default is :obj:`True`.
            
        bounds_show_yaxis: :class:`bool`, optional
            Whether to show the y-axis in the bounds. Default is :obj:`True`.
            
        bounds_show_zaxis: :class:`bool`, optional
            Whether to show the z-axis in the bounds. Default is :obj:`True`.
            
        bounds_show_xlabels: :class:`bool`, optional
            Whether to show the x-axis labels in the bounds. Default is :obj:`True`.
            
        bounds_show_ylabels: :class:`bool`, optional
            Whether to show the y-axis labels in the bounds. Default is :obj:`True`.
            
        bounds_show_zlabels: :class:`bool`, optional
            Whether to show the z-axis labels in the bounds. Default is :obj:`True`.
            
        show_bounds: :class:`bool`, optional
            Whether to show the axes bounds in the visualization. Default is :obj:`True`.    
        
        camera_position: Optional[ArrayLike], optional
            The position of the camera in the 3D space as a NumPy array of shape :math:`(3,)`.
            
        camera_focal_point: Optional[ArrayLike], optional
            The focal point of the camera in the 3D space as a NumPy array of shape :math:`(3,)`.
            
        camera_view_up: Optional[ArrayLike], optional
            The view up vector of the camera as a NumPy array of shape :math:`(3,)`.
            
        screenshot_path: Optional[:class:`str`], optional
            The file path to save a screenshot of the visualization. If :obj:`None`, no screenshot is saved.
            

        Examples
        --------
        Create a simple :class:`Mesh` instance and visualize it.
        
        .. code-block:: python
        
            import numpy as np
            from pysdic import Mesh
            
            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = Mesh(points, connectivity, element_type="triangle_3")
            
            mesh3d.visualize(
                vertex_color="red",
                edge_color="black",
                face_color="lightgray",
                vertex_size=10.0,
                edge_width=2.0,
                face_opacity=0.5,
                show_vertices=True,
                show_edges=True,
                show_faces=True,
                title="3D Mesh Visualization",
                title_color="blue",
                title_font_size=16,
            )
        
        
        """
        # Check empty vertices cloud
        if self.n_vertices == 0 or self.n_elements == 0:
            raise ValueError("Cannot visualize an empty mesh.")
        
        if self.element_type is None:
            raise ValueError("Visualization is not supported for meshes without defined element type. Please define an `element_type` for the mesh.")
        expected_K = self._get_expected_K()
        if expected_K is not None and expected_K != 2:
            raise NotImplementedError("Visualization is only supported for surfacique meshes (K=2).")
        vtk_cell_type = self._get_vtk_cell_type()
        if vtk_cell_type is None:
            raise NotImplementedError(f"Visualization is not supported as vtk_cell_type is not defined for mesh elements.")

        # Convert the mesh to 3D if necessary
        if self.n_dimensions < 1 or self.n_dimensions > 3:
            raise ValueError("Visualization is only supported for 1D, 2D, and 3D meshes.")
        
        if self.n_dimensions == 3:
            coordinates = self.points
        elif self.n_dimensions == 2:
            z_coords = numpy.zeros((self.n_vertices, 1), dtype=self.points.dtype)
            coordinates = numpy.hstack((self.points, z_coords))
        elif self.n_dimensions == 1:
            y_coords = numpy.zeros((self.n_vertices, 1), dtype=self.points.dtype)
            z_coords = numpy.zeros((self.n_vertices, 1), dtype=self.points.dtype)
            coordinates = numpy.hstack((self.points, y_coords, z_coords))
        else:
            raise ValueError("Visualization is only supported for 1D, 2D, and 3D meshes.")
        
        # Filter with point mask if provided
        if vertex_mask is not None:
            vertex_mask = numpy.asarray(vertex_mask, dtype=bool)
            if vertex_mask.ndim != 1:
                raise ValueError("vertex_mask must be a 1D boolean array.")
            if vertex_mask.shape[0] != self.n_vertices:
                raise ValueError("vertex_mask length must match the number of vertices in the mesh.")
        else:
            vertex_mask = numpy.ones(self.n_vertices, dtype=bool)
            
        # Filter with face mask if provided
        if face_mask is not None:
            face_mask = numpy.asarray(face_mask, dtype=bool)
            if face_mask.ndim != 1:
                raise ValueError("face_mask must be a 1D boolean array.")
            if face_mask.shape[0] != self.n_elements:
                raise ValueError("face_mask length must match the number of elements in the mesh.")
        else:
            face_mask = numpy.ones(self.n_elements, dtype=bool)
            
        # Keep only finite points
        finite_mask = self.vertices.is_finite()
        vertex_mask = numpy.logical_and(vertex_mask, finite_mask)

        if numpy.sum(vertex_mask) == 0:
            raise ValueError("No finite vertex coordinates available for visualization.")
        
        masked_coordinates = coordinates[vertex_mask]
        
        if numpy.sum(face_mask) == 0:
            raise ValueError("No faces available for visualization.")
        
        # Extract additional points if provided
        if additional_points is not None:
            if isinstance(additional_points, PointCloud):
                if additional_points.n_dimensions > 3:
                    raise ValueError("additional_points with more than 3 dimensions cannot be visualized.")
                additional_points = additional_points.points
            elif isinstance(additional_points, IntegrationPoints):
                self._check_integration_points(additional_points)
                additional_points = self.compute_property_interpolation(
                    property=self.vertices.points,
                    integration_points=additional_points,
                )
            additional_points = numpy.asarray(additional_points, dtype=numpy.float64)
            if additional_points.ndim != 2:
                raise ValueError("additional_points must be a 2D array of shape (N_ap, 3).")
            if additional_points.shape[0] == 0:
                raise ValueError("additional_points cannot be an empty array.")
            if additional_points.shape[1] == 3:
                additional_points_coordinates = additional_points
            elif additional_points.shape[1] == 2:
                z_coords = numpy.zeros((additional_points.shape[0], 1), dtype=additional_points.dtype)
                additional_points_coordinates = numpy.hstack((additional_points, z_coords))
            elif additional_points.shape[1] == 1:
                y_coords = numpy.zeros((additional_points.shape[0], 1), dtype=additional_points.dtype)
                z_coords = numpy.zeros((additional_points.shape[0], 1), dtype=additional_points.dtype)
                additional_points_coordinates = numpy.hstack((additional_points, y_coords, z_coords))
            else:
                raise ValueError("additional_points must have shape (N_ap, 3), (N_ap, 2), or (N_ap, 1).")
        else:
            additional_points_coordinates = None
        
        # Extract property array if provided
        _is_display_property = vertex_property is not None or element_property is not None
        _to_enough_properties =  vertex_property is not None and element_property is not None
        
        if _to_enough_properties:
            raise ValueError("Cannot provide both vertex_property and element_property for visualization. Please provide only one of them.")
        
        if _is_display_property and vertex_property is not None:
            if isinstance(vertex_property, str):
                if vertex_property not in self.vertices._properties:
                    raise ValueError(f"Property '{vertex_property}' not found in the point cloud.")
                property_values = self.vertices._properties[vertex_property]
            else:
                property_values = numpy.asarray(vertex_property)
                if property_values.ndim == 1:
                    if property_values.shape[0] != self.n_points:
                        raise ValueError(f"Property array must have shape ({self.n_points},) or ({self.n_points}, A).")
                elif property_values.ndim == 2:
                    if property_values.shape[0] != self.n_points:
                        raise ValueError(f"Property array must have shape ({self.n_points},) or ({self.n_points}, A).")
                else:
                    raise ValueError("Property array must be 1D or 2D.")

            # Reduce property to single attribute if necessary
            if property_values.ndim == 2:
                if property_axis is None:
                    property_values = numpy.linalg.norm(property_values, axis=1)
                else:
                    if property_axis < 0 or property_axis >= property_values.shape[1]:
                        raise ValueError(f"property_axis must be between 0 and {property_values.shape[1]-1}.")
                    property_values = property_values[:, property_axis]
                    
        elif _is_display_property and element_property is not None:
            if isinstance(element_property, str):
                if element_property not in self.connectivity._properties:
                    raise ValueError(f"Property '{element_property}' not found in the connectivity.")
                property_values = self.connectivity._properties[element_property]
            else:
                property_values = numpy.asarray(element_property)
                if property_values.ndim == 1:
                    if property_values.shape[0] != self.n_elements:
                        raise ValueError(f"Property array must have shape ({self.n_elements},) or ({self.n_elements}, A).")
                elif property_values.ndim == 2:
                    if property_values.shape[0] != self.n_elements:
                        raise ValueError(f"Property array must have shape ({self.n_elements},) or ({self.n_elements}, A).")
                else:
                    raise ValueError("Property array must be 1D or 2D.")

            # Reduce property to single attribute if necessary
            if property_values.ndim == 2:
                if property_axis is None:
                    property_values = numpy.linalg.norm(property_values, axis=1)
                else:
                    if property_axis < 0 or property_axis >= property_values.shape[1]:
                        raise ValueError(f"property_axis must be between 0 and {property_values.shape[1]-1}.")
                    property_values = property_values[:, property_axis]
                    
        else:
            property_values = None
                    
        # Create the displayed mesh (extension of the self mesh with duplicated faces for texturing)
        fictive_vertices = numpy.zeros((self.n_elements * self.n_vertices_per_element, 3), dtype=numpy.float64)
        for i in range(self.n_vertices_per_element):
            fictive_vertices[i::self.n_vertices_per_element, :] = self.vertices.points[self.elements[:, i], :]

        fictive_connectivity = numpy.arange(self.n_elements * self.n_vertices_per_element, dtype=numpy.int64).reshape(self.n_elements, self.n_vertices_per_element)
        fictive_n_cells = fictive_connectivity.shape[0]
        fictive_cells = numpy.hstack([numpy.full((fictive_n_cells, 1), self.n_vertices_per_element), fictive_connectivity]).astype(numpy.int64).ravel()
        fictive_celltypes = numpy.full(fictive_n_cells, vtk_cell_type, dtype=numpy.uint8)
        fictive_mesh = pyvista.UnstructuredGrid(fictive_cells, fictive_celltypes, fictive_vertices)
        
        if _is_display_property and vertex_property is not None:
            fictive_property_values = numpy.zeros((self.n_elements * self.n_vertices_per_element,), dtype=property_values.dtype)
            for i in range(self.n_vertices_per_element):
                fictive_property_values[i::self.n_vertices_per_element] = property_values[self.elements[:, i]]
        elif _is_display_property and element_property is not None:
            fictive_property_values = numpy.repeat(property_values, self.n_vertices_per_element)
        else:
            fictive_property_values = None
            
        # Propagate uvmap if texturing  #     # UV coordinates per vertex of each element
        uvmap = self.uvmap  # shape (N_e, 2 * N_vpe) or None
        
        if texture is not None:
            if not isinstance(texture, numpy.ndarray):
                raise ValueError("Texture must be a numpy array.")
            if uvmap is None:
                raise ValueError("UV mapping (uvmap) must be set for the mesh connectivity when a texture is provided.")
            
            fictive_mesh.active_texture_coordinates = numpy.zeros((self.n_elements * self.n_vertices_per_element, 2), dtype=numpy.float64)
            for i in range(self.n_vertices_per_element):
                fictive_mesh.active_texture_coordinates[i::self.n_vertices_per_element, 0] = uvmap[:, 2 * i]      # u_i
                fictive_mesh.active_texture_coordinates[i::self.n_vertices_per_element, 1] = uvmap[:, 2 * i + 1]  # v_i

            # Create a PyVista texture
            if texture.ndim == 2:
                color_texture = numpy.repeat(texture[:, :, numpy.newaxis], 3, axis=2).astype(numpy.uint8)
            elif texture.ndim == 3 and texture.shape[2] == 1:
                color_texture = numpy.repeat(texture, 3, axis=2).astype(numpy.uint8)
            elif texture.ndim == 3 and texture_rgb and texture.shape[2] == 3:
                color_texture = texture
            elif texture.ndim == 3 and not texture_rgb and texture.shape[2] == 3:
                gray_texture = numpy.round(numpy.dot(texture[..., :3], [0.2989, 0.5870, 0.1140])).astype(numpy.uint8)
                color_texture = numpy.repeat(gray_texture[:, :, numpy.newaxis], 3, axis=2).astype(numpy.uint8)
            else:
                raise ValueError("Invalid texture array shape.")
        
            pvtexture = pyvista.Texture(color_texture)
        
        else:
            pvtexture = None
            
        # Create a PyVista mesh
        plotter = pyvista.Plotter()
        
        plotter.add_mesh(
            fictive_mesh,
            color=face_color,
            opacity=face_opacity,
            scalars=fictive_property_values,
            texture=pvtexture,
            clim=property_clim,
            cmap=property_cmap,
            below_color=property_below_color,
            above_color=property_above_color,
            show_scalar_bar=False,  # Disable default scalar bar
            show_vertices=False,
            show_edges=False,
        )
        
        if show_bar and _is_display_property:
            plotter.add_scalar_bar(
                title=bar_title,
                width=bar_width,
                height=bar_height,
                position_x=bar_position_x,
                position_y=bar_position_y,
                font_family=bar_title_font_family,
                title_font_size=bar_title_font_size,
                label_font_size=bar_label_font_size,
                color=bar_title_color,
                bold=bar_title_bold,
                vertical=bar_vertical,
                interactive=bar_interactive,
            )
        
        if show_vertices:
            plotter.add_points(
                masked_coordinates,
                color=vertex_color,
                opacity=vertex_opacity,
                point_size=vertex_size,
                render_points_as_spheres=True,
                show_scalar_bar=False,
            )
            
        if additional_points_coordinates is not None:
            plotter.add_points(
                additional_points_coordinates,
                color=additional_points_color,
                opacity=additional_points_opacity,
                point_size=additional_points_size,
                render_points_as_spheres=True,
                show_scalar_bar=False,
            )
            
        if show_edges:
            plotter.add_mesh(
                fictive_mesh.extract_all_edges(),
                color=edge_color,
                opacity=edge_opacity,
                line_width=edge_width,
                show_vertices=False,
                show_scalar_bar=False,
            )
        
        if show_bounds:
            plotter.show_bounds(
                bounds=bounds,
                grid=bounds_grid,
                ticks=bounds_ticks,
                location=bounds_location,
                color=bounds_color,
                font_size=bounds_font_size,
                bold=bounds_bold,
                font_family=bounds_font_family,
                xtitle=bounds_title_x,
                ytitle=bounds_title_y,
                ztitle=bounds_title_z,
                n_xlabels=bounds_n_xlabels,
                n_ylabels=bounds_n_ylabels,
                n_zlabels=bounds_n_zlabels,
                fmt=bounds_fmt,
                show_xaxis=bounds_show_xaxis,
                show_yaxis=bounds_show_yaxis,
                show_zaxis=bounds_show_zaxis,
                show_xlabels=bounds_show_xlabels,
                show_ylabels=bounds_show_ylabels,
                show_zlabels=bounds_show_zlabels,
            )

        if show_title and title is not None:
            plotter.add_title(
                title, 
                font_size=title_font_size,
                color=title_color,
                font=title_font_family,
            )
            
        if camera_position is not None or camera_focal_point is not None or camera_view_up is not None:
            if camera_focal_point is None or camera_view_up is None or camera_position is None:
                raise ValueError("`camera_position`, `camera_focal_point`, and `camera_view_up` must all be provided together.")
            plotter.camera.position = camera_position
            plotter.camera.focal_point = camera_focal_point
            plotter.camera.up = camera_view_up
    
        plotter.show(screenshot=screenshot_path)
    
    
    