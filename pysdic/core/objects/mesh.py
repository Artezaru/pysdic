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

from typing import Optional, Tuple, Union, Dict, Callable
from numbers import Number, Integral

import numpy
import pyvista
import meshio
import os

import matplotlib.pyplot as plt

from .point_cloud import PointCloud

from ..shape_functions import (
    segment_2_shape_functions,
    segment_3_shape_functions,
    triangle_3_shape_functions,
    triangle_6_shape_functions,
    quadrangle_4_shape_functions,
    quadrangle_8_shape_functions,
)

from ..integration_points_operations import interpolate_property


class Mesh(ABC):
    r"""
    A Mesh is a collection of vertices (:class:`PointCloud`) and connectivity information that defines the elements of the mesh.
    
    This is an abstract base class for meshes.

    The vertices are represented as a PointCloud instance with shape (:math:`N_v`, :math`E`), where :math:`N_v` is the number of vertices (``n_vertices``) and :math:`E` is the embedding dimension (``n_dimensions``).
    The connectivity is represented as a numpy ndarray with shape (:math:`N_e`, :math:`N_{vpe}`), where :math:`N_e` is the number of elements (``n_elements``), and :math:`N_{vpe}` is the number of vertices per element (``n_vertices_per_element``).

    The coordinates of a point into the mesh can be accessed by the natural coordinates
    in the reference element. The number of natural coordinates :math:`(\xi, \eta, \zeta, ...)` depends on the type of element and
    is noted as :math:`K` (the topological dimension of the element) accessible through the property ``n_topological_dimensions``.

    Lets consider a mesh with :math:`N_{vpe}` vertices per element, and :math:`K` natural coordinates.
    Lets :math:`X` be the coordinates of a point in the mesh. The transformation from natural coordinates to global coordinates is given by:

    .. math::

        X = \sum_{i=1}^{N_{vpe}} N_i(\xi, \eta, \zeta, ...) X_i

    where :math:`N_i` are the shape functions associated with each vertex, and :math:`X_i` are the coordinates of the vertices of the element.

    .. seealso::

        - :doc:`./shape_functions` for more information on shape functions.

    If :obj:`elements_type` is provided, the mesh will enforce that all elements are of the specified type. 
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



    Parameters
    ----------
    vertices : :class:`PointCloud`
        The vertices of the mesh as a :class:`PointCloud` instance with shape (:math:`N_v`, :math:`E`), where :math:`N_v` is the number of vertices and :math:`E` is the embedding dimension.

    connectivity : :class:`numpy.ndarray`
        The connectivity of the mesh as a numpy ndarray with shape (:math:`N_e`, :math:`N_{vpe}`), where :math:`N_e` is the number of elements and :math:`N_{vpe}` is the number of vertices per element.

    vertices_properties : Optional[:class:`dict`], optional
        A dictionary to store properties of the vertices, each property should be a numpy ndarray of shape (:math:`N_v`, :math`A`) where :math:`N_v` is the number of vertices and :math:`A` is the number of attributes for that property, by default None.

    elements_properties : Optional[:class:`dict`], optional
        A dictionary to store properties of the elements, each property should be a numpy ndarray of shape (:math:`N_e`, :math`B`) where :math:`N_e` is the number of elements and :math:`B` is the number of attributes for that property, by default None.

    elements_type : Optional[:class:`str`], optional
        The expected type of elements in the mesh, by default None.

    internal_bypass : :class:`bool`, optional
        If :obj:`True`, internal checks are bypassed for better performance, by default :obj:`False`.
    
    """

    _mapping_elements_type_to_properties = {
        "segment_2": {
            "expected_N_vpe": 2,
            "expected_K": 1,
            "meshio_cell_type": "line",
            "vtk_cell_type": 3,
            "shape_functions_method": segment_2_shape_functions,
        },
        "segment_3": {
            "expected_N_vpe": 3,
            "expected_K": 1,
            "meshio_cell_type": "line3",
            "vtk_cell_type": 21,
            "shape_functions_method": segment_3_shape_functions,
        },
        "triangle_3": {
            "expected_N_vpe": 3,
            "expected_K": 2,
            "meshio_cell_type": "triangle",
            "vtk_cell_type": 5,
            "shape_functions_method": triangle_3_shape_functions,
        },
        "triangle_6": {
            "expected_N_vpe": 6,
            "expected_K": 2,
            "meshio_cell_type": "triangle6",
            "vtk_cell_type": 22,
            "shape_functions_method": triangle_6_shape_functions,
        },
        "quadrangle_4": {
            "expected_N_vpe": 4,
            "expected_K": 2,
            "meshio_cell_type": "quad",
            "vtk_cell_type": 9,
            "shape_functions_method": quadrangle_4_shape_functions,
        },
        "quadrangle_8": {
            "expected_N_vpe": 8,
            "expected_K": 2,
            "meshio_cell_type": "quad8",
            "vtk_cell_type": 23,
            "shape_functions_method": quadrangle_8_shape_functions,
        },
    }

    __slots__ = [
        '_internal_bypass',
        '_elements_type',
        '_vertices', 
        '_connectivity',
        '_vertices_properties',
        '_elements_properties',
        '_vertices_predefined_metadata',
        '_elements_predefined_metadata',
    ]

    def __init__(self, 
        vertices: PointCloud, 
        connectivity: numpy.ndarray, 
        vertices_properties: Optional[Dict] = None, 
        elements_properties: Optional[Dict] = None, 
        elements_type: Optional[str] = None,
        internal_bypass: bool = False
    ) -> None:
        # Define expected properties informations
        if not hasattr(self, "_vertices_predefined_metadata"):
            self._vertices_predefined_metadata = {}
        if not hasattr(self, "_elements_predefined_metadata"):
            self._elements_predefined_metadata = {}
        self._elements_type = None

        self._elements_predefined_metadata.update({
            "uvmap": {"check_method": self._internal_check_uvmap},
        })

        # Convert the inputs to the correct types
        if not isinstance(vertices, PointCloud):
            vertices = PointCloud(vertices)

        # Connectivity checks and conversion
        connectivity = numpy.asarray(connectivity, dtype=numpy.int64)
        if not connectivity.ndim == 2:
            raise ValueError(f"Connectivity must be a 2D array, got {connectivity.ndim}D array.")

        # Element type handling
        self.set_elements_type(elements_type)
        
        # Initialize attributes
        self._internal_bypass = True
        self._vertices = vertices
        self._connectivity = connectivity
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
        value : :class:`bool`
            If :obj:`True`, internal checks are bypassed. If :obj:`False`, internal checks are performed.

        Returns
        -------
        :class:`bool`
            :obj:`True` if internal checks are bypassed, :obj:`False` otherwise.

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

    def _get_expected_N_vpe(self) -> Optional[int]:
        r"""
        Internal method to get the expected number of vertices per element for the mesh.

        Returns
        -------
        Optional[:class:`int`]
            The expected number of vertices per element, or None if not set.
        """
        if self._elements_type is not None:
            return self._mapping_elements_type_to_properties[self._elements_type]["expected_N_vpe"]
        return None
        
    def _get_expected_K(self) -> Optional[int]:
        r"""
        Internal method to get the expected number of natural coordinates (topological dimensions) for the mesh.

        Returns
        -------
        Optional[:class:`int`]
            The expected number of natural coordinates (topological dimensions), or None if not set.
        """
        if self._elements_type is not None:
            return self._mapping_elements_type_to_properties[self._elements_type]["expected_K"]
        return None
        
    def _get_meshio_cell_type(self) -> Optional[str]:
        r"""
        Internal method to get the expected meshio cell type for the mesh.

        Returns
        -------
        Optional[:class:`str`]
            The expected meshio cell type, or None if not set.
        """
        if self._elements_type is not None:
            return self._mapping_elements_type_to_properties[self._elements_type]["meshio_cell_type"]
        return None
    
    def _get_vtk_cell_type(self) -> Optional[int]:
        r"""
        Internal method to get the expected VTK cell type for the mesh.

        Returns
        -------
        Optional[:class:`int`]
            The expected VTK cell type, or None if not set.
        """
        if self._elements_type is not None:
            return self._mapping_elements_type_to_properties[self._elements_type]["vtk_cell_type"]
        return None
    
    def _get_shape_functions_method(self) -> Optional[Callable]:
        r"""
        Internal method to get the shape functions method for the mesh.

        Returns
        -------
        Optional[:class:`Callable`]
            The shape functions method, or None if not set.
        """
        if self._elements_type is not None:
            return self._mapping_elements_type_to_properties[self._elements_type]["shape_functions_method"]
        return None

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
        if self.internal_bypass:
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
        if self.internal_bypass:
            return
        
        expected_N_vpe = self._get_expected_N_vpe()

        if not isinstance(self._connectivity, numpy.ndarray):
            raise TypeError(f"Connectivity must be a numpy ndarray, got {type(self._connectivity)}.")
        if self._connectivity.ndim != 2:
            raise ValueError(f"Connectivity must be a 2D array, got {self._connectivity.ndim}D array.")
        if expected_N_vpe is not None and not self._connectivity.shape[1] == expected_N_vpe:
            raise ValueError(f"Connectivity must have {expected_N_vpe} columns, got {self._connectivity.shape[1]}.")
        if numpy.any(self._connectivity < 0) or numpy.any(self._connectivity >= len(self._vertices)):
            raise ValueError("Connectivity contains invalid vertex indices.")
        if not self._connectivity.dtype == numpy.int64:
            raise TypeError(f"Connectivity must have type int64, got {self._connectivity.dtype}.")

    def _internal_check_vertices_property(self, key: str) -> None:
        r"""
        Internal method to check the validity of a specific vertices property.
        
        Parameters
        ----------
        key : :class:`str`
            The key of the vertices property to check.

        Raises
        ------
        TypeError
            If the vertices property is not a :class:`numpy.ndarray` or has invalid type.
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
            expected_dim = self._vertices_predefined_metadata[key].get("dim", None)
            check_method = self._vertices_predefined_metadata[key].get("check_method", None)
            if expected_dim is not None and value.shape[1] != expected_dim:
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
        key : :class:`str`
            The key of the elements property to check.

        Raises
        ------
        TypeError
            If the elements property is not a :class:`numpy.ndarray` or has invalid type.
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
            expected_dim = self._elements_predefined_metadata[key].get("dim", None)
            check_method = self._elements_predefined_metadata[key].get("check_method", None)
            if expected_dim is not None and value.shape[1] != expected_dim:
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

    def _get_vertices_property(self, key: Optional[None], default: Optional[numpy.ndarray] = None, raise_error: bool = False) -> Optional[numpy.ndarray]:
        r"""
        Internal method to get a vertices property or return a default value if the property does not exist.

        Parameters
        ----------
        key : Optional[:class:`str`]
            The key of the vertices property to retrieve. If None, returns the default value.

        default : Optional[:class:`numpy.ndarray`], optional
            The default value to return if the property does not exist, by default None.

        raise_error : :class:`bool`, optional
            If :obj:`True`, raises a KeyError if the property does not exist, by default :obj:`False`.

        Returns
        -------
        Optional[:class:`numpy.ndarray`]
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
        key : Optional[:class:`str`]
            The key of the elements property to retrieve. If None, returns the default value.

        default : Optional[:class:`numpy.ndarray`], optional
            The default value to return if the property does not exist, by default None.

        raise_error : :class:`bool`, optional
            If :obj:`True`, raises a KeyError if the property does not exist, by default :obj:`False`.

        Returns
        -------
        Optional[:class:`numpy.ndarray`]
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

    def set_elements_type(self, elements_type: Optional[str]) -> None:
        r"""
        Internal method to set the expected element type for the mesh.

        If :obj:`elements_type` is provided, the mesh will enforce that all elements are of the specified type. 
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

        Parameters
        ----------
        elements_type : :class:`str`
            The expected element type.

        Raises
        ------
        TypeError
            If the input is not a string.
        ValueError
            If the input is not a valid element type.
        """
        if elements_type is not None:
            if not isinstance(elements_type, str):
                raise TypeError(f"Element type must be a string, got {type(elements_type)}.")
            if elements_type not in self._mapping_elements_type_to_properties:
                raise ValueError(f"Invalid element type '{elements_type}'. Supported types are: {list(self._mapping_elements_type_to_properties.keys())}.")
        self._elements_type = elements_type
    

    # =======================
    # I/O Methods
    # =======================
    @classmethod
    def from_meshio(cls, mesh: meshio.Mesh, elements_type: Optional[str] = None, load_properties: bool = True, internal_bypass: bool = False) -> Mesh:
        r"""
        Create a Mesh instance from a :class:`meshio.Mesh` object.

        The following fields are extracted:

        - mesh.points → vertices
        - mesh.cells[0].data → connectivity
        - mesh.point_data → _vertex_properties as arrays of shape (N, A)
        - mesh.cell_data → _element_properties as arrays of shape (M, B)

        .. seealso::

            - :meth:`Mesh.to_meshio` for the reverse operation.
            - `meshio documentation <https://pypi.org/project/meshio/>`_ for more information.
            - :class:`Mesh` for more information on the Mesh class and element types.

        Parameters
        ----------
        mesh : :class:`meshio.Mesh`
            A meshio Mesh object to extract the first cell block and create the Mesh instance.

        elements_type : Optional[:class:`str`], optional
            The expected type of elements in the mesh, by default None.

        load_properties : :class:`bool`, optional
            If :obj:`True`, properties are extracted from the :class:`meshio.Mesh` object, by default :obj:`True`.

        internal_bypass : :class:`bool`, optional
            If :obj:`True`, internal checks are bypassed for better performance, by default :obj:`False`.

        Returns
        -------
        :class:`Mesh`
            A Mesh instance created from the :class:`meshio.Mesh` object.

        Raises
        ------
        TypeError
            If the input is not a :class:`meshio.Mesh` object.
        ValueError
            If the mesh structure is invalid.

        Examples
        --------

        Lets create a mesh using :class:`meshio.Mesh` and convert it to a :class:`Mesh` instance.

        .. code-block:: python

            import numpy as np
            import meshio
            from pysdic import Mesh

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            cells = [("triangle", np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]))]

            mesh = meshio.Mesh(points=points, cells=cells)

        Create a :class:`Mesh` instance from the :class:`meshio.Mesh` object.

        .. code-block:: python

            mesh3d = Mesh.from_meshio(mesh, elements_type="triangle_3")
            print(mesh3d.vertices)
            # Output: PointCloud with 4 points [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]

        """
        # Extract the expected properties based on element type
        expected_N_vpe = None
        expected_K = None
        meshio_cell_type = None

        if elements_type is not None:
            if not isinstance(elements_type, str):
                raise TypeError(f"Element type must be a string, got {type(elements_type)}.")
            if elements_type not in cls._mapping_elements_type_to_properties:
                raise ValueError(f"Invalid element type '{elements_type}'. Supported types are: {list(cls._mapping_elements_type_to_properties.keys())}.")
            expected_N_vpe = cls._mapping_elements_type_to_properties[elements_type]["expected_N_vpe"]
            expected_K = cls._mapping_elements_type_to_properties[elements_type]["expected_K"]
            meshio_cell_type = cls._mapping_elements_type_to_properties[elements_type]["meshio_cell_type"]

        # Validate the mesh structure
        if not isinstance(mesh, meshio.Mesh):
            raise TypeError(f"Input must be a meshio Mesh object, got {type(mesh)}.")
        if mesh.points.ndim != 2 or mesh.points.shape[1] < 1:
            raise ValueError("mesh.points must be a 2D array with at least one coordinate dimension.")
        if expected_K is not None and not mesh.points.shape[1] >= expected_K:
            raise ValueError(f"mesh.points must have embedding dimension greater or equal to {expected_K}, got {mesh.points.shape[1]}.")
        
        if not len(mesh.cells) == 1 or mesh.cells[0].data.ndim != 2 or mesh.cells[0].data.shape[1] < 1:
            raise ValueError("Invalid mesh structure.")
        if expected_N_vpe is not None and not mesh.cells[0].data.shape[1] == expected_N_vpe:
            raise ValueError(f"mesh.cells[0].data must have {expected_N_vpe} columns, got {mesh.cells[0].data.shape[1]}.")
        if meshio_cell_type is not None and not mesh.cells[0].type == meshio_cell_type:
            raise ValueError(f"mesh.cells[0].type must be '{meshio_cell_type}', got '{mesh.cells[0].type}'.")
        
        if not isinstance(load_properties, bool):
            raise TypeError(f"load_properties must be a boolean, got {type(load_properties)}.")
        
        if not isinstance(internal_bypass, bool):
            raise TypeError(f"internal_bypass must be a boolean, got {type(internal_bypass)}.")

        # Extract data
        vertices = PointCloud(mesh.points)
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
        return cls(vertices, connectivity, vertices_properties=vertices_properties, elements_properties=elements_properties, elements_type=elements_type, internal_bypass=internal_bypass)


    def to_meshio(self, save_properties: bool = True) -> meshio.Mesh:
        r"""
        Convert the :class:`Mesh` instance to a :class:`meshio.Mesh` object (:obj:`elements_type` must be defined).
        The mesh must not be empty. 

        .. warning::

            If the mesh does not have a defined element type, this method will raise a ValueError.
            See :meth:`set_elements_type` to define the element type before conversion.

        The following fields are created:

        - vertices → mesh.points
        - connectivity → mesh.cells[0].data
        - _vertex_properties as arrays of shape (N, A) → mesh.point_data
        - _element_properties as arrays of shape (M, B) → mesh.cell_data

        .. seealso::

            - :meth:`Mesh.from_meshio` for the reverse operation.
            - `meshio documentation <https://pypi.org/project/meshio/>`_ for more information.

        Parameters
        ----------
        save_properties : :class:`bool`, optional
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

            import numpy as np
            from pysdic import Mesh, PointCloud

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])

            mesh3d = Mesh(PointCloud.from_array(points), connectivity)

        Convert the :class:`Mesh` instance to a :class:`meshio.Mesh` object.

        .. code-block:: python

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
            raise ValueError("Cannot convert to meshio Mesh object without a defined element type. See method 'set_elements_type' to define it.")
        
        cells = [meshio.CellBlock(meshio_cell_type, data=self.connectivity)]
        
        if save_properties:
            point_data = {key: value for key, value in self._vertices_properties.items()}
            cell_data = {key: [value] for key, value in self._elements_properties.items()}
        else:
            point_data = {}
            cell_data = {}
        
        return meshio.Mesh(points=self.vertices.points, cells=cells, point_data=point_data, cell_data=cell_data)
    

    @classmethod
    def from_npz(cls, filename: str, elements_type: Optional[str] = None, load_properties: bool = True, internal_bypass: bool = False) -> Mesh:
        r"""
        Create a Mesh instance from a NPZ file.

        This method uses numpy to read the NPZ file and then converts it to a :class:`Mesh` instance.

        .. seealso::

            - :meth:`Mesh.to_npz` for the reverse operation.
            - `numpy documentation <https://numpy.org/doc/stable/reference/generated/numpy.load.html>`_ for more information.
            - :class:`Mesh` for more information on the Mesh class and element types.

        Parameters
        ----------
        filename : :class:`str`
            The path to the NPZ file.

        elements_type : Optional[:class:`str`], optional
            The expected type of elements in the mesh, by default None.

        load_properties : :class:`bool`, optional
            If :obj:`True`, properties are extracted from the NPZ file, by default :obj:`True`.

        internal_bypass : :class:`bool`, optional
            If :obj:`True`, internal checks are bypassed for better performance, by default :obj:`False`.

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

            import numpy as np
            from pysdic import Mesh

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            cells = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh = Mesh(vertices=points, connectivity=cells, elements_type="triangle_3")
            mesh.save_npz("simple_mesh.npz")

        Create a :class:`Mesh` instance from the NPZ file.

        .. code-block:: python

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

        return cls(
            vertices=PointCloud(points), 
            connectivity=connectivity, 
            vertices_properties=vertices_properties,
            elements_properties=elements_properties,
            elements_type=elements_type,
            internal_bypass=internal_bypass
        )
    

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
        filename : :class:`str`
            The path to the output NPZ file.

        save_properties : :class:`bool`, optional
            If :obj:`True`, properties are saved to the NPZ file, by default :obj:`True`.

        Raises
        ------
        ValueError
            If the file format is not supported or the mesh is empty.

            
        Examples
        --------
        Create a simple :class:`Mesh` instance.

        .. code-block:: python

            import numpy as np
            from pysdic import Mesh, PointCloud
            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = Mesh(PointCloud.from_array(points), connectivity)

        Save the :class:`Mesh` instance to a NPZ file.

        .. code-block:: python

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
                vertices=self.vertices.points,
                connectivity=self.connectivity,
                vertices_properties=self._vertices_properties,
                elements_properties=self._elements_properties
            )
        else:
            numpy.savez(
                path,
                vertices=self.vertices.points,
                connectivity=self.connectivity
            )
    

    @classmethod
    def from_vtk(cls, filename: str, elements_type: Optional[str] = None, load_properties: bool = True, internal_bypass: bool = False) -> Mesh:
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
        filename : :class:`str`
            The path to the VTK file.

        elements_type : Optional[:class:`str`], optional
            The expected type of elements in the mesh, by default None.

        load_properties : :class:`bool`, optional
            If :obj:`True`, properties are extracted from the VTK file, by default :obj:`True`.

        internal_bypass : :class:`bool`, optional
            If :obj:`True`, internal checks are bypassed for better performance, by default :obj:`False`.

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

            import numpy as np
            import meshio
            from pysdic import Mesh

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            cells = [("triangle", np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]))]

            mesh = meshio.Mesh(points=points, cells=cells)

        Save the meshio Mesh object to a VTK file.

        .. code-block:: python

            mesh.write("simple_mesh.vtk", file_format="vtk")

        Create a :class:`Mesh` instance from the VTK file.

        .. code-block:: python

            mesh3d = Mesh.from_vtk("simple_mesh.vtk")
            print(mesh3d.vertices)
            # Output: PointCloud with 4 points [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]      

        """
        path = os.path.abspath(os.path.expanduser(filename))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File '{filename}' does not exist.")
        
        mesh = meshio.read(filename, file_format="vtk")
        return cls.from_meshio(mesh, elements_type=elements_type, load_properties=load_properties, internal_bypass=internal_bypass)


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
        filename : :class:`str`
            The path to the output VTK file.
        
        save_properties : :class:`bool`, optional
            If :obj:`True`, properties are saved to the VTK file, by default :obj:`True`.

        Raises
        ------
        ValueError
            If the file format is not supported or the mesh is empty.

        Examples
        --------

        Create a simple :class:`Mesh` instance.

        .. code-block:: python

            import numpy as np
            from pysdic import Mesh, PointCloud

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = Mesh(PointCloud.from_array(points), connectivity)

        Save the :class:`Mesh` instance to a VTK file.

        .. code-block:: python

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
        value : Union[:class:`PointCloud`, :class:`numpy.ndarray`]
            The new vertices for the mesh with shape (:math:`N_v`, :math:`E`).

        Returns
        -------
        :class:`PointCloud`
            The vertices of the mesh as a PointCloud instance of shape (:math:`N_v`, :math:`E`).


        Examples
        --------

        Create a simple :class:`Mesh` instance.

        .. code-block:: python

            import numpy as np
            from pysdic import Mesh, PointCloud

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = Mesh(PointCloud.from_array(points), connectivity)

        Access the vertices of the mesh.

        .. code-block:: python

            print(mesh3d.vertices)
            # Output: PointCloud with 4 points [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        
        """
        return self._vertices
    
    @vertices.setter
    def vertices(self, value: Union[PointCloud, numpy.ndarray]) -> None:
        if not isinstance(value, PointCloud):
            value = PointCloud(value)
        self._vertices = value
        self._internal_check_vertices()

    @property
    def connectivity(self) -> numpy.ndarray:
        r"""
        [Get or Set] The connectivity of the mesh.

        The connectivity is represented as a numpy ndarray with shape (:math:`N_e`, :math:`N_{vpe}`)
        where :math:`N_e` is the number of elements and :math:`N_{vpe}` is the number of vertices per element.
        
        .. note::
            
            - An alias for this property is :attr:`mesh.elements`.
            - This property is settable.

        .. warning::

            If the connectivity is changed, the properties may become invalid. 
            Please ensure to recompute or update them accordingly.        

            If you change the connectivity, please ensure that all indices are valid with respect to the current vertices.
            To change the connectivity, it is recommended to create a new Mesh instance with the updated vertices and connectivity
            rather than modifying the connectivity in place. For memory considerations, you can also modify the connectivity in place, but please ensure that
            all necessary checks are performed before using this mode.

        Parameters
        ----------
        value : :class:`numpy.ndarray`
            The new connectivity for the mesh as an array-like of shape (:math:`N_e`, :math:`N_{vpe}`).

        Returns
        -------
        :class:`numpy.ndarray`
            The connectivity of the mesh.


        Examples
        --------

        Create a simple :class:`Mesh` instance.

        .. code-block:: python

            import numpy as np
            from pysdic import Mesh, PointCloud

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = Mesh(PointCloud.from_array(points), connectivity)

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
        [Get or Set] Alias for :attr:`connectivity` property.
        """
        return self.connectivity
    
    @elements.setter
    def elements(self, value: numpy.ndarray) -> None:
        self.connectivity = value

    @property
    def n_vertices(self) -> int:
        r"""
        [Get] The number of vertices :math:`N_v` in the mesh (same as :attr:`N_v`).

        .. note::

            Alias for `mesh.vertices.n_points`.
            You can also use `len(mesh.vertices)`

        Returns
        -------
        :class:`int`
            The number of vertices in the mesh.

        Examples
        --------

        Create a simple :class:`Mesh` instance.

        .. code-block:: python

            import numpy as np
            from pysdic import Mesh, PointCloud

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = Mesh(PointCloud.from_array(points), connectivity)

        Get the number of vertices in the mesh.

        .. code-block:: python

            print(mesh3d.n_vertices)
            # Output: 4

        """
        return len(self.vertices)

    
    @property
    def N_v(self) -> int:
        r"""
        [Get] Alias for :attr:`n_vertices` property.
        """
        return self.n_vertices
    

    @property
    def n_elements(self) -> int:
        r"""
        [Get] The number of elements :math:`N_e` in the mesh (same as :attr:`N_e`).

        Returns
        -------
        :class:`int`
            The number of elements in the mesh.

        Examples
        --------

        Create a simple :class:`Mesh` instance.

        .. code-block:: python

            import numpy as np
            from pysdic import Mesh, PointCloud

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = Mesh(PointCloud.from_array(points), connectivity)

        Get the number of elements in the mesh.

        .. code-block:: python

            print(mesh3d.n_elements)
            # Output: 4

        """
        return self.connectivity.shape[0]

    @property
    def N_e(self) -> int:
        r"""
        [Get] Alias for :attr:`n_elements` property.
        """
        return self.n_elements
    

    @property
    def n_dimensions(self) -> int:
        r"""
        [Get] The embedding dimension :math:`E` of the mesh (same as :attr:`E`).

        Returns
        -------
        :class:`int`
            The embedding dimension of the mesh.

        Examples
        --------

        Create a simple :class:`Mesh` instance.

        .. code-block:: python

            import numpy as np
            from pysdic import Mesh, PointCloud

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = Mesh(PointCloud.from_array(points), connectivity)

        Get the embedding dimension of the mesh.

        .. code-block:: python

            print(mesh3d.n_dimensions)
            # Output: 3

        """
        return self.vertices.n_dimensions
    
    @property
    def E(self) -> int:
        r"""
        [Get] Alias for :attr:`n_dimensions` property.
        """
        return self.n_dimensions

    @property
    def n_vertices_per_element(self) -> int:
        r"""
        [Get] The number of vertices per element :math:`N_{vpe}` in the mesh (same as :attr:`N_vpe`).

        Returns
        -------
        :class:`int`
            The number of vertices per element.

        
        Examples
        --------

        Create a simple :class:`Mesh` instance.

        .. code-block:: python

            import numpy as np
            from pysdic import Mesh, PointCloud

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = Mesh(PointCloud.from_array(points), connectivity)

        Get the number of vertices per element in the mesh.

        .. code-block:: python

            print(mesh3d.n_vertices_per_element)
            # Output: 3
        
        """
        return self.connectivity.shape[1]

    @property
    def N_vpe(self) -> int:
        r"""
        [Get] Alias for :attr:`n_vertices_per_element` property.
        """
        return self.n_vertices_per_element

    @property
    def n_topological_dimensions(self) -> int:
        r"""
        [Get] The topological dimension :math:`K` of the elements in the mesh (same as :attr:`K`).

        Returns
        -------
        :class:`int`
            The topological dimension of the elements.


        Examples
        --------

        Create a simple :class:`Mesh` instance.

        .. code-block:: python

            import numpy as np
            from pysdic import Mesh, PointCloud

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = Mesh(PointCloud.from_array(points), connectivity)

        Get the topological dimension of the elements in the mesh.

        .. code-block:: python

            print(mesh3d.n_topological_dimensions)
            # Output: 2
        
        """
        expected_K = self._get_expected_K()
        if expected_K is not None:
            return expected_K
        raise ValueError("Topological dimension cannot be deduced for meshes without a specified element type.")
    
    @property
    def K(self) -> int:
        r"""
        [Get] Alias for :attr:`n_topological_dimensions` property.
        """
        return self.n_topological_dimensions
    

    @property
    def elements_type(self) -> Optional[str]:
        r"""
        [Get or Set] The element type of the mesh.

        .. seealso::

            - :meth:`set_elements_type` to set the element type of the mesh.

        Returns
        -------
        :class:`str` or None
            The element type of the mesh, or None if not specified.

        
        Examples
        --------

        Create a simple :class:`Mesh` instance with a specified element type.

        .. code-block:: python

            import numpy as np
            from pysdic import Mesh, PointCloud

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = Mesh(PointCloud.from_array(points), connectivity, elements_type="triangle_3")

        Get the element type of the mesh.

        .. code-block:: python

            print(mesh3d.elements_type)
            # Output: triangle_3

        """
        return self._elements_type
    
    @elements_type.setter
    def elements_type(self, value: Optional[str]) -> None:
        self.set_elements_type(value)
    

    @property
    def elements_properties(self) -> Dict[str, numpy.ndarray]:
        r"""
        [Get] The properties associated with the elements of the mesh (see :meth:`set_elements_property` and :meth:`get_elements_property`).

        Returns
        -------
        :class:`Dict[str, numpy.ndarray]`
            A dictionary containing the properties of the elements, where keys are property names and values are numpy arrays.

        
        Examples
        --------

        Create a simple :class:`Mesh` instance with element properties.

        .. code-block:: python

            import numpy as np
            from pysdic import Mesh, PointCloud

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            elements_properties = {"material_id": np.array([1, 1, 2, 2])}
            mesh3d = Mesh(PointCloud.from_array(points), connectivity, elements_properties=elements_properties)

        Get the properties of the elements in the mesh.

        .. code-block:: python

            print(mesh3d.elements_properties)
            # Output: {'material_id': array([1, 1, 2, 2])}

        """
        return self._elements_properties
    

    @property
    def expected_N_vpe(self) -> Optional[int]:
        r"""
        [Get] The expected number of vertices per element :math:`N_{vpe}` for the mesh.

        .. seealso::

            - :meth:`set_elements_type` to set the element type of the mesh and update the expected number of vertices per element accordingly.

        Returns
        -------
        :class:`int` or None
            The expected number of vertices per element, or None if not specified.

        
        Examples
        --------

        Create a simple :class:`Mesh` instance with a specified element type.

        .. code-block:: python

            import numpy as np
            from pysdic import Mesh, PointCloud

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = Mesh(PointCloud.from_array(points), connectivity, elements_type="triangle_3")

        Get the expected number of vertices per element for the mesh.

        .. code-block:: python

            print(mesh3d.expected_N_vpe)
            # Output: 3

        """
        return self._get_expected_N_vpe()
    

    @property
    def expected_K(self) -> Optional[int]:
        r"""
        [Get] The expected topological dimension :math:`K` for the mesh.

        .. seealso::

            - :meth:`set_elements_type` to set the element type of the mesh and update the expected topological dimension accordingly.

        Returns
        -------
        :class:`int` or None
            The expected topological dimension, or None if not specified.

        
        Examples
        --------

        Create a simple :class:`Mesh` instance with a specified element type.

        .. code-block:: python

            import numpy as np
            from pysdic import Mesh, PointCloud

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = Mesh(PointCloud.from_array(points), connectivity, elements_type="triangle_3")

        Get the expected topological dimension for the mesh.

        .. code-block:: python

            print(mesh3d.expected_K)
            # Output: 2

        """
        return self._get_expected_K()
    
    @property
    def meshio_cell_type(self) -> Optional[str]:
        r"""
        [Get] The corresponding meshio cell type for the mesh.

        .. seealso::

            - :meth:`set_elements_type` to set the element type of the mesh and update the meshio cell type accordingly.

        Returns
        -------
        :class:`str` or None
            The meshio cell type, or None if not specified.

        
        Examples
        --------

        Create a simple :class:`Mesh` instance with a specified element type.

        .. code-block:: python

            import numpy as np
            from pysdic import Mesh, PointCloud

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = Mesh(PointCloud.from_array(points), connectivity, elements_type="triangle_3")

        Get the meshio cell type for the mesh.

        .. code-block:: python

            print(mesh3d.meshio_cell_type())
            # Output: triangle

        """
        return self._get_meshio_cell_type()
    


    @property
    def vertices_properties(self) -> Dict[str, numpy.ndarray]:
        r"""
        [Get] The properties associated with the vertices of the mesh (see :meth:`set_vertices_property` and :meth:`get_vertices_property`).

        Returns
        -------
        :class:`Dict[str, numpy.ndarray]`
            A dictionary containing the properties of the vertices, where keys are property names and values are numpy arrays.

        
        Examples
        --------

        Create a simple :class:`Mesh` instance with vertex properties.

        .. code-block:: python

            import numpy as np
            from pysdic import Mesh, PointCloud

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            vertices_properties = {"temperature": np.array([100, 150, 200, 250])}
            mesh3d = Mesh(PointCloud.from_array(points), connectivity, vertices_properties=vertices_properties)

        Get the properties of the vertices in the mesh.

        .. code-block:: python

            print(mesh3d.vertices_properties)
            # Output: {'temperature': array([100, 150, 200, 250])}

        """
        return self._vertices_properties
    

    
    @property
    def vtk_cell_type(self) -> Optional[int]:
        r"""
        [Get] The corresponding VTK cell type for the mesh.

        .. seealso::

            - :meth:`set_elements_type` to set the element type of the mesh and update the VTK cell type accordingly.

        Returns
        -------
        :class:`int` or None
            The VTK cell type, or None if not specified.

        
        Examples
        --------

        Create a simple :class:`Mesh` instance with a specified element type.

        .. code-block:: python

            import numpy as np
            from pysdic import Mesh, PointCloud

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = Mesh(PointCloud.from_array(points), connectivity, elements_type="triangle_3")

        Get the VTK cell type for the mesh.

        .. code-block:: python

            print(mesh3d.vtk_cell_type())
            # Output: 5

        """
        return self._get_vtk_cell_type()

    # =======================
    # Properties Methods
    # =======================
    def get_vertices_property(self, key: str) -> Optional[numpy.ndarray]:
        r"""
        Get a property associated with the vertices of the mesh with shape (:math:`N_v`, A).

        ``N_v`` is the number of vertices and ``A`` is the size of the property.

        .. seealso::

            - :meth:`Mesh.set_vertices_property` to set a vertices property.

        Parameters
        ----------
        key : :class:`str`
            The key of the property to retrieve.

        Returns
        -------
        :class:`numpy.ndarray` or None
            The property associated with the vertices, or None if the property does not exist.

        
        Examples
        --------

        Create a simple :class:`Mesh` instance with a vertex property.

        .. code-block:: python

            import numpy as np
            from pysdic import Mesh, PointCloud

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = Mesh(PointCloud.from_array(points), connectivity, elements_type="triangle_3")

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
        Get a property associated with the elements of the mesh with shape (:math:`N_e`, B)

        ``N_e`` is the number of elements and ``B`` is the size of the property.

        .. seealso::

            - :meth:`Mesh.set_elements_property` to set an elements property.

        Parameters
        ----------
        key : :class:`str`
            The key of the property to retrieve.

        Returns
        -------
        :class:`numpy.ndarray` or None
            The property associated with the elements, or None if the property does not exist.

        Examples
        --------

        Create a simple :class:`Mesh` instance with an element property.

        .. code-block:: python

            import numpy as np
            from pysdic import Mesh, PointCloud

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = Mesh(PointCloud.from_array(points), connectivity, elements_type="triangle_3")

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
        Set a property associated with the vertices of the mesh with shape (:math:`N_v`, A).

        ``N_v`` is the number of vertices and ``A`` is the size of the property.

        .. note::

            Even if the size of the property is 1, the property must be provided as a 2D array of shape (:math:`N_v`, 1).

        .. seealso::

            - :meth:`Mesh.get_vertices_property` to get a vertices property.

        Parameters
        ----------
        key : :class:`str`
            The key of the property to set.

        value : Optional[:class:`numpy.ndarray`]
            The property to associate with the vertices as an array-like of shape (:math:`N_v`, A),
            where :math:`N_v` is the number of vertices and A is the number of attributes for that property.
            If None, the property is removed.   

        Raises
        ------
        TypeError
            If value is not a :class:`numpy.ndarray` or None.
        ValueError
            If value does not have the correct shape.

        
        Examples
        --------

        Create a simple :class:`Mesh` instance.

        .. code-block:: python

            import numpy as np
            from pysdic import Mesh, PointCloud

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = Mesh(PointCloud.from_array(points), connectivity, elements_type="triangle_3")

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
        Set a property associated with the elements of the mesh with shape (:math:`N_e`, B).

        ``N_e`` is the number of elements and ``B`` is the size of the property.

        .. note::

            Even if the size of the property is 1, the property must be provided as a 2D array of shape (:math:`N_e`, 1).

        .. seealso::

            - :meth:`Mesh.get_elements_property` to get an elements property.

        Parameters
        ----------
        key : str
            The key of the property to set.

        value : Optional[:class:`numpy.ndarray`]
            The property to associate with the elements as an array-like of shape (:math:`N_e`, B),
            where :math:`N_e` is the number of elements and B is the number of attributes for that property.
            If None, the property is removed.

        Raises
        ------
        TypeError
            If value is not a :class:`numpy.ndarray` or None.
        ValueError
            If value does not have the correct shape.


        Examples
        --------

        Create a simple :class:`Mesh` instance.

        .. code-block:: python

            import numpy as np
            from pysdic import Mesh, PointCloud

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = Mesh(PointCloud.from_array(points), connectivity)

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
        key : :class:`str`
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
        key : :class:`str`
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
        Tuple[:class:`str`]
            A tuple containing all keys of the vertices properties.
        """
        return tuple(self._vertices_properties.keys())
    
    def list_elements_properties(self) -> Tuple[str]:
        r"""
        List all keys of the properties associated with the elements of the mesh.

        Returns
        -------
        Tuple[:class:`str`]
            A tuple containing all keys of the elements properties.
        """
        return tuple(self._elements_properties.keys())
    
    def has_vertices_property(self, key: str) -> bool:
        r"""
        Check if a property associated with the vertices of the mesh exists.

        Parameters
        ----------
        key : :class:`str`
            The key of the property to check.

        Returns
        -------
        :class:`bool`
            True if the property exists, False otherwise.
        """
        return key in self._vertices_properties
    
    def has_elements_property(self, key: str) -> bool:
        r"""
        Check if a property associated with the elements of the mesh exists.

        Parameters
        ----------
        key : :class:`str`
            The key of the property to check.

        Returns
        -------
        :class:`bool`
            True if the property exists, False otherwise.
        """
        return key in self._elements_properties

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
            The elements properties stored in the mesh are extended with default :obj:`numpy.nan` values for the new elements.

        Parameters
        ----------
        new_connectivity : :class:`numpy.ndarray`
            An array of shape (:math:`P`, :math:`N_{vpe}`) containing the connectivity of the new elements to add,
            where :math:`P` is the number of new elements and :math:`N_{vpe}` is the number of vertices per element.

        Raises
        ------
        ValueError
            If :obj:`new_connectivity` does not have the correct shape or contains invalid indices.

            
        Examples
        --------

        Create a simple :class:`Mesh` instance.

        .. code-block:: python

            import numpy as np
            from pysdic import Mesh, PointCloud

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
            connectivity = np.array([[0, 1, 2]])
            mesh3d = Mesh(PointCloud.from_array(points), connectivity, elements_type="triangle_3")

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


    def add_vertices(self, new_vertices: Union[PointCloud, numpy.ndarray]) -> None:
        r"""
        Add new vertices to the mesh by appending new vertex coordinates.

        .. note::

            The new vertices will be added to the end of the existing vertex list.
            The vertices properties stored in the mesh are extended with default :obj:`numpy.nan` values for the new vertices.

        Parameters
        ----------
        new_vertices : Union[:class:`PointCloud`, :class:`numpy.ndarray`]
            An array of shape (:math:`Q`, :math:`E`) containing the coordinates of the new vertices to add,
            where :math:`Q` is the number of new vertices and :math:`E` is the embedding dimension.

        Raises
        ------
        ValueError
            If :obj:`new_vertices` does not have the correct shape.

            
        Examples
        --------

        Create a simple :class:`Mesh` instance.

        .. code-block:: python

            import numpy as np
            from pysdic import Mesh, PointCloud

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
            connectivity = np.array([[0, 1, 2]])
            mesh3d = Mesh(PointCloud.from_array(points), connectivity, elements_type="triangle_3")

        Add new vertices to the mesh.

        .. code-block:: python

            new_vertices = np.array([[0, 0, 1], [1, 1, 1]])
            mesh3d.add_vertices(new_vertices)

            print(mesh3d.vertices)
            # Output: PointCloud with 5 points [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]

        """
        if not isinstance(new_vertices, PointCloud):
            new_vertices = PointCloud.from_array(new_vertices)
        if not new_vertices.n_dimensions == self.n_dimensions:
            raise ValueError(
                f"new_vertices must have the same embedding dimension as the mesh ({self.n_dimensions}). "
                f"Got {new_vertices.n_dimensions}."
            )
        
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
        vertex_indices : :class:`numpy.ndarray`
            An array of shape (:math:`R`,) containing the indices of the vertices to check.

        Returns
        -------
        :class:`numpy.ndarray`
            A boolean array of shape (:math:`R`,) where each entry indicates whether the corresponding vertex is used in the connectivity.

        Raises
        ------
        ValueError
            If any :obj:`vertex_indices` is out of bounds.

        
        Examples
        --------

        Create a simple :class:`Mesh` instance.

        .. code-block:: python

            import numpy as np
            from pysdic import Mesh, PointCloud

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = Mesh(PointCloud.from_array(points), connectivity, elements_type="triangle_3")

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
        :class:`bool`
            True if the mesh is empty, False otherwise.

        
        Examples
        --------

        Create an empty :class:`Mesh` instance.

        .. code-block:: python

            from pysdic import Mesh

            empty_mesh3d = Mesh(PointCloud.from_array(numpy.array([]).reshape(0, 3)), connectivity=numpy.array([]).reshape(0, 3))
            print(empty_mesh3d.is_empty())
            # Output: True

        Create a non-empty :class:`Mesh` instance.

        .. code-block:: python

            import numpy as np
            from pysdic import Mesh, PointCloud

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = Mesh(PointCloud.from_array(points), connectivity)

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
        element_indices : :class:`numpy.ndarray`
            An array of shape (:math:`R`,) containing the indices of the elements to keep,
            where :math:`R` is the number of elements to keep.

        Raises
        ------
        ValueError
            If :obj:`element_indices` does not have the correct shape or contains invalid indices.

        
        Examples
        --------

        Create a simple :class:`Mesh` instance.

        .. code-block:: python

            import numpy as np
            from pysdic import Mesh, PointCloud
        
            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = Mesh(PointCloud.from_array(points), connectivity)

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
        element_indices : :class:`numpy.ndarray`
            An array of shape (:math:`R`,) containing the indices of the elements to remove,
            where :math:`R` is the number of elements to remove.

        Raises
        ------
        ValueError
            If :obj:`element_indices` does not have the correct shape or contains invalid indices.

            
        Examples
        --------

        Create a simple :class:`Mesh` instance.

        .. code-block:: python

            import numpy as np
            from pysdic import Mesh, PointCloud

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = Mesh(PointCloud.from_array(points), connectivity)

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
        vertex_indices : :class:`numpy.ndarray`
            An array of shape (:math:`R`,) containing the indices of the vertices to remove,
            where :math:`R` is the number of vertices to remove.

        Raises
        ------
        ValueError
            If :obj:`vertex_indices` does not have the correct shape, contains invalid indices, or if any vertex is used in the connectivity.


        Examples
        --------

        Create a simple :class:`Mesh` instance.

        .. code-block:: python

            import numpy as np
            from pysdic import Mesh, PointCloud

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = Mesh(PointCloud.from_array(points), connectivity)

        Add some properties to the vertices.

        .. code-block:: python

            vertex_property = np.array([0.0, 1.0, 2.0, 3.0, 4.0]).reshape(-1, 1) # Shape (5, 1)
            mesh3d.set_vertices_property("my_vertex_property", vertex_property)

        Remove vertex with index 4 from the mesh.

        .. code-block:: python

            vertex_indices = np.array([4])
            mesh3d.remove_vertices(vertex_indices)

            print(mesh3d.vertices)
            # Output: PointCloud with 4 points [[0 0 0] [1 0 0] [0 1 0] [0 0 1]]

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

        Create a simple :class:`Mesh` instance.

        .. code-block:: python

            import numpy as np
            from pysdic import Mesh, PointCloud

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            mesh3d = Mesh(PointCloud.from_array(points), connectivity)

        Add some properties to the vertices.

        .. code-block:: python

            vertex_property = np.array([0.0, 1.0, 2.0, 3.0, 4.0]).reshape(-1, 1) # Shape (5, 1)
            mesh3d.set_vertices_property("my_vertex_property", vertex_property)

        Remove all unused vertices from the mesh.

        .. code-block:: python

            mesh3d.remove_unused_vertices()

            print(mesh3d.vertices)
            # Output: PointCloud with 4 points [[0 0 0] [1 0 0] [0 1 0] [0 0 1]]

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
    def copy(self, copy_properties: bool = True) -> Mesh:
        r"""
        Create a deep copy of the Mesh instance.

        .. note::

            This method creates a new :class:`Mesh` instance with copies of the vertices, connectivity, and properties.

        Parameters
        ----------
        copy_properties : :class:`bool`, optional
            If :obj:`True`, the vertices and elements properties are also copied, by default :obj:`True`.

        Returns
        -------
        :class:`Mesh`
            A deep copy of the Mesh instance.
        """
        vertices = self.vertices.copy()
        connectivity = numpy.copy(self.connectivity)

        if not copy_properties:
            return self.__class__(vertices, connectivity, elements_type=self.elements_type, internal_bypass=self.internal_bypass)
        
        new_vertices_properties = {key: numpy.copy(value) for key, value in self._vertices_properties.items()}
        new_elements_properties = {key: numpy.copy(value) for key, value in self._elements_properties.items()}
        return self.__class__(vertices, connectivity, new_vertices_properties, new_elements_properties, elements_type=self.elements_type, internal_bypass=self.internal_bypass)
    


    # =======================
    # Methods for shape functions
    # =======================
    def shape_functions(self, natural_coordinates: numpy.ndarray, return_derivatives: bool = False, default: Number = 0.0) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
        r"""
        Compute the shape functions and their first derivatives for the mesh elements at given natural coordinates.

        In a space of dimension :math:`E`, we consider a :math:`K`-dimensional element (with :math:`K \leq E`) defined by :math:`N_{vpe}` nodes/vertices.

        For an :math:`K`-dimensional element defined by :math:`N_{vpe}` nodes, points inside the element are represented in a local coordinate system :math:`(\xi, \eta, \zeta, ...)` also named ``natural coordinates``.
        Shape functions are defined in this local coordinate system in order to interpolate values at any point within the element based on the values at the nodes.

        .. math::

            P(\xi, \eta, \zeta, ...) = \sum_{i=1}^{N_{vpe}} N_i(\xi, \eta, \zeta, ...) P_i

        where :math:`P` is the interpolated value at the point, :math:`N_i` are the shape functions, and :math:`P_i` are the nodal values.


        Method to compute the shape functions and their first derivatives for the mesh elements.
        
        .. seealso::

            - :doc:`../api_documentation/shape_functions` for more information on shape functions.

        Parameters
        ----------
        natural_coordinates: :class:`numpy.ndarray`
            An array of shape (:math:`M`,:math:`K`) of :math:`M` points in the natural coordinate system with :math:`K` dimensions where the shape functions are to be evaluated.

        return_derivatives: :class:`bool`, optional
            If :obj:`True`, the method also returns the derivatives of the shape functions with respect to the local coordinates. Default is :obj:`False`.

        default: :class:`numbers.Real`, optional
            The default value to assign to shape functions for points outside the valid range. Default is :obj:`0.0`.


        Returns
        -------
        shape_functions: :class:`numpy.ndarray`
            An array of shape (:math:`M`, :math:`N_{vpe}`) containing the evaluated shape functions at the specified points for the :math:`N_{vpe}` nodes of the element.

        shape_function_derivatives: :class:`numpy.ndarray`, optional
            An array of shape (:math:`M`, :math:`N_{vpe}`, :math:`K`) containing the derivatives of the shape functions with respect to the local coordinates, if :obj:`return_derivatives` is True.


        """
        shape_functions_method = self._get_shape_functions_method()
        if shape_functions_method is None:
            raise NotImplementedError("Shape functions method is not implemented for this mesh type.")
        return shape_functions_method(natural_coordinates, return_derivatives=return_derivatives, default=default)

    # ===================================
    # New Properties for SurfaceMesh
    # ===================================
    def _internal_check_uvmap(self, uvmap: numpy.ndarray) -> None:
        r"""
        Internal method to check the validity of the uvmap property.

        Parameters
        ----------
        uvmap : :class:`numpy.ndarray`
            The uvmap property to check, should be of shape (:math:`N_e`, 2 * :math:`N_{vpe}`) where :math:`N_e` is the number of elements and :math:`N_{vpe}` is the number of vertices per element.

        Raises
        ------
        ValueError
            If any uv coordinate is not in the range [0, 1].
        """
        if uvmap.ndim != 2 or uvmap.shape[0] != self.n_elements or uvmap.shape[1] != 2 * self.n_vertices_per_element:
            raise ValueError(
                f"uvmap must be a 2D array with shape ({self.n_elements}, {2 * self.n_vertices_per_element}). "
                f"Got shape {uvmap.shape}."
            )
        if numpy.any(uvmap < 0) or numpy.any(uvmap > 1):
            raise ValueError("All UV coordinates must be in the range [0, 1].")

    @property
    def elements_uvmap(self) -> Optional[numpy.ndarray]:
        r"""
        [Get or Set] The UV mapping of each element in the mesh (only for surfacique meshes :math:`K=2`).

        The UV mapping is stored as a numpy ndarray of shape (:math:`N_e`, 2 * :math:`N_{vpe}`), where :math:`N_e` is the number of elements and :math:`N_{vpe}` is the number of vertices per element.

        The values correspond to the UV coordinates of the :math:`N_{vpe}` vertices of the element: :math:`(u_1, v_1, u_2, v_2, u_3, v_3, ..., u_{N_{vpe}}, v_{N_{vpe}})`.

        .. note::

            The UV coordinates are stored as an element property of the mesh under the key "uvmap".

        Parameters
        ----------
        value : :class:`numpy.ndarray`, optional
            A numpy ndarray of shape (:math:`N_e`, 2 * :math:`N_{vpe}`) to set as the UV mapping.

        Returns
        -------
        :class:`numpy.ndarray` or None
            An array of shape (:math:`N_e`, 2 * :math:`N_{vpe}`) where :math:`N_e` is the number of elements. Or None if not set.
        """
        expected_K = self._get_expected_K()
        if expected_K is not None and expected_K != 2:
            raise AttributeError("elements_uvmap property is only available for surfacique meshes (K=2).")
        return self.get_elements_property("uvmap")

    @elements_uvmap.setter
    def elements_uvmap(self, value: Optional[numpy.ndarray]) -> None:
        expected_K = self._get_expected_K()
        if expected_K is not None and expected_K != 2:
            raise AttributeError("elements_uvmap property is only available for surfacique meshes (K=2).")
        self.set_elements_property("uvmap", value)


    # =======================
    # Visualization Methods
    # =======================
    def visualize(
            self,
            vertices_color: str = "black",
            vertices_size: int = 5,
            vertices_opacity: float = 1.0,
            edges_color: str = "black",
            edges_width: int = 1,
            edges_opacity: float = 1.0,
            faces_color: str = "gray",   
            faces_opacity: float = 0.5,
            show_vertices: bool = True,
            show_edges: bool = True,
            show_faces: bool = True,
            title: Optional[str] = None,
            show_axes: bool = True,
            show_grid: bool = True,
        ) -> None:
        r"""
        Visualize the surface mesh using PyVista (only for :math:`E \leq 3` surfacique meshes :math:`K=2`).

        This method creates a 3D plot of the mesh, displaying its vertices, edges, and faces.
        The appearance of the vertices, edges, and faces can be customized using various parameters.

        .. seealso::

            - :meth:`visualize_vertices_property` to visualize a vertex property on the mesh.
            - :meth:`visualize_texture` to visualize the texture of the mesh.
            - :meth:`visualize_integration_points` to visualize integration points on the mesh.
            - :meth:`set_elements_type` to set the element type of the mesh and update the VTK cell type accordingly.

        Parameters
        ----------
        vertices_color : :class:`str`, optional
            Color of the vertices (points) in the mesh, by default :obj:`"black"`.

        vertices_size : :class:`int`, optional
            Size of the vertices (points) in the mesh, by default :obj:`5`.

        vertices_opacity : :class:`float`, optional
            Opacity of the vertices (points) in the mesh (0.0 to 1.0), by default :obj:`1.0`.

        edges_color : :class:`str`, optional
            Color of the edges in the mesh, by default :obj:`"black"`.

        edges_width : :class:`int`, optional
            Width of the edges in the mesh, by default :obj:`1`.

        edges_opacity : :class:`float`, optional
            Opacity of the edges in the mesh (0.0 to 1.0), by default :obj:`1.0`.

        faces_color : :class:`str`, optional
            Color of the faces in the mesh, by default :obj:`"gray"`.

        faces_opacity : :class:`float`, optional
            Opacity of the faces in the mesh (0.0 to 1.0), by default :obj:`0.5`.

        show_points : :class:`bool`, optional
            Whether to display the vertices (points) of the mesh, by default :obj:`True`.

        show_edges : :class:`bool`, optional
            Whether to display the edges of the mesh, by default :obj:`True`.

        show_faces : :class:`bool`, optional
            Whether to display the faces of the mesh, by default :obj:`True`.

        title : Optional[:class:`str`], optional
            Title of the plot, by default None.

        show_axes : :class:`bool`, optional
            Whether to display the axes in the plot, by default :obj:`True`.
            
        show_grid : :class:`bool`, optional
            Whether to display the grid in the plot, by default :obj:`True`.


        More Information
        -------------------------

        This method only display the mesh without additional elements.
        To display additional elements, use PyVista directly.

        .. seealso::
        
            - `PyVista Documentation <https://docs.pyvista.org>`_ for more details on visualization options.

            
        Examples
        --------

        Lets consider a simple linear triangular mesh in 3D:

        .. code-block:: python

            from pysdic import create_triangle_3_heightmap
            import numpy

            surface_mesh = create_triangle_3_heightmap(
                height_function=lambda x, y: 0.5 * numpy.sin(numpy.pi * x) * numpy.cos(numpy.pi * y),
                x_bounds=(-1.0, 1.0),
                y_bounds=(-1.0, 1.0),
                n_x=50,
                n_y=50,
            )
            surface_mesh.visualize(faces_color='green', faces_opacity=0.7, edges_color='black')

        .. figure:: /_static/meshes/mesh_visualize_example.png
           :width: 600
           :align: center
            
           Example of a 3D triangular mesh visualization using the `visualize` method.
            
        """
        # Check if visualizable
        if self.n_dimensions > 3:
            raise NotImplementedError("Visualization is only supported for meshes with embedding dimension E <= 3.")
        if self.n_dimensions == 3:
            mesh = self
        else:
            mesh = self.copy()
            mesh.vertices.extend_n_dimensions(3)

        expected_K = mesh._get_expected_K()
        if expected_K is not None and expected_K != 2:
            raise NotImplementedError("Visualization is only supported for surfacique meshes (K=2).")
        vtk_cell_type = mesh._get_vtk_cell_type()
        if vtk_cell_type is None:
            raise NotImplementedError(
                f"Visualization is not supported as vtk_cell_type is not defined for mesh elements. See method `set_elements_type` to define it."
            )
        
        # Check input data
        if mesh.n_vertices == 0:
            raise ValueError("Cannot visualize an empty mesh.")
        if mesh.n_elements == 0:
            raise ValueError("Cannot visualize a mesh without elements.")

        if not isinstance(vertices_color, str):
            raise ValueError("Vertices color must be a string.")
        if not (isinstance(vertices_size, Number) and vertices_size > 0):
            raise ValueError("Vertices size must be a positive number.")
        if not (isinstance(vertices_opacity, Number) and 0.0 <= vertices_opacity <= 1.0):
            raise ValueError("Vertices opacity must be a float between 0.0 and 1.0.")
        if not isinstance(edges_color, str):
            raise ValueError("Edges color must be a string.")
        if not (isinstance(edges_width, Number) and edges_width > 0):
            raise ValueError("Edges width must be a positive number.")
        if not (isinstance(edges_opacity, Number) and 0.0 <= edges_opacity <= 1.0):
            raise ValueError("Edges opacity must be a float between 0.0 and 1.0.")
        if not isinstance(faces_color, str):
            raise ValueError("Faces color must be a string.")
        if not (isinstance(faces_opacity, Number) and 0.0 <= faces_opacity <= 1.0):
            raise ValueError("Faces opacity must be a float between 0.0 and 1.0.")
        
        if not isinstance(show_vertices, bool):
            raise ValueError("show_vertices must be a boolean.")
        if not isinstance(show_edges, bool):
            raise ValueError("show_edges must be a boolean.")
        if not isinstance(show_faces, bool):
            raise ValueError("show_faces must be a boolean.")
        if not isinstance(show_axes, bool):
            raise ValueError("show_axis must be a boolean.")
        if not isinstance(show_grid, bool):
            raise ValueError("show_grid must be a boolean.")
        
        if title is not None and not isinstance(title, str):
            raise ValueError("Title must be a string.")
        
        # Create a PyVista mesh
        n_cells = mesh.n_elements

        cells = numpy.hstack([numpy.full((n_cells, 1), mesh.n_vertices_per_element), mesh.connectivity]).astype(numpy.int64).ravel()
        celltypes = numpy.full(n_cells, vtk_cell_type, dtype=numpy.uint8)

        pv_mesh = pyvista.UnstructuredGrid(cells, celltypes, mesh.vertices.points)

        # Create a PyVista plotter
        plotter = pyvista.Plotter()

        # Add faces if required
        if show_faces:
            plotter.add_mesh(
                pv_mesh, 
                color=faces_color, 
                opacity=faces_opacity
            )

        # Add edges if required
        if show_edges:
            edges = pv_mesh.extract_all_edges()
            plotter.add_mesh(
                edges, 
                color=edges_color, 
                line_width=edges_width,
                opacity=edges_opacity
            )

        # Add vertices if required
        if show_vertices:
            plotter.add_points(
                mesh.vertices.points, 
                color=vertices_color, 
                point_size=vertices_size,
                opacity=vertices_opacity,
                render_points_as_spheres=True
            )

        # Show the plot
        if title is not None:
            plotter.add_title(title)
        if show_axes:
            plotter.show_axes() 
        if show_grid:
            plotter.show_grid()
        plotter.show()



    
    def visualize_vertices_property(
            self,
            property_key: Optional[str] = None,
            property_array: Optional[numpy.ndarray] = None,
            property_axis: Optional[int] = None,
            property_label: Optional[str] = None,
            cmap: str = "magma",
            vmin : Optional[float] = None,
            vmax : Optional[float] = None,
            use_log_scale: bool = False,
            vertices_color: str = "black",
            vertices_size: int = 5,
            vertices_opacity: float = 1.0,
            edges_color: str = "black",
            edges_width: int = 1,
            edges_opacity: float = 1.0,
            faces_opacity: float = 1.0,
            show_vertices: bool = True,
            show_edges: bool = True,
            title: Optional[str] = None,
            show_axes: bool = True,
            show_grid: bool = True,
        ) -> None:
        r"""
        Visualize a vertex property on the 3D surface mesh using PyVista (only for :math:`E \leq 3` surfacique meshes :math:`K=2`).

        This method creates a 3D plot of the mesh, displaying its vertices colored according to the specified property.
        The appearance of the vertices can be customized using various parameters.

        .. seealso::

            - :meth:`visualize` to visualize the mesh without coloring by a property.
            - :meth:`visualize_texture` to visualize the texture of the mesh.
            - :meth:`visualize_integration_points` to visualize integration points on the mesh.
            - :meth:`set_elements_type` to set the element type of the mesh and update the VTK cell type accordingly.

        Parameters
        ----------
        property_key : :class:`str`, optional
            The name of the vertex property to visualize. If None, :obj:`property_array` must be provided, by default None.

        property_array : :class:`numpy.ndarray`, optional
            A numpy ndarray of shape (:math:`N_v`, A) where :math:`N_v` is the number of vertices and A is the number of attributes for that property.
            If None, :obj:`property_key` must be provided, by default None.

        property_axis : int, optional
            The axis of :obj:`property_array` to visualize (0 for x, 1 for y, 2 for z). If None, the magnitude of the property will be visualized, by default None.

        property_label : :class:`str`, optional
            The label to use for the property in the visualization legend. If None, :obj:`property_key` will be used, by default None.

        cmap : :class:`str`, optional
            The colormap to use for coloring the vertices based on the property values, by default "magma".

        vmin : :class:`float`, optional
            The minimum value for the colormap, by default None.

        vmax : :class:`float`, optional
            The maximum value for the colormap, by default None.

        use_log_scale : :class:`bool`, optional
            Whether to use a logarithmic scale for the colormap, by default False.

        vertices_color : :class:`str`, optional
            Color of the vertices (points) in the mesh, by default :obj:`"black"`.

        vertices_size : :class:`int`, optional
            Size of the vertices (points) in the mesh, by default :obj:`5`.

        vertices_opacity : :class:`float`, optional
            Opacity of the vertices (points) in the mesh (0.0 to 1.0), by default :obj:`1.0`.

        edges_color : :class:`str`, optional
            Color of the edges in the mesh, by default :obj:`"black"`.

        edges_width : :class:`int`, optional
            Width of the edges in the mesh, by default :obj:`1`.

        edges_opacity : :class:`float`, optional
            Opacity of the edges in the mesh (0.0 to 1.0), by default :obj:`1.0`.

        faces_opacity : :class:`float`, optional
            Opacity of the faces in the mesh (0.0 to 1.0), by default :obj:`1.0`.

        show_points : :class:`bool`, optional
            Whether to display the vertices (points) of the mesh, by default :obj:`True`.

        show_edges : :class:`bool`, optional
            Whether to display the edges of the mesh, by default :obj:`True`.

        title : Optional[:class:`str`], optional
            Title of the plot, by default None.

        show_axes : :class:`bool`, optional
            Whether to display the axes in the plot, by default :obj:`True`.
            
        show_grid : :class:`bool`, optional
            Whether to display the grid in the plot, by default :obj:`True`.

            
        More Information
        -------------------------

        This method only display the mesh without additional elements.
        To display additional elements, use PyVista directly.

        .. seealso::

            - `PyVista Documentation <https://docs.pyvista.org>`_ for more details on visualization options.

            
        Examples
        --------

        Lets consider a simple linear triangular mesh in 3D, and visualize the height (z-coordinate) of each vertex:

        .. code-block:: python

            from pysdic import create_triangle_3_heightmap
            import numpy

            surface_mesh = create_triangle_3_heightmap(
                height_function=lambda x, y: 0.5 * numpy.sin(numpy.pi * x) * numpy.cos(numpy.pi * y),
                x_bounds=(-1.0, 1.0),
                y_bounds=(-1.0, 1.0),
                n_x=50,
                n_y=50,
            )
            
            height = surface_mesh.vertices.points[:, 2].reshape(-1, 1)  # Use the z-coordinate as a property

            surface_mesh.visualize_vertices_property(
                property_array=height, 
                property_label='Height [m]',
                property_axis=0,
                cmap='terrain'
                )

        .. figure:: /_static/meshes/mesh_visualize_vertices_property_example.png
           :width: 600
           :align: center

           Example of a 3D triangular mesh visualization using the `visualize_vertices_property` method.

        """
        # Check if visualizable
        if self.n_dimensions > 3:
            raise NotImplementedError("Visualization is only supported for meshes with embedding dimension E <= 3.")
        if self.n_dimensions == 3:
            mesh = self
        else:
            mesh = self.copy()
            mesh.vertices.extend_n_dimensions(3)

        expected_K = mesh._get_expected_K()
        if expected_K is not None and expected_K != 2:
            raise NotImplementedError("Visualization is only supported for surfacique meshes (K=2).")
        vtk_cell_type = mesh._get_vtk_cell_type()
        if vtk_cell_type is None:
            raise NotImplementedError(
                f"Visualization is not supported as vtk_cell_type is not defined for mesh elements. See method `set_elements_type` to define it."
            )
        
        
        # Case of an empty mesh
        if mesh.n_vertices == 0:
            raise ValueError("Cannot visualize an empty mesh.")
        if mesh.n_elements == 0:
            raise ValueError("Cannot visualize a mesh without elements.")
        
        # Extract the property array
        if (property_key is None and property_array is None) or (property_key is not None and property_array is not None):
            raise ValueError("Either property_key or property_array must be provided, but not both.")
        property_array = self._get_vertices_property(property_key, property_array, raise_error=True)

        property_array = numpy.asarray(property_array)
        if property_array.ndim == 1:
            property_array = property_array.reshape(-1, 1)
        if property_array.shape[0] != mesh.n_vertices:
            raise ValueError(f"property_array must have shape ({mesh.n_vertices}, A) where A is the number of attributes.")
        if property_array.shape[1] == 0:
            raise ValueError("property_array must have at least one attribute (shape (N_v, A) with A >= 1).")   
        
        # Default parameters
        if property_label is None and property_key is not None:
            property_label = property_key
        elif property_label is None:
            property_label = "property"

        # Extract the desired axis
        if property_axis is not None:
            if not isinstance(property_axis, int):
                raise ValueError("property_axis must be an integer.")
            if property_axis < 0 or property_axis >= property_array.shape[1]:
                raise ValueError(f"property_axis must be between 0 and {property_array.shape[1]-1}.")
            property_array = property_array[:, property_axis]
            property_label = f"{property_label} (Axis {property_axis})"
        else:
            # Use the magnitude of the property
            property_array = numpy.linalg.norm(property_array, axis=1)
            property_label = f"{property_label} (Magnitude)"
        # Now property_array is of shape (N,)    

        # Determine vmin and vmax if not provided
        if vmin is None:
            vmin = numpy.min(property_array)
        if vmax is None:
            vmax = numpy.max(property_array)

        # Input checks
        if not isinstance(cmap, str):
            raise ValueError("cmap must be a string.")
        if not isinstance(property_label, str):
            raise ValueError("property_label must be a string.")
        if not isinstance(show_edges, bool):
            raise ValueError("show_edges must be a boolean.")
        if not (isinstance(vmin, Number) and isinstance(vmax, Number)):
            raise ValueError("vmin and vmax must be numbers.")
        if vmin >= vmax:
            raise ValueError("vmin must be less than vmax.")
        if not isinstance(vertices_color, str):
            raise ValueError("Vertices color must be a string.")
        if not (isinstance(vertices_size, Number) and vertices_size > 0):
            raise ValueError("Vertices size must be a positive number.")
        if not (isinstance(vertices_opacity, Number) and 0.0 <= vertices_opacity <= 1.0):
            raise ValueError("Vertices opacity must be a float between 0.0 and 1.0.")
        if not isinstance(edges_color, str):
            raise ValueError("Edges color must be a string.")
        if not (isinstance(edges_width, Number) and edges_width > 0):
            raise ValueError("Edges width must be a positive number.")
        if not (isinstance(edges_opacity, Number) and 0.0 <= edges_opacity <= 1.0):
            raise ValueError("Edges opacity must be a float between 0.0 and 1.0.")
        if not (isinstance(faces_opacity, Number) and 0.0 <= faces_opacity <= 1.0):
            raise ValueError("Faces opacity must be a float between 0.0 and 1.0.")
        if not isinstance(show_vertices, bool):
            raise ValueError("show_vertices must be a boolean.")
        if not isinstance(show_edges, bool):
            raise ValueError("show_edges must be a boolean.")
        if not isinstance(use_log_scale, bool):
            raise ValueError("use_log_scale must be a boolean.")
        if not isinstance(show_axes, bool):
            raise ValueError("show_axis must be a boolean.")
        if not isinstance(show_grid, bool):
            raise ValueError("show_grid must be a boolean.")
        if title is not None and not isinstance(title, str):
            raise ValueError("Title must be a string.")
    
        # Extract the cmap
        colormaps = plt.colormaps()
        if not cmap in colormaps:
            raise ValueError(f"cmap '{cmap}' is not a valid colormap. Available colormaps are: {colormaps}")

        # Create a PyVista mesh
        n_cells = mesh.n_elements

        cells = numpy.hstack([numpy.full((n_cells, 1), mesh.n_vertices_per_element), mesh.connectivity]).astype(numpy.int64).ravel()
        celltypes = numpy.full(n_cells, vtk_cell_type, dtype=numpy.uint8)

        pv_mesh = pyvista.UnstructuredGrid(cells, celltypes, mesh.vertices.points)

        # Add the property as point data
        pv_mesh.point_data[property_label] = property_array

        # Create a PyVista plotter
        plotter = pyvista.Plotter()

        # Add the mesh with the property colormap
        plotter.add_mesh(
            pv_mesh, 
            scalars=property_label, 
            cmap=cmap,
            clim=(vmin, vmax),
            log_scale=use_log_scale,
            opacity=faces_opacity,
        )

        # Add edges if required
        if show_edges:
            edges = pv_mesh.extract_all_edges()
            plotter.add_mesh(
                edges, 
                color=edges_color, 
                line_width=edges_width,
                opacity=edges_opacity
            )

        # Add vertices if required
        if show_vertices:
            plotter.add_points(
                self.vertices.points, 
                color=vertices_color, 
                point_size=vertices_size,
                opacity=vertices_opacity,
                render_points_as_spheres=True
            )

        # Show the plot
        if title is not None:
            plotter.add_title(title)
        if show_axes:
            plotter.show_axes()
        if show_grid:
            plotter.show_grid()
        plotter.show()



    def visualize_elements_property(
            self,
            property_key: Optional[str] = None,
            property_array: Optional[numpy.ndarray] = None,
            property_axis: Optional[int] = None,
            property_label: Optional[str] = None,
            cmap: str = "magma",
            vmin : Optional[float] = None,
            vmax : Optional[float] = None,
            use_log_scale: bool = False,
            vertices_color: str = "black",
            vertices_size: int = 5,
            vertices_opacity: float = 1.0,
            edges_color: str = "black",
            edges_width: int = 1,
            edges_opacity: float = 1.0,
            faces_opacity: float = 1.0,
            show_vertices: bool = True,
            show_edges: bool = True,
            title: Optional[str] = None,
            show_axes: bool = True,
            show_grid: bool = True,
        ) -> None:
        r"""
        Visualize an element property on the 3D surface mesh using PyVista (only for :math:`E \leq 3` surfacique meshes :math:`K=2`).

        This method creates a 3D plot of the mesh, displaying its faces colored according to the specified property.

        .. seealso::

            - :meth:`visualize` to visualize the mesh without coloring by a property.
            - :meth:`visualize_texture` to visualize the texture of the mesh.
            - :meth:`visualize_integration_points` to visualize integration points on the mesh.
            - :meth:`set_elements_type` to set the element type of the mesh and update the VTK cell type accordingly.

        Parameters
        ----------
        property_key : :class:`str`, optional
            The name of the element property to visualize. If None, :obj:`property_array` must be provided, by default None.

        property_array : :class:`numpy.ndarray`, optional
            A numpy ndarray of shape (:math:`N_e`, A) where :math:`N_e` is the number of elements and A is the number of attributes for that property.
            If None, :obj:`property_key` must be provided, by default None.

        property_axis : int, optional
            The axis of :obj:`property_array` to visualize (0 for x, 1 for y, 2 for z). If None, the magnitude of the property will be visualized, by default None.

        property_label : :class:`str`, optional
            The label to use for the property in the visualization legend. If None, :obj:`property_key` will be used, by default None.

        cmap : :class:`str`, optional
            The colormap to use for coloring the faces based on the property values, by default "magma".

        vmin : :class:`float`, optional
            The minimum value for the colormap, by default None.
        
        vmax : :class:`float`, optional
            The maximum value for the colormap, by default None.

        use_log_scale : :class:`bool`, optional
            Whether to use a logarithmic scale for the colormap, by default False.

        vertices_color : :class:`str`, optional
            Color of the vertices (points) in the mesh, by default :obj:`"black"`.

        vertices_size : :class:`int`, optional
            Size of the vertices (points) in the mesh, by default :obj:`5`.

        vertices_opacity : :class:`float`, optional
            Opacity of the vertices (points) in the mesh (0.0 to 1.0), by default :obj:`1.0`.

        edges_color : :class:`str`, optional
            Color of the edges in the mesh, by default :obj:`"black"`.

        edges_width : :class:`int`, optional
            Width of the edges in the mesh, by default :obj:`1`.

        edges_opacity : :class:`float`, optional
            Opacity of the edges in the mesh (0.0 to 1.0), by default :obj:`1.0`.

        faces_opacity : :class:`float`, optional
            Opacity of the faces in the mesh (0.0 to 1.0), by default :obj:`1.0`.

        show_points : :class:`bool`, optional
            Whether to display the vertices (points) of the mesh, by default :obj:`True`.

        show_edges : :class:`bool`, optional
            Whether to display the edges of the mesh, by default :obj:`True`.

        title : Optional[:class:`str`], optional
            Title of the plot, by default None.

        show_axes : :class:`bool`, optional
            Whether to display the axes in the plot, by default :obj:`True`.

        show_grid : :class:`bool`, optional
            Whether to display the grid in the plot, by default :obj:`True`.

            
        More Information
        -------------------------

        This method only display the mesh without additional elements.
        To display additional elements, use PyVista directly.

        .. seealso::
        
            - `PyVista Documentation <https://docs.pyvista.org>`_ for more details on visualization options.

        
        Examples
        --------

        Lets consider a simple linear triangular mesh in 3D, and visualize the area of each element:

        .. code-block:: python

            from pysdic import create_triangle_3_heightmap, triangle_3_compute_elements_areas
            import numpy

            surface_mesh = create_triangle_3_heightmap(
                height_function=lambda x, y: 0.5 * numpy.sin(numpy.pi * x) * numpy.cos(numpy.pi * y),
                x_bounds=(-1.0, 1.0),
                y_bounds=(-1.0, 1.0),
                n_x=50,
                n_y=50,
            )
            
            areas = triangle_3_compute_elements_areas(
                surface_mesh.vertices.points,
                surface_mesh.connectivity,
            )  # Compute the area of each triangle element (shape (N_e,))

            surface_mesh.visualize_elements_property(
                property_array=areas.reshape(-1, 1), 
                property_label='Element Area [m²]',
                property_axis=0,
                cmap='viridis'
                )

        
        .. figure:: /_static/meshes/mesh_visualize_elements_property_example.png
              :width: 600
              :align: center
    
              Example of a 3D triangular mesh visualization using the `visualize_elements_property` method.
    
        """
        # Check if visualizable
        if self.n_dimensions > 3:
            raise NotImplementedError("Visualization is only supported for meshes with embedding dimension E <= 3.")
        if self.n_dimensions == 3:
            mesh = self
        else:
            mesh = self.copy()
            mesh.vertices.extend_n_dimensions(3)

        expected_K = mesh._get_expected_K()
        if expected_K is not None and expected_K != 2:
            raise NotImplementedError("Visualization is only supported for surfacique meshes (K=2).")
        vtk_cell_type = mesh._get_vtk_cell_type()
        if vtk_cell_type is None:
            raise NotImplementedError(
                f"Visualization is not supported as vtk_cell_type is not defined for mesh elements. See method `set_elements_type` to define it."
            )
        
        # Case of an empty mesh
        if mesh.n_vertices == 0:
            raise ValueError("Cannot visualize an empty mesh.")
        if mesh.n_elements == 0:
            raise ValueError("Cannot visualize a mesh without elements.")
        
        # Extract the property array
        if (property_key is None and property_array is None) or (property_key is not None and property_array is not None):
            raise ValueError("Either property_key or property_array must be provided, but not both.")
        property_array = self._get_elements_property(property_key, property_array, raise_error=True)
        property_array = numpy.asarray(property_array)
        if property_array.ndim == 1:
            property_array = property_array.reshape(-1, 1)
        if property_array.shape[0] != mesh.n_elements:
            raise ValueError(f"property_array must have shape ({mesh.n_elements}, A) where A is the number of attributes.")
        if property_array.shape[1] == 0:
            raise ValueError("property_array must have at least one attribute (shape (N_e, A) with A >= 1).")
        
        # Default parameters
        if property_label is None and property_key is not None:
            property_label = property_key
        elif property_label is None:
            property_label = "property"

        # Extract the desired axis
        if property_axis is not None:
            if not isinstance(property_axis, int):
                raise ValueError("property_axis must be an integer.")
            if property_axis < 0 or property_axis >= property_array.shape[1]:
                raise ValueError(f"property_axis must be between 0 and {property_array.shape[1]-1}.")
            property_array = property_array[:, property_axis]
            property_label = f"{property_label} (Axis {property_axis})"
        else:
            # Use the magnitude of the property
            property_array = numpy.linalg.norm(property_array, axis=1)
            property_label = f"{property_label} (Magnitude)"
        # Now property_array is of shape (N,)

        # Determine vmin and vmax if not provided
        if vmin is None:
            vmin = numpy.min(property_array)
        if vmax is None:
            vmax = numpy.max(property_array)
        
        # Input checks
        if not isinstance(cmap, str):
            raise ValueError("cmap must be a string.")
        if not isinstance(property_label, str):
            raise ValueError("property_label must be a string.")
        if not isinstance(show_edges, bool):
            raise ValueError("show_edges must be a boolean.")
        if not (isinstance(vmin, Number) and isinstance(vmax, Number)):
            raise ValueError("vmin and vmax must be numbers.")
        if vmin >= vmax:
            raise ValueError("vmin must be less than vmax.")
        if not isinstance(vertices_color, str):
            raise ValueError("Vertices color must be a string.")
        if not (isinstance(vertices_size, Number) and vertices_size > 0):
            raise ValueError("Vertices size must be a positive number.")
        if not (isinstance(vertices_opacity, Number) and 0.0 <= vertices_opacity <= 1.0):
            raise ValueError("Vertices opacity must be a float between 0.0 and 1.0.")
        if not isinstance(edges_color, str):
            raise ValueError("Edges color must be a string.")
        if not (isinstance(edges_width, Number) and edges_width > 0):
            raise ValueError("Edges width must be a positive number.")
        if not (isinstance(edges_opacity, Number) and 0.0 <= edges_opacity <= 1.0):
            raise ValueError("Edges opacity must be a float between 0.0 and 1.0.")
        if not (isinstance(faces_opacity, Number) and 0.0 <= faces_opacity <= 1.0):
            raise ValueError("Faces opacity must be a float between 0.0 and 1.0.")
        if not isinstance(show_vertices, bool):
            raise ValueError("show_vertices must be a boolean.")
        if not isinstance(show_edges, bool):
            raise ValueError("show_edges must be a boolean.")
        if not isinstance(use_log_scale, bool):
            raise ValueError("use_log_scale must be a boolean.")
        if not isinstance(show_axes, bool):
            raise ValueError("show_axis must be a boolean.")
        if not isinstance(show_grid, bool):
            raise ValueError("show_grid must be a boolean.")
        if title is not None and not isinstance(title, str):
            raise ValueError("Title must be a string.")
        
        # Extract the cmap
        colormaps = plt.colormaps()
        if not cmap in colormaps:
            raise ValueError(f"cmap '{cmap}' is not a valid colormap. Available colormaps are: {colormaps}")
        
        # Create a PyVista mesh
        n_cells = mesh.n_elements
        cells = numpy.hstack([numpy.full((n_cells, 1), mesh.n_vertices_per_element), mesh.connectivity]).astype(numpy.int64).ravel()
        celltypes = numpy.full(n_cells, vtk_cell_type, dtype=numpy.uint8)

        pv_mesh = pyvista.UnstructuredGrid(cells, celltypes, mesh.vertices.points)

        # Add the property as cell data
        pv_mesh.cell_data[property_label] = property_array

        # Create a PyVista plotter
        plotter = pyvista.Plotter()

        # Add the mesh with the property colormap
        plotter.add_mesh(
            pv_mesh, 
            scalars=property_label, 
            cmap=cmap,
            clim=(vmin, vmax),
            log_scale=use_log_scale,
            opacity=faces_opacity,
        )

        # Add edges if required
        if show_edges:
            edges = pv_mesh.extract_all_edges()
            plotter.add_mesh(
                edges, 
                color=edges_color, 
                line_width=edges_width,
                opacity=edges_opacity
            )

        # Add vertices if required
        if show_vertices:
            plotter.add_points(
                self.vertices.points, 
                color=vertices_color, 
                point_size=vertices_size,
                opacity=vertices_opacity,
                render_points_as_spheres=True
            )

        # Show the plot
        if title is not None:
            plotter.add_title(title)
        if show_axes:
            plotter.show_axes()
        if show_grid:
            plotter.show_grid()
        plotter.show()


    def visualize_texture(
            self,
            texture: numpy.ndarray,
            use_rgb: bool = True,
            vertices_color: str = "black",
            vertices_size: int = 5,
            vertices_opacity: float = 1.0,
            edges_color: str = "black",
            edges_width: int = 1,
            edges_opacity: float = 1.0,
            faces_opacity: float = 1.0,
            show_vertices: bool = True,
            show_edges: bool = True,
            title: Optional[str] = None,
            show_axes: bool = True,
            show_grid: bool = True,
        ) -> None:
        r"""
        Visualize the texture of the mesh using a texture image and PyVista (only for :math:`E \leq 3` surfacique meshes :math:`K=2`).

        .. warning::

            The mesh must have the :obj:`uvmap` property set for this method to work.

        .. seealso::

            - :meth:`elements_uvmap` to set or get the UV mapping of the elements.

        This method creates a 3D plot of the mesh, displaying its faces textured with the provided image.
        The texture image should be a 2D (grayscale) or 3D (RGB) numpy array.

        .. seealso::

            - :meth:`visualize` to visualize the mesh without texture.
            - :meth:`visualize_vertices_property` to visualize a vertex property on the mesh.
            - :meth:`visualize_integration_points` to visualize integration points on the mesh.
            - :meth:`set_elements_type` to set the element type of the mesh and update the VTK cell type accordingly.

        Parameters
        ----------
        texture : :class:`numpy.ndarray`
            The texture image to apply to the mesh. Integer arrays with values in [0, 255] with dtype ``numpy.uint8``.
            Array must have shape (height, width, 3) for RGB textures or (height, width) for grayscale textures.

        use_rgb : :class:`bool`, optional
            Whether to interpret the texture as RGB (:obj:`True`). If :obj:`False`, any RGB texture will be converted to grayscale, by default :obj:`True`.

        vertices_color : :class:`str`, optional
            Color of the vertices (points) in the mesh, by default :obj:`"black"`.

        vertices_size : :class:`int`, optional
            Size of the vertices (points) in the mesh, by default :obj:`5`.

        vertices_opacity : :class:`float`, optional
            Opacity of the vertices (points) in the mesh (0.0 to 1.0), by default :obj:`1.0`.

        edges_color : :class:`str`, optional
            Color of the edges in the mesh, by default :obj:`"black"`.

        edges_width : :class:`int`, optional
            Width of the edges in the mesh, by default :obj:`1`.

        edges_opacity : :class:`float`, optional
            Opacity of the edges in the mesh (0.0 to 1.0), by default :obj:`1.0`.

        faces_opacity : :class:`float`, optional
            Opacity of the faces in the mesh (0.0 to 1.0), by default :obj:`1.0`.

        show_points : :class:`bool`, optional
            Whether to display the vertices (points) of the mesh, by default :obj:`True`.

        show_edges : :class:`bool`, optional
            Whether to display the edges of the mesh, by default :obj:`True`.

        title : Optional[:class:`str`], optional
            Title of the plot, by default None.

        show_axes : :class:`bool`, optional
            Whether to display the axes in the plot, by default :obj:`True`.

        show_grid : :class:`bool`, optional
            Whether to display the grid in the plot, by default :obj:`True`.


        More Information
        -------------------------
        This method only display the mesh without additional elements.
        To display additional elements, use PyVista directly.

        .. seealso::
        
            - `PyVista Documentation <https://docs.pyvista.org>`_ for more details on visualization options.

            
        Examples
        --------

        Lets consider a simple linear triangular mesh in 3D, and visualize a checkerboard texture on it:

        .. code-block:: python

            from pysdic import create_triangle_3_heightmap
            import numpy

            surface_mesh = create_triangle_3_heightmap(
                height_function=lambda x, y: 0.5 * numpy.sin(numpy.pi * x) * numpy.cos(numpy.pi * y),
                x_bounds=(-1.0, 1.0),
                y_bounds=(-1.0, 1.0),
                n_x=50,
                n_y=50,
            ) # UVMAP already set in the function

            # Create a texture image
            u = numpy.linspace(0, 1, 50)
            v = numpy.linspace(0, 1, 50)
            U, V = numpy.meshgrid(u, v)

            texture_image = numpy.round(255/2 + 255/2 * numpy.sin(U * 4 * numpy.pi)).astype(numpy.uint8)  # Example texture image with shape (50, 50)

            surface_mesh.visualize_texture(texture_image, show_edges=False, show_vertices=False)
            
        .. figure:: /_static/meshes/mesh_visualize_texture_example.png
            :width: 600
            :align: center

            Example of a 3D triangular mesh visualization using the `visualize_texture` method.

        """
        # Check if visualizable
        if self.n_dimensions > 3:
            raise NotImplementedError("Visualization is only supported for meshes with embedding dimension E <= 3.")
        if self.n_dimensions == 3:
            mesh = self
        else:
            mesh = self.copy()
            mesh.vertices.extend_n_dimensions(3)

        expected_K = mesh._get_expected_K()
        if expected_K is not None and expected_K != 2:
            raise NotImplementedError("Visualization is only supported for surfacique meshes (K=2).")
        vtk_cell_type = mesh._get_vtk_cell_type()
        if vtk_cell_type is None:
            raise NotImplementedError(
                f"Visualization is not supported as vtk_cell_type is not defined for mesh elements. See method `set_elements_type` to define it."
            )

        # Check input data
        if mesh.n_vertices == 0:
            raise ValueError("Cannot visualize an empty mesh.")
        if mesh.n_elements == 0:
            raise ValueError("Cannot visualize a mesh without elements.")
        if mesh.elements_uvmap is None:
            raise ValueError("The mesh must have the 'uvmap' property set to visualize texture.")
        
        if not isinstance(texture, numpy.ndarray):
            raise ValueError("texture must be a numpy ndarray.")
        
        if texture.ndim < 2 or texture.ndim > 3:
            raise ValueError("texture must be a 2D (grayscale) or 3D (RGB) array.")
        if texture.ndim == 3 and texture.shape[2] not in [1, 3]:
            raise ValueError("If texture is 3D, its third dimension must be 1 (grayscale) or 3 (RGB).")
        if texture.dtype != numpy.uint8:
            raise ValueError("texture array must have dtype numpy.uint8 with values in [0, 255].")
        
        if not isinstance(use_rgb, bool):
            raise ValueError("use_rgb must be a boolean.")

        if not isinstance(vertices_color, str):
            raise ValueError("Vertices color must be a string.")
        if not (isinstance(vertices_size, Number) and vertices_size > 0):
            raise ValueError("Vertices size must be a positive number.")
        if not (isinstance(vertices_opacity, Number) and 0.0 <= vertices_opacity <= 1.0):
            raise ValueError("Vertices opacity must be a float between 0.0 and 1.0.")
        if not isinstance(edges_color, str):
            raise ValueError("Edges color must be a string.")
        if not (isinstance(edges_width, Number) and edges_width > 0):
            raise ValueError("Edges width must be a positive number.")
        if not (isinstance(edges_opacity, Number) and 0.0 <= edges_opacity <= 1.0):
            raise ValueError("Edges opacity must be a float between 0.0 and 1.0.")
        if not (isinstance(faces_opacity, Number) and 0.0 <= faces_opacity <= 1.0):
            raise ValueError("Faces opacity must be a float between 0.0 and 1.0.")
        
        if not isinstance(show_vertices, bool):
            raise ValueError("show_vertices must be a boolean.")
        if not isinstance(show_edges, bool):
            raise ValueError("show_edges must be a boolean.")
        
        if not isinstance(show_axes, bool):
            raise ValueError("show_axis must be a boolean.")
        if not isinstance(show_grid, bool):
            raise ValueError("show_grid must be a boolean.")
        
        if title is not None and not isinstance(title, str):
            raise ValueError("Title must be a string.")

        # Duplicate points per face
        fictive_vertices = numpy.zeros((mesh.n_elements * mesh.n_vertices_per_element, 3), dtype=numpy.float64)
        for i in range(mesh.n_vertices_per_element):
            fictive_vertices[i::mesh.n_vertices_per_element, :] = mesh.vertices.points[mesh.connectivity[:, i], :]

        # Create connectivity for the fictive vertices
        fictive_connectivity = numpy.arange(mesh.n_elements * mesh.n_vertices_per_element, dtype=numpy.int64).reshape(mesh.n_elements, mesh.n_vertices_per_element)

        # Create a PyVista mesh
        n_cells = fictive_connectivity.shape[0]

        cells = numpy.hstack([numpy.full((n_cells, 1), mesh.n_vertices_per_element), fictive_connectivity]).astype(numpy.int64).ravel()
        celltypes = numpy.full(n_cells, vtk_cell_type, dtype=numpy.uint8)

        pv_mesh = pyvista.UnstructuredGrid(cells, celltypes, fictive_vertices)
        
        # Set texture coordinates
        pv_mesh.active_texture_coordinates = numpy.zeros((mesh.n_elements * mesh.n_vertices_per_element, 2), dtype=numpy.float64)

        # UV coordinates per vertex of each element
        uvmap = mesh.elements_uvmap  # shape (M, 6)
        for i in range(mesh.n_vertices_per_element):
            pv_mesh.active_texture_coordinates[i::mesh.n_vertices_per_element, 0] = uvmap[:, 2 * i]      # u_i
            pv_mesh.active_texture_coordinates[i::mesh.n_vertices_per_element, 1] = uvmap[:, 2 * i + 1]  # v_i

        # Create a PyVista texture
        if texture.ndim == 2:
            color_texture = numpy.repeat(texture[:, :, numpy.newaxis], 3, axis=2).astype(numpy.uint8)
        elif texture.ndim == 3 and texture.shape[2] == 1:
            color_texture = numpy.repeat(texture, 3, axis=2).astype(numpy.uint8)
        elif texture.ndim == 3 and use_rgb and texture.shape[2] == 3:
            color_texture = texture
        elif texture.ndim == 3 and not use_rgb and texture.shape[2] == 3:
            gray_texture = numpy.round(numpy.dot(texture[..., :3], [0.2989, 0.5870, 0.1140])).astype(numpy.uint8)
            color_texture = numpy.repeat(gray_texture[:, :, numpy.newaxis], 3, axis=2).astype(numpy.uint8)
        else:
            raise ValueError("Invalid texture array shape.")
        
        pvtexture = pyvista.Texture(color_texture)

        # Create a PyVista plotter
        plotter = pyvista.Plotter()

        # Add the mesh with the texture
        plotter.add_mesh(
            pv_mesh, 
            texture=pvtexture,
            opacity=faces_opacity,
        )

        # Add edges if required
        if show_edges:
            edges = pv_mesh.extract_all_edges()
            plotter.add_mesh(
                edges, 
                color=edges_color, 
                line_width=edges_width,
                opacity=edges_opacity
            )

        # Add vertices if required
        if show_vertices:
            plotter.add_points(
                self.vertices.points, 
                color=vertices_color, 
                point_size=vertices_size,
                opacity=vertices_opacity,
                render_points_as_spheres=True
            )

        # Show the plot
        if title is not None:
            plotter.add_title(title)
        if show_axes:
            plotter.show_axes()
        if show_grid:
            plotter.show_grid()
        plotter.show()



    def visualize_integration_points(
            self,
            natural_coordinates: numpy.ndarray,
            element_indices: numpy.ndarray,
            points_color: str = "red",
            points_size: int = 5,
            points_opacity: float = 1.0,
            vertices_color: str = "black",
            vertices_size: int = 5,
            vertices_opacity: float = 1.0,
            edges_color: str = "black",
            edges_width: int = 1,
            edges_opacity: float = 1.0,
            faces_color: str = "gray",   
            faces_opacity: float = 0.5,
            show_vertices: bool = True,
            show_edges: bool = True,
            show_faces: bool = True,
            title: Optional[str] = None,
            show_axes: bool = True,
            show_grid: bool = True,
        ) -> None:
        r"""
        Visualize the 3D surface mesh with integration points using PyVista (only for :math:`E \leq 3` surfacique meshes :math:`K=2`).

        This method creates a 3D plot of the mesh, displaying its vertices, edges, and faces,
        along with the integration points overlaid on the mesh.

        .. seealso::

            - :meth:`visualize` to visualize the mesh without integration points.
            - :meth:`visualize_vertices_property` to visualize a vertex property on the mesh.
            - :meth:`visualize_texture` to visualize the texture of the mesh.
            - :meth:`set_elements_type` to set the element type of the mesh and update the VTK cell type accordingly.

            
        Parameters
        ----------
        natural_coordinates : :class:`numpy.ndarray`
            A numpy ndarray of shape (:math:`N_p`, :math:`K`) containing the natural coordinates of the integration points,
            where :math:`N_p` is the number of integration points and :math:`K` is the dimension of the elements (should be 2 for surfacique meshes).

        element_indices : :class:`numpy.ndarray`
            A numpy ndarray of shape (:math:`N_p`,) containing the indices of the elements on which each integration point is located.

        points_color : str, optional
            Color of the integration points, by default "red".

        points_size : int, optional
            Size of the integration points, by default 5.
        
        points_opacity : float, optional
            Opacity of the integration points (0.0 to 1.0), by default 1.0.

        vertices_color : str, optional
            Color of the vertices (points) in the mesh, by default "black".

        vertices_size : int, optional
            Size of the vertices (points) in the mesh, by default 5.

        vertices_opacity : float, optional
            Opacity of the vertices (points) in the mesh (0.0 to 1.0), by default 1.0.

        edges_color : str, optional
            Color of the edges in the mesh, by default "black".

        edges_width : int, optional
            Width of the edges in the mesh, by default 1.

        edges_opacity : float, optional
            Opacity of the edges in the mesh (0.0 to 1.0), by default 1.0.

        faces_color : str, optional
            Color of the faces in the mesh, by default "gray".

        faces_opacity : float, optional
            Opacity of the faces in the mesh (0.0 to 1.0), by default 0.5.

        show_points : bool, optional
            Whether to display the vertices (points) of the mesh, by default True.

        show_edges : bool, optional
            Whether to display the edges of the mesh, by default True.

        show_faces : bool, optional
            Whether to display the faces of the mesh, by default True.

        title : Optional[str], optional
            Title of the plot, by default None.

        show_axes : bool, optional
            Whether to display the axes in the plot, by default True.

        show_grid : bool, optional
            Whether to display the grid in the plot, by default True.

            
        More Information
        -------------------------

        This method only display the mesh and integration points without additional elements.
        To display additional elements, use PyVista directly.

        .. seealso::

            - `PyVista Documentation <https://docs.pyvista.org>`_ for more details on visualization options.

        
        Examples
        --------

        Lets consider a simple linear triangular mesh in 3D, and visualize some intersection points on it:

        .. code-block:: python

            from pysdic import create_linear_triangle_heightmap
            import numpy

            surface_mesh = create_linear_triangle_heightmap(
                height_function=lambda x, y: 0.5 * numpy.sin(numpy.pi * x) * numpy.cos(numpy.pi * y),
                x_bounds=(-1.0, 1.0),
                y_bounds=(-1.0, 1.0),
                n_x=50,
                n_y=50,
            )

            # Create some rays to cast
            ray_origins = numpy.random.uniform(-1, 1, (100, 3))
            ray_origins[:, 2] = 3.0  # Start above the surface
            ray_directions = numpy.tile(numpy.array([[0, 0, -1]]), (100, 1))  # Pointing downwards

            intersection_points = surface_mesh.cast_rays(ray_origins, ray_directions)

            surface_mesh.visualize_integration_points(intersection_points, points_size=8)

            
        .. figure:: /_static/meshes/mesh_visualize_integration_points_example.png
            :width: 600
            :align: center

            Example of a 3D triangular mesh visualization using the `visualize_integration_points` method.

        """
        # Check input data
        if self.n_vertices == 0:
            raise ValueError("Cannot visualize an empty mesh.")
        if self.n_elements == 0:
            raise ValueError("Cannot visualize a mesh without elements.")
        
        expected_K = self._get_expected_K()
        if expected_K is not None and expected_K != 2:
            raise NotImplementedError("Visualization is only supported for surfacique meshes (K=2).")
        vtk_cell_type = self._get_vtk_cell_type()
        if vtk_cell_type is None:
            raise NotImplementedError(
                f"Visualization is not supported as vtk_cell_type is not defined for mesh elements. See method `set_elements_type` to define it."
            )
        
        natural_coordinates = numpy.asarray(natural_coordinates, dtype=numpy.float64)
        element_indices = numpy.asarray(element_indices, dtype=numpy.int64)
        if natural_coordinates.ndim != 2:
            raise ValueError("natural_coordinates must be a 2D array of shape (N_p, K).")
        if natural_coordinates.shape[1] != 2:
            raise ValueError("natural_coordinates must have K=2 columns for surfacique meshes.")
        N_p = natural_coordinates.shape[0]
        if element_indices.ndim != 1:
            raise ValueError("element_indices must be a 1D array of shape (N_p,).")
        if element_indices.shape[0] != N_p:
            raise ValueError("element_indices must have the same number of rows as natural_coordinates.")
        
        if not isinstance(points_color, str):
            raise ValueError("Points color must be a string.")
        if not (isinstance(points_size, Number) and points_size > 0):
            raise ValueError("Points size must be a positive number.")
        if not (isinstance(points_opacity, Number) and 0.0 <= points_opacity <= 1.0):
            raise ValueError("Points opacity must be a float between 0.0 and 1.0.")
        
        if not isinstance(vertices_color, str):
            raise ValueError("Vertices color must be a string.")
        if not (isinstance(vertices_size, Number) and vertices_size > 0):
            raise ValueError("Vertices size must be a positive number.")
        if not (isinstance(vertices_opacity, Number) and 0.0 <= vertices_opacity <= 1.0):
            raise ValueError("Vertices opacity must be a float between 0.0 and 1.0.")
        if not isinstance(edges_color, str):
            raise ValueError("Edges color must be a string.")
        if not (isinstance(edges_width, Number) and edges_width > 0):
            raise ValueError("Edges width must be a positive number.")
        if not (isinstance(edges_opacity, Number) and 0.0 <= edges_opacity <= 1.0):
            raise ValueError("Edges opacity must be a float between 0.0 and 1.0.")
        if not isinstance(faces_color, str):
            raise ValueError("Faces color must be a string.")
        if not (isinstance(faces_opacity, Number) and 0.0 <= faces_opacity <= 1.0):
            raise ValueError("Faces opacity must be a float between 0.0 and 1.0.")
        
        if not isinstance(show_vertices, bool):
            raise ValueError("show_vertices must be a boolean.")
        if not isinstance(show_edges, bool):
            raise ValueError("show_edges must be a boolean.")
        if not isinstance(show_faces, bool):
            raise ValueError("show_faces must be a boolean.")
        
        if not isinstance(show_axes, bool):
            raise ValueError("show_axis must be a boolean.")
        if not isinstance(show_grid, bool):
            raise ValueError("show_grid must be a boolean.")
        
        if title is not None and not isinstance(title, str):
            raise ValueError("Title must be a string.")
        
        # Create a PyVista mesh
        n_cells = self.n_elements

        cells = numpy.hstack([numpy.full((n_cells, 1), self.n_vertices_per_element), self.connectivity]).astype(numpy.int64).ravel()
        celltypes = numpy.full(n_cells, vtk_cell_type, dtype=numpy.uint8)

        pv_mesh = pyvista.UnstructuredGrid(cells, celltypes, self.vertices.points)

        # Create a PyVista plotter
        plotter = pyvista.Plotter()

        # Add faces if required
        if show_faces:
            plotter.add_mesh(
                pv_mesh, 
                color=faces_color, 
                opacity=faces_opacity
            )

        # Add edges if required
        if show_edges:
            edges = pv_mesh.extract_all_edges()
            plotter.add_mesh(
                edges, 
                color=edges_color, 
                line_width=edges_width,
                opacity=edges_opacity
            )

        # Add vertices if required
        if show_vertices:
            plotter.add_points(
                self.vertices.points, 
                color=vertices_color, 
                point_size=vertices_size,
                opacity=vertices_opacity,
                render_points_as_spheres=True
            )

        # Add integration points
        shape_functions = self.shape_functions(natural_coordinates)  # shape (N_p, n_vertices_per_element)
        points_coordinates = interpolate_property(
            self.vertices.points, 
            shape_functions, 
            self.connectivity,
            element_indices,
            skip_m1=True
        )

        plotter.add_points(
            points_coordinates, 
            color=points_color, 
            point_size=points_size,
            opacity=points_opacity,
            render_points_as_spheres=True
        )

        # Show the plot
        if title is not None:
            plotter.add_title(title)
        if show_axes:
            plotter.show_axes()
        if show_grid:
            plotter.show_grid()
        plotter.show()

















