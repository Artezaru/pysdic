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
from typing import Optional, Tuple, Union

import numpy

from .surface_mesh_3d import SurfaceMesh3D
from .point_cloud_3d import PointCloud3D
from .linear_triangle_mesh_3d import LinearTriangleMesh3D


class QuadraticTriangleMesh3D(SurfaceMesh3D):
    r"""
    Subclass of :class:`pysdic.geometry.Mesh3D` and :class:`pysdic.geometry.SurfaceMesh3D` representing a 3D mesh composed of quadratic triangular elements.

    The vertices are represented as a :class:`pysdic.geometry.PointCloud3D` instance with shape (N, 3),
    where N is the number of vertices. Each vertex has 3 coordinates (x, y, z).
    The elements are represented by a numpy ndarray with shape (M, 6),
    where M is the number of triangular elements and each element is defined by 6 vertex indices.

    The coordinates of a point into the mesh can be accessed by the natural coordinates
    in the reference element. The natural coordinates (:math:`\xi, \eta`) for a linear triangle satisfy:

    - :math:`0 <= \xi <= 1`
    - :math:`0 <= \eta <= 1`
    - :math:`\xi + \eta <= 1`

    We have K=6 vertices per element, and d=2 for the dimensions of the natural coordinates.

    Lets :math:`X` be the coordinates of a point in the mesh. The transformation from natural coordinates to global coordinates is given by:

    .. math::

        X = \sum_{i=1}^{K} N_i(\xi, \eta) X_i

    where :math:`N_i` are the shape functions associated with each vertex, and :math:`X_i` are the coordinates of the vertices of the element.

    The shape functions for a linear triangle are defined as:

    .. math::

        N_1(\xi, \eta) = 2 * (1 - \xi - \eta) * (1/2 - \xi - \eta)

    .. math::

        N_2(\xi, \eta) = 2 * \xi * (\xi - 1/2)

    .. math::

        N_3(\xi, \eta) = 2 * \eta * (\eta - 1/2)

    .. math::

        N_4(\xi, \eta) = 4 * \xi * (1 - \xi - \eta)

    .. math::

        N_5(\xi, \eta) = 4 * \xi * \eta

    .. math::

        N_6(\xi, \eta) = 4 * \eta * (1 - \xi - \eta)

    .. figure:: ../../../pysdic/resources/quadratic_triangle_reference_element.png
        :alt: Reference element for a quadratic triangle

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
    __slots__ = ["_vertices_predefined_metadata", "_elements_predefined_metadata"]

    _n_vertices_per_element: int = 6
    _n_dimensions: int = 2
    _meshio_cell_type: str = "triangle6"
    _vtk_cell_type: int = 22  # VTK_QUADRATIC_TRIANGLE
    _is_visualizable: bool = True

    def __init__(self, vertices: PointCloud3D, connectivity: numpy.ndarray, *, vertices_properties: dict = None, elements_properties: dict = None, internal_bypass: bool = False):
        # Define expected properties informations
        if not hasattr(self, "_vertices_predefined_metadata"):
            self._vertices_predefined_metadata = {}
        if not hasattr(self, "_elements_predefined_metadata"):
            self._elements_predefined_metadata = {}

        super().__init__(vertices, connectivity, vertices_properties=vertices_properties, elements_properties=elements_properties, internal_bypass=internal_bypass)


    # =======================
    # Conversion Methods
    # =======================
    @classmethod
    def from_lineartriangle(cls, mesh: LinearTriangleMesh3D, load_properties: bool = True) -> QuadraticTriangleMesh3D:
        r"""
        Create a QuadraticTriangleMesh3D from a :class:`pysdic.geometry.LinearTriangleMesh3D` by adding mid-side nodes.

        The mid-side nodes are computed as the midpoints of each edge of the linear triangle elements.

        Parameters
        ----------
        mesh : LinearTriangleMesh3D
            The linear triangle mesh to convert.

        load_properties : bool, optional
            If True, the vertex and element properties from the linear mesh are copied to the quadratic mesh where applicable, by default True.
            For the uv mapping properties, the mid-side nodes will have their uv coordinates computed as the average of the uv coordinates of the two vertices defining the edge.

        Returns
        -------
        QuadraticTriangleMesh3D
            The resulting quadratic triangle mesh.

    
        Example
        -------

        Create a linear triangle mesh and convert it to a quadratic triangle mesh:

        .. code-block:: python

            import numpy
            from pysdic.geometry import LinearTriangleMesh3D, QuadraticTriangleMesh3D

            vertices = numpy.array([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ])

            connectivity = numpy.array([
                [0, 1, 2],
                [0, 1, 3],
                [0, 2, 3],
                [1, 2, 3],
            ])

            mesh = LinearTriangleMesh3D(
                vertices=vertices,
                connectivity=connectivity,
            )

            displacement = numpy.array([
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.0, 0.1, 0.0],
                [0.0, 0.0, 0.1],
            ])

            mesh.set_vertices_property("displacement", displacement)

            quad_mesh = QuadraticTriangleMesh3D.from_lineartriangle(mesh)
            quad_mesh.visualize()

        """
        if not isinstance(mesh, LinearTriangleMesh3D):
            raise TypeError("Input mesh must be an instance of LinearTriangleMesh3D")
        if not isinstance(load_properties, bool):
            raise TypeError("load_properties must be a boolean")

        # Extract linear mesh data
        linear_vertices = mesh.vertices.points  # Shape (N, 3)
        linear_unique_edges = mesh.extract_unique_edges()  # Shape (E, 2)
        n_initial_vertices = linear_vertices.shape[0]
        n_elements = mesh.connectivity.shape[0]

        # Compute mid-side nodes
        mid_side_nodes = (linear_vertices[linear_unique_edges[:, 0]] + linear_vertices[linear_unique_edges[:, 1]]) / 2.0  # Shape (E, 3)
        n_mid_vertices = mid_side_nodes.shape[0]

        # Build a dictionary for edge indices to mid-side nodes (Shape (E,))
        edge_to_mid_node = {}
        for i, edge in enumerate(linear_unique_edges):
            # Store the index of the mid-node for each unique edge
            edge_to_mid_node[tuple(sorted(edge))] = i + linear_vertices.shape[0]

        # Create new connectivity array (Shape (M, 6))
        new_vertices = PointCloud3D(numpy.vstack((linear_vertices, mid_side_nodes)))  # Shape (N + E, 3)
        new_connectivity = numpy.full((n_elements, 6), -1, dtype=numpy.int64)

        # Loop over the elements to assign connectivity
        mid_node_to_element = {}
        mid_node_to_nat_coords = {}
        for i in range(n_elements):
            n1, n2, n3 = mesh.connectivity[i]
            
            # Sort edges and retrieve mid-node indices from the dictionary
            edge1 = tuple(sorted((n1, n2)))
            edge2 = tuple(sorted((n2, n3)))
            edge3 = tuple(sorted((n3, n1)))

            # Get the indices for the mid-side nodes
            m1 = edge_to_mid_node[edge1]
            m2 = edge_to_mid_node[edge2]
            m3 = edge_to_mid_node[edge3]

            # Map mid-side nodes to the current element
            mid_node_to_element[m1] = i
            mid_node_to_nat_coords[m1] = (0.5, 0.0)
            mid_node_to_element[m2] = i
            mid_node_to_nat_coords[m2] = (0.5, 0.5)
            mid_node_to_element[m3] = i
            mid_node_to_nat_coords[m3] = (0.0, 0.5)

            # Update the new connectivity with the node and mid-node indices
            new_connectivity[i] = [n1, n2, n3, m1, m2, m3]

        # Propagate properties if needed
        vertices_properties = {}
        elements_properties = {}
        if load_properties:
            # Create the integration points for mid-side nodes 
            natural_coords = numpy.full((len(mid_node_to_element), 2), numpy.nan, dtype=numpy.float64)
            element_indices = numpy.full((len(mid_node_to_element),), -1, dtype=numpy.int64)
            for index in range(n_mid_vertices):
                mid_node_index = index + n_initial_vertices
                natural_coords[index, :] = numpy.array(mid_node_to_nat_coords[mid_node_index], dtype=numpy.float64)
                element_indices[index] = mid_node_to_element[mid_node_index]

            # First copy vertex properties
            for key in mesh.list_vertices_properties():
                prop = mesh.get_vertices_property(key)
                vertices_properties[key] = numpy.concatenate(
                    [prop, mesh.interpolate_property_at_natural_coordinates(
                        natural_coords, # (N_mid_nodes, 2)
                        element_indices, # (N_mid_nodes,)
                        property_key=key
                    )], axis=0
                )

            # Then copy element properties
            for key in mesh.list_elements_properties():
                prop = mesh.get_elements_property(key)
                elements_properties[key] = prop.copy()

            # UV MAPPING SPECIAL CASE
            if "uvmap" in elements_properties:
                new_uvmap = numpy.full((n_elements, 12), -1.0, dtype=numpy.float64)
                for index in range(n_elements):
                    uvmap_index = elements_properties["uvmap"][index]  # Shape (6,)
                    n4_u = (uvmap_index[0] + uvmap_index[2]) / 2.0
                    n4_v = (uvmap_index[1] + uvmap_index[3]) / 2.0
                    n5_u = (uvmap_index[2] + uvmap_index[4]) / 2.0
                    n5_v = (uvmap_index[3] + uvmap_index[5]) / 2.0
                    n6_u = (uvmap_index[4] + uvmap_index[0]) / 2.0
                    n6_v = (uvmap_index[5] + uvmap_index[1]) / 2.0
                    new_uvmap[index] = numpy.array([
                        uvmap_index[0], uvmap_index[1],
                        uvmap_index[2], uvmap_index[3],
                        uvmap_index[4], uvmap_index[5],
                        n4_u, n4_v,
                        n5_u, n5_v,
                        n6_u, n6_v
                    ])
                elements_properties["uvmap"] = new_uvmap

        # Create and return the QuadraticTriangleMesh3D
        return cls(new_vertices, new_connectivity, vertices_properties=vertices_properties, elements_properties=elements_properties, internal_bypass=mesh.internal_bypass)
    

    def to_lineartriangle(self, save_properties: bool = True) -> LinearTriangleMesh3D:
        r"""
        Convert the QuadraticTriangleMesh3D to a :class:`pysdic.geometry.LinearTriangleMesh3D` by removing mid-side nodes for elements.

        .. note::

            The resulting LinearTriangleMesh3D may contain unused vertices . You can call the :meth:`pysdic.geometry.Mesh3D.remove_unused_vertices` method on the resulting linear mesh to remove them.

        Parameters
        ----------
        save_properties : bool, optional
            If True, the vertex and element properties from the quadratic mesh are copied to the linear mesh where applicable, by default True.

        Returns
        -------
        LinearTriangleMesh3D
            The resulting linear triangle mesh.

        
        Example
        -------

        Create a quadratic triangle mesh and convert it to a linear triangle mesh:

        .. code-block:: python

            vertices = numpy.array([
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.5],
                [0.0, 0.0, 1.0],
                [0.5, 0.5, 0.0],
                [0.5, 0.0, 0.5],
                [0.0, 0.5, 0.5],
            ])

            connectivity = numpy.array([
                [0, 2, 4, 1, 7, 3],
                [0, 2, 6, 1, 8, 5],
                [0, 4, 6, 3, 9, 5],
                [2, 4, 6, 7, 9, 8],
            ])

            mesh = QuadraticTriangleMesh3D(
                vertices=vertices,
                connectivity=connectivity,
            )

            linear_mesh = mesh.to_lineartriangle()
            linear_mesh.visualize()

        To remove extra vertices that are not used in the linear mesh, you can call the :meth:`pysdic.geometry.Mesh3D.remove_unused_vertices` method on the resulting linear mesh.

        .. code-block:: python

            linear_mesh.remove_unused_vertices()

        """
        # Extract vertices and connectivity
        quad_vertices = self.vertices  # PointCloud3D
        quad_connectivity = self.connectivity  # Shape (M, 6)
        n_elements = quad_connectivity.shape[0]

        # Create new connectivity for linear triangles (Shape (M, 3))
        linear_connectivity = quad_connectivity[:, :3]

        # Create new vertices PointCloud3D
        linear_vertices = quad_vertices

        # Propagate properties if needed
        vertices_properties = {}
        elements_properties = {}
        if save_properties:
            vertices_properties = {key : self.get_vertices_property(key) for key in self.list_vertices_properties()}
            elements_properties = {key : self.get_elements_property(key) for key in self.list_elements_properties()}

            # UV MAPPING SPECIAL CASE
            if "uvmap" in elements_properties:
                new_uvmap = numpy.full((n_elements, 6), -1.0, dtype=numpy.float64)
                for index in range(n_elements):
                    uvmap_index = elements_properties["uvmap"][index]  # Shape (12,)
                    new_uvmap[index] = uvmap_index[:6]
            
        # Create and return the LinearTriangleMesh3D
        linear_mesh = LinearTriangleMesh3D(
            linear_vertices,
            linear_connectivity,
            vertices_properties=vertices_properties,
            elements_properties=elements_properties,
            internal_bypass=self.internal_bypass
        )

        # Copy the data to avoid reference issues
        return linear_mesh.copy()


    # =======================
    # Parent Abstract Methods
    # =======================
    def shape_functions(self, natural_coords: numpy.ndarray, jacobian: bool = False) -> Tuple[numpy.ndarray, Optional[numpy.ndarray]]:
        r"""
        Compute the shape functions at given natural coordinates.

        Lets consider a mesh with K vertices per element, and d natural coordinates.
        The given natural coordinates should be (Np, d) where Np is the number of points to evaluate and d is the dimension of the natural coordinates (here d=2 for triangles).
        The returned shape functions will be of shape (Np, K) and each row will sum to 1 and contain the values of the shape functions associated with each vertex of the element.

        The shape fonctions :math:`N_i` are defined such that:

        .. math::

            X = \sum_{i=1}^{K} N_i(\xi, \eta) X_i

        where :math:`X` are the global coordinates of a point, and :math:`X_i` are the coordinates of the vertices of the element and :math:`(\xi, \eta)` are the natural coordinates.

        The shape functions for a linear triangle are defined as:

        .. math::

            N_1(\xi, \eta) = 2 * (1 - \xi - \eta) * (1/2 - \xi - \eta)

        .. math::

            N_2(\xi, \eta) = 2 * \xi * (\xi - 1/2)

        .. math::

            N_3(\xi, \eta) = 2 * \eta * (\eta - 1/2)

        .. math::

            N_4(\xi, \eta) = 4 * \xi * (1 - \xi - \eta)

        .. math::

            N_5(\xi, \eta) = 4 * \xi * \eta

        .. math::

            N_6(\xi, \eta) = 4 * \eta * (1 - \xi - \eta)


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
            An array-like of shape (Np, d) where Np is the number of points to evaluate and d=2 is the number of natural coordinates.

        Returns
        -------
        numpy.ndarray
            An array of shape (Np, K) where K is the number of nodes per element.

        Optional[numpy.ndarray]
            If ``jacobian`` is True, an array of shape (Np, K, d) where K is the number of nodes per element and d is the number of natural coordinates. Otherwise, None.

        """
        natural_coords = numpy.asarray(natural_coords, dtype=numpy.float64)
        if natural_coords.ndim != 2 or natural_coords.shape[1] != self._n_dimensions:
            raise ValueError(f"natural_coords must be of shape (Np, {self._n_dimensions})")
    
        # Extract natural coordinates
        xi = natural_coords[:, 0]
        eta = natural_coords[:, 1]

        # Compute shape functions
        N1 = 2 * (1 - xi - eta) * (0.5 - xi - eta)
        N2 = 2 * xi * (xi - 0.5)
        N3 = 2 * eta * (eta - 0.5)
        N4 = 4 * xi * (1 - xi - eta)
        N5 = 4 * xi * eta
        N6 = 4 * eta * (1 - xi - eta)
        shape_funcs = numpy.vstack((N1, N2, N3, N4, N5, N6)).T  # Shape (Np, 6)

        # Compute Jacobian if needed
        jacobian_matrix = None
        if jacobian:
            dN1_dxi = -2 * (1.5 - 2 * xi - 2 * eta)
            dN1_deta = -2 * (1.5 - 2 * xi - 2 * eta)
            dN2_dxi = 4 * xi - 1
            dN2_deta = 0.0
            dN3_dxi = 0.0
            dN3_deta = 4 * eta - 1
            dN4_dxi = 4 * (1 - 2 * xi - eta)
            dN4_deta = -4 * xi
            dN5_dxi = 4 * eta
            dN5_deta = 4 * xi
            dN6_dxi = -4 * eta
            dN6_deta = 4 * (1 - xi - 2 * eta)
            dN_dxi = numpy.vstack((dN1_dxi, dN2_dxi, dN3_dxi, dN4_dxi, dN5_dxi, dN6_dxi)).T  # Shape (Np, 6)
            dN_deta = numpy.vstack((dN1_deta, dN2_deta, dN3_deta, dN4_deta, dN5_deta, dN6_deta)).T  # Shape (Np, 6)

            jacobian_matrix = numpy.zeros((natural_coords.shape[0], self._n_vertices_per_element, self._n_dimensions), dtype=numpy.float64)
            jacobian_matrix[:, :, 0] = dN_dxi  # Derivative w.r.t xi
            jacobian_matrix[:, :, 1] = dN_deta  # Derivative w.r.t eta

        return shape_funcs, jacobian_matrix
    
