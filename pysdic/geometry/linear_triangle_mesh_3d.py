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
import open3d

from .surface_mesh_3d import SurfaceMesh3D
from .point_cloud_3d import PointCloud3D
from .integration_points import IntegrationPoints



class LinearTriangleMesh3D(SurfaceMesh3D):
    r"""
    Subclass of :class:`pysdic.geometry.Mesh3D` and :class:`pysdic.geometry.SurfaceMesh3D` representing a 3D mesh composed of linear triangular elements.

    The vertices are represented as a :class:`pysdic.geometry.PointCloud3D` instance with shape (N, 3),
    where N is the number of vertices. Each vertex has 3 coordinates (x, y, z).
    The elements are represented by a numpy ndarray with shape (M, 3),
    where M is the number of triangular elements and each element is defined by 3 vertex indices.

    The coordinates of a point into the mesh can be accessed by the natural coordinates
    in the reference element. The natural coordinates (:math:`\xi, \eta`) for a linear triangle satisfy:

    - :math:`0 <= \xi <= 1`
    - :math:`0 <= \eta <= 1`
    - :math:`\xi + \eta <= 1`

    We have K=3 vertices per element, and d=2 for the dimensions of the natural coordinates.

    Lets :math:`X` be the coordinates of a point in the mesh. The transformation from natural coordinates to global coordinates is given by:

    .. math::

        X = \sum_{i=1}^{K} N_i(\xi, \eta) X_i

    where :math:`N_i` are the shape functions associated with each vertex, and :math:`X_i` are the coordinates of the vertices of the element.

    The shape functions for a linear triangle are defined as:

    .. math::

        N_1(\xi, \eta) = 1 - \xi - \eta

    .. math::

        N_2(\xi, \eta) = \xi

    .. math::

        N_3(\xi, \eta) = \eta

    .. figure:: ../../../pysdic/resources/linear_triangle_reference_element.png
        :alt: Reference element for a linear triangle

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

    _n_vertices_per_element: int = 3
    _n_dimensions: int = 2
    _meshio_cell_type: str = "triangle"
    _vtk_cell_type: int = 5  # VTK_TRIANGLE
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
    def from_open3d(cls, mesh: Union[open3d.t.geometry.TriangleMesh, open3d.geometry.TriangleMesh]) -> LinearTriangleMesh3D:
        r"""
        Create a TriangleMesh3D instance from an Open3D TriangleMesh object.

        .. code-block:: python

            import open3d as o3d
            from pysdic.geometry import LinearTriangleMesh3D

            # Read the mesh from a file
            mesh = o3d.io.read_triangle_mesh("path/to/mesh.ply")

            # Create a LinearTriangleMesh3D instance from the Open3D object
            mesh = LinearTriangleMesh3D.from_open3d(mesh)

        .. warning::
            
            For now, the method only extracts the vertices, triangles, and UV map (if available) from the Open3D mesh.
            The other properties (normals, centroids, areas) are not extracted and must be computed separately.

        Parameters
        ----------
        mesh : Union[open3d.t.geometry.TriangleMesh, open3d.geometry.TriangleMesh]
            An Open3D TriangleMesh object containing the mesh data.

        Returns
        -------
        LinearTriangleMesh3D
            A LinearTriangleMesh3D instance containing the mesh data.
        """
        if not isinstance(mesh, (open3d.t.geometry.TriangleMesh, open3d.geometry.TriangleMesh)):
            raise TypeError(f"Expected an Open3D TriangleMesh object, got {type(mesh)}.")

        if isinstance(mesh, open3d.geometry.TriangleMesh): # Legacy Open3D mesh
            vertices = numpy.asarray(mesh.vertices, dtype=numpy.float64)
            triangles = numpy.asarray(mesh.triangles, dtype=numpy.int64)
            mesh_instance = cls(vertices=PointCloud3D.from_array(vertices), connectivity=triangles)
            mesh_instance.validate()  # Validate the mesh structure

            # Check if UV mapping is available
            if mesh.triangle_uvs is not None and numpy.asarray(mesh.triangle_uvs).size > 0:
                uvmap = numpy.asarray(mesh.triangle_uvs, dtype=numpy.float64)
                # Convert UV map to the format (M, 6) - u1, v1, u2, v2, u3, v3
                uvmap = uvmap.reshape(-1, 6)
                mesh_instance.elements_uvmap = uvmap

        else: # Open3D T geometry mesh
            vertices = numpy.asarray(mesh.vertex.positions.numpy(), dtype=numpy.float64)
            triangles = numpy.asarray(mesh.triangle.indices.numpy(), dtype=numpy.int64)
            mesh_instance = cls(vertices=vertices, connectivity=triangles)
            mesh_instance.validate()  # Validate the mesh structure

            # Check if UV mapping is available
            if any(key == "texture_uvs" for key, _ in mesh.triangle.items()):
                uvmap = numpy.asarray(mesh.triangle.texture_uvs.numpy(), dtype=numpy.float64)
                # Convert UV map to the format (M, 6) - u1, v1, u2, v2, u3, v3
                uvmap = uvmap.reshape(-1, 6)
                mesh_instance.elements_uvmap = uvmap

        return mesh_instance
    

    def to_open3d(self, legacy: bool = False, uvmap: bool = True) -> Union[open3d.t.geometry.TriangleMesh, open3d.geometry.TriangleMesh]:
        r"""
        Convert the LinearTriangleMesh3D instance to an Open3D TriangleMesh object.
        The mesh must not be empty.

        If `legacy` is True, the method returns a legacy Open3D TriangleMesh object.
        Otherwise, it returns a T geometry TriangleMesh object.

        .. code-block:: python

            import open3d as o3d
            from pysdic.geometry import LinearTriangleMesh3D

            # Create a LinearTriangleMesh3D instance
            mesh = LinearTriangleMesh3D(vertices=..., connectivity=...)
            
            # Convert the mesh to an Open3D object
            open3d_mesh = mesh.to_open3d()

        .. warning::

            For now, the method only converts the vertices, triangles, and UV map (if available) to the Open3D mesh.
            The other properties stored in the LinearTriangleMesh3D instance are not transferred.

        Parameters
        ----------
        legacy : bool, optional
            If True, return a legacy Open3D TriangleMesh object. Default is False.

        uvmap : bool, optional
            If True, include the UV mapping in the Open3D mesh if available. Default is True.

        Returns
        -------
        Union[open3d.t.geometry.TriangleMesh, open3d.geometry.TriangleMesh]
            An Open3D TriangleMesh object containing the mesh data.

        Raises
        ------
        ValueError
            If the mesh is empty.   
        """
        if self.n_vertices == 0 or self.n_elements == 0:
            raise ValueError("Cannot write an empty mesh to file.")
        if not isinstance(legacy, bool):
            raise TypeError(f"Expected a boolean for legacy, got {type(legacy)}.")
        if not isinstance(uvmap, bool):
            raise TypeError(f"Expected a boolean for uvmap, got {type(uvmap)}.")

        if legacy:
            o3d_mesh = open3d.geometry.TriangleMesh()
            o3d_mesh.vertices = open3d.utility.Vector3dVector(self.vertices.to_array())
            o3d_mesh.triangles = open3d.utility.Vector3iVector(self.connectivity)

            # Check if UV mapping is available
            if self.elements_uvmap is not None and uvmap:
                uvmap = self.elements_uvmap.reshape(-1, 2)
                o3d_mesh.triangle_uvs = open3d.utility.Vector2dVector(uvmap)

        else:
            o3d_mesh = open3d.t.geometry.TriangleMesh()
            o3d_mesh.vertex.positions = open3d.core.Tensor(self.vertices.to_array(), dtype=open3d.core.float32)
            o3d_mesh.triangle.indices = open3d.core.Tensor(self.connectivity, dtype=open3d.core.int32)

            # Check if UV mapping is available
            if self.elements_uvmap is not None and uvmap:
                uvmap = self.elements_uvmap.reshape(self.n_elements, 3, 2)  # Reshape to (M, 3, 2) for Open3D T geometry
                o3d_mesh.triangle.texture_uvs = open3d.core.Tensor(uvmap, dtype=open3d.core.float32)

        return o3d_mesh


    # =======================
    # Computation Methods
    # =======================
    def compute_elements_areas(self) -> numpy.ndarray:
        r"""
        Compute and set the areas of each triangular element in the mesh.

        The areas are computed using the cross product of two edges of each triangle.

        The areas are stored in a numpy ndarray of shape (M, 1)

        .. code-block:: python

            areas = mesh.compute_elements_areas() # shape (M, 1)

        Returns
        -------
        numpy.ndarray
            An array of shape (M, 1) where M is the number of elements, representing the area of each triangular element.

        """
        if self.n_elements == 0:
            self.elements_areas = numpy.empty((0, 1), dtype=numpy.float64)
            return

        v0 = self.vertices.points[self.connectivity[:, 0], :]
        v1 = self.vertices.points[self.connectivity[:, 1], :]
        v2 = self.vertices.points[self.connectivity[:, 2], :]

        # Compute the vectors for two edges of each triangle
        edge1 = v1 - v0
        edge2 = v2 - v0

        # Compute the cross product of the two edge vectors
        cross_product = numpy.cross(edge1, edge2)

        # Compute the area of each triangle (half the magnitude of the cross product)
        areas = 0.5 * numpy.linalg.norm(cross_product, axis=1)
        areas = areas.reshape(-1, 1)

        return areas


    def compute_elements_normals(self) -> numpy.ndarray:
        r"""
        Compute and set the normal vectors of each triangular element in the mesh.

        The normal vectors are computed using the cross product of two edges of each triangle
        and are normalized to have unit length.

        The normal vectors are stored in a numpy ndarray of shape (M, 3) where the last dimension represents the (x, y, z) components of the normal vector.

        .. code-block:: python

            normals = mesh.compute_elements_normals()  # shape (M, 3)

        Returns
        -------
        numpy.ndarray
            An array of shape (M, 3) where M is the number of elements, representing the normal vector of each triangular element.

        """
        if self.n_elements == 0:
            self.elements_normal_vectors = numpy.empty((0, 3), dtype=numpy.float64)
            return
        
        v0 = self.vertices.points[self.connectivity[:, 0], :]
        v1 = self.vertices.points[self.connectivity[:, 1], :]
        v2 = self.vertices.points[self.connectivity[:, 2], :]

        # Compute the vectors for two edges of each triangle
        edge1 = v1 - v0
        edge2 = v2 - v0

        # Compute the cross product of the two edge vectors to get the normal vector
        normals = numpy.cross(edge1, edge2)

        # Normalize the normal vectors to have unit length
        norms = numpy.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1.0 # avoid division by zero
        normals = normals / norms
        
        return normals

    def compute_vertices_normals(self, elements_normals: Optional[numpy.ndarray] = None, elements_areas: Optional[numpy.ndarray] = None) -> numpy.ndarray:
        r"""
        Compute and set the normal vectors at each vertex of the mesh.

        The normal vector at each vertex is computed as the average of the normal vectors
        of the adjacent triangular elements, weighted by the area of each element.
        The vertex normals are then normalized to have unit length.

        The vertex normal vectors are stored in a numpy ndarray of shape (N, 3) where the last dimension represents the (x, y, z) components of the normal vector.

        .. code-block:: python

            normals = mesh.compute_vertices_normals() # shape (N, 3)

        Parameters
        ----------
        elements_normals : Optional[numpy.ndarray], optional
            Precomputed normal vectors of the elements, of shape (M, 3). If None, the method will compute them, by default None.

        elements_areas : Optional[numpy.ndarray], optional
            Precomputed areas of the elements, of shape (M, 1). If None, the method will compute them, by default None.

        """
        if elements_normals is None:
            elements_normals = self.compute_elements_normals()
        if not isinstance(elements_normals, numpy.ndarray) or elements_normals.shape != (self.n_elements, 3):
            raise ValueError(f"elements_normals must be of shape ({self.n_elements}, 3)")

        if elements_areas is None:
            elements_areas = self.compute_elements_areas()
        if not isinstance(elements_areas, numpy.ndarray) or elements_areas.shape != (self.n_elements, 1):
            raise ValueError(f"elements_areas must be of shape ({self.n_elements}, 1)")

        if self.n_vertices == 0:
            self.vertices_normal_vectors = numpy.empty((0, 3), dtype=numpy.float64)
            return self.vertices_normal_vectors

        vertex_normals = numpy.zeros((self.n_vertices, 3), dtype=numpy.float64)

        # Normales pondérées par l'aire
        weighted_normals = elements_normals * elements_areas  # (M, 3)

        # Pour chaque triangle, répéter les normales 3 fois (une par sommet)
        repeated_normals = numpy.repeat(weighted_normals, 3, axis=0)  # (M*3, 3)
        vertex_indices = self.connectivity.reshape(-1)  # (M*3,)

        # Accumuler avec numpy.add.at
        vertex_normals = numpy.zeros((self.n_vertices, 3), dtype=numpy.float64)
        numpy.add.at(vertex_normals, vertex_indices, repeated_normals)

        # Normalize to unit length
        norms = numpy.linalg.norm(vertex_normals, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # avoid division by zero
        vertex_normals /= norms

        return vertex_normals
    

    def compute_elements_centroids(self) -> numpy.ndarray:
        r"""
        Compute and set the centroids of each triangular element in the mesh.

        The centroid of a triangle is computed as the average of the coordinates of its three vertices.

        The centroids are stored in a numpy ndarray of shape (M, 3) where the last dimension represents the (x, y, z) coordinates of the centroid.

        .. code-block:: python

            centroids = mesh.compute_elements_centroids() # shape (M, 3)

        Returns
        -------
        numpy.ndarray
            An array of shape (M, 3) where M is the number of elements, representing the centroid of each triangular element.

        """
        if self.n_elements == 0:
            self.elements_centroids = numpy.empty((0, 3), dtype=numpy.float64)
            return self.elements_centroids

        # Get the vertices of each triangle
        v0 = self.vertices.points[self.connectivity[:, 0], :]
        v1 = self.vertices.points[self.connectivity[:, 1], :]
        v2 = self.vertices.points[self.connectivity[:, 2], :]

        # Compute the centroid as the average of the three vertices
        centroids = (v0 + v1 + v2) / 3.0

        return centroids   


    def cast_rays(self, ray_origins: numpy.ndarray, ray_directions: numpy.ndarray, weights: Optional[numpy.ndarray] = None, nan_open3d_errors: bool = False) -> IntegrationPoints:
        r"""
        Cast rays into the mesh and compute the intersection points.

        This method uses Open3D to perform ray-mesh intersection tests.

        .. code-block:: python

            import numpy as np
            from pysdic.geometry import PointCloud3D, LinearTriangleMesh3D

            # Create a point cloud from a NumPy array
            points_array = numpy.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]) # shape (4, 3)

            point_cloud = PointCloud3D.from_array(points_array)
            mesh_connectivity = numpy.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]) # shape (4, 3)

            mesh = LinearTriangleMesh3D(point_cloud, mesh_connectivity)

            ray_origins = numpy.array([[0.1, 0.1, -1], [0.5, 0.5, -1]])  # shape (2, 3)
            ray_directions = numpy.array([[0, 0, 1], [0, 0, 1]])  # shape (2, 3)

            intersection_points = mesh.cast_rays(ray_origins, ray_directions)  # Returns an IntegrationPoints instance with intersection points

        .. seealso::

            - :class:`IntegrationPoints` for more information on the structure of the returned intersection points.
            - `Open3D Ray Casting Documentation <http://www.open3d.org/docs/release/tutorial/geometry/ray_casting.html>`_ for more details on ray casting.

        .. warning::

            This method converts the rays into a float32 format for compatibility with Open3D.

        Parameters
        ----------
        ray_origins : numpy.ndarray
            An array of shape (Nr, 3) where Nr is the number of rays, representing the origins of the rays.

        ray_directions : numpy.ndarray
            An array of shape (Nr, 3) where Nr is the number of rays, representing the directions of the rays.

        weights : Optional[numpy.ndarray], optional
            An array of shape (Nr, ) representing weights for each ray, by default None. Meaning all weights are 1.

        nan_open3d_errors : bool, optional
            Due to float32 precision issues in Open3D, some rays may produce coordinates with natural coordinates slightly outside the valid range [0, 1]. Setting this to True will replace such errors with NaN values in the output, by default False (an error will be raised instead).

        Returns
        -------
        IntegrationPoints
            An IntegrationPoints instance containing the intersection points and related information.

        """
        # Combine ray origins and directions into a single array of shape (..., 6)
        rays_origins = numpy.asarray(ray_origins, dtype=numpy.float64)
        rays_directions = numpy.asarray(ray_directions, dtype=numpy.float64)
        rays = numpy.concatenate((rays_origins, rays_directions), axis=-1)  # Shape: (..., 6)

        # Extract the Open3D mesh for the specified frame
        o3d_mesh = self.to_open3d(legacy=False, uvmap=False)

        # Convert rays_origins and rays_directions to numpy arrays
        rays = numpy.asarray(rays, dtype=numpy.float32)
        if rays.shape[-1] != 6:
            raise ValueError("Rays must have shape (..., 6).")

        # Convert numpy arrays to Open3D point clouds (ray origins and directions)
        rays_o3d = open3d.core.Tensor(rays, open3d.core.float32)  # Shape: (..., 6)

        # Create the scene and add the mesh
        raycaster = open3d.t.geometry.RaycastingScene()
        raycaster.add_triangles(o3d_mesh)

        # Cast the rays
        results = raycaster.cast_rays(rays_o3d)

        # Prepare output arrays
        natural_coordinates = numpy.full((*rays.shape[:-1], self._n_dimensions), numpy.nan, dtype=numpy.float64)
        element_indices = numpy.full(*rays.shape[:-1], -1, dtype=int)

        # Extract the intersection points
        intersect_true = results["t_hit"].isfinite().numpy()
        natural_coordinates[intersect_true] = results["primitive_uvs"].numpy().astype(numpy.float64)[intersect_true]
        element_indices[intersect_true] = results["primitive_ids"].numpy().astype(int)[intersect_true]

        # Handle NaN errors due to Open3D float32 precision issues
        if nan_open3d_errors:
            invalid_coords = numpy.logical_or.reduce((
                natural_coordinates[..., 0] < 0,
                natural_coordinates[..., 0] > 1,
                natural_coordinates[..., 1] < 0,
                natural_coordinates[..., 1] > 1,
                natural_coordinates[..., 0] + natural_coordinates[..., 1] > 1,
            ))
            natural_coordinates[invalid_coords] = numpy.nan
            element_indices[invalid_coords] = -1
        else:
            if numpy.any(natural_coordinates[..., 0] < 0) or numpy.any(natural_coordinates[..., 0] > 1) or \
               numpy.any(natural_coordinates[..., 1] < 0) or numpy.any(natural_coordinates[..., 1] > 1) or \
               numpy.any(natural_coordinates[..., 0] + natural_coordinates[..., 1] > 1):
                raise ValueError("Some intersection natural coordinates are out of bounds [0, 1] with open3d. "
                                 "Consider setting nan_open3d_errors=True to handle these cases.")

        # Construct the output
        intersect_points = IntegrationPoints(natural_coordinates, element_indices, weights=weights, n_dimensions=self._n_dimensions)

        return intersect_points


    def extract_unique_edges(self) -> numpy.ndarray:
        r"""
        Extract the unique edges from the triangular mesh.

        Each edge is represented as a pair of vertex indices, sorted in ascending order.
        The method returns a numpy ndarray of shape (E, 2) where E is the number of unique edges.

        .. code-block:: python

            edges = mesh.extract_unique_edges()  # shape (E, 2)

        Returns
        -------
        numpy.ndarray
            An array of shape (E, 2) where E is the number of unique edges, representing the vertex indices of each edge.

        
        Example
        -------

        Consider a triangular mesh with the following connectivity:

        .. code-block:: python

            import numpy as np
            from pysdic.geometry import PointCloud3D, LinearTriangleMesh3D

            vertices_array = np.array([[0, 0, 0],
                                       [1, 0, 0],
                                       [0, 1, 0],
                                       [1, 1, 0]])  # shape (4, 3)

            point_cloud = PointCloud3D.from_array(vertices_array)

            mesh_connectivity = np.array([[0, 1, 2],
                                          [1, 3, 2]])  # shape (2, 3)

            mesh = LinearTriangleMesh3D(point_cloud, mesh_connectivity)

            edges = mesh.extract_unique_edges()
            print(edges)
            # Output:
            # [[0 1]
            #  [0 2]
            #  [1 2]
            #  [1 3]
            #  [2 3]]

        """
        if self.n_elements == 0:
            return numpy.empty((0, 2), dtype=numpy.int64)

        # Extract edges from triangles
        edges = numpy.vstack((
            self.connectivity[:, [0, 1]],
            self.connectivity[:, [1, 2]],
            self.connectivity[:, [2, 0]]
        )) # Shape (M*3, 2)

        # Sort each edge to ensure (min, max) ordering
        edges = numpy.sort(edges, axis=1)

        # Conversion to void type for easy comparison
        dtype = numpy.dtype((numpy.void, edges.dtype.itemsize * edges.shape[1]))

        # Create a view of the points as a 1D array of void type
        a = numpy.ascontiguousarray(edges).view(dtype).ravel()

        # Use numpy.unique to find unique edges
        unique_a = numpy.unique(a)

        # Convert back to original edge format
        unique_edges = unique_a.view(edges.dtype).reshape(-1, edges.shape[1])

        return unique_edges
        
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

            N_1(\xi, \eta) = 1 - \xi - \eta

        .. math::

            N_2(\xi, \eta) = \xi

        .. math::

            N_3(\xi, \eta) = \eta


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
        N1 = 1.0 - xi - eta
        N2 = xi
        N3 = eta
        shape_funcs = numpy.vstack((N1, N2, N3)).T  # Shape (Np, 3)

        # Compute Jacobian if needed
        jacobian_matrix = None
        if jacobian:
            dN_dxi = numpy.array([-1.0, 1.0, 0.0])
            dN_deta = numpy.array([-1.0, 0.0, 1.0])
            jacobian_matrix = numpy.zeros((natural_coords.shape[0], self._n_vertices_per_element, self._n_dimensions), dtype=numpy.float64)
            jacobian_matrix[:, :, 0] = dN_dxi  # Derivative w.r.t xi
            jacobian_matrix[:, :, 1] = dN_deta  # Derivative w.r.t eta

        return shape_funcs, jacobian_matrix
    
