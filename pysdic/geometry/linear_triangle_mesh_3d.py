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
from numbers import Number

import numpy
import pyvista
import open3d

from .mesh_3d import Mesh3D
from .point_cloud_3d import PointCloud3D
from .integrated_points import IntegratedPoints

class LinearTriangleMesh3D(Mesh3D):
    r"""
    Subclass of Mesh3D representing a 3D mesh composed of linear triangular elements.

    The vertices are represented as a PointCloud3D instance with shape (N, 3),
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

    Additional Properties
    ---------------------

    The new elements property specific to this class is:

    - `uvmap`: A numpy ndarray of shape (M, 6) representing the UV mapping of each triangular element.

    Instanciation
    --------------
    To create a LinearTriangleMesh3D instance, you need to provide the vertices and connectivity as follows

    .. code-block:: python

        from pysdic import LinearTriangleMesh3D, PointCloud3D

        vertices = PointCloud3D(numpy.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]))  # Example vertices
        connectivity = numpy.array([[0, 1, 2, 3]])  # Example connectivity
        mesh = LinearTriangleMesh3D(vertices, connectivity)

    You can also load a mesh from a file using the appropriate method for the file format.

    - :meth:`from_meshio` to load from a meshio Mesh object.
    - :meth:`from_vtk` to load from a VTK file.

    To vizualize the mesh, you can use the `visualize` method:

    .. code-block:: python

        mesh.visualize()

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

    _n_nodes_per_element: int = 3
    _n_dimensions: int = 2
    _meshio_cell_type: str = "triangle"

    def __init__(self, vertices: PointCloud3D, connectivity: numpy.ndarray, *, vertices_properties: dict = None, elements_properties: dict = None, internal_bypass: bool = False):
        # Define expected properties informations
        if not hasattr(self, "_vertices_predefined_metadata"):
            self._vertices_predefined_metadata = {}
        if not hasattr(self, "_elements_predefined_metadata"):
            self._elements_predefined_metadata = {}

        self._elements_predefined_metadata.update({
            "uvmap": {"dim": 6, "type": numpy.float64, "check_method": self._internal_check_uvmap},
        })

        super().__init__(vertices, connectivity, vertices_properties=vertices_properties, elements_properties=elements_properties, internal_bypass=internal_bypass)


    # ======================
    # Internal Checks Methods
    # ======================        
    def _internal_check_uvmap(self, uvmap: numpy.ndarray) -> None:
        r"""
        Internal method to check the validity of the uvmap property.

        Parameters
        ----------
        uvmap : numpy.ndarray
            The uvmap property to check, should be of shape (M, 6) where M is the number of elements.

        Raises
        ------
        ValueError
            If any uv coordinate is not in the range [0, 1].
        """
        if numpy.any(uvmap < 0) or numpy.any(uvmap > 1):
            raise ValueError("All UV coordinates must be in the range [0, 1].")


    # =======================
    # New Properties
    # =======================
    @property
    def elements_uvmap(self) -> Optional[numpy.ndarray]:
        r"""
        Get or set the UV mapping of each triangular element.

        The UV mapping is stored as a numpy ndarray of shape (M, 6), where M is the number of elements.

        The 6 values correspond to the UV coordinates of the 3 vertices of the triangle: (u1, v1, u2, v2, u3, v3).

        Parameters
        ----------
        value : numpy.ndarray, optional
            A numpy ndarray of shape (M, 6) to set as the UV mapping.

        Returns
        -------
        numpy.ndarray
            An array of shape (M, 6) where M is the number of elements. Or None if not set.
        """
        return self.get_elements_property("uvmap")

    @elements_uvmap.setter
    def elements_uvmap(self, value: Optional[numpy.ndarray]) -> None:
        self.set_elements_property("uvmap", value)


    # =======================
    # Conversion Methods
    # =====================
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
            mesh_instance = cls(vertices=vertices, connectivity=triangles)
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
    

    def to_open3d(self, legacy: bool = False) -> Union[open3d.t.geometry.TriangleMesh, open3d.geometry.TriangleMesh]:
        r"""
        Convert the LinearTriangleMesh3D instance to an Open3D TriangleMesh object.

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

        Returns
        -------
        Union[open3d.t.geometry.TriangleMesh, open3d.geometry.TriangleMesh]
            An Open3D TriangleMesh object containing the mesh data.
        """
        if legacy:
            o3d_mesh = open3d.geometry.TriangleMesh()
            o3d_mesh.vertices = open3d.utility.Vector3dVector(self.vertices)
            o3d_mesh.triangles = open3d.utility.Vector3iVector(self.connectivity)

            # Check if UV mapping is available
            if self.elements_uvmap is not None:
                uvmap = self.elements_uvmap.reshape(-1, 2)
                o3d_mesh.triangle_uvs = open3d.utility.Vector2dVector(uvmap)

        else:
            o3d_mesh = open3d.t.geometry.TriangleMesh()
            o3d_mesh.vertex.positions = open3d.core.Tensor(self.vertices, dtype=open3d.core.float32)
            o3d_mesh.triangle.indices = open3d.core.Tensor(self.connectivity, dtype=open3d.core.int32)

            # Check if UV mapping is available
            if self.uvmap is not None:
                uvmap = self.uvmap.reshape(self.Ntriangles, 3, 2)  # Reshape to (M, 3, 2) for Open3D T geometry
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


    def cast_rays(self, ray_origins: numpy.ndarray, ray_directions: numpy.ndarray, weights: Optional[numpy.ndarray] = None) -> IntegratedPoints:
        r"""
        Cast rays into the mesh and compute the intersection points.

        This method uses Open3D to perform ray-mesh intersection tests.

        .. code-block:: python

            import numpy as np
            from pysdic.geometry import PointCloud3D, LinearTriangleMesh3D

            # Create a point cloud from a NumPy array
            points_array = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]) # shape (4, 3)

            point_cloud = PointCloud3D.from_array(points_array)
            mesh_connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]) # shape (4, 3)

            mesh = LinearTriangleMesh3D(point_cloud, mesh_connectivity)

            ray_origins = np.array([[0.1, 0.1, -1], [0.5, 0.5, -1]])  # shape (2, 3)
            ray_directions = np.array([[0, 0, 1], [0, 0, 1]])  # shape (2, 3)

            intersection_points = mesh.cast_rays(ray_origins, ray_directions)  # Returns an IntegratedPoints instance with intersection points

        .. seealso::

            - :class:`IntegratedPoints` for more information on the structure of the returned intersection points.
            - `Open3D Ray Casting Documentation <http://www.open3d.org/docs/release/tutorial/geometry/ray_casting.html>`_ for more details on ray casting.

        ..warning::

            This method converts the rays into a float32 format for compatibility with Open3D.

        Parameters
        ----------
        ray_origins : numpy.ndarray
            An array of shape (Nr, 3) where Nr is the number of rays, representing the origins of the rays.

        ray_directions : numpy.ndarray
            An array of shape (Nr, 3) where Nr is the number of rays, representing the directions of the rays.

        weights : Optional[numpy.ndarray], optional
            An array of shape (Nr, ) representing weights for each ray, by default None. Meaning all weights are 1.

        Returns
        -------
        IntegratedPoints
            An IntegratedPoints instance containing the intersection points and related information.

        """
        # Combine ray origins and directions into a single array of shape (..., 6)
        rays_origins = numpy.asarray(ray_origins, dtype=numpy.float64)
        rays_directions = numpy.asarray(ray_directions, dtype=numpy.float64)
        rays = numpy.concatenate((rays_origins, rays_directions), axis=-1)  # Shape: (..., 6)

        # Extract the Open3D mesh for the specified frame
        o3d_mesh = self.to_open3d(legacy=False)

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

        # Construct the output
        intersect_points = IntegratedPoints(natural_coordinates, element_indices, weights=weights, n_dimensions=self._n_dimensions)

        return intersect_points

        
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
            jacobian_matrix = numpy.zeros((natural_coords.shape[0], self._n_nodes_per_element, self._n_dimensions), dtype=numpy.float64)
            jacobian_matrix[:, :, 0] = dN_dxi  # Derivative w.r.t xi
            jacobian_matrix[:, :, 1] = dN_deta  # Derivative w.r.t eta

        return shape_funcs, jacobian_matrix
    

    # =======================
    # Visualization Methods
    # =======================
    def visualize(
            self,
            point_color: str = "black",
            point_size: int = 5,
            edge_color: str = "blue",
            edge_width: int = 1,
            face_color: str = "gray",   
            face_opacity: float = 0.5,
            show_edges: bool = True,
            show_faces: bool = True,
        ) -> None:
        r"""
        Visualize the 3D triangular mesh using PyVista.

        This method creates a 3D plot of the mesh, displaying its vertices, edges, and faces.
        The appearance of the vertices, edges, and faces can be customized using various parameters.

        .. code-block:: python

            import numpy as np

            from pysdic.geometry import PointCloud3D, LinearTriangleMesh3D

            # Create a point cloud from a NumPy array
            points_array = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]]) # shape (4, 3)

            point_cloud = PointCloud3D.from_array(points_array)

            mesh_connectivity = np.array([[0, 1, 2], [1, 2, 3]]) # shape (2, 3)

            mesh = LinearTriangleMesh3D(point_cloud, mesh_connectivity)
            mesh.visualize(face_color='green', face_opacity=0.7, edge_color='black')

        .. seealso::

            - :meth:`visualize_vertices_property` to visualize a vertex property on the mesh.
            - :meth:`visualize_texture` to visualize the texture of the mesh.

        Parameters
        ----------
        point_color : str, optional
            Color of the vertices (points) in the mesh, by default "black".

        point_size : int, optional
            Size of the vertices (points) in the mesh, by default 5.

        edge_color : str, optional
            Color of the edges in the mesh, by default "blue".

        edge_width : int, optional
            Width of the edges in the mesh, by default 1.

        face_color : str, optional
            Color of the faces in the mesh, by default "gray".

        face_opacity : float, optional
            Opacity of the faces in the mesh (0.0 to 1.0), by default 0.5.

        show_edges : bool, optional
            Whether to display the edges of the mesh, by default True.

        show_faces : bool, optional
            Whether to display the faces of the mesh, by default True.

            
        More Information
        -------------------------

        This method only display the mesh without additional elements.
        To display additional elements, use PyVista directly.

        .. seealso::
        
            - `PyVista Documentation <https://docs.pyvista.org>`_ for more details on visualization options.

        Examples
        --------

        .. code-block:: python

            from pysdic.geometry import create_linear_triangle_heightmap
            import numpy as np

            surface_mesh = create_linear_triangle_heightmap(
                height_function=lambda x, y: 0.5 * np.sin(np.pi * x) * np.cos(np.pi * y),
                x_bounds=(-1.0, 1.0),
                y_bounds=(-1.0, 1.0),
                n_x=50,
                n_y=50,
            )
            surface_mesh.visualize(face_color='green', face_opacity=0.7, edge_color='black')

        .. figure:: ../../../pysdic/resources/linear_triangle_mesh_3d_visualize_example.png
           :width: 600
           :align: center
            
           Example of a 3D triangular mesh visualization using the `visualize` method.
            
        """
        # Check input data
        if self.n_vertices == 0:
            raise ValueError("Cannot visualize an empty mesh.")
        if self.n_elements == 0:
            raise ValueError("Cannot visualize a mesh without elements.")

        if not isinstance(point_color, str):
            raise ValueError("Point color must be a string.")
        if not (isinstance(point_size, Number) and point_size > 0):
            raise ValueError("Point size must be a positive number.")
        if not isinstance(edge_color, str):
            raise ValueError("Edge color must be a string.")
        if not (isinstance(edge_width, Number) and edge_width > 0):
            raise ValueError("Edge width must be a positive number.")
        if not isinstance(face_color, str):
            raise ValueError("Face color must be a string.")
        if not (isinstance(face_opacity, Number) and 0.0 <= face_opacity <= 1.0):
            raise ValueError("Face opacity must be a float between 0.0 and 1.0.")
        if not isinstance(show_edges, bool):
            raise ValueError("show_edges must be a boolean.")
        if not isinstance(show_faces, bool):
            raise ValueError("show_faces must be a boolean.")
        
        # Create a PyVista mesh
        pv_mesh = pyvista.PolyData(self.vertices.points, numpy.hstack((numpy.full((self.n_elements, 1), 3), self.connectivity)).astype(numpy.int64))

        # Create a PyVista plotter
        plotter = pyvista.Plotter()

        # Add faces if required
        if show_faces:
            plotter.add_mesh(pv_mesh, color=face_color, opacity=face_opacity, show_edges=show_edges, edge_color=edge_color, line_width=edge_width)

        # Add edges if required
        elif show_edges:
            edges = pv_mesh.extract_feature_edges()
            plotter.add_mesh(edges, color=edge_color, line_width=edge_width)

        # Add points if required
        plotter.add_points(self.vertices.points, color=point_color, point_size=point_size)

        # Show the plot
        plotter.show_axes() 
        plotter.show_grid()
        plotter.show()


    def visualize_vertices_property(
            self,
            property_key: Optional[str] = None,
            property_array: Optional[numpy.ndarray] = None,
            property_label: Optional[str] = None,
            cmap: str = "viridis",
            vmin : Optional[float] = None,
            vmax : Optional[float] = None,
            show_edges: bool = True,
            ) -> None:
        r"""
        Visualize a vertex property on the 3D triangular mesh using PyVista.

        This method creates a 3D plot of the mesh, displaying its vertices colored according to the specified property.
        The appearance of the vertices can be customized using various parameters.

        .. code-block:: python
        
            import numpy as np
            from pysdic.geometry import PointCloud3D, LinearTriangleMesh3D

            # Create a point cloud from a NumPy array
            points_array = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]]) # shape (4, 3)

            point_cloud = PointCloud3D.from_array(points_array)
            mesh_connectivity = np.array([[0, 1, 2], [1, 2, 3]]) # shape (2, 3)

            mesh = LinearTriangleMesh3D(point_cloud, mesh_connectivity)

            # Define a vertex property (e.g., temperature)
            temperature = np.array([[100], [150], [200], [250]])  # shape (4, 1)
            mesh.set_vertices_property("temperature", temperature)  

            # Visualize the temperature property on the mesh
            mesh.visualize_vertices_property(property_key="temperature", cmap='hot')

        If the property is not found in the mesh, it can be provided directly as a numpy array.

        If the property is a (N, 1) array, it will be used directly.
        If the property is a (N, A) array with A > 1, the norm of the attributes will be computed and used for coloring.

        .. seealso::

            - :meth:`visualize` to visualize the mesh without coloring by a property.
            - :meth:`visualize_texture` to visualize the texture of the mesh.

        Parameters
        ----------
        property_key : str, optional
            The name of the vertex property to visualize. If None, `property_array` must be provided, by default None.

        property_array : numpy.ndarray, optional
            A numpy ndarray of shape (N, A) where N is the number of vertices and A is the number of attributes for that property.
            If None, `property_key` must be provided, by default None.

        property_label : str, optional
            The label to use for the property in the visualization legend. If None, `property_key` will be used, by default None.

        cmap : str, optional
            The colormap to use for coloring the vertices based on the property values, by default "viridis".

        vmin : float, optional
            The minimum value for the colormap, by default None.

        vmax : float, optional
            The maximum value for the colormap, by default None.

        show_edges : bool, optional
            Whether to show the mesh edges in the visualization, by default True.

        More Information
        -------------------------

        This method only display the mesh without additional elements.
        To display additional elements, use PyVista directly.

        .. seealso::

            - `PyVista Documentation <https://docs.pyvista.org>`_ for more details on visualization options.

        Examples
        --------

        .. code-block:: python

            from pysdic.geometry import create_linear_triangle_heightmap
            import numpy as np

            surface_mesh = create_linear_triangle_heightmap(
                height_function=lambda x, y: 0.5 * np.sin(np.pi * x) * np.cos(np.pi * y),
                x_bounds=(-1.0, 1.0),
                y_bounds=(-1.0, 1.0),
                n_x=50,
                n_y=50,
            )
            
            height = surface_mesh.vertices.points[:, 2].reshape(-1, 1)  # Use the z-coordinate as a property

            surface_mesh.visualize_vertices_property(property_array=height, property_label='Height (m)', cmap='terrain')

        .. figure:: ../../../pysdic/resources/linear_triangle_mesh_3d_visualize_vertices_property_example.png
           :width: 600
           :align: center

           Example of a 3D triangular mesh visualization using the `visualize_vertices_property` method.

        """
        # Case of an empty mesh
        if self.n_vertices == 0:
            raise ValueError("Cannot visualize an empty mesh.")
        if self.n_elements == 0:
            raise ValueError("Cannot visualize a mesh without elements.")
        
        # Extract the property array
        if (property_key is None and property_array is None) or (property_key is not None and property_array is not None):
            raise ValueError("Either property_key or property_array must be provided, but not both.")
        property_array = self._get_vertices_property(property_key, property_array, raise_error=True)
        if property_array.shape[1] != 1:
            property_array = numpy.linalg.norm(property_array, axis=1, keepdims=True) # use the norm if multiple attributes

        # Default parameters
        if property_label is None:
            property_label = "property"
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
        
        # Create a PyVista mesh
        pv_mesh = pyvista.PolyData(self.vertices.points, numpy.hstack((numpy.full((self.n_elements, 1), 3), self.connectivity)).astype(numpy.int64))

        # Add the property as point data
        pv_mesh.point_data[property_label] = property_array.flatten()

        # Create a PyVista plotter
        plotter = pyvista.Plotter()

        # Add the mesh with the property colormap
        plotter.add_mesh(pv_mesh, scalars=property_label, cmap=cmap, show_edges=show_edges, clim=[vmin, vmax])

        # Show the plot
        plotter.add_scalar_bar(title=property_label, n_labels=5, vertical=True)
        plotter.show_axes()
        plotter.show_grid()
        plotter.show()


    def visualize_texture(
            self,
            texture: numpy.ndarray,
            show_edges: bool = True,
        ) -> None:
        r"""
        Visualize the texture of the mesh using a texture image.

        .. warning::

            The mesh must have the `uvmap` property set for this method to work.

        .. seealso::

            - :meth:`elements_uvmap` to set or get the UV mapping of the elements.

        This method creates a 3D plot of the mesh, displaying its faces textured with the provided image.
        The texture image should be a 2D (grayscale) or 3D (RGB/RGBA) numpy array.

        .. code-block:: python

            import numpy as np
            from pysdic.geometry import PointCloud3D, LinearTriangleMesh3D

            # Create a point cloud from a NumPy array
            points_array = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]) # shape (4, 3)

            point_cloud = PointCloud3D.from_array(points_array)
            mesh_connectivity = np.array([[0, 1, 2], [1, 2, 3]]) # shape (2, 3)
            mesh = LinearTriangleMesh3D(point_cloud, mesh_connectivity)

            # Define UV mapping for the elements
            uvmap = np.array([[0, 0, 1, 0, 0, 1], [1, 0, 1, 1, 0, 1]])  # shape (2, 6)
            mesh.elements_uvmap = uvmap
            
            # Load a texture image (e.g., from an image file)
            texture_image = np.random.rand(256, 256, 3)  # Example texture image

            # Visualize the mesh with the texture
            mesh.visualize_texture(texture_image)

        Parameters
        ----------
        texture : numpy.ndarray
            The texture image to apply to the mesh. It must have the same number of vertices as the mesh.

        show_edges : bool, optional
            Whether to show the mesh edges in the visualization, by default True.

        More Information
        -------------------------
        This method only display the mesh without additional elements.
        To display additional elements, use PyVista directly.

        .. seealso::
        
            - `PyVista Documentation <https://docs.pyvista.org>`_ for more details on visualization options.

        Examples
        --------

        .. code-block:: python

            from pysdic.geometry import create_linear_triangle_heightmap
            import numpy as np

            surface_mesh = create_linear_triangle_heightmap(
                height_function=lambda x, y: 0.5 * np.sin(np.pi * x) * np.cos(np.pi * y),
                x_bounds=(-1.0, 1.0),
                y_bounds=(-1.0, 1.0),
                n_x=50,
                n_y=50,
            ) # UVMAP already set in the function

            texture_image = np.random.rand(256, 256, 3)  # Example texture image

            surface_mesh.visualize_texture(texture_image, show_edges=False)

            
        .. figure:: ../../../pysdic/resources/linear_triangle_mesh_3d_visualize_texture_example.png
            :width: 600
            :align: center

            Example of a 3D triangular mesh visualization using the `visualize_texture` method.
        """
        # Check input data
        if self.n_vertices == 0:
            raise ValueError("Cannot visualize an empty mesh.")
        if self.n_elements == 0:
            raise ValueError("Cannot visualize a mesh without elements.")
        if self.elements_uvmap is None:
            raise ValueError("The mesh must have the 'uvmap' property set to visualize texture.")
        
        if not isinstance(texture, numpy.ndarray):
            raise ValueError("texture must be a numpy ndarray.")
        
        if texture.ndim < 2 or texture.ndim > 3:
            raise ValueError("texture must be a 2D (grayscale) or 3D (RGB/RGBA) array.")
        if texture.ndim == 3 and texture.shape[2] not in [1, 3, 4]:
            raise ValueError("If texture is 3D, its third dimension must be 1 (grayscale), 3 (RGB), or 4 (RGBA).")
        
        np = numpy
        pv = pyvista

        # Duplicate points per face
        fictive_vertices = np.zeros((self.n_elements * 3, 3), dtype=np.float64)
        fictive_vertices[0::3, :] = self.vertices.points[self.connectivity[:, 0], :]
        fictive_vertices[1::3, :] = self.vertices.points[self.connectivity[:, 1], :]
        fictive_vertices[2::3, :] = self.vertices.points[self.connectivity[:, 2], :]

        # Create connectivity for the fictive vertices
        fictive_connectivity = np.arange(self.n_elements * 3, dtype=np.int64).reshape(self.n_elements, 3)

        # Create a PyVista mesh
        pv_mesh = pv.PolyData(fictive_vertices, np.hstack((np.full((self.n_elements, 1), 3), fictive_connectivity)).astype(np.int64))

        # Set texture coordinates
        pv_mesh.active_texture_coordinates = np.zeros((self.n_elements * 3, 2), dtype=np.float64)

        # UV coordinates per vertex of each element
        uvmap = self.elements_uvmap  # shape (M, 6)
        pv_mesh.active_texture_coordinates[0::3, 0] = uvmap[:, 0]  # u1
        pv_mesh.active_texture_coordinates[0::3, 1] = uvmap[:, 1]  # v1
        pv_mesh.active_texture_coordinates[1::3, 0] = uvmap[:, 2]  # u2
        pv_mesh.active_texture_coordinates[1::3, 1] = uvmap[:, 3]  # v2
        pv_mesh.active_texture_coordinates[2::3, 0] = uvmap[:, 4]  # u3
        pv_mesh.active_texture_coordinates[2::3, 1] = uvmap[:, 5]  # v3

        # Create a PyVista texture
        texture = pv.Texture(texture)

        # Create a PyVista plotter
        plotter = pyvista.Plotter()

        # Add the mesh with the texture
        plotter.add_mesh(pv_mesh, texture=texture, show_edges=show_edges)

        # Show the plot
        plotter.show_axes()
        plotter.show_grid()
        plotter.show()