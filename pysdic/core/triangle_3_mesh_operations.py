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
from typing import Union, Tuple, Optional

import numpy
import open3d

from collections import deque

from .objects.mesh import Mesh
from .objects.point_cloud import PointCloud


def triangle_3_mesh_from_open3d(mesh: Union[open3d.t.geometry.TriangleMesh, open3d.geometry.TriangleMesh], internal_bypass: bool = False) -> Mesh:
    r"""
    Create a :class:`Mesh` (:obj:`elements_type` = "triangle_3") instance from an Open3D TriangleMesh object. (Only for 3D embedding dimension meshes :math:`E=3`)

    .. warning::
        
        For now, the method only extracts the vertices, triangles, and UV map (if available) from the Open3D mesh.
        The other properties (normals, centroids, areas) are not extracted and must be computed separately.

    .. seealso::

        - :func:`triangle_3_mesh_to_open3d` for creating an Open3D TriangleMesh object from a :class:`Mesh` instance.

    Parameters
    ----------
    mesh : Union[:class:`open3d.t.geometry.TriangleMesh`, :class:`open3d.geometry.TriangleMesh`]
        An Open3D TriangleMesh object containing the mesh data.

    internal_bypass : bool, optional
        Internal use only. If :obj:`True`, bypasses certain checks for performance, by default :obj:`False`.

    Returns
    -------
    :class:`Mesh`
        A :class:`Mesh` instance containing the mesh data.


    Example
    -------
        
    .. code-block:: python

        import open3d as o3d
        from pysdic import Mesh, from_open3d

        # Read the mesh from a file
        o3dmesh = o3d.io.read_triangle_mesh("path/to/mesh.ply")

        # Create a Mesh instance from the Open3D object
        mesh = from_open3d(o3dmesh)

    """
    if not isinstance(mesh, (open3d.t.geometry.TriangleMesh, open3d.geometry.TriangleMesh)):
        raise TypeError(f"Expected an Open3D TriangleMesh object, got {type(mesh)}.")

    if isinstance(mesh, open3d.geometry.TriangleMesh): # Legacy Open3D mesh
        vertices = numpy.asarray(mesh.vertices, dtype=numpy.float64)
        triangles = numpy.asarray(mesh.triangles, dtype=numpy.int64)
        mesh_instance = Mesh(vertices=PointCloud.from_array(vertices), connectivity=triangles, elements_type="triangle_3", internal_bypass=internal_bypass)
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
        mesh_instance = Mesh(vertices=vertices, connectivity=triangles, elements_type="triangle_3", internal_bypass=internal_bypass)
        mesh_instance.validate()  # Validate the mesh structure

        # Check if UV mapping is available
        if any(key == "texture_uvs" for key, _ in mesh.triangle.items()):
            uvmap = numpy.asarray(mesh.triangle.texture_uvs.numpy(), dtype=numpy.float64)
            # Convert UV map to the format (M, 6) - u1, v1, u2, v2, u3, v3
            uvmap = uvmap.reshape(-1, 6)
            mesh_instance.elements_uvmap = uvmap

    return mesh_instance


def triangle_3_mesh_to_open3d(mesh: Mesh, legacy: bool = False, uvmap: bool = True) -> Union[open3d.t.geometry.TriangleMesh, open3d.geometry.TriangleMesh]:
    r"""
    Convert the :class:`Mesh` (:obj:`elements_type` = "triangle_3") instance to an Open3D TriangleMesh object. (Only for 3D embedding dimension meshes :math:`E=3`)

    The mesh must not be empty.

    If :obj:`legacy` is :obj:`True`, the method returns a legacy Open3D TriangleMesh object.
    Otherwise, it returns a T geometry TriangleMesh object.

    .. warning::

        For now, the method only converts the vertices, triangles, and UV map (if available) to the Open3D mesh.
        The other properties stored in the :class:`Triangle3Mesh` instance are not transferred.

    .. seealso::

        - :func:`triangle_3_mesh_from_open3d` for creating a :class:`Mesh` instance from an Open3D TriangleMesh object.

    Parameters
    ----------
    mesh : :class:`Mesh`
        A :class:`Mesh` instance containing the mesh data.

    legacy : :class:`bool`, optional
        If :obj:`True`, return a legacy Open3D TriangleMesh object. Default is :obj:`False`.

    uvmap : :class:`bool`, optional
        If :obj:`True`, include the UV mapping in the Open3D mesh if available. Default is :obj:`True`.

    Returns
    -------
    Union[:class:`open3d.t.geometry.TriangleMesh`, :class:`open3d.geometry.TriangleMesh`]
        An Open3D TriangleMesh object containing the mesh data.

    Raises
    ------
    ValueError
        If the mesh is empty.   


    Example
    -------

    .. code-block:: python

        import open3d as o3d
        from pysdic import Mesh, to_open3d

        # Create a Mesh instance
        mesh = Mesh(vertices=..., connectivity=...)
        
        # Convert the mesh to an Open3D object
        o3dmesh = to_open3d(mesh)

    """
    if not mesh.elements_type == "triangle_3":
        raise TypeError("to_open3d function only supports 'triangle_3' element type meshes.")
    if mesh.n_vertices == 0 or mesh.n_elements == 0:
        raise ValueError("Cannot write an empty mesh to file.")
    if mesh.n_dimensions != 3:
        raise ValueError("Only 3D embedding dimension meshes can be converted to Open3D TriangleMesh.")
    if not isinstance(legacy, bool):
        raise TypeError(f"Expected a boolean for legacy, got {type(legacy)}.")
    if not isinstance(uvmap, bool):
        raise TypeError(f"Expected a boolean for uvmap, got {type(uvmap)}.")

    if legacy:
        o3d_mesh = open3d.geometry.TriangleMesh()
        o3d_mesh.vertices = open3d.utility.Vector3dVector(mesh.vertices.to_array())
        o3d_mesh.triangles = open3d.utility.Vector3iVector(mesh.connectivity)

        # Check if UV mapping is available
        if mesh.elements_uvmap is not None and uvmap:
            uvmap = mesh.elements_uvmap.reshape(-1, 2)
            o3d_mesh.triangle_uvs = open3d.utility.Vector2dVector(uvmap)

    else:
        o3d_mesh = open3d.t.geometry.TriangleMesh()
        o3d_mesh.vertex.positions = open3d.core.Tensor(mesh.vertices.to_array(), dtype=open3d.core.float32)
        o3d_mesh.triangle.indices = open3d.core.Tensor(mesh.connectivity, dtype=open3d.core.int32)

        # Check if UV mapping is available
        if mesh.elements_uvmap is not None and uvmap:
            uvmap = mesh.elements_uvmap.reshape(mesh.n_elements, 3, 2)  # Reshape to (M, 3, 2) for Open3D T geometry
            o3d_mesh.triangle.texture_uvs = open3d.core.Tensor(uvmap, dtype=open3d.core.float32)

    return o3d_mesh




def triangle_3_compute_elements_areas(
    vertices_coordinates: numpy.ndarray,
    connectivity: numpy.ndarray,
) -> numpy.ndarray:
    r"""
    Compute the areas of triangular elements defined by the given vertices and connectivity.

    The area of each triangular element is computed using the cross product of two edges of the triangle.

    .. math::

        \text{Area} = \frac{1}{2} \| \vec{AB} \times \vec{AC} \|

    .. seealso::

        - :func:`triangle_3_compute_elements_normals` for computing the normal vectors of triangular elements.
        - :func:`triangle_3_compute_vertices_normals` for computing the normal vectors at each vertex of the mesh.

    Parameters
    ----------
    vertices_coordinates : :class:`numpy.ndarray`
        An array of shape (:math:`N_v`, 3) representing the coordinates of the vertices.

    connectivity : :class:`numpy.ndarray`
        An array of shape (:math:`N_e`, 3) representing the connectivity of the triangular elements.

    Returns
    -------
    :class:`numpy.ndarray`
        An array of shape (:math:`N_e`,) representing the area of each triangular element.

    Raises
    ------
    ValueError
        If the input arrays do not have the correct shapes.

    
    Example
    -------

    .. code-block:: python

        import numpy as np
        from pysdic import triangle_3_compute_elements_areas

        vertices = np.array([[0, 0, 0],
                             [1, 0, 0],
                             [0, 1, 0],
                             [0, 0, 1]])  # shape (4, 3)

        connectivity = np.array([[0, 1, 2],
                                 [0, 1, 3],
                                 [0, 2, 3],
                                 [1, 2, 3]])  # shape (4, 3)

        areas = triangle_3_compute_elements_areas(vertices, connectivity)
        print(areas)
        
    The output will be:

    .. code-block:: console

        [0.5, 0.5, 0.5, 0.8660254]


    """
    vertices_coordinates = numpy.asarray(vertices_coordinates)
    if not numpy.issubdtype(vertices_coordinates.dtype, numpy.floating):
        vertices_coordinates = vertices_coordinates.astype(numpy.float64)
    
    connectivity = numpy.asarray(connectivity)
    if not numpy.issubdtype(connectivity.dtype, numpy.integer):
        connectivity = connectivity.astype(numpy.int64)

    if vertices_coordinates.ndim != 2 or vertices_coordinates.shape[1] != 3:
        raise ValueError(f"vertices_coordinates must be of shape (N_v, 3), got {vertices_coordinates.shape}.")
    
    if connectivity.ndim != 2 or connectivity.shape[1] != 3:
        raise ValueError(f"connectivity must be of shape (N_e, 3), got {connectivity.shape}.")
    
    v0 = vertices_coordinates[connectivity[:, 0], :]
    v1 = vertices_coordinates[connectivity[:, 1], :]
    v2 = vertices_coordinates[connectivity[:, 2], :]

    # Compute the vectors for two edges of each triangle
    edge1 = v1 - v0
    edge2 = v2 - v0

    # Compute the cross product of the two edge vectors
    cross_product = numpy.cross(edge1, edge2)

    # Compute the area of each triangle (half the magnitude of the cross product)
    areas = 0.5 * numpy.linalg.norm(cross_product, axis=1)

    return areas


def triangle_3_compute_elements_normals(
    vertices_coordinates: numpy.ndarray,
    connectivity: numpy.ndarray,
) -> numpy.ndarray:
    r"""
    Compute the normal vectors of triangular elements defined by the given vertices and connectivity.

    The normal vector of each triangular element is computed using the cross product of two edges of the triangle
    and is normalized to have unit length.

    .. math::

        \vec{n} = \frac{\vec{AB} \times \vec{AC}}{\| \vec{AB} \times \vec{AC} \|}

    .. seealso::

        - :func:`triangle_3_compute_elements_areas` for computing the areas of triangular elements.
        - :func:`triangle_3_compute_vertices_normals` for computing the normal vectors at each vertex of the mesh.

    Parameters
    ----------
    vertices_coordinates : :class:`numpy.ndarray`
        An array of shape (:math:`N_v`, 3) representing the coordinates of the vertices.

    connectivity : :class:`numpy.ndarray`
        An array of shape (:math:`N_e`, 3) representing the connectivity of the triangular elements.

    Returns
    -------
    :class:`numpy.ndarray`
        An array of shape (:math:`N_e`, 3) representing the normal vector of each triangular element.

    Raises
    ------
    ValueError
        If the input arrays do not have the correct shapes.


    Example
    -------
    
    .. code-block:: python

        import numpy as np
        from pysdic import triangle_3_compute_elements_normals

        vertices = np.array([[0, 0, 0],
                             [1, 0, 0],
                             [0, 1, 0],
                             [0, 0, 1]])  # shape (4, 3)

        connectivity = np.array([[0, 1, 2],
                                 [0, 1, 3],
                                 [0, 2, 3],
                                 [1, 2, 3]])  # shape (4, 3)

        normals = triangle_3_compute_elements_normals(vertices, connectivity)
        print(normals)

    The output will be:

    .. code-block:: console

        [[ 0.  0.  1.]
         [ 0. -1.  0.]
         [ 1.  0.  0.]
         [ 0.57735027  0.57735027  0.57735027]]

    """
    vertices_coordinates = numpy.asarray(vertices_coordinates)
    if not numpy.issubdtype(vertices_coordinates.dtype, numpy.floating):
        vertices_coordinates = vertices_coordinates.astype(numpy.float64)

    connectivity = numpy.asarray(connectivity)
    if not numpy.issubdtype(connectivity.dtype, numpy.integer):
        connectivity = connectivity.astype(numpy.int64)

    if vertices_coordinates.ndim != 2 or vertices_coordinates.shape[1] != 3:
        raise ValueError(f"vertices_coordinates must be of shape (N_v, 3), got {vertices_coordinates.shape}.")
    
    if connectivity.ndim != 2 or connectivity.shape[1] != 3:
        raise ValueError(f"connectivity must be of shape (N_e, 3), got {connectivity.shape}.")
    
    v0 = vertices_coordinates[connectivity[:, 0], :]
    v1 = vertices_coordinates[connectivity[:, 1], :]
    v2 = vertices_coordinates[connectivity[:, 2], :]

    # Compute the vectors for two edges of each triangle
    edge1 = v1 - v0
    edge2 = v2 - v0

    # Compute the cross product of the two edge vectors to get the normal vector
    normals = numpy.cross(edge1, edge2)

    # Normalize the normal vectors to have unit length
    norms = numpy.linalg.norm(normals, axis=1, keepdims=True)

    norms[norms <= 1e-10] = 1.0  # avoid division by zero
    normals = normals / norms

    return normals


def triangle_3_compute_vertices_normals(
    vertices_coordinates: numpy.ndarray,
    connectivity: numpy.ndarray,
) -> numpy.ndarray:
    r"""
    Compute the normal vectors at each vertex of a triangular mesh.

    The normal vector at each vertex is computed as the average of the normal vectors
    of the adjacent triangular elements, weighted by the area of each element.

    .. math::

        \vec{n}_v = \frac{\sum_{e \in \text{adj}(v)} \text{Area}_e \cdot \vec{n}_e}{\| \sum_{e \in \text{adj}(v)} \text{Area}_e \cdot \vec{n}_e \|}

    .. seealso::

        - :func:`triangle_3_compute_elements_areas` for computing the areas of triangular elements.
        - :func:`triangle_3_compute_elements_normals` for computing the normal vectors of triangular elements.

    Parameters
    ----------
    vertices_coordinates : :class:`numpy.ndarray`
        An array of shape (:math:`N_v`, 3) representing the coordinates of the vertices.

    connectivity : :class:`numpy.ndarray`
        An array of shape (:math:`N_e`, 3) representing the connectivity of the triangular elements.

    Returns
    -------
    :class:`numpy.ndarray`
        An array of shape (:math:`N_v`, 3) representing the normal vectors at each vertex.

    Raises
    ------
    ValueError
        If the input arrays do not have the correct shapes.

    
    Example
    -------

    .. code-block:: python

        import numpy as np
        from pysdic import triangle_3_compute_vertices_normals

        vertices = np.array([[0, 0, 0],
                             [1, 0, 0],
                             [0, 1, 0],
                             [0, 0, 1]])  # shape (4, 3)

        connectivity = np.array([[0, 1, 2],
                                 [0, 1, 3],
                                 [0, 2, 3],
                                 [1, 2, 3]])  # shape (4, 3)

        vertex_normals = triangle_3_compute_vertices_normals(vertices, connectivity)
        print(vertex_normals)

    The output will be:

    .. code-block:: console

        [[ 0.57735027 -0.57735027  0.57735027]
         [ 0.4472136   0.          0.89442719]
         [ 0.66666667  0.33333333  0.66666667]
         [ 0.89442719  0.          0.4472136 ]]

    """
    vertices_coordinates = numpy.asarray(vertices_coordinates)
    if not numpy.issubdtype(vertices_coordinates.dtype, numpy.floating):
        vertices_coordinates = vertices_coordinates.astype(numpy.float64)

    connectivity = numpy.asarray(connectivity)
    if not numpy.issubdtype(connectivity.dtype, numpy.integer):
        connectivity = connectivity.astype(numpy.int64)

    if vertices_coordinates.ndim != 2 or vertices_coordinates.shape[1] != 3:
        raise ValueError(f"vertices_coordinates must be of shape (N_v, 3), got {vertices_coordinates.shape}.")
    
    if connectivity.ndim != 2 or connectivity.shape[1] != 3:
        raise ValueError(f"connectivity must be of shape (N_e, 3), got {connectivity.shape}.")

    # Compute element normals and areas
    element_normals = triangle_3_compute_elements_normals(vertices_coordinates, connectivity)  # (N_e, 3)
    element_areas = triangle_3_compute_elements_areas(vertices_coordinates, connectivity).reshape(-1, 1)  # (N_e, 1)

    # Initialize vertex normals
    n_vertices = vertices_coordinates.shape[0]
    vertex_normals = numpy.zeros((n_vertices, 3), dtype=numpy.float64)

    # Weighted normals
    weighted_normals = element_normals * element_areas  # (N_e, 3)

    # For each triangle, repeat the normals 3 times (one per vertex)
    repeated_normals = numpy.repeat(weighted_normals, 3, axis=0)  # (N_e*3, 3)
    vertex_indices = connectivity.reshape(-1)  # (N_e*3,)

    # Accumulate with numpy.add.at
    numpy.add.at(vertex_normals, vertex_indices, repeated_normals)

    # Normalize to unit length
    norms = numpy.linalg.norm(vertex_normals, axis=1, keepdims=True)
    norms[norms <= 1e-10] = 1.0  # avoid division by zero
    vertex_normals /= norms

    return vertex_normals


def triangle_3_cast_rays(
    vertices_coordinates: numpy.ndarray,
    connectivity: numpy.ndarray,
    ray_origins: numpy.ndarray,
    ray_directions: numpy.ndarray,
    nan_open3d_errors: bool = True,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    r"""
    Cast rays into a triangular mesh and compute the intersection points.

    This function uses Open3D to perform ray-mesh intersection tests.

    Parameters
    ----------
    vertices_coordinates : :class:`numpy.ndarray`
        An array of shape (:math:`N_v`, 3) representing the coordinates of the vertices.

    connectivity : :class:`numpy.ndarray`
        An array of shape (:math:`N_e`, 3) representing the connectivity of the triangular elements.

    ray_origins : :class:`numpy.ndarray`
        An array of shape (:math:`N_r`, 3) representing the origins of the rays.

    ray_directions : :class:`numpy.ndarray`
        An array of shape (:math:`N_r`, 3) representing the directions of the rays.

    Returns
    -------
    natural_coordinates : :class:`numpy.ndarray`
        An array of shape (:math:`N_r`, 2) representing the natural coordinates (:math:`\xi, \eta`) of the intersection points.

    element_indices : :class:`numpy.ndarray`
        An array of shape (:math:`N_r`,) representing the indices of the intersected elements. If a ray does not intersect any element, the index is -1.

    nan_open3d_errors : :class:`bool`, optional
        If :obj:`True`, handle NaN errors due to Open3D float32 precision issues by setting invalid intersections to NaN and -1, by default :obj:`True`.

    Raises
    ------
    ValueError
        If the input arrays do not have the correct shapes.

    
    Example
    -------
    
    .. code-block:: python

        import numpy as np
        from pysdic import triangle_3_cast_rays

        vertices = np.array([[0, 0, 0],
                             [1, 0, 0],
                             [0, 1, 0],
                             [0, 0, 1]])  # shape (4, 3)

        connectivity = np.array([[0, 1, 2],
                                 [0, 1, 3],
                                 [0, 2, 3],
                                 [1, 2, 3]])  # shape (4, 3)

        ray_origins = np.array([[0.1, 0.1, -1],
                                [0.5, 0.5, -1]])  # shape (2, 3)

        ray_directions = np.array([[0, 0, 1],
                                   [0, 0, 1]])  # shape (2, 3)

        natural_coords, element_indices = triangle_3_cast_rays(vertices, connectivity, ray_origins, ray_directions)
        print(natural_coords)
        print(element_indices)

    The output will be:

    .. code-block:: console

        [[0.1 0.1]
         [0.5 0.5]]

        [0 0]

    """
    vertices_coordinates = numpy.asarray(vertices_coordinates)
    if not numpy.issubdtype(vertices_coordinates.dtype, numpy.floating):
        vertices_coordinates = vertices_coordinates.astype(numpy.float64)

    connectivity = numpy.asarray(connectivity)
    if not numpy.issubdtype(connectivity.dtype, numpy.integer):
        connectivity = connectivity.astype(numpy.int64)

    if vertices_coordinates.ndim != 2 or vertices_coordinates.shape[1] != 3:
        raise ValueError(f"vertices_coordinates must be of shape (N_v, 3), got {vertices_coordinates.shape}.")
    
    if connectivity.ndim != 2 or connectivity.shape[1] != 3:
        raise ValueError(f"connectivity must be of shape (N_e, 3), got {connectivity.shape}.")
    
    ray_origins = numpy.asarray(ray_origins)
    if not numpy.issubdtype(ray_origins.dtype, numpy.floating):
        ray_origins = ray_origins.astype(numpy.float64)
    
    ray_directions = numpy.asarray(ray_directions)
    if not numpy.issubdtype(ray_directions.dtype, numpy.floating):
        ray_directions = ray_directions.astype(numpy.float64)

    if ray_origins.ndim != 2 or ray_origins.shape[1] != 3:
        raise ValueError(f"ray_origins must be of shape (N_r, 3), got {ray_origins.shape}.")
    
    if ray_directions.ndim != 2 or ray_directions.shape[1] != 3:
        raise ValueError(f"ray_directions must be of shape (N_r, 3), got {ray_directions.shape}.")
    if ray_origins.shape[0] != ray_directions.shape[0]:
        raise ValueError(f"ray_origins and ray_directions must have the same number of rays, got {ray_origins.shape[0]} and {ray_directions.shape[0]}.")

    n_rays = ray_origins.shape[0]

    # Extract the Open3D mesh for the specified frame
    o3d_mesh = triangle_3_mesh_to_open3d(Mesh(
        vertices=vertices_coordinates,
        connectivity=connectivity,
        elements_type="triangle_3",
        internal_bypass=True,
    ), legacy=False, uvmap=False)

    # Convert numpy arrays to Open3D point clouds (ray origins and directions)
    rays_o3d = open3d.core.Tensor(numpy.concatenate((ray_origins, ray_directions), axis=-1).astype(numpy.float32), open3d.core.float32)  # Shape: (..., 6)

    # Create the scene and add the mesh
    raycaster = open3d.t.geometry.RaycastingScene()
    raycaster.add_triangles(o3d_mesh)

    # Cast the rays
    results = raycaster.cast_rays(rays_o3d)

    # Prepare output arrays
    natural_coordinates = numpy.full((n_rays, 2), numpy.nan, dtype=numpy.float64)
    element_indices = numpy.full((n_rays,), -1, dtype=int)

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

    return natural_coordinates, element_indices


def triangle_3_extract_unique_edges(
    connectivity: numpy.ndarray,
) -> numpy.ndarray:
    r"""
    Extract the unique edges from the triangular mesh connectivity.

    Each edge is represented as a pair of vertex indices, sorted in ascending order.
    The method returns a numpy ndarray of shape (:math:`N_{ed}`, 2) where :math:`N_{ed}` is the number of unique edges.

    Parameters
    ----------
    connectivity : :class:`numpy.ndarray`
        An array of shape (:math:`N_e`, 3) representing the connectivity of the triangular elements.

    Returns
    -------
    :class:`numpy.ndarray`
        An array of shape (:math:`N_{ed}`, 2) where :math:`N_{ed}` is the number of unique edges, representing the vertex indices of each edge.

    
    Example
    -------
    
    .. code-block:: python

        import numpy as np
        from pysdic import triangle_3_extract_unique_edges

        connectivity = np.array([[0, 1, 2],
                                 [1, 3, 2]])  # shape (2, 3)

        edges = triangle_3_extract_unique_edges(connectivity)
        print(edges)

    The output will be:

    .. code-block:: console

        [[0 1]
         [0 2]
         [1 2]
         [1 3]
         [2 3]]

    """
    connectivity = numpy.asarray(connectivity)
    if not numpy.issubdtype(connectivity.dtype, numpy.integer):
        connectivity = connectivity.astype(numpy.int64)

    if connectivity.ndim != 2 or connectivity.shape[1] != 3:
        raise ValueError(f"connectivity must be of shape (N_e, 3), got {connectivity.shape}.")
    
    if connectivity.shape[0] == 0:
        return numpy.empty((0, 2), dtype=numpy.int64)
    
    # Extract edges from triangles
    edges = numpy.vstack((
        connectivity[:, [0, 1]],
        connectivity[:, [1, 2]],
        connectivity[:, [2, 0]]
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



def _bfs_distance(graph: dict, start_index: int, n_vertices: int) -> numpy.ndarray:
    r"""
    Perform a breadth-first search (BFS) to compute the shortest path distances from a starting vertex to all other vertices in the graph.

    .. warning::

        This is an internal function and should not be called directly.

    Parameters
    ----------
    graph : :class:`dict`
        A dictionary representing the adjacency list of the graph, where keys are vertex indices and values are lists of adjacent vertex indices.

    start_index : :class:`int`
        The starting vertex index for the BFS.

    n_points : :class:`int`
        The total number of vertices in the graph.

    Returns
    -------
    :class:`numpy.ndarray`
        An array of shape (:math:`N_v`,) representing the shortest path distances from the starting vertex to all other vertices.

    """
    # Breadth-first search (BFS) to find shortest paths from vertex i
    distances = [-1] * n_vertices  # -1 indicates unvisited vertices
    distances[start_index] = 0
    queue = deque([start_index])

    while queue:
        node = queue.popleft()
        current_distance = distances[node]
        
        for neighbor in graph[node]:
            if distances[neighbor] == -1:  # Not visited neighbor
                distances[neighbor] = current_distance + 1
                queue.append(neighbor)

    return numpy.array(distances, dtype=numpy.int64)



def triangle_3_build_vertices_adjacency_matrix(
    edges: numpy.ndarray,
    n_vertices: Optional[int] = None,
) -> numpy.ndarray:
    r"""
    Build the vertices path distance matrix (adjacency matrix) from the edges of a triangular mesh.

    The distance between two vertices is defined as the minimum number of edges that must be traversed to go from one vertex to another.
    The output is a symmetric square matrix of shape (:math:`N_v`, :math:`N_v`), where :math:`N_v` is the number of vertices.

    .. seealso::

        - :func:`triangle_3_build_elements_adjacency_matrix` for building the elements path distance matrix.

    Parameters
    ----------
    edges : numpy.ndarray
        An array of shape (:math:`N_{ed}`, 2) representing the vertex indices.

    n_vertices : Optional[int], optional
        The total number of vertices in the mesh. If :obj:`None`, it is inferred from the maximum vertex index in the edges, by default :obj:`None`.

    Returns
    -------
    numpy.ndarray
        A square array of shape (:math:`N_v`, :math:`N_v`) representing the path distance matrix between vertices.


    Example
    -------
    .. code-block:: python

        import numpy as np
        from pysdic import triangle_3_build_vertices_adjacency_matrix

        edges = np.array([[0, 1],
                          [1, 2],
                          [2, 3],
                          [3, 4],
                          [1, 3]])  # shape (5, 2)

        distance_matrix = triangle_3_build_vertices_adjacency_matrix(edges)
        print(distance_matrix)

    The output will be:

    .. code-block:: console

        [[0 1 2 2 3]
         [1 0 1 1 2]
         [2 1 0 1 2]
         [2 1 1 0 1]
         [3 2 2 1 0]]
        
    """
    edges = numpy.asarray(edges)
    if not numpy.issubdtype(edges.dtype, numpy.integer):
        edges = edges.astype(numpy.int64)

    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"edges must be of shape (N_ed, 2), got {edges.shape}.")
    
    if n_vertices is not None and (not isinstance(n_vertices, int) or n_vertices <= 0):
        raise ValueError(f"n_vertices must be a positive integer or None, got {n_vertices}.")
    
    if edges.shape[0] == 0:
        if n_vertices is None:
            return numpy.empty((0, 0), dtype=numpy.int64)
        else:
            return numpy.zeros((n_vertices, n_vertices), dtype=numpy.int64)
        
    if n_vertices is None:
        n_vertices = numpy.max(edges) + 1
    
    # Build the graph adjacency matrix
    graph = {}
    for (u, v) in edges:
        if u not in graph:
            graph[u] = []
        if v not in graph:
            graph[v] = []
        graph[u].append(v)
        graph[v].append(u)

    # Initialize the distance matrix with zeros
    distance_matrix = numpy.zeros((n_vertices, n_vertices), dtype=numpy.int64)

    # Fill the upper triangle of the distance matrix
    for i in range(n_vertices):
        distances = _bfs_distance(graph, i, n_vertices)
        distance_matrix[i, :] = distances

    # Make the distance matrix symmetric
    distance_matrix = numpy.maximum(distance_matrix, distance_matrix.T)

    return distance_matrix


def triangle_3_build_elements_adjacency_matrix(
    connectivity: numpy.ndarray,
) -> numpy.ndarray:
    r"""
    Build the elements path distance matrix (adjacency matrix) from the connectivity of a triangular mesh.

    The distance between two elements is defined as the minimum number of shared edges that must be traversed to go from one element to another.
    The output is a symmetric square matrix of shape (:math:`N_e`, :math:`N_e`), where :math:`N_e` is the number of elements.

    .. seealso::

        - :func:`triangle_3_build_vertices_adjacency_matrix` for building the vertices path distance matrix.

    Parameters
    ----------
    connectivity : numpy.ndarray
        An array of shape (:math:`N_e`, 3) representing the connectivity of the triangular elements.

    Returns
    -------
    numpy.ndarray
        A square array of shape (:math:`N_e`, :math:`N_e`) representing the path distance matrix between elements.


    Example
    -------
    .. code-block:: python

        import numpy as np
        from pysdic import triangle_3_build_elements_adjacency_matrix

        connectivity = np.array([[0, 1, 2],
                                 [1, 2, 3],
                                 [2, 3, 4],
                                 [0, 2, 4]])  # shape (4, 3)

        distance_matrix = triangle_3_build_elements_adjacency_matrix(connectivity)
        print(distance_matrix)

    The output will be:

    .. code-block:: console

        [[0 1 2 1]
         [1 0 1 2]
         [2 1 0 3]
         [1 2 3 0]]
        
    """
    connectivity = numpy.asarray(connectivity)
    if not numpy.issubdtype(connectivity.dtype, numpy.integer):
        connectivity = connectivity.astype(numpy.int64)

    if connectivity.ndim != 2 or connectivity.shape[1] != 3:
        raise ValueError(f"connectivity must be of shape (N_e, 3), got {connectivity.shape}.")
    
    n_elements = connectivity.shape[0]
    if n_elements == 0:
        return numpy.empty((0, 0), dtype=numpy.int64)

    # Build element adjacency graph based on shared edges
    graph = {i: [] for i in range(n_elements)}

    for i in range(n_elements):
        vertices = set(connectivity[i])
        for j in range(i + 1, n_elements):
            other_vertices = set(connectivity[j])
            if len(vertices & other_vertices) == 2:  # Shared edge
                graph[i].append(j)
                graph[j].append(i)
        
    # Initialize the distance matrix with zeros
    distance_matrix = numpy.zeros((n_elements, n_elements), dtype=numpy.int64)

    # Fill the upper triangle of the distance matrix
    for i in range(n_elements):
        distances = _bfs_distance(graph, i, n_elements)
        distance_matrix[i, :] = distances

    # Make the distance matrix symmetric
    distance_matrix = numpy.maximum(distance_matrix, distance_matrix.T)

    return distance_matrix