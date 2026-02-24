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
from numpy.typing import ArrayLike

import numpy
import open3d

from ..objects.mesh import Mesh
from ..objects.point_cloud import PointCloud
from .gauss_points import get_triangle_3_gauss_points
from .shape_functions import compute_triangle_3_shape_functions
from .integration_points_operations import compute_property_derivative


def triangle_3_mesh_from_open3d(mesh: Union[open3d.t.geometry.TriangleMesh, open3d.geometry.TriangleMesh], internal_bypass: bool = False) -> Mesh:
    r"""
    Create a :class:`Mesh` (:obj:`element_type` = "triangle_3") instance from an Open3D TriangleMesh object. (Only for 3D embedding dimension meshes :math:`E=3`)

    .. warning::
        
        For now, the method only extracts the vertices, triangles, and UV map (if available) from the Open3D mesh.
        The other properties (normals, centroids, areas) are not extracted and must be computed separately.


    Parameters
    ----------
    mesh : Union[:class:`open3d.t.geometry.TriangleMesh`, :class:`open3d.geometry.TriangleMesh`]
        An Open3D TriangleMesh object containing the mesh data.

    internal_bypass : :class:`bool`, optional
        Internal use only. If :obj:`True`, bypasses certain checks for performance, by default :obj:`False`.


    Returns
    -------
    :class:`Mesh`
        A :class:`Mesh` instance containing the mesh data.
        
        
    Raises
    ------
    TypeError
        If the input mesh is not an Open3D TriangleMesh object.
        
    ValueError
        If the mesh is empty.
        
        
    See Also
    --------
    pysdic.triangle_3_mesh_to_open3d
        For converting a :class:`Mesh` instance to an Open3D TriangleMesh object.


    Example
    -------
    Reading a mesh using Open3D and converting it to a :class:`Mesh` instance:
    
    .. code-block:: python
        :linenos:

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
        mesh_instance = Mesh(vertices=PointCloud.from_array(vertices), connectivity=triangles, element_type="triangle_3", internal_bypass=internal_bypass)
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
        mesh_instance = Mesh(vertices=vertices, connectivity=triangles, element_type="triangle_3", internal_bypass=internal_bypass)
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
    Convert the :class:`Mesh` (:obj:`element_type` = "triangle_3") instance to an Open3D TriangleMesh object. (Only for 3D embedding dimension meshes :math:`E=3`)

    If :obj:`legacy` is :obj:`True`, the method returns a legacy Open3D TriangleMesh object.
    Otherwise, it returns a T geometry TriangleMesh object.

    .. warning::

        For now, the method only converts the vertices, triangles, and UV map (if available) to the Open3D mesh.
        The other properties stored in the :class:`Triangle3Mesh` instance are not transferred.


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
    TypeError
        If the input mesh is not of type "triangle_3".
        
    ValueError
        If the mesh is empty or not 3D embedding dimension.


    See Also
    --------
    pysdic.triangle_3_mesh_from_open3d
        For converting an Open3D TriangleMesh object to a :class:`Mesh` instance.
        
        
    Examples
    --------
    Converting a :class:`Mesh` instance to an Open3D TriangleMesh object:
    
    .. code-block:: python
        :linenos:

        import open3d as o3d
        from pysdic import Mesh, to_open3d

        # Create a Mesh instance
        mesh = Mesh(vertices=..., connectivity=...)
        
        # Convert the mesh to an Open3D object
        o3dmesh = to_open3d(mesh)

    """
    if not isinstance(mesh, Mesh):
        raise TypeError(f"Expected a Mesh instance, got {type(mesh)}.")
    if not mesh.element_type == "triangle_3":
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
    vertices_coordinates: ArrayLike,
    connectivity: ArrayLike,
) -> numpy.ndarray:
    r"""
    Compute the areas of triangular elements defined by the given vertices and connectivity.

    The area of each triangular element is computed using the cross product of two edges of the triangle.

    .. math::

        \text{Area} = \frac{1}{2} \| \vec{AB} \times \vec{AC} \|
        
    .. note::

        The input :obj:`vertices_coordinates` will be converted to :obj:`numpy.float64` for computation.
        The input :obj:`connectivity` will be converted to :obj:`numpy.int64` for computation.
        
    .. warning::

        No tests are performed to check if the inputs array content are consistent (e.g., if the connectivity contains invalid vertex indices, ...).
        Only the shapes and types of the input arrays are validated. The behavior of the function is undefined if the input arrays are not consistent.
        

    Parameters
    ----------
    vertices_coordinates: ArrayLike
        An array of shape :math:`(N_v, 3)` representing the coordinates of the vertices.

    connectivity: ArrayLike
        An array of shape :math:`(N_e, 3)` representing the connectivity of the triangular elements.


    Returns
    -------
    :class:`numpy.ndarray`
        An array of shape :math:`(N_e,)` representing the area of each triangular element.


    Raises
    ------
    TypeError
        If the inputs are not numpy arrays.
        
    ValueError
        If the input arrays do not have the correct shapes.

    
    See Also
    --------
    pysdic.triangle_3_compute_elements_normals
        For computing the normal vectors of triangular elements.
        
    pysdic.triangle_3_compute_vertices_normals
        For computing the normal vectors at each vertex of the mesh.
        
        
    Examples
    --------
    Computing the areas of triangular elements:

    .. code-block:: python
        :linenos:

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

    .. code-block:: console

        [0.5, 0.5, 0.5, 0.8660254]


    """
    vertices_coordinates = numpy.asarray(vertices_coordinates, dtype=numpy.float64)
    connectivity = numpy.asarray(connectivity, dtype=numpy.int64)

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
    vertices_coordinates: ArrayLike,
    connectivity: ArrayLike,
) -> numpy.ndarray:
    r"""
    Compute the normal vectors of triangular elements defined by the given vertices and connectivity.

    The normal vector of each triangular element is computed using the cross product of two edges of the triangle
    and is normalized to have unit length.

    .. math::

        \vec{n} = \frac{\vec{AB} \times \vec{AC}}{\| \vec{AB} \times \vec{AC} \|}

    .. note::

        The input :obj:`vertices_coordinates` will be converted to :obj:`numpy.float64` for computation.
        The input :obj:`connectivity` will be converted to :obj:`numpy.int64` for computation.
        
    .. warning::

        No tests are performed to check if the inputs array content are consistent (e.g., if the connectivity contains invalid vertex indices, ...).
        Only the shapes and types of the input arrays are validated. The behavior of the function is undefined if the input arrays are not consistent.

    Parameters
    ----------
    vertices_coordinates: ArrayLike
        An array of shape :math:`(N_v, 3)` representing the coordinates of the vertices.

    connectivity: ArrayLike
        An array of shape :math:`(N_e, 3)` representing the connectivity of the triangular elements.

    Returns
    -------
    :class:`numpy.ndarray`
        An array of shape :math:`(N_e, 3)` representing the normal vector of each triangular element.


    Raises
    ------
    TypeError
        If the inputs are not numpy arrays.
        
    ValueError
        If the input arrays do not have the correct shapes.


    See Also
    --------
    pysdic.triangle_3_compute_elements_areas
        For computing the areas of triangular elements.
        
    pysdic.triangle_3_compute_vertices_normals
        For computing the normal vectors at each vertex of the mesh.
        

    Examples
    --------
    Computing the normal vectors of triangular elements:
    
    .. code-block:: python
        :linenos:
        
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

    .. code-block:: console

        [[ 0.  0.  1.]
         [ 0. -1.  0.]
         [ 1.  0.  0.]
         [ 0.57735027  0.57735027  0.57735027]]

    """
    vertices_coordinates = numpy.asarray(vertices_coordinates, dtype=numpy.float64) 
    connectivity = numpy.asarray(connectivity, dtype=numpy.int64)

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
    vertices_coordinates: ArrayLike,
    connectivity: ArrayLike,
) -> numpy.ndarray:
    r"""
    Compute the normal vectors at each vertex of a triangular mesh.

    The normal vector at each vertex is computed as the average of the normal vectors
    of the adjacent triangular elements, weighted by the area of each element.

    .. math::

        \vec{n}_v = \frac{\sum_{e \in \text{adj}(v)} \text{Area}_e \cdot \vec{n}_e}{\| \sum_{e \in \text{adj}(v)} \text{Area}_e \cdot \vec{n}_e \|}
        
    .. note::

        The input :obj:`vertices_coordinates` will be converted to :obj:`numpy.float64` for computation.
        The input :obj:`connectivity` will be converted to :obj:`numpy.int64` for computation.
        
    .. warning::

        No tests are performed to check if the inputs array content are consistent (e.g., if the connectivity contains invalid vertex indices, ...).
        Only the shapes and types of the input arrays are validated. The behavior of the function is undefined if the input arrays are not consistent.


    Parameters
    ----------
    vertices_coordinates: ArrayLike
        An array of shape :math:`(N_v, 3)` representing the coordinates of the vertices.

    connectivity: ArrayLike
        An array of shape :math:`(N_e, 3)` representing the connectivity of the triangular elements.


    Returns
    -------
    :class:`numpy.ndarray`
        An array of shape :math:`(N_v, 3)` representing the normal vectors at each vertex.


    Raises
    ------
    TypeError
        If the inputs are not numpy arrays.
        
    ValueError
        If the input arrays do not have the correct shapes.
        
        
    See Also
    --------
    pysdic.triangle_3_compute_elements_areas
        For computing the areas of triangular elements.
        
    pysdic.triangle_3_compute_elements_normals
        For computing the normal vectors of triangular elements.

    
    Examples
    --------
    Computing the normal vectors at each vertex of a triangular mesh:

    .. code-block:: python
        :linenos:
        
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
        
    .. code-block:: console

        [[ 0.57735027 -0.57735027  0.57735027]
         [ 0.4472136   0.          0.89442719]
         [ 0.66666667  0.33333333  0.66666667]
         [ 0.89442719  0.          0.4472136 ]]

    """
    vertices_coordinates = numpy.asarray(vertices_coordinates, dtype=numpy.float64) 
    connectivity = numpy.asarray(connectivity, dtype=numpy.int64)

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
    vertices_coordinates: ArrayLike,
    connectivity: ArrayLike,
    ray_origins: ArrayLike,
    ray_directions: ArrayLike,
    nan_open3d_errors: bool = True,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    r"""
    Cast rays into a triangular mesh and compute the intersection points.

    This function uses Open3D to perform ray-mesh intersection tests.
    
    .. note::

        The inputs :obj:`vertices_coordinates`, :obj:`ray_origins`, and :obj:`ray_directions` will be converted to :obj:`numpy.float64` for computation.
        The input :obj:`connectivity` will be converted to :obj:`numpy.int64` for computation.
        
    .. warning::

        No tests are performed to check if the inputs array content are consistent (e.g., if the connectivity contains invalid vertex indices, if the rays are valid, ...).
        Only the shapes and types of the input arrays are validated. The behavior of the function is undefined if the input arrays are not consistent.


    Parameters
    ----------
    vertices_coordinates: ArrayLike
        An array of shape :math:`(N_v, 3)` representing the coordinates of the vertices.

    connectivity: ArrayLike
        An array of shape :math:`(N_e, 3)` representing the connectivity of the triangular elements.

    ray_origins: ArrayLike
        An array of shape :math:`(N_r, 3)` representing the origins of the rays.

    ray_directions: ArrayLike
        An array of shape :math:`(N_r, 3)` representing the directions of the rays.
    
    nan_open3d_errors: :class:`bool`, optional
        If :obj:`True`, handle NaN errors due to Open3D float32 precision issues by setting invalid intersections to NaN and -1, by default :obj:`True`.


    Returns
    -------
    natural_coordinates: :class:`numpy.ndarray`
        An array of shape :math:`(N_r, 2)` representing the natural coordinates (:math:`\xi, \eta`) of the intersection points.
        
    element_indices: :class:`numpy.ndarray`
        An array of shape :math:`(N_r,)` representing the indices of the intersected elements. If a ray does not intersect any element, the index is -1.


    Raises
    ------
    TypeError
        If the inputs are not numpy arrays.

    ValueError
        If the input arrays do not have the correct shapes.

    
    See Also
    --------
    pysdic.triangle_3_mesh_to_open3d
        For converting a :class:`Mesh` instance to an Open3D TriangleMesh object.
        

    Examples
    --------
    Casting rays into a triangular mesh:
    
    .. code-block:: python
        :linenos:
        
        import numpy as np
        from pysdic import triangle_3_cast_rays

        vertices_coordinates = np.array([[0, 0, 0],
                             [1, 0, 0],
                             [0, 1, 0],
                             [0, 0, 1]])  # shape (Nv, E) = (4, 3)

        connectivity = np.array([[0, 1, 2],
                                 [0, 1, 3],
                                 [0, 2, 3],
                                 [1, 2, 3]])  # shape (Ne, Nvpe) = (4, 3)

        ray_origins = np.array([[0.1, 0.1, -1],
                                [0.5, 0.5, -1]])  # shape (Nr, E) = (2, 3)

        ray_directions = np.array([[0, 0, 1],
                                   [0, 0, 1]])  # shape (Nr, E) = (2, 3)

        natural_coordinates, element_indices = triangle_3_cast_rays(vertices_coordinates, connectivity, ray_origins, ray_directions)
        print(natural_coordinates) # shape (Nr, K)
        print(element_indices) # shape (Nr,)
        
    .. code-block:: console

        [[0.1 0.1]
         [0.5 0.5]]

        [0 0]

    Deduce the coordinates of the intersection points from the natural coordinates and the element connectivity.
    
    .. code-block:: python
        :linenos:
        
        from pysdic import triangle_3_shape_functions, interpolate_property
        
        shape_functions = triangle_3_shape_functions(natural_coordinates)  # shape (Nr, Nvpe) = (2, 3)
        
        points_coordinates = interpolate_property(
            vertices_coordinates, 
            shape_functions, 
            connectivity, 
            element_indices
        )  # shape (Nr, E) = (2, 3)

    """
    vertices_coordinates = numpy.asarray(vertices_coordinates, dtype=numpy.float64) 
    connectivity = numpy.asarray(connectivity, dtype=numpy.int64)

    if vertices_coordinates.ndim != 2 or vertices_coordinates.shape[1] != 3:
        raise ValueError(f"vertices_coordinates must be of shape (N_v, 3), got {vertices_coordinates.shape}.")
    
    if connectivity.ndim != 2 or connectivity.shape[1] != 3:
        raise ValueError(f"connectivity must be of shape (N_e, 3), got {connectivity.shape}.")
    
    ray_origins = numpy.asarray(ray_origins, dtype=numpy.float64)
    ray_directions = numpy.asarray(ray_directions, dtype=numpy.float64)

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
        element_type="triangle_3",
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


def triangle_3_compute_elements_strains(
    vertices_coordinates: ArrayLike,
    connectivity: ArrayLike,
    displacements: ArrayLike,
) -> numpy.ndarray:
    r"""
    Compute the strain tensor for each triangular element in the mesh given the vertex displacements.

    The strain tensor is computed using the linear strain-displacement relationship for triangular elements.

    .. math::

        \boldsymbol{\varepsilon} = \frac{1}{2} \left( \nabla \mathbf{U} + (\nabla \mathbf{U})^T \right)
        
    where :math:`\mathbf{U}` is the displacement vector.
    
    .. note::

        The inputs :obj:`vertices_coordinates` and :obj:`displacements` will be converted to :obj:`numpy.float64` for computation.
        The input :obj:`connectivity` will be converted to :obj:`numpy.int64` for computation.
        
    .. warning::

        No tests are performed to check if the inputs array content are consistent (e.g., if the connectivity contains invalid vertex indices, if the displacements are valid, ...).
        Only the shapes and types of the input arrays are validated. The behavior of the function is undefined if the input arrays are not consistent.
    
    
    Parameters
    ----------
    vertices_coordinates: ArrayLike
        An array of shape :math:`(N_v, E)` representing the coordinates of the vertices in the embedded space of dimension :math:`E`.
        
    connectivity: ArrayLike
        An array of shape :math:`(N_e, 3)` representing the connectivity of the triangular elements.
        
    displacements: ArrayLike
        An array of shape :math:`(N_v, E)` representing the displacements of the vertices in the embedded space of dimension :math:`E`.
        
        
    Returns
    -------
    :class:`numpy.ndarray`
        An array of shape :math:`(N_e, E, E)` representing the strain tensor for each triangular element along the global coordinates :math:`(x, y, z, ...)`.
        
        
    Raises
    ------
    TypeError
        If the inputs are not numpy arrays.
        
    ValueError
        If the input arrays do not have the correct shapes.
        
        
    Notes
    -----
    The strain tensor is computed as the symmetric part of the displacement gradient tensor.
    
    .. math::
    
        \boldsymbol{\varepsilon} = \frac{1}{2} \left( \nabla \mathbf{U} + (\nabla \mathbf{U})^T \right)
        
    The displacement gradient tensor :math:`\nabla \mathbf{U}` is obtained by differentiating the displacement field
    with respect to the global coordinates using the shape function derivatives for triangular elements such as:
    
    .. math::

        \nabla_X U(\xi, \eta, \zeta, ...) = J \cdot (J^T J)^{-1} \cdot \sum_{i=1}^{N_{vpe}} \nabla_{\xi} N_i(\xi, \eta, \zeta, ...) U_i

    where :math:`J` is the Jacobian matrix of the transformation from natural coordinates to global coordinates,
    :math:`N_i` are the shape functions, and :math:`U_i` are the nodal displacements.
    
    If the embedded space dimension :math:`E` is equal to the topological dimension of the element (:math:`K=2` for triangular elements),
    the strain tensor can be directly compute using Jacobian inverse : 
    
    .. math::

        \nabla_X U(\xi, \eta, \zeta, ...) = J^{-1} \cdot \sum_{i=1}^{N_{vpe}} \nabla_{\xi} N_i(\xi, \eta, \zeta, ...) U_i
    
    
    See Also
    --------
    pysdic.compute_jacobian_matrix
        For constructing the Jacobian matrix :math:`J` of the transformation from natural coordinates :math:`(\xi, \eta, \zeta, ...)` to
        global coordinates :math:`(x, y, z, ...)`.
        
    pysdic.compute_property_derivative
        Derivate a property defined at the nodes of a mesh to given integration points within elements with respect to global coordinates :math:`(x,y,z,...)` using shape function derivatives.
    
    
    Examples
    --------
    Computing the strain tensor for triangular elements:
    
    .. code-block:: python
        :linenos:

        import numpy as np
        from pysdic import triangle_3_compute_elements_strains

        vertices = np.array([[0, 0, 0],
                             [1, 0, 0],
                             [0, 1, 0],
                             [0, 0, 1]])  # shape (4, 3)

        connectivity = np.array([[0, 1, 2],
                                 [0, 1, 3],
                                 [0, 2, 3],
                                 [1, 2, 3]])  # shape (4, 3)

        displacements = np.array([[0, 0, 0],
                                  [0.1, 0, 0],
                                  [0, 0.1, 0],
                                  [0, 0, 0.1]])  # shape (4, 3)

        strains = triangle_3_compute_elements_strains(vertices, connectivity, displacements)
        print(strains)

    .. code-block:: console

        [[[ 0.1  0.   0. ]
          [ 0.   0.1  0. ]
          [ 0.   0.   0. ]]

         [[ 0.1  0.   0. ]
          [ 0.   0.   0. ]
          [ 0.   0.   0.1]]

         [[ 0.   0.   0. ]
          [ 0.   0.1  0. ]
          [ 0.   0.   0.1]]

         [[ 0.1 -0.1 -0.1]
          [-0.1  0.1 -0.1]
          [-0.1 -0.1  0.1]]]
          
    """
    vertices_coordinates = numpy.asarray(vertices_coordinates)
    if not numpy.issubdtype(vertices_coordinates.dtype, numpy.floating):
        vertices_coordinates = vertices_coordinates.astype(numpy.float64)
        
    connectivity = numpy.asarray(connectivity)
    if not numpy.issubdtype(connectivity.dtype, numpy.integer):
        connectivity = connectivity.astype(numpy.int64)
        
    displacements = numpy.asarray(displacements)
    if not numpy.issubdtype(displacements.dtype, numpy.floating):
        displacements = displacements.astype(numpy.float64)
        
    if vertices_coordinates.ndim != 2:
        raise ValueError(f"vertices_coordinates must be of shape (N_v, E), got {vertices_coordinates.shape}.")
    
    if connectivity.ndim != 2 or connectivity.shape[1] != 3:
        raise ValueError(f"connectivity must be of shape (N_e, 3), got {connectivity.shape}.")
    
    if displacements.ndim != 2 or displacements.shape != vertices_coordinates.shape:
        raise ValueError(f"displacements must be of shape (N_v, E), got {displacements.shape}.")
    
    # Get the gauss point of each triangular element.
    natural_coordinates = get_triangle_3_gauss_points() # shape (1, 2) // single gauss point at the centroid of the triangle.
    natural_coordinates = numpy.repeat(natural_coordinates, connectivity.shape[0], axis=0) # shape (N_e, 2)
    element_indices = numpy.arange(connectivity.shape[0], dtype=numpy.int64)  # shape (N_e,)
        
    # Derivate the displacements at the gauss points with respect to global coordinates
    displacement_gradients = compute_property_derivative(
        property_array=displacements,
        vertices_coordinates=vertices_coordinates,
        connectivity=connectivity,
        element_type="triangle_3",
        natural_coordinates=natural_coordinates,
        element_indices=element_indices, 
    ) # shape (N_e, N_d, E) where N_d is the size of property (here displacements N_d=E) and E is the embedded space dimension.
    
    # Compute the strain tensor as the symmetric part of the displacement gradient tensor
    strain = 0.5 * (displacement_gradients + numpy.transpose(displacement_gradients, (0, 2, 1)))  # shape (N_e, E, E)
    
    return strain
    