# Copyright 2025 Artezaru
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .__version__ import __version__

__all__ = [
    "__version__",
]

from .core.shape_functions import (
    compute_shape_functions,
    compute_segment_2_shape_functions,
    compute_segment_3_shape_functions,
    compute_triangle_3_shape_functions,
    compute_triangle_6_shape_functions,
    compute_quadrangle_4_shape_functions,
    compute_quadrangle_8_shape_functions,
)
__all__.extend([
    "compute_shape_functions",
    "compute_segment_2_shape_functions",
    "compute_segment_3_shape_functions",
    "compute_triangle_3_shape_functions",
    "compute_triangle_6_shape_functions",
    "compute_quadrangle_4_shape_functions",
    "compute_quadrangle_8_shape_functions",
])

from .core.gauss_points import (
    get_gauss_points,
    get_segment_2_gauss_points,
    get_segment_3_gauss_points,
    get_triangle_3_gauss_points,
    get_triangle_6_gauss_points,
    get_quadrangle_4_gauss_points,
    get_quadrangle_8_gauss_points,
)
__all__.extend([
    "get_gauss_points",
    "get_segment_2_gauss_points",
    "get_segment_3_gauss_points",
    "get_triangle_3_gauss_points",
    "get_triangle_6_gauss_points",
    "get_quadrangle_4_gauss_points",
    "get_quadrangle_8_gauss_points",
])

from .core.integration_points_operations import (
    assemble_shape_function_matrix,
    compute_shape_function_matrix,
    assemble_jacobian_matrix,
    compute_jacobian_matrix,
    assemble_property_derivative,
    compute_property_derivative,
    assemble_property_interpolation,
    compute_property_interpolation,
    assemble_property_projection,
    compute_property_projection,
    remap_vertices_coordinates,
)
__all__.extend([
    "assemble_shape_function_matrix",
    "compute_shape_function_matrix",
    "assemble_jacobian_matrix",
    "compute_jacobian_matrix",
    "assemble_property_derivative",
    "compute_property_derivative",
    "assemble_property_interpolation",
    "compute_property_interpolation",
    "assemble_property_projection",
    "compute_property_projection",
    "remap_vertices_coordinates",
])

from .core.adjacency_operations import (
    bfs_distance,
    bfs_neighborhood,
    create_vertices_adjacency_graph,
    create_elements_adjacency_graph,
    compute_adjacency_matrix,
    compute_vertices_adjacency_matrix,
    compute_elements_adjacency_matrix,
    compute_neighborhood,
    compute_vertices_neighborhood,
    compute_elements_neighborhood,
    compute_neighborhood_statistics,
)
__all__.extend([
    "bfs_distance",
    "bfs_neighborhood",
    "create_vertices_adjacency_graph",
    "create_elements_adjacency_graph",
    "compute_adjacency_matrix",
    "compute_vertices_adjacency_matrix",
    "compute_elements_adjacency_matrix",
    "compute_neighborhood",
    "compute_vertices_neighborhood",
    "compute_elements_neighborhood",
    "compute_neighborhood_statistics",
])

from .core.photometric_operations import (
    compute_bouguer_law,
    compute_brdf_beckmann,
    compute_brdf_ward,
)
__all__.extend([
    "compute_bouguer_law",
    "compute_brdf_beckmann",
    "compute_brdf_ward",
])

from .core.temporal_derivation import (
    compute_forward_finite_difference_coefficients,
    compute_central_finite_difference_coefficients,
    compute_backward_finite_difference_coefficients,
    apply_forward_finite_difference,
    apply_central_finite_difference,
    apply_backward_finite_difference,
    assemble_central_finite_difference_matrix,
    assemble_backward_finite_difference_matrix,
    assemble_forward_finite_difference_matrix,
)
__all__.extend([
    "compute_central_finite_difference_coefficients",
    "compute_backward_finite_difference_coefficients",
    "compute_forward_finite_difference_coefficients",
    "apply_central_finite_difference",
    "apply_backward_finite_difference",
    "apply_forward_finite_difference",
    "assemble_central_finite_difference_matrix",
    "assemble_backward_finite_difference_matrix",
    "assemble_forward_finite_difference_matrix",
])

from .core.optical_flow import (
    compute_optical_flow, 
    display_optical_flow, 
)
__all__.extend([
    "compute_optical_flow",
    "display_optical_flow",
])

from .core.build_operators import build_displacement_operator
__all__.extend([
    "build_displacement_operator",
])

from .core.textures import get_lena_texture
__all__.extend([
    "get_lena_texture",
])

from .core.create_3D_triangle_3_meshes import (
    create_triangle_3_heightmap,
    create_triangle_3_axisymmetric,
)
__all__.extend([
    "create_triangle_3_heightmap",
    "create_triangle_3_axisymmetric",
])

from .core.triangle_3_mesh_operations import (
    triangle_3_mesh_from_open3d,
    triangle_3_mesh_to_open3d,
    triangle_3_compute_elements_areas,
    triangle_3_compute_elements_normals,
    triangle_3_compute_vertices_normals,
    triangle_3_cast_rays,
    triangle_3_compute_elements_strains,
)
__all__.extend([
    "triangle_3_mesh_from_open3d",
    "triangle_3_mesh_to_open3d",
    "triangle_3_compute_elements_areas",
    "triangle_3_compute_elements_normals",
    "triangle_3_compute_vertices_normals",
    "triangle_3_cast_rays",
    "triangle_3_compute_elements_strains",
])


from .objects.point_cloud import PointCloud
__all__.extend([
    "PointCloud",
])

from .objects.connectivity import Connectivity
__all__.extend([
    "Connectivity",
])

from .objects.mesh import Mesh
__all__.extend([
    "Mesh",
])

from .objects.integration_points import IntegrationPoints
__all__.extend([
    "IntegrationPoints",
])

from .objects.image import Image
__all__.extend([
    "Image",
])

from .objects.camera import Camera
__all__.extend([
    "Camera",
])

from .objects.projection_result import ProjectionResult
__all__.extend([
    "ProjectionResult",
])

from .objects.image_projection_result import ImageProjectionResult
__all__.extend([
    "ImageProjectionResult",
])

from .objects.view import View
__all__.extend([
    "View",
])

# Import methods into blender and workflows submodules
from . import workflows as workflows
