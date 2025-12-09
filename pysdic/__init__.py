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

# -------------------------------------------------
# Shape functions manipulation
# -------------------------------------------------

from .core.objects.point_cloud import PointCloud
__all__.extend([
    "PointCloud",
])

from .core.objects.mesh import Mesh
__all__.extend([
    "Mesh",
])

from .core.objects.integration_points import IntegrationPoints
__all__.extend([
    "IntegrationPoints",
])

from .core.objects.image import Image
__all__.extend([
    "Image",
])

from .core.objects.camera import Camera
__all__.extend([
    "Camera",
])

from .core.objects.projection_result import ProjectionResult
__all__.extend([
    "ProjectionResult",
])

from .core.objects.image_projection_result import ImageProjectionResult
__all__.extend([
    "ImageProjectionResult",
])

from .core.objects.view import View
__all__.extend([
    "View",
])

from .core.shape_functions import (
    segment_2_shape_functions,
    segment_3_shape_functions,
    triangle_3_shape_functions,
    triangle_6_shape_functions,
    quadrangle_4_shape_functions,
    quadrangle_8_shape_functions,
)
__all__.extend([
    "segment_2_shape_functions",
    "segment_3_shape_functions",
    "triangle_3_shape_functions",
    "triangle_6_shape_functions",
    "quadrangle_4_shape_functions",
    "quadrangle_8_shape_functions",
])


from .core.gauss_points import (
    segment_2_gauss_points,
    segment_3_gauss_points,
    triangle_3_gauss_points,
    triangle_6_gauss_points,
    quadrangle_4_gauss_points,
    quadrangle_8_gauss_points,
)
__all__.extend([
    "segment_2_gauss_points",
    "segment_3_gauss_points",
    "triangle_3_gauss_points",
    "triangle_6_gauss_points",
    "quadrangle_4_gauss_points",
    "quadrangle_8_gauss_points",
])


from .core.integration_points_operations import (
    assemble_shape_function_matrix,
    construct_jacobian,
    derivate_property,
    interpolate_property,
    project_property_to_vertices,
    remap_vertices_coordinates,
)
__all__.extend([
    "assemble_shape_function_matrix",
    "construct_jacobian",
    "derivate_property",
    "interpolate_property",
    "project_property_to_vertices",
    "remap_vertices_coordinates",
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

from .core.create_3D_surface_meshes import (
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
    triangle_3_extract_unique_edges,
    triangle_3_build_vertices_adjacency_matrix,
    triangle_3_build_elements_adjacency_matrix
)
__all__.extend([
    "triangle_3_mesh_from_open3d",
    "triangle_3_mesh_to_open3d",
    "triangle_3_compute_elements_areas",
    "triangle_3_compute_elements_normals",
    "triangle_3_compute_vertices_normals",
    "triangle_3_cast_rays",
    "triangle_3_extract_unique_edges",
    "triangle_3_build_vertices_adjacency_matrix",
    "triangle_3_build_elements_adjacency_matrix"
])


from .core.photometric_operations import (
    compute_bouguer_law,
)
__all__.extend([
    "compute_bouguer_law",
])


from .core.BRDF_models import (
    compute_BRDF_ward,
    compute_BRDF_beckmann,
)
__all__.extend([
    "compute_BRDF_ward",
    "compute_BRDF_beckmann",
])


from . import sdic