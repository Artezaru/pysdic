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

__all__ = []

from .point_cloud_3d import PointCloud3D
__all__.extend(['PointCloud3D'])

from .integration_points import IntegrationPoints
__all__.extend(['IntegrationPoints'])

from .mesh_3d import Mesh3D
__all__.extend(['Mesh3D'])

from .surface_mesh_3d import SurfaceMesh3D
__all__.extend(['SurfaceMesh3D'])

from .linear_triangle_mesh_3d import LinearTriangleMesh3D
__all__.extend(['LinearTriangleMesh3D'])

from .quadratic_triangle_mesh_3d import QuadraticTriangleMesh3D
__all__.extend(['QuadraticTriangleMesh3D'])

from .create_linear_triangle_axisymmetric import create_linear_triangle_axisymmetric
from .create_linear_triangle_heightmap import create_linear_triangle_heightmap
__all__.extend(['create_linear_triangle_axisymmetric', 'create_linear_triangle_heightmap'])












from .shape_functions import (
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

from .integrated_points_operations import (
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

from .point_cloud import PointCloud

__all__.extend([
    "PointCloud",
])

from .mesh_ABC import (
    Mesh,
    SurfaceMesh,
)
__all__.extend([
    "Mesh",
    "SurfaceMesh",
])