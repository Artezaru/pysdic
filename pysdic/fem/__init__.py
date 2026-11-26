"""
pysdic - Python Stereo-Digital Image Correlation (Computer Vision
Cameras Models, Image Processing, Meshes and tools for SDIC analysis)
Copyright (C) 2026 Artezaru, artezaru.github@proton.me

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from .shape_functions import (
    compute_segment_2_shape_functions,
    compute_segment_3_shape_functions,
    compute_triangle_3_shape_functions,
    compute_triangle_6_shape_functions,
    compute_quadrangle_4_shape_functions,
    compute_quadrangle_8_shape_functions,
    compute_shape_functions,
)

from .gauss_points import (
    get_segment_2_gauss_points,
    get_segment_3_gauss_points,
    get_triangle_3_gauss_points,
    get_triangle_6_gauss_points,
    get_quadrangle_4_gauss_points,
    get_quadrangle_8_gauss_points,
    get_gauss_points,
)

from .operations import (
    assemble_shape_function_matrix,
    compute_shape_function_matrix,
    remap_vertices_coordinates,
    assemble_jacobian_matrix,
    compute_jacobian_matrix,
    assemble_property_derivative,
    compute_property_derivative,
    assemble_property_interpolation,
    compute_property_interpolation,
    assemble_property_projection,
    compute_property_projection,
)

from .displacement_jacobian import assemble_displacement_jacobian_matrix

from .adjacency import (
    bfs_distance,
    bfs_neighborhood,
    assemble_vertices_adjacency_graph,
    assemble_elements_adjacency_graph,
    compute_distance_matrix,
    compute_vertices_distance_matrix,
    compute_elements_distance_matrix,
    compute_neighborhood,
    compute_vertices_neighborhood,
    compute_elements_neighborhood,
    remap_integration_points_neighborhood,
    compute_integration_points_neighborhood,
    compute_neighborhood_statistics,
)

from .geometry import (
    compute_elements_areas,
    compute_integration_points_normals,
    compute_elements_normals,
    compute_vertices_normals,
    compute_strains,
    compute_elements_centers,
)

from .cast_rays import triangle_3_cast_rays, cast_rays
from .locate_points import locate_points