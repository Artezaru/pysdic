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

from .__version__ import __version__

# =============================================================================
# Parametrizations and transformations
# =============================================================================
from .parametrizations.parametrization import (
    ParametrizationResult,
    Parametrization,
    ComposedParametrization,
)
from .parametrizations.affine_parametrization import AffineParametrization
from .parametrizations.subset_parametrization import SubsetParametrization

from .transformations.transformation import (
    TransformationResult,
    Transformation,
    ComposedTransformation,
)
from .transformations.reparametrized_transformation import ReparametrizedTransformation
from .transformations.rays import Rays
from .transformations.identity_transformation import IdentityTransformation
from .transformations.no_transformation import NoIntrinsic, NoDistortion, NoExtrinsic
from .transformations.pinhole_intrinsic import PinholeIntrinsic
from .transformations.brown_distortion import BrownDistortion
from .transformations.zernike_distortion import ZernikeDistortion, ZernikePolynomialsCache
from .transformations.rigid_extrinsic import (
    RigidExtrinsic,
    PerspectiveExtrinsic,
    OrthographicExtrinsic,
)
from .transformations.world_plane_rigid_extrinsic import (
    WorldPlanePerspectiveExtrinsic,
    WorldPlaneOrthographicExtrinsic,
)
from .transformations.planar_rigid_extrinsic import PlanarRigidExtrinsic
from .transformations.planar_homographic_extrinsic import PlanarHomographicExtrinsic

# =============================================================================
# Optimization of transformations
# =============================================================================
from .transformation_optimizations.optimize_transformation_parameters import (
    build_transformation_term,
    build_chains_transformation_term,
    optimize_transformation_parameters,
    optimize_chains_transformation_parameters,
)
from .transformation_optimizations.optimize_transformation_input_points import (
    build_transformation_input_points_term,
    build_chains_transformation_input_points_term,
    optimize_transformation_input_points,
    optimize_chains_transformation_input_points,
)

# =============================================================================
# Computer vision
# =============================================================================
from .computer_vision.project_points import (
    ProjectionResult,
    compose_projection,
    project_points,
)
from .computer_vision.distorting_points import (
    DistortionResult,
    distort_points,
    undistort_points,
)
from .computer_vision.distorting_images import distort_image, undistort_image
from .computer_vision.display_distortion_field import display_distortion_field
from .computer_vision.compute_rays import compute_rays
from .computer_vision.intersect_rays import intersect_rays
from .computer_vision.optical_flow import (
    compute_optical_flow,
    display_optical_flow,
    display_optical_flow_quiver,
)
from .core.disflow_config import DisflowConfig

# =============================================================================
# Resources
# =============================================================================
from .resources.get_lena_image import get_lena_image

# =============================================================================
# Geometry
# =============================================================================
from .geometry.point_cloud import PointCloud
from .geometry.connectivity import Connectivity
from .geometry.mesh import Mesh
from .geometry.integration_points import IntegrationPoints
from .geometry.mesh_conversion import (
    quadrangle_4_to_triangle_3,
    triangle_3_to_triangle_6,
    quadrangle_4_to_quadrangle_8,
)
from .geometry.create_3D_surface_meshes import (
    create_uniform_quadrangle_4_mesh,
    create_3d_quadrangle_4_heightmap,
    create_3d_triangle_3_heightmap,
    create_3d_quadrangle_4_cylindrical,
    create_3d_triangle_3_cylindrical,
    create_3d_quadrangle_4_spherical,
    create_3d_triangle_3_spherical,
)
from .geometry.create_2D_surface_meshes import (
    create_2d_quadrangle_4_rectangle,
    create_2d_triangle_3_rectangle,
    create_2d_quadrangle_4_polar,
    create_2d_triangle_3_polar,
)
from .geometry.plotter3d import Plotter3D

# =============================================================================
# Imaging
# =============================================================================
from .imaging.image import Image
from .imaging.camera import Camera
from .imaging.view import ImageProjectionResult, View
from .imaging.image_plotter import ImagePlotter
from .imaging.mesh_projection import (
    compute_projected_bounding_box,
    compute_projected_roi,
    compute_projected_mesh_mask,
    compute_projected_elements_areas,
    compute_projected_uvmap,
    compute_visibility,
    compute_pixel_integration_points,
    display_projected_bounding_box,
    display_projected_roi,
    display_projected_mesh_mask,
    display_projected_elements_areas,
    display_visibility,
    display_pixel_integration_points,
)

# =============================================================================
# Synthetic images
# =============================================================================
from .synthetic.speckle import compute_wave_speckle
from .synthetic.noise import (
    add_gaussian_blur,
    add_poisson_noise,
    add_gaussian_noise,
    add_salt_and_pepper_noise,
    quantize_image,
)
from .synthetic.synthetic_image import (
    interpolate_texture,
    generate_synthetic_image,
)

# =============================================================================
# SDIC
# =============================================================================
from .sdic.sdic_analysis import SDICAnalysis
from .sdic._load_previous_sdic_analysis import PreviousSaveWarning, DeprecatedError
from .sdic.setups import (
    create_multicamera_analysis,
    create_bicamera_analysis,
    create_splitscreen_analysis,
    create_cordin_analysis,
)
from .sdic.pairing import Pairing
from .sdic.gl_correction import (
    GrayLevelGroups,
    compute_gray_level_groups,
    znssd_correction,
    znssd_correction_exact,
    matching_correction,
    matching_correction_exact,
)
from .sdic.sdic_unknowns import SDICUnknowns
from .sdic.sdic_term import SDICCorrelationTerm
from .sdic.temporal_regularization_term import SDICTemporalRegularizationTerm
from .sdic.sdic_parametrization import (
    compute_supported_vertices,
    SDICFixedParametrization,
)
from .sdic.problem import check_same_problem
from .sdic.residuals import (
    PairingResiduals,
    compute_residuals,
    compute_pixel_residual_maps,
    display_residual_maps,
)

# =============================================================================
# Subpackages (called as pysdic.<subpackage>.<function>)
# =============================================================================
from . import cordin      # Captures of the CORDIN high-speed cameras
from . import workflows   # Bank of procedures built on the API


__all__ = [
    "__version__",
    # Parametrizations and transformations
    "ParametrizationResult",
    "Parametrization",
    "ComposedParametrization",
    "AffineParametrization",
    "SubsetParametrization",
    "TransformationResult",
    "Transformation",
    "ComposedTransformation",
    "ReparametrizedTransformation",
    "Rays",
    "IdentityTransformation",
    "NoIntrinsic",
    "NoDistortion",
    "NoExtrinsic",
    "PinholeIntrinsic",
    "BrownDistortion",
    "ZernikeDistortion",
    "ZernikePolynomialsCache",
    "RigidExtrinsic",
    "PerspectiveExtrinsic",
    "OrthographicExtrinsic",
    "WorldPlanePerspectiveExtrinsic",
    "WorldPlaneOrthographicExtrinsic",
    "PlanarRigidExtrinsic",
    "PlanarHomographicExtrinsic",
    # Optimization of transformations
    "build_transformation_term",
    "build_chains_transformation_term",
    "optimize_transformation_parameters",
    "optimize_chains_transformation_parameters",
    "build_transformation_input_points_term",
    "build_chains_transformation_input_points_term",
    "optimize_transformation_input_points",
    "optimize_chains_transformation_input_points",
    # Computer vision
    "ProjectionResult",
    "compose_projection",
    "project_points",
    "DistortionResult",
    "distort_points",
    "undistort_points",
    "distort_image",
    "undistort_image",
    "display_distortion_field",
    "compute_rays",
    "intersect_rays",
    "compute_optical_flow",
    "display_optical_flow",
    "display_optical_flow_quiver",
    "DisflowConfig",
    # Resources
    "get_lena_image",
    # Geometry
    "PointCloud",
    "Connectivity",
    "Mesh",
    "IntegrationPoints",
    "quadrangle_4_to_triangle_3",
    "triangle_3_to_triangle_6",
    "quadrangle_4_to_quadrangle_8",
    "create_uniform_quadrangle_4_mesh",
    "create_3d_quadrangle_4_heightmap",
    "create_3d_triangle_3_heightmap",
    "create_3d_quadrangle_4_cylindrical",
    "create_3d_triangle_3_cylindrical",
    "create_3d_quadrangle_4_spherical",
    "create_3d_triangle_3_spherical",
    "create_2d_quadrangle_4_rectangle",
    "create_2d_triangle_3_rectangle",
    "create_2d_quadrangle_4_polar",
    "create_2d_triangle_3_polar",
    "Plotter3D",
    # Imaging
    "Image",
    "Camera",
    "ImageProjectionResult",
    "View",
    "ImagePlotter",
    "compute_projected_bounding_box",
    "compute_projected_roi",
    "compute_projected_mesh_mask",
    "compute_projected_elements_areas",
    "compute_projected_uvmap",
    "compute_visibility",
    "compute_pixel_integration_points",
    "display_projected_bounding_box",
    "display_projected_roi",
    "display_projected_mesh_mask",
    "display_projected_elements_areas",
    "display_visibility",
    "display_pixel_integration_points",
    # Synthetic images
    "compute_wave_speckle",
    "add_gaussian_blur",
    "add_poisson_noise",
    "add_gaussian_noise",
    "add_salt_and_pepper_noise",
    "quantize_image",
    "interpolate_texture",
    "generate_synthetic_image",
    # SDIC
    "SDICAnalysis",
    "PreviousSaveWarning",
    "DeprecatedError",
    "create_multicamera_analysis",
    "create_bicamera_analysis",
    "create_splitscreen_analysis",
    "create_cordin_analysis",
    "Pairing",
    "GrayLevelGroups",
    "compute_gray_level_groups",
    "znssd_correction",
    "znssd_correction_exact",
    "matching_correction",
    "matching_correction_exact",
    "SDICUnknowns",
    "SDICCorrelationTerm",
    "SDICTemporalRegularizationTerm",
    "compute_supported_vertices",
    "SDICFixedParametrization",
    "check_same_problem",
    "PairingResiduals",
    "compute_residuals",
    "compute_pixel_residual_maps",
    "display_residual_maps",
    # Subpackages
    "cordin",
    "workflows",
]