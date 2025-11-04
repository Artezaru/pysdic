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

import numpy
import scipy

from typing import Tuple, Union, Optional

from ..geometry import Mesh3D, IntegrationPoints, PointCloud3D
from ..imaging import View, ImageProjectionResult


def assembly_SDIC_distortion(
    mesh: Mesh3D,
    integration_points: IntegrationPoints,
    displacement_key_1: Optional[str],
    displacement_key_2: Optional[str],
    view_1: View,
    view_2: View,
    is_operative_1: bool,
    is_operative_2: bool,
    *,
    bypass_world_points_1: Optional[PointCloud3D] = None,
    bypass_world_points_2: Optional[PointCloud3D] = None,
    bypass_image_projection_result_1: Optional[ImageProjectionResult] = None,
    bypass_image_projection_result_2: Optional[ImageProjectionResult] = None,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    r"""
    Assemble the 3D residual and Jacobian for SDIC distortion optimisation between two views.

    Lets consider a 3D mesh as :class:`Mesh3D` with a displacement field defined at its vertices and stored as vertex properties.
    Lets consider a set of integration points defined over the mesh as :class:`IntegrationPoints`.

    Lets consider two views :class:`View`.

    The cost function for the SDIC distortion measurement :math:`C(\mathbf{dp})` is defined as:

    .. math::

        C(\mathbf{dp}) = \frac{1}{2} || \sqrt{W} \left(I_1 \circ P_1(\mathbf{X} + \mathbf{U_1}(\mathbf{X}), \mathbf{p_1} + \mathbf{dp}) - I_2 \circ P_2(\mathbf{X} + \mathbf{U_2}(\mathbf{X}), \mathbf{p_2} + \mathbf{dp})\right) ||^2

    where :math:`I_1` and :math:`I_2` are the gray level images of the two views, :math:`P_1` and :math:`P_2` are the projection functions of the two views, :math:`\mathbf{X}` are the 3D coordinates of the integration points at the reference state, :math:`W` the weights of each integration point and :math:`\mathbf{U_1}` and :math:`\mathbf{U_2}` are the displacement fields of the two current views.
    Note that if operative flags are set to False, the corresponding :math:`\mathbf{dp}` is considered zero in the cost function.

    We can denote :math:`\mathbf{X_1} = \mathbf{X} + \mathbf{U_1}(\mathbf{X})` and :math:`\mathbf{X_2} = \mathbf{X} + \mathbf{U_2}(\mathbf{X})` as the deformed coordinates of the integration points in the two views.

    The goal is to find the optimal incremental distortion parameters :math:`\mathbf{dp}` that minimizes the cost function.

    By linearizing the cost function around the current distortion parameters, we can derive the residual vector and the Jacobian matrix with respect to the incremental distortion parameters.

    .. math::

        C(\mathbf{dp}) \approx \frac{1}{2} || \sqrt{W} \left(I_1 \circ P_1(\mathbf{X_1}) - I_2 \circ P_2(\mathbf{X_2}) + \nabla I_1 \cdot J_{P_1} \cdot \mathbf{dp} - \nabla I_2 \cdot J_{P_2} \cdot \mathbf{dp}\right) ||^2

    where :math:`\nabla I_1` and :math:`\nabla I_2` are the image gradients at the projected points, and :math:`J_{P_1}` and :math:`J_{P_2}` are the Jacobians of the projection functions with respect to the distortion parameters.

    The cost function can be rewritten as:

    .. math::

        C(\mathbf{dp}) \approx \frac{1}{2} || J \cdot \mathbf{dp} - \mathbf{r} ||^2

    where the residual vector :math:`\mathbf{r}` and the Jacobian matrix :math:`J` are defined as:

    .. math::

        \mathbf{r} = \text{diag}(\sqrt{W}) (I_2 \circ P_2(\mathbf{X_2}) - I_1 \circ P_1(\mathbf{X_1}))

    .. math::

        J = \text{diag}(\sqrt{W}) (\nabla I_1 \cdot J_{P_1} - \nabla I_2 \cdot J_{P_2})

    .. note::

        - If `is_operative_1` is False, the contribution from view 1 in :math:`J` is ignored in the cost function.
        - If `is_operative_2` is False, the contribution from view 2 in :math:`J` is ignored in the cost function.

    .. note::

        The method returns the residual :math:`\mathbf{r}` with shape (Np,) and Jacobian :math:`J` with shape (Np, N_{params}) assembled over all integration points in the mesh. Where Np is the number of integration points and N_{params} is the number of distortion parameters.

    The user can then use the assembled residual and Jacobian to solve for the optimal incremental distortion parameters :math:`\mathbf{dp}` using standard optimization techniques.

    .. math::

        \mathbf{dp} = (J^T J)^{-1} J^T \mathbf{r}

    If no regularization is added.

    Notice that the integrated points with no valid projection in any of the views are ignored in the assembly.

    Parameters
    ----------
    mesh : Mesh3D
        The 3D mesh representing the object being imaged.

    integration_points : IntegrationPoints
        The integration points over the mesh where the residual and Jacobian are evaluated.

    displacement_key_1 : Optional[str]
        The key for the displacement field vertex property in the mesh for view 1.
        If None, zero displacement is assumed.

    displacement_key_2 : Optional[str]
        The key for the displacement field vertex property in the mesh for view 2.
        If None, zero displacement is assumed.

    view_1 : View
        The first view used for the DIC measurement.

    view_2 : View
        The second view used for the DIC measurement.

    is_operative_1 : bool
        Flag indicating if view 1 is operative in the measurement.

    is_operative_2 : bool
        Flag indicating if view 2 is operative in the measurement.

    bypass_world_points_1 : Optional[PointCloud3D], optional
        If provided, these world points are used directly for view 1 projection instead of computing them from the mesh and displacement, by default None.

    bypass_world_points_2 : Optional[PointCloud3D], optional
        If provided, these world points are used directly for view 2 projection instead of computing them from the mesh and displacement, by default None.

    bypass_image_projection_result_1 : Optional[ImageProjectionResult], optional
        If provided, this image projection result is used directly for view 1 instead of computing it, by default None.
    
    bypass_image_projection_result_2 : Optional[ImageProjectionResult], optional
        If provided, this image projection result is used directly for view 2 instead of computing it, by default None.

    Returns
    -------
    residual : numpy.ndarray
        The assembled residual vector for the SDIC distortion measurement.
        Shape (Np,)

    jacobian : numpy.ndarray
        The assembled Jacobian matrix for the SDIC distortion measurement.
    """
    # Check inputs
    if not isinstance(mesh, Mesh3D):
        raise TypeError(f"mesh must be a Mesh3D instance, got {type(mesh)}.")
    if not isinstance(integration_points, IntegrationPoints):
        raise TypeError(f"integration_points must be an IntegrationPoints instance, got {type(integration_points)}.")
    if displacement_key_1 is not None and not isinstance(displacement_key_1, str):
        raise TypeError(f"displacement_key_1 must be a string or None, got {type(displacement_key_1)}.")
    if displacement_key_2 is not None and not isinstance(displacement_key_2, str):
        raise TypeError(f"displacement_key_2 must be a string or None, got {type(displacement_key_2)}.")
    if not isinstance(view_1, View):
        raise TypeError(f"view_1 must be a View instance, got {type(view_1)}.")
    if not isinstance(view_2, View):
        raise TypeError(f"view_2 must be a View instance, got {type(view_2)}.")
    if not isinstance(is_operative_1, bool):
        raise TypeError(f"is_operative_1 must be a boolean, got {type(is_operative_1)}.")
    if not isinstance(is_operative_2, bool):
        raise TypeError(f"is_operative_2 must be a boolean, got {type(is_operative_2)}.")
    if not is_operative_1 and not is_operative_2:
        raise ValueError("At least one of the views must be operative.")
    
    if not (bypass_world_points_1 is None or isinstance(bypass_world_points_1, PointCloud3D)):
        raise TypeError(f"bypass_world_points_1 must be a PointCloud3D instance or None, got {type(bypass_world_points_1)}.")
    if not (bypass_world_points_2 is None or isinstance(bypass_world_points_2, PointCloud3D)):
        raise TypeError(f"bypass_world_points_2 must be a PointCloud3D instance or None, got {type(bypass_world_points_2)}.")

    if not (bypass_image_projection_result_1 is None or isinstance(bypass_image_projection_result_1, ImageProjectionResult)):
        raise TypeError(f"bypass_image_projection_result_1 must be an ImageProjectionResult instance or None, got {type(bypass_image_projection_result_1)}.")
    if not (bypass_image_projection_result_2 is None or isinstance(bypass_image_projection_result_2, ImageProjectionResult)):
        raise TypeError(f"bypass_image_projection_result_2 must be an ImageProjectionResult instance or None, got {type(bypass_image_projection_result_2)}.")

    # If bypass image projection results are provided, use them directly
    if bypass_image_projection_result_1 is not None:
        image_projection_result_1 = bypass_image_projection_result_1
    if bypass_image_projection_result_2 is not None:
        image_projection_result_2 = bypass_image_projection_result_2

    # Compute the image projection results if not bypassed
    if not bypass_image_projection_result_1:
        # If bypass world points are provided, use them directly for view 1 projection
        if bypass_world_points_1 is not None:
            world_points_1 = bypass_world_points_1
        else:
            vertices_coordinates_1 = mesh.vertices.points
            if displacement_key_1 is not None:
                displacement_1 = mesh.get_vertices_property(displacement_key_1)
                if displacement_1 is None:
                    raise ValueError(f"Displacement property '{displacement_key_1}' not found in mesh.")
                vertices_coordinates_1 = mesh.vertices.points + displacement_1
    
            world_points_1 = mesh.interpolate_property_at_integration_points(integration_points, property_array=vertices_coordinates_1)
        # Project world points to view 1
        image_projection_result_1 = view_1.image_project(world_points_1, dx=False, dintrinsic=False, ddistortion=is_operative_1, dextrinsic=False) # gray_levels : (Np,), jacobian_dx: (Np, 1, Nparams) or None

    if not bypass_image_projection_result_2:
        if bypass_world_points_2 is not None:
            world_points_2 = bypass_world_points_2
        else:
            vertices_coordinates_2 = mesh.vertices.points
            if displacement_key_2 is not None:
                displacement_2 = mesh.get_vertices_property(displacement_key_2)
                if displacement_2 is None:
                    raise ValueError(f"Displacement property '{displacement_key_2}' not found in mesh.")
                vertices_coordinates_2 = mesh.vertices.points + displacement_2

            world_points_2 = mesh.interpolate_property_at_integration_points(integration_points, property_array=vertices_coordinates_2)
        # Project world points to view 2
        image_projection_result_2 = view_2.image_project(world_points_2, dx=False, dintrinsic=False, ddistortion=is_operative_2, dextrinsic=False) # gray_levels : (Np,), jacobian_dx: (Np, 1, Nparams) or None

    # Build The residual
    Residual = image_projection_result_2.gray_levels - image_projection_result_1.gray_levels  # Shape (Np, channels)

    if not Residual.shape[1] == 1:
        raise ValueError(f"Only single channel images are supported for FESDIC displacement measurement, got {Residual.shape[1]} channels.")
    Residual = Residual[:, 0]  # Shape (Np,)

    # Build The Jacobian
    # First case: both views are operative
    if is_operative_1 and is_operative_2:
        if image_projection_result_1.jacobian_ddistortion is None or image_projection_result_2.jacobian_ddistortion is None:
            raise ValueError("Jacobian with respect to distortion parameters is required from both views for FESDIC displacement measurement when both views are operative.")
        jacobian_ddistortion = image_projection_result_1.jacobian_ddistortion - image_projection_result_2.jacobian_ddistortion  # Shape (Np, channels, Nparams)
    # Second case: only view 1 is operative
    elif is_operative_1:
        if image_projection_result_1.jacobian_ddistortion is None:
            raise ValueError("Jacobian with respect to distortion parameters is required from view 1 for FESDIC displacement measurement when only view 1 is operative.")
        jacobian_ddistortion = image_projection_result_1.jacobian_ddistortion  # Shape (Np, channels, Nparams)
    # Third case: only view 2 is operative
    else:  # is_operative_2
        jacobian_ddistortion = -image_projection_result_2.jacobian_ddistortion  # Shape (Np, channels, Nparams)
        if jacobian_ddistortion is None:
            raise ValueError("Jacobian with respect to distortion parameters is required from view 2 for FESDIC displacement measurement when only view 2 is operative.")
    jacobian_ddistortion = jacobian_ddistortion[:, 0, :]  # Shape (Np, Nparams)

    return Residual, jacobian_ddistortion
