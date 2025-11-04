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


def assembly_FESDIC_displacement(
    mesh: Mesh3D,
    integration_points: IntegrationPoints,
    displacement_key_1: Optional[str],
    displacement_key_2: Optional[str],
    view_1: View,
    view_2: View,
    is_operative_1: bool,
    is_operative_2: bool,
    sparse: bool = False,
    *,
    bypass_world_points_1: Optional[PointCloud3D] = None,
    bypass_world_points_2: Optional[PointCloud3D] = None,
    bypass_image_projection_result_1: Optional[ImageProjectionResult] = None,
    bypass_image_projection_result_2: Optional[ImageProjectionResult] = None,
) -> Tuple[numpy.ndarray, Union[numpy.ndarray, scipy.sparse.csr_matrix]]:
    r"""
    Assemble the 3D residual and Jacobian for FE-SDIC displacement measurement between two views.

    Lets consider a 3D mesh as :class:`Mesh3D` with a displacement field defined at its vertices and stored as vertex properties.
    Lets consider a set of integration points defined over the mesh as :class:`IntegrationPoints`.

    Lets consider two views :class:`View`.

    The cost function for the FESDIC displacement measurement :math:`C(\mathbf{dU})` is defined as:

    .. math::

        C(\mathbf{dU}) = \frac{1}{2} || \sqrt{W} \left(I_1 \circ P_1(\mathbf{X} + \mathbf{U_1}(\mathbf{X}) + \mathbf{dU}(\mathbf{X})) - I_2 \circ P_2(\mathbf{X} + \mathbf{U_2}(\mathbf{X}) + \mathbf{dU}(\mathbf{X}))\right) ||^2

    where :math:`I_1` and :math:`I_2` are the gray level images of the two views, :math:`P_1` and :math:`P_2` are the projection functions of the two views, :math:`\mathbf{X}` are the 3D coordinates of the integration points at the reference state, :math:`W` the weights of each integration point and :math:`\mathbf{U_1}` and :math:`\mathbf{U_2}` are the displacement fields of the two current views.
    Note that if operative flags are set to False, the corresponding :math:`\mathbf{dU}` is considered zero in the cost function.

    We can denote :math:`\mathbf{X_1} = \mathbf{X} + \mathbf{U_1}(\mathbf{X})` and :math:`\mathbf{X_2} = \mathbf{X} + \mathbf{U_2}(\mathbf{X})` as the deformed coordinates of the integration points in the two views.

    The goal is to find the optimal incremental displacement field :math:`\mathbf{dU}(\mathbf{X})` that minimizes the cost function.

    By linearizing the cost function around the current displacement fields, we can derive the residual vector and the Jacobian matrix with respect to the incremental displacement field.

    .. math::

        C(\mathbf{dU}) \approx \frac{1}{2} || \sqrt{W} \left(I_1 \circ P_1(\mathbf{X_1}) - I_2 \circ P_2(\mathbf{X_2}) + \nabla I_1 \cdot J_{P_1} \cdot \mathbf{dU} - \nabla I_2 \cdot J_{P_2} \cdot \mathbf{dU}\right) ||^2

    where :math:`\nabla I_1` and :math:`\nabla I_2` are the image gradients at the projected points, and :math:`J_{P_1}` and :math:`J_{P_2}` are the Jacobians of the projection functions with respect to the 3D coordinates.

    The incremental displacement field :math:`\mathbf{dU}(\mathbf{X})` is interpolated from the nodal values defined at the mesh vertices using the shape functions of the mesh elements such that:

    .. math::

        \mathbf{dU}(\mathbf{X}) = N(\mathbf{X}) \cdot \mathbf{dU_{nodes}}

    where :math:`N(\mathbf{X})` are the shape functions evaluated at the integration points, and :math:`\mathbf{dU_{nodes}}` are the incremental displacements at the mesh vertices.

    The cost function can be rewritten as:

    .. math::

        C(\mathbf{dU}) \approx \frac{1}{2} || J \cdot \mathbf{dU} - \mathbf{r} ||^2

    where the residual vector :math:`\mathbf{r}` and the Jacobian matrix :math:`J` are defined as:

    .. math::

        \mathbf{r} = \text{diag}(\sqrt{W}) \left(I_2 \circ P_2(\mathbf{X_2}) - I_1 \circ P_1(\mathbf{X_1})\right)

    .. math::

        J = \text{diag}(\sqrt{W}) \left(\nabla I_1 \cdot J_{P_1} - \nabla I_2 \cdot J_{P_2}\right) \cdot N(\mathbf{X})

    .. note::

        - If `is_operative_1` is False, the contribution from view 1 in :math:`J` is ignored in the cost function.
        - If `is_operative_2` is False, the contribution from view 2 in :math:`J` is ignored in the cost function.

    .. note::

        The method returns the residual :math:`\mathbf{r}` with shape (Np,) and Jacobian :math:`J` with shape (Np, 3*Nn) assembled over all integration points in the mesh. Where Np is the number of integration points and Nn is the number of mesh vertices.
        The nodes are ordered as [Ux_node1, Uy_node1, Uz_node1, Ux_node2, Uy_node2, Uz_node2, ..., Ux_nodeNn, Uy_nodeNn, Uz_nodeNn].

    The user can then use the assembled residual and Jacobian to solve for the optimal incremental displacement field :math:`\mathbf{dU_{nodes}}` at the mesh vertices using standard optimization techniques.

    .. math::

        \mathbf{dU_{nodes}} = (J^T J)^{-1} J^T \mathbf{r}

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

    sparse : bool, optional
        If True, the Jacobian matrix is returned as a sparse matrix, by default False.

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
        The assembled residual vector for the FEDIC displacement measurement.
        Shape (Np,)

    jacobian : Union[numpy.ndarray, scipy.sparse.csr_matrix]
        The assembled Jacobian matrix for the FEDIC displacement measurement.
        Shape (Np, 3*Nn)
        Where Np is the number of integration points and Nn is the number of mesh vertices.
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
    if not isinstance(sparse, bool):
        raise TypeError(f"sparse must be a boolean, got {type(sparse)}.")
    
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
        image_projection_result_1 = view_1.image_project(world_points_1, dx=is_operative_1, dintrinsic=False, ddistortion=False, dextrinsic=False) # gray_levels : (Np,), jacobian_dx: (Np, 1, 3) or None

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
        image_projection_result_2 = view_2.image_project(world_points_2, dx=is_operative_2, dintrinsic=False, ddistortion=False, dextrinsic=False) # gray_levels : (Np,), jacobian_dx: (Np, 1, 3) or None

    # Build The residual
    Residual = image_projection_result_2.gray_levels - image_projection_result_1.gray_levels  # Shape (Np, channels)

    if not Residual.shape[1] == 1:
        raise ValueError(f"Only single channel images are supported for FESDIC displacement measurement, got {Residual.shape[1]} channels.")
    Residual = Residual[:, 0]  # Shape (Np,)

    # Build The Jacobian
    # First case: both views are operative
    if is_operative_1 and is_operative_2:
        if image_projection_result_1.jacobian_dx is None or image_projection_result_2.jacobian_dx is None:
            raise ValueError("Jacobian with respect to 3D points is required from both views for FESDIC displacement measurement when both views are operative.")
        jacobian_dx = image_projection_result_1.jacobian_dx - image_projection_result_2.jacobian_dx  # Shape (Np, channels, 3)
    # Second case: only view 1 is operative
    elif is_operative_1:
        if image_projection_result_1.jacobian_dx is None:
            raise ValueError("Jacobian with respect to 3D points is required from view 1 for FESDIC displacement measurement when only view 1 is operative.")
        jacobian_dx = image_projection_result_1.jacobian_dx  # Shape (Np, channels, 3)
    # Third case: only view 2 is operative
    else:  # is_operative_2
        jacobian_dx = -image_projection_result_2.jacobian_dx  # Shape (Np, channels, 3)
        if jacobian_dx is None:
            raise ValueError("Jacobian with respect to 3D points is required from view 2 for FESDIC displacement measurement when only view 2 is operative.")
    jacobian_dx = jacobian_dx[:, 0, :]  # Shape (Np, 3)

    # Get the shape function matrix with shape (Np, Nn)
    shape_function, _ = mesh.shape_functions(integration_points.natural_coordinates, jacobian=False)  # Shape (Np, N_vertices_par_element)

    # Assemble the Jacobian with shape (Np, 3*Nn)
    # case of numpy array
    if not sparse:
        # basic sizes
        Np = integration_points.n_points
        Ndof = mesh.n_vertices * 3

        # Global column indices for each integration point
        # ------------------------------------------------------------------
        # `mesh.connectivity` maps an element → the *global vertex ids* of its nodes.
        # For every integration point we know which element it belongs to:
        elem_ids = integration_points.element_indices # (Np,)
        node_ids_per_point = mesh.connectivity[integration_points.element_indices , :] # (Np, Nn_per_elem)

        # Offsets for the three displacement components of a vertex:
        dof_offsets = numpy.arange(3) # (3,)

        # Global column index for (point i, node a, component c):
        #   col = 3 * vertex_id + c
        column_indices = (3 * node_ids_per_point[..., numpy.newaxis] + dof_offsets).reshape(Np, -1) # (Np, 3*Nn_per_elem)

        # Row indices (simply repeat each point index for its 3*Nn_per_elem entries)
        # ------------------------------------------------------------------
        row_indices = numpy.repeat(numpy.arange(Np), 3 * node_ids_per_point.shape[1]) # (Np*3*Nn_per_elem,)

        # Values that go into the matrix
        # ------------------------------------------------------------------
        #
        # Vectorised version:
        #   - `jacobian_dx` : (Np, 3, 3) → we need only the *first* spatial direction
        #     (the original sparse code used `jacobian_dx[:, :, None]`,
        #      i.e. all three rows, but then reshaped to (Np, 3*Nn).  
        #   - To stay faithful we keep the full 3‑vector `jacobian_dx[:, :, :]`
        #     and broadcast against the shape‑function values.
        #
        # Resulting `data` will have shape (Np*3*Nn_per_elem,).

        # Broadcast multiplication:
        #   shape_function : (Np, Nn) → (Np, Nn, 1)
        #   jacobian_dx    : (Np, 3, 3) → (Np, 1, 3, 3)
        #   We want a tensor (Np, Nn, 3) = N_a(i) * (∂x/∂ξ)_i[:,0:3] (the three components)
        #   The original code multiplies each component of the *row* of the Jacobian
        #   by the scalar shape function, then flattens.
        J_weighted = jacobian_dx[:, :, numpy.newaxis] * shape_function[:, numpy.newaxis, :] # (Np, 3, Nn)

        # Rearrange to match the ordering used by the sparse version:
        #   (Np, 3, Nn) → transpose → (Np, Nn, 3) → flatten row‑wise
        data = J_weighted.transpose(0, 2, 1).reshape(Np, -1).ravel() # (Np*3*Nn,)

        # Assemble the **dense** Jacobian matrix
        # ------------------------------------------------------------------
        Jacobian = numpy.zeros((Np, 3 * Ndof), dtype=jacobian_dx.dtype)

        # Advanced indexing: for each point i we write the whole row slice at once.
        rows = numpy.arange(Np)[:, None] # (Np,1) broadcastable
        Jacobian[rows, column_indices] = data.reshape(Np, -1)

        # Apply quadrature weights (if they exist)
        # ------------------------------------------------------------------
        if integration_points._weights is not None: # use _weights for if test to avoid computing on ones weights then use the classic weight property
            sqrt_w = numpy.sqrt(integration_points.weights) # (Np,)
            # Weight the residual vector element‑wise
            Residual *= sqrt_w
            # Weight each Jacobian row by the same sqrt(weight)
            Jacobian *= sqrt_w[:, numpy.newaxis] # broadcasting

    # case of sparse matrix
    else:
        Np = integration_points.n_points
        Ndof = mesh.n_vertices * 3
        node_ids_per_point = mesh.connectivity[integration_points.element_indices, :]  # Shape (Np, Nn_per_element) 

        # Create the column indices for each point
        dof_offsets = numpy.arange(3) # Shape (3,)
        column_indices = (3 * node_ids_per_point[..., numpy.newaxis] + dof_offsets).reshape(Np, -1)  # Shape (Np, 3*Nn_per_element)

        # Reapeat indices of lines 3*Nn_per_element times
        row_indices = numpy.repeat(numpy.arange(Np), 3 * node_ids_per_point.shape[1]) # Shape (Np * 3*Nn_per_element,)

        # Compute the values to fill the sparse matrix
        J_weighted = jacobian_dx[:, :, numpy.newaxis] * shape_function[:, numpy.newaxis, :]  # Shape (Np, 3, Nn_per_element)
        data = J_weighted.transpose(0, 2, 1).reshape(Np, -1).ravel()  # Shape (Np * 3*Nn_per_element,)

        # Assemble the sparse Jacobian matrix
        Jacobian = scipy.sparse.coo_matrix((data, (row_indices, column_indices.ravel())), shape=(Np, Ndof)).tocsr()  # Shape (Np, 3*Nn)
        
        # Add weights to residual and Jacobian
        if integration_points._weights is not None: # use _weights for if test to avoid computing on ones weights then use the classic weight property
            sqrt_weights = numpy.sqrt(integration_points.weights)  # Shape (Np,)
            Residual *= sqrt_weights # Shape (Np,)
            D_sqrt_weights = scipy.sparse.diags(sqrt_weights)  # Shape (Np, Np)
            Jacobian = D_sqrt_weights.dot(Jacobian)  # Shape (Np, 3*Nn)
        
    return Residual, Jacobian