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

from typing import Optional, Union
from numbers import Real, Integral
from numpy.typing import ArrayLike

import numpy
import scipy


def build_displacement_operator(
    jacobian_dxyz: ArrayLike,
    shape_function: ArrayLike,
    element_connectivity: ArrayLike,
    element_indices: ArrayLike,
    vertices_number: Optional[Integral] = None,
    *,
    sparse: bool = False,
    skip_m1: bool = True,
    default: Real = 0.0,
) -> Union[numpy.ndarray, scipy.sparse.csr_matrix]:
    r"""
    Build the displacement operator that maps nodal displacements to image displacements at integration points.
    
    In a space of dimension :math:`E` with a mesh constituted of :math:`N_{e}` elements and :math:`N_{v}` nodes,
    The mesh is composed of :math:`K`-dimensional elements (with :math:`K \leq E`) defined by :math:`N_{vpe}` nodes for each element.

    Lets consider :math:`M` integration points located within elements. 
    We have a jacobian array of shape :math:`(M, P, E)` that describes how displacements in the :math:`E`-dimensional space affect each property at the integration points, where :math:`P` (e.g., gray levels).
    We want to build the displacement operator that maps nodal displacements from property displacements at these integration points with shape :math:`(M \times P, N_{v} \times E)`.

    The objectif is to build the jacobian operator :math:`\mathbf{J} \in \mathbb{R}^{M \times (N_{v} \cdot E)}` such that:

    .. math::

        J dU = R

    where :math:`dU \in \mathbb{R}^{N_{v} \cdot E}` is the vector of nodal displacements and :math:`R \in \mathbb{R}^{M \times P}` is the residual vector.

    .. note::

        For (:math:`P \neq 1`), the lines are groupes by property first (i.e., all integration points for property 1, followed by all integration points for property 2, etc.).
        Thus the residual :math:`R` associated with the output must be constructed accordingly.

        .. math::

            R = \begin{bmatrix}
            R_{1}^{1} \\
            R_{2}^{1} \\
            \vdots \\
            R_{M}^{1} \\
            R_{1}^{2} \\
            R_{2}^{2} \\
            \vdots \\
            R_{M}^{2} \\
            \vdots \\
            R_{1}^{P} \\
            R_{2}^{P} \\
            \vdots \\
            R_{M}^{P}
            \end{bmatrix}

        where :math:`R_{i}^{j}` represents the residual at integration point :math:`i` for property :math:`j`.

        .. code-block:: python

            residual = numpy.random.rand(M, P)
            residual_vector = residual.flatten(order='F')  # Column-major flattening

    .. note::

        The columns of the operator correspond to the nodal displacements organized by direction first (i.e., all nodes for direction 1, followed by all nodes for direction 2, etc.).
        Thus the displacement vector :math:`dU` must be constructed accordingly.

        .. math::

            dU = \begin{bmatrix}
            dU_{1}^{x} \\
            dU_{2}^{x} \\
            \vdots \\
            dU_{N_{v}}^{x} \\
            dU_{1}^{y} \\
            dU_{2}^{y} \\
            \vdots \\
            dU_{N_{v}}^{y} \\
            \vdots \\
            dU_{1}^{E} \\
            dU_{2}^{E} \\
            \vdots \\
            dU_{N_{v}}^{E}
            \end{bmatrix}

        where :math:`dU_{i}^{j}` represents the displacement of node :math:`i` in direction :math:`j`.

        .. code-block:: python

            displacement_vector = numpy.random.rand(N_v * E) # resulting vector of size N_v * E
            displacement = displacement_vector.reshape((N_v, E), order='F')  # Column-major reshaping

    .. note::
    
        The inputs :obj:`jacobian_dxyz`, and :obj:`shape_function` will be converted to :obj:`numpy.float64`
        The inputs :obj:`element_connectivity` and :obj:`element_indices` will be converted to :obj:`numpy.int64`
        
    .. warning::
    
        No tests are performed to check the consistency of the input data (e.g., whether the element indices are valid, ...).
        Only the shapes and types of the input arrays are checked. 
        The behavior of the function is undefined if the input data is not consistent.

            
    Parameters
    ----------
    jacobian_dxyz: ArrayLike of shape :math:`(M, E)` or :math:`(M, P, E)`
        Array containing the jacobian of the image properties with respect to displacements in the :math:`E`-dimensional space at :math:`M` integration points.

    shape_function: ArrayLike of shape :math:`(M, N_{vpe})`
        Array containing the shape function values evaluated at :math:`M` points for the :math:`N_{vpe}` nodes of the element.

    element_connectivity: ArrayLike of shape :math:`(N_{e}, N_{vpe})`
        Array defining the connectivity of the elements in the mesh, where each row contains the indices of the nodes that form an element.

    element_indices: ArrayLike of shape :math:`(M,)`
        An array containing the indices of each element corresponding to the :math:`M` integration points.

    vertices_number: Optional[Integral], optional
        Total number of vertices :math:`N_{v}` in the mesh. If not provided, it will be inferred from the :obj:`element_connectivity`.

    sparse: :class:`bool`, optional
        If set to :obj:`True`, the function will return a sparse matrix representation of the displacement operator. Default is :obj:`False`.

    skip_m1: :class:`bool`, optional
        If set to :obj:`True`, any element index of -1 in :obj:`element_indices` will result in the corresponding rows in the displacement operator being set to :obj:`default`. Default is :obj:`True`.

    default: Real, optional
        The default value to use for entries corresponding to integration points with element index -1 when :obj:`skip_m1` is :obj:`True`. Default is 0.0.


    Returns
    -------
    displacement_operator: Union[:class:`numpy.ndarray`, :class:`scipy.sparse.csr_matrix`] of shape :math:`(M \times P, N_{v} \times E)`
        The displacement operator mapping nodal displacements from property displacements at integration points.
        
        
    Raises
    ------
    ValueError
        If the input arrays do not have the expected shapes or types, or if the parameters are not valid.

    """
    # Input validation
    jacobian_dxyz = numpy.asarray(jacobian_dxyz, dtype=numpy.float64)
    if not (jacobian_dxyz.ndim == 2 or jacobian_dxyz.ndim == 3):
        raise ValueError("jacobian_dxyz must be a 2D or 3D array.")
    if jacobian_dxyz.ndim == 2:
        jacobian_dxyz = jacobian_dxyz[:, numpy.newaxis, :]  # Add property dimension

    shape_function = numpy.asarray(shape_function, dtype=numpy.float64)
    if shape_function.ndim != 2:
        raise ValueError("shape_function must be a 2D array.")

    element_connectivity = numpy.asarray(element_connectivity, dtype=numpy.int64)
    if element_connectivity.ndim != 2:
        raise ValueError("element_connectivity must be a 2D array.")

    element_indices = numpy.asarray(element_indices, dtype=numpy.int64)
    if element_indices.ndim != 1:
        raise ValueError("element_indices must be a 1D array.")

    if not isinstance(skip_m1, bool):
        raise ValueError("skip_m1 must be a boolean value.")
    if not isinstance(sparse, bool):
        raise ValueError("sparse must be a boolean value.")
    if not isinstance(default, Real):
        raise ValueError("default must be a real number.")

    if (
        not jacobian_dxyz.shape[0]
        == shape_function.shape[0]
        == element_indices.shape[0]
    ):
        raise ValueError(
            "The first dimension of jacobian_dxyz, shape_function, and element_indices must be equal (number of integration points M)."
        )
    if not shape_function.shape[1] == element_connectivity.shape[1]:
        raise ValueError(
            "The second dimension of shape_function must be equal to the second dimension of element_connectivity (number of vertices per element N_vpe)."
        )

    # Extract number of vertices
    if vertices_number is None:
        vertices_number = numpy.max(element_connectivity) + 1

    if not isinstance(vertices_number, Integral) or vertices_number <= 0:
        raise ValueError("vertices_number must be a positive integer.")

    # Extract dimensions
    M = shape_function.shape[0]  # Number of integration points
    N_vpe = shape_function.shape[1]  # Number of vertices per element
    E = jacobian_dxyz.shape[2]  # Spatial dimensions
    P = jacobian_dxyz.shape[1]  # Number of properties
    N_v = vertices_number  # Total number of vertices

    # Handle skip_m1 option
    if skip_m1:
        m1_mask = element_indices == -1
        numpy.logical_not(m1_mask, out=m1_mask)
        valid_indices = element_indices[m1_mask]
        M_valid = valid_indices.shape[0]
    else:
        m1_mask = numpy.ones(M, dtype=bool)
        valid_indices = element_indices
        M_valid = M

    # Extract the vertices indices for each point
    vertex_indices = element_connectivity[valid_indices, :]  # (M_valid, N_vpe)

    # Create the columns indices for the displacement operator
    # col = node_index + direction * N_v
    column_indices = (
        vertex_indices[:, :, numpy.newaxis]
        + numpy.arange(E)[numpy.newaxis, numpy.newaxis, :] * N_v
    )  # (M_valid, N_vpe, E)
    column_indices = column_indices.reshape(
        M_valid, -1, order="C"
    )  # (M_valid, N_vpe * E)

    # Create the lines indices for the displacement operator
    # row = point_index + property_index * M
    row_indices = (
        numpy.arange(M_valid)[:, numpy.newaxis] + numpy.arange(P)[numpy.newaxis, :] * M
    )  # (M_valid, P)
    row_indices = row_indices.reshape(-1, 1, order="C")  # (M_valid * P, 1)

    # Create the data for the displacement operator
    # data = shape_function_value * jacobian_value
    shape_function_expanded = shape_function[
        m1_mask, :, numpy.newaxis
    ]  # (M_valid, N_vpe, 1)
    jacobian_expanded = jacobian_dxyz[
        m1_mask, numpy.newaxis, :, :
    ]  # (M_valid, 1, P, E)
    data = (
        shape_function_expanded[:, :, numpy.newaxis, :] * jacobian_expanded
    )  # (M_valid, N_vpe, P, E)
    data = data.reshape(M_valid * P, -1, order="C")  # (M_valid * P, N_vpe * E)

    # Build the output matrix.
    if not sparse:
        jacobian = numpy.full((M * P, N_v * E), default, dtype=numpy.float64)
        jacobian[row_indices[:, 0], column_indices[0, :]] = data[:, :]
    else:
        jacobian = scipy.sparse.csr_matrix(
            (
                data.flatten(order="C"),
                (
                    row_indices.repeat(column_indices.shape[1], axis=1).flatten(
                        order="C"
                    ),
                    column_indices.repeat(P, axis=0).flatten(order="C"),
                ),
            ),
            shape=(M * P, N_v * E),
        )

    return jacobian


def build_image_jacobian_operator(
    I1: ArrayLike,
    I2: ArrayLike,
    dI1_dX: Optional[ArrayLike] = None,
    dI2_dX: Optional[ArrayLike] = None,
    dX1_dp: Optional[ArrayLike] = None,
    dX2_dp: Optional[ArrayLike] = None,
    shape_function: Optional[ArrayLike] = None,
    element_connectivity: Optional[ArrayLike] = None,
    element_indices: Optional[ArrayLike] = None,
    vertices_number: Optional[Integral] = None,
    *,
    sparse: bool = False,
    skip_m1: bool = True,
    default: Real = 0.0,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Build the image jacobian operator that maps nodal displacements to image intensity changes at integration points.

    In a space of dimension :math:`E` with a mesh constituted of :math:`N_{e}` elements and :math:`N_{v}` nodes,
    The mesh is composed of :math:`K`-dimensional elements (with :math:`K \leq E`) defined by :math:`N_{vpe}` nodes for each element.

    Lets consider :math:`M` integration points located within elements.
    We have two images :math:`I_{1}` and :math:`I_{2}` defined at these integration points, and we want to build the jacobian operator that maps nodal displacements to changes in these images at the integration points.

    The objectif is to build the jacobian operator :math:`\mathbf{J} \in \mathbb{R}^{M \times (N_{v} \cdot E)}` such that:

    .. math::

        J dU = R

    where :math:`dU \in \mathbb{R}^{N_{v} \cdot E}` is the vector of nodal displacements and :math:`R \in \mathbb{R}^{M}` is the residual vector representing the difference between the two images at the integration points.

    .. note::

        The inputs :obj:`I1`, :obj:`I2`, :obj:`dI1_dX`, and :obj:`dI2_dX` will be converted to :obj:`numpy.float64`
        The inputs :obj:`dX1_dp`, and :obj:`dX2_dp` will be converted to :obj:`numpy.float64` if provided, otherwise they will be set to None.
        The inputs :obj:`shape_function`, :obj:`element_connectivity`, and :obj:`element_indices` will be converted to :obj:`numpy.int64` if provided, otherwise they will be set to None.
    """
    pass
