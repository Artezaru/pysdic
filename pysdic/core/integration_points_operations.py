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
from typing import Tuple, Optional, Union
from numbers import Real, Integral
from numpy.typing import ArrayLike

import numpy
import scipy

from .shape_functions import compute_shape_functions


def assemble_shape_function_matrix(
    shape_functions: ArrayLike,
    connectivity: ArrayLike,
    element_indices: ArrayLike,
    n_vertices: Optional[Integral] = None,
    *,
    sparse: bool = False,
    skip_m1: bool = True,
    default: Real = 0.0,
) -> Union[numpy.ndarray, scipy.sparse.csr_matrix]:
    r"""
    Assemble the shape function matrix :math:`\mathcal{N}` at given integration
    points within elements with shape :math:`(N_{p}, N_{v})`, from precomputed shape
    functions values with zero-filling for non-associated nodes/vertices.

    ``shape_function_matrix[i, j]`` contains the shape function value at integration
    point :math:`i` for vertex :math:`j` and a non-zero value only if vertex
    :math:`j` is part of the element containing integration point :math:`i`.

    .. note::

        - Input :obj:`shape_functions` will be converted to :obj:`numpy.float64`.
        - Inputs :obj:`connectivity` and :obj:`element_indices` will be converted to :obj:`numpy.int64`.

    .. warning::

        No tests are performed to check if the inputs array are consistent
        (e.g., if the connectivity contains invalid vertex indices, ...).
        Only shapes are validated.
        The behavior of the function is undefined in this case.

    .. important::

        When using :obj:`-1` in :obj:`element_indices` for invalid elements,
        ensure to set :obj:`skip_m1` to :obj:`True` to avoid indexing errors.


    Parameters
    ----------
    shape_functions: ArrayLike
        An array of shape :math:`(N_{p}, N_{vpe})` containing the shape function values
        evaluated at :math:`N_{p}` points for the :math:`N_{vpe}` nodes of the element.

    connectivity: ArrayLike
        An array of shape :math:`(N_{e}, N_{vpe})` defining the connectivity of the
        elements in the mesh, where each row contains the indices of the nodes that
        form an element.

    element_indices: ArrayLike
        An array of shape :math:`(N_{p},)` containing the indices of each element
        corresponding to the :math:`N_{p}` integration points.

    n_vertices: Optional[Integral], optional
        The total number of vertices :math:`N_{v}` in the mesh.
        If not provided, it will be inferred as the maximum node index in
        :obj:`connectivity` plus one.

    sparse: :class:`bool`, optional
        If set to :obj:`True`, the function will use :obj:`scipy.sparse` to create a
        sparse matrix representation of the shape function matrix. Default is
        :obj:`False`.

    skip_m1: :class:`bool`, optional
        If set to :obj:`True`, any element index of -1 in :obj:`element_indices` will
        result in the corresponding shape function values being set to :obj:`default`.
        Default is :obj:`True`.

    default: Real, optional
        The default value to assign to shape function values for integration points
        associated with an element index of -1 when :obj:`skip_m1` is :obj:`True`.
        Default is :obj:`0.0`. Only used for non-sparse matrix construction.
        For sparse matrices, zero-filling is used.


    Returns
    -------
    shape_function_matrix: Union[numpy.ndarray, scipy.sparse.csr_matrix]
        An array of shape :math:`(N_{p}, N_{v})` or sparse matrix containing the shape
        function values at each of the :math:`N_{p}` integration points for all
        :math:`N_{v}` nodes in the mesh.


    See Also
    --------
    pysdic.compute_shape_functions:
        To compute the shape functions at given natural coordinates within elements
        with shape :math:`(N_{p}, N_{vpe})`.

    pysdic.compute_shape_function_matrix:
        To compute the shape function matrix directly from natural coordinates
        and mesh connectivity.


    Examples
    --------
    Lets construct a simple 2D mesh and build the shape function matrix
    at given integration points for triangular elements.

    .. code-block:: python
        :linenos:

        import numpy
        from pysdic import assemble_shape_function_matrix
        from pysdic import compute_triangle_3_shape_functions

        vertices_coordinates = numpy.array(
            [[0.0, 0.0],
             [1.0, 0.0],
             [1.0, 1.0],
             [0.0, 1.0]]
        )

        connectivity = numpy.array(
            [[0, 1, 2],
             [0, 2, 3]]
        )

        natural_coordinates = numpy.array(
            [[0.2, 0.3],
             [0.6, 0.2]]
        )

        element_indices = numpy.array([0, 1])

        shape_functions = compute_triangle_3_shape_functions(natural_coordinates)

        shape_func_matrix = assemble_shape_function_matrix(
            shape_functions=shape_functions,
            connectivity=connectivity,
            element_indices=element_indices
        )

        print(f"shape function matrix (shape={shape_func_matrix.shape}):")
        print(shape_func_matrix)

    .. code-block:: console

        shape function matrix (shape=(2, 4)):
        [[0.5 0.2 0.3 0. ]
         [0.2 0.  0.6 0.2]]

    """
    # Validate input dimensions
    shape_functions = numpy.asarray(shape_functions, dtype=numpy.float64)
    connectivity = numpy.asarray(connectivity, dtype=numpy.int64)
    element_indices = numpy.asarray(element_indices, dtype=numpy.int64)

    if shape_functions.ndim != 2:
        raise ValueError(
            f"'shape_functions' must be a 2D array of shape (N_p, N_npe)."
            f" Got {shape_functions.ndim}D array instead."
        )
    if connectivity.ndim != 2:
        raise ValueError(
            f"'connectivity' must be a 2D array of shape (N_e, N_npe)."
            f" Got {connectivity.ndim}D array instead."
        )
    if element_indices.ndim != 1:
        raise ValueError(
            f"'element_indices' must be a 1D array of shape (N_p,)."
            f" Got {element_indices.ndim}D array instead."
        )
    if not shape_functions.shape[0] == element_indices.shape[0]:
        raise ValueError(
            f"The first dimension (N_p) of 'shape_functions' and 'element_indices'"
            f" must match. Got {shape_functions.shape[0]} and"
            f" {element_indices.shape[0]} respectively."
        )
    if not shape_functions.shape[1] == connectivity.shape[1]:
        raise ValueError(
            f"The second dimension (N_npe) of 'shape_functions' and 'connectivity' "
            f"must match. Got {shape_functions.shape[1]} and"
            f" {connectivity.shape[1]} respectively."
        )
    if not isinstance(sparse, bool):
        raise ValueError(
            f"'sparse' must be a boolean value. Got {type(sparse)} instead."
        )
    if not isinstance(skip_m1, bool):
        raise ValueError(
            f"'skip_m1' must be a boolean value. Got {type(skip_m1)} instead."
        )
    if not isinstance(default, Real):
        raise ValueError(
            f"'default' must be a real number. Got {type(default)} instead."
        )

    # Extract number of vertices
    if n_vertices is None:
        n_vertices = numpy.max(connectivity) + 1

    if not isinstance(n_vertices, Integral) or n_vertices <= 0:
        raise ValueError(
            f"n_vertices must be a positive integer. Got {n_vertices} instead."
        )

    # Extract dimensions
    N_p = shape_functions.shape[0]
    N_npe = shape_functions.shape[1]
    N_v = n_vertices

    # Handle skip_m1 option
    if skip_m1:
        m1_mask = element_indices == -1
        numpy.logical_not(m1_mask, out=m1_mask)
        valid_indices = element_indices[m1_mask]
        shape_functions = shape_functions[m1_mask, :]
    else:
        m1_mask = numpy.ones(N_p, dtype=bool)
        valid_indices = element_indices

    # Extract the vertices indices for each point
    vertex_indices = connectivity[valid_indices, :]  # (N_valid, N_npe)

    # Build the numpy arrays
    if not sparse:
        shape_functions_matrix = numpy.zeros((N_p, N_v), dtype=numpy.float64)
        sfm_mask = numpy.arange(N_p)[m1_mask, None]
        shape_functions_matrix[sfm_mask, vertex_indices] = shape_functions

        # Handle skip_m1 option
        if skip_m1:
            shape_functions_matrix[~m1_mask, :] = default

    # Build the scipy sparse matrices
    else:
        row_idx = numpy.repeat(numpy.arange(N_p), N_npe)
        col_idx = vertex_indices.ravel()
        data = shape_functions.ravel()
        shape_functions_matrix = scipy.sparse.csr_matrix(
            (data, (row_idx, col_idx)), shape=(N_p, N_v)
        )

    return shape_functions_matrix


def compute_shape_function_matrix(
    connectivity: ArrayLike,
    element_type: str,
    natural_coordinates: ArrayLike,
    element_indices: ArrayLike,
    n_vertices: Optional[Integral] = None,
    *,
    sparse: bool = False,
    skip_m1: bool = True,
    default: Real = 0.0,
) -> Union[numpy.ndarray, scipy.sparse.csr_matrix]:
    r"""
    Compute and assemble the shape function matrix :math:`\mathcal{N}` at given
    integration points within specified elements with shape :math:`(N_{p}, N_{v})`.

    This function combines the computation of shape functions at given natural
    coordinates with the assembly of the shape function matrix for the entire mesh.

    ``shape_function_matrix[i, j]`` contains the shape function value at integration
    point :math:`i` for vertex :math:`j` and a non-zero value only if vertex
    :math:`j` is part of the element containing integration point :math:`i`.

    .. note::

        - Input :obj:`natural_coordinates` will be converted to :obj:`numpy.float64`.
        - Inputs :obj:`connectivity` and :obj:`element_indices` will be converted to :obj:`numpy.int64`.

    .. warning::

        No tests are performed to check if the inputs array are consistent
        (e.g., if the connectivity contains invalid vertex indices, ...).
        Only shapes are validated.
        The behavior of the function is undefined in this case.

    .. important::

        When using :obj:`-1` in :obj:`element_indices` for invalid elements,
        ensure to set :obj:`skip_m1` to :obj:`True` to avoid indexing errors.


    Parameters
    ----------
    connectivity: ArrayLike
        An array of shape :math:`(N_{e}, N_{vpe})` defining the connectivity of
        the elements in the mesh, where each row contains the indices of the nodes
        that form an element.

    element_type: :class:`str`
        A string specifying the type of element (e.g., 'segment_2', 'triangle_3', etc.)
        to determine which shape function to use.

    natural_coordinates: ArrayLike
        An array of shape :math:`(N_{p}, K)` containing the natural coordinates of the
        integration points within elements where :math:`K` is the topological dimension
        of the element (e.g., 1 for segments, 2 for triangles/quadrangles, etc.).

    element_indices: ArrayLike
        An array of shape :math:`(N_{p},)` containing the indices of each element
        corresponding to the :math:`N_{p}` integration points.

    n_vertices: Optional[Integral], optional
        The total number of vertices :math:`N_{v}` in the mesh.
        If not provided, it will be inferred as the maximum node index in
        :obj:`connectivity` plus one.

    sparse: :class:`bool`, optional
        If set to :obj:`True`, the function will use :obj:`scipy.sparse` to create a
        sparse matrix representation of the shape function matrix.
        Default is :obj:`False`.

    skip_m1: :class:`bool`, optional
        If set to :obj:`True`, any element index of -1 in :obj:`element_indices` will
        result in the corresponding shape function values being set to :obj:`default`.
        Default is :obj:`True`.

    default: Real, optional
        The default value to assign to shape function values for integration points
        associated with an element index of -1 when :obj:`skip_m1` is :obj:`True`.
        Default is :obj:`0.0`. Only used for non-sparse matrix construction.
        For sparse matrices, zero-filling is used.


    Returns
    -------
    shape_function_matrix: Union[numpy.ndarray, scipy.sparse.csr_matrix]
        An array of shape :math:`(N_{p}, N_{v})` or sparse matrix containing the shape
        function values at each of the :math:`N_{p}` integration points for all
        :math:`N_{v}` nodes in the mesh.


    Notes
    -----
    This function first computes the shape function values at the provided natural
    coordinates using the appropriate shape function based on the specified
    :obj:`element_type`. It then assembles these values into a global shape function
    matrix for the entire mesh.


    See Also
    --------
    pysdic.compute_shape_functions:
        To compute the shape functions at given natural coordinates within elements
        with shape :math:`(N_{p}, N_{vpe})`.

    pysdic.assemble_shape_function_matrix:
        To assemble the shape function matrix from precomputed shape function values.


    Examples
    --------
    Lets construct a simple 2D mesh and build the shape function matrix at given
    integration points for triangular elements.

    .. code-block:: python
        :linenos:

        import numpy
        from pysdic import compute_shape_function_matrix

        vertices_coordinates = numpy.array(
            [[0.0, 0.0],
             [1.0, 0.0],
             [1.0, 1.0],
             [0.0, 1.0]]
        )

        connectivity = numpy.array(
            [[0, 1, 2],
             [0, 2, 3]]
        )

        natural_coordinates = numpy.array(
            [[0.2, 0.3],
             [0.6, 0.2]]
        )

        element_indices = numpy.array([0, 1])

        shape_func_matrix = compute_shape_function_matrix(
            connectivity=connectivity,
            element_type='triangle_3',
            natural_coordinates=natural_coordinates,
            element_indices=element_indices
        )

        print(f"shape function matrix (shape={shape_func_matrix.shape}):")
        print(shape_func_matrix)

    .. code-block:: console

        shape function matrix (shape=(2, 4)):
        [[0.5 0.2 0.3 0. ]
         [0.2 0.  0.6 0.2]]

    """
    # Validate input dimensions
    connectivity = numpy.asarray(connectivity, dtype=numpy.int64)
    element_indices = numpy.asarray(element_indices, dtype=numpy.int64)
    natural_coordinates = numpy.asarray(natural_coordinates, dtype=numpy.float64)

    if connectivity.ndim != 2:
        raise ValueError(
            f"'connectivity' must be a 2D array of shape (N_e, N_npe)."
            f" Got {connectivity.ndim}D array instead."
        )
    if element_indices.ndim != 1:
        raise ValueError(
            f"'element_indices' must be a 1D array of shape (N_p,)."
            f" Got {element_indices.ndim}D array instead."
        )
    if natural_coordinates.ndim != 2:
        raise ValueError(
            f"'natural_coordinates' must be a 2D array of shape (N_p, K)."
            f" Got {natural_coordinates.ndim}D array instead."
        )
    if not natural_coordinates.shape[0] == element_indices.shape[0]:
        raise ValueError(
            f"The first dimension (N_p) of 'natural_coordinates' and 'element_indices'"
            f" must match. Got {natural_coordinates.shape[0]} and"
            f" {element_indices.shape[0]} respectively."
        )
    if not isinstance(element_type, str):
        raise ValueError(
            f"'element_type' must be a string. Got {type(element_type)} instead."
        )
    if not isinstance(sparse, bool):
        raise ValueError(
            f"'sparse' must be a boolean value. Got {type(sparse)} instead."
        )
    if not isinstance(skip_m1, bool):
        raise ValueError(
            f"'skip_m1' must be a boolean value. Got {type(skip_m1)} instead."
        )
    if not isinstance(default, Real):
        raise ValueError(
            f"'default' must be a real number. Got {type(default)} instead."
        )

    # Compute shape functions at given natural coordinates
    shape_functions_values = compute_shape_functions(
        natural_coordinates=natural_coordinates,
        element_type=element_type,
        default=default,
    )

    # Assemble the shape function matrix
    shape_function_matrix = assemble_shape_function_matrix(
        shape_functions_values,
        connectivity,
        element_indices,
        n_vertices=n_vertices,
        sparse=sparse,
        skip_m1=skip_m1,
        default=default,
    )
    return shape_function_matrix


def remap_vertices_coordinates(
    vertices_coordinates: ArrayLike,
    connectivity: ArrayLike,
    element_indices: ArrayLike,
    *,
    skip_m1: bool = True,
    default: Real = numpy.nan,
) -> numpy.ndarray:
    r"""
    Remap the global coordinates of the vertices to given integration points within
    elements based on the element connectivity, and the remapped coordinates
    will have shape :math:`(N_{p}, N_{vpe}, E)`.

    ``remapped_coordinates[i]`` contains the global coordinates of the vertices for
    the element containing integration point :math:`i`.

    .. note::

        - Input :obj:`vertices_coordinates` will be converted to :obj:`numpy.float64`.
        - Inputs :obj:`connectivity` and :obj:`element_indices` will be converted to :obj:`numpy.int64`.

    .. warning::

        No tests are performed to check if the inputs array are consistent
        (e.g., if the connectivity contains invalid vertex indices, ...).
        Only shapes are validated.
        The behavior of the function is undefined in this case.

    .. important::

        When using :obj:`-1` in :obj:`element_indices` for invalid elements,
        ensure to set :obj:`skip_m1` to :obj:`True` to avoid indexing errors.


    Parameters
    ----------
    vertices_coordinates: ArrayLike
        An array of shape :math:`(N_{v}, E)` containing the global coordinates of the
        vertices in the mesh.

    connectivity: ArrayLike
        An array of shape :math:`(N_{e}, N_{vpe})` defining the connectivity of the
        elements in the mesh, where each row contains the indices of the nodes that
        form an element.

    element_indices: ArrayLike
        An array of shape :math:`(N_{p},)` containing the indices of each element
        corresponding to the :math:`N_{p}` integration points.

    skip_m1: :class:`bool`, optional
        If set to :obj:`True`, any element index of -1 in :obj:`element_indices` will
        result in the corresponding remapped coordinates being set to :obj:`default`.
        Default is :obj:`True`.

    default: Real, optional
        The default value to assign to remapped coordinates for integration points
        associated with an element index of -1 when :obj:`skip_m1` is :obj:`True`.
        Default is :obj:`numpy.nan`.


    Returns
    -------
    remapped_coordinates: :class:`numpy.ndarray`
        An array of shape :math:`(N_{p}, N_{vpe}, E)` containing the remapped global
        coordinates of the vertices for each integration point within the elements.


    Notes
    -----
    In a space of dimension :math:`E` with a mesh constituted of :math:`N_{e}` elements
    and :math:`N_{v}` nodes. The mesh is composed of :math:`K`-dimensional elements
    (with :math:`K \leq E`) defined by :math:`N_{vpe}` nodes for each element.

    For a given set of :math:`N_{p}` integration points located within elements, the
    remapped coordinates array has shape :math:`(N_{p}, N_{vpe}, E)` where each entry
    corresponds to the global coordinates of the nodes associated with the element
    containing the integration point.


    See Also
    --------
    pysdic.assemble_jacobian_matrix:
        To compute the Jacobian matrices for the transformation between natural and
        global coordinates.


    Examples
    --------
    Lets construct a simple 2D mesh and remap the vertex coordinates to given
    integration points for triangular elements.

    .. code-block:: python
        :linenos:

        import numpy
        from pysdic import remap_vertices_coordinates

        vertices_coordinates = numpy.array(
            [[0.0, 0.0],
             [1.0, 0.0],
             [1.0, 1.0],
             [0.0, 1.0]]
        )

        connectivity = numpy.array(
            [[0, 1, 2],
             [0, 2, 3]]
        )

        natural_coordinates = numpy.array(
            [[0.2, 0.3],
             [0.6, 0.2]]
        )

        element_indices = numpy.array([0, 1, 0])

        remapped_coords = remap_vertices_coordinates(
            vertices_coordinates=vertices_coordinates,
            connectivity=connectivity,
            element_indices=element_indices
        )

        print(f"remapped coordinates (shape={remapped_coords.shape}):")
        print(remapped_coords)

    .. code-block:: console

        remapped coordinates (shape=(3, 3, 2)):
        [[[0. 0.]
          [1. 0.]
          [1. 1.]]

         [[0. 0.]
          [1. 1.]
          [0. 1.]]]

         [[0. 0.]
          [1. 0.]
          [1. 1.]]]

    """
    # Validate input dimensions
    vertices_coordinates = numpy.asarray(vertices_coordinates, dtype=numpy.float64)
    connectivity = numpy.asarray(connectivity, dtype=numpy.int64)
    element_indices = numpy.asarray(element_indices, dtype=numpy.int64)

    if vertices_coordinates.ndim != 2:
        raise ValueError(
            f"'vertices_coordinates' must be a 2D array of shape (N_v, E)."
            f" Got {vertices_coordinates.ndim}D array instead."
        )
    if connectivity.ndim != 2:
        raise ValueError(
            f"'connectivity' must be a 2D array of shape (N_e, N_npe)."
            f" Got {connectivity.ndim}D array instead."
        )
    if element_indices.ndim != 1:
        raise ValueError(
            f"'element_indices' must be a 1D array of shape (N_p,)."
            f" Got {element_indices.ndim}D array instead."
        )
    if not isinstance(skip_m1, bool):
        raise ValueError(
            f"'skip_m1' must be a boolean value. Got {type(skip_m1)} instead."
        )
    if not isinstance(default, Real):
        raise ValueError(
            f"'default' must be a real number. Got {type(default)} instead."
        )

    # remap coordinates initialization
    N_p = element_indices.shape[0]
    N_npe = connectivity.shape[1]
    E = vertices_coordinates.shape[1]

    # Extract -1 mask if skip_m1 is True
    if skip_m1:
        m1_mask = element_indices == -1
        numpy.logical_not(m1_mask, out=m1_mask)
        valid_indices = element_indices[m1_mask]

    else:
        valid_indices = element_indices

    # Remap coordinates
    N_valid = valid_indices.shape[0]
    connectivity_per_point = connectivity[valid_indices, :]  # Shape: (N_valid, N_npe)
    remapped_coordinates_valid = vertices_coordinates[
        connectivity_per_point, :
    ]  # Shape: (N_valid, N_npe, E)

    # Handle -1 entries if skip_m1 is True
    if skip_m1:
        remapped_coordinates = numpy.full(
            (N_p, N_npe, E), default, dtype=remapped_coordinates_valid.dtype
        )
        remapped_coordinates[m1_mask, :, :] = remapped_coordinates_valid
    else:
        remapped_coordinates = remapped_coordinates_valid

    return remapped_coordinates


def assemble_jacobian_matrix(
    shape_function_derivatives: ArrayLike,
    remapped_coordinates: ArrayLike,
) -> numpy.ndarray:
    r"""
    Assemble the Jacobian (derivative) matrix :math:`J` of the transformation from 
    natural coordinates :math:`(\xi, \eta, \zeta, ...)` to global coordinates 
    :math:`(x, y, z, ...)` with shape :math:`(N_{p}, E, K)` from precomputed shape 
    function derivatives and remapped vertex coordinates.
    
    ``jacobians[i]`` contains the Jacobian matrix for the integration point :math:`i`.

    .. note::

        - Inputs :obj:`shape_function_derivatives` and :obj:`remapped_coordinates` will be converted to :obj:`numpy.float64`.
        
    .. warning::

        No tests are performed to check if the inputs array are consistent
        (e.g., if the connectivity contains invalid vertex indices, ...).
        The behavior of the function is undefined in this case.
        Only shapes are validated.
        
        
    Parameters
    ----------
    shape_function_derivatives: ArrayLike
        An array of shape :math:`(N_{p}, N_{vpe}, K)` containing the derivatives of 
        the shape functions with respect to the local coordinates,
        evaluated at :math:`N_{p}` points for the :math:`N_{vpe}` nodes of the element.

    remapped_coordinates: ArrayLike
        An array of shape :math:`(N_{p}, N_{vpe}, E)` containing the remapped global 
        coordinates of the vertices for each integration point within the elements


    Returns
    -------
    jacobians: :class:`numpy.ndarray`
        An array of shape :math:`(N_{p}, E, K)` containing the Jacobian matrices for 
        each of the :math:`N_{p}` points. Each Jacobian matrix has dimensions 
        :math:`(E \times K)`.
        
    
    Notes
    -----
    In a space of dimension :math:`E` with a mesh constituted of :math:`N_{e}` elements
    and :math:`N_{v}` nodes, the mesh is composed of :math:`K`-dimensional elements 
    (with :math:`K \leq E`) defined by :math:`N_{vpe}` nodes for each element.
     
    The Jacobian matrix has dimensions :math:`(E \times K)` and constructed as follows:

    .. math::

        J = J_{X/\Xi} = \frac{dX}{d\Xi} = \begin{bmatrix}
        \frac{\partial x}{\partial \xi} & \frac{\partial x}{\partial \eta} & \frac{\partial x}{\partial \zeta} & \ldots \\
        \frac{\partial y}{\partial \xi} & \frac{\partial y}{\partial \eta} & \frac{\partial y}{\partial \zeta} & \ldots \\
        \frac{\partial z}{\partial \xi} & \frac{\partial z}{\partial \eta} & \frac{\partial z}{\partial \zeta} & \ldots \\
        \vdots & \vdots & \vdots & \ddots
        \end{bmatrix}

    where each entry is computed as:

    .. math::

        \frac{\partial x_i}{\partial \xi_j} = \sum_{a=1}^{N_{vpe}} \frac{\partial N_a}{\partial \xi_j} x_{i,a}

    Here, :math:`N_a` are the shape functions associated with each node, and 
    :math:`x_{i,a}` are the global coordinates of the nodes.
    
    .. math::
    
        \nabla_{\Xi} N = J^{T} \nabla_{X} N = J_{X/\Xi}^{T} \nabla_{X} N
    
    To obtain the jacobian matrix for the transformation from global coordinates to 
    natural coordinates noted :math:`J_{\Xi/X}` such as:
    
    .. math::

        \nabla_{X} N = J_{\Xi/X}^{T} \nabla_{\Xi} N
        
    we get :
    
    .. math::
    
        \nabla_{X} N = (J^{T})^{-1} \nabla_{\Xi} N = (J^{-1})^{T} \nabla_{\Xi} N
        
    So the jacobian matrix for the transformation from global coordinates to natural 
    coordinates is the inverse of the jacobian matrix for the transformation from 
    natural coordinates to global coordinates.
    
    .. math::
    
        J_{\Xi/X} = J^{-1} = J_{X/\Xi}^{-1}
        
    For :math:`K \le E`, the Jacobian matrix is not square and thus not invertible.
    In this case, the pseudo-inverse can be used to compute the transformation from 
    global coordinates to natural coordinates. The Moore-Penrose pseudo-inverse of the 
    Jacobian matrix :math:`J \in \mathbb{R}^{E \times K}` is given by 
    :math:`J^{+} = (J^{T} J)^{-1} J^{T}` because :math:`J` has full column rank 
    (i.e., the columns of :math:`J` are linearly independent).
    

    See Also
    --------
    pysdic.compute_shape_functions:
        To compute the shape functions and their derivatives at given natural 
        coordinates within elements.
    
    pysdic.remap_vertices_coordinates:
        To remap the vertices coordinates for each element at given integration 
        points.
        
    pysdic.compute_jacobian_matrix:
        To compute the Jacobian matrices directly from natural coordinates and mesh 
        connectivity.


    Examples
    --------
    Lets construct a simple 2D mesh and compute the Jacobian at given integration points 
    for triangular elements.

    .. code-block:: python
        :linenos:

        import numpy
        from pysdic import (
            remap_vertices_coordinates, 
            assemble_jacobian_matrix,
            compute_triangle_3_shape_functions
        )

        vertices_coordinates = numpy.array(
            [[0.0, 0.0],
             [1.0, 0.0], 
             [1.0, 1.0],
             [0.0, 1.0]]
        )

        connectivity = numpy.array(
            [[0, 1, 2],
             [0, 2, 3]]
        )

        natural_coordinates = numpy.array(
            [[0.2, 0.3],
             [0.6, 0.2]]
        )
        
        element_indices = numpy.array([0, 1])

        remapped_coords = remap_vertices_coordinates(
            vertices_coordinates=vertices_coordinates,
            connectivity=connectivity,
            element_indices=element_indices
        )

        _, shape_func_derivs = compute_triangle_3_shape_functions(
            natural_coordinates=natural_coordinates,
            return_derivatives=True
        )
        
        jacobians = assemble_jacobian_matrix(
            shape_function_derivatives=shape_func_derivs, 
            remapped_coordinates=remapped_coords
        )
        
        print(f"jacobians (shape={jacobians.shape}):")
        print(jacobians)
        
    .. code-block:: console

        jacobians (shape=(2, 2, 2)):
        [[[1. 0.]
          [1. 1.]]]

         [[1. 1.]
          [0. 1.]]]

    """
    # Validate input dimensions
    shape_function_derivatives = numpy.asarray(
        shape_function_derivatives, dtype=numpy.float64
    )
    remapped_coordinates = numpy.asarray(remapped_coordinates, dtype=numpy.float64)

    if shape_function_derivatives.ndim != 3:
        raise ValueError(
            f"'shape_function_derivatives' must be a 3D array of shape (N_p, N_npe, K)."
            f" Got {shape_function_derivatives.ndim}D array instead."
        )
    if remapped_coordinates.ndim != 3:
        raise ValueError(
            f"'remapped_coordinates' must be a 3D array of shape (N_p, N_npe, E)."
            f" Got {remapped_coordinates.ndim}D array instead."
        )
    if not shape_function_derivatives.shape[0] == remapped_coordinates.shape[0]:
        raise ValueError(
            f"The first dimension (N_p) of 'shape_function_derivatives' and"
            f" 'remapped_coordinates' must match."
            f" Got {shape_function_derivatives.shape[0]} and"
            f" {remapped_coordinates.shape[0]} respectively."
        )
    if not shape_function_derivatives.shape[1] == remapped_coordinates.shape[1]:
        raise ValueError(
            f"The second dimension (N_npe) of 'shape_function_derivatives' and"
            f" 'remapped_coordinates' must match."
            f" Got {shape_function_derivatives.shape[1]} and"
            f" {remapped_coordinates.shape[1]} respectively."
        )

    # Extract dimensions
    N_p = shape_function_derivatives.shape[0]
    N_npe = shape_function_derivatives.shape[1]
    K = shape_function_derivatives.shape[2]
    E = remapped_coordinates.shape[2]

    # Compute Jacobians
    jacobians = numpy.einsum(
        "mjk,mja->mak", shape_function_derivatives, remapped_coordinates
    )
    return jacobians


def compute_jacobian_matrix(
    vertices_coordinates: ArrayLike,
    connectivity: ArrayLike,
    element_type: str,
    natural_coordinates: ArrayLike,
    element_indices: ArrayLike,
    *,
    skip_m1: bool = True,
    default: Real = numpy.nan,
) -> numpy.ndarray:
    r"""
    Compute and assemble the Jacobian (derivative) matrix :math:`J` of the 
    transformation from natural coordinates :math:`(\xi, \eta, \zeta, ...)` to
    global coordinates :math:`(x, y, z, ...)` with shape :math:`(N_{p}, E, K)` at 
    given integration points within specified elements.
    
    ``jacobians[i]`` contains the Jacobian matrix for the integration point :math:`i`.
    
    This function combines the remapping of vertex coordinates, computation of shape 
    function derivatives, and assembly of the Jacobian matrices for the entire mesh.

    .. note::

        - Inputs :obj:`vertices_coordinates` and :obj:`natural_coordinates` will be converted to :obj:`numpy.float64`.
        - Inputs :obj:`connectivity` and :obj:`element_indices` will be converted to :obj:`numpy.int64`.

    .. warning::

        No tests are performed to check if the inputs array are consistent
        (e.g., if the connectivity contains invalid vertex indices, ...).
        Only shapes are validated.
        The behavior of the function is undefined in this case.

    .. important::

        When using :obj:`-1` in :obj:`element_indices` for invalid elements,
        ensure to set :obj:`skip_m1` to :obj:`True` to avoid indexing errors.

        
    Parameters
    ----------
    vertices_coordinates: ArrayLike
        An array of shape :math:`(N_{v}, E)` containing the global coordinates of the 
        vertices in the mesh.
        
    connectivity: ArrayLike
        An array of shape :math:`(N_{e}, N_{vpe})` defining the connectivity of the 
        elements in the mesh, where each row contains the indices of the nodes that 
        form an element.

    element_type: :class:`str`
        A string specifying the type of element (e.g., 'segment_2', 'triangle_3', etc.) 
        to determine which shape function to use.

    natural_coordinates: ArrayLike
        An array of shape :math:`(N_{p}, K)` containing the natural coordinates of the 
        integration points within elements where :math:`K` is the topological dimension 
        of the element (e.g., 1 for segments, 2 for triangles/quadrangles, etc.).

    element_indices: ArrayLike
        An array of shape :math:`(N_{p},)` containing the indices of each element 
        corresponding to the :math:`N_{p}` integration points.

    skip_m1: :class:`bool`, optional
        If set to :obj:`True`, any element index of -1 in :obj:`element_indices` will 
        result in the corresponding Jacobian being set to :obj:`default`.
        Default is :obj:`True`.

    default: Real, optional
        The default value to assign to Jacobians for integration points associated 
        with an element index of -1 when :obj:`skip_m1` is :obj:`True`.
        The out-of-range natural coordinates will also result in Jacobians being set 
        to this value. Default is :obj:`numpy.nan`.
        

    Returns
    -------
    jacobians: :class:`numpy.ndarray`
        An array of shape :math:`(N_{p}, E, K)` containing the Jacobian matrices for 
        each of the :math:`N_{p}` points. Each Jacobian matrix has dimensions 
        :math:`(E \times K)`.
        
    
    Notes
    -----
    In a space of dimension :math:`E` with a mesh constituted of :math:`N_{e}` elements 
    and :math:`N_{v}` nodes, the mesh is composed of :math:`K`-dimensional elements 
    (with :math:`K \leq E`) defined by :math:`N_{vpe}` nodes for each element.
     
    The Jacobian matrix has dimensions :math:`(E \times K)` and constructed as follows:

    .. math::

        J = J_{X/\Xi} = \frac{dX}{d\Xi} = \begin{bmatrix}
        \frac{\partial x}{\partial \xi} & \frac{\partial x}{\partial \eta} & \frac{\partial x}{\partial \zeta} & \ldots \\
        \frac{\partial y}{\partial \xi} & \frac{\partial y}{\partial \eta} & \frac{\partial y}{\partial \zeta} & \ldots \\
        \frac{\partial z}{\partial \xi} & \frac{\partial z}{\partial \eta} & \frac{\partial z}{\partial \zeta} & \ldots \\
        \vdots & \vdots & \vdots & \ddots
        \end{bmatrix}

    where each entry is computed as:

    .. math::

        \frac{\partial x_i}{\partial \xi_j} = \sum_{a=1}^{N_{vpe}} \frac{\partial N_a}{\partial \xi_j} x_{i,a}

    Here, :math:`N_a` are the shape functions associated with each node, and 
    :math:`x_{i,a}` are the global coordinates of the nodes.
    
    See the notes in :func:`pysdic.assemble_jacobian_matrix` for more details on the 
    transformation between natural and global coordinates and the use of pseudo-inverse 
    for non-square Jacobian matrices.
    
    
    See Also
    --------
    pysdic.compute_shape_functions:
        To compute the shape functions and their derivatives at given natural 
        coordinates within elements.
        
    pysdic.remap_vertices_coordinates:
        To remap the vertices coordinates for each element at given integration points.
        
    pysdic.assemble_jacobian_matrix:
        To assemble the Jacobian matrices from shape function derivatives and remapped 
        coordinates.
        
        
    Examples
    --------
    Lets construct a simple 2D mesh and compute the Jacobian at given integration points
    for triangular elements.
    
    .. code-block:: python
        :linenos:
    
        import numpy
        from pysdic import compute_jacobian_matrix

        vertices_coordinates = numpy.array(
            [[0.0, 0.0],
             [1.0, 0.0], 
             [1.0, 1.0],
             [0.0, 1.0]]
        )

        connectivity = numpy.array(
            [[0, 1, 2],
             [0, 2, 3]]
        )

        natural_coordinates = numpy.array(
            [[0.2, 0.3],
             [0.6, 0.2]]
        )
        
        element_indices = numpy.array([0, 1])

        jacobians = compute_jacobian_matrix(
            vertices_coordinates=vertices_coordinates,
            connectivity=connectivity,
            element_type='triangle_3',
            natural_coordinates=natural_coordinates,
            element_indices=element_indices,
        )

        print(f"jacobians (shape={jacobians.shape}):")
        print(jacobians)
        
    .. code-block:: console

        jacobians (shape=(2, 2, 2)):
        [[[1. 1.]
          [0. 1.]]

         [[1. 0.]
          [1. 1.]]]

    """
    # Validate input dimensions
    vertices_coordinates = numpy.asarray(vertices_coordinates, dtype=numpy.float64)
    connectivity = numpy.asarray(connectivity, dtype=numpy.int64)
    element_indices = numpy.asarray(element_indices, dtype=numpy.int64)
    natural_coordinates = numpy.asarray(natural_coordinates, dtype=numpy.float64)

    if vertices_coordinates.ndim != 2:
        raise ValueError(
            f"'vertices_coordinates' must be a 2D array of shape (N_v, E)."
            f" Got {vertices_coordinates.ndim}D array instead."
        )
    if connectivity.ndim != 2:
        raise ValueError(
            f"'connectivity' must be a 2D array of shape (N_e, N_npe)."
            f" Got {connectivity.ndim}D array instead."
        )
    if element_indices.ndim != 1:
        raise ValueError(
            f"'element_indices' must be a 1D array of shape (N_p,)."
            f" Got {element_indices.ndim}D array instead."
        )
    if natural_coordinates.ndim != 2:
        raise ValueError(
            f"'natural_coordinates' must be a 2D array of shape (N_p, K)."
            f" Got {natural_coordinates.ndim}D array instead."
        )
    if not natural_coordinates.shape[0] == element_indices.shape[0]:
        raise ValueError(
            f"The first dimension (N_p) of 'natural_coordinates' and 'element_indices'"
            f" must match. Got {natural_coordinates.shape[0]} and"
            f" {element_indices.shape[0]} respectively."
        )
    if not isinstance(element_type, str):
        raise ValueError(
            f"'element_type' must be a string. Got {type(element_type)} instead."
        )
    if not isinstance(skip_m1, bool):
        raise ValueError(
            f"'skip_m1' must be a boolean value. Got {type(skip_m1)} instead."
        )
    if not isinstance(default, Real):
        raise ValueError(
            f"'default' must be a real number. Got {type(default)} instead."
        )

    # Remap vertices coordinates
    remapped_coords = remap_vertices_coordinates(
        vertices_coordinates=vertices_coordinates,
        connectivity=connectivity,
        element_indices=element_indices,
        skip_m1=skip_m1,
        default=default,
    )

    # Compute shape function derivatives at given natural coordinates
    _, shape_func_derivs = compute_shape_functions(
        natural_coordinates=natural_coordinates,
        element_type=element_type,
        return_derivatives=True,
        default=default,
    )

    # Assemble Jacobians
    jacobians = assemble_jacobian_matrix(
        shape_function_derivatives=shape_func_derivs,
        remapped_coordinates=remapped_coords,
    )
    return jacobians


def assemble_property_derivative(
    property_array: ArrayLike,
    connectivity: ArrayLike,
    element_indices: ArrayLike,
    shape_function_derivatives: ArrayLike,
    jacobian_matrix: ArrayLike,
    *,
    skip_m1: bool = True,
    default: Real = numpy.nan,
) -> numpy.ndarray:
    r"""
    Assemble the derivative of a property defined at the nodes of a mesh to given
    integration points within elements with respect to global coordinates
    :math:`(x,y,z,...)` from precomputed shape function derivatives and Jacobian
    matrices.

    ``derivated_properties[i]`` contains the derivated property values at the
    integration point :math:`i`.

    .. note::

        - Inputs :obj:`property_array`, :obj:`shape_function_derivatives` and :obj:`jacobian_matrix` will be converted to :obj:`numpy.float64`.
        - Inputs :obj:`connectivity` and :obj:`element_indices` will be converted to :obj:`numpy.int64`.

    .. warning::

        No tests are performed to check if the inputs array are consistent
        (e.g., if the connectivity contains invalid vertex indices, ...).
        Only shapes are validated.
        The behavior of the function is undefined in this case.

    .. important::

        When using :obj:`-1` in :obj:`element_indices` for invalid elements,
        ensure to set :obj:`skip_m1` to :obj:`True` to avoid indexing errors.


    Parameters
    ----------
    property_array: ArrayLike
        An array of shape :math:`(N_{v}, P)` containing the property values defined at
        the nodes of the mesh. If 1D-array is provided, it will be treated as a
        single-component property of shape :math:`(N_{v}, 1)`.

    connectivity: ArrayLike
        An array of shape :math:`(N_{e}, N_{vpe})` defining the connectivity of the
        elements in the mesh, where each row contains the indices of the nodes that
        form an element.

    element_indices: ArrayLike
        An array of shape :math:`(N_{p},)` containing the indices of each element
        corresponding to the :math:`N_{p}` integration points.

    shape_function_derivatives: ArrayLike
        An array of shape :math:`(N_{p}, N_{vpe}, K)` containing the derivatives of
        the shape functions evaluated at :math:`N_{p}` points for the :math:`N_{vpe}`
        nodes of the element.

    jacobian_matrix: ArrayLike
        An array of shape :math:`(N_{p}, E, K)` containing the Jacobian matrices for
        each of the :math:`N_{p}` points. Each Jacobian matrix has dimensions
        :math:`(E \times K)`.

    skip_m1: :class:`bool`, optional
        If set to :obj:`True`, any element index of -1 in :obj:`element_indices` will
        result in the corresponding derivated properties being set to :obj:`default`.
        Default is :obj:`True`.

    default: Real, optional
        The default value to assign to derivated properties when the Jacobian matrix is
        singular or not invertible.
        Default is :obj:`numpy.nan`.


    Returns
    -------
    derivated_properties: :class:`numpy.ndarray`
        An array of shape :math:`(N_{p}, P, E)` containing the derivated property values
        at each of the :math:`N_{p}` integration points with respect to the global
        coordinates :math:`(x, y, z, ...)`.


    Notes
    -----
    In a space of dimension :math:`E` with a mesh constituted of :math:`N_{e}` elements
    and :math:`N_{v}` nodes, the mesh is composed of :math:`K`-dimensional elements
    (with :math:`K \leq E`) defined by :math:`N_{vpe}` nodes for each element.

    For a given set of :math:`N_p` integration points located within elements, the
    derivated property array has shape :math:`(N_p, P, K)` where :math:`P` is the number
    of property components (e.g., 1 for scalar properties, 3 for vector properties),
    and :math:`K` is the number of local coordinates.

    The derivative of the property at each integration point is computed as:

    If the dimension of the elements :math:`K` is equal to the space dimension
    :math:`E`:

    .. math::

        \nabla_X P(\xi, \eta, \zeta, ...) = J^{-T} \cdot \sum_{i=1}^{N_{vpe}} \nabla_{\Xi} N_i(\xi, \eta, \zeta, ...) P_i

    If the dimension of the elements :math:`K` is less than the space dimension
    :math:`E`:

    .. math::

        \nabla_X P(\xi, \eta, \zeta, ...) = J (J^T J)^{-1} \cdot \sum_{i=1}^{N_{vpe}} \nabla_{\Xi} N_i(\xi, \eta, \zeta, ...) P_i

    where :math:`N_i` are the shape functions associated with each node, :math:`P_i` are
    the property values at the nodes of the element, and :math:`J` is the Jacobian
    matrix of the transformation from natural to global coordinates.


    Demonstration
    --------------
    The property field is defined at the nodes of the mesh and can be interpolate to any
    point within the elements using shape functions.

    .. math::

        P(\Xi) = \sum_{i=1}^{N_{vpe}} N_i(\Xi) P_i

    The spatial derivative of the property field with respect to global coordinates is
    defined as:

    .. math::

        \nabla_X P(\Xi) = \sum_{i=1}^{N_{vpe}} \nabla_{X} N_i(\Xi) P_i

    The derivative of the shape functions with respect to global coordinates is obtained
    via the chain rule:

    .. math::

        \nabla_{X} N = J_{\Xi/X}^T \nabla_{\Xi} N

    where :math:`J_{\Xi/X}` is the Jacobian matrix of the transformation from global
    coordinates to natural coordinates, which is the inverse of the Jacobian matrix
    :math:`J = J_{X/\Xi}` of the transformation from natural coordinates to global
    coordinates.

    For :math:`K = E`, the Jacobian matrix is square and invertible :

    .. math::

        J_{\Xi/X} = J^{-1} \rightarrow \nabla_{X} N = J^{-T} \nabla_{\Xi} N

    And we obtain the formula as mentioned above:

    .. math::

        \nabla_X P(\Xi) = J^{-T} \cdot \sum_{i=1}^{N_{vpe}} \nabla_{\Xi} N_i(\Xi) P_i

    For :math:`K < E`, the Jacobian matrix is not square and thus not invertible.
    The pseudo-inverse is :

    .. math::

        J_{\Xi/X} = (J^T J)^{-1} J^T \rightarrow \nabla_{X} N = J (J^T J)^{-1} \nabla_{\Xi} N

    And we obtain the formula as mentioned above:

    .. math::

        \nabla_X P(\Xi) = J (J^T J)^{-1} \cdot \sum_{i=1}^{N_{vpe}} \nabla_{\Xi} N_i(\Xi) P_i


    See Also
    --------
    pysdic.compute_shape_functions:
        To compute the shape functions and their derivatives at given natural
        coordinates within elements.

    pysdic.assemble_jacobian_matrix:
        To assemble the Jacobian matrices from shape function derivatives and
        remapped coordinates.

    pysdic.compute_property_derivative:
        To compute the property derivatives directly from mesh and integration point
        information.


    Examples
    --------
    Lets construct a simple 2D mesh and compute the derivative of a scalar property at
    given integration points for triangular elements.

    .. code-block:: python
        :linenos:

        import numpy
        from pysdic import (
            remap_vertices_coordinates,
            assemble_jacobian_matrix,
            assemble_property_derivative,
            compute_triangle_3_shape_functions
        )

        vertices_coordinates = numpy.array(
            [[0.0, 0.0],
             [1.0, 0.0],
             [1.0, 1.0],
             [0.0, 1.0]]
        )

        connectivity = numpy.array(
            [[0, 1, 2],
             [0, 2, 3]]
        )

        natural_coordinates = numpy.array(
            [[0.2, 0.3],
             [0.6, 0.2]]
        )

        element_indices = numpy.array([0, 1])

        property_array = numpy.array([10.0, 20.0, 30.0, 40.0])  # Scalar property

        remapped_coords = remap_vertices_coordinates(
            vertices_coordinates=vertices_coordinates,
            connectivity=connectivity,
            element_indices=element_indices
        )

        _, shape_func_derivs = compute_triangle_3_shape_functions(
            natural_coordinates=natural_coordinates,
            return_derivatives=True
        )

        jacobians = assemble_jacobian_matrix(
            shape_function_derivatives=shape_func_derivs,
            remapped_coordinates=remapped_coords
        )

        derivated_properties = assemble_property_derivative(
            property_array=property_array,
            connectivity=connectivity,
            element_indices=element_indices,
            shape_function_derivatives=shape_func_derivs,
            jacobian_matrix=jacobians
        )

        print(f"derivated properties (shape={derivated_properties.shape}):")
        print(derivated_properties)

    .. code-block:: console

        derivated properties (shape=(2, 1, 2)):
        [[[ 10.  10.]]

         [[-10.  30.]]]

    """
    # Validate input dimensions
    property_array = numpy.asarray(property_array, dtype=numpy.float64)
    connectivity = numpy.asarray(connectivity, dtype=numpy.int64)
    element_indices = numpy.asarray(element_indices, dtype=numpy.int64)
    shape_function_derivatives = numpy.asarray(
        shape_function_derivatives, dtype=numpy.float64
    )
    jacobian_matrix = numpy.asarray(jacobian_matrix, dtype=numpy.float64)

    if property_array.ndim == 1:
        property_array = property_array[:, numpy.newaxis]
    if property_array.ndim != 2:
        raise ValueError(
            f"'property_array' must be a 1D or 2D array of shape (N_v,) or (N_v, P)."
            f" Got {property_array.ndim}D array instead."
        )
    if connectivity.ndim != 2:
        raise ValueError(
            f"'connectivity' must be a 2D array of shape (N_e, N_npe)."
            f" Got {connectivity.ndim}D array instead."
        )
    if element_indices.ndim != 1:
        raise ValueError(
            f"'element_indices' must be a 1D array of shape (N_p,)."
            f" Got {element_indices.ndim}D array instead."
        )
    if shape_function_derivatives.ndim != 3:
        raise ValueError(
            f"'shape_function_derivatives' must be a 3D array of shape (N_p, N_npe, K)."
            f" Got {shape_function_derivatives.ndim}D array instead."
        )
    if jacobian_matrix.ndim != 3:
        raise ValueError(
            f"'jacobian_matrix' must be a 3D array of shape (N_p, E, K)."
            f" Got {jacobian_matrix.ndim}D array instead."
        )
    if not shape_function_derivatives.shape[0] == jacobian_matrix.shape[0]:
        raise ValueError(
            f"The first dimension (N_p) of 'shape_function_derivatives' and"
            f" 'jacobian_matrix' must match. Got {shape_function_derivatives.shape[0]}"
            f" and {jacobian_matrix.shape[0]} respectively."
        )
    if not connectivity.shape[1] == shape_function_derivatives.shape[1]:
        raise ValueError(
            f"The second dimension (N_npe) of 'connectivity' and"
            f" 'shape_function_derivatives' must match. Got {connectivity.shape[1]}"
            f" and {shape_function_derivatives.shape[1]} respectively."
        )
    if not element_indices.shape[0] == shape_function_derivatives.shape[0]:
        raise ValueError(
            f"The first dimension (N_p) of 'element_indices' and"
            f" 'shape_function_derivatives' must match. Got {element_indices.shape[0]}"
            f" and {shape_function_derivatives.shape[0]} respectively."
        )
    if not shape_function_derivatives.shape[2] == jacobian_matrix.shape[2]:
        raise ValueError(
            f"The third dimension (K) of 'shape_function_derivatives' and"
            f" 'jacobian_matrix' must match. Got {shape_function_derivatives.shape[2]}"
            f" and {jacobian_matrix.shape[2]} respectively."
        )
    if not isinstance(skip_m1, bool):
        raise ValueError(
            f"'skip_m1' must be a boolean value. Got {type(skip_m1)} instead."
        )
    if not isinstance(default, Real):
        raise ValueError(
            f"'default' must be a real number. Got {type(default)} instead."
        )

    # Extract dimensions
    N_p = shape_function_derivatives.shape[0]
    N_npe = shape_function_derivatives.shape[1]
    K = shape_function_derivatives.shape[2]
    E = jacobian_matrix.shape[1]
    P = property_array.shape[1]

    # Extract -1 mask if skip_m1 is True
    if skip_m1:
        m1_mask = element_indices == -1
        numpy.logical_not(m1_mask, out=m1_mask)
        valid_indices = element_indices[m1_mask]
        jacobian_matrix = jacobian_matrix[m1_mask, :, :]
        Npv = valid_indices.shape[0]
    else:
        valid_indices = element_indices
        jacobian_matrix = jacobian_matrix
        Npv = N_p

    # Invert jacobians
    if E == K:
        # here M = J^-T
        jacobians_inv = numpy.linalg.inv(jacobian_matrix)  # Shape: (N_pv, K, E)
        jacobians_inv = numpy.transpose(
            jacobians_inv, axes=(0, 2, 1)
        )  # Transpose to get J^{-T} with shape: (N_pv, E, K)
    else:
        # Moore-Penrose pseudo-inverse for non-square Jacobian matrices
        # K < E case: J is a (E x K) matrix, we compute the pseudo-inverse
        # Here M = J (J^T J)^{-1}
        JTJ = numpy.einsum(
            "mjk,mjl->mkl", jacobian_matrix, jacobian_matrix
        )  # Shape: (N_pv, K, K)
        JTJ_inv = numpy.linalg.inv(JTJ)
        jacobians_inv = numpy.einsum(
            "mjk,mkl->mjl", jacobian_matrix, JTJ_inv
        )  # Shape: (N_pv, E, K)

    # Compute derivated properties
    vertices_properties = property_array[
        connectivity[valid_indices, :]
    ]  # Shape: (N_p, N_npe, P)
    derivative_sum = numpy.einsum(
        "mjp,mjk->mpk",
        vertices_properties,
        (
            shape_function_derivatives[valid_indices, :, :]
            if skip_m1
            else shape_function_derivatives
        ),
    )  # Shape: (N_pv, P, K)
    derivated_properties_valid = numpy.einsum(
        "mek,mpk->mpe", jacobians_inv, derivative_sum
    )  # Shape: (N_pv, P, E)

    if skip_m1:
        derivated_properties = numpy.full((N_p, P, E), default, dtype=numpy.float64)
        derivated_properties[m1_mask, :, :] = derivated_properties_valid
    else:
        derivated_properties = derivated_properties_valid

    return derivated_properties


def compute_property_derivative(
    property_array: ArrayLike,
    vertices_coordinates: ArrayLike,
    connectivity: ArrayLike,
    element_type: str,
    natural_coordinates: ArrayLike,
    element_indices: ArrayLike,
    *,
    skip_m1: bool = True,
    default: Real = numpy.nan,
) -> numpy.ndarray:
    r"""
    Compute and assemble the derivative of a property defined at the nodes of a mesh to
    given integration points within elements with respect to global coordinates
    :math:`(x,y,z,...)` from mesh and integration point information.

    ``derivated_properties[i]`` contains the derivated property at the integration point
    :math:`i`.

    This function combines the remapping of vertex coordinates, computation of shape
    function derivatives, assembly of the Jacobian matrices, and assembly of the
    property derivatives for the entire mesh.

    .. note::

        - Inputs :obj:`property_array`, :obj:`vertices_coordinates` and :obj:`natural_coordinates` will be converted to :obj:`numpy.float64`.
        - Inputs :obj:`connectivity` and :obj:`element_indices` will be converted to :obj:`numpy.int64`.

    .. warning::

        No tests are performed to check if the inputs array are consistent
        (e.g., if the connectivity contains invalid vertex indices, ...).
        Only shapes are validated.
        The behavior of the function is undefined in this case.

    .. important::

        When using :obj:`-1` in :obj:`element_indices` for invalid elements,
        ensure to set :obj:`skip_m1` to :obj:`True` to avoid indexing errors.


    Parameters
    ----------
    property_array: ArrayLike
        An array of shape :math:`(N_{v}, P)` containing the property values defined at
        the nodes of the mesh. If 1D-array is provided, it will be treated as a
        single-component property of shape :math:`(N_{v}, 1)`.

    vertices_coordinates: ArrayLike
        An array of shape :math:`(N_{v}, E)` containing the global coordinates
        :math:`(x, y, z, ...)` of the vertices in the mesh.

    connectivity: ArrayLike
        An array of shape :math:`(N_{e}, N_{vpe})` defining the connectivity of the
        elements in the mesh, where each row contains the indices of the nodes that
        form an element.

    element_type: :class:`str`
        A string specifying the type of element (e.g., 'segment_2', 'triangle_3', etc.)
        to determine which shape function to use.

    natural_coordinates: ArrayLike
        An array of shape :math:`(N_{p}, K)` containing the natural coordinates of the
        integration points within elements where :math:`K` is the topological dimension
        of the element (e.g., 1 for segments, 2 for triangles/quadrangles, etc.).

    element_indices: ArrayLike
        An array of shape :math:`(N_{p},)` containing the indices of each element
        corresponding to the :math:`N_{p}` integration points.

    skip_m1: :class:`bool`, optional
        If set to :obj:`True`, any element index of -1 in :obj:`element_indices` will
        result in the corresponding derivated properties being set to :obj:`default`.
        Default is :obj:`True`.

    default: Real, optional
        The default value to assign to derivated properties for integration points
        associated with an element index of -1 when :obj:`skip_m1` is :obj:`True`.
        The out-of-range natural coordinates will also result in derivated properties
        being set to this value.
        Default is :obj:`numpy.nan`.


    Returns
    -------
    derivated_properties: :class:`numpy.ndarray`
        An array of shape :math:`(N_{p}, P, E)` containing the derivated property values
        at each of the :math:`N_{p}` integration points with respect to the global
        coordinates :math:`(x, y, z, ...)`.


    Notes
    -----
    In a space of dimension :math:`E` with a mesh constituted of :math:`N_{e}` elements
    and :math:`N_{v}` nodes, the mesh is composed of :math:`K`-dimensional elements
    (with :math:`K \leq E`) defined by :math:`N_{vpe}` nodes for each element.

    For a given set of :math:`N_p` integration points located within elements, the
    derivated property array has shape :math:`(N_p, P, K)` where :math:`P` is the number
    of property components (e.g., 1 for scalar properties, 3 for vector properties),
    and :math:`K` is the number of local coordinates.

    The derivative of the property at each integration point is computed as:

    If the dimension of the elements :math:`K` is equal to the space dimension
    :math:`E`:

    .. math::

        \nabla_X P(\xi, \eta, \zeta, ...) = J^{-T} \cdot \sum_{i=1}^{N_{vpe}} \nabla_{\Xi} N_i(\xi, \eta, \zeta, ...) P_i

    See the notes in :func:`pysdic.assemble_property_derivative` for more details on the
    behaviour when :math:`K < E` and the use of pseudo-inverse for
    non-square Jacobian matrices.


    See Also
    --------
    pysdic.compute_shape_functions:
        To compute the shape functions and their derivatives at given natural
        coordinates within elements.

    pysdic.compute_jacobian_matrix:
        To compute the Jacobian matrices from mesh and integration point information.

    pysdic.assemble_property_derivative:
        To assemble the property derivatives from precomputed shape function derivatives
        and Jacobian matrices.


    Examples
    --------
    Lets construct a simple 2D mesh and compute the derivative of a scalar property at
    given integration points for triangular elements.

    .. code-block:: python
        :linenos:

        import numpy
        from pysdic import compute_property_derivative

        vertices_coordinates = numpy.array(
            [[0.0, 0.0],
             [1.0, 0.0],
             [1.0, 1.0],
             [0.0, 1.0]]
        )

        connectivity = numpy.array(
            [[0, 1, 2],
             [0, 2, 3]]
        )

        natural_coordinates = numpy.array(
            [[0.2, 0.3],
             [0.6, 0.2]]
        )

        element_indices = numpy.array([0, 1])

        property_array = numpy.array([10.0, 20.0, 30.0, 40.0])  # Scalar property

        derivated_properties = compute_property_derivative(
            property_array=property_array,
            vertices_coordinates=vertices_coordinates,
            connectivity=connectivity,
            element_type='triangle_3',
            natural_coordinates=natural_coordinates,
            element_indices=element_indices
        )

        print(f"derivated properties (shape={derivated_properties.shape}):")
        print(derivated_properties)


    .. code-block:: console

        derivated properties (shape=(2, 1, 2)):
        [[[ 10.  10.]]

         [[-10.  30.]]]

    """
    # Validate input dimensions
    property_array = numpy.asarray(property_array, dtype=numpy.float64)
    vertices_coordinates = numpy.asarray(vertices_coordinates, dtype=numpy.float64)
    connectivity = numpy.asarray(connectivity, dtype=numpy.int64)
    natural_coordinates = numpy.asarray(natural_coordinates, dtype=numpy.float64)
    element_indices = numpy.asarray(element_indices, dtype=numpy.int64)

    if property_array.ndim == 1:
        property_array = property_array[:, numpy.newaxis]
    if property_array.ndim != 2:
        raise ValueError(
            f"'property_array' must be a 1D or 2D array of shape (N_v,) or (N_v, P)."
            f" Got {property_array.ndim}D array instead."
        )
    if vertices_coordinates.ndim != 2:
        raise ValueError(
            f"'vertices_coordinates' must be a 2D array of shape (N_v, E)."
            f" Got {vertices_coordinates.ndim}D array instead."
        )
    if connectivity.ndim != 2:
        raise ValueError(
            f"'connectivity' must be a 2D array of shape (N_e, N_npe)."
            f" Got {connectivity.ndim}D array instead."
        )
    if natural_coordinates.ndim != 2:
        raise ValueError(
            f"'natural_coordinates' must be a 2D array of shape (N_p, K)."
            f" Got {natural_coordinates.ndim}D array instead."
        )
    if element_indices.ndim != 1:
        raise ValueError(
            f"'element_indices' must be a 1D array of shape (N_p,)."
            f" Got {element_indices.ndim}D array instead."
        )
    if not natural_coordinates.shape[0] == element_indices.shape[0]:
        raise ValueError(
            f"The first dimension (N_p) of 'natural_coordinates' and 'element_indices'"
            f" must match. Got {natural_coordinates.shape[0]} and"
            f" {element_indices.shape[0]} respectively."
        )
    if not isinstance(element_type, str):
        raise ValueError(
            f"'element_type' must be a string. Got {type(element_type)} instead."
        )
    if not isinstance(skip_m1, bool):
        raise ValueError(
            f"'skip_m1' must be a boolean value. Got {type(skip_m1)} instead."
        )
    if not isinstance(default, Real):
        raise ValueError(
            f"'default' must be a real number. Got {type(default)} instead."
        )

    E = vertices_coordinates.shape[1]
    K = natural_coordinates.shape[1]
    Np = natural_coordinates.shape[0]
    Nvpe = connectivity.shape[1]

    # Remap vertices coordinates
    remapped_coords = remap_vertices_coordinates(
        vertices_coordinates=vertices_coordinates,
        connectivity=connectivity,
        element_indices=element_indices,
        skip_m1=skip_m1,
        default=default,
    )  # Shape: (N_p, N_vpe, E)

    # Compute shape function derivatives
    _, shape_func_derivs = compute_shape_functions(
        natural_coordinates=natural_coordinates,
        element_type=element_type,
        return_derivatives=True,
        default=default,
    )  # Shape: (N_p, N_vpe, K)

    # Assemble Jacobian matrices
    jacobians = assemble_jacobian_matrix(
        shape_function_derivatives=shape_func_derivs,
        remapped_coordinates=remapped_coords,
    )  # Shape: (N_p, E, K)

    # Assemble property derivatives
    derivated_properties = assemble_property_derivative(
        property_array=property_array,
        connectivity=connectivity,
        element_indices=element_indices,
        shape_function_derivatives=shape_func_derivs,
        jacobian_matrix=jacobians,
        skip_m1=skip_m1,
        default=default,
    )

    return derivated_properties


def assemble_property_interpolation(
    property_array: ArrayLike,
    connectivity: ArrayLike,
    element_indices: ArrayLike,
    shape_functions: ArrayLike,
    *,
    skip_m1: bool = True,
    default: Union[Real, ArrayLike] = numpy.nan,
) -> numpy.ndarray:
    r"""
    Interpolate a property defined at the nodes of a mesh to given integration points
    within elements using precomputed shape functions .

    ``interpolated_properties[i]`` contains the interpolated property at the integration
    point :math:`i`.

    .. note::

        - Inputs :obj:`property_array` and :obj:`shape_functions` will be converted to :obj:`numpy.float64`.
        - Inputs :obj:`connectivity` and :obj:`element_indices` will be converted to :obj:`numpy.int64`.

    .. warning::

        No tests are performed to check if the inputs array are consistent
        (e.g., if the connectivity contains invalid vertex indices, ...).
        Only shapes are validated.
        The behavior of the function is undefined in this case.

    .. important::

        When using :obj:`-1` in :obj:`element_indices` for invalid elements,
        ensure to set :obj:`skip_m1` to :obj:`True` to avoid indexing errors.


    Parameters
    ----------
    property_array: ArrayLike
        An array of shape :math:`(N_{v}, P)` containing the property values defined at
        the nodes of the mesh. If 1D-array is provided, it will be treated as a
        single-component property of shape :math:`(N_{v}, 1)`.

    connectivity: ArrayLike
        An array of shape :math:`(N_{e}, N_{vpe})` defining the connectivity of the
        elements in the mesh, where each row contains the indices of the nodes that
        form an element.

    element_indices: ArrayLike
        An array of shape :math:`(N_{p},)` containing the indices of each element
        corresponding to the :math:`N_{p}` integration points.

    shape_functions: ArrayLike
        An array of shape :math:`(N_{p}, N_{vpe})` containing the shape function values
        evaluated at :math:`N_{p}` points for the :math:`N_{vpe}` nodes of the element.

    skip_m1: :class:`bool`, optional
        If set to :obj:`True`, any element index of -1 in :obj:`element_indices` will
        result in the corresponding interpolated property being set to :obj:`default`.
        Default is :obj:`True`.

    default: Union[Real, ArrayLike], optional
        The default value to assign to interpolated properties for integration points
        associated with an element index of -1 when :obj:`skip_m1` is :obj:`True`.
        Default is :obj:`numpy.nan`. The input can also be a :class:`numpy.ndarray` of
        shape (P,) to assign different default values for each property component.


    Returns
    -------
    interpolated_properties: :class:`numpy.ndarray`
        An array of shape :math:`(N_{p}, P)` containing the interpolated property values
        at each of the :math:`N_{p}` integration points.


    Notes
    -----
    In a space of dimension :math:`E` with a mesh constituted of :math:`N_{e}` elements
    and :math:`N_{v}` nodes, the mesh is composed of :math:`K`-dimensional elements
    (with :math:`K \leq E`) defined by :math:`N_{vpe}` nodes for each element.

    For a given set of :math:`N_{p}` integration points located within elements, the
    interpolated property array has shape :math:`(N_{p}, P)` where :math:`P` is the
    number of property components (e.g., 1 for scalar properties, 3 for vector
    properties).

    The property at each integration point is computed as:

    .. math::

        P(\xi, \eta, \zeta, ...) = \sum_{i=1}^{N_{vpe}} N_i(\xi, \eta, \zeta, ...) P_i

    where :math:`N_i` are the shape functions associated with each node, and :math:`P_i`
    are the property values at the nodes of the element.


    See Also
    --------
    pysdic.compute_shape_functions:
        To compute the shape functions at given natural coordinates within elements.

    pysdic.compute_property_interpolation:
        To compute the property interpolation directly from mesh and integration point
        information.

    pysdic.assemble_property_interpolation:
        To project properties defined at integration points back to the nodes of the
        mesh.


    Examples
    --------
    Lets construct a simple 2D mesh and interpolate a scalar property at given
    integration points for triangular elements.

    .. code-block:: python
        :linenos:

        import numpy
        from pysdic import (
            compute_shape_functions,
            assemble_property_interpolation
        )

        vertices_coordinates = numpy.array(
            [[0.0, 0.0],
             [1.0, 0.0],
             [1.0, 1.0],
             [0.0, 1.0]]
        )

        connectivity = numpy.array(
            [[0, 1, 2],
             [0, 2, 3]]
        )

        natural_coordinates = numpy.array(
            [[0.2, 0.3],
             [0.6, 0.2]]
        )

        element_indices = numpy.array([0, 1])

        property_array = numpy.array([10.0, 20.0, 30.0, 40.0])  # Scalar property

        shape_functions = compute_shape_functions(
            natural_coordinates=natural_coordinates,
            element_type='triangle_3',
            return_derivatives=False
        )

        interpolated_props = assemble_property_interpolation(
            property_array=property_array,
            connectivity=connectivity,
            element_indices=element_indices,
            shape_functions=shape_functions
        )

        print(f"interpolated properties (shape={interpolated_props.shape}):")
        print(interpolated_props)

    .. code-block:: console

        interpolated properties (shape=(2, 1)):
        [[18.  ]
         [28. ]]

    """
    # Validate input dimensions
    property_array = numpy.asarray(property_array, dtype=numpy.float64)
    if property_array.ndim == 1:
        property_array = property_array[
            :, numpy.newaxis
        ]  # Convert to 2D array with shape (N_v, 1)

    shape_functions = numpy.asarray(shape_functions, dtype=numpy.float64)
    connectivity = numpy.asarray(connectivity, dtype=numpy.int64)
    element_indices = numpy.asarray(element_indices, dtype=numpy.int64)

    if property_array.ndim != 2:
        raise ValueError(
            f"'property_array' must be a 2D array of shape (N_v, P)."
            f" Got {property_array.ndim}D array instead."
        )
    if shape_functions.ndim != 2:
        raise ValueError(
            f"'shape_functions' must be a 2D array of shape (N_p, N_npe)."
            f" Got {shape_functions.ndim}D array instead."
        )
    if connectivity.ndim != 2:
        raise ValueError(
            f"'connectivity' must be a 2D array of shape (N_e, N_npe)."
            f" Got {connectivity.ndim}D array instead."
        )
    if element_indices.ndim != 1:
        raise ValueError(
            f"'element_indices' must be a 1D array of shape (N_p,)."
            f" Got {element_indices.ndim}D array instead."
        )
    if not shape_functions.shape[0] == element_indices.shape[0]:
        raise ValueError(
            f"The first dimension (N_p) of 'shape_functions' and 'element_indices' must"
            f" match. Got {shape_functions.shape[0]} and {element_indices.shape[0]}"
            f" respectively."
        )
    if not shape_functions.shape[1] == connectivity.shape[1]:
        raise ValueError(
            f"The second dimension (N_npe) of 'shape_functions' and 'connectivity' must"
            f" match. Got {shape_functions.shape[1]} and {connectivity.shape[1]}"
            f" respectively."
        )

    if not isinstance(skip_m1, bool):
        raise ValueError(
            f"'skip_m1' must be a boolean value. Got {type(skip_m1)} instead."
        )
    if isinstance(default, Real):
        default = numpy.full(
            (property_array.shape[1],), default, dtype=numpy.float64
        )  # Convert to array of shape (P,)
    if not (
        isinstance(default, Real)
        or (
            isinstance(default, numpy.ndarray)
            and default.shape == (property_array.shape[1],)
        )
    ):
        raise ValueError(
            f"'default' must be a real number or a numpy.ndarray of shape (P,)."
            f" Got {type(default)} with shape {getattr(default, 'shape', None)}"
            f" instead."
        )

    # Extract dimensions
    N_p = shape_functions.shape[0]
    N_npe = shape_functions.shape[1]
    P = property_array.shape[1]

    # Extract -1 mask if skip_m1 is True
    if skip_m1:
        m1_mask = element_indices == -1
        numpy.logical_not(m1_mask, out=m1_mask)
        valid_indices = element_indices[m1_mask]
    else:
        valid_indices = element_indices

    # Interpolate properties
    vertices_properties = property_array[
        connectivity[valid_indices, :], :
    ]  # Shape: (N_p, N_npe, P)
    interpolated_properties_valid = numpy.einsum(
        "mi,mip->mp",
        shape_functions[m1_mask] if skip_m1 else shape_functions,
        vertices_properties,
    )  # Shape: (N_p, P)

    # Handle -1 entries if skip_m1 is True
    if skip_m1:
        interpolated_properties = numpy.tile(default, (N_p, 1))
        interpolated_properties[m1_mask, :] = interpolated_properties_valid
    else:
        interpolated_properties = interpolated_properties_valid

    return interpolated_properties


def compute_property_interpolation(
    property_array: ArrayLike,
    connectivity: ArrayLike,
    element_type: str,
    natural_coordinates: ArrayLike,
    element_indices: ArrayLike,
    *,
    skip_m1: bool = True,
    default: Union[Real, ArrayLike] = numpy.nan,
) -> numpy.ndarray:
    r"""
    Interpolate a property defined at the nodes of a mesh to given integration points
    within elements using mesh and integration point information.

    This function combines the remapping of vertex coordinates, computation of shape
    functions, and assembly of the property interpolation for the entire mesh.

    ``interpolated_properties[i]`` contains the interpolated property at the integration
    point :math:`i`.

    .. note::

        - Inputs :obj:`property_array` and :obj:`natural_coordinates` will be converted to :obj:`numpy.float64`.
        - Inputs :obj:`connectivity` and :obj:`element_indices` will be converted to :obj:`numpy.int64`.

    .. warning::

        No tests are performed to check if the inputs array are consistent
        (e.g., if the connectivity contains invalid vertex indices, ...).
        Only shapes are validated.
        The behavior of the function is undefined in this case.

    .. important::

        When using :obj:`-1` in :obj:`element_indices` for invalid elements,
        ensure to set :obj:`skip_m1` to :obj:`True` to avoid indexing errors.


    Parameters
    ----------
    property_array: ArrayLike
        An array of shape :math:`(N_{v}, P)` containing the property values defined at
        the nodes of the mesh. If 1D-array is provided, it will be treated as a
        single-component property of shape :math:`(N_{v}, 1)`.

    connectivity: ArrayLike
        An array of shape :math:`(N_{e}, N_{vpe})` defining the connectivity of the
        elements in the mesh, where each row contains the indices of the nodes that
        form an element.

    element_type: :class:`str`
        A string specifying the type of element (e.g., 'segment_2', 'triangle_3', etc.)
        to determine which shape function to use.

    natural_coordinates: ArrayLike
        An array of shape :math:`(N_{p}, K)` containing the natural coordinates of the
        integration points within elements where :math:`K` is the topological dimension
        of the element (e.g., 1 for segments, 2 for triangles/quadrangles, etc.).

    element_indices: ArrayLike
        An array of shape :math:`(N_{p},)` containing the indices of each element
        corresponding to the :math:`N_{p}` integration points.

    skip_m1: :class:`bool`, optional
        If set to :obj:`True`, any element index of -1 in :obj:`element_indices` will
        result in the corresponding interpolated property being set to :obj:`default`.
        Default is :obj:`True`.

    default: Union[Real, ArrayLike], optional
        The default value to assign to interpolated properties for integration points
        associated with an element index of -1 when :obj:`skip_m1` is :obj:`True`.
        Default is :obj:`numpy.nan`. The input can also be a :class:`numpy.ndarray` of
        shape (P,) to assign different default values for each property component.


    Returns
    -------
    interpolated_properties: :class:`numpy.ndarray`
        An array of shape :math:`(N_{p}, P)` containing the interpolated property values
        at each of the :math:`N_{p}` integration points.


    Notes
    -----
    In a space of dimension :math:`E` with a mesh constituted of :math:`N_{e}` elements
    and :math:`N_{v}` nodes, the mesh is composed of :math:`K`-dimensional elements
    (with :math:`K \leq E`) defined by :math:`N_{vpe}` nodes for each element.

    For a given set of :math:`N_{p}` integration points located within elements, the
    interpolated property array has shape :math:`(N_{p}, P)` where :math:`P` is the
    number of property components (e.g., 1 for scalar properties, 3 for vector
    properties).

    The property at each integration point is computed as:

    .. math::

        P(\xi, \eta, \zeta, ...) = \sum_{i=1}^{N_{vpe}} N_i(\xi, \eta, \zeta, ...) P_i

    where :math:`N_i` are the shape functions associated with each node, and :math:`P_i`
    are the property values at the nodes of the element.


    See Also
    --------
    pysdic.compute_shape_functions:
        To compute the shape functions at given natural coordinates within elements.

    pysdic.assemble_property_interpolation:
        To assemble the property interpolation from precomputed shape functions.

    pysdic.compute_property_derivative:
        To compute the derivative of a property at integration points within elements.

    pysdic.compute_property_projection:
        To project properties defined at integration points back to the nodes of the
        mesh.


    Examples
    --------
    Lets construct a simple 2D mesh and interpolate a scalar property at given
    integration points for triangular elements.

    .. code-block:: python
        :linenos:

        import numpy
        from pysdic import compute_property_interpolation

        vertices_coordinates = numpy.array(
            [[0.0, 0.0],
             [1.0, 0.0],
             [1.0, 1.0],
             [0.0, 1.0]]
        )

        connectivity = numpy.array(
            [[0, 1, 2],
             [0, 2, 3]]
        )

        natural_coordinates = numpy.array(
            [[0.2, 0.3],
             [0.6, 0.2]]
        )

        element_indices = numpy.array([0, 1])

        property_array = numpy.array([10.0, 20.0, 30.0, 40.0])  # Scalar property

        interpolated_props = compute_property_interpolation(
            property_array=property_array,
            connectivity=connectivity,
            element_type='triangle_3',
            natural_coordinates=natural_coordinates,
            element_indices=element_indices
        )

        print(f"interpolated properties (shape={interpolated_props.shape}):")
        print(interpolated_props)


    .. code-block:: console

        interpolated properties (shape=(2, 1)):
        [[18.  ]
         [28. ]]

    """
    # Validate input dimensions
    property_array = numpy.asarray(property_array, dtype=numpy.float64)
    connectivity = numpy.asarray(connectivity, dtype=numpy.int64)
    natural_coordinates = numpy.asarray(natural_coordinates, dtype=numpy.float64)
    element_indices = numpy.asarray(element_indices, dtype=numpy.int64)

    if property_array.ndim == 1:
        property_array = property_array[:, numpy.newaxis]
    if property_array.ndim != 2:
        raise ValueError(
            f"'property_array' must be a 1D or 2D array of shape (N_v,) or (N_v, P)."
            f" Got {property_array.ndim}D array instead."
        )
    if connectivity.ndim != 2:
        raise ValueError(
            f"'connectivity' must be a 2D array of shape (N_e, N_npe)."
            f" Got {connectivity.ndim}D array instead."
        )
    if natural_coordinates.ndim != 2:
        raise ValueError(
            f"'natural_coordinates' must be a 2D array of shape (N_p, K)."
            f" Got {natural_coordinates.ndim}D array instead."
        )
    if element_indices.ndim != 1:
        raise ValueError(
            f"'element_indices' must be a 1D array of shape (N_p,)."
            f" Got {element_indices.ndim}D array instead."
        )
    if not natural_coordinates.shape[0] == element_indices.shape[0]:
        raise ValueError(
            f"The first dimension (N_p) of 'natural_coordinates' and 'element_indices'"
            f" must match. Got {natural_coordinates.shape[0]} and"
            f" {element_indices.shape[0]} respectively."
        )
    if not isinstance(element_type, str):
        raise ValueError(
            f"'element_type' must be a string. Got {type(element_type)} instead."
        )
    if not isinstance(skip_m1, bool):
        raise ValueError(
            f"'skip_m1' must be a boolean value. Got {type(skip_m1)} instead."
        )

    # Compute shape functions
    shape_functions = compute_shape_functions(
        natural_coordinates=natural_coordinates,
        element_type=element_type,
        return_derivatives=False,
        default=default,
    )

    # Assemble property interpolation
    interpolated_properties = assemble_property_interpolation(
        property_array=property_array,
        connectivity=connectivity,
        element_indices=element_indices,
        shape_functions=shape_functions,
        skip_m1=skip_m1,
        default=default,
    )

    return interpolated_properties


def assemble_property_projection(
    property_array: ArrayLike,
    shape_function_matrix: Union[ArrayLike, scipy.sparse.csr_matrix],
    point_weights: Optional[ArrayLike] = None,
    *,
    return_unaffected: bool = False,
) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
    r"""
    Project a property defined at integration points within elements back to the nodes
    of a mesh using a precomputed shape function matrix.

    ``projected_properties[i]`` contains the projected property at the node :math:`i`.

    .. note::

        - Inputs :obj:`property_array`, :obj:`shape_function_matrix` and :obj:`point_weights` will be converted to :obj:`numpy.float64`.

    .. warning::

        No tests are performed to check if the inputs array are consistent
        (e.g., if the connectivity contains invalid vertex indices, ...).
        Only shapes are validated.
        The behavior of the function is undefined in this case.


    Parameters
    ----------
    property_array: ArrayLike
        An array of shape :math:`(N_{p}, P)` containing the property values defined at
        the :math:`N_{p}` integration points. If 1D-array is provided, it will be
        treated as a single-component property of shape :math:`(N_{p}, 1)`.

    shape_function_matrix: Union[ArrayLike, scipy.sparse.csr_matrix]
        A precomputed shape functions matrix of shape :math:`(N_{p}, N_{v})` where
        :math:`N_{v}` is the number of nodes in the mesh.

    point_weights: Optional[ArrayLike], optional
        An array of shape :math:`(N_{p},)` containing the weights associated with each
        integration point. If not provided, all weights will be assumed to be equal to
        one.

    return_unaffected: :class:`bool`, optional
        If set to :obj:`True`, the function will return a tuple containing the projected
        properties and a boolean mask indicating which nodes were unaffected by any
        integration point.
        Default is :obj:`False`.


    Returns
    -------
    projected_properties: :class:`numpy.ndarray`
        An array of shape :math:`(N_{v}, P)` containing the projected property values at
        the nodes of the mesh.

    unaffected_mask: :class:`numpy.ndarray`, optional
        A boolean array of shape :math:`(N_{v},)` indicating which nodes were unaffected
        by any integration point. This is only returned if :obj:`return_unaffected` is
        set to :obj:`True`.
        :obj:`True` indicates the node was unaffected, while :obj:`False` indicates it
        was affected. So using :obj:`projected_properties[unaffected_mask, :] = default`
        will set the unaffected nodes to :obj:`default`.


    Notes
    -----
    In a space of dimension :math:`E` with a mesh constituted of :math:`N_{e}` elements
    and :math:`N_{v}` nodes, the mesh is composed of :math:`K`-dimensional elements
    (with :math:`K \leq E`) defined by :math:`N_{vpe}` nodes for each element.

    The evaluation of a property at the given integration points is represented by an
    array of shape :math:`(N_{p}, P)` and is performed as:

    .. math::

        P_{points} = N_f \cdot P_{nodes} \Leftrightarrow P(\xi, \eta, \zeta, ...) = \sum_{i=1}^{N_{vpe}} N_i(\xi, \eta, \zeta, ...) P_i

    where :math:`N_f` is the shape functions matrix assembled for all :math:`N_{p}`
    integration points, :math:`P_{points}` is the property array at the integration
    points, and :math:`P_{nodes}` is the property array at the mesh nodes.

    To project back to the nodes, we solve the following system using the pseudo-inverse
    of the shape functions matrix:

    .. math::

        P_{nodes} = (N_f^T W N_f)^{-1} N_f^T W P_{points}

    where :math:`W` is a diagonal matrix of weights associated with each integration
    point.


    Demonstration
    --------------
    Lets consider the following notations:

    - :math:`N_v` the number of nodes in the mesh.
    - :math:`N_{p}` the number of integration points.
    - :math:`P` the number of property components (e.g., 1 for scalar properties, 3 for vector properties).
    - :math:`P_{points}` the property array at the integration points with shape :math:`(N_{p}, P)`.
    - :math:`P_{nodes}` the property array at the mesh nodes with shape :math:`(N_v, P)`.
    - :math:`N_f` the shape functions matrix with shape :math:`(N_{p}, N_v)`.
    - :math:`W` the diagonal matrix of weights with shape :math:`(N_{p}, N_{p})`.

    We know that:

    .. math::

        P_{points} = N_f \cdot P_{nodes}

    To project back to the vertices of the mesh, we want tot minimize the following
    weighted least squares problem:

    .. math::

        \hat{P_{nodes}} = \min_{P_{nodes}} ( \frac{1}{2}\sum_{i=1}^{N_p} w_i \| P_{points,i} - N_{f,i} \cdot P_{nodes} \|^2 )

    .. math::

        \hat{P_{nodes}} = \min_{P_{nodes}} ( \frac{1}{2} (P_{points} - N_f \cdot P_{nodes})^T W (P_{points} - N_f \cdot P_{nodes}) )

    The gradient of the objective function is:

    .. math::

        \nabla_{P_{nodes}} \hat{P_{nodes}} = - N_f^T W (P_{points} - N_f \cdot P_{nodes})

    To find the optimal solution, we set the gradient to zero and solve for
    :math:`P_{nodes}`:

    .. math::

        \nabla_{P_{nodes}} \hat{P_{nodes}}  = 0 \Rightarrow N_f^T W P_{points} = N_f^T W N_f \cdot P_{nodes}

    So if we denote :math:`A = N_f^T W N_f` invertible, we have:

    .. math::

        \hat{P_{nodes}} = A^{-1} N_f^T W P_{points}


    See Also
    --------
    pysdic.compute_shape_functions:
        To compute the shape functions at given natural coordinates within elements.

    pysdic.compute_property_projection:
        To project properties defined at integration points back to the nodes of the
        mesh using mesh and integration point information.

    pysdic.assemble_property_interpolation:
        To interpolate properties defined at the nodes of the mesh to integration points
        within elements.


    Examples
    --------
    Lets construct a simple 2D mesh and project a scalar property defined at given
    integration points back to the nodes for triangular elements.

    .. code-block:: python
        :linenos:

        import numpy
        from pysdic import (
            compute_shape_function_matrix,
            assemble_property_projection,
            compute_property_interpolation
        )

        vertices_coordinates = numpy.array(
            [[0.0, 0.0],
             [1.0, 0.0],
             [1.0, 1.0],
             [0.0, 1.0]]
        )

        connectivity = numpy.array(
            [[0, 1, 2],
             [0, 2, 3]]
        )

        N_e = connectivity.shape[0]

        property_array = numpy.array([10.0, 20.0, 30.0,  40.0])  # Scalar property

        # Define natural coordinates for interpolation (e.g., 4 points per element)
        natural_coordinates = numpy.array(
            [[0.3, 0.3],
             [0.2, 0.5],
             [0.5, 0.2],
             [0.1, 0.1]]
        )
        N_p = natural_coordinates.shape[0]
        natural_coordinates = numpy.vstack([natural_coordinates] * N_e)
        element_indices = numpy.repeat(numpy.arange(N_e), N_p)
        N_p = natural_coordinates.shape[0]

        interpolate_property = compute_property_interpolation(
            property_array=property_array,
            connectivity=connectivity,
            element_type='triangle_3',
            natural_coordinates=natural_coordinates,
            element_indices=element_indices
        )

        # Now project back to the nodes
        shape_function_matrix = compute_shape_function_matrix(
            connectivity=connectivity,
            element_type='triangle_3',
            natural_coordinates=natural_coordinates,
            element_indices=element_indices,
            n_vertices=vertices_coordinates.shape[0],
            sparse=False
        )

        projected_props = assemble_property_projection(
            property_array=property_array,
            shape_function_matrix=shape_function_matrix
        )

        print(f"projected properties (shape={projected_props.shape}):")
        print(projected_props)


    .. code-block:: console

        projected properties (shape=(4, 1)):
        [[10.]
         [20.]
         [30.]
         [40.]]

    """
    # Validate input dimensions
    property_array = numpy.asarray(property_array, dtype=numpy.float64)
    if property_array.ndim == 1:
        property_array = property_array[
            :, numpy.newaxis
        ]  # Convert to 2D array with shape (N_p, 1)

    if not (
        isinstance(shape_function_matrix, numpy.ndarray)
        or scipy.sparse.issparse(shape_function_matrix)
    ):
        raise TypeError(
            f"'shape_function_matrix' must be a numpy.ndarray or a scipy.sparse matrix."
            f" Got {type(shape_function_matrix)} instead."
        )
    if isinstance(shape_function_matrix, numpy.ndarray):
        shape_function_matrix = shape_function_matrix.astype(numpy.float64)

    if property_array.ndim != 2:
        raise ValueError(
            f"'property_array' must be a 2D array of shape (N_p, P)."
            f" Got {property_array.ndim}D array instead."
        )
    if shape_function_matrix.ndim != 2:
        raise ValueError(
            f"'shape_function_matrix' must be a 2D array of shape (N_p, N_v)."
            f" Got {shape_function_matrix.ndim}D array instead."
        )
    if not isinstance(return_unaffected, bool):
        raise ValueError(
            f"'return_unaffected' must be a boolean value."
            f" Got {type(return_unaffected)} instead."
        )

    # Extract dimensions
    N_p = shape_function_matrix.shape[0]
    N_v = shape_function_matrix.shape[1]
    P = property_array.shape[1]

    # Handle weights
    if point_weights is None:
        point_weights = 1.0
    else:
        point_weights = numpy.asarray(point_weights, dtype=numpy.float64)
        if point_weights.ndim != 1 or point_weights.shape[0] != property_array.shape[0]:
            raise ValueError(
                f"'point_weights' must be a 1D array of shape (N_p,)."
                f" Got array with shape {point_weights.shape} instead."
            )

    # Initialize projected property
    projected_property = numpy.zeros(
        (N_v, property_array.shape[1]), dtype=numpy.float64
    )

    # Construct weights matrix
    if scipy.sparse.issparse(shape_function_matrix):
        W = scipy.sparse.diags(point_weights, format="csr")  # (N_p, N_p)
        A = shape_function_matrix.T @ W @ shape_function_matrix  # (N_v, N_v)
        b = shape_function_matrix.T @ W @ property_array  # (N_v, P)

        # Extract valid rows to avoid singular matrix issues
        valid_rows = numpy.unique(shape_function_matrix.nonzero()[1])

        reduced_A = A[valid_rows, :][:, valid_rows]  # (N_valid, N_valid)
        reduced_b = b[valid_rows, :]  # (N_valid, P)

        # Solve the normal equations
        reduced_projected_property = scipy.sparse.linalg.spsolve(
            reduced_A, reduced_b
        )  # (N_valid, P)
        if reduced_projected_property.ndim == 1:
            reduced_projected_property = reduced_projected_property[
                :, numpy.newaxis
            ]  # Convert to 2D array with shape (N_valid, 1)
        projected_property[valid_rows, :] = reduced_projected_property

    else:
        W = numpy.diag(point_weights)  # (N_p, N_p)
        A = shape_function_matrix.T @ W @ shape_function_matrix  # (N_v, N_v)
        b = shape_function_matrix.T @ W @ property_array  # (N_v, P)

        # Extract valid rows to avoid singular matrix issues
        valid_rows = numpy.unique(numpy.nonzero(shape_function_matrix)[1])
        A = A[valid_rows, :][:, valid_rows]  # (N_valid, N_valid)
        b = b[valid_rows, :]  # (N_valid, P)

        # Solve the normal equations
        reduced_projected_property = numpy.linalg.solve(A, b)  # (N_valid, P)
        if reduced_projected_property.ndim == 1:
            reduced_projected_property = reduced_projected_property[
                :, numpy.newaxis
            ]  # Convert to 2D array with shape (N_valid, 1)
        projected_property[valid_rows, :] = reduced_projected_property

    # Return unaffected mask if requested
    if return_unaffected:
        unaffected_mask = numpy.ones((N_v,), dtype=bool)
        unaffected_mask[valid_rows] = False
        return projected_property, unaffected_mask

    else:
        return projected_property


def compute_property_projection(
    property_array: ArrayLike,
    connectivity: ArrayLike,
    element_type: str,
    natural_coordinates: ArrayLike,
    element_indices: ArrayLike,
    point_weights: Optional[ArrayLike] = None,
    n_vertices: Optional[Integral] = None,
    *,
    sparse: bool = False,
    skip_m1: bool = True,
    return_unaffected: bool = False,
) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
    r"""
    Project a property defined at integration points within elements back to the nodes
    of a mesh using mesh and integration point information.

    This function combines the remapping of vertex coordinates, computation of shape
    functions, and assembly of the property projection for the entire mesh.

    ``projected_properties[i]`` contains the projected property at the node :math:`i`.

    .. note::

        - Inputs :obj:`property_array`, :obj:`natural_coordinates` and :obj:`point_weights` will be converted to :obj:`numpy.float64`.
        - Inputs :obj:`connectivity` and :obj:`element_indices` will be converted to :obj:`numpy.int64`.

    .. warning::

        No tests are performed to check if the inputs array are consistent
        (e.g., if the connectivity contains invalid vertex indices, ...).
        Only shapes are validated.
        The behavior of the function is undefined in this case.

    .. important::

        When using :obj:`-1` in :obj:`element_indices` for invalid elements,
        ensure to set :obj:`skip_m1` to :obj:`True` to avoid indexing errors.



    Parameters
    ----------
    property_array: ArrayLike
        An array of shape :math:`(N_{p}, P)` containing the property values defined at
        the :math:`N_{p}` integration points. If 1D-array is provided, it will be
        treated as a single-component property of shape :math:`(N_{p}, 1)`.

    connectivity: ArrayLike
        An array of shape :math:`(N_{e}, N_{vpe})` defining the connectivity of the
        elements in the mesh, where each row contains the indices of the nodes that
        form an element.

    element_type: :class:`str`
        A string specifying the type of element (e.g., 'segment_2', 'triangle_3', etc.)
        to determine which shape function to use.

    natural_coordinates: ArrayLike
        An array of shape :math:`(N_{p}, K)` containing the natural coordinates of the
        integration points within elements where :math:`K` is the topological dimension
        of the element (e.g., 1 for segments, 2 for triangles/quadrangles, etc.).

    element_indices: ArrayLike
        An array of shape :math:`(N_{p},)` containing the indices of each element
        corresponding to the :math:`N_{p}` integration points.

    point_weights: Optional[ArrayLike], optional
        An array of shape :math:`(N_{p},)` containing the weights associated with each
        integration point. If not provided, all weights will be assumed to be equal to
        one.

    n_vertices: Optional[Integral], optional
        The total number of vertices :math:`N_{v}` in the mesh. If not provided, it
        will be inferred as the maximum index in :obj:`connectivity` plus one.

    sparse: :class:`bool`, optional
        If set to :obj:`True`, the shape functions matrix will be constructed as a
        sparse matrix to optimize memory usage for large meshes.
        Default is :obj:`False`.

    skip_m1: :class:`bool`, optional
        If set to :obj:`True`, any element index of -1 in :obj:`element_indices` will
        result in the corresponding integration point being ignored during the
        projection.
        Default is :obj:`True`.

    return_unaffected: :class:`bool`, optional
        If set to :obj:`True`, the function will return a tuple containing the projected
        properties and a boolean mask indicating which nodes were unaffected by any
        integration point.
        Default is :obj:`False`.


    Returns
    -------
    projected_properties: :class:`numpy.ndarray`
        An array of shape :math:`(N_{v}, P)` containing the projected property values at
        the nodes of the mesh.

    unaffected_mask: :class:`numpy.ndarray`, optional
        A boolean array of shape :math:`(N_{v},)` indicating which nodes were unaffected
        by any integration point. This is only returned if :obj:`return_unaffected` is
        set to :obj:`True`.
        :obj:`True` indicates the node was unaffected, while :obj:`False` indicates it
        was affected. So using :obj:`projected_properties[unaffected_mask, :] = default`
        will set the unaffected nodes to :obj:`default`.


    Notes
    -----
    See the documentation of :func:`assemble_property_projection` for the mathematical
    formulation and demonstration of the projection process.


    See Also
    --------
    pysdic.compute_shape_functions:
        To compute the shape functions at given natural coordinates within elements.

    pysdic.assemble_property_projection:
        To project properties defined at integration points back to the nodes of the
        mesh using a precomputed shape functions matrix.

    pysdic.compute_property_interpolation:
        To interpolate properties defined at the nodes of the mesh to integration points
        within elements.


    Examples
    --------
    Lets construct a simple 2D mesh and project a scalar property defined at given
    integration points back to the nodes for triangular elements.

    .. code-block:: python
        :linenos:

        import numpy
        from pysdic import (
            compute_property_projection,
            compute_property_interpolation
        )

        vertices_coordinates = numpy.array(
            [[0.0, 0.0],
             [1.0, 0.0],
             [1.0, 1.0],
             [0.0, 1.0]]
        )

        connectivity = numpy.array(
            [[0, 1, 2],
             [0, 2, 3]]
        )

        N_e = connectivity.shape[0]

        property_array = numpy.array([10.0, 20.0, 30.0,  40.0])  # Scalar property

        # Define natural coordinates for interpolation (e.g., 4 points per element)
        natural_coordinates = numpy.array(
            [[0.3, 0.3],
             [0.2, 0.5],
             [0.5, 0.2],
             [0.1, 0.1]]
        )
        N_p = natural_coordinates.shape[0]
        natural_coordinates = numpy.vstack([natural_coordinates] * N_e)
        element_indices = numpy.repeat(numpy.arange(N_e), N_p)
        N_p = natural_coordinates.shape[0]

        interpolate_property = compute_property_interpolation(
            property_array=property_array,
            connectivity=connectivity,
            element_type='triangle_3',
            natural_coordinates=natural_coordinates,
            element_indices=element_indices
        )

        # Now project back to the nodes
        projected_props = compute_property_projection(
            property_array=interpolate_property,
            connectivity=connectivity,
            element_type='triangle_3',
            natural_coordinates=natural_coordinates,
            element_indices=element_indices,
            sparse=False
        )

        print(f"projected properties (shape={projected_props.shape}):")
        print(projected_props)


    .. code-block:: console

        projected properties (shape=(4, 1)):
        [[10.]
         [20.]
         [30.]
         [40.]]

    """
    # Validate input dimensions
    property_array = numpy.asarray(property_array, dtype=numpy.float64)
    if property_array.ndim == 1:
        property_array = property_array[
            :, numpy.newaxis
        ]  # Convert to 2D array with shape (N_p, 1)

    natural_coordinates = numpy.asarray(natural_coordinates, dtype=numpy.float64)
    connectivity = numpy.asarray(connectivity, dtype=numpy.int64)
    element_indices = numpy.asarray(element_indices, dtype=numpy.int64)

    if property_array.ndim != 2:
        raise ValueError(
            f"'property_array' must be a 2D array of shape (N_p, P)."
            f" Got {property_array.ndim}D array instead."
        )
    if natural_coordinates.ndim != 2:
        raise ValueError(
            f"'natural_coordinates' must be a 2D array of shape (N_p, K)."
            f" Got {natural_coordinates.ndim}D array instead."
        )
    if connectivity.ndim != 2:
        raise ValueError(
            f"'connectivity' must be a 2D array of shape (N_e, N_vpe)."
            f" Got {connectivity.ndim}D array instead."
        )
    if element_indices.ndim != 1:
        raise ValueError(
            f"'element_indices' must be a 1D array of shape (N_p,)."
            f" Got {element_indices.ndim}D array instead."
        )
    if not natural_coordinates.shape[0] == element_indices.shape[0]:
        raise ValueError(
            f"The first dimension (N_p) of 'natural_coordinates' and 'element_indices'"
            f" must match. Got {natural_coordinates.shape[0]} and"
            f" {element_indices.shape[0]} respectively."
        )
    if not isinstance(element_type, str):
        raise TypeError(
            f"'element_type' must be a string. Got {type(element_type)} instead."
        )
    if not isinstance(sparse, bool):
        raise TypeError(
            f"'sparse' must be a boolean value. Got {type(sparse)} instead."
        )
    if not isinstance(skip_m1, bool):
        raise TypeError(
            f"'skip_m1' must be a boolean value. Got {type(skip_m1)} instead."
        )
    if not isinstance(return_unaffected, bool):
        raise ValueError(
            f"'return_unaffected' must be a boolean value."
            f" Got {type(return_unaffected)} instead."
        )

    # Extract number of vertices
    if n_vertices is None:
        n_vertices = numpy.max(connectivity) + 1

    if not isinstance(n_vertices, Integral) or n_vertices <= 0:
        raise ValueError(
            f"'n_vertices' must be a positive integer. Got {n_vertices} instead."
        )

    # Extract weights
    if point_weights is None:
        point_weights = numpy.ones(
            shape=(property_array.shape[0],), dtype=numpy.float64
        )
    else:
        point_weights = numpy.asarray(point_weights, dtype=numpy.float64)
        if point_weights.ndim != 1 or point_weights.shape[0] != property_array.shape[0]:
            raise ValueError(
                f"'point_weights' must be a 1D array of shape (N_p,)."
                f" Got array with shape {point_weights.shape} instead."
            )

    # Compute shape functions matrix
    shape_function_matrix = compute_shape_function_matrix(
        connectivity=connectivity,
        element_type=element_type,
        natural_coordinates=natural_coordinates,
        element_indices=element_indices,
        n_vertices=n_vertices,
        sparse=sparse,
        skip_m1=skip_m1,
    )

    # Assemble property projection
    return assemble_property_projection(
        property_array=property_array,
        shape_function_matrix=shape_function_matrix,
        point_weights=point_weights,
        return_unaffected=return_unaffected,
    )
