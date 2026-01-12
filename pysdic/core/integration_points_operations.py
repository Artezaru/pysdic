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

import numpy
import scipy



def assemble_shape_function_matrix(
    shape_functions: numpy.ndarray,
    element_connectivity: numpy.ndarray,
    element_indices: numpy.ndarray,
    n_vertices: Optional[int] = None,
    *,
    sparse: bool = False,
    skip_m1: bool = True,
    default: Real = 0.0
) -> Union[numpy.ndarray, scipy.sparse.csr_matrix]:
    r"""
    Construct the matrix of shape functions at given integration points within elements, from :math:`(N_{p}, N_{vpe})` to :math:`(N_{p}, N_{v})` with zero-filling for non-associated nodes/vertices.

    .. note::

        The input :obj:`shape_functions` will be converted to :obj:`numpy.float64` for computation.
        The inputs :obj:`element_connectivity`, and :obj:`element_indices` will be converted to :obj:`numpy.int64` for computation.

    .. warning::

        When using :obj:`-1` in :obj:`element_indices` for invalid elements, ensure to set :obj:`skip_m1` to :obj:`True` to avoid indexing errors as :obj:`-1` is equivalent to the last element in Python indexing.

        
    Parameters
    ----------
    shape_functions: :class:`numpy.ndarray`
        An array of shape :math:`(N_{p}, N_{vpe})` containing the shape function values evaluated at :math:`N_{p}` points for the :math:`N_{vpe}` nodes of the element.

    element_connectivity: :class:`numpy.ndarray`
        An array of shape :math:`(N_{e}, N_{vpe})` defining the connectivity of the elements in the mesh, where each row contains the indices of the nodes that form an element.

    element_indices: :class:`numpy.ndarray`
        An array of shape :math:`(N_{p},)` containing the indices of each element corresponding to the :math:`N_{p}` integration points.

    n_vertices: :class:`int`, optional
        The total number of vertices :math:`N_{v}` in the mesh.
        If not provided, it will be inferred as the maximum node index in :obj:`element_connectivity` plus one. 

    sparse: :class:`bool`, optional
        If set to :obj:`True`, the function will use :obj:`scipy.sparse` to create a sparse matrix representation of the shape function matrix.
        Default is :obj:`False`.

    skip_m1: :class:`bool`, optional
        If set to :obj:`True`, any element index of -1 in :obj:`element_indices` will result in the corresponding shape function values being set to :obj:`default`.
        Default is :obj:`True`.

    default: :class:`Real`, optional
        The default value to assign to shape function values for integration points associated with an element index of -1 when :obj:`skip_m1` is :obj:`True`.
        Default is :obj:`0.0`. Only used for non-sparse matrix construction. For sparse matrices, zero-filling is used.

        
    Returns 
    -------
    shape_function_matrix: Union[numpy.ndarray, scipy.sparse.csr_matrix]
        An array of shape :math:`(N_{p}, N_{v})` or sparse matrix containing the shape function values at each of the :math:`N_{p}` integration points for all :math:`N_{v}` nodes in the mesh.

    
    Raises
    ------
    TypeError
        - If :obj:`shape_functions` cannot be converted to a :obj:`numpy.ndarray` of floating type.
        - If :obj:`element_connectivity` or :obj:`element_indices` cannot be converted to a :obj:`numpy.ndarray` of integer type.
        - If :obj:`n_vertices` is provided and is not an integer.
        - If :obj:`sparse` is not a boolean.
        - If :obj:`skip_m1` is not a boolean.
        - If :obj:`default` is not a real number.

    ValueError
        - If :obj:`shape_functions` is not a 2D array of shape :math:`(N_{p}, N_{vpe})`.
        - If :obj:`element_connectivity` is not a 2D array of shape :math:`(N_{e}, N_{vpe})`.
        - If :obj:`element_indices` is not a 1D array of shape :math:`(N_{p},)`.


    Notes
    -----
    The computation of the shape function values are returned in a matrix with dimensions :math:`(N_{p}, N_{vpe})`.
    This method constructs a larger matrix of dimensions :math:`(N_{p}, N_{v})` where each row corresponds to an integration point and each column to a node in the mesh.
    The values from the input :obj:`shape_functions` are placed in the appropriate columns based on the :obj:`element_connectivity` and :obj:`element_indices`.

    .. code-block:: python

        point_index = 0  # Index of the integration point
        element_index = element_indices[point_index]  # Element index for the integration point
        vertices_indices = element_connectivity[element_index, :]  # Node indices for the element of the integration point
        shape_function_matrix[point_index, vertices_indices] = shape_functions[point_index, :] # Assigning shape function values

        
    See Also
    --------
    pysdic.shape_functions:
        To compute the shape functions at given natural coordinates within elements with shape :math:`(N_{p}, N_{vpe})`.


    Examples
    --------
    Lets construct a simple 2D mesh and build the shape function matrix at given integration points for triangular elements.

    .. code-block:: python

        import numpy
        from pysdic import assemble_shape_function_matrix
        from pysdic import triangle_3_shape_functions

        vertices_coordinates = numpy.array([[0.0, 0.0],
                                            [1.0, 0.0], 
                                            [1.0, 1.0],
                                            [0.0, 1.0]])

        element_connectivity = numpy.array([[0, 1, 2],
                                            [0, 2, 3]])

        natural_coordinates = numpy.array([[0.2, 0.3],
                                           [0.6, 0.2]])

        shape_functions = triangle_3_shape_functions(natural_coordinates)
        element_indices = numpy.array([0, 1])

        shape_func_matrix = assemble_shape_function_matrix(shape_functions,
                                                           element_connectivity,
                                                           element_indices)

    
        print(shape_func_matrix)

    The output will be:

    .. code-block:: console

        [[0.5 0.2 0.3 0. ]
         [0.2 0.  0.6 0.2]]

    """
    # Validate input dimensions
    shape_functions = numpy.asarray(shape_functions, dtype=numpy.float64)
    element_connectivity = numpy.asarray(element_connectivity, dtype=numpy.int64)
    element_indices = numpy.asarray(element_indices, dtype=numpy.int64)

    if shape_functions.ndim != 2:
        raise ValueError(f"'shape_functions' must be a 2D array of shape (N_p, N_npe). Got {shape_functions.ndim}D array instead.")
    if element_connectivity.ndim != 2:
        raise ValueError(f"'element_connectivity' must be a 2D array of shape (N_e, N_npe). Got {element_connectivity.ndim}D array instead.")
    if element_indices.ndim != 1:
        raise ValueError(f"'element_indices' must be a 1D array of shape (N_p,). Got {element_indices.ndim}D array instead.")
    if not shape_functions.shape[0] == element_indices.shape[0]:
        raise ValueError(f"The first dimension (N_p) of 'shape_functions' and 'element_indices' must match. Got {shape_functions.shape[0]} and {element_indices.shape[0]} respectively.")
    if not shape_functions.shape[1] == element_connectivity.shape[1]: 
        raise ValueError(f"The second dimension (N_npe) of 'shape_functions' and 'element_connectivity' must match. Got {shape_functions.shape[1]} and {element_connectivity.shape[1]} respectively.")
    if not isinstance(sparse, bool):
        raise ValueError(f"'sparse' must be a boolean value. Got {type(sparse)} instead.")
    if not isinstance(skip_m1, bool):
        raise ValueError(f"'skip_m1' must be a boolean value. Got {type(skip_m1)} instead.")
    if not isinstance(default, Real):
        raise ValueError(f"'default' must be a real number. Got {type(default)} instead.")
    
    # Extract number of vertices
    if n_vertices is None:
        n_vertices = numpy.max(element_connectivity) + 1
    
    if not isinstance(n_vertices, Integral) or n_vertices <= 0:
        raise ValueError(f"n_vertices must be a positive integer. Got {n_vertices} instead.")
    
    # Extract dimensions
    N_p = shape_functions.shape[0]
    N_npe = shape_functions.shape[1]
    N_v = n_vertices

    # Handle skip_m1 option
    if skip_m1:
        m1_mask = element_indices == -1
        numpy.logical_not(m1_mask, out=m1_mask)
        valid_indices = element_indices[m1_mask]
    else:
        m1_mask = numpy.ones(N_p, dtype=bool)
        valid_indices = element_indices

    # Extract the vertices indices for each point
    vertex_indices = element_connectivity[valid_indices, :]  # (N_valid, N_npe)

    # Build the numpy arrays
    if not sparse:
        shape_functions_matrix = numpy.zeros((N_p, N_v), dtype=numpy.float64)
        # Indexation vectorisée
        shape_functions_matrix[numpy.arange(N_p)[m1_mask, None], vertex_indices] = shape_functions[m1_mask, :] if skip_m1 else shape_functions

        # Handle skip_m1 option
        if skip_m1:
            shape_functions_matrix[~m1_mask, :] = default

    # Build the scipy sparse matrices
    else:
        row_idx = numpy.repeat(numpy.arange(N_p), N_npe)
        col_idx = vertex_indices.ravel()
        data = shape_functions[m1_mask, :].ravel() if skip_m1 else shape_functions.ravel()
        shape_functions_matrix = scipy.sparse.csr_matrix((data, (row_idx, col_idx)), shape=(N_p, N_v))

    return shape_functions_matrix



def construct_jacobian(
    shape_function_derivatives: numpy.ndarray,
    remapped_coordinates: numpy.ndarray,
) -> numpy.ndarray:
    r"""
    Construct the Jacobian matrix :math:`J` of the transformation from natural coordinates :math:`(\xi, \eta, \zeta, ...)` to
    global coordinates :math:`(x, y, z, ...)`.

    .. note::

        The inputs :obj:`shape_function_derivatives` and :obj:`remapped_coordinates` will be converted to :obj:`numpy.float64` for computation.

        
    Parameters
    ----------
    shape_function_derivatives: :class:`numpy.ndarray`
        An array of shape :math:`(N_{p}, N_{vpe}, K)` containing the derivatives of the shape functions with respect to the local coordinates,
        evaluated at :math:`N_{p}` points for the :math:`N_{vpe}` nodes of the element.

    remapped_coordinates : :class:`numpy.ndarray`
        An array of shape :math:`(N_{p}, N_{vpe}, E)` containing the remapped global coordinates of the vertices for each integration point within the elements


    Returns
    -------
    jacobians: :class:`numpy.ndarray`
        An array of shape :math:`(N_{p}, E, K)` containing the Jacobian matrices for each of the :math:`N_{p}` points. Each Jacobian matrix has dimensions :math:`(E \times K)`.

    
    Raises
    ------
    TypeError
        - If :obj:`shape_function_derivatives` cannot be converted to a :obj:`numpy.ndarray` of floating type.
        - If :obj:`remapped_coordinates` cannot be converted to a :obj:`numpy.ndarray` of floating type.

    ValueError
        - If :obj:`shape_function_derivatives` is not a 3D array of shape :math:`(N_{p}, N_{vpe}, K)`.
        - If :obj:`remapped_coordinates` is not a 3D array of shape :math:`(N_{p}, N_{vpe}, E)`.

    
    Notes
    -----
    In a space of dimension :math:`E` with a mesh constituted of :math:`N_{e}` elements and :math:`N_{v}` nodes,
    The mesh is composed of :math:`K`-dimensional elements (with :math:`K \leq E`) defined by :math:`N_{vpe}` nodes for each element.
     
    The Jacobian matrix has dimensions :math:`(E \times K)` and constructed as follows:

    .. math::

        J = \begin{bmatrix}
        \frac{\partial x}{\partial \xi} & \frac{\partial x}{\partial \eta} & \frac{\partial x}{\partial \zeta} & \ldots \\
        \frac{\partial y}{\partial \xi} & \frac{\partial y}{\partial \eta} & \frac{\partial y}{\partial \zeta} & \ldots \\
        \frac{\partial z}{\partial \xi} & \frac{\partial z}{\partial \eta} & \frac{\partial z}{\partial \zeta} & \ldots \\
        \vdots & \vdots & \vdots & \ddots
        \end{bmatrix}

    where each entry is computed as:

    .. math::

        \frac{\partial x_i}{\partial \xi_j} = \sum_{a=1}^{N_{vpe}} \frac{\partial N_a}{\partial \xi_j} x_{i,a}

    Here, :math:`N_a` are the shape functions associated with each node, and :math:`x_{i,a}` are the global coordinates of the nodes.


    See Also
    --------
    pysdic.shape_functions:
        To compute the shape functions and their derivatives at given natural coordinates within elements.
    
    pysdic.remap_vertices_coordinates:
        To remap the vertices coordinates for each element at given integration points.


    Examples
    --------
    Lets construct a simple 2D mesh and compute the Jacobian at given integration points for triangular elements.

    .. code-block:: python

        import numpy
        from pysdic import (remap_vertices_coordinates, construct_jacobian)
        from pysdic import triangle_3_shape_functions

        vertices_coordinates = numpy.array([[0.0, 0.0],
                                            [1.0, 0.0], 
                                            [1.0, 1.0],
                                            [0.0, 1.0]])

        element_connectivity = numpy.array([[0, 1, 2],
                                            [0, 2, 3]])

        element_indices = numpy.array([0, 1])

        remapped_coords = remap_vertices_coordinates(vertices_coordinates,
                                                     element_connectivity,
                                                     element_indices)

        natural_coordinates = numpy.array([[0.2, 0.3],
                                           [0.6, 0.2]])

        _, shape_func_derivs = triangle_3_shape_functions(natural_coordinates, return_derivatives=True)
        jacobians = construct_jacobian(shape_func_derivs, remapped_coords)
        print(jacobians)

    The output will be:

    .. code-block:: console

        [[[1. 0.]
          [1. 1.]]]

         [[1. 1.]
          [0. 1.]]]

    """
    # Validate input dimensions
    shape_function_derivatives = numpy.asarray(shape_function_derivatives, dtype=numpy.float64)
    remapped_coordinates = numpy.asarray(remapped_coordinates, dtype=numpy.float64)

    if shape_function_derivatives.ndim != 3:
        raise ValueError(f"'shape_function_derivatives' must be a 3D array of shape (N_p, N_npe, K). Got {shape_function_derivatives.ndim}D array instead.")
    if remapped_coordinates.ndim != 3:
        raise ValueError(f"'remapped_coordinates' must be a 3D array of shape (N_p, N_npe, E). Got {remapped_coordinates.ndim}D array instead.")
    if not shape_function_derivatives.shape[0] == remapped_coordinates.shape[0]:
        raise ValueError(f"The first dimension (N_p) of 'shape_function_derivatives' and 'remapped_coordinates' must match. Got {shape_function_derivatives.shape[0]} and {remapped_coordinates.shape[0]} respectively.")
    if not shape_function_derivatives.shape[1] == remapped_coordinates.shape[1]:
        raise ValueError(f"The second dimension (N_npe) of 'shape_function_derivatives' and 'remapped_coordinates' must match. Got {shape_function_derivatives.shape[1]} and {remapped_coordinates.shape[1]} respectively.")

    # Extract dimensions
    N_p = shape_function_derivatives.shape[0]
    N_npe = shape_function_derivatives.shape[1]
    K = shape_function_derivatives.shape[2]
    E = remapped_coordinates.shape[2]

    # Compute Jacobians
    jacobians = numpy.einsum('mjk,mja->mka', shape_function_derivatives, remapped_coordinates)
    return jacobians



def derivate_property(
    property_array: numpy.ndarray,
    vertices_coordinates: numpy.ndarray,
    shape_function_derivatives: numpy.ndarray,
    element_connectivity: numpy.ndarray,
    element_indices: numpy.ndarray,
    *,
    skip_m1: bool = True,
    default: Real = numpy.nan,
) -> numpy.ndarray:
    r"""
    Derivate a property defined at the nodes of a mesh to given integration points within elements with respect to global coordinates :math:`(x,y,z,...)` using shape function derivatives.

    .. note::

        The inputs :obj:`property_array`, :obj:`vertices_coordinates`, and :obj:`shape_function_derivatives` will be converted to :obj:`numpy.float64` for computation.
        The inputs :obj:`element_connectivity`, and :obj:`element_indices` will be converted to :obj:`numpy.int64` for computation.


    Parameters
    ----------
    property_array: :class:`numpy.ndarray`
        An array of shape :math:`(N_{v}, P)` containing the property values defined at the nodes of the mesh. If 1D-array is provided, it will be treated as a single-component property of shape :math:`(N_{v}, 1)`.

    vertices_coordinates: :class:`numpy.ndarray`
        An array of shape :math:`(N_{v}, E)` containing the global coordinates :math:`(x, y, z, ...)` of the vertices in the mesh.

    shape_function_derivatives: :class:`numpy.ndarray`
        An array of shape :math:`(N_{p}, N_{vpe}, K)` containing the derivatives of the shape functions evaluated at :math:`N_{p}` points for the :math:`N_{vpe}` nodes of the element.

    element_connectivity: :class:`numpy.ndarray`
        An array of shape :math:`(N_{e}, N_{vpe})` defining the connectivity of the elements in the mesh, where each row contains the indices of the nodes that form an element.

    element_indices: :class:`numpy.ndarray`
        An array of shape :math:`(N_{p},)` containing the indices of each element corresponding to the :math:`N_{p}` integration points.

    skip_m1: :class:`bool`, optional
        If set to :obj:`True`, any element index of -1 in :obj:`element_indices` will result in the corresponding derivated property being set to :obj:`default`.
        Default is :obj:`True`.

    default: :class:`Real`, optional
        The default value to assign to derivated properties for integration points associated with an element index of -1 when :obj:`skip_m1` is :obj:`True`.
        Default is :obj:`numpy.nan`.

    
    Returns
    -------
    derivated_properties: :class:`numpy.ndarray`
        An array of shape :math:`(N_{p}, P, E)` containing the derivated property values at each of the :math:`N_{p}` integration points with respect to the global coordinates :math:`(x, y, z, ...)`.

        
    Raises
    ------
    TypeError
        - If :obj:`property_array`, :obj:`vertices_coordinates`, or :obj:`shape_function_derivatives` cannot be converted to a :obj:`numpy.ndarray` of floating type.
        - If :obj:`element_connectivity` or :obj:`element_indices` cannot be converted to a :obj:`numpy.ndarray` of integer type.
        - If :obj:`skip_m1` is not a boolean.
        - If :obj:`default` is not a real number.

    ValueError
        - If :obj:`property_array` is not a 1D or 2D array of shape :math:`(N_{v},)` or :math:`(N_{v}, P)`.
        - If :obj:`vertices_coordinates` is not a 2D array of shape :math:`(N_{v}, E)`.
        - If :obj:`shape_function_derivatives` is not a 3D array of shape :math:`(N_{p}, N_{vpe}, K)`.
        - If :obj:`element_connectivity` is not a 2D array of shape :math:`(N_{e}, N_{vpe})`.
        - If :obj:`element_indices` is not a 1D array of shape :math:`(N_{p},)`.


    Notes
    -----
    In a space of dimension :math:`E` with a mesh constituted of :math:`N_{e}` elements and :math:`N_{v}` nodes,
    The mesh is composed of :math:`K`-dimensional elements (with :math:`K \leq E`) defined by :math:`N_{vpe}` nodes for each element.

    For a given set of :math:`N_p` integration points located within elements, the derivated property array has shape :math:`(N_p, P, K)`
    where :math:`P` is the number of property components (e.g., 1 for scalar properties, 3 for vector properties), and :math:`K` is the number of local coordinates.

    The derivative of the property at each integration point is computed as:

    .. math::

        \nabla_X P(\xi, \eta, \zeta, ...) = J \cdot (J^T J)^{-1} \cdot \sum_{i=1}^{N_{vpe}} \nabla_{\xi} N_i(\xi, \eta, \zeta, ...) P_i

    where :math:`N_i` are the shape functions associated with each node, :math:`P_i` are the property values at the nodes of the element, and :math:`J` is the Jacobian matrix of the transformation from natural to global coordinates.

    If the dimension of the elements :math:`K` is equal to the space dimension :math:`E`, the formula simplifies to:

    .. math::

        \nabla_X P(\xi, \eta, \zeta, ...) = J^{-1} \cdot \sum_{i=1}^{N_{vpe}} \nabla_{\xi} N_i(\xi, \eta, \zeta, ...) P_i


    Demonstration
    --------------

    The property field is defined at the nodes of the mesh and can be interpolate to any point within the elements using shape functions.

    .. math::

        P(\xi, \eta, \zeta, ...) = \sum_{i=1}^{N_{vpe}} N_i(\xi, \eta, \zeta, ...) P_i

    The spatial derivative of the property field with respect to global coordinates is defined as:

    .. math::

        \nabla_X P(\xi, \eta, \zeta, ...) = \sum_{i=1}^{N_{vpe}} \nabla_{X} N_i(\xi, \eta, \zeta, ...) P_i

    The derivative of the shape functions with respect to global coordinates is obtained via the chain rule:

    .. math::

        \frac{\partial N_i}{\partial \xi} = J^T \cdot \nabla_X N_i(\xi, \eta, \zeta, ...)

    where :math:`J = \frac{\partial X}{\partial \xi}` is the Jacobian matrix of the transformation from natural to global coordinates.

    .. math::

        J = \sum_{a=1}^{N_{vpe}} \nabla_{\xi} N_a(\xi, \eta, \zeta, ...) X_a

    The system can be inverted to express the derivative with respect to global coordinates if the element is not degenerate:

    .. math::

        \nabla_X N_i(\xi, \eta, \zeta, ...) = J \cdot (J^T J)^{-1} \cdot \frac{\partial N_i}{\partial \xi}

    Thus, the derivative of the property field with respect to global coordinates becomes:

    .. math::

        \nabla_X P(\xi, \eta, \zeta, ...) = J \cdot (J^T J)^{-1} \cdot \sum_{i=1}^{N_{vpe}} \frac{\partial N_i}{\partial \xi} P_i

        
    See Also
    --------
    pysdic.shape_functions:
        To compute the shape functions and their derivatives at given natural coordinates within elements.

    pysdic.construct_jacobian:
        To construct the Jacobian matrix using remapped coordinates.

    pysdic.interpolate_property:
        To interpolate properties at integration points.
    

    Examples
    --------
    Lets construct a simple 2D mesh and derivate a scalar property at given integration points for triangular elements.

    .. code-block:: python

        import numpy
        from pysdic import derivate_property
        from pysdic import triangle_3_shape_functions

        vertices_coordinates = numpy.array([[0.0, 0.0],
                                            [1.0, 0.0], 
                                            [1.0, 1.0],
                                            [0.0, 1.0]])

        element_connectivity = numpy.array([[0, 1, 2],
                                            [0, 2, 3]])

        property_array = numpy.array([10.0,
                                    20.0,
                                    30.0,
                                    40.0])  # Scalar property at each node

        natural_coordinates = numpy.array([[0.2, 0.3],
                                           [0.6, 0.2]])

        _, shape_func_derivs = triangle_3_shape_functions(natural_coordinates, return_derivatives=True)

        element_indices = numpy.array([0, 1])

        derivated_props = derivate_property(property_array,
                                            vertices_coordinates,
                                            shape_func_derivs,
                                            element_connectivity,
                                            element_indices)

        print(derivated_props)

    
    The output will be:

    .. code-block:: console

        [[[-10.  20.]]
         [[ 20.  10.]]]

    """
    # Validate input dimensions
    property_array = numpy.asarray(property_array, dtype=numpy.float64)
    if property_array.ndim == 1:
        property_array = property_array[:, numpy.newaxis] # Convert to 2D array with shape (N_v, 1)

    vertices_coordinates = numpy.asarray(vertices_coordinates, dtype=numpy.float64)
    shape_function_derivatives = numpy.asarray(shape_function_derivatives, dtype=numpy.float64)
    element_connectivity = numpy.asarray(element_connectivity, dtype=numpy.int64)
    element_indices = numpy.asarray(element_indices, dtype=numpy.int64)

    if property_array.ndim != 2:
        raise ValueError(f"'property_array' must be a 2D array of shape (N_v, P). Got {property_array.ndim}D array instead.")
    if vertices_coordinates.ndim != 2:
        raise ValueError(f"'vertices_coordinates' must be a 2D array of shape (N_v, E). Got {vertices_coordinates.ndim}D array instead.")
    if shape_function_derivatives.ndim != 3:
        raise ValueError(f"'shape_function_derivatives' must be a 3D array of shape (N_p, N_npe, K). Got {shape_function_derivatives.ndim}D array instead.")
    if element_connectivity.ndim != 2:
        raise ValueError(f"'element_connectivity' must be a 2D array of shape (N_e, N_npe). Got {element_connectivity.ndim}D array instead.")
    if element_indices.ndim != 1:
        raise ValueError(f"'element_indices' must be a 1D array of shape (N_p,). Got {element_indices.ndim}D array instead.")
    if not shape_function_derivatives.shape[0] == element_indices.shape[0]:
        raise ValueError(f"The first dimension (N_p) of 'shape_function_derivatives' and 'element_indices' must match. Got {shape_function_derivatives.shape[0]} and {element_indices.shape[0]} respectively.")
    if not shape_function_derivatives.shape[1] == element_connectivity.shape[1]:
        raise ValueError(f"The second dimension (N_npe) of 'shape_function_derivatives' and 'element_connectivity' must match. Got {shape_function_derivatives.shape[1]} and {element_connectivity.shape[1]} respectively.")
    if not vertices_coordinates.shape[0] == property_array.shape[0]:
        raise ValueError(f"The first dimension (N_v) of 'vertices_coordinates' and 'property_array' must match. Got {vertices_coordinates.shape[0]} and {property_array.shape[0]} respectively.")
    if not isinstance(skip_m1, bool):
        raise ValueError(f"'skip_m1' must be a boolean value. Got {type(skip_m1)} instead.")
    if not isinstance(default, Real):
        raise ValueError(f"'default' must be a real number. Got {type(default)} instead.")
    
    # Extract dimensions
    N_p = shape_function_derivatives.shape[0]
    N_npe = shape_function_derivatives.shape[1]
    K = shape_function_derivatives.shape[2]
    P = property_array.shape[1]
    E = vertices_coordinates.shape[1]
    N_v = property_array.shape[0]
    
    # Extract -1 mask if skip_m1 is True
    if skip_m1:
        m1_mask = element_indices == -1
        numpy.logical_not(m1_mask, out=m1_mask)
        valid_indices = element_indices[m1_mask]
    else:
        valid_indices = element_indices

    # Compute the jacobians for valid indices
    remapped_coordinates = remap_vertices_coordinates(
        vertices_coordinates=vertices_coordinates,
        element_connectivity=element_connectivity,
        element_indices=valid_indices,
    )  # Shape: (N_valid, N_npe, E)

    jacobians = construct_jacobian(
        shape_function_derivatives=shape_function_derivatives[valid_indices, :, :],
        remapped_coordinates=remapped_coordinates,
    )  # Shape: (N_valid, E, K)
 
    # Invert jacobians
    if E == K:
        jacobians_inv = numpy.linalg.inv(jacobians)  # Shape: (N_p, K, E)
    else:
        jacobians_inv = numpy.einsum('mij,mjk->mik', jacobians, numpy.linalg.inv(numpy.einsum('mji,mjk->mik', jacobians, jacobians))) # Shape: (N_p, K, E)

    # Derivate properties
    vertices_properties = property_array[element_connectivity[valid_indices, :], :]  # Shape: (N_p, N_npe, P)
    derivative_sum = numpy.einsum('mjp,mjk->mpk', vertices_properties, shape_function_derivatives[m1_mask, :, :] if skip_m1 else shape_function_derivatives)  # Shape: (N_p, P, K)
    derivated_properties_valid = numpy.einsum('mke,mpk->mpe', jacobians_inv, derivative_sum)  # Shape: (N_p, P, E)

    # Handle -1 entries if skip_m1 is True
    if skip_m1:
        derivated_properties = numpy.full((N_p, P, E), default, dtype=derivated_properties_valid.dtype)
        derivated_properties[m1_mask, :, :] = derivated_properties_valid
    else:
        derivated_properties = derivated_properties_valid

    return derivated_properties



def interpolate_property(
    property_array: numpy.ndarray,
    shape_functions: numpy.ndarray,
    element_connectivity: numpy.ndarray,
    element_indices: numpy.ndarray,
    *,
    skip_m1: bool = True,
    default: Union[Real, numpy.ndarray] = numpy.nan,
) -> numpy.ndarray:
    r"""
    Interpolate a property defined at the nodes of a mesh to given integration points within elements using shape functions.

    .. note::

        The inputs :obj:`property_array` and :obj:`shape_functions` will be converted to :obj:`numpy.float64` for computation.
        The inputs :obj:`element_connectivity`, and :obj:`element_indices` will be converted to :obj:`numpy.int64` for computation.

    .. warning::

        When using :obj:`-1` in :obj:`element_indices` for invalid elements, ensure to set :obj:`skip_m1` to :obj:`True` to avoid indexing errors as :obj:`-1` is equivalent to the last element in Python indexing.

        
    Parameters
    ----------
    property_array: :class:`numpy.ndarray`
        An array of shape :math:`(N_{v}, P)` containing the property values defined at the nodes of the mesh. If 1D-array is provided, it will be treated as a single-component property of shape :math:`(N_{v}, 1)`.

    shape_functions: :class:`numpy.ndarray`
        An array of shape :math:`(N_{p}, N_{vpe})` containing the shape function values evaluated at :math:`N_{p}` points for the :math:`N_{vpe}` nodes of the element.

    element_connectivity: :class:`numpy.ndarray`
        An array of shape :math:`(N_{e}, N_{vpe})` defining the connectivity of the elements in the mesh, where each row contains the indices of the nodes that form an element.

    element_indices: :class:`numpy.ndarray`
        An array of shape :math:`(N_{p},)` containing the indices of each element corresponding to the :math:`N_{p}` integration points.

    skip_m1: :class:`bool`, optional
        If set to :obj:`True`, any element index of -1 in :obj:`element_indices` will result in the corresponding interpolated property being set to :obj:`default`.
        Default is :obj:`True`.

    default: :class:`Real` or :class:`numpy.ndarray`, optional
        The default value to assign to interpolated properties for integration points associated with an element index of -1 when :obj:`skip_m1` is :obj:`True`.
        Default is :obj:`numpy.nan`. The input can also be a :class:`numpy.ndarray` of shape (P,) to assign different default values for each property component.


    Returns
    -------
    interpolated_properties: :class:`numpy.ndarray`
        An array of shape :math:`(N_{p}, P)` containing the interpolated property values at each of the :math:`N_{p}` integration points.

    
    Raises
    ------
    TypeError
        - If :obj:`property_array` or :obj:`shape_functions` cannot be converted to a :obj:`numpy.ndarray` of floating type.
        - If :obj:`element_connectivity` or :obj:`element_indices` cannot be converted to a :obj:`numpy.ndarray` of integer type.
        - If :obj:`skip_m1` is not a boolean.
        - If :obj:`default` is neither a real number nor a :obj:`numpy.ndarray` of shape (P,).

    ValueError
        - If :obj:`property_array` is not a 1D or 2D array of shape :math:`(N_{v},)` or :math:`(N_{v}, P)`.
        - If :obj:`shape_functions` is not a 2D array of shape :math:`(N_{p}, N_{vpe})`.
        - If :obj:`element_connectivity` is not a 2D array of shape :math:`(N_{e}, N_{vpe})`.
        - If :obj:`element_indices` is not a 1D array of shape :math:`(N_{p},)`.

        
    Notes
    -----
    In a space of dimension :math:`E` with a mesh constituted of :math:`N_{e}` elements and :math:`N_{v}` nodes,
    The mesh is composed of :math:`K`-dimensional elements (with :math:`K \leq E`) defined by :math:`N_{vpe}` nodes for each element.

    For a given set of :math:`N_{p}` integration points located within elements, the interpolated property array has shape :math:`(N_{p}, P)`
    where :math:`P` is the number of property components (e.g., 1 for scalar properties, 3 for vector properties).

    The property at each integration point is computed as:

    .. math::

        P(\xi, \eta, \zeta, ...) = \sum_{i=1}^{N_{vpe}} N_i(\xi, \eta, \zeta, ...) P_i

    where :math:`N_i` are the shape functions associated with each node, and :math:`P_i` are the property values at the nodes of the element.


    See Also
    --------
    pysdic.shape_functions:
        To compute the shape functions at given natural coordinates within elements.

    pysdic.project_property_to_vertices:
        To project properties defined at integration points back to the nodes of the mesh.

    
    Examples
    --------
    Lets construct a simple 2D mesh and interpolate a scalar property at given integration points for triangular elements.

    .. code-block:: python

        import numpy
        from pysdic import interpolate_property
        from pysdic import triangle_3_shape_functions

        vertices_coordinates = numpy.array([[0.0, 0.0],
                                            [1.0, 0.0], 
                                            [1.0, 1.0],
                                            [0.0, 1.0]])

        element_connectivity = numpy.array([[0, 1, 2],
                                            [0, 2, 3]])

        property_array = numpy.array([10.0,
                                      20.0,
                                      30.0,
                                      40.0])  # Scalar property at each node

        natural_coordinates = numpy.array([[0.2, 0.3],
                                           [0.6, 0.2]])

        shape_functions, _ = triangle_3_shape_functions(natural_coordinates, return_derivatives=True)
        element_indices = numpy.array([0, 1])

        interpolated_props = interpolate_property(property_array,
                                                 shape_functions,
                                                 element_connectivity,
                                                 element_indices)
        print(interpolated_props)

    The output will be:

    .. code-block:: console

        [[18.  ]
         [28. ]]

    """
    # Validate input dimensions
    property_array = numpy.asarray(property_array, dtype=numpy.float64)
    if property_array.ndim == 1:
        property_array = property_array[:, numpy.newaxis] # Convert to 2D array with shape (N_v, 1)

    shape_functions = numpy.asarray(shape_functions, dtype=numpy.float64)
    element_connectivity = numpy.asarray(element_connectivity, dtype=numpy.int64)
    element_indices = numpy.asarray(element_indices, dtype=numpy.int64)

    if property_array.ndim != 2:
        raise ValueError(f"'property_array' must be a 2D array of shape (N_v, P). Got {property_array.ndim}D array instead.")
    if shape_functions.ndim != 2:
        raise ValueError(f"'shape_functions' must be a 2D array of shape (N_p, N_npe). Got {shape_functions.ndim}D array instead.")
    if element_connectivity.ndim != 2:
        raise ValueError(f"'element_connectivity' must be a 2D array of shape (N_e, N_npe). Got {element_connectivity.ndim}D array instead.")
    if element_indices.ndim != 1:
        raise ValueError(f"'element_indices' must be a 1D array of shape (N_p,). Got {element_indices.ndim}D array instead.")
    if not shape_functions.shape[0] == element_indices.shape[0]:
        raise ValueError(f"The first dimension (N_p) of 'shape_functions' and 'element_indices' must match. Got {shape_functions.shape[0]} and {element_indices.shape[0]} respectively.")
    if not shape_functions.shape[1] == element_connectivity.shape[1]:
        raise ValueError(f"The second dimension (N_npe) of 'shape_functions' and 'element_connectivity' must match. Got {shape_functions.shape[1]} and {element_connectivity.shape[1]} respectively.")
    
    if not isinstance(skip_m1, bool):
        raise ValueError(f"'skip_m1' must be a boolean value. Got {type(skip_m1)} instead.")
    if isinstance(default, Real):
        default = numpy.full((property_array.shape[1],), default, dtype=numpy.float64) # Convert to array of shape (P,)
    if not (isinstance(default, Real) or (isinstance(default, numpy.ndarray) and default.shape == (property_array.shape[1],))):
        raise ValueError(f"'default' must be a real number or a numpy.ndarray of shape (P,). Got {type(default)} with shape {getattr(default, 'shape', None)} instead.")
    
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
    vertices_properties = property_array[element_connectivity[valid_indices, :], :]  # Shape: (N_p, N_npe, P)
    interpolated_properties_valid = numpy.einsum('mi,mip->mp', shape_functions[m1_mask] if skip_m1 else shape_functions, vertices_properties)  # Shape: (N_p, P)

    # Handle -1 entries if skip_m1 is True
    if skip_m1:
        interpolated_properties = numpy.tile(default, (N_p, 1))
        interpolated_properties[m1_mask, :] = interpolated_properties_valid
    else:
        interpolated_properties = interpolated_properties_valid
    
    return interpolated_properties



def project_property_to_vertices(
    property_array: numpy.ndarray,
    shape_functions: numpy.ndarray,
    element_connectivity: numpy.ndarray,
    element_indices: numpy.ndarray,
    n_vertices: Optional[int] = None,
    points_weights: Optional[numpy.ndarray] = None,
    *,
    sparse: bool = False,
    skip_m1: bool = True,
    return_unaffected: bool = False,
) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
    r"""
    Project a property defined at integration points within elements back to the nodes of a mesh using shape functions.

    .. note::

        The inputs :obj:`property_array`, :obj:`shape_functions`, and :obj:`points_weights` will be converted to :obj:`numpy.float64` for computation.
        The inputs :obj:`element_connectivity`, and :obj:`element_indices` will be converted to :obj:`numpy.int64` for computation.

    .. note::

        By default, nodes that are not affected by any integration point will have their projected property value set to zero. To change this behavior, you can set :obj:`return_unaffected` to :obj:`True` to return the mask with shape :math:`(N_{v},)` indicating which nodes were unaffected (i.e., not influenced by any integration point).


    Parameters
    ----------
    property_array: :class:`numpy.ndarray`
        An array of shape :math:`(N_{p}, P)` containing the property values defined at the :math:`N_{p}` integration points. If 1D-array is provided, it will be treated as a single-component property of shape :math:`(N_{p}, 1)`.

    shape_functions: :class:`numpy.ndarray`
        An array of shape :math:`(N_{p}, N_{vpe})` containing the shape function values evaluated at :math:`N_{p}` points for the :math:`N_{vpe}` nodes of the element.

    element_connectivity: :class:`numpy.ndarray`
        An array of shape :math:`(N_{e}, N_{vpe})` defining the connectivity of the elements in the mesh, where each row contains the indices of the nodes that form an element.

    element_indices: :class:`numpy.ndarray` 
        An array of shape :math:`(N_{p},)` containing the indices of each element corresponding to the :math:`N_{p}` integration points.

    n_vertices: :class:`int`, optional
        The total number of vertices :math:`N_{v}` in the mesh. If not provided, it will be inferred as the maximum index in :obj:`element_connectivity` plus one.

    points_weights: :class:`numpy.ndarray`, optional
        An array of shape :math:`(N_{p},)` containing the weights associated with each integration point. If not provided, all weights will be assumed to be equal to one.

    sparse: :class:`bool`, optional
        If set to :obj:`True`, the shape functions matrix will be constructed as a sparse matrix to optimize memory usage for large meshes.
        Default is :obj:`False`.

    skip_m1: :class:`bool`, optional
        If set to :obj:`True`, any element index of -1 in :obj:`element_indices` will result in the corresponding integration point being ignored during the projection.
        Default is :obj:`True`. 

    return_unaffected: :class:`bool`, optional
        If set to :obj:`True`, the function will return a tuple containing the projected properties and a boolean mask indicating which nodes were unaffected by any integration point.
        Default is :obj:`False`.


    Returns
    -------
    projected_properties: :class:`numpy.ndarray`
        An array of shape :math:`(N_{v}, P)` containing the projected property values at the nodes of the mesh.

    unaffected_mask: :class:`numpy.ndarray`, optional
        A boolean array of shape :math:`(N_{v},)` indicating which nodes were unaffected by any integration point. This is only returned if :obj:`return_unaffected` is set to :obj:`True`.
        :obj:`True` indicates the node was unaffected, while :obj:`False` indicates it was affected. So using :obj:`projected_properties[unaffected_mask, :] = default` will set the unaffected nodes to :obj:`default`.
        

    Raises
    ------
    TypeError
        - If :obj:`property_array`, :obj:`shape_functions`, or :obj:`points_weights` cannot be converted to a :obj:`numpy.ndarray` of floating type.
        - If :obj:`element_connectivity` or :obj:`element_indices` cannot be converted to a :obj:`numpy.ndarray` of integer type.
        - If :obj:`sparse` is not a boolean.
        - If :obj:`skip_m1` is not a boolean.
        - If :obj:`return_unaffected` is not a boolean.

    ValueError
        - If :obj:`property_array` is not a 1D or 2D array of shape :math:`(N_{p},)` or :math:`(N_{p}, P)`.
        - If :obj:`shape_functions` is not a 2D array of shape :math:`(N_{p}, N_{vpe})`.
        - If :obj:`element_connectivity` is not a 2D array of shape :math:`(N_{e}, N_{vpe})`.
        - If :obj:`element_indices` is not a 1D array of shape :math:`(N_{p},)`.
        - If :obj:`points_weights` is not a 1D array of shape :math:`(N_{p},)`.
        - If :obj:`n_vertices` is provided and is not a positive integer or is less than the maximum index in :obj:`element_connectivity` plus one.


    Notes
    -----
    In a space of dimension :math:`E` with a mesh constituted of :math:`N_{e}` elements and :math:`N_{v}` nodes,
    The mesh is composed of :math:`K`-dimensional elements (with :math:`K \leq E`) defined by :math:`N_{vpe}` nodes for each element.

    The evaluation of a property at the given integration points is represented by an array of shape :math:`(N_{p}, P)` and is performed as:

    .. math::

        P_{points} = N_f \cdot P_{nodes} \Leftrightarrow P(\xi, \eta, \zeta, ...) = \sum_{i=1}^{N_{vpe}} N_i(\xi, \eta, \zeta, ...) P_i

    where :math:`N_f` is the shape functions matrix assembled for all :math:`N_{p}` integration points, :math:`P_{points}` is the property array at the integration points, and :math:`P_{nodes}` is the property array at the mesh nodes.

    To project back to the nodes, we solve the following system using the pseudo-inverse of the shape functions matrix:

    .. math::

        P_{nodes} = (N_f^T W N_f)^{-1} N_f^T W P_{points}

    where :math:`W` is a diagonal matrix of weights associated with each integration point.


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

    To project back to the vertices of the mesh, we want tot minimize the following weighted least squares problem:

    .. math::

        \hat{P_{nodes}} = \min_{P_{nodes}} ( \frac{1}{2}\sum_{i=1}^{N_p} w_i \| P_{points,i} - N_{f,i} \cdot P_{nodes} \|^2 ) 
        
    .. math::

        \hat{P_{nodes}} = \min_{P_{nodes}} ( \frac{1}{2} (P_{points} - N_f \cdot P_{nodes})^T W (P_{points} - N_f \cdot P_{nodes}) )

    The gradient of the objective function is:

    .. math::

        \nabla_{P_{nodes}} \hat{P_{nodes}} = - N_f^T W (P_{points} - N_f \cdot P_{nodes})

    To find the optimal solution, we set the gradient to zero and solve for :math:`P_{nodes}`:

    .. math::

        \nabla_{P_{nodes}} \hat{P_{nodes}}  = 0 \Rightarrow N_f^T W P_{points} = N_f^T W N_f \cdot P_{nodes}

    So if we denote :math:`A = N_f^T W N_f` invertible, we have:

    .. math::

        \hat{P_{nodes}} = A^{-1} N_f^T W P_{points}

    
    See Also
    --------
    pysdic.shape_functions:
        To compute the shape functions at given natural coordinates within elements.

    pysdic.interpolate_property:
        To interpolate properties at integration points.

    pysdic.assemble_shape_function_matrix:
        To assemble the shape functions matrix for the projection operation.

        
    Examples
    --------
    Lets interpolate a scalar property at given integration points for triangular elements and project it back to the mesh nodes.

    .. code-block:: python

        import numpy
        from pysdic import project_property_to_vertices
        from pysdic import triangle_3_shape_functions
        from pysdic import interpolate_property

        vertices_coordinates = numpy.array([[0.0, 0.0],
                                            [1.0, 0.0], 
                                            [1.0, 1.0],
                                            [0.0, 1.0]])

        element_connectivity = numpy.array([[0, 1, 2],
                                            [0, 2, 3]])
        N_e = element_connectivity.shape[0]

        property_array = numpy.array([10.0,
                                      20.0,
                                      30.0,
                                      40.0])  # Scalar property at each node

        natural_coordinates = np.array([[0.3, 0.3], [0.2, 0.5], [0.5, 0.2], [0.1, 0.1]])
        N_p = natural_coordinates.shape[0]
        natural_coordinates = np.vstack([natural_coordinates] * N_e)
        element_indices = np.repeat(np.arange(N_e), N_p)
        N_p = natural_coordinates.shape[0]

        shape_functions, _ = triangle_3_shape_functions(natural_coordinates, return_derivatives=True)

        interpolated_props = interpolate_property(property_array,
                                                 shape_functions,
                                                 element_connectivity,
                                                 element_indices)

        projected_props = project_property_to_vertices(interpolated_props,
                                                       shape_functions,
                                                       element_connectivity,
                                                       element_indices,
                                                       n_vertices=vertices_coordinates.shape[0])

        print("Original Properties at Nodes:")
        print(property_array)
        print("Projected Properties at Nodes:")
        print(projected_props)

    The output will be:

    .. code-block:: console

        Original Properties at Nodes:
        [10. 20. 30. 40.]
        Projected Properties at Nodes:
        [[10.]
        [20.]
        [30.]
        [40.]]

    """
    # Validate input dimensions
    property_array = numpy.asarray(property_array, dtype=numpy.float64)
    if property_array.ndim == 1:
        property_array = property_array[:, numpy.newaxis] # Convert to 2D array with shape (N_p, 1)

    shape_functions = numpy.asarray(shape_functions, dtype=numpy.float64)
    element_connectivity = numpy.asarray(element_connectivity, dtype=numpy.int64)
    element_indices = numpy.asarray(element_indices, dtype=numpy.int64)


    if property_array.ndim != 2:
        raise ValueError(f"'property_array' must be a 2D array of shape (N_p, P). Got {property_array.ndim}D array instead.")
    if shape_functions.ndim != 2:
        raise ValueError(f"'shape_functions' must be a 2D array of shape (N_p, N_npe). Got {shape_functions.ndim}D array instead.")
    if element_connectivity.ndim != 2:
        raise ValueError(f"'element_connectivity' must be a 2D array of shape (N_e, N_npe). Got {element_connectivity.ndim}D array instead.")
    if element_indices.ndim != 1:
        raise ValueError(f"'element_indices' must be a 1D array of shape (N_p,). Got {element_indices.ndim}D array instead.")
    if not shape_functions.shape[0] == element_indices.shape[0]:
        raise ValueError(f"The first dimension (N_p) of 'shape_functions' and 'element_indices' must match. Got {shape_functions.shape[0]} and {element_indices.shape[0]} respectively.")
    if not shape_functions.shape[1] == element_connectivity.shape[1]:
        raise ValueError(f"The second dimension (N_npe) of 'shape_functions' and 'element_connectivity' must match. Got {shape_functions.shape[1]} and {element_connectivity.shape[1]} respectively.")
    if not isinstance(sparse, bool):
        raise ValueError(f"'sparse' must be a boolean value. Got {type(sparse)} instead.")
    if not isinstance(skip_m1, bool):
        raise ValueError(f"'skip_m1' must be a boolean value. Got {type(skip_m1)} instead.")
    
    # Extract number of vertices
    if n_vertices is None:
        n_vertices = numpy.max(element_connectivity) + 1
    
    if not isinstance(n_vertices, Integral) or n_vertices <= 0:
        raise ValueError(f"'n_vertices' must be a positive integer. Got {n_vertices} instead.")
    
    # Extract weights
    if points_weights is None:
        points_weights = numpy.ones(shape=(property_array.shape[0],), dtype=numpy.float64)
    else:
        points_weights = numpy.asarray(points_weights, dtype=numpy.float64)
        if points_weights.ndim != 1 or points_weights.shape[0] != property_array.shape[0]:
            raise ValueError(f"'points_weights' must be a 1D array of shape (N_p,). Got array with shape {points_weights.shape} instead.")
        
    # Assemble shape functions matrix
    shape_functions_matrix = assemble_shape_function_matrix(
        shape_functions=shape_functions,
        element_connectivity=element_connectivity,
        element_indices=element_indices,
        n_vertices=n_vertices,
        sparse=sparse,
        skip_m1=skip_m1,
        default=0.0,
    )  # Shape: (N_p, N_v)

    # Initialize projected property
    projected_property = numpy.zeros((n_vertices, property_array.shape[1]), dtype=numpy.float64)

    # Construct weights matrix
    if sparse:
        W = scipy.sparse.diags(points_weights, format='csr')  # (N_p, N_p)
        A = shape_functions_matrix.T @ W @ shape_functions_matrix  # (N_v, N_v)
        b = shape_functions_matrix.T @ W @ property_array  # (N_v, P)

        # Extract valid rows to avoid singular matrix issues
        valid_rows = numpy.unique(shape_functions_matrix.nonzero()[1])

        reduced_A = A[valid_rows, :][:, valid_rows]  # (N_valid, N_valid)
        reduced_b = b[valid_rows, :]  # (N_valid, P)

        # Solve the normal equations
        reduced_projected_property = scipy.sparse.linalg.spsolve(reduced_A, reduced_b)  # (N_valid, P)
        if reduced_projected_property.ndim == 1:
            reduced_projected_property = reduced_projected_property[:, numpy.newaxis]  # Convert to 2D array with shape (N_valid, 1)
        projected_property[valid_rows, :] = reduced_projected_property

    else:
        W = numpy.diag(points_weights)  # (N_p, N_p)
        A = shape_functions_matrix.T @ W @ shape_functions_matrix  # (N_v, N_v)
        b = shape_functions_matrix.T @ W @ property_array  # (N_v, P)

        # Extract valid rows to avoid singular matrix issues
        valid_rows = numpy.unique(numpy.nonzero(shape_functions_matrix)[1])
        A = A[valid_rows, :][:, valid_rows]  # (N_valid, N_valid)
        b = b[valid_rows, :]  # (N_valid, P)

        # Solve the normal equations
        reduced_projected_property = numpy.linalg.solve(A, b)  # (N_valid, P)
        if reduced_projected_property.ndim == 1:
            reduced_projected_property = reduced_projected_property[:, numpy.newaxis]  # Convert to 2D array with shape (N_valid, 1)
        projected_property[valid_rows, :] = reduced_projected_property
    
    # Return unaffected mask if requested
    if return_unaffected:
        unaffected_mask = numpy.ones((n_vertices,), dtype=bool)
        unaffected_mask[valid_rows] = False
        return projected_property, unaffected_mask
    
    else:
        return projected_property
    


def remap_vertices_coordinates(
    vertices_coordinates: numpy.ndarray,
    element_connectivity: numpy.ndarray,
    element_indices: numpy.ndarray,
    *,
    skip_m1: bool = True,
    default: Real = numpy.nan,
) -> numpy.ndarray:
    r"""
    Remap the global coordinates of the vertices to given integration points within elements based on the element connectivity.

    .. note::

        If the input :obj:`vertices_coordinates` is not a numpy floating type, it will be converted to :obj:`numpy.float64`.
        If the inputs :obj:`element_connectivity` or :obj:`element_indices` are not of integer type, they will be converted to :obj:`numpy.int64`.

    .. warning::

        When using :obj:`-1` in :obj:`element_indices` for invalid elements, ensure to set :obj:`skip_m1` to :obj:`True` to avoid indexing errors as :obj:`-1` is equivalent to the last element in Python indexing.

        
    Parameters
    ----------
    vertices_coordinates: :class:`numpy.ndarray`
        An array of shape :math:`(N_{v}, E)` containing the global coordinates of the vertices in the mesh.

    element_connectivity: :class:`numpy.ndarray`
        An array of shape :math:`(N_{e}, N_{vpe})` defining the connectivity of the elements in the mesh, where each row contains the indices of the nodes that form an element.

    element_indices: :class:`numpy.ndarray`
        An array of shape :math:`(N_{p},)` containing the indices of each element corresponding to the :math:`N_{p}` integration points. 

    skip_m1: :class:`bool`, optional
        If set to :obj:`True`, any element index of -1 in :obj:`element_indices` will result in the corresponding remapped coordinates being set to :obj:`default`.
        Default is :obj:`True`.

    default: :class:`Real`, optional
        The default value to assign to remapped coordinates for integration points associated with an element index of -1 when :obj:`skip_m1` is :obj:`True`.
        Default is :obj:`numpy.nan`.


    Returns
    -------
    remapped_coordinates: :class:`numpy.ndarray`
        An array of shape :math:`(N_{p}, N_{vpe}, E)` containing the remapped global coordinates of the vertices for each integration point within the elements.


    Raises
    ------
    TypeError
        - If :obj:`vertices_coordinates` cannot be converted to a :obj:`numpy.ndarray` of floating type.
        - If :obj:`element_connectivity` or :obj:`element_indices` cannot be converted to a :obj:`numpy.ndarray` of integer type.
        - If :obj:`skip_m1` is not a boolean.
        - If :obj:`default` is not a real number.
    
    ValueError
        - If :obj:`vertices_coordinates` is not a 2D array of shape :math:`(N_{v}, E)`.
        - If :obj:`element_connectivity` is not a 2D array of shape :math:`(N_{e}, N_{vpe})`.
        - If :obj:`element_indices` is not a 1D array of shape :math:`(N_{p},)`.

    
    Notes
    -----
    In a space of dimension :math:`E` with a mesh constituted of :math:`N_{e}` elements and :math:`N_{v}` nodes,
    The mesh is composed of :math:`K`-dimensional elements (with :math:`K \leq E`) defined by :math:`N_{vpe}` nodes for each element.

    For a given set of :math:`N_{p}` integration points located within elements, the remapped coordinates array has shape :math:`(N_{p}, N_{vpe}, E)`
    where each entry corresponds to the global coordinates of the nodes associated with the element containing the integration point.


    See Also
    --------
    pysdic.construct_jacobian:
        To compute the Jacobian matrices for the transformation between natural and global coordinates.


    Examples
    --------
    Lets construct a simple 2D mesh

    .. code-block:: python

        import numpy
        from pysdic import remap_vertices_coordinates

        vertices_coordinates = numpy.array([[0.0, 0.0],
                                            [1.0, 0.0], 
                                            [1.0, 1.0],
                                            [0.0, 1.0]])

        element_connectivity = numpy.array([[0, 1, 2],
                                            [0, 2, 3]])

        element_indices = numpy.array([0, 1, 0])

        remapped_coords = remap_vertices_coordinates(vertices_coordinates,
                                                    element_connectivity,
                                                    element_indices)
        print(remapped_coords)

    The output will be:

    .. code-block:: console

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
    element_connectivity = numpy.asarray(element_connectivity, dtype=numpy.int64)
    element_indices = numpy.asarray(element_indices, dtype=numpy.int64)

    
    if vertices_coordinates.ndim != 2:
        raise ValueError(f"'vertices_coordinates' must be a 2D array of shape (N_v, E). Got {vertices_coordinates.ndim}D array instead.")
    if element_connectivity.ndim != 2:
        raise ValueError(f"'element_connectivity' must be a 2D array of shape (N_e, N_npe). Got {element_connectivity.ndim}D array instead.")
    if element_indices.ndim != 1:
        raise ValueError(f"'element_indices' must be a 1D array of shape (N_p,). Got {element_indices.ndim}D array instead.")
    if not isinstance(skip_m1, bool):
        raise ValueError(f"'skip_m1' must be a boolean value. Got {type(skip_m1)} instead.")
    if not isinstance(default, Real):
        raise ValueError(f"'default' must be a real number. Got {type(default)} instead.")
    
    # remap coordinates initialization
    N_p = element_indices.shape[0]
    N_npe = element_connectivity.shape[1]
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
    connectivity_per_point = element_connectivity[valid_indices, :]  # Shape: (N_valid, N_npe)
    remapped_coordinates_valid = vertices_coordinates[connectivity_per_point, :]  # Shape: (N_valid, N_npe, E)

    # Handle -1 entries if skip_m1 is True
    if skip_m1:
        remapped_coordinates = numpy.full((N_p, N_npe, E), default, dtype=remapped_coordinates_valid.dtype)
        remapped_coordinates[m1_mask, :, :] = remapped_coordinates_valid
    else:
        remapped_coordinates = remapped_coordinates_valid

    return remapped_coordinates



