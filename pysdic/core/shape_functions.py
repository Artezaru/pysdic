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
from typing import Tuple, Union, Callable, Optional
from numbers import Real

import numpy


def _input_shape_functions_validation(
    natural_coordinates: numpy.ndarray, 
    return_derivatives: bool, 
    default: Real, 
    topological_dimension: int,
) -> numpy.ndarray:
    r"""
    Validate the input parameters for shape functions computation.

    
    Parameters
    ----------
    natural_coordinates: :class:`numpy.ndarray`
        Natural coordinates where to evaluate the shape functions with shape :math:`(N_{p}, K)`, where :math:`N_{p}` is the number of points and :math:`K` is the topological dimension.

    return_derivatives: :class:`bool`
        If :obj:`True`, the function will also return the first derivatives of the shape functions with respect to the natural coordinates.

    default: :class:`Real`
        Default value to assign to shape functions when the input natural coordinates are out of the valid range.

    topological_dimension: :class:`int`
        Topological dimension of the element (1 for segments, 2 for triangles, etc.).

        
    Returns
    -------
    natural_coordinates: :class:`numpy.ndarray`
        Validated natural coordinates as a numpy array of shape :math:`(N_{p}, K)` with dtype :obj:`numpy.float64`.

        
    Raises
    ------
    TypeError
        - If the input :obj:`natural_coordinates` cannot be converted to a numpy array.
        - If the input :obj:`return_derivatives` is not a boolean.
        - If the input :obj:`default` is not a real number.

    ValueError
        - If the input :obj:`natural_coordinates` does not have shape :math:`(N_{p}, K)`.
    """
    natural_coordinates = numpy.asarray(natural_coordinates, dtype=numpy.float64)

    # Accept 1D arrays and convert to 2D for K=1 elements
    if natural_coordinates.ndim == 1 and topological_dimension == 1:
        natural_coordinates = natural_coordinates[:, numpy.newaxis]

    # Input validation
    if not isinstance(return_derivatives, bool):
        raise TypeError(f"Input 'return_derivatives' must be a boolean value. Got type {type(return_derivatives)}.")
    if not isinstance(default, Real):
        raise TypeError(f"Input 'default' must be a real number. Got type {type(default)}.")
    if natural_coordinates.ndim != 2 or natural_coordinates.shape[1] != topological_dimension:
        raise ValueError(f"Input 'natural_coordinates' must have shape (N_p, {topological_dimension}) for the given element type. Got shape {natural_coordinates.shape}.")

    # Return validated natural coordinates
    return natural_coordinates




def segment_2_shape_functions(
    natural_coordinates: numpy.ndarray, 
    return_derivatives: bool = False, 
    *, 
    default: Real = 0.0
) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
    r"""
    Compute the shape functions for a 2-node segment for given :obj:`natural_coordinates` :math:`\xi`.

    .. note::

        The input :obj:`natural_coordinates` will be converted to :obj:`numpy.float64` for computation.
        The output arrays will be :obj:`numpy.float64` as well.

        
    Parameters
    ----------
    natural_coordinates: :class:`numpy.ndarray`
        Natural coordinates where to evaluate the shape functions. The array must have shape :math:`(N_{p},)` or :math:`(N_{p}, 1)`,
        where :math:`N_{p}` is the number of points to evaluate.

    return_derivatives: :class:`bool`, optional
        If :obj:`True`, the function will also return the first derivatives of the shape functions with respect to the natural coordinate.
        By default, :obj:`False`.

    default: :class:`Real`, optional
        Default value to assign to shape functions when the input natural coordinate is out of the valid range
        (i.e., not in :math:`[-1, 1]`). By default, :obj:`0.0`.


    Returns
    -------
    shape_functions: :class:`numpy.ndarray`
        Shape functions evaluated at the given natural coordinates. The returned array has shape :math:`(N_{p}, 2)`,
        where each row corresponds to a point and each column to a node.

    shape_function_derivatives: :class:`numpy.ndarray`, optional
        If :obj:`return_derivatives` is :obj:`True`, the function also returns an array of the first derivatives of the shape functions
        with respect to the natural coordinate. The returned array has shape :math:`(N_{p}, 2, 1)`.

    
    Raises
    ------
    TypeError
        - If the input :obj:`natural_coordinates` cannot be converted to a numpy array.
        - If the input :obj:`return_derivatives` is not a boolean.
        - If the input :obj:`default` is not a real number.

    ValueError
        - If the input :obj:`natural_coordinates` does not have shape :math:`(N_{p},)` or :math:`(N_{p}, 1)`.


    Notes
    -----
    A 2-node segment represented in the figure below has the following shape functions:

    +----------+-----------------+-----------------------------------------+-----------------------------------------------------+
    | Node No. | :math:`(\xi)`   | Shape Function :math:`N`                | First Derivative :math:`(\frac{dN}{d\xi})`          |
    +==========+=================+=========================================+=====================================================+
    | 1        | :math:`-1`      | :math:`N_1(\xi) = \frac{1}{2}(1 - \xi)` | :math:`-\frac{1}{2}`                                |
    +----------+-----------------+-----------------------------------------+-----------------------------------------------------+
    | 2        | :math:`1`       | :math:`N_2(\xi) = \frac{1}{2}(1 + \xi)` | :math:`\frac{1}{2}`                                 |
    +----------+-----------------+-----------------------------------------+-----------------------------------------------------+

    .. figure:: /_static/shape_functions/segment_2.png
        :alt: 2-node segment element
        :align: center
        :width: 200px
        

    See Also
    --------
    pysdic.segment_3_shape_functions:
        Shape functions for 3-node segment 1D-elements.
    
    pysdic.segment_2_gauss_points:
        Gauss integration points for 2-node segment 1D-elements.
        

    Examples
    --------
    Compute shape functions without derivatives for 3 valid points and 1 invalid point:

    .. code-block:: python

        import numpy
        from pysdic import segment_2_shape_functions

        coords = numpy.array([[-1.0], [0.0], [1.0], [1.5]])
        shape_functions = segment_2_shape_functions(coords)
        print("Shape function values:")
        print(shape_functions)

    .. code-block:: console

        Shape function values:
        [[ 1. ,  0. ],
         [ 0.5,  0.5],
         [ 0. ,  1. ],
         [ 0. ,  0. ]]

    Compute shape functions with first derivatives for 3 valid points and 1 invalid point:
    
    .. code-block:: python

        import numpy
        from pysdic import segment_2_shape_functions

        coords = numpy.array([[-1.0], [0.0], [1.0], [1.5]])
        values, derivatives = segment_2_shape_functions(coords, return_derivatives=True)
        print("Shape function values:")
        print(values)
        print("Shape function derivatives:")
        print(derivatives)

    .. code-block:: console

        Shape function values:
        [[ 1. ,  0. ],
         [ 0.5,  0.5],
         [ 0. ,  1. ],
         [ 0. ,  0. ]]
        Shape function derivatives:
        [[[-0.5]
          [ 0.5]]

         [[-0.5]
          [ 0.5]]

         [[-0.5]
          [ 0.5]]

         [[ 0. ]
          [ 0. ]]]

    Set a different default value for out-of-range coordinates:

    .. code-block:: python

        import numpy
        from pysdic import segment_2_shape_functions

        coords = numpy.array([[-1.0], [0.0], [1.0], [1.5]])
        shape_functions = segment_2_shape_functions(coords, default=numpy.nan)
        print("Shape function values with custom default:")
        print(shape_functions)

    .. code-block:: console

        Shape function values with custom default:
        [[ 1.  0.]
         [ 0.5 0.5]
         [ 0.  1.]
         [nan nan]]

    """
    # Input_validation and conversion
    natural_coordinates = _input_shape_functions_validation(
        natural_coordinates, 
        return_derivatives, 
        default, 
        topological_dimension=1
    )
    
    # Number of points
    n_points = natural_coordinates.shape[0]
    
    # Initialize shape functions array
    shape_functions = numpy.full((n_points, 2), default, dtype=numpy.float64)

    # Valid range mask
    valid_mask = (natural_coordinates >= -1.0) & (natural_coordinates <= 1.0)
    xi = natural_coordinates[valid_mask[:, 0], 0]

    # Compute shape functions for valid coordinates
    shape_functions[valid_mask[:, 0], 0] = 0.5 * (1.0 - xi)
    shape_functions[valid_mask[:, 0], 1] = 0.5 * (1.0 + xi)

    # Compute derivatives if requested
    if return_derivatives:
        # Initialize derivatives array
        derivatives = numpy.full((n_points, 2, 1), default, dtype=numpy.float64)

        # Derivatives are constant for valid coordinates
        derivatives[valid_mask[:, 0], 0, 0] = -0.5
        derivatives[valid_mask[:, 0], 1, 0] = 0.5

        return shape_functions, derivatives

    return shape_functions



        

def segment_3_shape_functions(
    natural_coordinates: numpy.ndarray, 
    return_derivatives: bool = False, 
    *,
    default: Real = 0.0
) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
    r"""
    Compute the shape functions for a 3-node segment for given :obj:`natural_coordinates` :math:`\xi`.

    .. note::

        The input :obj:`natural_coordinates` will be converted to :obj:`numpy.float64` for computation.
        The output arrays will be :obj:`numpy.float64` as well.

        
    Parameters
    ----------
    natural_coordinates: :class:`numpy.ndarray`
        Natural coordinates where to evaluate the shape functions. The array must have shape :math:`(N_{p},)` or :math:`(N_{p}, 1)`,
        where :math:`N_{p}` is the number of points to evaluate.

    return_derivatives: :class:`bool`, optional
        If :obj:`True`, the function will also return the first derivatives of the shape functions with respect to the natural coordinate.
        By default, :obj:`False`.

    default: :class:`Real`, optional
        Default value to assign to shape functions when the input natural coordinate is out of the valid range
        (i.e., not in :math:`[-1, 1]`). By default, :obj:`0.0`.

    
    Returns
    -------
    shape_functions: :class:`numpy.ndarray`
        Shape functions evaluated at the given natural coordinates. The returned array has shape :math:`(N_{p}, 3)`,
        where each row corresponds to a point and each column to a node.

    shape_function_derivatives: :class:`numpy.ndarray`, optional
        If :obj:`return_derivatives` is :obj:`True`, the function also returns an array of the first derivatives of the shape functions
        with respect to the natural coordinate. The returned array has shape :math:`(N_{p}, 3, 1)`.

    
    Raises
    ------
    TypeError
        - If the input :obj:`natural_coordinates` cannot be converted to a numpy array.
        - If the input :obj:`return_derivatives` is not a boolean.
        - If the input :obj:`default` is not a real number.

    ValueError
        - If the input :obj:`natural_coordinates` does not have shape :math:`(N_{p},)` or :math:`(N_{p}, 1)`.

        
    Notes
    -----
    A 3-node segment represented in the figure below has the following shape functions:

    +----------+-----------------+-----------------------------------------------+---------------------------------------------------------+
    | Node No. | :math:`(\xi)`   | Shape Function :math:`N`                      | First Derivative :math:`(\frac{dN}{d\xi})`              |
    +==========+=================+===============================================+=========================================================+
    | 1        | :math:`-1`      | :math:`N_1(\xi) = \frac{1}{2}\xi(\xi - 1)`    | :math:`\xi - \frac{1}{2}`                               |
    +----------+-----------------+-----------------------------------------------+---------------------------------------------------------+
    | 2        | :math:`1`       | :math:`N_2(\xi) = \frac{1}{2}\xi(\xi + 1)`    | :math:`\xi + \frac{1}{2}`                               |
    +----------+-----------------+-----------------------------------------------+---------------------------------------------------------+
    | 3        | :math:`0`       | :math:`N_3(\xi) = 1 - \xi^2`                  | :math:`-2\xi`                                           |
    +----------+-----------------+-----------------------------------------------+---------------------------------------------------------+

    .. figure:: /_static/shape_functions/segment_3.png
        :alt: 3-node segment element
        :align: center
        :width: 200px

    
    See Also
    --------
    pysdic.segment_2_shape_functions:
        Shape functions for 2-node segment 1D-elements.

    pysdic.segment_3_gauss_points:
        Gauss integration points for 3-node segment 1D-elements.


    Examples
    --------
    Compute shape functions without derivatives for 3 valid points and 1 invalid point:

    .. code-block:: python

        import numpy
        from pysdic import segment_3_shape_functions

        coords = numpy.array([[-1.0], [0.0], [1.0], [1.5]])
        shape_functions = segment_3_shape_functions(coords)
        print("Shape function values:")
        print(shape_functions)

    .. code-block:: console

        Shape function values:
        [[ 1. -0.  0.]
         [-0.  0.  1.]
         [ 0.  1.  0.]
         [ 0.  0.  0.]]

    Compute shape functions with first derivatives for 3 valid points and 1 invalid point:

    .. code-block:: python

        import numpy
        from pysdic import segment_3_shape_functions

        coords = numpy.array([[-1.0], [0.0], [1.0], [1.5]])
        values, derivatives = segment_3_shape_functions(coords, return_derivatives=True)
        print("Shape function values:")
        print(values)
        print("Shape function derivatives:")
        print(derivatives)

    .. code-block:: console

        Shape function values:
        [[ 1. -0.  0.]
         [-0.  0.  1.]
         [ 0.  1.  0.]
         [ 0.  0.  0.]]
        Shape function derivatives:
        [[[-1.5]
          [-0.5]
          [ 2. ]]

         [[-0.5]
          [ 0.5]
          [-0. ]]

         [[ 0.5]
          [ 1.5]
          [-2. ]]

         [[ 0. ]
          [ 0. ]
          [ 0. ]]]

    Set a different default value for out-of-range coordinates:

    .. code-block:: python

        import numpy
        from pysdic import segment_3_shape_functions

        coords = numpy.array([[-1.0], [0.0], [1.0], [1.5]])
        shape_functions = segment_3_shape_functions(coords, default=numpy.nan)
        print("Shape function values with custom default:")
        print(shape_functions)

    .. code-block:: console

        Shape function values with custom default:
        [[ 1. -0.  0.]
         [-0.  0.  1.]
         [ 0.  1.  0.]
         [nan nan nan]]

    """
    # Input_validation and conversion
    natural_coordinates = _input_shape_functions_validation(
        natural_coordinates, 
        return_derivatives, 
        default, 
        topological_dimension=1
    )
    
    # Number of points
    n_points = natural_coordinates.shape[0]
    
    # Initialize shape functions array
    shape_functions = numpy.full((n_points, 3), default, dtype=numpy.float64)

    # Valid range mask
    valid_mask = (natural_coordinates >= -1.0) & (natural_coordinates <= 1.0)
    xi = natural_coordinates[valid_mask[:, 0], 0]

    # Compute shape functions for valid coordinates
    shape_functions[valid_mask[:, 0], 0] = 0.5 * xi * (xi - 1.0)
    shape_functions[valid_mask[:, 0], 1] = 0.5 * xi * (xi + 1.0)
    shape_functions[valid_mask[:, 0], 2] = 1.0 - xi**2

    # Compute derivatives if requested
    if return_derivatives: 
        # Initialize derivatives array
        derivatives = numpy.full((n_points, 3, 1), default, dtype=numpy.float64)

        # Derivatives for valid coordinates
        derivatives[valid_mask[:, 0], 0, 0] = xi - 0.5
        derivatives[valid_mask[:, 0], 1, 0] = xi + 0.5
        derivatives[valid_mask[:, 0], 2, 0] = -2.0 * xi

        return shape_functions, derivatives
    
    return shape_functions



def triangle_3_shape_functions(
    natural_coordinates: numpy.ndarray, 
    return_derivatives: bool = False, 
    *, 
    default: Real = 0.0
) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
    r"""
    Compute the shape functions for a 3-node triangle for given :obj:`natural_coordinates` :math:`(\xi, \eta)`.

    .. note::

        The input :obj:`natural_coordinates` will be converted to :obj:`numpy.float64` for computation.
        The output arrays will be :obj:`numpy.float64` as well.

        
    Parameters
    ----------
    natural_coordinates: :class:`numpy.ndarray`
        Natural coordinates where to evaluate the shape functions. The array must have shape :math:`(N_{p}, 2)`,
        where :math:`N_{p}` is the number of points to evaluate.

    return_derivatives: :class:`bool`, optional
        If :obj:`True`, the function will also return the first derivatives of the shape functions with respect to the natural coordinate.
        By default, :obj:`False`.

    default: :class:`Real`, optional
        Default value to assign to shape functions when the input natural coordinate is out of the valid range
        (i.e., not in :math:`\xi, \eta \in [0, 1]` with :math:`\xi + \eta \leq 1`). By default, :obj:`0.0`.

    
    Returns
    -------
    shape_functions: :class:`numpy.ndarray`
        Shape functions evaluated at the given natural coordinates. The returned array has shape :math:`(N_{p}, 3)`,
        where each row corresponds to a point and each column to a node.

    shape_function_derivatives: :class:`numpy.ndarray`, optional
        If :obj:`return_derivatives` is :obj:`True`, the function also returns an array of the first derivatives of the shape functions
        with respect to the natural coordinate. The returned array has shape :math:`(N_{p}, 3, 2)`.

    
    Raises
    ------
    TypeError
        - If the input :obj:`natural_coordinates` cannot be converted to a numpy array.
        - If the input :obj:`return_derivatives` is not a boolean.
        - If the input :obj:`default` is not a real number.

    ValueError
        - If the input :obj:`natural_coordinates` does not have shape :math:`(N_{p}, 2)`.


    Notes
    -----
    A 3-node triangle represented in the figure below has the following shape functions:

    +----------+-----------------------+-----------------------------------------------+--------------------------------------------------------------+
    | Node No. | :math:`(\xi, \eta)`   | Shape Function :math:`N`                      | First Derivative :math:`(\frac{dN}{d\xi}, \frac{dN}{d\eta})` |
    +==========+=======================+===============================================+==============================================================+
    | 1        | :math:`(0, 0)`        | :math:`N_1(\xi, \eta) = 1 - \xi - \eta`       | :math:`(-1, -1)`                                             |
    +----------+-----------------------+-----------------------------------------------+--------------------------------------------------------------+
    | 2        | :math:`(1, 0)`        | :math:`N_2(\xi, \eta) = \xi`                  | :math:`(1, 0)`                                               |
    +----------+-----------------------+-----------------------------------------------+--------------------------------------------------------------+
    | 3        | :math:`(0, 1)`        | :math:`N_3(\xi, \eta) = \eta`                 | :math:`(0, 1)`                                               |
    +----------+-----------------------+-----------------------------------------------+--------------------------------------------------------------+

    .. figure:: /_static/shape_functions/triangle_3.png
        :alt: 3-node triangle element
        :align: center
        :width: 200px


    See Also
    --------
    pysdic.triangle_6_shape_functions:
        Shape functions for 6-node triangle 2D-elements.

    pysdic.triangle_3_gauss_points:
        Gauss integration points for 3-node triangle 2D-elements.

    
    Examples
    --------
    Compute shape functions without derivatives for 3 valid points and 1 invalid point:

    .. code-block:: python

        import numpy
        from pysdic import triangle_3_shape_functions

        coords = numpy.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.7, 0.5]])
        shape_functions = triangle_3_shape_functions(coords)
        print("Shape function values:")
        print(shape_functions)

    .. code-block:: console

        Shape function values:
        [[1.  0.  0. ]
         [0.5 0.5 0. ]
         [0.5 0.  0.5]
         [0.  0.  0. ]]

    Compute shape functions with first derivatives for 3 valid points and 1 invalid point:

    .. code-block:: python

        import numpy
        from pysdic import triangle_3_shape_functions

        coords = numpy.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.7, 0.5]])
        values, derivatives = triangle_3_shape_functions(coords, return_derivatives=True)
        print("Shape function values:")
        print(values)
        print("Shape function derivatives:")
        print(derivatives)

    .. code-block:: console

        Shape function values:
        [[1.  0.  0. ]
         [0.5 0.5 0. ]
         [0.5 0.  0.5]
         [ 0.   0.   0. ]]
        Shape function derivatives:
        [[[-1. -1.]
          [ 1.  0.]
          [ 0.  1.]]

         [[-1. -1.]
          [ 1.  0.]
          [ 0.  1.]]

         [[-1. -1.]
          [ 1.  0.]
          [ 0.  1.]]

         [[ 0.   0. ]
          [ 0.   0. ]
          [ 0.   0. ]]]

    Set a different default value for out-of-range coordinates:

    .. code-block:: python

        import numpy
        from pysdic import triangle_3_shape_functions

        coords = numpy.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.7, 0.5]])
        shape_functions = triangle_3_shape_functions(coords, default=numpy.nan)
        print("Shape function values with custom default:")
        print(shape_functions)

    .. code-block:: console

        Shape function values with custom default:
        [[ 1.  0.  0.]
         [ 0.5 0.5 0.]
         [ 0.5 0.  0.5]
         [nan nan nan]]

    """
    # Input_validation and conversion
    natural_coordinates = _input_shape_functions_validation(
        natural_coordinates, 
        return_derivatives, 
        default, 
        topological_dimension=2
    )
    
    # Number of points
    n_points = natural_coordinates.shape[0]
    
    # Initialize shape functions array
    shape_functions = numpy.full((n_points, 3), default, dtype=numpy.float64)

    # Valid range mask
    valid_mask = (natural_coordinates[:, 0] >= 0.0) & (natural_coordinates[:, 1] >= 0.0) & ((natural_coordinates[:, 0] + natural_coordinates[:, 1]) <= 1.0)
    xi = natural_coordinates[valid_mask, 0]
    eta = natural_coordinates[valid_mask, 1]

    # Compute shape functions for valid coordinates
    shape_functions[valid_mask, 0] = 1.0 - xi - eta
    shape_functions[valid_mask, 1] = xi
    shape_functions[valid_mask, 2] = eta

    # Compute derivatives if requested
    if return_derivatives:
        # Initialize derivatives array
        derivatives = numpy.full((n_points, 3, 2), default, dtype=numpy.float64)

        # Derivatives are constant for valid coordinates
        derivatives[valid_mask, 0, 0] = -1.0
        derivatives[valid_mask, 0, 1] = -1.0
        derivatives[valid_mask, 1, 0] = 1.0
        derivatives[valid_mask, 1, 1] = 0.0
        derivatives[valid_mask, 2, 0] = 0.0
        derivatives[valid_mask, 2, 1] = 1.0

        return shape_functions, derivatives
    
    return shape_functions




def triangle_6_shape_functions(
    natural_coordinates: numpy.ndarray, 
    return_derivatives: bool = False, 
    *, 
    default: Real = 0.0
) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
    r"""
    Compute the shape functions for a 6-node triangle for given :obj:`natural_coordinates` :math:`(\xi, \eta)`.

    .. note::

        The input :obj:`natural_coordinates` will be converted to :obj:`numpy.float64` for computation.
        The output arrays will be :obj:`numpy.float64` as well.

        
    Parameters
    ----------
    natural_coordinates: :class:`numpy.ndarray`
        Natural coordinates where to evaluate the shape functions. The array must have shape :math:`(N_{p}, 2)`,
        where :math:`N_{p}` is the number of points to evaluate.

    return_derivatives: :class:`bool`, optional
        If :obj:`True`, the function will also return the first derivatives of the shape functions with respect to the natural coordinate.
        By default, :obj:`False`.

    default: :class:`Real`, optional
        Default value to assign to shape functions when the input natural coordinate is out of the valid range
        (i.e., not in :math:`\xi, \eta \in [0, 1]` with :math:`\xi + \eta \leq 1`). By default, :obj:`0.0`.

    
    Returns
    -------
    shape_functions: :class:`numpy.ndarray`
        Shape functions evaluated at the given natural coordinates. The returned array has shape :math:`(N_{p}, 6)`,
        where each row corresponds to a point and each column to a node.

    shape_function_derivatives: :class:`numpy.ndarray`, optional
        If :obj:`return_derivatives` is :obj:`True`, the function also returns an array of the first derivatives of the shape functions
        with respect to the natural coordinate. The returned array has shape :math:`(N_{p}, 6, 2)`.

    
    Raises
    ------
    TypeError
        - If the input :obj:`natural_coordinates` cannot be converted to a numpy array.
        - If the input :obj:`return_derivatives` is not a boolean.
        - If the input :obj:`default` is not a real number.

    ValueError
        - If the input :obj:`natural_coordinates` does not have shape :math:`(N_{p}, 2)`.


    Notes
    -----
    A 6-node triangle represented in the figure below has the following shape functions:

    +----------+-----------------------+--------------------------------------------------------------------------------------+--------------------------------------------------------------------+
    | Node No. | :math:`(\xi, \eta)`   | Shape Function :math:`N`                                                             | First Derivative :math:`(\frac{dN}{d\xi}, \frac{dN}{d\eta})`       |
    +==========+=======================+======================================================================================+====================================================================+
    | 1        | :math:`(0, 0)`        | :math:`N_1(\xi, \eta) = (1 - \xi - \eta)(1 - 2\xi - 2\eta)`                          | :math:`(-3 + 4\xi + 4\eta, -3 + 4\xi + 4\eta)`                     |
    +----------+-----------------------+--------------------------------------------------------------------------------------+--------------------------------------------------------------------+
    | 2        | :math:`(1, 0)`        | :math:`N_2(\xi, \eta) = \xi(2\xi - 1)`                                               | :math:`(4\xi - 1, 0)`                                              |
    +----------+-----------------------+--------------------------------------------------------------------------------------+--------------------------------------------------------------------+
    | 3        | :math:`(0, 1)`        | :math:`N_3(\xi, \eta) = \eta(2\eta - 1)`                                             | :math:`(0, 4\eta - 1)`                                             |
    +----------+-----------------------+--------------------------------------------------------------------------------------+--------------------------------------------------------------------+
    | 4        | :math:`(0.5, 0)`      | :math:`N_4(\xi, \eta) = 4\xi(1 - \xi - \eta)`                                        | :math:`(4 - 8\xi - 4\eta, -4\xi)`                                  |
    +----------+-----------------------+--------------------------------------------------------------------------------------+--------------------------------------------------------------------+
    | 5        | :math:`(0.5, 0.5)`    | :math:`N_5(\xi, \eta) = 4\xi\eta`                                                    | :math:`(4\eta, 4\xi)`                                              |
    +----------+-----------------------+--------------------------------------------------------------------------------------+--------------------------------------------------------------------+
    | 6        | :math:`(0, 0.5)`      | :math:`N_6(\xi, \eta) = 4\eta(1 - \xi - \eta)`                                       | :math:`(-4\eta, 4 - 4\xi - 8\eta)`                                 |
    +----------+-----------------------+--------------------------------------------------------------------------------------+--------------------------------------------------------------------+

    .. figure:: /_static/shape_functions/triangle_6.png
        :alt: 6-node triangle element
        :align: center
        :width: 200px


    See Also
    --------
    pysdic.triangle_3_shape_functions:
        Shape functions for 3-node triangle 2D-elements.

    pysdic.triangle_6_gauss_points:
        Gauss integration points for 6-node triangle 2D-elements.

    
    Examples
    --------

    .. code-block:: python

        import numpy
        from pysdic import triangle_6_shape_functions

        coords = numpy.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.7, 0.5]])
        shape_functions = triangle_6_shape_functions(coords)
        print("Shape function values:")
        print(shape_functions)

    .. code-block:: console

        Shape function values:
        [[1.   0.   0.   0.   0.   0.  ]
         [0.   0    0.   1.   0.   0.  ]
         [0.   0.   0.   0.   0.   1.  ]
         [0.   0.   0.   0.   0.   0.  ]]

    .. code-block:: python

        import numpy
        from pysdic import triangle_6_shape_functions

        coords = numpy.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.7, 0.5]])
        values, derivatives = triangle_6_shape_functions(coords, return_derivatives=True)
        print("Shape function values:")
        print(values)
        print("Shape function derivatives:")
        print(derivatives)

    .. code-block:: console

        Shape function values:
        [[1.  -0.  -0.   0.   0.   0.  ]
         [0.   0   -0.   1.   0.   0.  ]
         [0.  -0.   0.   0.   0.   1.  ]
         [0.   0.   0.   0.   0.   0.  ]]
        Shape function derivatives:
        [[[-3. -3.]
          [-1.  0.]
          [ 0. -1.]
          [ 4. -0.]
          [ 0.  0.]
          [-0.  4.]]

         [[-1. -1.]
          [ 1.  0.]
          [ 0. -1.]
          [ 0. -2.]
          [ 0.  2.]
          [-0.  2.]]

         [[-1. -1.]
          [-1.  0.]
          [ 0.  1.]
          [ 2. -0.]
          [ 2.  0.]
          [-2.  0.]]

         [[ 0.  0. ]
          [ 0.  0. ]
          [ 0.  0. ]
          [ 0.  0. ]
          [ 0.  0. ]
          [ 0.  0. ]]]

    Set a different default value for out-of-range coordinates:

    .. code-block:: python

        import numpy
        from pysdic import triangle_6_shape_functions

        coords = numpy.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.7, 0.5]])
        shape_functions = triangle_6_shape_functions(coords, default=numpy.nan)
        print("Shape function values with custom default:")
        print(shape_functions)

    .. code-block:: console

        Shape function values with custom default:
        [[ 1.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  1.  0.  0.]
         [ 0.  0.  0.  0.  0.  1.]
         [nan nan nan nan nan nan]]

    """
    # Input_validation and conversion
    natural_coordinates = _input_shape_functions_validation(
        natural_coordinates,
        return_derivatives,
        default,
        topological_dimension=2
    )
    
    # Number of points
    n_points = natural_coordinates.shape[0]

    # Initialize shape functions array
    shape_functions = numpy.full((n_points, 6), default, dtype=numpy.float64)

    # Valid range mask
    valid_mask = (natural_coordinates[:, 0] >= 0.0) & (natural_coordinates[:, 1] >= 0.0) & ((natural_coordinates[:, 0] + natural_coordinates[:, 1]) <= 1.0)
    xi = natural_coordinates[valid_mask, 0]
    eta = natural_coordinates[valid_mask, 1]

    # Compute shape functions for valid coordinates
    shape_functions[valid_mask, 0] = (1.0 - xi - eta) * (1.0 - 2.0 * xi - 2.0 * eta)
    shape_functions[valid_mask, 1] = xi * (2.0 * xi - 1.0)
    shape_functions[valid_mask, 2] = eta * (2.0 * eta - 1.0)
    shape_functions[valid_mask, 3] = 4.0 * xi * (1.0 - xi - eta)
    shape_functions[valid_mask, 4] = 4.0 * xi * eta
    shape_functions[valid_mask, 5] = 4.0 * eta * (1.0 - xi - eta)

    # Compute derivatives if requested
    if return_derivatives:
        # Initialize derivatives array
        derivatives = numpy.full((n_points, 6, 2), default, dtype=numpy.float64)

        # Derivatives for valid coordinates
        derivatives[valid_mask, 0, 0] = -3.0 + 4.0 * xi + 4.0 * eta
        derivatives[valid_mask, 0, 1] = -3.0 + 4.0 * xi + 4.0 * eta
        derivatives[valid_mask, 1, 0] = 4.0 * xi - 1.0
        derivatives[valid_mask, 1, 1] = 0.0
        derivatives[valid_mask, 2, 0] = 0.0
        derivatives[valid_mask, 2, 1] = 4.0 * eta - 1.0
        derivatives[valid_mask, 3, 0] = 4.0 - 8.0 * xi - 4.0 * eta
        derivatives[valid_mask, 3, 1] = -4.0 * xi
        derivatives[valid_mask, 4, 0] = 4.0 * eta
        derivatives[valid_mask, 4, 1] = 4.0 * xi
        derivatives[valid_mask, 5, 0] = -4.0 * eta
        derivatives[valid_mask, 5, 1] = 4.0 - 4.0 * xi - 8.0 * eta

        return shape_functions, derivatives
    
    return shape_functions




def quadrangle_4_shape_functions(
    natural_coordinates: numpy.ndarray, 
    return_derivatives: bool = False, 
    *, 
    default: Real = 0.0
) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
    r"""
    Compute the shape functions for a 4-node quadrangle for given :obj:`natural_coordinates` :math:`(\xi, \eta)`.

    .. note::

        The input :obj:`natural_coordinates` will be converted to :obj:`numpy.float64` for computation.
        The output arrays will be :obj:`numpy.float64` as well.

        
    Parameters
    ----------
    natural_coordinates: :class:`numpy.ndarray`
        Natural coordinates where to evaluate the shape functions. The array must have shape :math:`(N_{p}, 2)`,
        where :math:`N_{p}` is the number of points to evaluate.

    return_derivatives: :class:`bool`, optional
        If :obj:`True`, the function will also return the first derivatives of the shape functions with respect to the natural coordinate.
        By default, :obj:`False`.

    default: :class:`Real`, optional
        Default value to assign to shape functions when the input natural coordinate is out of the valid range
        (i.e., not in :math:`\xi, \eta \in [-1, 1]`). By default, :obj:`0.0`.


    Returns
    -------
    shape_functions: :class:`numpy.ndarray`
        Shape functions evaluated at the given natural coordinates. The returned array has shape :math:`(N_{p}, 4)`,
        where each row corresponds to a point and each column to a node.

    shape_function_derivatives: :class:`numpy.ndarray`, optional
        If :obj:`return_derivatives` is :obj:`True`, the function also returns an array of the first derivatives of the shape functions
        with respect to the natural coordinate. The returned array has shape :math:`(N_{p}, 4, 2)`.

    
    Raises
    ------
    TypeError
        - If the input :obj:`natural_coordinates` cannot be converted to a numpy array.
        - If the input :obj:`return_derivatives` is not a boolean.
        - If the input :obj:`default` is not a real number.

    ValueError
        - If the input :obj:`natural_coordinates` does not have shape :math:`(N_{p}, 2)`.

    
    Notes
    -----
    A 4-node quadrangle represented in the figure below has the following shape functions:

    +----------+-----------------------+---------------------------------------------------------------+--------------------------------------------------------------------+
    | Node No. | :math:`(\xi, \eta)`   | Shape Function :math:`N`                                      | First Derivative :math:`(\frac{dN}{d\xi}, \frac{dN}{d\eta})`       |
    +==========+=======================+===============================================================+====================================================================+
    | 1        | :math:`(-1, -1)`      | :math:`N_1(\xi, \eta) = \frac{1}{4}(1 - \xi)(1 - \eta)`       | :math:`(\frac{1}{4}(\eta - 1), \frac{1}{4}(\xi - 1))`              |
    +----------+-----------------------+---------------------------------------------------------------+--------------------------------------------------------------------+
    | 2        | :math:`(1, -1)`       | :math:`N_2(\xi, \eta) = \frac{1}{4}(1 + \xi)(1 - \eta)`       | :math:`(\frac{1}{4}(1 - \eta), -\frac{1}{4}(1 + \xi))`             |
    +----------+-----------------------+---------------------------------------------------------------+--------------------------------------------------------------------+
    | 3        | :math:`(1, 1)`        | :math:`N_3(\xi, \eta) = \frac{1}{4}(1 + \xi)(1 + \eta)`       | :math:`(\frac{1}{4}(1 + \eta), \frac{1}{4}(1 + \xi))`              |
    +----------+-----------------------+---------------------------------------------------------------+--------------------------------------------------------------------+
    | 4        | :math:`(-1, 1)`       | :math:`N_4(\xi, \eta) = \frac{1}{4}(1 - \xi)(1 + \eta)`       | :math:`(-\frac{1}{4}(1 + \eta), \frac{1}{4}(1 - \xi))`             |
    +----------+-----------------------+---------------------------------------------------------------+--------------------------------------------------------------------+

    .. figure:: /_static/shape_functions/quadrangle_4.png
        :alt: 4-node quadrangle element
        :align: center
        :width: 200px

    
    See Also
    --------
    pysdic.quadrangle_8_shape_functions:
        Shape functions for 8-node quadrangle 2D-elements.

    pysdic.quadrangle_4_gauss_points:
        Gauss integration points for 4-node quadrangle 2D-elements.


    Examples
    --------
    Compute shape functions without derivatives for 3 valid points and 1 invalid point:

    .. code-block:: python

        import numpy
        from pysdic import quadrangle_4_shape_functions

        coords = numpy.array([[-1.0, -1.0], [0.0, -1.0], [1.0, 0.0], [1.5, 0.5]])
        shape_functions = quadrangle_4_shape_functions(coords)
        print("Shape function values:")
        print(shape_functions)

    .. code-block:: console

        Shape function values:
        [[1.   0.   0.   0.  ]
         [0.5  0.5  0.   0.  ]
         [0.   0.5  0.5  0.  ]
         [0.   0.   0.   0.  ]]

    Compute shape functions with first derivatives for 3 valid points and 1 invalid point:

    .. code-block:: python

        import numpy
        from pysdic import quadrangle_4_shape_functions

        coords = numpy.array([[-1.0, -1.0], [0.0, -1.0], [1.0, 0.0], [1.5, 0.5]])
        values, derivatives = quadrangle_4_shape_functions(coords, return_derivatives=True)
        print("Shape function values:")
        print(values)
        print("Shape function derivatives:")
        print(derivatives)

    .. code-block:: console

        Shape function values:
        [[1.   0.   0.   0.  ]
         [0.5  0.5  0.   0.  ]
         [0.   0.5  0.5  0.  ]
         [0.   0.   0.   0.  ]]
        Shape function derivatives:
        [[[-0.5  -0.5 ]
          [ 0.5  -0.  ]
          [ 0.    0.  ]
          [-0.    0.5 ]]

         [[-0.5  -0.25]
          [ 0.5  -0.25]
          [ 0.    0.25]
          [-0.    0.25]]

         [[-0.25  0.  ]
          [ 0.25 -0.5 ]
          [ 0.25  0.5 ]
          [-0.25  0.  ]]

         [[ 0.    0.  ]
          [ 0.    0.  ]
          [ 0.    0.  ]
          [ 0.    0.  ]]]

    Set a different default value for out-of-range coordinates:

    .. code-block:: python

        import numpy
        from pysdic import quadrangle_4_shape_functions

        coords = numpy.array([[-1.0, -1.0], [0.0, -1.0], [1.0, 0.0], [1.5, 0.5]])
        shape_functions = quadrangle_4_shape_functions(coords, default=numpy.nan)
        print("Shape function values with custom default:")
        print(shape_functions)

    .. code-block:: console

        Shape function values with custom default:
        [[ 1.  0.  0.  0.]
         [ 0.5 0.5 0.  0.]
         [ 0.  0.5 0.5 0.]
         [nan nan nan nan]]

    """
    # Input_validation and conversion
    natural_coordinates = _input_shape_functions_validation(
        natural_coordinates,
        return_derivatives,
        default,
        topological_dimension=2
    )
    
    # Number of points
    n_points = natural_coordinates.shape[0]
    
    # Initialize shape functions array
    shape_functions = numpy.full((n_points, 4), default, dtype=numpy.float64)

    # Valid range mask
    valid_mask = (natural_coordinates[:, 0] >= -1.0) & (natural_coordinates[:, 0] <= 1.0) & (natural_coordinates[:, 1] >= -1.0) & (natural_coordinates[:, 1] <= 1.0)
    xi = natural_coordinates[valid_mask, 0]
    eta = natural_coordinates[valid_mask, 1]

    # Compute shape functions for valid coordinates
    shape_functions[valid_mask, 0] = 0.25 * (1.0 - xi) * (1.0 - eta)
    shape_functions[valid_mask, 1] = 0.25 * (1.0 + xi) * (1.0 - eta)
    shape_functions[valid_mask, 2] = 0.25 * (1.0 + xi) * (1.0 + eta)
    shape_functions[valid_mask, 3] = 0.25 * (1.0 - xi) * (1.0 + eta)

    # Compute derivatives if requested
    if return_derivatives:
        # Initialize derivatives array
        derivatives = numpy.full((n_points, 4, 2), default, dtype=numpy.float64)

        # Derivatives for valid coordinates
        derivatives[valid_mask, 0, 0] = 0.25 * (eta - 1.0)
        derivatives[valid_mask, 0, 1] = 0.25 * (xi - 1.0)
        derivatives[valid_mask, 1, 0] = 0.25 * (1.0 - eta)
        derivatives[valid_mask, 1, 1] = -0.25 * (1.0 + xi)
        derivatives[valid_mask, 2, 0] = 0.25 * (1.0 + eta)
        derivatives[valid_mask, 2, 1] = 0.25 * (1.0 + xi)
        derivatives[valid_mask, 3, 0] = -0.25 * (1.0 + eta)
        derivatives[valid_mask, 3, 1] = 0.25 * (1.0 - xi)

        return shape_functions, derivatives
    
    return shape_functions




def quadrangle_8_shape_functions(
    natural_coordinates: numpy.ndarray, 
    return_derivatives: bool = False, 
    *, 
    default: Real = 0.0
) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
    r"""
    Compute the shape functions for a 8-node quadrangle for given :obj:`natural_coordinates` :math:`(\xi, \eta)`.

    .. note::

        The input :obj:`natural_coordinates` will be converted to :obj:`numpy.float64` for computation.
        The output arrays will be :obj:`numpy.float64` as well.

        
    Parameters
    ----------
    natural_coordinates: :class:`numpy.ndarray`
        Natural coordinates where to evaluate the shape functions. The array must have shape :math:`(N_{p}, 2)`,
        where :math:`N_{p}` is the number of points to evaluate.

    return_derivatives: :class:`bool`, optional
        If :obj:`True`, the function will also return the first derivatives of the shape functions with respect to the natural coordinate.
        By default, :obj:`False`.

    default: :class:`Real`, optional
        Default value to assign to shape functions when the input natural coordinate is out of the valid range
        (i.e., not in :math:`\xi, \eta \in [-1, 1]`). By default, :obj:`0.0`.

    
    Returns
    -------
    shape_functions: :class:`numpy.ndarray`
        Shape functions evaluated at the given natural coordinates. The returned array has shape :math:`(N_{p}, 8)`,
        where each row corresponds to a point and each column to a node.

    shape_function_derivatives: :class:`numpy.ndarray`, optional
        If :obj:`return_derivatives` is :obj:`True`, the function also returns an array of the first derivatives of the shape functions
        with respect to the natural coordinate. The returned array has shape :math:`(N_{p}, 8, 2)`.

    
    Raises
    ------
    TypeError
        - If the input :obj:`natural_coordinates` cannot be converted to a numpy array.
        - If the input :obj:`return_derivatives` is not a boolean.
        - If the input :obj:`default` is not a real number.

    ValueError
        - If the input :obj:`natural_coordinates` does not have shape :math:`(N_{p}, 2)`.

    
    Notes
    -----
    An 8-node quadrangle represented in the figure below has the following shape functions:

    +----------+-----------------------+--------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------+
    | Node No. | :math:`(\xi, \eta)`   | Shape Function :math:`N`                                                                   | First Derivative :math:`(\frac{dN}{d\xi}, \frac{dN}{d\eta})`                                 |
    +==========+=======================+============================================================================================+==============================================================================================+
    | 1        | :math:`(-1, -1)`      | :math:`N_1(\xi, \eta) = \frac{1}{4}(1 - \xi)(1 - \eta)(- \xi - \eta - 1)`                  | :math:`(\frac{1}{4}(1 - \eta)(2\xi + \eta), \frac{1}{4}(1 - \xi)(\xi + 2\eta))`              |
    +----------+-----------------------+--------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------+
    | 2        | :math:`(1, -1)`       | :math:`N_2(\xi, \eta) = \frac{1}{4}(1 + \xi)(1 - \eta)(\xi - \eta - 1)`                    | :math:`(\frac{1}{4}(1 - \eta)(2\xi - \eta), -\frac{1}{4}(1 + \xi)(\xi - 2\eta))`             |
    +----------+-----------------------+--------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------+
    | 3        | :math:`(1, 1)`        | :math:`N_3(\xi, \eta) = \frac{1}{4}(1 + \xi)(1 + \eta)(\xi + \eta - 1)`                    | :math:`(\frac{1}{4}(1 + \eta)(2\xi + \eta), \frac{1}{4}(1 + \xi)(\xi + 2\eta))`              |
    +----------+-----------------------+--------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------+
    | 4        | :math:`(-1, 1)`       | :math:`N_4(\xi, \eta) = \frac{1}{4}(1 - \xi)(1 + \eta)(- \xi + \eta - 1)`                  | :math:`(\frac{1}{4}(1 + \eta)(2\xi - \eta), \frac{1}{4}(1 - \xi)(- \xi + 2\eta))`            |
    +----------+-----------------------+--------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------+
    | 5        | :math:`(0, -1)`       | :math:`N_5(\xi, \eta) = \frac{1}{2}(1 - \xi^2)(1 - \eta)`                                  | :math:`(-\xi(1 - \eta), -\frac{1}{2}(1 - \xi^2))`                                            |
    +----------+-----------------------+--------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------+
    | 6        | :math:`(1, 0)`        | :math:`N_6(\xi, \eta) = \frac{1}{2}(1 + \xi)(1 - \eta^2)`                                  | :math:`(\frac{1}{2}(1 - \eta^2), -\eta(1 + \xi))`                                            |
    +----------+-----------------------+--------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------+
    | 7        | :math:`(0, 1)`        | :math:`N_7(\xi, \eta) = \frac{1}{2}(1 - \xi^2)(1 + \eta)`                                  | :math:`(-\xi(1 + \eta), \frac{1}{2}(1 - \xi^2))`                                             |
    +----------+-----------------------+--------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------+
    | 8        | :math:`(-1, 0)`       | :math:`N_8(\xi, \eta) = \frac{1}{2}(1 - \xi)(1 - \eta^2)`                                  | :math:`(-\frac{1}{2}(1 - \eta^2), -\eta(1 - \xi))`                                           |
    +----------+-----------------------+--------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------+

    .. figure:: /_static/shape_functions/quadrangle_8.png
        :alt: 8-node quadrangle element
        :align: center
        :width: 200px


    See Also
    --------
    pysdic.quadrangle_4_shape_functions:
        Shape functions for 4-node quadrangle 2D-elements.

    pysdic.quadrangle_8_gauss_points:
        Gauss integration points for 8-node quadrangle 2D-elements.

    
    Examples
    --------

    .. code-block:: python

        import numpy
        from pysdic import quadrangle_8_shape_functions

        coords = numpy.array([[-1.0, -1.0], [0.0, -1.0], [1.0, 0.0], [1.5, 0.5]])
        shape_functions = quadrangle_8_shape_functions(coords)
        print("Shape function values:")
        print(shape_functions)

    .. code-block:: console

        Shape function values:
        [[ 1. -0. -0. -0.  0.  0.  0.  0.]
         [ 0.  0. -0. -0.  1.  0.  0.  0.]
         [-0.  0.  0. -0.  0.  1.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  0.]]

    .. code-block:: python

        import numpy
        from pysdic import quadrangle_8_shape_functions
        
        coords = numpy.array([[-1.0, -1.0], [0.0, -1.0], [1.0, 0.0], [1.5, 0.5]])
        values, derivatives = quadrangle_8_shape_functions(coords, return_derivatives=True)
        print("Shape function values:")
        print(values)
        print("Shape function derivatives:")
        print(derivatives)

    .. code-block:: console

        Shape function values:
        [[ 1. -0. -0. -0.  0.  0.  0.  0.]
         [ 0.  0. -0. -0.  1.  0.  0.  0.]
         [-0.  0.  0. -0.  0.  1.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  0.]]
        Shape function derivatives:
        [[[-1.5 -1.5]
          [-0.5 -0. ]
          [-0.  -0. ]
          [-0.  -0.5]
          [ 2.  -0. ]
          [ 0.   0. ]
          [ 0.   0. ]
          [-0.   2. ]]

         [[-0.5 -0.5]
          [ 0.5 -0.5]
          [-0.  -0.5]
          [ 0.  -0.5]
          [-0.  -0.5]
          [ 0.   1. ]
          [-0.   0.5]
          [-0.   1. ]]

         [[ 0.5  0. ]
          [ 0.5 -0.5]
          [ 0.5  0.5]
          [ 0.5 -0. ]
          [-1.  -0. ]
          [ 0.5 -0. ]
          [-1.   0. ]
          [-0.5 -0. ]]

         [[ 0.   0. ]
          [ 0.   0. ]
          [ 0.   0. ]
          [ 0.   0. ]
          [ 0.   0. ]
          [ 0.   0. ]
          [ 0.   0. ]
          [ 0.   0. ]]]

    """
    # Input_validation and conversion
    natural_coordinates = _input_shape_functions_validation(
        natural_coordinates,
        return_derivatives,
        default,
        topological_dimension=2
    )

    # Number of points
    n_points = natural_coordinates.shape[0]

    # Initialize shape functions array
    shape_functions = numpy.full((n_points, 8), default, dtype=numpy.float64)

    # Valid range mask
    valid_mask = (natural_coordinates[:, 0] >= -1.0) & (natural_coordinates[:, 0] <= 1.0) & (natural_coordinates[:, 1] >= -1.0) & (natural_coordinates[:, 1] <= 1.0)
    xi = natural_coordinates[valid_mask, 0]
    eta = natural_coordinates[valid_mask, 1]

    # Compute shape functions for valid coordinates
    shape_functions[valid_mask, 0] = 0.25 * (1.0 - xi) * (1.0 - eta) * (-xi - eta - 1.0)
    shape_functions[valid_mask, 1] = 0.25 * (1.0 + xi) * (1.0 - eta) * (xi - eta - 1.0)
    shape_functions[valid_mask, 2] = 0.25 * (1.0 + xi) * (1.0 + eta) * (xi + eta - 1.0)
    shape_functions[valid_mask, 3] = 0.25 * (1.0 - xi) * (1.0 + eta) * (-xi + eta - 1.0)
    shape_functions[valid_mask, 4] = 0.5 * (1.0 - xi**2) * (1.0 - eta)
    shape_functions[valid_mask, 5] = 0.5 * (1.0 + xi) * (1.0 - eta**2)
    shape_functions[valid_mask, 6] = 0.5 * (1.0 - xi**2) * (1.0 + eta)
    shape_functions[valid_mask, 7] = 0.5 * (1.0 - xi) * (1.0 - eta**2)

    # Compute derivatives if requested
    if return_derivatives:
        # Initialize derivatives array
        derivatives = numpy.full((n_points, 8, 2), default, dtype=numpy.float64)

        # Derivatives for valid coordinates
        derivatives[valid_mask, 0, 0] = 0.25 * (1.0 - eta) * (2.0 * xi + eta)
        derivatives[valid_mask, 0, 1] = 0.25 * (1.0 - xi) * (xi + 2.0 * eta)
        derivatives[valid_mask, 1, 0] = 0.25 * (1.0 - eta) * (2.0 * xi - eta)
        derivatives[valid_mask, 1, 1] = -0.25 * (1.0 + xi) * (xi - 2.0 * eta)
        derivatives[valid_mask, 2, 0] = 0.25 * (1.0 + eta) * (2.0 * xi + eta)
        derivatives[valid_mask, 2, 1] = 0.25 * (1.0 + xi) * (xi + 2.0 * eta)
        derivatives[valid_mask, 3, 0] = 0.25 * (1.0 + eta) * (2.0 * xi - eta)
        derivatives[valid_mask, 3, 1] = 0.25 * (1.0 - xi) * (-xi + 2.0 * eta)
        derivatives[valid_mask, 4, 0] = -xi * (1.0 - eta)
        derivatives[valid_mask, 4, 1] = -0.5 * (1.0 - xi**2)
        derivatives[valid_mask, 5, 0] = 0.5 * (1.0 - eta**2)
        derivatives[valid_mask, 5, 1] = -eta * (1.0 + xi)
        derivatives[valid_mask, 6, 0] = -xi * (1.0 + eta)
        derivatives[valid_mask, 6, 1] = 0.5 * (1.0 - xi**2)
        derivatives[valid_mask, 7, 0] = -0.5 * (1.0 - eta**2)
        derivatives[valid_mask, 7, 1] = -eta * (1.0 - xi)

        return shape_functions, derivatives
    
    return shape_functions





def shape_functions(
    natural_coordinates: numpy.ndarray,
    element_type: str,
    return_derivatives: bool = False,
    *,
    default: Real = 0.0
) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
    r"""
    Compute the shape functions for a given :obj:`element_type` at specified :obj:`natural_coordinates`.

    .. note::

        The input :obj:`natural_coordinates` will be converted to :obj:`numpy.float64` for computation.
        The output arrays will be :obj:`numpy.float64` as well.

        
    Parameters
    ----------
    natural_coordinates: :class:`numpy.ndarray`
        Natural coordinates where to evaluate the shape functions. The array must have shape :math:`(N_{p}, K)`,
        where :math:`N_{p}` is the number of points to evaluate and :math:`K` is the topological dimension of the element.

    element_type: :class:`str`
        Type of the finite element. Supported types are:
        
        - 'segment_2': 2-node segment element.
        - 'segment_3': 3-node segment element.
        - 'triangle_3': 3-node triangle element.
        - 'triangle_6': 6-node triangle element.
        - 'quadrangle_4': 4-node quadrangle element.
        - 'quadrangle_8': 8-node quadrangle element.

    return_derivatives: :class:`bool`, optional
        If :obj:`True`, the function will also return the first derivatives of the shape functions with respect to the natural coordinate.
        By default, :obj:`False`.

    default: :class:`Real`, optional
        Default value to assign to shape functions when the input natural coordinate is out of the valid range
        (i.e., not in the valid range for the specified element type). By default, :obj:`0.0`.

    
    Returns
    -------
    shape_functions: :class:`numpy.ndarray`
        Shape functions evaluated at the given natural coordinates. The returned array has shape :math:`(N_{p}, N_{vpe})`,
        where each row corresponds to a point and each column to a node, and :math:`N_{vpe}` is the number of vertices per element for the specified element type.

    shape_function_derivatives: :class:`numpy.ndarray`, optional
        If :obj:`return_derivatives` is :obj:`True`, the function also returns an array of the first derivatives of the shape functions
        with respect to the natural coordinate. The returned array has shape :math:`(N_{p}, N_{vpe}, K)`.

    
    Raises
    ------
    TypeError
        - If the input :obj:`natural_coordinates` cannot be converted to a numpy array.
        - If the input :obj:`element_type` is not a string.
        - If the input :obj:`return_derivatives` is not a boolean.
        - If the input :obj:`default` is not a real number.

    ValueError
        - If the input :obj:`natural_coordinates` does not have shape :math:`(N_{p}, K)`.
        - If the input :obj:`element_type` is not supported.


    See Also
    --------
    pysdic.segment_2_shape_functions:
        Shape functions for 2-node segment 1D-elements.

    pysdic.segment_3_shape_functions:
        Shape functions for 3-node segment 1D-elements.

    pysdic.triangle_3_shape_functions:
        Shape functions for 3-node triangle 2D-elements.

    pysdic.triangle_6_shape_functions:
        Shape functions for 6-node triangle 2D-elements.

    pysdic.quadrangle_4_shape_functions:
        Shape functions for 4-node quadrangle 2D-elements.
        
    pysdic.quadrangle_8_shape_functions:
        Shape functions for 8-node quadrangle 2D-elements.


    Examples
    --------
    Compute shape functions for a 6-node triangle element for 3 valid points and 1 invalid point:

    .. code-block:: python

        import numpy
        from pysdic import shape_functions

        coords = numpy.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.7, 0.5]])
        shape_funcs = shape_functions(coords, element_type='triangle_6')
        print("Shape function values for triangle_6:")
        print(shape_funcs)

    .. code-block:: console

        Shape function values for triangle_6:
        [[ 1.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  1.  0.  0.]
         [ 0.  0.  0.  0.  1.  0.]
         [ 0.  0.  0.  0.  0.  0.]]

    Compute shape functions for an 8-node quadrangle element with derivatives for 3 valid points and 1 invalid point:

    .. code-block:: python

        import numpy
        from pysdic import shape_functions

        coords = numpy.array([[-1.0, -1.0], [0.0, -1.0], [1.0, 0.0], [1.5, 0.5]])
        values, derivatives = shape_functions(coords, element_type='quadrangle_8', return_derivatives=True)
        print("Shape function values for quadrangle_8:")
        print(values)
        print("Shape function derivatives for quadrangle_8:")
        print(derivatives)

    .. code-block:: console

        Shape function values for quadrangle_8:
        [[ 1. -0. -0. -0.  0.  0.  0.  0.]
         [ 0.  0. -0. -0.  1.  0.  0.  0.]
         [-0.  0.  0. -0.  0.  1.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  0.]]
        Shape function derivatives for quadrangle_8:
        [[[-1.5 -1.5]
          [-0.5 -0. ]
          [-0.  -0. ]
          [-0.  -0.5]
          [ 2.  -0. ]
          [ 0.   0. ]
          [ 0.   0. ]
          [-0.   2. ]]

         [[-0.5 -0.5]
          [ 0.5 -0.5]
          [-0.  -0.5]
          [ 0.  -0.5]
          [-0.  -0.5]
          [ 0.   1. ]
          [-0.   0.5]
          [-0.   1. ]]

         [[ 0.5  0. ]
          [ 0.5 -0.5]
          [ 0.5  0.5]
          [ 0.5 -0. ]
          [-1.  -0. ]
          [ 0.5 -0. ]
          [-1.   0. ]
          [-0.5 -0. ]]

         [[ 0.   0. ]
          [ 0.   0. ]
          [ 0.   0. ]
          [ 0.   0. ]
          [ 0.   0. ]
          [ 0.   0. ]
          [ 0.   0. ]
          [ 0.   0. ]]]

    Set a different default value for out-of-range coordinates for a 3-node triangle element:

    .. code-block:: python

        import numpy
        from pysdic import shape_functions

        coords = numpy.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.7, 0.5]])
        shape_funcs = shape_functions(coords, element_type='triangle_3', default=numpy.nan)
        print("Shape function values for triangle_3 with custom default:")
        print(shape_funcs)

    .. code-block:: console

        Shape function values for triangle_3 with custom default:
        [[ 1.  0.  0.]
         [ 0.5 0.5 0.]
         [ 0.5 0.  0.5]
         [nan nan nan]]

    """
    # Dispatcher dictionary
    SHAPE_FUNCTIONS_DISPATCHER = {
        'segment_2': segment_2_shape_functions,
        'segment_3': segment_3_shape_functions,
        'triangle_3': triangle_3_shape_functions,
        'triangle_6': triangle_6_shape_functions,
        'quadrangle_4': quadrangle_4_shape_functions,
        'quadrangle_8': quadrangle_8_shape_functions
    }

    # Validate element_type
    if not isinstance(element_type, str):
        raise TypeError("The 'element_type' parameter must be a string.")
    element_type = element_type.lower()
    if not element_type in SHAPE_FUNCTIONS_DISPATCHER:
        raise ValueError(f"Unsupported element_type '{element_type}'. Supported types are: {list(SHAPE_FUNCTIONS_DISPATCHER.keys())}.")

    # Call the appropriate shape function
    shape_function_func = SHAPE_FUNCTIONS_DISPATCHER[element_type]
    return shape_function_func(
        natural_coordinates,
        return_derivatives=return_derivatives,
        default=default
    )



