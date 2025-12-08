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
from typing import Tuple, Union
from numbers import Real

import numpy

def segment_2_shape_functions(natural_coordinates: numpy.ndarray, return_derivatives: bool = False, *, default: Real = 0.0) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
    r"""
    Compute the shape functions for a 2-node segment for given natural coordinates :math:`\xi`.

    A 2-node segment represented in the figure below has the following shape functions:

    +----------+-----------------+-----------------------------------------+-----------------------------------------------------+
    | Node No. | (:math:`\xi`)   | Shape Function :math:`N`                | First Derivative (:math:`\frac{dN}{d\xi}`)          |
    +==========+=================+=========================================+=====================================================+
    | 1        | -1              | :math:`N_1(\xi) = \frac{1}{2}(1 - \xi)` | :math:`-\frac{1}{2}`                                |
    +----------+-----------------+-----------------------------------------+-----------------------------------------------------+
    | 2        | 1               | :math:`N_2(\xi) = \frac{1}{2}(1 + \xi)` | :math:`\frac{1}{2}`                                 |
    +----------+-----------------+-----------------------------------------+-----------------------------------------------------+

    .. figure:: /_static/shape_functions/segment_2.png
        :alt: 2-node segment element
        :align: center
        :width: 200px

    .. note::

        If the input :obj:`natural_coordinates` is not a numpy floating type, it will be converted to :obj:`numpy.float64`.

    .. seealso::

        - :func:`pysdic.segment_3_shape_functions` : Shape functions for 3-node segment 1D-elements.
        - :func:`pysdic.segment_2_gauss_points` : Gauss integration points for 2-node segment 1D-elements.

    Parameters
    ----------
    natural_coordinates: :class:`numpy.ndarray`
        Natural coordinates where to evaluate the shape functions. The array must have shape (n_points,) or (n_points, 1),
        where n_points is the number of points to evaluate.

    return_derivatives: :class:`bool`, optional
        If :obj:`True`, the function will also return the first derivatives of the shape functions with respect to the natural coordinate.
        By default, :obj:`False`.

    default: :class:`Real`, optional
        Default value to assign to shape functions when the input natural coordinate is out of the valid range
        (i.e., not in [-1, 1]). By default, :obj:`0.0`.


    Returns
    -------
    shape_functions: :class:`numpy.ndarray`
        Shape functions evaluated at the given natural coordinates. The returned array has shape (n_points, 2),
        where each row corresponds to a point and each column to a node.

    shape_function_derivatives: :class:`numpy.ndarray`, optional
        If :obj:`return_derivatives` is :obj:`True`, the function also returns an array of the first derivatives of the shape functions
        with respect to the natural coordinate. The returned array has shape (n_points, 2, 1).

    
    Raises
    ------
    TypeError
        If the input :obj:`natural_coordinates` cannot be converted to a numpy array.
        If the input :obj:`return_derivatives` is not a boolean.
        If the input :obj:`default` is not a real number.

    ValueError
        If the input :obj:`natural_coordinates` does not have shape (n_points, ) or (n_points, 1).

    
    Examples
    --------

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

    """
    # Convert coords to a 2D array if necessary
    coords = numpy.asarray(natural_coordinates)
    if coords.ndim == 1:
        coords = coords[:, numpy.newaxis]
    if not numpy.issubdtype(coords.dtype, numpy.floating):
        coords = coords.astype(numpy.float64)

    # Input validation
    if coords.ndim != 2 or coords.shape[1] != 1:
        raise ValueError("Input 'coords' for the natural coordinates must have shape (n_points,) or (n_points, 1) for 2-node segment elements.")
    if not isinstance(return_derivatives, bool):
        raise ValueError("Input 'return_derivatives' must be a boolean value.")
    if not isinstance(default, Real):
        raise ValueError("Input 'default' must be a real number.")
    
    # Number of points
    n_points = coords.shape[0]
    
    # Initialize shape functions array
    shape_functions = numpy.full((n_points, 2), default, dtype=coords.dtype)

    # Valid range mask
    valid_mask = (coords >= -1.0) & (coords <= 1.0)
    xi = coords[valid_mask[:, 0], 0]

    # Compute shape functions for valid coordinates
    shape_functions[valid_mask[:, 0], 0] = 0.5 * (1.0 - xi)
    shape_functions[valid_mask[:, 0], 1] = 0.5 * (1.0 + xi)

    # Compute derivatives if requested
    if return_derivatives:
        # Initialize derivatives array
        derivatives = numpy.full((n_points, 2, 1), default, dtype=coords.dtype)

        # Derivatives are constant for valid coordinates
        derivatives[valid_mask[:, 0], 0, 0] = -0.5
        derivatives[valid_mask[:, 0], 1, 0] = 0.5

        return shape_functions, derivatives

    return shape_functions



        

def segment_3_shape_functions(natural_coordinates: numpy.ndarray, return_derivatives: bool = False, *, default: Real = 0.0) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
    r"""
    Compute the shape functions for a 3-node segment for given natural coordinates :math:`\xi`.

    A 3-node segment represented in the figure below has the following shape functions:

    +----------+-----------------+-----------------------------------------------+---------------------------------------------------------+
    | Node No. | (:math:`\xi`)   | Shape Function :math:`N`                      | First Derivative (:math:`\frac{dN}{d\xi}`)              |
    +==========+=================+===============================================+=========================================================+
    | 1        | -1              | :math:`N_1(\xi) = \frac{1}{2}\xi(\xi - 1)`    | :math:`\xi - \frac{1}{2}`                               |
    +----------+-----------------+-----------------------------------------------+---------------------------------------------------------+
    | 2        | 1               | :math:`N_2(\xi) = \frac{1}{2}\xi(\xi + 1)`    | :math:`\xi + \frac{1}{2}`                               |
    +----------+-----------------+-----------------------------------------------+---------------------------------------------------------+
    | 3        | 0               | :math:`N_3(\xi) = 1 - \xi^2`                  | :math:`-2\xi`                                           |
    +----------+-----------------+-----------------------------------------------+---------------------------------------------------------+

    .. figure:: /_static/shape_functions/segment_3.png
        :alt: 3-node segment element
        :align: center
        :width: 200px

    .. note::

        If the input :obj:`natural_coordinates` is not a numpy floating type, it will be converted to :obj:`numpy.float64`.

    .. seealso::

        - :func:`pysdic.segment_2_shape_functions` : Shape functions for 2-node segment 1D-elements.
        - :func:`pysdic.segment_3_gauss_points` : Gauss integration points for 3-node segment 1D-elements.

    Parameters
    ----------
    natural_coordinates: :class:`numpy.ndarray`
        Natural coordinates where to evaluate the shape functions. The array must have shape (n_points,)
        or (n_points, 1), where n_points is the number of points to evaluate.

    return_derivatives: :class:`bool`, optional
        If :obj:`True`, the function will also return the first derivatives of the shape functions with respect to the natural coordinate.
        By default, :obj:`False`.

    default: :class:`numbers.Real`, optional
        Default value to assign to shape functions when the input natural coordinate is out of the valid range
        (i.e., not in [-1, 1]). By default, :obj:`:0.0`.

    
    Returns
    -------
    shape_functions: :class:`numpy.ndarray`
        Shape functions evaluated at the given natural coordinates. The returned array has shape (n_points, 3),
        where each row corresponds to a point and each column to a node.

    shape_function_derivatives: :class:`numpy.ndarray`, optional
        If :obj:`return_derivatives` is :obj:`True`, the function also returns an array of the first derivatives of the shape functions
        with respect to the natural coordinate. The returned array has shape (n_points, 3, 1).


    Raises
    ------
    TypeError
        If the input :obj:`natural_coordinates` cannot be converted to a numpy array.
        If the input :obj:`return_derivatives` is not a boolean.
        If the input :obj:`default` is not a real number.

    ValueError
        If the input :obj:`natural_coordinates` does not have shape (n_points, ) or (n_points, 1).


    Examples
    --------

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

    """
    # Convert coords to a 2D array if necessary
    coords = numpy.asarray(natural_coordinates)
    if coords.ndim == 1:
        coords = coords[:, numpy.newaxis]
    if not numpy.issubdtype(coords.dtype, numpy.floating):
        coords = coords.astype(numpy.float64)

    # Input validation
    if coords.ndim != 2 or coords.shape[1] != 1:
        raise ValueError("Input 'coords' for the natural coordinates must have shape (n_points,) or (n_points, 1) for 3-node segment elements.")
    if not isinstance(return_derivatives, bool):
        raise ValueError("Input 'return_derivatives' must be a boolean value.")
    if not isinstance(default, Real):
        raise ValueError("Input 'default' must be a real number.")
    
    # Number of points
    n_points = coords.shape[0]
    
    # Initialize shape functions array
    shape_functions = numpy.full((n_points, 3), default, dtype=coords.dtype)

    # Valid range mask
    valid_mask = (coords >= -1.0) & (coords <= 1.0)
    xi = coords[valid_mask[:, 0], 0]

    # Compute shape functions for valid coordinates
    shape_functions[valid_mask[:, 0], 0] = 0.5 * xi * (xi - 1.0)
    shape_functions[valid_mask[:, 0], 1] = 0.5 * xi * (xi + 1.0)
    shape_functions[valid_mask[:, 0], 2] = 1.0 - xi**2

    # Compute derivatives if requested
    if return_derivatives: 
        # Initialize derivatives array
        derivatives = numpy.full((n_points, 3, 1), default, dtype=coords.dtype)

        # Derivatives for valid coordinates
        derivatives[valid_mask[:, 0], 0, 0] = xi - 0.5
        derivatives[valid_mask[:, 0], 1, 0] = xi + 0.5
        derivatives[valid_mask[:, 0], 2, 0] = -2.0 * xi

        return shape_functions, derivatives
    
    return shape_functions



def triangle_3_shape_functions(natural_coordinates: numpy.ndarray, return_derivatives: bool = False, *, default: Real = 0.0) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
    r"""
    Compute the shape functions for a 3-node triangle for given natural coordinates :math:`(\xi, \eta)`.

    A 3-node triangle represented in the figure below has the following shape functions:

    +----------+-----------------------+-----------------------------------------------+--------------------------------------------------------------+
    | Node No. | (:math:`\xi, \eta`)   | Shape Function :math:`N`                      | First Derivative (:math:`\frac{dN}{d\xi}, \frac{dN}{d\eta}`) |
    +==========+=======================+===============================================+==============================================================+
    | 1        | (0, 0)                | :math:`N_1(\xi, \eta) = 1 - \xi - \eta`       | :math:`(-1, -1)`                                             |
    +----------+-----------------------+-----------------------------------------------+--------------------------------------------------------------+
    | 2        | (1, 0)                | :math:`N_2(\xi, \eta) = \xi`                  | :math:`(1, 0)`                                               |
    +----------+-----------------------+-----------------------------------------------+--------------------------------------------------------------+
    | 3        | (0, 1)                | :math:`N_3(\xi, \eta) = \eta`                 | :math:`(0, 1)`                                               |
    +----------+-----------------------+-----------------------------------------------+--------------------------------------------------------------+

    .. figure:: /_static/shape_functions/triangle_3.png
        :alt: 3-node triangle element
        :align: center
        :width: 200px

    .. note::

        If the input :obj:`natural_coordinates` are not numpy floating type, they will be converted to :obj:`numpy.float64`.

    .. seealso::

        - :func:`pysdic.triangle_6_shape_functions` : Shape functions for 6-node triangle 2D-elements.
        - :func:`pysdic.quadrangle_4_shape_functions` : Shape functions for 4-node quadrangle 2D-elements.
        - :func:`pysdic.quadrangle_8_shape_functions` : Shape functions for 8-node quadrangle 2D-elements.
        - :func:`pysdic.triangle_3_gauss_points` : Gauss integration points for 3-node triangle 2D-elements.

    Parameters
    ----------
    natural_coordinates: :class:`numpy.ndarray`
        Natural coordinates where to evaluate the shape functions. The array must have shape (n_points, 2),
        where n_points is the number of points to evaluate.

    return_derivatives: :class:`bool`, optional
        If :obj:`True`, the function will also return the first derivatives of the shape functions with respect to the natural coordinates.
        By default, :obj:`False`.

    default: :class:`numbers.Real`, optional
        Default value to assign to shape functions when the input natural coordinates are out of the valid range
        (i.e., not in the triangle defined by (0,0), (1,0), (0,1)). By default, :obj:`0.0`.

    
    Returns
    -------
    shape_functions: :class:`numpy.ndarray`
        Shape functions evaluated at the given natural coordinates. The returned array has shape (n_points, 3),
        where each row corresponds to a point and each column to a node.

    shape_function_derivatives: :class:`numpy.ndarray`, optional
        If :obj:`return_derivatives` is :obj:`True`, the function also returns an array of the first derivatives of the shape functions
        with respect to the natural coordinates. The returned array has shape (n_points, 3, 2).

    
    Raises
    ------
    TypeError
        If the input :obj:`natural_coordinates` cannot be converted to a numpy array.
        If the input :obj:`return_derivatives` is not a boolean.
        If the input :obj:`default` is not a real number.

    ValueError
        If the input :obj:`natural_coordinates` does not have shape (n_points, 2).

    
    Examples
    --------

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

    """
    # Convert coords to a 2D array if necessary
    coords = numpy.asarray(natural_coordinates)
    if not numpy.issubdtype(coords.dtype, numpy.floating):
        coords = coords.astype(numpy.float64)

    # Input validation
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("Input 'coords' for the natural coordinates must have shape (n_points, 2) for 3-node triangle elements.")
    if not isinstance(return_derivatives, bool):
        raise ValueError("Input 'return_derivatives' must be a boolean value.")
    if not isinstance(default, Real):
        raise ValueError("Input 'default' must be a real number.")
    
    # Number of points
    n_points = coords.shape[0]
    
    # Initialize shape functions array
    shape_functions = numpy.full((n_points, 3), default, dtype=coords.dtype)

    # Valid range mask
    valid_mask = (coords[:, 0] >= 0.0) & (coords[:, 1] >= 0.0) & ((coords[:, 0] + coords[:, 1]) <= 1.0)
    xi = coords[valid_mask, 0]
    eta = coords[valid_mask, 1]

    # Compute shape functions for valid coordinates
    shape_functions[valid_mask, 0] = 1.0 - xi - eta
    shape_functions[valid_mask, 1] = xi
    shape_functions[valid_mask, 2] = eta

    # Compute derivatives if requested
    if return_derivatives:
        # Initialize derivatives array
        derivatives = numpy.full((n_points, 3, 2), default, dtype=coords.dtype)

        # Derivatives are constant for valid coordinates
        derivatives[valid_mask, 0, 0] = -1.0
        derivatives[valid_mask, 0, 1] = -1.0
        derivatives[valid_mask, 1, 0] = 1.0
        derivatives[valid_mask, 1, 1] = 0.0
        derivatives[valid_mask, 2, 0] = 0.0
        derivatives[valid_mask, 2, 1] = 1.0

        return shape_functions, derivatives
    
    return shape_functions




def triangle_6_shape_functions(natural_coordinates: numpy.ndarray, return_derivatives: bool = False, *, default: Real = 0.0) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
    r"""
    Compute the shape functions for a 6-node triangle for given natural coordinates :math:`(\xi, \eta)`.

    A 6-node triangle represented in the figure below has the following shape functions:

    +----------+-----------------------+--------------------------------------------------------------------------------------+--------------------------------------------------------------------+
    | Node No. | (:math:`\xi, \eta`)   | Shape Function :math:`N`                                                             | First Derivative (:math:`\frac{dN}{d\xi}, \frac{dN}{d\eta}`)       |
    +==========+=======================+======================================================================================+====================================================================+
    | 1        | (0, 0)                | :math:`N_1(\xi, \eta) = (1 - \xi - \eta)(1 - 2\xi - 2\eta)`                          | :math:`(-3 + 4\xi + 4\eta, -3 + 4\xi + 4\eta)`                     |
    +----------+-----------------------+--------------------------------------------------------------------------------------+--------------------------------------------------------------------+
    | 2        | (1, 0)                | :math:`N_2(\xi, \eta) = \xi(2\xi - 1)`                                               | :math:`(4\xi - 1, 0)`                                              |
    +----------+-----------------------+--------------------------------------------------------------------------------------+--------------------------------------------------------------------+
    | 3        | (0, 1)                | :math:`N_3(\xi, \eta) = \eta(2\eta - 1)`                                             | :math:`(0, 4\eta - 1)`                                             |
    +----------+-----------------------+--------------------------------------------------------------------------------------+--------------------------------------------------------------------+
    | 4        | (0.5, 0)              | :math:`N_4(\xi, \eta) = 4\xi(1 - \xi - \eta)`                                        | :math:`(4 - 8\xi - 4\eta, -4\xi)`                                  |
    +----------+-----------------------+--------------------------------------------------------------------------------------+--------------------------------------------------------------------+
    | 5        | (0.5, 0.5)            | :math:`N_5(\xi, \eta) = 4\xi\eta`                                                    | :math:`(4\eta, 4\xi)`                                              |
    +----------+-----------------------+--------------------------------------------------------------------------------------+--------------------------------------------------------------------+
    | 6        | (0, 0.5)              | :math:`N_6(\xi, \eta) = 4\eta(1 - \xi - \eta)`                                       | :math:`(-4\eta, 4 - 4\xi - 8\eta)`                                 |
    +----------+-----------------------+--------------------------------------------------------------------------------------+--------------------------------------------------------------------+

    .. figure:: /_static/shape_functions/triangle_6.png
        :alt: 6-node triangle element
        :align: center
        :width: 200px

    .. note::

        If the input :obj:`natural_coordinates` are not numpy floating type, they will be converted to :obj:`numpy.float64`.

    .. seealso::

        - :func:`pysdic.triangle_3_shape_functions` : Shape functions for 3-node triangle 2D-elements.
        - :func:`pysdic.quadrangle_4_shape_functions` : Shape functions for 4-node quadrangle 2D-elements.
        - :func:`pysdic.quadrangle_8_shape_functions` : Shape functions for 8-node quadrangle 2D-elements.
        - :func:`pysdic.triangle_6_gauss_points` : Gauss integration points for 6-node triangle 2D-elements.

    Parameters
    ----------
    natural_coordinates: :class:`numpy.ndarray`
        Natural coordinates where to evaluate the shape functions. The array must have shape (n_points, 2),
        where n_points is the number of points to evaluate.

    return_derivatives: :class:`bool`, optional
        If :obj:`True`, the function will also return the first derivatives of the shape functions with respect to the natural coordinates.
        By default, :obj:`False`.

    default: :class:`numbers.Real`, optional
        Default value to assign to shape functions when the input natural coordinates are out of the valid range
        (i.e., not in the triangle defined by (0,0), (1,0), (0,1)). By default, :obj:`0.0`.

    
    Returns
    -------
    shape_functions: :class:`numpy.ndarray`
        Shape functions evaluated at the given natural coordinates. The returned array has shape (n_points, 6),
        where each row corresponds to a point and each column to a node.

    shape_function_derivatives: :class:`numpy.ndarray`, optional
        If :obj:`return_derivatives` is :obj:`True`, the function also returns an array of the first derivatives of the shape functions
        with respect to the natural coordinates. The returned array has shape (n_points, 6, 2).

    
    Raises
    ------
    TypeError
        If the input :obj:`natural_coordinates` cannot be converted to a numpy array.
        If the input :obj:`return_derivatives` is not a boolean.
        If the input :obj:`default` is not a real number.

    ValueError
        If the input :obj:`natural_coordinates` does not have shape (n_points, 2).

    
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

    """
    # Convert coords to a 2D array if necessary
    coords = numpy.asarray(natural_coordinates)
    if not numpy.issubdtype(coords.dtype, numpy.floating):
        coords = coords.astype(numpy.float64)

    # Input validation
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("Input 'coords' for the natural coordinates must have shape (n_points, 2) for 6-node triangle elements.")
    if not isinstance(return_derivatives, bool):
        raise ValueError("Input 'return_derivatives' must be a boolean value.")
    if not isinstance(default, Real):
        raise ValueError("Input 'default' must be a real number.")
    
    # Number of points
    n_points = coords.shape[0]

    # Initialize shape functions array
    shape_functions = numpy.full((n_points, 6), default, dtype=coords.dtype)

    # Valid range mask
    valid_mask = (coords[:, 0] >= 0.0) & (coords[:, 1] >= 0.0) & ((coords[:, 0] + coords[:, 1]) <= 1.0)
    xi = coords[valid_mask, 0]
    eta = coords[valid_mask, 1]

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
        derivatives = numpy.full((n_points, 6, 2), default, dtype=coords.dtype)

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




def quadrangle_4_shape_functions(natural_coordinates: numpy.ndarray, return_derivatives: bool = False, *, default: Real = 0.0) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
    r"""
    Compute the shape functions for a 4-node quadrangle for given natural coordinates :math:`(\xi, \eta)`.

    A 4-node quadrangle represented in the figure below has the following shape functions:

    +----------+-----------------------+---------------------------------------------------------------+--------------------------------------------------------------------+
    | Node No. | (:math:`\xi, \eta`)   | Shape Function :math:`N`                                      | First Derivative (:math:`\frac{dN}{d\xi}, \frac{dN}{d\eta}`)       |
    +==========+=======================+===============================================================+====================================================================+
    | 1        | (-1, -1)              | :math:`N_1(\xi, \eta) = \frac{1}{4}(1 - \xi)(1 - \eta)`       | :math:`(\frac{1}{4}(\eta - 1), \frac{1}{4}(\xi - 1))`              |
    +----------+-----------------------+---------------------------------------------------------------+--------------------------------------------------------------------+
    | 2        | (1, -1)               | :math:`N_2(\xi, \eta) = \frac{1}{4}(1 + \xi)(1 - \eta)`       | :math:`(\frac{1}{4}(1 - \eta), -\frac{1}{4}(1 + \xi))`             |
    +----------+-----------------------+---------------------------------------------------------------+--------------------------------------------------------------------+
    | 3        | (1, 1)                | :math:`N_3(\xi, \eta) = \frac{1}{4}(1 + \xi)(1 + \eta)`       | :math:`(\frac{1}{4}(1 + \eta), \frac{1}{4}(1 + \xi))`              |
    +----------+-----------------------+---------------------------------------------------------------+--------------------------------------------------------------------+
    | 4        | (-1, 1)               | :math:`N_4(\xi, \eta) = \frac{1}{4}(1 - \xi)(1 + \eta)`       | :math:`(-\frac{1}{4}(1 + \eta), \frac{1}{4}(1 - \xi))`             |
    +----------+-----------------------+---------------------------------------------------------------+--------------------------------------------------------------------+

    .. figure:: /_static/shape_functions/quadrangle_4.png
        :alt: 4-node quadrangle element
        :align: center
        :width: 200px

    .. note:: 

        If the input :obj:`natural_coordinates` are not numpy floating type, they will be converted to :obj:`numpy.float64`.

    .. seealso::

        - :func:`pysdic.triangle_3_shape_functions` : Shape functions for 3-node triangle 2D-elements.
        - :func:`pysdic.triangle_6_shape_functions` : Shape functions for 6-node triangle 2D-elements.
        - :func:`pysdic.quadrangle_8_shape_functions` : Shape functions for 8-node quadrangle 2D-elements.
        - :func:`pysdic.quadrangle_4_gauss_points` : Gauss integration points for 4-node quadrangle 2D-elements.

    Parameters
    ----------
    natural_coordinates: :class:`numpy.ndarray`
        Natural coordinates where to evaluate the shape functions. The array must have shape (n_points, 2),
        where n_points is the number of points to evaluate.

    return_derivatives: :class:`bool`, optional
        If :obj:`True`, the function will also return the first derivatives of the shape functions with respect to the natural coordinates.
        By default, :obj:`False`.

    default: :class:`numbers.Real`, optional
        Default value to assign to shape functions when the input natural coordinates are out of the valid range
        (i.e., not in the square defined by (-1,-1), (1,-1), (1,1), (-1,1)). By default, :obj:`0.0`.

        
    Returns
    -------
    shape_functions: :class:`numpy.ndarray`
        Shape functions evaluated at the given natural coordinates. The returned array has shape (n_points, 4),
        where each row corresponds to a point and each column to a node.

    shape_function_derivatives: :class:`numpy.ndarray`, optional
        If :obj:`return_derivatives` is :obj:`True`, the function also returns an array of the first derivatives of the shape functions
        with respect to the natural coordinates. The returned array has shape (n_points, 4, 2).

    
    Raises
    ------
    TypeError
        If the input :obj:`natural_coordinates` cannot be converted to a numpy array.
        If the input :obj:`return_derivatives` is not a boolean.
        If the input :obj:`default` is not a real number.

    ValueError
        If the input :obj:`natural_coordinates` does not have shape (n_points, 2).


    Examples
    --------

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

    """
    # Convert coords to a 2D array if necessary
    coords = numpy.asarray(natural_coordinates)
    if not numpy.issubdtype(coords.dtype, numpy.floating):
        coords = coords.astype(numpy.float64)

    # Input validation
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("Input 'coords' for the natural coordinates must have shape (n_points, 2) for 4-node quadrangle elements.")
    if not isinstance(return_derivatives, bool):
        raise ValueError("Input 'return_derivatives' must be a boolean value.")
    if not isinstance(default, Real):
        raise ValueError("Input 'default' must be a real number.")
    
    # Number of points
    n_points = coords.shape[0]
    
    # Initialize shape functions array
    shape_functions = numpy.full((n_points, 4), default, dtype=coords.dtype)

    # Valid range mask
    valid_mask = (coords[:, 0] >= -1.0) & (coords[:, 0] <= 1.0) & (coords[:, 1] >= -1.0) & (coords[:, 1] <= 1.0)
    xi = coords[valid_mask, 0]
    eta = coords[valid_mask, 1]

    # Compute shape functions for valid coordinates
    shape_functions[valid_mask, 0] = 0.25 * (1.0 - xi) * (1.0 - eta)
    shape_functions[valid_mask, 1] = 0.25 * (1.0 + xi) * (1.0 - eta)
    shape_functions[valid_mask, 2] = 0.25 * (1.0 + xi) * (1.0 + eta)
    shape_functions[valid_mask, 3] = 0.25 * (1.0 - xi) * (1.0 + eta)

    # Compute derivatives if requested
    if return_derivatives:
        # Initialize derivatives array
        derivatives = numpy.full((n_points, 4, 2), default, dtype=coords.dtype)

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




def quadrangle_8_shape_functions(natural_coordinates: numpy.ndarray, return_derivatives: bool = False, *, default: Real = 0.0) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
    r"""
    Compute the shape functions for a 8-node quadrangle for given natural coordinates :math:`(\xi, \eta)`.

    An 8-node quadrangle represented in the figure below has the following shape functions:

    +----------+-----------------------+--------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------+
    | Node No. | (:math:`\xi, \eta`)   | Shape Function :math:`N`                                                                   | First Derivative (:math:`\frac{dN}{d\xi}, \frac{dN}{d\eta}`)                                 |
    +==========+=======================+============================================================================================+==============================================================================================+
    | 1        | (-1, -1)              | :math:`N_1(\xi, \eta) = \frac{1}{4}(1 - \xi)(1 - \eta)(- \xi - \eta - 1)`                  | :math:`(\frac{1}{4}(1 - \eta)(2\xi + \eta), \frac{1}{4}(1 - \xi)(\xi + 2\eta))`              |
    +----------+-----------------------+--------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------+
    | 2        | (1, -1)               | :math:`N_2(\xi, \eta) = \frac{1}{4}(1 + \xi)(1 - \eta)(\xi - \eta - 1)`                    | :math:`(\frac{1}{4}(1 - \eta)(2\xi - \eta), -\frac{1}{4}(1 + \xi)(\xi - 2\eta))`             |
    +----------+-----------------------+--------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------+
    | 3        | (1, 1)                | :math:`N_3(\xi, \eta) = \frac{1}{4}(1 + \xi)(1 + \eta)(\xi + \eta - 1)`                    | :math:`(\frac{1}{4}(1 + \eta)(2\xi + \eta), \frac{1}{4}(1 + \xi)(\xi + 2\eta))`              |
    +----------+-----------------------+--------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------+
    | 4        | (-1, 1)               | :math:`N_4(\xi, \eta) = \frac{1}{4}(1 - \xi)(1 + \eta)(- \xi + \eta - 1)`                  | :math:`(\frac{1}{4}(1 + \eta)(2\xi - \eta), \frac{1}{4}(1 - \xi)(- \xi + 2\eta))`            |
    +----------+-----------------------+--------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------+
    | 5        | (0, -1)               | :math:`N_5(\xi, \eta) = \frac{1}{2}(1 - \xi^2)(1 - \eta)`                                  | :math:`(-\xi(1 - \eta), -\frac{1}{2}(1 - \xi^2))`                                            |
    +----------+-----------------------+--------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------+
    | 6        | (1, 0)                | :math:`N_6(\xi, \eta) = \frac{1}{2}(1 + \xi)(1 - \eta^2)`                                  | :math:`(\frac{1}{2}(1 - \eta^2), -\eta(1 + \xi))`                                            |
    +----------+-----------------------+--------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------+
    | 7        | (0, 1)                | :math:`N_7(\xi, \eta) = \frac{1}{2}(1 - \xi^2)(1 + \eta)`                                  | :math:`(-\xi(1 + \eta), \frac{1}{2}(1 - \xi^2))`                                             |
    +----------+-----------------------+--------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------+
    | 8        | (-1, 0)               | :math:`N_8(\xi, \eta) = \frac{1}{2}(1 - \xi)(1 - \eta^2)`                                  | :math:`(-\frac{1}{2}(1 - \eta^2), -\eta(1 - \xi))`                                           |
    +----------+-----------------------+--------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------+

    .. figure:: /_static/shape_functions/quadrangle_8.png
        :alt: 8-node quadrangle element
        :align: center
        :width: 200px

    .. note::

        If the input :obj:`natural_coordinates` are not numpy floating type, they will be converted to :obj:`numpy.float64`.

    .. seealso::

        - :func:`pysdic.triangle_3_shape_functions` : Shape functions for 3-node triangle 2D-elements.
        - :func:`pysdic.triangle_6_shape_functions` : Shape functions for 6-node triangle 2D-elements.
        - :func:`pysdic.quadrangle_4_shape_functions` : Shape functions for 4-node quadrangle 2D-elements.
        - :func:`pysdic.quadrangle_8_gauss_points` : Gauss integration points for 8-node quadrangle 2D-elements.

    Parameters
    ----------
    natural_coordinates: :class:`numpy.ndarray`
        Natural coordinates where to evaluate the shape functions. The array must have shape (n_points, 2),
        where n_points is the number of points to evaluate.

    return_derivatives: :class:`bool`, optional
        If :obj:`True`, the function will also return the first derivatives of the shape functions with respect to the natural coordinates.
        By default, :obj:`False`.

    default: :class:`numbers.Real`, optional
        Default value to assign to shape functions when the input natural coordinates are out of the valid range
        (i.e., not in the square defined by (-1,-1), (1,-1), (1,1), (-1,1)). By default, :obj:`0.0`.

    
    Returns
    -------
    shape_functions: :class:`numpy.ndarray`
        Shape functions evaluated at the given natural coordinates. The returned array has shape (n_points, 8),
        where each row corresponds to a point and each column to a node.

    shape_function_derivatives: :class:`numpy.ndarray`, optional
        If :obj:`return_derivatives` is :obj:`True`, the function also returns an array of the first derivatives of the shape functions
        with respect to the natural coordinates. The returned array has shape (n_points, 8, 2).

    
    Raises
    ------
    TypeError
        If the input :obj:`natural_coordinates` cannot be converted to a numpy array.
        If the input :obj:`return_derivatives` is not a boolean.
        If the input :obj:`default` is not a real number.

    ValueError
        If the input :obj:`natural_coordinates` does not have shape (n_points, 2).

    
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
    # Convert coords to a 2D array if necessary
    coords = numpy.asarray(natural_coordinates)
    if not numpy.issubdtype(coords.dtype, numpy.floating):
        coords = coords.astype(numpy.float64)

    # Input validation
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("Input 'coords' for the natural coordinates must have shape (n_points, 2) for 8-node quadrangle elements.")
    if not isinstance(return_derivatives, bool):
        raise ValueError("Input 'return_derivatives' must be a boolean value.")
    if not isinstance(default, Real):
        raise ValueError("Input 'default' must be a real number.")

    # Number of points
    n_points = coords.shape[0]

    # Initialize shape functions array
    shape_functions = numpy.full((n_points, 8), default, dtype=coords.dtype)

    # Valid range mask
    valid_mask = (coords[:, 0] >= -1.0) & (coords[:, 0] <= 1.0) & (coords[:, 1] >= -1.0) & (coords[:, 1] <= 1.0)
    xi = coords[valid_mask, 0]
    eta = coords[valid_mask, 1]

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
        derivatives = numpy.full((n_points, 8, 2), default, dtype=coords.dtype)

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

