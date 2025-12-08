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

import numpy


def segment_2_gauss_points(return_weights: bool = False) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
    r"""
    Get the natural coordinates :math:`\xi` and weights of Gauss quadrature points for a 2-node segment element.

    A 2-node segment element uses the following Gauss points and weights for numerical integration:

    +------------+-----------------+-----------------+
    | Point No.  | (:math:`\xi`)   | Weight :math:`w`|
    +============+=================+=================+
    | 1          | :math:`0`       | :math:`2`       |
    +------------+-----------------+-----------------+

    .. figure:: /_static/shape_functions/segment_2_gp.png
        :alt: Gauss quadrature points for 2-node segment element
        :align: center
        :width: 200px

    .. seealso::

        - :func:`pysdic.segment_2_shape_functions` : Shape functions for 2-node segment 1D-elements.

    Parameters
    ----------
    return_weights: :class:`bool`, optional
        If :obj:`True`, the function will also return the weights associated with each Gauss point. Default is :obj:`False`.

    Returns
    -------
    gauss_points: :class:`numpy.ndarray`
        Natural coordinates of the Gauss points. The returned array has shape (1, 1).

    weights: :class:`numpy.ndarray`, optional
        If :obj:`return_weights` is :obj:`True`, the function also returns an array of weights with shape (1,) associated with each Gauss point.

    Raises
    ------
    TypeError
        If the input :obj:`return_weights` is not a boolean.

    
    Examples
    --------

    .. code-block:: python

        import numpy
        from pysdic import segment_2_gauss_points

        gauss_points = segment_2_gauss_points()
        print("Gauss points:")
        print(gauss_points)

    .. code-block:: console

        Gauss points:
        [[0.0]]

    """
    # Input validation
    if not isinstance(return_weights, bool):
        raise ValueError("Input 'return_weights' must be a boolean value.")

    # Gauss points and weights for Gauss quadrature
    gauss_points = numpy.array([[0.0]], dtype=numpy.float64)
    weights = numpy.array([2.0], dtype=numpy.float64)

    if return_weights:
        return gauss_points, weights
    return gauss_points



def segment_3_gauss_points(return_weights: bool = False) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
    r"""
    Get the natural coordinates :math:`\xi` and weights of Gauss quadrature points for a 3-node segment element.

    A 3-node segment element uses the following Gauss points and weights for numerical integration:

    +------------+-----------------------------+-----------------+
    | Point No.  | (:math:`\xi`)               | Weight :math:`w`|
    +============+=============================+=================+
    | 1          | :math:`-\frac{1}{\sqrt{3}}` | :math:`1`       |
    +------------+-----------------------------+-----------------+
    | 2          | :math:`\frac{1}{\sqrt{3}}`  | :math:`1`       |
    +------------+-----------------------------+-----------------+

    
    .. figure:: /_static/shape_functions/segment_3_gp.png
        :alt: Gauss quadrature points for 3-node segment element
        :align: center
        :width: 200px


    .. seealso::

        - :func:`pysdic.segment_3_shape_functions` : Shape functions for 3-node segment 1D-elements.

        

    Parameters
    ----------
    return_weights: :class:`bool`, optional
        If :obj:`True`, the function will also return the weights associated with each Gauss point. Default is :obj:`False`.

    Returns
    -------
    gauss_points: :class:`numpy.ndarray`
        Natural coordinates of the Gauss points. The returned array has shape (2, 1).

    weights: :class:`numpy.ndarray`, optional
        If :obj:`return_weights` is :obj:`True`, the function also returns an array of weights with shape (2,) associated with each Gauss point.

    Raises
    ------
    TypeError
        If the input :obj:`return_weights` is not a boolean.

    
    Examples
    --------
    .. code-block:: python

        import numpy
        from pysdic import segment_3_gauss_points

        gauss_points = segment_3_gauss_points()
        print("Gauss points:")
        print(gauss_points)

    .. code-block:: console

        Gauss points:
        [[-0.57735027]
         [ 0.57735027]]

    """
    # Input validation
    if not isinstance(return_weights, bool):
        raise ValueError("Input 'return_weights' must be a boolean value.")

    # Gauss points and weights Gauss quadrature
    gauss_points = numpy.array([[-1.0 / numpy.sqrt(3)], [1.0 / numpy.sqrt(3)]], dtype=numpy.float64)
    weights = numpy.array([1.0, 1.0], dtype=numpy.float64)

    if return_weights:
        return gauss_points, weights
    
    return gauss_points



def triangle_3_gauss_points(return_weights: bool = False) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
    r"""
    Get the natural coordinates :math:`(\xi, \eta)` and weights of Gauss quadrature points for a 3-node triangle element.

    A 3-node triangle element uses the following Gauss points and weights for numerical integration:

    +------------+-----------------------------------------+---------------------+
    | Point No.  | (:math:`\xi`, :math:`\eta`)             | Weight :math:`w`    |
    +============+=========================================+=====================+
    | 1          | :math:`(\frac{1}{3}, \frac{1}{3})`      | :math:`\frac{1}{2}` |
    +------------+-----------------------------------------+---------------------+

    .. figure:: /_static/shape_functions/triangle_3_gp.png
        :alt: Gauss quadrature points for 3-node triangle element
        :align: center
        :width: 200px

    .. seealso::

        - :func:`pysdic.triangle_3_shape_functions` : Shape functions for 3-node triangle 2D-elements.

    Parameters
    ----------
    return_weights: :class:`bool`, optional
        If :obj:`True`, the function will also return the weights associated with each Gauss point. Default is :obj:`False`.

    Returns
    -------
    gauss_points: :class:`numpy.ndarray`
        Natural coordinates of the Gauss points. The returned array has shape (3, 2).

    weights: :class:`numpy.ndarray`, optional
        If :obj:`return_weights` is :obj:`True`, the function also returns an array of weights with shape (3,) associated with each Gauss point.

    Raises
    ------
    TypeError
        If the input :obj:`return_weights` is not a boolean.

    
    Examples
    --------

    .. code-block:: python

        import numpy
        from pysdic import triangle_3_gauss_points

        gauss_points = triangle_3_gauss_points()
        print("Gauss points:")
        print(gauss_points)

    .. code-block:: console

        Gauss points:
        [[0.33333333 0.33333333]]

    """
    # Input validation
    if not isinstance(return_weights, bool):
        raise ValueError("Input 'return_weights' must be a boolean value.")

    # Gauss points and weights for Gauss quadrature
    gauss_points = numpy.array([[1.0 / 3.0, 1.0 / 3.0]], dtype=numpy.float64)
    weights = numpy.array([0.5], dtype=numpy.float64)

    if return_weights:
        return gauss_points, weights
    
    return gauss_points


def triangle_6_gauss_points(return_weights: bool = False) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
    r"""
    Get the natural coordinates :math:`(\xi, \eta)` and weights of Gauss quadrature points for a 6-node triangle element.

    A 6-node triangle element uses the following Gauss points and weights for numerical integration:

    +------------+-----------------------------------------+---------------------+
    | Point No.  | (:math:`\xi`, :math:`\eta`)             | Weight :math:`w`    |
    +============+=========================================+=====================+
    | 1          | :math:`(\frac{1}{6}, \frac{1}{6})`      | :math:`\frac{1}{6}` |
    +------------+-----------------------------------------+---------------------+
    | 2          | :math:`(\frac{2}{3}, \frac{1}{6})`      | :math:`\frac{1}{6}` |
    +------------+-----------------------------------------+---------------------+
    | 3          | :math:`(\frac{1}{6}, \frac{2}{3})`      | :math:`\frac{1}{6}` |
    +------------+-----------------------------------------+---------------------+

    .. figure:: /_static/shape_functions/triangle_6_gp.png
        :alt: Gauss quadrature points for 6-node triangle element
        :align: center
        :width: 200px

    .. seealso::

        - :func:`pysdic.triangle_6_shape_functions` : Shape functions for 6-node triangle 2D-elements.

    Parameters
    ----------
    return_weights: :class:`bool`, optional
        If :obj:`True`, the function will also return the weights associated with each Gauss point. Default is :obj:`False`.

    Returns
    -------
    gauss_points: :class:`numpy.ndarray`
        Natural coordinates of the Gauss points. The returned array has shape (3, 2).

    weights: :class:`numpy.ndarray`, optional
        If :obj:`return_weights` is :obj:`True`, the function also returns an array of weights with shape (3,) associated with each Gauss point.

    Raises
    ------
    TypeError
        If the input :obj:`return_weights` is not a boolean.

    
    Examples
    --------

    .. code-block:: python

        import numpy
        from pysdic import triangle_6_gauss_points

        gauss_points = triangle_6_gauss_points()
        print("Gauss points:")
        print(gauss_points)

    .. code-block:: console

        Gauss points:
        [[0.16666667 0.16666667]
         [0.66666667 0.16666667]
         [0.16666667 0.66666667]]

    """
    # Input validation
    if not isinstance(return_weights, bool):
        raise ValueError("Input 'return_weights' must be a boolean value.")
    
    # Gauss points and weights for Gauss quadrature
    gauss_points = numpy.array([
        [1.0 / 6.0, 1.0 / 6.0],
        [2.0 / 3.0, 1.0 / 6.0],
        [1.0 / 6.0, 2.0 / 3.0]
    ], dtype=numpy.float64)
    weights = numpy.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0], dtype=numpy.float64)

    if return_weights:
        return gauss_points, weights
    
    return gauss_points



def quadrangle_4_gauss_points(return_weights: bool = False) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
    r"""
    Get the natural coordinates :math:`(\xi, \eta)` and weights of Gauss quadrature points for a 4-node quadrangle element.

    A 4-node quadrangle element uses the following Gauss points and weights for numerical integration:

    +------------+----------------------------------------------------+---------------------+
    | Point No.  | (:math:`\xi`, :math:`\eta`)                        | Weight :math:`w`    |
    +============+====================================================+=====================+
    | 1          | :math:`(-\frac{1}{\sqrt{3}}, -\frac{1}{\sqrt{3}})` | :math:`1`           |
    +------------+----------------------------------------------------+---------------------+
    | 2          | :math:`(\frac{1}{\sqrt{3}}, -\frac{1}{\sqrt{3}})`  | :math:`1`           |
    +------------+----------------------------------------------------+---------------------+
    | 3          | :math:`(\frac{1}{\sqrt{3}}, \frac{1}{\sqrt{3}})`   | :math:`1`           |
    +------------+----------------------------------------------------+---------------------+
    | 4          | :math:`(-\frac{1}{\sqrt{3}}, \frac{1}{\sqrt{3}})`  | :math:`1`           |
    +------------+----------------------------------------------------+---------------------+

    .. figure:: /_static/shape_functions/quadrangle_4_gp.png
        :alt: Gauss quadrature points for 4-node quadrangle element
        :align: center
        :width: 200px

    .. seealso::

        - :func:`pysdic.quadrangle_4_shape_functions` : Shape functions for 4-node quadrangle 2D-elements.

    Parameters
    ----------
    return_weights: :class:`bool`, optional
        If :obj:`True`, the function will also return the weights associated with each Gauss point. Default is :obj:`False`.

    Returns
    -------
    gauss_points: :class:`numpy.ndarray`
        Natural coordinates of the Gauss points. The returned array has shape (4, 2).

    weights: :class:`numpy.ndarray`, optional
        If :obj:`return_weights` is :obj:`True`, the function also returns an array of weights with shape (4,) associated with each Gauss point.

    
    Raises
    ------
    TypeError
        If the input :obj:`return_weights` is not a boolean.

    
    Examples
    --------

    .. code-block:: python

        import numpy
        from pysdic import quadrangle_4_gauss_points

        gauss_points = quadrangle_4_gauss_points()
        print("Gauss points:")
        print(gauss_points)

    .. code-block:: console

        Gauss points:
        [[-0.57735027 -0.57735027]
         [ 0.57735027 -0.57735027]
         [ 0.57735027  0.57735027]
         [-0.57735027  0.57735027]]

    """
    # Input validation
    if not isinstance(return_weights, bool):
        raise ValueError("Input 'return_weights' must be a boolean value.")
    
    # Gauss points and weights for Gauss quadrature
    gp_coord = 1.0 / numpy.sqrt(3.0)
    gauss_points = numpy.array([
        [-gp_coord, -gp_coord],
        [ gp_coord, -gp_coord],
        [ gp_coord,  gp_coord],
        [-gp_coord,  gp_coord]
    ], dtype=numpy.float64)
    weights = numpy.array([1.0, 1.0, 1.0, 1.0], dtype=numpy.float64)

    if return_weights:
        return gauss_points, weights
    
    return gauss_points



def quadrangle_8_gauss_points(return_weights: bool = False) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
    r"""
    Get the natural coordinates :math:`(\xi, \eta)` and weights of Gauss quadrature points for an 8-node quadrangle element.

    An 8-node quadrangle element uses the following Gauss points and weights for numerical integration:

    +------------+----------------------------------------------------+----------------------+
    | Point No.  | (:math:`\xi`, :math:`\eta`)                        | Weight :math:`w`     |
    +============+====================================================+======================+
    | 1          | :math:`(0,0)`                                      | :math:`\frac{64}{81}`|
    +------------+----------------------------------------------------+----------------------+
    | 2          | :math:`(\sqrt(\frac{3}{5}), \sqrt(\frac{3}{5}))`   | :math:`\frac{25}{81}`|
    +------------+----------------------------------------------------+----------------------+
    | 3          | :math:`(-\sqrt{\frac{3}{5}}, \sqrt{\frac{3}{5}})`  | :math:`\frac{25}{81}`|
    +------------+----------------------------------------------------+----------------------+
    | 4          | :math:`(-\sqrt{\frac{3}{5}}, -\sqrt{\frac{3}{5}})` | :math:`\frac{25}{81}`|
    +------------+----------------------------------------------------+----------------------+
    | 5          | :math:`(\sqrt{\frac{3}{5}}, -\sqrt{\frac{3}{5}})`  | :math:`\frac{25}{81}`|
    +------------+----------------------------------------------------+----------------------+
    | 6          | :math:`(0, \sqrt{\frac{3}{5}})`                    | :math:`\frac{40}{81}`|
    +------------+----------------------------------------------------+----------------------+
    | 7          | :math:`(-\sqrt{\frac{3}{5}}, 0)`                   | :math:`\frac{40}{81}`|
    +------------+----------------------------------------------------+----------------------+
    | 8          | :math:`(0, -\sqrt{\frac{3}{5}})`                   | :math:`\frac{40}{81}`|
    +------------+----------------------------------------------------+----------------------+
    | 9          | :math:`(\sqrt{\frac{3}{5}}, 0)`                    | :math:`\frac{40}{81}`|
    +------------+----------------------------------------------------+----------------------+

    .. figure:: /_static/shape_functions/quadrangle_8_gp.png
        :alt: Gauss quadrature points for 8-node quadrangle element
        :align: center
        :width: 200px

    .. seealso::

        - :func:`pysdic.quadrangle_8_shape_functions` : Shape functions for 8-node quadrangle 2D-elements.

    Parameters
    ----------
    return_weights: :class:`bool`, optional
        If :obj:`True`, the function will also return the weights associated with each Gauss point. Default is :obj:`False`.

    Returns
    -------
    gauss_points: :class:`numpy.ndarray`
        Natural coordinates of the Gauss points. The returned array has shape (9, 2).

    weights: :class:`numpy.ndarray`, optional
        If :obj:`return_weights` is :obj:`True`, the function also returns an array of weights with shape (9,) associated with each Gauss point.


    Raises
    ------
    TypeError
        If the input :obj:`return_weights` is not a boolean.

    
    Examples
    --------

    .. code-block:: python

        import numpy
        from pysdic import quadrangle_8_gauss_points

        gauss_points = quadrangle_8_gauss_points()
        print("Gauss points:")
        print(gauss_points)

    .. code-block:: console

        Gauss points:
        [[ 0.          0.        ]
         [ 0.77459667  0.77459667]
         [-0.77459667  0.77459667]
         [-0.77459667 -0.77459667]
         [ 0.77459667 -0.77459667]
         [ 0.          0.77459667]
         [-0.77459667  0.        ]
         [ 0.         -0.77459667]
         [ 0.77459667  0.        ]]

    """
    # Input validation
    if not isinstance(return_weights, bool):
        raise ValueError("Input 'return_weights' must be a boolean value.")
    
    # Gauss points and weights for Gauss quadrature
    gp_coord = numpy.sqrt(3.0 / 5.0)
    gauss_points = numpy.array([
        [ 0.0,         0.0        ],
        [ gp_coord,    gp_coord   ],
        [-gp_coord,    gp_coord   ],
        [-gp_coord,   -gp_coord   ],
        [ gp_coord,   -gp_coord   ],
        [ 0.0,         gp_coord   ],
        [-gp_coord,    0.0        ],
        [ 0.0,        -gp_coord   ],
        [ gp_coord,    0.0        ]
    ], dtype=numpy.float64)
    weights = numpy.array([
        64.0 / 81.0,
        25.0 / 81.0,
        25.0 / 81.0,
        25.0 / 81.0,
        25.0 / 81.0,
        40.0 / 81.0,
        40.0 / 81.0,
        40.0 / 81.0,
        40.0 / 81.0
    ], dtype=numpy.float64)

    if return_weights:
        return gauss_points, weights
    
    return gauss_points