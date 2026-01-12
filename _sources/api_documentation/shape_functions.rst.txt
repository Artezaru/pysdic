.. currentmodule:: pysdic

Shape Functions Collection (1D, 2D, 3D)
===========================================

.. contents:: Table of Contents
   :local:
   :depth: 2
   :backlinks: none


Description and mathematical background
-----------------------------------------

The package ``pysdic`` provides shape functions for various types of elements used in finite element analysis. 
Shape functions are mathematical functions that describe how the displacement within an element varies with respect to its nodal displacements.

In a space of dimension :math:`E`, we consider a :math:`K`-dimensional element (with :math:`K \leq E`) defined by :math:`N_{vpe}` nodes/vertices.

For an :math:`K`-dimensional element defined by :math:`N_{vpe}` nodes, points inside the element are represented in a local coordinate system :math:`(\xi, \eta, \zeta, ...)` also named ``natural coordinates``.
Shape functions are defined in this local coordinate system in order to interpolate values at any point within the element based on the values at the nodes.

.. math::

    P(\xi, \eta, \zeta, ...) = \sum_{i=1}^{N_{vpe}} N_i(\xi, \eta, \zeta, ...) P_i

where :math:`P` is the interpolated value at the point, :math:`N_i` are the shape functions, and :math:`P_i` are the nodal values.

.. seealso::

    - :doc:`./integration_points_operations` for operations on integrated points using shape functions.
    - :doc:`./gauss_points` for Gauss quadrature points and weights.


Function signatures
--------------------

All of the shape function methods follow a similar interface. 

.. note::

    The input :obj:`natural_coordinates` will be converted to :obj:`numpy.float64` for computation.
    The output arrays will be :obj:`numpy.float64` as well.

Parameters
~~~~~~~~~~
**natural_coordinates**: :class:`numpy.ndarray`
    Natural coordinates where to evaluate the shape functions. The array must have shape :math:`(N_{p}, K)`,
    where :math:`N_{p}` is the number of points to evaluate and :math:`K` is the dimension of the element.

**return_derivatives**: :class:`bool`, optional
    If :obj:`True`, the method also returns the derivatives of the shape functions with respect to the natural coordinates. Default is :obj:`False`.

**default**: :class:`numbers.Real`, optional
    The default value to assign to shape functions for points outside the valid range. Default is :obj:`0.0`.


Returns
~~~~~~~
**shape_functions**: :class:`numpy.ndarray`
    Shape functions evaluated at the given natural coordinates. The returned array has shape :math:`(N_{p}, N_{vpe})`,
    where each row corresponds to a point and each column to a node.

**shape_function_derivatives**: :class:`numpy.ndarray`, optional
    If :obj:`return_derivatives` is :obj:`True`, the function also returns an array of the first derivatives of the shape functions
    with respect to the natural coordinate. The returned array has shape :math:`(N_{p}, N_{vpe}, K)`, where each slice along the first dimension corresponds to a point,
    each column to a node, and each slice along the last dimension to a natural coordinate direction.


Raises
~~~~~~
**TypeError**
    - If :obj:`natural_coordinates` can't be converted to a :class:`numpy.ndarray`.
    - If :obj:`return_derivatives` is not a :class:`bool`.
    - If :obj:`default` is not a :class:`numbers.Real`.

**ValueError**
    - If :obj:`natural_coordinates` does not have shape :math:`(N_{p}, K)` where :math:`K` is the dimension of the element.






Implemented Shape Functions
-----------------------------

1-Dimensional Elements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: ../generated/

   segment_2_shape_functions
   segment_3_shape_functions


2-Dimensional Elements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: ../generated/

   triangle_3_shape_functions
   triangle_6_shape_functions
   quadrangle_4_shape_functions
   quadrangle_8_shape_functions


Dispatcher function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: ../generated/

   shape_functions




Usage
-----------------

Lets illustrate the usage of one of the shape function methods with an example for a 2-node line element.

.. code-block:: python

    import numpy
    from pysdic import segment_2_shape_functions

    # Define local coordinates for points where shape functions are to be evaluated
    coords = numpy.array([
        [-0.5],
        [0.0],
        [0.5],
    ])

    # Evaluate shape functions and their derivatives at the specified local coordinates
    shape_functions = segment_2_shape_functions(coords)

    print("Shape Functions:\n", shape_functions)

The output will be:

.. code-block:: console

   Shape Functions:
    [[0.75 0.25]
     [0.5  0.5 ]
     [0.25 0.75]]

To also obtain the derivatives of the shape functions with respect to the local coordinate, set the :obj:`return_derivatives` parameter to :obj:`True`.

.. code-block:: python

    import numpy
    from pysdic import segment_2_shape_functions

    # Define local coordinates for points where shape functions are to be evaluated
    coords = numpy.array([
        [-0.5],
        [0.0],
        [0.5],
    ])

    # Evaluate shape functions and their derivatives at the specified local coordinates
    shape_funcs, shape_func_derivs = segment_2_shape_functions(coords, return_derivatives=True)

    print("Shape Functions:\n", shape_funcs)
    print("Shape Function Derivatives:\n", shape_func_derivs)

The output will be:

.. code-block:: console

   Shape Functions:
    [[0.75 0.25]
     [0.5  0.5 ]
     [0.25 0.75]]
   Shape Function Derivatives:
    [[[-0.5]
      [ 0.5]]

     [[-0.5]
      [ 0.5]]

     [[-0.5]
      [ 0.5]]]


Set the :obj:`default` parameter to specify a default value for shape functions at points outside the valid range.

.. code-block:: python

    import numpy
    from pysdic import segment_2_shape_functions

    # Define local coordinates for points where shape functions are to be evaluated
    coords = numpy.array([
        [-1.5],  # Outside valid range
        [0.0],
        [1.5],   # Outside valid range
    ])

    # Evaluate shape functions with a default value for out-of-range points
    shape_functions = segment_2_shape_functions(coords, default=numpy.nan)

    print("Shape Functions with Default for Out-of-Range Points:\n", shape_functions)

The output will be:

.. code-block:: console

   Shape Functions with Default for Out-of-Range Points:
    [[ nan  nan]
     [0.5  0.5 ]
     [ nan  nan]]