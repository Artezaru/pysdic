.. currentmodule:: pysdic

Shape functions collection (1D, 2D)
===========================================

.. contents:: Table of Contents
   :local:
   :depth: 3
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

    - Input :obj:`natural_coordinates` will be converted to :obj:`numpy.float64`.
    - Output arrays will be :obj:`numpy.float64`.

.. admonition:: Parameters
    
    **natural_coordinates**: Arraylike
        Natural coordinates where to evaluate the shape functions. The array must have shape :math:`(N_{p}, K)`,
        where :math:`N_{p}` is the number of points to evaluate and :math:`K` is the dimension of the element.

    **return_derivatives**: :class:`bool`, optional
        If :obj:`True`, the method also returns the derivatives of the shape functions with respect to the natural coordinates. Default is :obj:`False`.

    **default**: Real, optional
        The default value to assign to shape functions for points outside the valid range. Default is :obj:`0.0`.

.. admonition:: Returns

    **shape_functions**: :class:`numpy.ndarray`
        Shape functions evaluated at the given natural coordinates. The returned array has shape :math:`(N_{p}, N_{vpe})`,
        where each row corresponds to a point and each column to a node.

    **shape_function_derivatives**: :class:`numpy.ndarray`, optional
        If :obj:`return_derivatives` is :obj:`True`, the function also returns an array of the first derivatives of the shape functions
        with respect to the natural coordinates. The returned array has shape :math:`(N_{p}, N_{vpe}, K)`, where each slice along the first dimension corresponds to a point,
        each column to a node, and each slice along the last dimension to a natural coordinate direction.


Usage
-----------------

Lets illustrate the usage of the shape functions with an example. We will compute the shape functions for a 2-node line element (segment_2) at 3 valid points and 1 invalid point.

.. code-block:: python
    :linenos:

    import numpy
    import pysdic

    # Define the natural coordinate where to evaluate the shape functions
    natural_coordinates = numpy.array([[-1.0], [0.0], [1.0], [1.5]]) 
    print("Expected shape of natural_coordinates: (Np, K)")
    print("Natural coordinates shape:", natural_coordinates.shape)

    # Compute the shape functions for a 2-node line element (segment_2)
    shape_functions = pysdic.compute_segment_2_shape_functions(natural_coordinates)
    print("Expected shape of shape_functions: (Np, Nvpe)")
    print("Shape functions shape:", shape_functions.shape)
    print("Shape function values:")
    print(shape_functions)

.. code-block:: console

    Expected shape of natural_coordinates: (Np, K)
    Natural coordinates shape: (4, 1)
    Expected shape of shape_functions: (Np, Nvpe)
    Shape functions shape: (4, 2)
    Shape function values:
    [[1. 0.]
     [0.5 0.5]
     [0. 1.]
     [0. 0.]]


Implemented shape functions
-----------------------------

1-Dimensional elements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: ../_autosummary/
   :signatures: none

   compute_segment_2_shape_functions
   compute_segment_3_shape_functions


2-Dimensional elements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: ../_autosummary/
   :signatures: none

   compute_triangle_3_shape_functions
   compute_triangle_6_shape_functions
   compute_quadrangle_4_shape_functions
   compute_quadrangle_8_shape_functions

3-Dimensional elements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

    3D shape functions are not yet implemented. Please contact the maintainers if you need this feature.


Dispatcher function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: ../_autosummary/
   :signatures: none

   compute_shape_functions


References elements for implemented shape functions
----------------------------------------------------

+---------------------------------+-------------------------------------------------------------+
| Element Type                    | Reference Element                                           |
+=================================+=============================================================+
| ``segment_2``                   | .. figure:: /_static/shape_functions/segment_2.png          |
| (2-node line element)           |     :width: 200px                                           |
+---------------------------------+-------------------------------------------------------------+
| ``segment_3``                   | .. figure:: /_static/shape_functions/segment_3.png          |
| (3-node line element)           |     :width: 200px                                           |
+---------------------------------+-------------------------------------------------------------+
| ``triangle_3``                  | .. figure:: /_static/shape_functions/triangle_3.png         |
| (3-node triangular element)     |     :width: 200px                                           |
+---------------------------------+-------------------------------------------------------------+
| ``triangle_6``                  | .. figure:: /_static/shape_functions/triangle_6.png         |
| (6-node triangular element)     |     :width: 200px                                           |
+---------------------------------+-------------------------------------------------------------+
| ``quadrangle_4``                | .. figure:: /_static/shape_functions/quadrangle_4.png       |
| (4-node quadrilateral element)  |     :width: 200px                                           |
+---------------------------------+-------------------------------------------------------------+
| ``quadrangle_8``                | .. figure:: /_static/shape_functions/quadrangle_8.png       |
| (8-node quadrilateral element)  |     :width: 200px                                           |
+---------------------------------+-------------------------------------------------------------+
