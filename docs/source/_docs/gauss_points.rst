.. currentmodule:: pysdic

Gauss quadrature points collection
===================================================

.. contents:: Table of Contents
   :local:
   :depth: 2
   :backlinks: none
   

Description and mathematical background
-----------------------------------------

The package ``pysdic`` provides shape functions for various types of elements used in finite element analysis. 
To facilitate numerical integration over these elements, Gauss quadrature points and weights are provided.

A Gauss quadrature rule approximates the integral of a function over an element by evaluating the function at specific points (the Gauss points) and weighting these evaluations appropriately.
In a space of dimension :math:`E`, we consider a :math:`K`-dimensional element (with :math:`K \leq E`) defined by :math:`N_{vpe}` nodes/vertices.

The integral of a function :math:`f` over the element can be approximated as:

.. math::

    \int_{Element} f(\xi, \eta, \zeta, ...) dV \approx \sum_{i=1}^{N_{gp}} w_i f(\xi_i, \eta_i, \zeta_i, ...)

where :math:`N_{gp}` is the number of Gauss points, :math:`(\xi_i, \eta_i, \zeta_i, ...)` are the coordinates of the Gauss points in the local coordinate system, and :math:`w_i` are the corresponding weights.

.. seealso::

    - :doc:`./integration_points_operations` for operations on integrated points using Gauss points.
    - :doc:`./shape_functions` for shape functions used in conjunction with Gauss quadrature.


Function signatures
--------------------

All of the gauss points methods follow a similar interface. 

.. admonition:: Parameters
    
    **return_weights**: :class:`bool`, optional
        If :obj:`True`, the method also returns the weights associated with each Gauss point. Default is :obj:`False`.

.. admonition:: Returns

    **gauss_points**: :class:`numpy.ndarray`
        Natural coordinates of the Gauss points. The returned array has shape :math:`(N_{gp}, K)` where :math:`N_{gp}` is the number of Gauss points and :math:`K` is the dimension of the element.

    **weights**: :class:`numpy.ndarray`, optional
        If :obj:`return_weights` is :obj:`True`, the method also returns an array of weights with shape :math:`(N_{gp},)` associated with each Gauss point.


Usage
-----------------


Lets illustrate the usage of one of the gauss points methods with an example for a 2-node line element.

.. code-block:: python
    :linenos:

    import numpy
    import pysdic

    # Get Gauss points and weights for 2-node segment element
    gauss_points, weights = pysdic.get_segment_2_gauss_points(return_weights=True)
    print("Expected Gauss points shape: (Np, K)")
    print("Gauss Points shape:", gauss_points.shape)
    print("Gauss Points values")
    print(gauss_points)
    print("Expected Weights shape: (Np,)")
    print("Weights shape:", weights.shape)
    print("Weights values")
    print(weights)

.. code-block:: console

    Expected Gauss points shape: (Np, K)
    Gauss Points shape: (1, 1)
    Gauss Points values
    [[0.]]
    Expected Weights shape: (Np,)
    Weights shape: (1,)
    Weights values
    [2.]



Implemented Gauss points methods
---------------------------------

1-Dimensional elements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: ../_autosummary/
   :signatures: none

   get_segment_2_gauss_points
   get_segment_3_gauss_points


2-Dimensional elements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: ../_autosummary/
   :signatures: none

   get_triangle_3_gauss_points
   get_triangle_6_gauss_points
   get_quadrangle_4_gauss_points
   get_quadrangle_8_gauss_points

3-Dimensional elements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

    3D gauss points are not yet implemented. Please contact the maintainers if you need this feature.


Dispatcher function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: ../_autosummary/
   :signatures: none

   get_gauss_points



References elements for implemented Gauss points methods
----------------------------------------------------------

+---------------------------------+-------------------------------------------------------------+
| Element Type                    | Reference Element                                           |
+=================================+=============================================================+
| ``segment_2``                   | .. figure:: /_static/shape_functions/segment_2_gp.png       |
| (2-node line element)           |     :width: 200px                                           |
+---------------------------------+-------------------------------------------------------------+
| ``segment_3``                   | .. figure:: /_static/shape_functions/segment_3_gp.png       |
| (3-node line element)           |     :width: 200px                                           |
+---------------------------------+-------------------------------------------------------------+
| ``triangle_3``                  | .. figure:: /_static/shape_functions/triangle_3_gp.png      |
| (3-node triangular element)     |     :width: 200px                                           |
+---------------------------------+-------------------------------------------------------------+
| ``triangle_6``                  | .. figure:: /_static/shape_functions/triangle_6_gp.png      |
| (6-node triangular element)     |     :width: 200px                                           |
+---------------------------------+-------------------------------------------------------------+
| ``quadrangle_4``                | .. figure:: /_static/shape_functions/quadrangle_4_gp.png    |
| (4-node quadrilateral element)  |     :width: 200px                                           |
+---------------------------------+-------------------------------------------------------------+
| ``quadrangle_8``                | .. figure:: /_static/shape_functions/quadrangle_8_gp.png    |
| (8-node quadrilateral element)  |     :width: 200px                                           |
+---------------------------------+-------------------------------------------------------------+
