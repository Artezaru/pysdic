.. currentmodule:: pysdic

Gauss Quadrature Points Collection
===================================================

.. contents:: Table of Contents
   :local:
   :depth: 2
   :backlinks: top
   

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


Function signatures
--------------------

All of the gauss points methods follow a similar interface. 

Parameters
~~~~~~~~~~
- **return_weights** : :class:`bool`, optional
    If :obj:`True`, the function will also return the weights associated with each Gauss point. Default is :obj:`False`.


Returns
~~~~~~~
- **gauss_points** : :class:`numpy.ndarray`
    Natural coordinates of the Gauss points. The returned array has shape (:math:`N_{gp}`, :math:`K`) where :math:`N_{gp}` is the number of Gauss points and :math:`K` is the dimension of the element.

- **weights** : :class:`numpy.ndarray`, optional
    If :obj:`return_weights` is :obj:`True`, the function also returns an array of weights with shape (:math:`N_{gp}`,) associated with each Gauss point.





Implemented Gauss Points Methods
---------------------------------

1-Dimensional Elements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: ../generated/

   segment_2_gauss_points
   segment_3_gauss_points


2-Dimensional Elements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: ../generated/

   triangle_3_gauss_points
   triangle_6_gauss_points
   quadrangle_4_gauss_points
   quadrangle_8_gauss_points



