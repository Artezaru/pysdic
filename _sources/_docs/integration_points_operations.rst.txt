.. currentmodule:: pysdic

Operations on Integration Points (interpolation, projection, etc.)
===================================================================

.. contents:: Table of Contents
   :local:
   :depth: 2
   :backlinks: top


In this module:

- ``compute_*`` functions perform the full computation starting from
  mesh geometry and integration points. They internally re-compute shape
  functions, Jacobians, and other required quantities.

- ``assemble_*`` functions operate on precomputed quantities
  (shape functions, Jacobians, weights, etc.) and perform only the final
  algebraic operation. They are intended for advanced usage and
  performance-critical workflows.


Description
-----------------------------------------

The package ``pysdic`` provides functions to perform operations for various types of elements used in finite element analysis. 

In a space of dimension :math:`E` with a mesh constituted of :math:`N_{e}` elements and :math:`N_{v}` nodes/vertices.
The mesh is composed of :math:`K`-dimensional elements (with :math:`K \leq E`) defined by :math:`N_{vpe}` nodes for each element.

The objective of these operations is to interpolate or project values defined at integration points to nodes or other points in space, and vice versa.

If we consider a property :math:`P` defined at the vertices of the elements, we can interpolate this property at any point within the element using shape functions:

.. math::

    P(\xi, \eta, \zeta, ...) = \sum_{i=1}^{N_{vpe}} N_i(\xi, \eta, \zeta, ...) P_i

where :math:`P` is the interpolated value at the point, :math:`N_i` are the shape functions, and :math:`P_i` are the nodal values.

.. seealso::

    - :doc:`./shape_functions` for shape functions definitions and details.


Building operation matrices and precomputed quantities
--------------------------------------------------------------------

In this section:

- ``shape_function`` refers to the computed shape functions values at the integration points with shape :math:`(N_{p}, N_{vpe})`.

- ``shape_function_derivative`` refers to the computed shape function derivatives at the integration points with shape :math:`(N_{p}, N_{vpe}, K)` with respect to the natural coordinates.

- ``shape_function_matrix`` refers to the assembled shape function matrix at the integration points with shape :math:`(N_{p}, N_{v})`.

- ``jacobian_matrix`` refers to the Jacobian matrix  of the transformation from natural coordinates to global coordinates at the integration points with shape :math:`(N_{p}, E, K)` (ie :math:`J = \frac{\partial (x, y, z, ...)}{\partial (\xi, \eta, \zeta, ...)}`).


.. autosummary::
    :toctree: ../_autosummary/
    :signatures: none

    assemble_shape_function_matrix
    compute_shape_function_matrix
    assemble_jacobian_matrix
    compute_jacobian_matrix
    remap_vertices_coordinates


Performing property interpolation, projection, and derivatives at integration points
--------------------------------------------------------------------------------------

In this section:

- ``interpolation`` refers to interpolating vertices properties :math:`(N_v, P)` to integration points :math:`(N_{p}, P)`.

- ``projection`` refers to projecting properties defined at integration points :math:`(N_{p}, P)` back to vertices :math:`(N_v, P)`.

- ``derivative`` refers to computing the spatial derivatives of properties defined at vertices :math:`(N_v, P)` at integration points :math:`(N_{p}, \nabla P)` with respect to the global coordinates.

.. autosummary::
    :toctree: ../_autosummary/
    :signatures: none

    assemble_property_interpolation
    compute_property_interpolation
    assemble_property_projection
    compute_property_projection
    assemble_property_derivative
    compute_property_derivative

