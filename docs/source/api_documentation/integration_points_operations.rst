.. currentmodule:: pysdic

Operations on Integration Points (interpolation, projection, etc.)
===================================================================

.. contents:: Table of Contents
   :local:
   :depth: 1
   :backlinks: top


Description
-----------------------------------------

The package ``pysdic`` provides functions to perform operations for various types of elements used in finite element analysis. 

.. seealso::

    - :doc:`./shape_functions` for shape functions definitions and details.

In a space of dimension :math:`E` with a mesh constituted of :math:`N_{e}` elements and :math:`N_{v}` nodes/vertices.
The mesh is composed of :math:`K`-dimensional elements (with :math:`K \leq E`) defined by :math:`N_{vpe}` nodes for each element.

The objective of these operations is to interpolate or project values defined at integration points to nodes or other points in space, and vice versa.

If we consider a property :math:`P` defined at the vertices of the elements, we can interpolate this property at any point within the element using shape functions:

.. math::

    P(\xi, \eta, \zeta, ...) = \sum_{i=1}^{N_{vpe}} N_i(\xi, \eta, \zeta, ...) P_i

where :math:`P` is the interpolated value at the point, :math:`N_i` are the shape functions, and :math:`P_i` are the nodal values.


Implemented Functions
-----------------------------

The following functions are implemented in this module:

.. autosummary::
   :toctree: ../generated/

   assemble_shape_function_matrix
   construct_jacobian
   derivate_property
   interpolate_property
   project_property_to_vertices
   remap_vertices_coordinates



