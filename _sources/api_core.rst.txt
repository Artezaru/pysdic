Core Package API Reference
==============================

.. contents:: Table of Contents
   :local:
   :depth: 1
   :backlinks: top

Shape Functions and Gauss Points
-------------------------------------

The package ``pysdic`` provides a set of core numerical operations that are fundamental for finite element analyses and SDIC analyses. 
These operations include functions to evaluate shape functions, compute Gauss points and weights, and perform numerical integration.

- :doc:`Shape functions <_docs/shape_functions>`: Functions to evaluate shape functions for different element types and orders.
- :doc:`Gauss points <_docs/gauss_points>`: Functions to compute the coordinates and weights of Gauss points for numerical integration.

.. toctree::
   :maxdepth: 1
   :hidden:

   _docs/shape_functions
   _docs/gauss_points


Mesh operations (interpolations, adjacency, etc.)
-------------------------------------------------------------

The package also includes core operations for working with meshes and connectivity graphs, such as interpolation of fields, computation of adjacency matrices, and operations specific to triangle meshes.

- :doc:`Mesh interpolation operations <_docs/integration_points_operations>`: Functions to perform interpolation of fields defined on meshes, including interpolation from vertices to Gauss points and vice versa.
- :doc:`Adjacency operations <_docs/adjacency_operations>`: Functions to compute adjacency matrices for meshes, which are essential for various analyses and operations on meshes.
- :doc:`Triangle mesh operations <_docs/triangle_3_meshes_operations>`: Functions specific to triangle 3 meshes, such as operations for computing triangle areas, normals, and other geometric properties.

.. toctree::
   :maxdepth: 1
   :hidden:

   _docs/integration_points_operations
   _docs/adjacency_operations
   _docs/triangle_3_meshes_operations


Build operators for SDIC analyses
-------------------------------------

The package provides functions to build operators for SDIC analyses, such as temporal derivation operators and displacement operators.
These operators are essential for performing SDIC analyses on time series data.

- :doc:`Temporal derivation operators <_docs/temporal_derivation>`: Functions to build finite difference operators for temporal derivation of time series data, including forward, backward, and central finite difference operators.
- :doc:`Build displacement and parameters operators <_docs/build_operator>`: Functions to build displacement operators for SDIC analyses and build the :math:`JdU = R` system of equations.

.. toctree::
   :maxdepth: 1
   :hidden:

   _docs/temporal_derivation
   _docs/build_operator


Additional operations
------------------------------------

The package also includes additional core operations that are useful for various analyses and operations on meshes and time series data.
These include operations for computing the Bouguer law, geometric extent, and bidirectional reflectance distribution function (BRDF) for photometric analyses.

- :doc:`Optical flow operations <_docs/optical_flow>`: Functions to compute the optical flow between two images, which can be used for motion analysis and other applications.
- :doc:`Photometric operations <_docs/photometric_operations>`: Functions to compute the Bouguer law, geometric extent, and bidirectional reflectance distribution function (BRDF) for photometric analyses.

.. toctree::
   :maxdepth: 1
   :hidden:

   _docs/optical_flow
   _docs/photometric_operations
