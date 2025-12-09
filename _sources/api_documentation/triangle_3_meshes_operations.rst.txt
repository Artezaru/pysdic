.. currentmodule:: pysdic

Manage and Operate on Triangle 3 Meshes
=================================================

.. contents:: Table of Contents
   :local:
   :depth: 2
   :backlinks: top

The package ``pysdic`` provides functions to operate on 3-node triangular surface meshes embedded in 3D space.

Create and manipulate triangle 3 meshes from open3d
---------------------------------------------------------------

.. autosummary::
   :toctree: ../generated/

   triangle_3_mesh_from_open3d
   triangle_3_mesh_to_open3d


Compute geometric properties of triangle 3 meshes
---------------------------------------------------------------

.. autosummary::
   :toctree: ../generated/

   triangle_3_build_vertices_adjacency_matrix
   triangle_3_build_elements_adjacency_matrix
   triangle_3_cast_rays
   triangle_3_compute_elements_areas    
   triangle_3_compute_elements_normals
   triangle_3_compute_vertices_normals
   triangle_3_extract_unique_edges


