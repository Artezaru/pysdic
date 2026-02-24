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

The following functions allow you to create and manipulate triangle 3 meshes from open3d:

.. autosummary::
   :toctree: ../_autosummary/

   triangle_3_mesh_from_open3d
   triangle_3_mesh_to_open3d


Compute geometric properties of triangle 3 meshes
---------------------------------------------------------------

The following functions allow you to compute geometric properties of triangle 3 meshes, such as areas, strains, normals, etc.:

.. autosummary::
   :toctree: ../_autosummary/

   triangle_3_cast_rays
   triangle_3_compute_elements_areas
   triangle_3_compute_elements_strains    
   triangle_3_compute_elements_normals
   triangle_3_compute_vertices_normals


