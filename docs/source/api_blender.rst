Blender Integration API Reference
==================================

The package :py:mod:`pysdic.blender` provides classes and functions to integrate with Blender for rendering and visualization purposes. Below is a reference to the main classes available in this package.

- :class:`pysdic.Mesh`: A class representing a 3D mesh, including vertices and connectivity information. For Blender integration, the mesh must be a 3D triangular mesh with a UVMAP defined.
- :doc:`Blender Camera <_docs/blender_camera>`: A class for managing camera properties and settings in Blender.
- :doc:`Blender Experiment <_docs/blender_experiment>`: A class for managing experiments in Blender, including adding meshes, lights, and cameras, and rendering scenes.
- :doc:`Blender Material BSDF <_docs/blender_material_bsdf>`: A class for managing material properties and settings in Blender, including BSDF properties.
- :doc:`Blender SpotLight <_docs/blender_spotlight>`: A class for managing spotlight properties and settings in Blender.

.. toctree::
   :maxdepth: 2
   :glob:
   :hidden:

   _docs/blender_camera
   _docs/blender_experiment
   _docs/blender_material_bsdf
   _docs/blender_spotlight
