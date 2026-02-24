API Reference
==============

This section contains a detailed description of the functions, modules, and objects included in ``pysdic``. The reference describes how the methods work and which parameters can be used. It assumes that you have an understanding of the key concepts.
The API is organized into several sections, each corresponding to a specific aspect of the package:

.. grid:: 2

    .. grid-item-card:: 
      :img-top: /_static/_icons/core.png
      :text-align: center

      Core Functions
      ^^^

      This section provide the API reference for the core functions of the package without requesting the package implemented objects. These functions are used to perform various operations on meshes, integration points, images, cameras, and views directly on NumPy arrays.

      +++

      .. button-ref:: api_core
         :expand:
         :color: secondary
         :click-parent:

         To the core functions reference guide

    .. grid-item-card::
      :img-top: /_static/_icons/objects.png
      :text-align: center

      Objects Reference
      ^^^

      This section provide the API reference for the objects implemented in the package. These objects are used to store and manipulate geometrical entities such as point clouds, meshes, integration points, images, cameras, and views.
      
      +++ 

      .. button-ref:: api_objects
         :expand:
         :color: secondary
         :click-parent:

         To the objects reference guide



.. grid:: 2

    .. grid-item-card:: 
      :img-top: /_static/_icons/blender.png
      :text-align: center

      Blender Integration
      ^^^ 

      This section provide the API reference for the classes and functions to generate and manipulate 3D scenes in Blender. This integration allows users to visualize and analyze SDIC results within the Blender environment.

      +++

      .. button-ref:: api_blender
         :expand:
         :color: secondary
         :click-parent:

         To the Blender integration reference guide


    .. grid-item-card:: 
      :img-top: /_static/_icons/workflow.png
      :text-align: center

      Workflow Integration
      ^^^ 

      This section provide the API reference for the workflow functions on the specific objects of the package. These functions are used to perform SDIC for a given configuration.

      +++

      .. button-ref:: api_workflow
         :expand:
         :color: secondary
         :click-parent:

         To the workflow integration reference guide


.. toctree::
   :caption: Contents:
   :hidden:

   api_core
   api_objects
   api_blender
   api_workflow




