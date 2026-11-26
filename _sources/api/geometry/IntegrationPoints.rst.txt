pysdic.IntegrationPoints
========================

.. currentmodule:: pysdic

.. contents:: Table of Contents
   :local:
   :depth: 2

The :class:`pysdic.IntegrationPoints` stores :math:`N_p` points located in the elements of a mesh by their element index and their natural coordinates :math:`\Xi` (shape :math:`(N_p, K)`), with a weight and optional properties for each point. A point outside the mesh is **unlocated** (element index ``-1``).


IntegrationPoints class
-----------------------

.. autoclass:: IntegrationPoints
   :no-members:
   :show-inheritance:


Instantiation
-------------

The :class:`IntegrationPoints` class can be instantiated in two ways:

- **From natural coordinates and element indices** with the constructor (and optional weights).
- **From the Gauss points** of all the elements of a mesh using the classmethod :meth:`from_gauss_points`.

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   IntegrationPoints.from_gauss_points


Implemented methods and attributes
----------------------------------

The following sections provide an overview of the attributes and methods defined
in the :class:`pysdic.IntegrationPoints` class.


Attributes
^^^^^^^^^^

The points can carry properties stored in a property container (see :doc:`Core Tools <../core>`), accessible with ``obj.properties`` or directly with ``obj[name]``.

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   IntegrationPoints.element_indices
   IntegrationPoints.n_located
   IntegrationPoints.n_points
   IntegrationPoints.n_properties
   IntegrationPoints.n_topological_dimensions
   IntegrationPoints.natural_coordinates
   IntegrationPoints.properties
   IntegrationPoints.shape
   IntegrationPoints.weights


Structure methods
^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   IntegrationPoints.concatenate
   IntegrationPoints.copy
   IntegrationPoints.filter_points
   IntegrationPoints.keep_points_by_index
   IntegrationPoints.remove_points_by_index
   IntegrationPoints.keep_points_by_element
   IntegrationPoints.remove_points_by_element
   IntegrationPoints.remove_unlocated
   IntegrationPoints.unlocate_points_by_index
   IntegrationPoints.unlocate_points_by_element


Update after a modification of the mesh
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When the elements of the mesh are filtered, removed or kept, these methods unlocate the points of the removed elements and renumber the element indices of the other points.

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   IntegrationPoints.filter_elements
   IntegrationPoints.keep_elements
   IntegrationPoints.remove_elements


Checks
^^^^^^

Use :meth:`pysdic.Mesh.validate_integration_points` to check the consistency of the points with a mesh.

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   IntegrationPoints.validate
   IntegrationPoints.is_located


Serialization methods
^^^^^^^^^^^^^^^^^^^^^

The :class:`pysdic.IntegrationPoints` is converted to a flat dictionary with ``to_dict`` (and created back with
``from_dict``). It inherits the file methods of the serialization mixins
(see :doc:`Core Tools <../core>`).

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   IntegrationPoints.to_dict
   IntegrationPoints.from_dict
   IntegrationPoints.to_json_file
   IntegrationPoints.from_json_file
   IntegrationPoints.to_hdf5_file
   IntegrationPoints.from_hdf5_file
   IntegrationPoints.to_npz_file
   IntegrationPoints.from_npz_file
