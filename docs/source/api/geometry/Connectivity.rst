pysdic.Connectivity
===================

.. currentmodule:: pysdic

.. contents:: Table of Contents
   :local:
   :depth: 2

The :class:`pysdic.Connectivity` stores the :math:`N_e` elements of a mesh, each defined by the indices of its :math:`N_{vpe}` vertices (shape :math:`(N_e, N_{vpe})`), with an optional element type and properties for each element.


Connectivity class
------------------

.. autoclass:: Connectivity
   :no-members:
   :show-inheritance:


Instantiation
-------------

The :class:`Connectivity` class can be instantiated:

- **From an array of elements** of shape :math:`(N_e, N_{vpe})` with the constructor or the classmethod :meth:`from_array`.

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   Connectivity.from_array


Implemented methods and attributes
----------------------------------

The following sections provide an overview of the attributes and methods defined
in the :class:`pysdic.Connectivity` class.


Attributes
^^^^^^^^^^

The elements can carry properties stored in a property container (see :doc:`Core Tools <../core>`), accessible with ``obj.properties`` or directly with ``obj[name]``.

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   Connectivity.element_type
   Connectivity.elements
   Connectivity.n_elements
   Connectivity.n_properties
   Connectivity.n_topological_dimensions
   Connectivity.n_vertices_per_element
   Connectivity.properties
   Connectivity.shape


Structure methods
^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   Connectivity.concatenate
   Connectivity.copy
   Connectivity.filter_elements
   Connectivity.keep_elements_by_index
   Connectivity.remove_elements_by_index


Checks
^^^^^^

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   Connectivity.validate
   Connectivity.all_close
   Connectivity.all_positive
   Connectivity.all_lower_than
   Connectivity.is_positive
   Connectivity.has_negative
   Connectivity.has_higher_than


Conversion
^^^^^^^^^^

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   Connectivity.to_array


Serialization methods
^^^^^^^^^^^^^^^^^^^^^

The :class:`pysdic.Connectivity` is converted to a flat dictionary with ``to_dict`` (and created back with
``from_dict``). It inherits the file methods of the serialization mixins
(see :doc:`Core Tools <../core>`).

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   Connectivity.to_dict
   Connectivity.from_dict
   Connectivity.to_json_file
   Connectivity.from_json_file
   Connectivity.to_hdf5_file
   Connectivity.from_hdf5_file
   Connectivity.to_npz_file
   Connectivity.from_npz_file
