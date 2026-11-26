pysdic.core.property_container.PropertyContainer
================================================

.. currentmodule:: pysdic.core.property_container

.. contents:: Table of Contents
   :local:
   :depth: 2

.. note::

   This class is internal: it is not exported in the ``pysdic`` namespace. It is used to store
   the properties of the objects of the package (for example the properties of the points of a
   point cloud), and its methods are available through these objects.


PropertyContainer class
-----------------------

.. autoclass:: PropertyContainer
   :no-members:
   :show-inheritance:


Instantiation
-------------

The :class:`PropertyContainer` class can be instantiated in two ways:

- **With the number of entities** :math:`N` and optional initial properties: ``PropertyContainer(N, properties)``.
- **From a dictionary of properties** using the classmethod :meth:`from_dict` (:math:`N` is inferred from the first property).

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   PropertyContainer.from_dict


Implemented methods and attributes
----------------------------------

Attributes
^^^^^^^^^^

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   PropertyContainer.N
   PropertyContainer.force_2d
   PropertyContainer.n_properties


Dictionary-like interface
^^^^^^^^^^^^^^^^^^^^^^^^^

The container can be used as a dictionary: ``container[key]``, ``container[key] = value``,
``del container[key]``, ``key in container``, ``len(container)`` and ``for key in container``.
The arrays are copied when they are set and returned as read-only views.

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   PropertyContainer.get
   PropertyContainer.set
   PropertyContainer.delete
   PropertyContainer.clear
   PropertyContainer.has
   PropertyContainer.keys
   PropertyContainer.items
   PropertyContainer.values


Inspection and validation
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   PropertyContainer.get_dtype
   PropertyContainer.get_shape
   PropertyContainer.get_ndim
   PropertyContainer.check
   PropertyContainer.validate


Reshaping a property
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   PropertyContainer.flatten
   PropertyContainer.reshape


Structure methods
^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   PropertyContainer.copy
   PropertyContainer.concatenate
   PropertyContainer.update
   PropertyContainer.filter


Serialization methods
^^^^^^^^^^^^^^^^^^^^^

The container is converted to a dictionary of arrays with :meth:`to_dict` (and created back
with :meth:`from_dict`). It inherits the serialization methods of the mixins
:class:`pysdic.core.serialization.JsonSerializerMixin`, :class:`pysdic.core.serialization.Hdf5SerializerMixin`
and :class:`pysdic.core.serialization.NpzSerializerMixin`.

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   PropertyContainer.to_dict
   PropertyContainer.to_json_file
   PropertyContainer.from_json_file
   PropertyContainer.to_hdf5_file
   PropertyContainer.from_hdf5_file
   PropertyContainer.to_npz_file
   PropertyContainer.from_npz_file
