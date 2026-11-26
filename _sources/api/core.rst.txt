Core Tools (internal classes)
=============================

.. contents:: Table of Contents
   :local:
   :depth: 1

The submodule ``pysdic.core`` contains the internal building blocks shared by the objects of the package.

.. important::

   These classes are **internal**: they are not exported in the ``pysdic`` namespace
   (``from pysdic import PropertyContainer`` fails, except :class:`pysdic.DisflowConfig`) and are not meant to be
   instantiated directly by the users. They are documented because their methods are available on the public objects
   that use or inherit from them (for example the serialization methods ``to_json_file``,
   ``from_hdf5_file``, ... of the transformations and parametrizations).

   If needed, they can still be imported from ``pysdic.core``:

   .. code-block:: python

      from pysdic.core import PropertyContainer, LockableMixin
      from pysdic.core import SerializerMixin, JsonSerializerMixin, NpzSerializerMixin, Hdf5SerializerMixin


Property container
------------------

A container of named arrays sharing the same first dimension :math:`N` (the number of entities, for
example the points of a point cloud), with a dictionary-like interface and validated shapes.

- :doc:`PropertyContainer <core/PropertyContainer>`: Container of properties of shape :math:`(N, \dots)` associated with :math:`N` entities.

.. toctree::
   :maxdepth: 1
   :hidden:

   ./core/PropertyContainer.rst


DIS configuration
-----------------

The parameters of the optical flow of :func:`pysdic.compute_optical_flow`, based on the ``OpenCV`` ``DIS``
(Dense Inverse Search) algorithm. Defined in ``pysdic.core``, the class is also exported as
:class:`pysdic.DisflowConfig`.

- :doc:`DisflowConfig <core/DisflowConfig>`: Parameters of the DIS optical flow algorithm.

.. toctree::
   :maxdepth: 1
   :hidden:

   ./core/DisflowConfig.rst


Serialization mixins
--------------------

The serialization mixins add the methods to save and load an object in several file formats. A class
inheriting from them only implements ``to_dict`` and ``from_dict``, and the mixins convert this dictionary
to and from the files:

+---------------------------------------------------------------+--------------------------------------------------+------------------------------------------+
| Mixin                                                         | Methods                                          | Notes                                    |
+===============================================================+==================================================+==========================================+
| :doc:`JsonSerializerMixin <core/JsonSerializerMixin>`         | ``to_json_string``, ``from_json_string``,        | Arrays are converted to lists.           |
|                                                               | ``to_json_file``, ``from_json_file``             |                                          |
+---------------------------------------------------------------+--------------------------------------------------+------------------------------------------+
| :doc:`Hdf5SerializerMixin <core/Hdf5SerializerMixin>`         | ``to_hdf5_file``, ``from_hdf5_file``             | Requires ``h5py``. Nested dictionaries   |
|                                                               |                                                  | are stored as groups.                    |
+---------------------------------------------------------------+--------------------------------------------------+------------------------------------------+
| :doc:`NpzSerializerMixin <core/NpzSerializerMixin>`           | ``to_npz_file``, ``from_npz_file``               | Flat dictionaries only, without pickle.  |
+---------------------------------------------------------------+--------------------------------------------------+------------------------------------------+

- :doc:`SerializerMixin <core/SerializerMixin>`: Abstract base class of the serialization mixins (``to_dict`` and ``from_dict``).

All the file methods accept ``make_dirs=True`` to create the missing parent directories.

.. toctree::
   :maxdepth: 1
   :hidden:

   ./core/SerializerMixin.rst
   ./core/JsonSerializerMixin.rst
   ./core/Hdf5SerializerMixin.rst
   ./core/NpzSerializerMixin.rst


Lock mixin
----------

The lock mixin adds a lock to an object: a locked object can still be read, copied and saved, but
the methods that modify it in place raise a :class:`RuntimeError`, with a message explaining why the
object is locked and what to do instead. The lock is set by the objects of the package that own
another object (for example a ``ReparametrizedTransformation`` owns its transformation and its
parametrization): there is no public method to lock or unlock an object, and a copy is always unlocked.
Some methods offer an explicit bypass, e.g. ``Transformation.set_parameters(..., override_lock=True)``.

- :doc:`LockableMixin <core/LockableMixin>`: Mixin adding ``is_locked`` and ``locked_message`` to an object (and the internal ``_set_locked`` and ``_check_unlocked``).

.. toctree::
   :maxdepth: 1
   :hidden:

   ./core/LockableMixin.rst