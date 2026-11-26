pysdic.PointCloud
=================

.. currentmodule:: pysdic

.. contents:: Table of Contents
   :local:
   :depth: 2

The :class:`pysdic.PointCloud` stores :math:`N_p` points in a space of dimension :math:`E` (coordinates of shape :math:`(N_p, E)`) with optional properties for each point.


PointCloud class
----------------

.. autoclass:: PointCloud
   :no-members:
   :show-inheritance:


Instantiation
-------------

The :class:`PointCloud` class can be instantiated in several ways:

- **From an array of coordinates** of shape :math:`(N_p, E)` with the constructor or the classmethod :meth:`from_array`.
- **From other libraries**: ``meshio`` (:meth:`from_meshio`), ``pyvista`` (:meth:`from_pyvista`) or ``vtk`` (:meth:`from_vtk`, :meth:`from_vtk_file`).

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   PointCloud.from_array
   PointCloud.from_meshio
   PointCloud.from_pyvista
   PointCloud.from_vtk
   PointCloud.from_vtk_file


Implemented methods and attributes
----------------------------------

The following sections provide an overview of the attributes and methods defined
in the :class:`pysdic.PointCloud` class.


Attributes
^^^^^^^^^^

The points can carry properties stored in a property container (see :doc:`Core Tools <../core>`), accessible with ``obj.properties`` or directly with ``obj[name]``.

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   PointCloud.coordinates
   PointCloud.n_dimensions
   PointCloud.n_points
   PointCloud.n_properties
   PointCloud.properties
   PointCloud.shape


Structure methods
^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   PointCloud.concatenate
   PointCloud.copy
   PointCloud.filter_points
   PointCloud.keep_points_by_index
   PointCloud.remove_points_by_index
   PointCloud.insert_dimension
   PointCloud.remove_dimension


Checks and geometry
^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   PointCloud.validate
   PointCloud.all_close
   PointCloud.all_finite
   PointCloud.is_finite
   PointCloud.has_nan
   PointCloud.has_inf
   PointCloud.bounding_box
   PointCloud.bounding_sphere
   PointCloud.frame_transform


Conversion and visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Export to arrays and other libraries (the ``meshio``, ``pyvista``, ``vtk`` and ``py3dframe`` methods require the corresponding optional packages).

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   PointCloud.to_array
   PointCloud.to_meshio
   PointCloud.to_pyvista
   PointCloud.to_vtk
   PointCloud.to_vtk_file
   PointCloud.quick_visualize


Serialization methods
^^^^^^^^^^^^^^^^^^^^^

The :class:`pysdic.PointCloud` is converted to a flat dictionary with ``to_dict`` (and created back with
``from_dict``). It inherits the file methods of the serialization mixins
(see :doc:`Core Tools <../core>`).

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   PointCloud.to_dict
   PointCloud.from_dict
   PointCloud.to_json_file
   PointCloud.from_json_file
   PointCloud.to_hdf5_file
   PointCloud.from_hdf5_file
   PointCloud.to_npz_file
   PointCloud.from_npz_file
