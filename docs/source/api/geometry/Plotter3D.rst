pysdic.Plotter3D
================

.. currentmodule:: pysdic

.. contents:: Table of Contents
   :local:
   :depth: 2

The :class:`pysdic.Plotter3D` displays the geometrical objects of ``pysdic`` (:class:`pysdic.PointCloud`, :class:`pysdic.Mesh` and :class:`pysdic.IntegrationPoints`) with ``pyvista``.

.. important::

   Requires the optional package ``pyvista`` (``pip install pyvista``). Use ``Plotter3D.is_available()`` to check it.


Plotter3D class
---------------

.. autoclass:: Plotter3D
   :no-members:
   :show-inheritance:


Instantiation
-------------

The :class:`Plotter3D` class takes no argument: ``plotter = Plotter3D()``. The objects are added with the ``add_*`` methods and displayed with :meth:`show`.


Implemented methods and attributes
----------------------------------

The following sections provide an overview of the attributes and methods defined
in the :class:`pysdic.Plotter3D` class.


Attributes
^^^^^^^^^^

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   Plotter3D.plotter


Adding objects
^^^^^^^^^^^^^^

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   Plotter3D.add_point_cloud
   Plotter3D.add_mesh
   Plotter3D.add_integration_points
   Plotter3D.add_scalar_bar
   Plotter3D.add_axes


Scene settings and display
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   Plotter3D.set_title
   Plotter3D.set_bounds
   Plotter3D.set_camera
   Plotter3D.clear
   Plotter3D.show
