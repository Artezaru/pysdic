.. currentmodule:: pysdic.geometry

pysdic.geometry.LinearTriangleMesh3D
===========================================

.. autoclass:: LinearTriangleMesh3D


Instantiate a LinearTriangleMesh3D object
-------------------------------------------

The LinearTriangleMesh3D is a subclass of Mesh3D and can be instantiated directly or using the methods inherited from :class:`pysdic.geometry.Mesh3D`.

The LinearTriangleMesh3D class can also be instantiated from an Open3D TriangleMesh object using the class method :meth:`from_open3d`.

.. autosummary::
   :toctree: ../generated/

    LinearTriangleMesh3D.from_open3d
    LinearTriangleMesh3D.to_open3d

Additional LinearTriangleMesh3D attributes
-------------------------------------------

.. autosummary::
   :toctree: ../generated/

    LinearTriangleMesh3D.elements_uvmap

Manipulating LinearTriangleMesh3D objects
-------------------------------------------

To manipulate only the geometry of the mesh, access the ``vertices`` attribute (:class:`pysdic.geometry.PointCloud3D`) and use its methods.
See the class :class:`pysdic.geometry.Mesh3D` for other inherited methods.

.. autosummary::
   :toctree: ../generated/

    LinearTriangleMesh3D.cast_rays
    LinearTriangleMesh3D.compute_elements_areas
    LinearTriangleMesh3D.compute_elements_normals
    LinearTriangleMesh3D.compute_vertices_normals
    LinearTriangleMesh3D.shape_functions
    LinearTriangleMesh3D.visualize
    LinearTriangleMesh3D.visualize_texture
    LinearTriangleMesh3D.visualize_vertices_property

