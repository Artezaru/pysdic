Installation
============

.. contents:: Table of Contents
   :local:
   :depth: 1

``pysdic`` requires Python 3.11 or later. To install it, you can use the following command:

.. code-block:: bash

    pip install pysdic

Or install it directly from the GitHub repository using:

.. code-block:: bash

    pip install git+https://github.com/Artezaru/pysdic.git


Required dependencies
---------------------

The following dependencies are installed with ``pysdic``:

- **NumPy** (https://numpy.org/): the arrays of all the objects of the package.
- **SciPy** (https://scipy.org/): sparse matrices, interpolation of the images, spatial searches.
- **OpenCV** (https://opencv.org/): image processing, distortion of images and optical flow.
- **Matplotlib** (https://matplotlib.org/): 2D displays (:class:`pysdic.ImagePlotter` and the ``display_*`` functions).
- **Pysolve-GN** (https://Artezaru.github.io/pysolve-gn/): the Gauss-Newton least squares solver of the
  optimizations (calibration, SDIC terms).


Optional dependencies
---------------------

Some features need an optional package. The optional packages are **not imported** when ``pysdic`` is loaded: a
feature that needs one of them imports it at its first call, and raises an explicit ``ImportError`` with the
installation instructions if it is missing. The rest of the package works without them.

.. list-table::
   :header-rows: 1
   :widths: 15 30 15 40

   * - Package
     - Installation
     - Extra
     - Used by
   * - `py3dframe <https://github.com/Artezaru/py3dframe>`_
     - ``pip install py3dframe``
     - ``frames``
     - The ``frame`` argument of the mesh creators and :meth:`pysdic.PointCloud.frame_transform`.
   * - `pyzernike <https://github.com/Artezaru/pyzernike>`_
     - ``pip install pyzernike-polynomials``
     - ``zernike``
     - :class:`pysdic.ZernikeDistortion`.
   * - `meshio <https://github.com/nschloe/meshio>`_
     - ``pip install meshio``
     - ``meshio``
     - Conversion to and from ``meshio`` and the VTK files with ``use_module="meshio"`` (default).
   * - `PyVista <https://docs.pyvista.org/>`_
     - ``pip install pyvista``
     - ``pyvista``
     - 3D displays (:class:`pysdic.Plotter3D`, ``quick_visualize`` of the point clouds and meshes) and conversion
       to and from ``pyvista``.
   * - `VTK <https://vtk.org>`_
     - ``pip install vtk``
     - ``vtk``
     - Conversion to and from ``vtk`` objects and the VTK files with ``use_module="vtk"``.
   * - `Open3D <https://www.open3d.org/>`_
     - ``pip install open3d``
     - ``open3d``
     - Ray casting on 3D meshes: :meth:`pysdic.Mesh.cast_rays`, :func:`pysdic.compute_visibility`, the 3D case of
       :func:`pysdic.generate_synthetic_image` and :func:`pysdic.compute_pixel_integration_points`.
   * - `h5py <https://github.com/h5py/h5py>`_
     - ``pip install h5py``
     - ``hdf5``
     - HDF5 files (``to_hdf5_file``, ``from_hdf5_file``).

All the optional packages are installed with:

.. code-block:: bash

    pip install pysdic[all]

or one by one with the extras of the table, e.g.:

.. code-block:: bash

    pip install pysdic[pyvista,open3d]

.. note::

   For ``vtk``, only the light package ``vtkmodules`` is imported (``import vtk`` loads every VTK module, which is
   slow).

The classes that need an optional package can be checked before their use, e.g. ``pysdic.Plotter3D.is_available()``.


Rendering with Blender (``pysdic-blender``)
-------------------------------------------

The rendering of synthetic images with Blender (lighting, shading, materials) is provided by a **separate package**,
``pysdic-blender``, built on ``pysdic`` (see :doc:`Blender Rendering <api/blender>`):

.. code-block:: bash

    pip install pysdic-blender

It installs ``pysdic`` and ``bpy`` (Blender as a Python module). ``bpy`` exists for only one version of Python per
Blender release: ``pysdic-blender`` requires **Python 3.13** (``bpy`` 5.1). Use a dedicated environment for the
rendering, e.g.:

.. code-block:: bash

    python3.13 -m venv blender-env
    source blender-env/bin/activate
    pip install pysdic-blender