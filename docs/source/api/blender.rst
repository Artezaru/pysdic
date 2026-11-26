Blender Rendering (``pysdic-blender``)
======================================

.. contents:: Table of Contents
   :local:
   :depth: 1

:func:`pysdic.generate_synthetic_image` renders the exact gray levels of a textured mesh, without lighting. To
render the same scene with lighting, shading and the reflections of a material, a separate package,
**pysdic-blender**, drives Blender (Cycles renderer) from the objects of ``pysdic``.

.. important::

   ``pysdic-blender`` is **not part of** ``pysdic`` and has no documentation of its own: this page is its reference.

   - Source code: `pysdic-blender on GitHub <https://github.com/Artezaru/pysdic-blender>`_
   - Installation: ``pip install pysdic-blender`` (installs ``pysdic`` and ``bpy``)

   It is a separate package because ``bpy`` (Blender as a Python module) exists for only one version of Python per
   Blender release: ``pysdic-blender`` requires **Python 3.13** (``bpy`` 5.1), while ``pysdic`` supports
   Python 3.11 and later. Use a dedicated environment for the rendering.


When can a scene be rendered by Blender?
----------------------------------------

Blender only knows the pinhole camera model and splits non-planar polygons its own way. A scene is accepted only if:

- the **camera** follows the OpenCV pinhole model: a :class:`pysdic.PerspectiveExtrinsic`, a
  :class:`pysdic.PinholeIntrinsic` without skew and a :class:`pysdic.NoDistortion` (checked on test points, an error
  is raised otherwise);
- the **mesh** is a ``triangle_3`` mesh with a UV map (convert a quadrangle mesh with
  :func:`pysdic.quadrangle_4_to_triangle_3`).

The pose of the camera is converted from the OpenCV convention (:math:`z` forward, :math:`y` down) to the Blender
convention (:math:`z` backward, :math:`y` up): with the world-to-camera rotation :math:`R` and translation
:math:`\vec{t}` of the extrinsic, the center of the camera is :math:`\vec{C} = -R^T \vec{t}` and its orientation in
Blender is :math:`R^T \, \text{diag}(1, -1, -1)`.


Objects
-------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Object
     - Role
   * - ``BlenderCamera(camera, *, pixel_size=0.01, clip_start=1e-3, clip_end=1e6)``
     - A Blender camera from a :class:`pysdic.Camera` (OpenCV pinhole only).
   * - ``BlenderMesh(mesh, texture=...)``
     - A Blender mesh from a ``triangle_3`` :class:`pysdic.Mesh` with its UV map and its texture (float image,
       "Non-Color").
   * - ``BlenderMaterialBSDF`` and ``get_*_material()``
     - The material of a mesh (Principled BSDF): ``get_matte_paint_material``, ``get_steel_material``,
       ``get_iron_material``, ``get_copper_material``, ``get_titanium_material``, ``get_mirror_material``.
   * - ``BlenderSpotLight(position, direction, ...)``, ``BlenderSpotLight.looking_at(position, target, ...)``
     - A spot light.
   * - ``BlenderExperiment``
     - The Blender scene: ``add_camera``, ``add_mesh`` (with a material), ``update_mesh`` (e.g. a displaced mesh at
       each frame), ``add_spotlight``, ``set_background``, ``remove_*``, and ``render(camera, *, n_samples, ...)``
       which returns the rendered image (scene-linear values).
   * - ``render_blender_image``
     - Render one image in a single call.

.. note::

   Blender has a single scene per process: only one :class:`BlenderExperiment` is active at a time (a new experiment
   resets the scene).


Example
-------

.. code-block:: python

   import numpy
   from pysdic import Camera, PerspectiveExtrinsic, PinholeIntrinsic, compute_wave_speckle, create_3d_triangle_3_heightmap
   from pysdic_blender import BlenderCamera, BlenderExperiment, BlenderMesh, BlenderSpotLight, get_matte_paint_material

   camera = Camera(
       180, 240,
       extrinsic=PerspectiveExtrinsic.from_rt(rvec=[numpy.pi, 0.0, 0.0], tvec=[0.0, 0.0, 5.0]),
       intrinsic=PinholeIntrinsic.from_matrix([[300.0, 0.0, 119.5], [0.0, 300.0, 89.5], [0.0, 0.0, 1.0]]),
   )
   plate = create_3d_triangle_3_heightmap(x_bounds=(-2.5, 2.5), y_bounds=(-2.0, 2.0), n_x=11, n_y=9)
   x, y = numpy.linspace(-2.5, 2.5, 500), numpy.linspace(2.0, -2.0, 400)
   speckle = compute_wave_speckle(numpy.stack(numpy.meshgrid(x, y), axis=-1), wavelength=0.15)

   experiment = BlenderExperiment(background_strength=0.1)
   experiment.add_camera("camera", BlenderCamera(camera))
   experiment.add_mesh("plate", BlenderMesh(plate, texture=speckle), material=get_matte_paint_material())
   experiment.add_spotlight("spot", BlenderSpotLight.looking_at([3.0, 0.0, 6.0], [0.0, 0.0, 0.0], energy=2000.0))
   image = experiment.render("camera", n_samples=64)  # (180, 240), scene-linear

.. seealso::

   :func:`pysdic.generate_synthetic_image` (:doc:`Synthetic Images <synthetic>`) for the rendering without lighting,
   available in ``pysdic`` with any camera model and any mesh.