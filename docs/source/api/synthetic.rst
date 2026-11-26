Synthetic Images
================

.. currentmodule:: pysdic

.. contents:: Table of Contents
   :local:
   :depth: 1

The package ``pysdic`` provides tools to generate synthetic images of textured surfaces seen by cameras, with a known
displacement: they are the reference to test and validate a SDIC analysis (see :doc:`SDIC Analysis <sdic>`).

.. code-block:: text

   speckle texture ──┐
   Mesh + UV map ────┼──► generate_synthetic_image ──► add_*_noise / add_gaussian_blur ──► quantize_image ──► Image
   Camera ───────────┤        (float image)                (acquisition)                    (uint8, uint16)
   displacement ─────┘


Textures
--------

The texture of a surface mesh is an image mapped on the elements by the UV map of the mesh (see :attr:`pysdic.Mesh.uvmap`),
with the OpenGL convention: :math:`(u, v) \in [0, 1]^2` covers the whole texture, with :math:`v` pointing upward.
The texture is evaluated at the natural coordinates of the material points, so it is attached to the material and
follows the displacement of the mesh.

- :doc:`compute_wave_speckle <synthetic/compute_wave_speckle>`: Synthetic speckle pattern (sum of plane waves of random directions and phases, saturated by a hyperbolic tangent) at given points, e.g. a grid of UV coordinates.
- :doc:`interpolate_texture <synthetic/interpolate_texture>`: Values of the texture of a surface mesh at integration points, through its UV map.
- :doc:`get_lena_image <synthetic/get_lena_image>`: A reference image shipped with the package, to use as a default texture for quick tests.

The UV map of a mesh can also be computed from the projection of its vertices on a camera
(:func:`pysdic.compute_projected_uvmap`), to use an acquired image as texture.

.. toctree::
   :maxdepth: 1
   :hidden:

   ./synthetic/compute_wave_speckle.rst
   ./synthetic/interpolate_texture.rst
   ./synthetic/get_lena_image.rst


Generation of images
--------------------

Each pixel of the sensor is sampled by :math:`n \times n` points. The material point seen by each sample gets the value
of the texture, the samples that see no point of the mesh get a background value, and the value of a pixel is the mean of
its samples (anti-aliasing of the edges). The case is chosen from the dimension :math:`E` of the world points of the camera,
which must be the dimension of the mesh:

.. list-table::
   :header-rows: 1
   :widths: 15 45 40

   * - :math:`E`
     - Cameras
     - Material point seen by a sample
   * - 2 (2D mesh)
     - Camera of the plane (``PlanarRigidExtrinsic``, ``PlanarHomographicExtrinsic``) or 3D camera observing
       the plane :math:`Z_w = 0` (``WorldPlanePerspectiveExtrinsic``, ``WorldPlaneOrthographicExtrinsic``)
     - Inversion of the camera, then location of the point in the mesh (:meth:`pysdic.Mesh.locate_points`)
   * - 3 (3D mesh)
     - 3D camera (``PerspectiveExtrinsic``, ``OrthographicExtrinsic``, ...)
     - First intersection of the ray of the sample with the mesh (:meth:`pysdic.Mesh.cast_rays`)

- :doc:`generate_synthetic_image <synthetic/generate_synthetic_image>`: Image of shape :math:`(H, W, C)` of a textured surface mesh, optionally displaced, seen by a camera.

.. important::

   The 3D case uses the ray casting of :meth:`pysdic.Mesh.cast_rays`, which requires the optional package ``open3d``
   (``pip install open3d``). The 2D case has no optional dependency.

.. seealso::

   :doc:`Blender Rendering <blender>` (separate package ``pysdic-blender``) to render the same scene with lighting,
   shading and the reflections of a material.

.. toctree::
   :maxdepth: 1
   :hidden:

   ./synthetic/generate_synthetic_image.rst


Acquisition
-----------

The generated image is the exact gray level of the texture: these functions simulate the defects of an acquisition,
and are applied in this order (blur of the optics, noise of the sensor, quantization of the gray levels).

- :doc:`add_gaussian_blur <synthetic/add_gaussian_blur>`: Gaussian blur of the image (optics, defocus).
- :doc:`add_gaussian_noise <synthetic/add_gaussian_noise>`: Additive Gaussian noise (electronic noise of the sensor).
- :doc:`add_poisson_noise <synthetic/add_poisson_noise>`: Poisson noise (photon noise), depending on the gray level.
- :doc:`add_salt_and_pepper_noise <synthetic/add_salt_and_pepper_noise>`: Random saturated pixels (dead or hot pixels).
- :doc:`quantize_image <synthetic/quantize_image>`: Clip and round the gray levels on a given number of bits (e.g. ``uint8``, ``uint16``).

.. code-block:: python

   from pysdic import add_gaussian_noise, generate_synthetic_image, quantize_image

   rendered = generate_synthetic_image(camera, mesh, texture, displacement=displacement)[..., 0]
   image = quantize_image(add_gaussian_noise(rendered, std=1.0, seed=0), bit_depth=8)

.. toctree::
   :maxdepth: 1
   :hidden:

   ./synthetic/add_gaussian_blur.rst
   ./synthetic/add_gaussian_noise.rst
   ./synthetic/add_poisson_noise.rst
   ./synthetic/add_salt_and_pepper_noise.rst
   ./synthetic/quantize_image.rst