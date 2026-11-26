Imaging Model (Cameras, Images and Views)
=========================================

.. currentmodule:: pysdic

.. contents:: Table of Contents
   :local:
   :depth: 1

The imaging model of ``pysdic`` combines the camera models (see :doc:`Computer Vision Cameras Models <camera_models>`)
with the images acquired by the cameras, to project the geometry of the observed surfaces (see :doc:`Geometry <geometry>`)
to the gray levels of the images, as required by the SDIC.

.. code-block:: text

   world_points --[Camera]--> image_points --[Image]--> gray_levels
                \______________________ View ______________________/


Coordinate systems of the images
--------------------------------

Two coordinate systems are used for the 2D points of an image of height :math:`H` and width :math:`W`:

- ``image_points``: the points :math:`(x, y)` of the **image coordinate system** returned by the cameras,
  with :math:`x` the column index and :math:`y` the row index.
- ``pixel_points``: the points :math:`(u, v)` of the **pixel coordinate system** of the arrays,
  with :math:`u` the row index and :math:`v` the column index.

The centers of the pixels are at integer coordinates, so the sensor covers
:math:`[-0.5, W - 0.5] \times [-0.5, H - 0.5]` in image points. The conversion between both systems
is a swap of the coordinates (see :meth:`pysdic.Image.pixel_points_to_image_points` and
:meth:`pysdic.Image.image_points_to_pixel_points`).


Cameras, images and views
-------------------------

- :doc:`Camera <imaging/Camera>`: A camera model: the composition :math:`\text{Intrinsic} \circ \text{Distortion} \circ \text{Extrinsic}` with the size of its sensor, to project world points and compute the rays of the pixels.
- :doc:`Image <imaging/Image>`: An image (grayscale or color) with interpolation functions to evaluate its values and derivatives at any point of a region of interest.
- :doc:`View <imaging/View>`: A camera and an image acquired by this camera, to project world points to the gray levels of the image with their Jacobians.
- :doc:`ImageProjectionResult <imaging/ImageProjectionResult>`: Result of :meth:`pysdic.View.image_project`, with the blocks of the Jacobian for the extrinsic, distortion and intrinsic parameters.

.. toctree::
   :maxdepth: 1
   :hidden:

   ./imaging/Camera.rst
   ./imaging/Image.rst
   ./imaging/View.rst
   ./imaging/ImageProjectionResult.rst


Projection of meshes
--------------------

Compute quantities of a surface mesh (see :class:`pysdic.Mesh`) seen by a camera (or a view).

- :doc:`compute_projected_bounding_box <imaging/compute_projected_bounding_box>`: Bounding box ``(x_min, y_min, x_max, y_max)`` of the projected vertices, with an optional margin (fast restriction of a computation on the pixels).
- :doc:`compute_projected_roi <imaging/compute_projected_roi>`: The same box in integer pixels, clipped to the sensor, in the ROI format ``(x, y, width, height)`` of :class:`pysdic.Image`.
- :doc:`compute_projected_uvmap <imaging/compute_projected_uvmap>`: UV map :math:`(N_e, 2 N_{vpe})` of the projected vertices on the image (OpenGL convention, NaN out of the image), to use the image of a camera as the texture of the mesh.
- :doc:`compute_projected_mesh_mask <imaging/compute_projected_mesh_mask>`: Boolean mask of the pixels whose center is inside the projection of the mesh (without the occlusions).
- :doc:`compute_projected_elements_areas <imaging/compute_projected_elements_areas>`: Area in pixels² of the projection of each element on the sensor (optionally signed, to detect the elements seen from the back).
- :doc:`compute_visibility <imaging/compute_visibility>`: Visibility of integration points: projected on the sensor (and in the ROI of the image for a view) and not hidden by the mesh.
- :doc:`compute_pixel_integration_points <imaging/compute_pixel_integration_points>`: Material points of the mesh seen by the pixels of the camera, as integration points (with the occlusions).

.. important::

   :func:`pysdic.compute_visibility` and :func:`pysdic.compute_pixel_integration_points` (3D case) use the ray casting
   of :meth:`pysdic.Mesh.cast_rays`, which requires the optional package ``open3d`` (``pip install open3d``),
   a 3D surface mesh and a camera with 3D world points. :func:`pysdic.compute_pixel_integration_points` also accepts a
   2D mesh (inversion of the camera, no optional dependency).

.. toctree::
   :maxdepth: 1
   :hidden:

   ./imaging/compute_projected_bounding_box.rst
   ./imaging/compute_projected_roi.rst
   ./imaging/compute_projected_uvmap.rst
   ./imaging/compute_projected_mesh_mask.rst
   ./imaging/compute_projected_elements_areas.rst
   ./imaging/compute_visibility.rst
   ./imaging/compute_pixel_integration_points.rst


Display of the projections
--------------------------

One display function per computation of the projection of meshes, to check the results visually.
The 2D displays draw on the sensor of a camera, or on the image of a view, with a :class:`pysdic.ImagePlotter`
and return the ``matplotlib`` figure. The 3D displays use a :class:`pysdic.Plotter3D` (``pip install pyvista``)
and return it.

- :doc:`display_projected_bounding_box <imaging/display_projected_bounding_box>`: The projected mesh with its bounding box drawn as a rectangle.
- :doc:`display_projected_roi <imaging/display_projected_roi>`: The projected mesh with its ROI drawn along the borders of its pixels.
- :doc:`display_projected_mesh_mask <imaging/display_projected_mesh_mask>`: The image of a view seen through the projected mesh (the pixels out of the mask are not drawn), or the mask itself for a camera.
- :doc:`display_projected_elements_areas <imaging/display_projected_elements_areas>`: The projected mesh colored by the area in pixels² of its elements, optionally written in each element with the density of the integration points (points per pixel).
- :doc:`display_visibility <imaging/display_visibility>`: 3D display of the mesh with the visible integration points in blue and the other ones in red, and the center of the camera.
- :doc:`display_pixel_integration_points <imaging/display_pixel_integration_points>`: 3D display of the material points seen by the pixels on the mesh, colored by the gray levels of the image for a view.

.. toctree::
   :maxdepth: 1
   :hidden:

   ./imaging/display_projected_bounding_box.rst
   ./imaging/display_projected_roi.rst
   ./imaging/display_projected_mesh_mask.rst
   ./imaging/display_projected_elements_areas.rst
   ./imaging/display_visibility.rst
   ./imaging/display_pixel_integration_points.rst


Visualization
-------------

- :doc:`ImagePlotter <imaging/ImagePlotter>`: Draw images, the sensor of a camera and the (projected) point clouds, meshes and integration points in the image coordinate system with ``matplotlib``.

The ``quick_visualize`` methods of :class:`pysdic.Camera`, :class:`pysdic.Image` and :class:`pysdic.View`
give a fast preview without creating a plotter.

.. toctree::
   :maxdepth: 1
   :hidden:

   ./imaging/ImagePlotter.rst