.. currentmodule:: pysdic.imaging

pysdic.imaging.Camera
======================

.. contents:: Table of Contents
   :local:
   :depth: 1
   :backlinks: top


Camera class
-------------------------------------------

.. autoclass:: Camera

Instantiate a Camera object
-------------------------------------------

To instantiate a Camera object, you need to provide the sensor dimensions (height and width in pixels), the intrinsic parameters (as a :class:`pycvcam.Cv2Intrinsic` object), the distortion parameters (as a :class:`pycvcam.Cv2Distortion` object) and the extrinsic parameters (as a :class:`pycvcam.Cv2Extrinsic` object).

Lets consider a simple OpenCV pinhole camera model without distortion.
The intrinsic parameters can be defined from the camera matrix, and the extrinsic parameters from the rotation and translation vectors.

.. code-block:: python

    from pycvcam import Cv2Extrinsic, Cv2Intrinsic, Cv2Distortion
    from pysdic.imaging import Camera
    import numpy

    rotation_vector = numpy.array([0.1, 0.2, 0.3])
    translation_vector = numpy.array([12.0, 34.0, 56.0])

    extrinsic = Cv2Extrinsic.from_rt(rotation_vector, translation_vector)

    intrinsic = Cv2Intrinsic.from_matrix(
        numpy.array([[1000, 0, 320],
                    [0, 1000, 240],
                    [0, 0, 1]])
    )

    camera = Camera(
        sensor_height=480,
        sensor_width=640,
        intrinsic=intrinsic,
        distortion=None,
        extrinsic=extrinsic,
    )

To load the extrinsic, intrinsic and distortion parameters from ``.json`` files, you can use the methods ``write_transform`` and ``read_transform`` from the package ``pycvcam`` (https://github.com/Artezaru/pycvcam).

Accessing Camera attributes
-------------------------------------------

You can access the camera attributes such as sensor dimensions, intrinsic, distortion and extrinsic parameters.

.. autosummary::
   :toctree: ../generated/

    Camera.sensor_height
    Camera.sensor_width
    Camera.distortion
    Camera.extrinsic
    Camera.intrinsic
    Camera.internal_bypass

Several cameras can have the same transformations (``Intrinsic``, ``Distortion``, ``Extrinsic``).
Any modification of these transformations will be reflected in all the cameras sharing them.

.. warning::

    If you use the ``internal_bypass = True`` feature to avoid checking the consistency of the parameters, the updates methods below must be called manually after any modification of the parameters on a transformations to propagate the changes to the camera.

.. autosummary::
   :toctree: ../generated/

    Camera.update
    Camera.intrinsic_update
    Camera.distortion_update
    Camera.extrinsic_update
    Camera.size_update


Manipulating Camera objects
-------------------------------------------

The camera can be used to project 3D points into the 2D image plane and to compute the rays emmited by the camera for each pixel.

.. autosummary::
   :toctree: ../generated/

    Camera.image_points_to_pixel_points
    Camera.get_camera_normalized_points
    Camera.get_camera_pixel_points
    Camera.get_camera_rays
    Camera.pixel_points_to_image_points
    Camera.project
    Camera.project_points


Visualize 2D Projections
-------------------------------------------

The Camera class provides a method to visualize the 2D projection of 3D points onto the image plane.

.. autosummary::
   :toctree: ../generated/

    Camera.visualize_projected_point_cloud
    Camera.visualize_projected_mesh

Usage
-----
Creating a camera with only intrinsic and extrinsic transformations:

.. code-block:: python

    import numpy
    from pysdic.imaging import Camera
    from pycvcam import Cv2Extrinsic, Cv2Intrinsic

    rotation_vector = numpy.array([0.1, 0.2, 0.3])
    translation_vector = numpy.array([12.0, 34.0, 56.0])

    extrinsic = Cv2Extrinsic.from_rt(rotation_vector, translation_vector)

    intrinsic = Cv2Intrinsic.from_matrix(
        numpy.array([[1000, 0, 320],
                    [0, 1000, 240],
                    [0, 0, 1]])
    )

    camera = Camera(
        sensor_height=480,
        sensor_width=640,
        intrinsic=intrinsic,
        extrinsic=extrinsic,
    )

To project 3D points into the 2D image plane, you can use the :meth:`project` method.

.. code-block:: python

    points_3d = numpy.array([[0, 0, 0],
                            [1, 1, 1],
                            [2, 2, 2]]) # world_points with shape (N, 3)

    result = camera.project(points_3d)
    image_points = result.image_points

.. warning::

    ``image_points`` are expressed in the (x,y) coordinate system and not as ``pixel_points`` in the (u,v) coordinate system.

You can also access the jacobian of the projection (``dx``, ``dintrinsic``, ``ddistortion``, ``dextrinsic``).

.. code-block:: python

    result = camera.project(points_3d, dintrinsic=True) # Compute also the jacobian of the projection with respect to the intrinsic parameters

    image_points = result.image_points
    jacobian = result.jacobian_dintrinsic

For reverse transformation, you can construct the rays emmited by the camera for each pixel by using the :meth:`get_camera_rays` method.
A mask with shape (height, width) can be used to filter the rays.
To construct the rays for any 2D image points, you can use the package ``pycvcam`` (https://github.com/Artezaru/pycvcam) and the method ``compute_rays``.

.. code-block:: python

    rays = camera.get_camera_rays(mask=mask)
    origins = rays[:, :3]  # shape (N, 3)
    directions = rays[:, 3:]  # shape (N, 3)

