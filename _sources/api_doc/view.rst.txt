.. currentmodule:: pysdic.imaging

pysdic.imaging.View
======================

.. contents:: Table of Contents
   :local:
   :depth: 1
   :backlinks: top


View class
-------------------------------------------

.. autoclass:: View

Instantiate a View object
-------------------------------------------

To instantiate a View object, you need to provide the camera (as a :class:`Camera` object`) and the image (as a :class:`numpy.ndarray` object).

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

The view can be instantiated with an image as follows:

.. code-block:: python

    import numpy
    from pysdic.imaging import View, Image

    image = Image.from_array(numpy.zeros((480, 640, 3), dtype=numpy.uint8))

    view = View(
        camera=camera,
        image=image,
    )

To load the extrinsic, intrinsic and distortion parameters from ``.json`` files, you can use the methods ``write_transform`` and ``read_transform`` from the package ``pycvcam`` (https://github.com/Artezaru/pycvcam).

Accessing View attributes
-------------------------------------------

You can access the view attributes such as camera and image.

.. autosummary::
   :toctree: ../generated/

    View.camera
    View.image
    View.image_shape
    View.camera_size

Manipulating View objects
-------------------------------------------

The view can be used to project 3D points into the 2D image plane.

.. autosummary::
   :toctree: ../generated/

    View.project
    View.project_points
    View.image_project
    View.image_project_points


Visualize 2D Projections
-------------------------------------------

The View class provides a method to visualize the 2D projection of 3D points onto the image plane.

.. autosummary::
   :toctree: ../generated/

    View.visualize_projected_point_cloud
    View.visualize_projected_mesh
