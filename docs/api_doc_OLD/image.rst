.. currentmodule:: pysdic.imaging

pysdic.imaging.Image
======================

.. contents:: Table of Contents
   :local:
   :depth: 1
   :backlinks: top


Image class
-------------------------------------------

.. autoclass:: Image


Instantiate an Image object
-------------------------------------------

To instantiate an Image object, you need to provide the pixel data as a 2D numpy array or the file path to an image.

.. autosummary::
   :toctree: ../generated/

   Image.from_array
   Image.from_file


Export Image data
-------------------------------------------

You can export the image data to a file or a numpy array.

.. autosummary::
   :toctree: ../generated/

   Image.to_array
   Image.to_file


Accessing Image attributes
-------------------------------------------

You can access the image attributes through the Image object using the following properties.

.. autosummary::
   :toctree: ../generated/

   Image.dtype
   Image.is_color
   Image.is_grayscale
   Image.image
   Image.height
   Image.n_channels
   Image.ndim
   Image.shape
   Image.width

If the image is modified externally, you must call the method ``image_update`` to refresh the image data.

.. autosummary::
   :toctree: ../generated/

   Image.image_update
   Image.construct_interpolation_functions


Operating with Image data
-------------------------------------------

You can use the image data to interpolate pixel values at specific coordinates.

.. autosummary::
   :toctree: ../generated/

   Image.copy
   Image.evaluate_image_at_image_points
   Image.evaluate_image_at_pixel_points
   Image.evaluate_image_jacobian_dx_at_pixel_points
   Image.evaluate_image_jacobian_dy_at_pixel_points
   Image.evaluate_image_jacobian_du_at_pixel_points
   Image.evaluate_image_jacobian_dv_at_pixel_points
   Image.evaluate_image_jacobian_dx_at_image_points
   Image.evaluate_image_jacobian_dy_at_image_points
   Image.evaluate_image_jacobian_du_at_image_points
   Image.evaluate_image_jacobian_dv_at_image_points
   Image.get_image_pixel_points
   Image.get_image_image_points
   Image.get_interpolation_function
   Image.image_points_to_pixel_points
   Image.pixel_points_to_image_points


Visualize Image data
-------------------------------------------

You can visualize the image using the built-in visualization method.

.. autosummary::
   :toctree: ../generated/

   Image.visualize