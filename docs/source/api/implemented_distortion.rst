Implemented Distortion Transformations
======================================

The package ``pysdic`` provides a set of implemented distortion transformation models
that can be used to transform 2D **normalized points** to 2D **distorted points**:

.. code-block:: text

    world_points → normalized_points → distorted_points → image_points

- :doc:`NoDistortion <morphisms/NoDistortion>`: Identity transformation that does not apply any distortion (the **distorted points** are equal to the **normalized points**).
- :doc:`BrownDistortion <morphisms/BrownDistortion>`: Brown-Conrady distortion model as implemented by OpenCV, with radial, tangential, thin prism and tilt distortions (0, 4, 5, 8, 12 or 14 parameters).
- :doc:`ZernikeDistortion <morphisms/ZernikeDistortion>`: Distortion model adding a displacement field expanded on the Zernike polynomials along the two axes, defined on an ellipse containing the field of view.

The inverse transformation (undistortion) of the :class:`pysdic.BrownDistortion` and :class:`pysdic.ZernikeDistortion`
has no closed form and is computed with Newton iterations.

- :doc:`ZernikePolynomialsCache <morphisms/ZernikePolynomialsCache>`: Explicit cache of the Zernike polynomials of a :class:`pysdic.ZernikeDistortion`, reused when the same points are distorted with different coefficients (e.g. the calibration of the distortion alone).

.. toctree::
   :maxdepth: 1
   :hidden:

   morphisms/NoDistortion.rst
   morphisms/BrownDistortion.rst
   morphisms/ZernikeDistortion.rst
   morphisms/ZernikePolynomialsCache.rst