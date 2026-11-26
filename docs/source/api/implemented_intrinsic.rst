Implemented Intrinsic Transformations
=====================================

The ``pysdic`` package provides a set of intrinsic transformation models
for mapping 2D **distorted points** to 2D **image points** (in pixels):

.. code-block:: text

    world_points → normalized_points → distorted_points → image_points

- :doc:`NoIntrinsic <morphisms/NoIntrinsic>`
  Identity transformation: no intrinsic mapping is applied. The output
  **image points** are identical to the input **distorted points**.

- :doc:`PinholeIntrinsic <morphisms/PinholeIntrinsic>`
  OpenCV-style intrinsic transformation: applies a linear mapping to the
  **distorted points** using the intrinsic matrix :math:`K` with the focal lengths
  :math:`(f_x, f_y)`, the principal point :math:`(c_x, c_y)` and an optional
  skew :math:`s` (``use_skew=True``).

.. toctree::
  :maxdepth: 1
  :hidden:

  morphisms/NoIntrinsic.rst
  morphisms/PinholeIntrinsic.rst
