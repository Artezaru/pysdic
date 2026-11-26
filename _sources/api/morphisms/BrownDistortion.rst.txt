pysdic.BrownDistortion
========================

.. currentmodule:: pysdic

.. contents:: Table of Contents
   :local:
   :depth: 2


BrownDistortion class
----------------------

.. autoclass:: BrownDistortion
   :no-members:
   :show-inheritance:

Instantiation
-------------

The :class:`BrownDistortion` class can be instantiated in two ways:

- **From a parameter vector** of shape ``(N,)`` with :math:`N \in \{0, 4, 5, 8, 12, 14\}` passed to the constructor.
- **From a number of parameters** ``n_parameters`` passed to the constructor, then initialized with zeros using :meth:`set_zero`.

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   BrownDistortion.set_zero

Implemented methods and attributes
------------------------------------

The following sections provide an overview of the attributes and methods defined
in the :class:`pysdic.BrownDistortion` class.

Attributes
^^^^^^^^^^

The distortion parameters can be accessed via the base class
:class:`pysdic.Transformation`:

- ``parameters``: returns a vector of distortion parameters, the first :math:`N` of
  ``[k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, tau_x, tau_y]``.

The :class:`pysdic.BrownDistortion` defines the following attributes
(each coefficient is only available if the model has enough parameters):

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   BrownDistortion.k1
   BrownDistortion.k2
   BrownDistortion.k3
   BrownDistortion.k4
   BrownDistortion.k5
   BrownDistortion.k6
   BrownDistortion.p1
   BrownDistortion.p2
   BrownDistortion.s1
   BrownDistortion.s2
   BrownDistortion.s3
   BrownDistortion.s4
   BrownDistortion.tau_x
   BrownDistortion.tau_y


Performing Projections
----------------------

Use the ``apply`` and ``invert`` methods from
:class:`pysdic.Transformation` to perform distortion projections:

- ``apply``: maps ``normalized_points`` → ``distorted_points``.
- ``invert``: maps ``distorted_points`` → ``normalized_points`` (Newton iterations, or OpenCV / ``pysolvegn`` with the ``opencv`` and ``pysolvegn`` flags).

Jacobians with respect to the input points and distortion parameters
can be computed by enabling the ``dx`` and ``dp`` flags:

- ``dx=True``: compute ∂distorted_points / ∂normalized_points
- ``dp=True``: compute ∂distorted_points / ∂distortion_parameters

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   BrownDistortion._apply
   BrownDistortion._invert
