pysdic.ZernikeDistortion
========================
 
.. currentmodule:: pysdic
 
.. contents:: Table of Contents
   :local:
   :depth: 2
 
 
ZernikeDistortion class
-----------------------
 
.. autoclass:: ZernikeDistortion
   :no-members:
   :show-inheritance:
 
Instantiation
-------------
 
The :class:`ZernikeDistortion` class can be instantiated in two ways:
 
- **From a parameter vector** of shape ``(N_params,)`` with :math:`N_{params} = (N_{zer}+1)(N_{zer}+2)` passed to the constructor.
- **From a Zernike order** ``n_zernike`` (or a number of parameters ``n_parameters``) passed to the constructor, then initialized with zeros using :meth:`set_zero`.
 
The ellipse of definition is set at the instantiation with the constants ``radius_x``, ``radius_y`` and ``center``.
A cache of the Zernike polynomials can be given with ``polynomials_cache`` (see :doc:`ZernikePolynomialsCache <ZernikePolynomialsCache>`).
 
.. autosummary::
   :toctree: ./_generated/
   :signatures: long
 
   ZernikeDistortion.set_zero
 
Implemented methods and attributes
----------------------------------
 
The following sections provide an overview of the attributes and methods defined
in the :class:`pysdic.ZernikeDistortion` class.
 
Attributes
^^^^^^^^^^
 
The distortion parameters can be accessed via the base class
:class:`pysdic.Transformation`:
 
- ``parameters``: returns a vector of distortion parameters ``[Cx(0, 0), Cy(0, 0), Cx(1, -1), Cy(1, -1), ...]`` (OSA/ANSI order, alternating x and y).
 
The :class:`pysdic.ZernikeDistortion` defines the following attributes:
 
.. autosummary::
   :toctree: ./_generated/
   :signatures: long
 
   ZernikeDistortion.center
   ZernikeDistortion.center_x
   ZernikeDistortion.center_y
   ZernikeDistortion.n_zernike
   ZernikeDistortion.parameter_x_names
   ZernikeDistortion.parameter_y_names
   ZernikeDistortion.parameters_x
   ZernikeDistortion.parameters_y
   ZernikeDistortion.polynomials_cache
   ZernikeDistortion.radius
   ZernikeDistortion.radius_x
   ZernikeDistortion.radius_y
 
The individual Zernike coefficients can be accessed with the following methods:
 
.. autosummary::
   :toctree: ./_generated/
   :signatures: long
 
   ZernikeDistortion.get_index
   ZernikeDistortion.get_Cx
   ZernikeDistortion.get_Cy
   ZernikeDistortion.set_Cx
   ZernikeDistortion.set_Cy
 
 
Performing Projections
----------------------
 
Use the ``apply`` and ``invert`` methods from
:class:`pysdic.Transformation` to perform distortion projections:
 
- ``apply``: maps ``normalized_points`` → ``distorted_points`` (NaN outside the ellipse of definition).
- ``invert``: maps ``distorted_points`` → ``normalized_points`` (Newton iterations, or ``pysolvegn`` with the ``pysolvegn`` flag).
 
Jacobians with respect to the input points and distortion parameters
can be computed by enabling the ``dx`` and ``dp`` flags:
 
- ``dx=True``: compute ∂distorted_points / ∂normalized_points
- ``dp=True``: compute ∂distorted_points / ∂distortion_parameters
 
.. autosummary::
   :toctree: ./_generated/
   :signatures: long
 
   ZernikeDistortion._apply
   ZernikeDistortion._invert
 
 
Cache of the Zernike polynomials
--------------------------------
 
The Zernike polynomials only depend on the points and on the ellipse of definition, not on the coefficients: when the
distortion is applied several times to the **same points** (e.g. the calibration of the distortion alone), they can be
computed once and reused with a :class:`pysdic.ZernikePolynomialsCache`, attached with the property
``polynomials_cache`` (no cache by default).
 
.. seealso::
 
   :doc:`ZernikePolynomialsCache <ZernikePolynomialsCache>` for the usage, the rules and the memory of the cache.
 
