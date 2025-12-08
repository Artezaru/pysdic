.. currentmodule:: pysdic

Bidirectional Reflectance Distribution Function (BRDF) Lighting Models
=========================================================================

.. contents:: Table of Contents
   :local:
   :depth: 2
   :backlinks: top

.. seealso::

    :doc:`./photometric_quantities` to understand the photometric quantities used in lighting models.



Mathematical Definition
---------------------------------------------------------------

The package ``pysdic`` provides implementations of various Bidirectional Reflectance Distribution Function (BRDF) lighting models.

For a given surface, the BRDF is defined as the ratio of the radiance in the direction :math:`(\theta_o, \phi_o)` to the irradiance from direction :math:`(\theta_i, \phi_i)`.

.. math::

    BRDF(\theta_i, \phi_i, \theta_o, \phi_o) = \frac{dL_o(\theta_o, \phi_o)}{dE_i(\theta_i, \phi_i)}

where :math:`L_o` is the outgoing radiance in Watts per steradian per square meter and :math:`E_i` is the incoming irradiance in Watts per square meter.

This package includes the following BRDF models:


Function signatures
--------------------

All BRDF model functions follow a similar interface.

.. note::

    By default, the functions return a 3D array of shape (:math:`N_p`, :math:`N_l`, :math:`N_o`) when the inputs are 2D arrays of shapes (:math:`N_p`, :math:`E`), (:math:`N_l`, :math:`E`), and (:math:`N_o`, :math:`E`).

    If any of the input arrays are 1D arrays of shape (:math:`E`,), they are treated as single positions and the corresponding output array dimensions are omitted.

    - If :obj:`surface_points` and :obj:`surface_normals` are 1D arrays, the first dimension is omitted (shape (:math:`N_l`, :math:`N_o`)).
    - If :obj:`light_positions` is a 1D array, the second dimension is omitted (shape (:math:`N_p`, :math:`N_o`)).
    - If :obj:`observer_positions` is a 1D array, the third dimension is omitted (shape (:math:`N_p`, :math:`N_l`)).
    - If :obj:`light_positions` and :obj:`observer_positions` are 1D arrays, both the second and third dimensions are omitted (shape (:math:`N_p`,)).
    - ...
    - If all inputs are 1D arrays, the output is a scalar value.


Parameters
~~~~~~~~~~
- **surface_points** : :class:`numpy.ndarray` of shape (:math:`N_p`, :math:`E`) or (:math:`E`,)
    An array of :math:`N_p` points on the surface where the BRDF is to be evaluated.

- **surface_normals** : :class:`numpy.ndarray` of shape (:math:`N_p`, :math:`E`) or (:math:`E`,)
    An array of :math:`N_p` normal vectors at the surface points.

- **light_positions** : :class:`numpy.ndarray` of shape (:math:`N_l`, :math:`E`) or (:math:`E`,)
    An array of :math:`N_l` positions of light sources.

- **observer_positions** : :class:`numpy.ndarray` of shape (:math:`N_o`, :math:`E`) or (:math:`E`,)
    An array of :math:`N_o` positions of observers or cameras.

- **diffuse_coefficient** : :class:`float`
    The diffuse reflection coefficient of the surface.

- **specular_coefficient** : :class:`float`
    The specular reflection coefficient of the surface.

- **other model-specific parameters**
    Additional parameters specific to each BRDF model (e.g., roughness, shininess, rms).

Returns
~~~~~~~
- **brdf_values** : Union[:class:`numpy.ndarray`, :class:`float`] of shape (:math:`N_p`, :math:`N_l`, :math:`N_o`)
    An array containing the computed BRDF values at each surface point for each combination of light source and observer.
    The shape of the output array is adjusted based on whether :obj:`surface_points`, :obj:`light_positions`, or :obj:`observer_positions` are provided as 1D arrays.


Implemented BRDF Models
------------------------------

.. autosummary::
   :toctree: ../generated/

   compute_BRDF_ward
   compute_BRDF_beckmann


