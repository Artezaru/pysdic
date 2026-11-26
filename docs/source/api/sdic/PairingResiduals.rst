pysdic.PairingResiduals
=======================

.. currentmodule:: pysdic

.. contents:: Table of Contents
   :local:
   :depth: 2

The :class:`pysdic.PairingResiduals` stores the gray levels and the residuals of the material points (integration
points of the mesh) in the two views of a :class:`pysdic.Pairing`, as compared by a
:class:`pysdic.SDICCorrelationTerm` (gray level correction included, without the weights of the points):

.. math::

   r_p = g'_2(\vec{X}_p) - g'_1(\vec{X}_p)

with :math:`g'_k` the gray levels of the view :math:`k` after the gray level correction (the raw gray levels without
correction).


PairingResiduals class
----------------------

.. autoclass:: PairingResiduals
   :no-members:
   :show-inheritance:


Instantiation
-------------

A :class:`PairingResiduals` is a frozen dataclass returned by the functions below, not created by the user:

- :func:`pysdic.compute_residuals`: the residuals of a pairing for the **current state** of the analysis, with or
  without gray level correction;
- :meth:`pysdic.SDICCorrelationTerm.compute_residuals`: the residuals of the pairings of a term for the state given by
  a **complete** vector of unknowns :math:`\vec{p}` (by default the current state of the analysis), with the gray
  level correction of the term.

.. code-block:: python

   from pysdic import Pairing, compute_residuals, display_residual_maps

   raw = compute_residuals(analysis, Pairing.stereo(1, 0, 1), integration_points)       # Raw gray levels
   corrected = correlation.compute_residuals()                                         # One per pairing of the term

   print(raw.label, raw.n_points, raw.n_channels)    # "raw", N_p, C
   rms = numpy.sqrt(numpy.nanmean(raw.residuals**2))
   display_residual_maps(analysis, corrected[0], fill_radius=2.0)

.. note::

   The points projected out of the ROI of an image have NaN gray levels and NaN residuals: use the ``nan*``
   functions of numpy for the statistics. The RMS per element is computed by
   :func:`pysdic.workflows.compute_elements_residual_rms`.


Attributes
----------

The fields of the dataclass:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Content
   * - ``pairing``
     - The :class:`pysdic.Pairing` (the two views).
   * - ``gray_levels_1``, ``gray_levels_2``
     - The (corrected) gray levels in the two views, shape :math:`(N_p, C)`, NaN out of the ROI.
   * - ``residuals``
     - ``gray_levels_2 - gray_levels_1``, shape :math:`(N_p, C)`.
   * - ``image_points_1``, ``image_points_2``
     - The projections :math:`(x, y)` of the points in the two views (pixels), shape :math:`(N_p, 2)`.
   * - ``displacement_1``, ``displacement_2``
     - The (total) displacements :math:`(N_v, E)` of the frames of the two views used for the evaluation.
   * - ``gl_correction``
     - The gray level correction (complete options), or None for the raw gray levels.

And the derived properties:

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   PairingResiduals.n_points
   PairingResiduals.n_channels
   PairingResiduals.label


Residual maps
-------------

The residuals of the points are drawn as images in the two views of the pairing:

- :func:`pysdic.compute_pixel_residual_maps`: the residuals splatted on the pixels of each view (arrays of the size
  of the images);
- :func:`pysdic.display_residual_maps`: the two maps displayed side by side.
