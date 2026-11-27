SDIC Analysis
=============

.. currentmodule:: pysdic

.. contents:: Table of Contents
   :local:
   :depth: 1

A stereo digital image correlation (SDIC) analysis measures the displacement of a surface, described by a mesh
(see :doc:`Geometry <geometry>`), from the images acquired by one or several cameras (see :doc:`Imaging Model <imaging>`)
at several frames.

.. code-block:: text

   SDICAnalysis = reference mesh + displacement of each frame + views (camera, image) of each frame
   SDICUnknowns = the optimized frames + the optimized transformations (does not know the analysis)
   Pairing      = (frame_1, view_1) -> (frame_2, view_2)    # Two views whose gray levels are compared
   SDICCorrelationTerm(analysis, unknowns, integration_points, pairings)   # The pysolvegn.Term (property term)

For a pairing, the gray levels of the material points :math:`\vec{X}` of the mesh are compared between the two views:

.. math::

   r(\vec{X}) = [I_2 \circ P_2](\vec{X} + \vec{u}_2(\vec{X})) - [I_1 \circ P_1](\vec{X} + \vec{u}_1(\vec{X}))

where :math:`\vec{u}_1` and :math:`\vec{u}_2` are the (total) displacements of the frames, :math:`P_1`, :math:`P_2`
the projections of the cameras and :math:`I_1`, :math:`I_2` the images of the views.


The analysis
------------

- :doc:`SDICAnalysis <sdic/SDICAnalysis>`: The data of the analysis: the reference mesh, the displacement of its vertices at each frame and the views acquired at each frame, with the structure of the objects shared between the views.

.. toctree::
   :maxdepth: 1
   :hidden:

   ./sdic/SDICAnalysis.rst


Acquisition setups
------------------

Create a :class:`pysdic.SDICAnalysis` with the views of the usual acquisition setups. The displacements of the frames
are not set.

- :doc:`create_multicamera_analysis <sdic/create_multicamera_analysis>`: :math:`N` cameras that each acquire their own series of images (view :math:`i` of the frame :math:`f` = ``View(cameras[i], images[i][f])``).
- :doc:`create_bicamera_analysis <sdic/create_bicamera_analysis>`: Stereo setup of two cameras that each acquire their own series of images.
- :doc:`create_splitscreen_analysis <sdic/create_splitscreen_analysis>`: :math:`N` virtual cameras (stereo mirror adapter) sharing one sensor and one series of images.
- :doc:`create_cordin_analysis <sdic/create_cordin_analysis>`: Cordin-like high-speed camera: one sensor seen through :math:`N` points of view, with a distortion changing with the frame (rotating mirror).

.. note::

   The cameras (and for :func:`pysdic.create_cordin_analysis` the transformations) are shared between the views, not
   copied: use :meth:`pysdic.SDICAnalysis.display_structure` to see which objects are shared, and
   :meth:`pysdic.SDICAnalysis.get_transformation` to access one of them.

.. toctree::
   :maxdepth: 1
   :hidden:

   ./sdic/create_multicamera_analysis.rst
   ./sdic/create_bicamera_analysis.rst
   ./sdic/create_splitscreen_analysis.rst
   ./sdic/create_cordin_analysis.rst


Unknowns, pairings and correlation term
---------------------------------------

The unknowns of the analysis are optimized with ``pysolvegn``. The vector of unknowns
:math:`\vec{p} = [\vec{p}_U, \vec{p}_T]` is made of the **total** displacements (from the reference mesh, not
increments) of the optimized frames, each one in Fortran order, in the order of the frames, and of the parameters of
the optimized transformations of the cameras, given as ``(kind, index)`` (see
:meth:`pysdic.SDICAnalysis.display_structure`):

- **measurement**: only optimized frames, the cameras are fixed (``SDICUnknowns([1], [])``);
- **calibration**: only optimized transformations, the displacements of all the frames are fixed
  (``SDICUnknowns([], [("extrinsic", 1)])``);
- **joint optimization**: both.

The objects of a problem are created in this order:

1. :class:`pysdic.SDICUnknowns`, **created by the user**: the optimized frames and transformations, in their order.
   As a :class:`pysdic.Pairing`, it **does not know the analysis**: the analysis validates it
   (:meth:`pysdic.SDICAnalysis.validate_unknowns`) and builds the vector :math:`\vec{p}`
   (:meth:`pysdic.SDICAnalysis.pack_unknowns`, :meth:`pysdic.SDICAnalysis.unpack_unknowns`,
   :meth:`pysdic.SDICAnalysis.apply_unknowns`).
2. :class:`pysdic.Pairing`: the views to compare.
3. :class:`pysdic.SDICCorrelationTerm` ``(analysis, unknowns, integration_points, pairings, ...)``: compares the gray
   levels of material points (integration points of the mesh) between the views of the pairings. Its
   ``pysolvegn.Term`` is the property ``term``.

All the objects of a problem (correlation term, temporal regularization, fixed parametrization) receive the **same
analysis object and the same unknowns object**: :func:`pysdic.check_same_problem` checks it (by identity) before
solving. ``pysolvegn`` only sees functions: two terms on different unknowns of the same size would be added without
error.

.. important::

   The terms (:class:`pysdic.SDICCorrelationTerm`, :class:`pysdic.SDICTemporalRegularizationTerm`) **do not build the
   vectors of unknowns**. The vectors are built only by the
   analysis (complete vector :math:`\vec{p}`: ``pack_unknowns``, ``unpack_unknowns``, ``apply_unknowns``) and by the
   parametrization (reduced vector :math:`\vec{q}`: ``reparametrized_pack``, ``reparametrized_unpack``,
   ``reparametrized_apply``).

.. seealso::

   The optimization problems are solved with the Gauss-Newton least squares solver ``pysolvegn``
   (`pysolve-gn on GitHub <https://github.com/Artezaru/pysolve-gn>`_, ``pip install pysolvegn``): a ``pysolvegn.Term``
   gives the residuals and the Jacobian (mode ``"rJ"``) or the gradient and the Hessian (mode ``"gH"``) of a part of the
   cost, with its weight and its loss function, and several terms with the same unknowns can be combined in the same
   problem (e.g. the correlation term and a regularization term). See the documentation of ``pysolvegn`` for the solver
   options (``max_iteration``, stopping criteria, damping, ...).

.. code-block:: python

   import pysolvegn
   from pysdic import Pairing, SDICCorrelationTerm, SDICUnknowns

   unknowns = SDICUnknowns([1], [("extrinsic", 1)])
   pairings = [Pairing.temporal(0, 0, 1), Pairing.temporal(1, 0, 1), Pairing.stereo(0, 0, 1), Pairing.stereo(1, 0, 1)]
   correlation = SDICCorrelationTerm(analysis, unknowns, integration_points, pairings,
                                     gl_correction={"mode": "ZNSSD", "scale": 255}, use_cache=True)

   result = pysolvegn.solve(terms=[correlation.term], p0=analysis.pack_unknowns(unknowns), max_iteration=20)
   analysis.apply_unknowns(unknowns, result.parameters)  # Displacement of the frame 1 and extrinsic 1 written in the analysis

   # Next frame: new unknowns, new term (same integration points and options)
   unknowns_2 = SDICUnknowns([2], [("extrinsic", 1)])
   correlation_2 = correlation.with_unknowns(unknowns_2, pairings=pairings_2)
   result = pysolvegn.solve(terms=[correlation_2.term], p0=analysis.pack_unknowns(unknowns_2), max_iteration=20)
   analysis.apply_unknowns(unknowns_2, result.parameters)

- :doc:`SDICUnknowns <sdic/SDICUnknowns>`: The description of the unknowns of a problem: the optimized frames and transformations (the vector :math:`\vec{p}` is built by the analysis).
- :doc:`check_same_problem <sdic/check_same_problem>`: Check that the objects of a problem share the same analysis and the same unknowns objects.
- :doc:`Pairing <sdic/Pairing>`: A link between two views of the analysis (stereo pairing at the same frame, or temporal pairing of the same view at two frames).
- :doc:`SDICCorrelationTerm <sdic/SDICCorrelationTerm>`: The correlation term on the unknowns (modes ``"rJ"`` and ``"gH"``, gray level corrections, caches), immutable: new terms with ``with_unknowns``, ``with_pairings``, ``with_integration_points`` and ``with_options``.

.. note::

   An evaluation of the term never modifies the analysis, it reads its current state (displacements of the reference
   frames, parameters of the transformations): apply a result with :meth:`pysdic.SDICAnalysis.apply_unknowns`. A term cannot be
   modified: the ``with_*`` methods create a new term (new caches, new structures). The objects built on other
   unknowns (regularization, parametrization) must be built on the same new unknowns (their own ``with_unknowns``).

.. note::

   With ``use_cache=True``, the term keeps the operators of its last evaluation (returned again for the same
   parameters if nothing has changed in the analysis) and the projections of its reference views (views whose frame
   and camera have no unknown, projected again only when their displacement or their camera change). Without cache,
   everything is computed again at each call. A new term (``with_*``) starts with empty caches.

.. toctree::
   :maxdepth: 1
   :hidden:

   ./sdic/SDICUnknowns.rst
   ./sdic/check_same_problem.rst
   ./sdic/Pairing.rst
   ./sdic/SDICCorrelationTerm.rst


Residuals
---------

The residuals of a pairing at the integration points, and their maps on the images, to check a measure (see also
:func:`pysdic.workflows.compute_elements_residual_rms` for the RMS per element):

- :doc:`PairingResiduals <sdic/PairingResiduals>`: The gray levels and the residuals of the points in the two views of a pairing (frozen dataclass).
- :doc:`compute_residuals <sdic/compute_residuals>`: The residuals of a pairing for the current state of the analysis, with or without gray level correction.
- :doc:`compute_pixel_residual_maps <sdic/compute_pixel_residual_maps>`: The residuals splatted on the pixels of the two views.
- :doc:`display_residual_maps <sdic/display_residual_maps>`: Display the residual maps of the two views.

.. code-block:: python

   from pysdic import compute_residuals, display_residual_maps

   raw = compute_residuals(analysis, Pairing.stereo(1, 0, 1), integration_points)  # Current state, raw gray levels
   corrected = correlation.compute_residuals()  # One PairingResiduals per pairing, correction of the term
   display_residual_maps(analysis, corrected[0], fill_radius=2.0)

.. toctree::
   :maxdepth: 1
   :hidden:

   ./sdic/PairingResiduals.rst
   ./sdic/compute_residuals.rst
   ./sdic/compute_pixel_residual_maps.rst
   ./sdic/display_residual_maps.rst


Gray level corrections
----------------------

The gray level corrections are applied by the option ``gl_correction`` of :class:`pysdic.SDICCorrelationTerm`, or directly with the functions below.
The gray levels of the same material point can differ between the two views of a pairing (lighting, exposure,
reflections, ...). The corrections remove an affine change of the gray levels, estimated by the mean :math:`m` and
the standard deviation :math:`s` over groups of integration points:

- ``ZNSSD`` (zero-mean normalized): both views are normalized, :math:`g' = \text{scale} \cdot (g - m) / s`.
- ``matching``: the second view is mapped onto the statistics of the first one,
  :math:`g_2' = \text{scale} \cdot ((g_2 - m_2) \, s_1 / s_2 + m_1)` and :math:`g_1' = \text{scale} \cdot g_1`.

.. warning::

   With several pairings, the ``matching`` correction keeps the contrast of the first view of each pairing: the
   pairings no longer have the same weight in the optimization. Prefer ``ZNSSD`` when several pairings are combined.

The statistics depend on the gray levels, so on the unknowns. Two derivatives are available for each correction:

- **approximate** (default): :math:`m` and :math:`s` are considered constant, the Jacobians of the gray levels are
  only multiplied by the factor of the correction and remain sparse (one point per row).
- **exact**: the derivatives of :math:`m` and :math:`s` are included. The corrected projections are returned without
  Jacobians, with a sparse matrix ``derivative`` to apply to the raw Jacobians of both views.

The groups of points are created with :func:`pysdic.compute_gray_level_groups`: a single group (global correction)
or one group per element with the points of its neighborhood (local correction). They only depend on the mesh and on
the integration points: by default a correlation term computes them from ``gl_correction["neighborhood"]``, but
they can be computed once and given to several terms with ``gray_level_groups`` (e.g. one calibration per frame;
the terms created with ``with_*`` share them).

.. code-block:: python

   from pysdic import compute_gray_level_groups

   groups = compute_gray_level_groups(analysis.mesh, integration_points, neighborhood=1)  # Once
   correlation = SDICCorrelationTerm(analysis, unknowns, integration_points, pairings,
                                     gl_correction={"mode": "ZNSSD"}, gray_level_groups=groups)

- :doc:`GrayLevelGroups <sdic/GrayLevelGroups>`: The groups of integration points used to compute the statistics of the corrections.
- :doc:`compute_gray_level_groups <sdic/compute_gray_level_groups>`: Compute the groups of points, global or by neighborhood of the elements.
- :doc:`znssd_correction <sdic/znssd_correction>`: Normalize the gray levels of both views (ZNSSD), approximate derivative.
- :doc:`znssd_correction_exact <sdic/znssd_correction_exact>`: Normalize the gray levels of both views (ZNSSD), exact derivative.
- :doc:`matching_correction <sdic/matching_correction>`: Map the gray levels of the second view onto the first one, approximate derivative.
- :doc:`matching_correction_exact <sdic/matching_correction_exact>`: Map the gray levels of the second view onto the first one, exact derivative.

.. note::

   The non-finite gray levels (points out of the ROI) are ignored by the statistics, and a group with a zero standard
   deviation gives non-finite corrected values. The exact derivative has :math:`\sum_G |G|^2` entries per channel:
   a global group gives a dense matrix, limited to small numbers of points.

.. toctree::
   :maxdepth: 1
   :hidden:

   ./sdic/GrayLevelGroups.rst
   ./sdic/compute_gray_level_groups.rst
   ./sdic/znssd_correction.rst
   ./sdic/znssd_correction_exact.rst
   ./sdic/matching_correction.rst
   ./sdic/matching_correction_exact.rst


Temporal regularization
-----------------------

The displacements of the optimized frames can be regularized along the time by penalizing their temporal derivative
of order :math:`n` (1 for the velocity, 2 for the acceleration). :class:`pysdic.SDICTemporalRegularizationTerm` builds
a ``pysolvegn.Term`` (property ``term``) on an analysis and a :class:`pysdic.SDICUnknowns` (zero Jacobian for the
transformations): with the analysis and the unknowns of the correlation term, both terms are combined in the same
problem:

.. math::

   r_{i,v,e} = \sqrt{w_e \, \omega_i} \left( \sum_{k} c_{i,k} \, u_{f_k, v, e} - a_{f_i, v, e} \right)

- The frames of the regularization are the optimized frames and the fixed frames whose displacement is set (by
  default), sorted by their ``times``. The frames without displacement are not given: the step between their
  neighbors is larger. No displacement is interpolated or extrapolated.
- The finite difference coefficients :math:`c_{i,k}` are computed for the actual times of the frames (Fornberg),
  with centered stencils inside, the most centered stencils around larger steps and one-sided stencils at the first and
  last frames (see :func:`pysdic.numeric.assemble_finite_difference_matrix`). With ``max_gap``, no stencil spans a larger
  step.
- Only the equations involving an optimized frame are kept, including those of the fixed frames next to the
  optimized ones (e.g. the reference frame).
- The weights :math:`w_e` of the components and :math:`\omega_i` of the frames (``row_weights="time"``: the duration
  represented by each frame, so that the cost approximates :math:`\int \| \vec{u}^{(n)} - \vec{a} \|^2 dt`) balance
  the equations.

.. code-block:: python

   import pysolvegn
   from pysdic import SDICCorrelationTerm, SDICTemporalRegularizationTerm, SDICUnknowns

   unknowns = SDICUnknowns(list(range(1, 10)), [])
   correlation = SDICCorrelationTerm(analysis, unknowns, integration_points, pairings)
   regularization = SDICTemporalRegularizationTerm(analysis, unknowns, order=2, weight=1e-2)

   result = pysolvegn.solve([correlation.term, regularization.term], analysis.pack_unknowns(unknowns), max_iteration=20)
   analysis.apply_unknowns(unknowns, result.parameters)

   softer = regularization.with_options(order=1, weight=1e-1)  # A new term, same unknowns

- :doc:`SDICTemporalRegularizationTerm <sdic/SDICTemporalRegularizationTerm>`: The term penalizing the temporal derivative of the displacements of the optimized frames (linear residual, constant Jacobian), immutable: new terms with ``with_unknowns`` and ``with_options``.

.. note::

   The unknowns of the regularization are fixed at its creation: with new unknowns, create a new regularization with
   ``regularization.with_unknowns(unknowns)``.

.. note::

   The residual is linear in the displacements: its Jacobian is computed once. The one-sided stencils of the first and
   last frames have larger coefficients than the centered ones, so these frames are more regularized; the coefficients
   across a larger step are smaller (a displacement change over a longer duration is a smaller derivative).

.. toctree::
   :maxdepth: 1
   :hidden:

   ./sdic/SDICTemporalRegularizationTerm.rst


Fixed parametrization
---------------------

The terms always work on the complete vector :math:`\vec{p}`. A parametrization lets the solver optimize a reduced
vector :math:`\vec{q}`, a selection of :math:`\vec{p}` (some vertices and some components of the displacements; all
the parameters of the optimized transformations), the other entries keeping their values in the analysis:

.. math::

   \vec{p} = P(\vec{q}) = S \, \vec{q} + \vec{p}_{\text{fixed}}, \qquad \mathbf{J}_q = \mathbf{J}_p \, S

:class:`pysdic.SDICFixedParametrization` is built on the analysis and the :class:`pysdic.SDICUnknowns` of the problem, and gives the
``pysolvegn.Parametrization`` with its property ``parametrization``. The fixed values :math:`\vec{p}_{\text{fixed}}`
are **read from the analysis at each call**: after a convergence written in the analysis, the same parametrization
can be used again without building it again.

.. warning::

   **With a parametrization, always use** ``reparametrized_pack``, ``reparametrized_unpack`` **and**
   ``reparametrized_apply`` on the vectors of the solver: ``p0`` and ``result.parameters`` are reduced vectors
   :math:`\vec{q}`. ``analysis.pack_unknowns``, ``unpack_unknowns`` and ``apply_unknowns`` work on the complete vector
   :math:`\vec{p}`: giving them :math:`\vec{q}` raises an error (wrong size) or, worse, misplaces the values if the
   sizes happen to match.

   ==========================================  =================================================
   Without parametrization                     With a parametrization
   ==========================================  =================================================
   ``analysis.pack_unknowns(unknowns)``        ``parametrization.reparametrized_pack()``
   ``analysis.unpack_unknowns(unknowns, p)``   ``parametrization.reparametrized_unpack(q)``
   ``analysis.apply_unknowns(unknowns, p)``    ``parametrization.reparametrized_apply(q)``
   ==========================================  =================================================

.. code-block:: python

   import pysolvegn
   from pysdic import SDICFixedParametrization, check_same_problem, compute_supported_vertices

   unknowns = SDICUnknowns(frames, [])
   correlation = SDICCorrelationTerm(analysis, unknowns, integration_points, pairings)
   regularization = SDICTemporalRegularizationTerm(analysis, unknowns, order=2, weight=1e-2)

   vertices = compute_supported_vertices(analysis.mesh, integration_points)  # Vertices carrying points
   parametrization = SDICFixedParametrization(analysis, unknowns, vertices=vertices, components=[0, 1])
   check_same_problem([correlation, regularization, parametrization])  # Same analysis, same unknowns

   result = pysolvegn.solve([correlation.term, regularization.term], parametrization.reparametrized_pack(),
                            parametrization=parametrization.parametrization, max_iteration=20)
   parametrization.reparametrized_apply(result.parameters)  # NOT analysis.apply_unknowns(...)

- :doc:`SDICFixedParametrization <sdic/SDICFixedParametrization>`: The parametrization optimizing a selection of the unknowns (vertices and components), the other ones read from the analysis at each call.
- :doc:`compute_supported_vertices <sdic/compute_supported_vertices>`: The vertices of the elements containing at least one integration point (the other vertices have a zero column in the Jacobian).

.. note::

   Do not confuse ``pysolvegn.Parametrization`` (the map :math:`\vec{q} \mapsto \vec{p}` of the solver, given by
   :attr:`pysdic.SDICFixedParametrization.parametrization`) with :class:`pysdic.Parametrization` (the parameters of a
   transformation of a camera).

.. toctree::
   :maxdepth: 1
   :hidden:

   ./sdic/SDICFixedParametrization.rst
   ./sdic/compute_supported_vertices.rst