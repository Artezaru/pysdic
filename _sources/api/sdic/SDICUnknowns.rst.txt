pysdic.SDICUnknowns
===================

.. currentmodule:: pysdic

.. contents:: Table of Contents
   :local:
   :depth: 2


SDICUnknowns class
------------------

.. autoclass:: SDICUnknowns
   :no-members:

Instantiation
-------------

A :class:`SDICUnknowns` is **created by the user** with the optimized frames and the optimized transformations
``(kind, index)``. It **does not know the analysis**: as a :class:`pysdic.Pairing`, it is a light frozen description,
validated by an analysis with :meth:`pysdic.SDICAnalysis.validate_unknowns` (the objects of a problem call it at their
creation). The same unknowns can be used with several analyses.

.. code-block:: python

   from pysdic import SDICUnknowns

   unknowns = SDICUnknowns([1], [])                  # Measurement of the frame 1
   unknowns = SDICUnknowns([], [("extrinsic", 1)])   # Calibration of the extrinsic 1
   unknowns = SDICUnknowns([1], [("extrinsic", 1)])  # Joint optimization

   analysis.validate_unknowns(unknowns)  # The frames and transformations exist, have parameters, are not locked

The objects of a ``pysolvegn`` problem (:class:`pysdic.SDICCorrelationTerm`,
:class:`pysdic.SDICTemporalRegularizationTerm`, :class:`pysdic.SDICFixedParametrization`) are created with the **same
analysis object and the same unknowns object**: :func:`pysdic.check_same_problem` checks it by identity. To change the
unknowns (e.g. the next frame of a measurement), create new unknowns and give them to each object of the problem with
its ``with_unknowns`` method.

Attributes
----------

.. autosummary::
   :toctree: ../_generated/
   :signatures: long

   SDICUnknowns.optimized_frames
   SDICUnknowns.optimized_transformations
   SDICUnknowns.n_frames
   SDICUnknowns.n_transformations
   SDICUnknowns.is_empty

Vector of unknowns
------------------

The vector :math:`\vec{p} = [\vec{p}_U, \vec{p}_T]`: the **total** displacement :math:`(N_v, E)` of each optimized
frame (from the reference mesh, not an increment) in Fortran order (the component :math:`e` of the vertex :math:`v`
at the optimized frame :math:`k` has the index :math:`k N_v E + e N_v + v`), then the parameters of the optimized
transformations.

The unknowns do not know the sizes nor the values: the vector is built by the analysis.

.. code-block:: python

   p0 = analysis.pack_unknowns(unknowns)                                 # Current state -> p
   displacements, parameters = analysis.unpack_unknowns(unknowns, p)     # p -> copies
   analysis.apply_unknowns(unknowns, result.parameters)                  # p -> analysis

- :meth:`pysdic.SDICAnalysis.validate_unknowns`
- :meth:`pysdic.SDICAnalysis.n_unknowns`
- :meth:`pysdic.SDICAnalysis.pack_unknowns`, :meth:`pysdic.SDICAnalysis.unpack_unknowns`, :meth:`pysdic.SDICAnalysis.apply_unknowns`
- :meth:`pysdic.SDICAnalysis.get_unknowns_frame_slice`, :meth:`pysdic.SDICAnalysis.get_unknowns_transformation_slice`, :meth:`pysdic.SDICAnalysis.compute_unknowns_indices`

.. warning::

   **With a parametrization, do not use** ``pack_unknowns``, ``unpack_unknowns`` **or** ``apply_unknowns`` **on the
   vectors of the solver**: they are reduced vectors :math:`\vec{q}`. Use ``parametrization.reparametrized_pack()``,
   ``reparametrized_unpack(q)`` and ``reparametrized_apply(q)`` (see :class:`pysdic.SDICFixedParametrization`).