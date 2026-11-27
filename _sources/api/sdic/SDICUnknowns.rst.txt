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
``(kind, index)`` (see the description of the class above).

.. code-block:: python

   from pysdic import SDICUnknowns

   unknowns = SDICUnknowns([1], [])                  # Measurement of the frame 1
   unknowns = SDICUnknowns([], [("extrinsic", 1)])   # Calibration of the extrinsic 1
   unknowns = SDICUnknowns([1], [("extrinsic", 1)])  # Joint optimization

   analysis.validate_unknowns(unknowns)  # The frames and transformations exist, have parameters, are not locked

To change the unknowns (e.g. the next frame of a measurement), create new unknowns and give them to each object of
the problem with its ``with_unknowns`` method.

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

The order of the vector :math:`\vec{p}` is described with the class above. The unknowns do not know the sizes nor
the values: the vector is built by the analysis.

.. code-block:: python

   p0 = analysis.pack_unknowns(unknowns)                                 # Current state -> p
   displacements, parameters = analysis.unpack_unknowns(unknowns, p)     # p -> copies
   analysis.apply_unknowns(unknowns, result.parameters)                  # p -> analysis

- :meth:`pysdic.SDICAnalysis.validate_unknowns`
- :meth:`pysdic.SDICAnalysis.n_unknowns`
- :meth:`pysdic.SDICAnalysis.pack_unknowns`, :meth:`pysdic.SDICAnalysis.unpack_unknowns`, :meth:`pysdic.SDICAnalysis.apply_unknowns`
- :meth:`pysdic.SDICAnalysis.get_unknowns_frame_slice`, :meth:`pysdic.SDICAnalysis.get_unknowns_transformation_slice`, :meth:`pysdic.SDICAnalysis.compute_unknowns_indices`

.. seealso::

   With a parametrization, the vectors of the solver are reduced vectors :math:`\vec{q}`: see
   :class:`pysdic.SDICFixedParametrization`.