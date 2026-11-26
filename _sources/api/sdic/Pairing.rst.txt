pysdic.Pairing
==============

.. currentmodule:: pysdic

.. contents:: Table of Contents
   :local:
   :depth: 2

The :class:`pysdic.Pairing` is a link between two views of a :class:`pysdic.SDICAnalysis`: the view ``view_index_1``
of the frame ``frame_index_1`` and the view ``view_index_2`` of the frame ``frame_index_2``. The gray levels of the
material points are compared between the two views with the residual:

.. math::

   r(\vec{X}) = [I_2 \circ P_2](\vec{X} + \vec{u}_2(\vec{X})) - [I_1 \circ P_1](\vec{X} + \vec{u}_1(\vec{X}))

- **Stereo pairing**: two views of the same frame (``frame_index_1 == frame_index_2``), to compare two cameras at the same instant.
- **Temporal pairing**: the same view index at two frames, to follow one camera along the time.

.. note::

   A pairing is a frozen object that does not know the analysis: use :meth:`pysdic.SDICAnalysis.validate_pairing` to
   check that its views exist in an analysis. The pairings are given to a :class:`pysdic.SDICCorrelationTerm` with the
   unknowns and the integration points (``SDICCorrelationTerm(unknowns, integration_points, pairings)``); a pairing
   without unknown (both views fixed) is ignored by the term.


Pairing class
-------------

.. autoclass:: Pairing
   :no-members:
   :show-inheritance:


Instantiation
-------------

The :class:`Pairing` class is instantiated with its constructor ``Pairing(frame_index_1, view_index_1, frame_index_2, view_index_2)``
or with the following class methods. The indices must be non-negative integers and the two views must be different.

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   Pairing.stereo
   Pairing.temporal

.. code-block:: python

   from pysdic import Pairing

   Pairing.stereo(0, 0, 1)    # Pairing((frame 0, view 0) -> (frame 0, view 1))
   Pairing.temporal(0, 0, 3)  # Pairing((frame 0, view 0) -> (frame 3, view 0))


Implemented methods and attributes
----------------------------------

The following sections provide an overview of the attributes and methods defined
in the :class:`pysdic.Pairing` class.


Attributes
^^^^^^^^^^

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   Pairing.frame_index_1
   Pairing.view_index_1
   Pairing.frame_index_2
   Pairing.view_index_2
   Pairing.view_1
   Pairing.view_2
   Pairing.is_stereo


Operations
^^^^^^^^^^

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   Pairing.reversed
