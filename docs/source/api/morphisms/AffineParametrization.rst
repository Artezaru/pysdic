pysdic.AffineParametrization
=================================

.. currentmodule:: pysdic

.. contents:: Table of Contents
   :local:
   :depth: 2


AffineParametrization class
------------------------------------

.. autoclass:: AffineParametrization
   :no-members:
   :show-inheritance:

Instantiation
-------------

The :class:`AffineParametrization` class is instantiated directly with its constructor:

- **With a matrix of modes** :math:`\mathbf{M}` of shape :math:`(n_o, n_i)`: ``AffineParametrization(modes)`` (zero offset).
- **With a matrix of modes and an offset** :math:`\vec{p_0}` of shape :math:`(n_o,)`: ``AffineParametrization(modes, offset)``.

The ``decode`` method is only available if :math:`\mathbf{M}` is square with full rank.


Implemented methods and attributes
------------------------------------

The following sections provide an overview of the attributes and methods defined
in the :class:`pysdic.AffineParametrization` class.

Attributes
^^^^^^^^^^

The :class:`pysdic.AffineParametrization` defines the following attributes
(the constants ``modes`` and ``offset`` are read-only):

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   AffineParametrization.input_dim
   AffineParametrization.modes
   AffineParametrization.offset
   AffineParametrization.output_dim


Encoding and decoding
---------------------

Use the ``encode`` and ``decode`` methods from
:class:`pysdic.Parametrization` to perform the parametrization:

- ``encode``: maps ``p_i`` → ``p_o = M p_i + p_0``.
- ``decode``: maps ``p_o`` → ``p_i = M^{-1} (p_o - p_0)`` (only if :math:`\mathbf{M}` is invertible).

The Jacobian with respect to the parameters can be computed by enabling the ``dp`` flag:

- ``dp=True``: compute ∂p_o / ∂p_i (constant, equal to :math:`\mathbf{M}`)

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   AffineParametrization._encode
   AffineParametrization._decode
