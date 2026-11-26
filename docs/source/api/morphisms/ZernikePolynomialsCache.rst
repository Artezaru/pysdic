pysdic.ZernikePolynomialsCache
==============================

.. currentmodule:: pysdic

.. contents:: Table of Contents
   :local:
   :depth: 2

.. _zernike-polynomials-cache:

The Zernike polynomials :math:`Z_{n,m}(\rho, \theta)` of a :class:`pysdic.ZernikeDistortion` only depend on the points
and on the ellipse of definition :math:`(R_x, R_y, x_0, y_0)`, not on the coefficients. When a distortion is applied
several times to the **same points** with different coefficients, the polynomials can be computed once and reused
with a :class:`ZernikePolynomialsCache`, attached to the distortion with the property ``polynomials_cache``.


ZernikePolynomialsCache class
-----------------------------

.. autoclass:: ZernikePolynomialsCache
   :no-members:
   :show-inheritance:


Usage
-----

The typical use is the calibration of a distortion alone: the extrinsic does not change, so the normalized points of
a view are the same at each iteration, only the coefficients are optimized. In a measurement, the points move at each
iteration: the cache is useless.

.. code-block:: python

   import pysdic

   cache = pysdic.ZernikePolynomialsCache(max_entries=2)  # One entry per view (L and R share the distortion)
   distortion.polynomials_cache = cache                   # Or ZernikeDistortion(..., polynomials_cache=cache)

   ...                                                    # Optimization of the coefficients of the distortion

   print(cache)                                           # n_entries, nbytes, hits, misses
   distortion.polynomials_cache = None                    # No cache anymore
   cache.clear()                                          # Free the memory

The rules of the cache:

- **No cache by default**: the polynomials are computed at each call unless a cache is set explicitly. The cache never
  changes the results.
- **Shareable**: several distortions (e.g. of different orders, with the same ellipse) can use the same cache. It is
  not serialized and not copied (a copy of the distortion has no cache).
- **Identification of an entry**: the points (hash and exact comparison with a stored copy) and the ellipse (exact
  comparison). An entry of order :math:`N` also serves the orders :math:`N' \leq N`; the derivatives
  :math:`\partial Z / \partial x` and :math:`\partial Z / \partial y` are stored only when requested (Jacobian
  ``dx``).
- **LRU**: at most ``max_entries`` entries are kept, the least recently used one is removed first. ``max_entries``
  can be changed at any time.
- **Only for** ``apply``: the undistortion (``invert``) evaluates the polynomials at points that change at each
  Newton iteration and never uses the cache.

.. important::

   The memory of an entry is about :math:`8 N_p (2 + D P)` bytes for :math:`N_p` points and :math:`P` polynomials,
   with :math:`D = 1` (values) or :math:`D = 3` (with the derivatives), e.g. about 115 MB for 627 264 points at the
   order 5 without derivatives. Call :meth:`ZernikePolynomialsCache.clear` (or remove the cache from the
   distortions) when it is not needed anymore.


Implemented methods and attributes
----------------------------------

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   ZernikePolynomialsCache.max_entries
   ZernikePolynomialsCache.n_entries
   ZernikePolynomialsCache.nbytes
   ZernikePolynomialsCache.hits
   ZernikePolynomialsCache.misses
   ZernikePolynomialsCache.clear

.. seealso::

   :doc:`ZernikeDistortion <ZernikeDistortion>`, the distortion using the cache.