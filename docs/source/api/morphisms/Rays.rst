pysdic.Rays
========================================================================

.. currentmodule:: pysdic

.. contents:: Table of Contents
   :local:
   :depth: 2

The :class:`pysdic.Rays` provides a standardized way to store and manage a collection of rays in 3D space,
expressed in world coordinates. Each ray is stored with its origin and its unit direction in an array
with shape :math:`(\dots, 6)`.

.. seealso::

   - The ``compute_rays`` method of the extrinsic transformations (for example :class:`pysdic.PerspectiveExtrinsic`),
     which returns a :class:`pysdic.Rays` object.

Rays class
----------

.. autoclass:: Rays
   :members:

