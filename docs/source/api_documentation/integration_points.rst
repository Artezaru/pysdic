.. currentmodule:: pysdic

pysdic.IntegrationPoints
===========================================

.. autoclass:: IntegrationPoints


I/O IntegrationPoints objects
-------------------------------------------

The :class:`IntegrationPoints` class can be instantiated using the following constructor:

.. autosummary::
   :toctree: ../generated/

    IntegrationPoints.from_npz

The :class:`IntegrationPoints` can be exported to a .npz file using the following method:

.. autosummary::
   :toctree: ../generated/

    IntegrationPoints.to_npz


Accessing IntegrationPoints attributes
-------------------------------------------

.. autosummary::
   :toctree: ../generated/

    IntegrationPoints.element_indices
    IntegrationPoints.internal_bypass
    IntegrationPoints.n_points
    IntegrationPoints.n_topological_dimensions
    IntegrationPoints.n_valids
    IntegrationPoints.natural_coordinates
    IntegrationPoints.shape
    IntegrationPoints.weights

Manipulating IntegrationPoints objects
-------------------------------------------

.. autosummary::
   :toctree: ../generated/

    IntegrationPoints.add_points
    IntegrationPoints.concatenate
    IntegrationPoints.copy
    IntegrationPoints.disable_points
    IntegrationPoints.remove_invalids
    IntegrationPoints.remove_points
    IntegrationPoints.validate


Operating on IntegrationPoints objects
-------------------------------------------

The following methods can be used to operate on :class:`IntegrationPoints` objects:

- ``+`` operator: Concatenate two :class:`IntegrationPoints` objects.
- ``+=`` operator: In-place concatenation of two :class:`IntegrationPoints` objects.
- ``len()`` function: Get the number of points in a :class:`IntegrationPoints` object.

