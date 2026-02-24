.. currentmodule:: pysdic

IntegrationPoints structures
===========================================

.. autoclass:: IntegrationPoints


I/O IntegrationPoints objects
-------------------------------------------

The :class:`IntegrationPoints` class can be instantiated using the following constructor:

.. autosummary::
   :toctree: ../_autosummary/

    IntegrationPoints.from_npz

The :class:`IntegrationPoints` can be exported to a .npz file using the following method:

.. autosummary::
   :toctree: ../_autosummary/

    IntegrationPoints.to_npz

Accessing IntegrationPoints attributes
-------------------------------------------

.. autosummary::
   :toctree: ../_autosummary/

    IntegrationPoints.element_indices
    IntegrationPoints.n_points
    IntegrationPoints.n_topological_dimensions
    IntegrationPoints.n_valids
    IntegrationPoints.natural_coordinates
    IntegrationPoints.shape
    IntegrationPoints.weights

Add and manage IntegrationPoints points properties
----------------------------------------------------

The properties of the points in a :class:`IntegrationPoints` object can be managed using the following methods:

.. note::

   Point properties are stored as named NumPy arrays of shape :math:`(N_p, A)` where :math:`N_p` is the number of points and :math:`A` is the number of property components (e.g., 3 for RGB color).
   All the point properties must have the same number of points as the :class:`IntegrationPoints` object and are stored as :obj:`numpy.float64` arrays.

.. autosummary::
   :toctree: ../_autosummary/

   IntegrationPoints.set_property
   IntegrationPoints.get_property
   IntegrationPoints.has_property
   IntegrationPoints.delete_property
   IntegrationPoints.list_properties
   IntegrationPoints.copy_properties
   IntegrationPoints.clear_properties


Precomputed quantities of the integration points
--------------------------------------------------------------------

To perform geometric computations and property interpolations on the mesh, some quantities can be precomputed and stored in the integration points for later use.
The precomputed quantities of the integration points are stored in a dictionary, and they can be accessed and managed using the following methods:

.. warning::

   If any modification is made to the points of the integration points, the precomputed properties may become invalid and should be cleared using :meth:`IntegrationPoints.clear_precomputed` to avoid using outdated properties in subsequent computations.
   It is the responsibility of the user to ensure that the precomputed properties are consistent with the current state of the integration points, especially after any modifications to the points.

.. autosummary::
   :toctree: ../_autosummary/

   IntegrationPoints.get_precomputed
   IntegrationPoints.list_precomputed
   IntegrationPoints.clear_precomputed

The method with :obj:`precompute` argument allows to compute a quantity and store it in the precomputed properties of the integration points for later use (or into :class:`IntegrationPoints`).
Set to :obj:`True` to store the computed quantity in the precomputed properties of the integration points, and set to :obj:`False` to compute the quantity without storing it in the precomputed properties of the integration points.

The precomputed quantities available are (and are precomputed when using the corresponding methods in :class:`Mesh`):

- ``integration_points_neighborhood``
- ``shape_functions``
- ``shape_function_derivatives``


Manipulating IntegrationPoints objects
-------------------------------------------

.. autosummary::
   :toctree: ../_autosummary/

    IntegrationPoints.concatenate
    IntegrationPoints.copy
    IntegrationPoints.disable_points
    IntegrationPoints.remove_invalids
    IntegrationPoints.remove_points
    IntegrationPoints.validate
