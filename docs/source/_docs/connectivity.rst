.. currentmodule:: pysdic


Connectivity structures
==================================================================

.. contents:: Table of Contents
   :local:
   :depth: 1
   :backlinks: top

Connectivity class
-------------------------------------------

.. autoclass:: Connectivity

Instantiate and export Connectivity object
--------------------------------------------------

To Instantiate a :class:`Connectivity` object, use one of the following class methods:

.. autosummary::
   :toctree: ../_autosummary/

   Connectivity.from_array
   Connectivity.from_npz

The :class:`Connectivity` can then be exported to different formats using the following methods:

.. autosummary::
   :toctree: ../_autosummary/

   Connectivity.to_array
   Connectivity.to_npz


Accessing Connectivity attributes
-------------------------------------------

The public attributes of a :class:`Connectivity` object can be accessed using the following properties:

.. autosummary::
   :toctree: ../_autosummary/

   Connectivity.elements
   Connectivity.element_type
   Connectivity.n_elements
   Connectivity.n_topological_dimensions
   Connectivity.n_vertices_per_element
   Connectivity.shape


Add and manage Connectivity elements properties
------------------------------------------------

The properties of the elements in a :class:`Connectivity` object can be managed using the following methods:

.. note::

   Elements properties are stored as named NumPy arrays of shape :math:`(N_e, A)` where :math:`N_e` is the number of elements and :math:`A` is the number of property components (e.g., 3 for RGB color).
   All the element properties must have the same number of elements as the :class:`Connectivity` object and are stored as :obj:`numpy.float64` arrays.
   
.. autosummary::
   :toctree: ../_autosummary/

   Connectivity.set_property
   Connectivity.get_property
   Connectivity.has_property
   Connectivity.delete_property
   Connectivity.list_properties
   Connectivity.copy_properties
   Connectivity.clear_properties


Add, remove or modify points in Connectivity objects
-----------------------------------------------------

The elements of a :class:`Connectivity` object can be manipulated using the following methods:

.. autosummary::
   :toctree: ../_autosummary/

   Connectivity.concatenate
   Connectivity.copy
   Connectivity.filter_elements
   Connectivity.keep_elements
   Connectivity.remove_elements














