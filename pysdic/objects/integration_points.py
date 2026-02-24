# Copyright 2025 Artezaru
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Optional, Tuple, Dict, List, Any
from numpy.typing import ArrayLike

import numpy
import os


class IntegrationPoints(object):
    r"""
    Base class to store integration points for numerical integration over elements and localisation of points into :class:`pysdic.Mesh` objects.

    Integration points are defined in the reference element using natural coordinates (:math:`\xi, \eta, \zeta, ...`), element indices, and weights.

    The number of natural coordinates depends on the topological dimension of the element :math:`K` noted ``n_topological_dimensions``:
    
    .. note::
    
        The coordinates and the properties of the points are stored as copy-on-write (cow) NumPy arrays of type :obj:`numpy.float64`. 
        Accesing these arrays directly will return unwritable views to prevent accidental modifications.
        The element indices are stored as copy-on-write (cow) NumPy arrays of type :obj:`numpy.int64`.

    .. note::

        If the weights are not provided, equal weights are assumed for all integration points.
        By default the weights are set to 1 for all integration points (and not :math:`1/N_p`).

    If a specific specific point is not include in any element, it can be identified by setting its element ID to -1 and its natural coordinates to NaN.

    Parameters
    ----------
    natural_coordinates: ArrayLike
        The natural coordinates of the integration points as a numpy ndarray with shape (:math:`N_p`, :math:`K`),
        where :math:`N_p` is the number of integration points and :math:`K` is the topological dimension of the element.

    element_indices: ArrayLike
        The element indices of the integration points as a numpy ndarray with shape (:math:`N_p`,),
        where :math:`N_p` is the number of integration points.

    weights: Optional[ArrayLike], optional
        The weights of the integration points as a numpy ndarray with shape (:math:`N_p`,),
        where :math:`N_p` is the number of integration points, by default None means equal weights of 1 for all points.

    n_topological_dimensions: Optional[Integral], optional
        The topological dimension of the element, by default None means it will be inferred from the shape of the natural_coordinates.
        Otherwise, an error will be raised if the provided dimension does not match the shape of the natural_coordinates.
        
    properties: Optional[Dict[:class:`str`, ArrayLike]], optional
        A dictionary of additional properties to associate with the integrated points. Keys are property names, and values are NumPy arrays of property data of shape :math:`(N_{p}, A)` where :math:`A` is the number of attributes per point for the given property.
        Default is :obj:`None`, meaning no additional properties are set.

    """
    __slots__ = [
        '_natural_coordinates', 
        '_element_indices', 
        '_weights', 
        '_properties',
        '_precomputed'
    ]

    def __init__(
        self, 
        natural_coordinates: ArrayLike, 
        element_indices: numpy.ndarray, 
        weights: Optional[numpy.ndarray] = None, 
        n_topological_dimensions: Optional[int] = None,
        properties: Optional[dict[str, ArrayLike]] = None,
    ):
        # Type checks and conversions
        natural_coordinates = numpy.asarray(natural_coordinates, dtype=numpy.float64)
        element_indices = numpy.asarray(element_indices, dtype=numpy.int64)
        if weights is not None:
            weights = numpy.asarray(weights, dtype=numpy.float64)

        if natural_coordinates.ndim != 2:
            raise ValueError("natural_coordinates should be a 2D array of shape (Np, K).")
        if element_indices.ndim != 1:
            raise ValueError("element_indices should be a 1D array of shape (Np,).")
        if natural_coordinates.shape[0] != element_indices.shape[0]:
            raise ValueError("The number of integration points Np in natural_coordinates and element_indices should match.")
        if weights is not None and weights.ndim != 1:
            raise ValueError("weights should be a 1D array of shape (Np,).")
        if weights is not None and weights.shape[0] != element_indices.shape[0]:
            raise ValueError("The number of integration points Np in weights and element_indices should match.")
        if n_topological_dimensions is not None and (not isinstance(n_topological_dimensions, int) or n_topological_dimensions <= 0):
            raise ValueError("n_topological_dimensions should be a positive integer.")
        if n_topological_dimensions is not None and natural_coordinates.shape[1] != n_topological_dimensions:
            raise ValueError("The provided n_topological_dimensions does not match the shape of the natural_coordinates.")
        
        if properties is None:
            properties = {}
        if not isinstance(properties, dict):
            raise ValueError("properties must be a dictionary of string keys and NumPy array values.")
        for key, values in properties.items():
            if not isinstance(key, str):
                raise TypeError("Property key must be a string.")
            values = numpy.asarray(values, dtype=numpy.float64)
            if values.ndim == 1:
                values = values.reshape((-1, 1))
            if not (values.ndim == 2 and values.shape[0] == natural_coordinates.shape[0]):
                raise ValueError(f"Property values must be a 2D NumPy array with shape (N_p, A) where N_p={natural_coordinates.shape[0]}.")
           
        # Set the attributes
        self._natural_coordinates = natural_coordinates
        self._element_indices = element_indices
        self._n_topological_dimensions = natural_coordinates.shape[1] if n_topological_dimensions is None else n_topological_dimensions
        self._weights = weights
        self._properties = {}
        self._precomputed = {}
        
        # Validate and save properties
        for key, values in properties.items():
            self.set_property(key, values)
        self.validate()

    # =======================
    # Internals
    # =======================
    def _internal_check_consistency(self) -> None:
        r"""
        Internal method to check the consistency of the attributes.

        All checks are skipped if internal_bypass is :obj:`True`.

        Otherwise, the shape and type of the attributes are always checked and the content of arrays is checked based on the input flags.
                    
        """
        if not isinstance(self._n_topological_dimensions, int):
            raise TypeError("dimension should be an integer.")
        if self._n_topological_dimensions <= 0:
            raise ValueError("dimension should be a positive integer.")

        if not isinstance(self._natural_coordinates, numpy.ndarray):
            raise TypeError("natural_coordinates should be a numpy ndarray.")
        if self._natural_coordinates.ndim != 2:
            raise ValueError("natural_coordinates should be a 2D array of shape (Np, K).")
        if not self._natural_coordinates.shape == (self.n_points, self.n_topological_dimensions):
            raise ValueError("natural_coordinates shape does not match (Np, K).")
        if not self._natural_coordinates.dtype == numpy.float64:
            raise TypeError("natural_coordinates should be of type numpy.float64.")
        
        if not isinstance(self._element_indices, numpy.ndarray):
            raise TypeError("element_indices should be a numpy ndarray.")
        if self._element_indices.ndim != 1:
            raise ValueError("element_indices should be a 1D array of shape (Np,).")
        if not self._element_indices.shape[0] == self.n_points:
            raise ValueError("element_indices shape does not match (Np,).")
        if not self._element_indices.dtype == int:
            raise TypeError("element_indices should be of type int.")
        
        if self._weights is not None and not isinstance(self._weights, numpy.ndarray):
            raise TypeError("weights should be a numpy ndarray.")
        if self._weights is not None and self._weights.ndim != 1:
            raise ValueError("weights should be a 1D array of shape (Np,).")
        if self._weights is not None and not self._weights.shape[0] == self.n_points:
            raise ValueError("weights shape does not match (Np,).")
        if self._weights is not None and not self._weights.dtype == numpy.float64:
            raise TypeError("weights should be of type numpy.float64.")
        
    
    def _set_precomputed(self, key: str, value: Any) -> None:
        r"""
        Internal method to set a precomputed value in the mesh.

        Parameters
        ----------
        key : :class:`str`
            The key to associate with the precomputed value.

        value : Any
            The value to store as precomputed.
        """
        if not isinstance(key, str):
            raise TypeError(f"Key must be a string, got {type(key)}.")
        
        self._precomputed[key] = value
        
    def _get_precomputed(self, key: str) -> Optional[Any]:
        r"""
        Internal method to get a precomputed value from the mesh.

        Parameters
        ----------
        key : :class:`str`
            The key associated with the precomputed value to retrieve.

        Returns
        -------
        Optional[Any]
            The precomputed value associated with the key, or None if the key does not exist.
        """
        if not isinstance(key, str):
            raise TypeError(f"Key must be a string, got {type(key)}.")
        
        return self._precomputed.get(key, None)
    
    def _has_precomputed(self, key: str) -> bool:
        r"""
        Internal method to check if a precomputed value exists in the mesh.

        Parameters
        ----------
        key : :class:`str`
            The key associated with the precomputed value to check.

        Returns
        -------
        bool
            True if a precomputed value associated with the key exists, False otherwise.
        """
        if not isinstance(key, str):
            raise TypeError(f"Key must be a string, got {type(key)}.")
        
        return key in self._precomputed
    
    def _clear_precomputed(self) -> None:
        r"""
        Internal method to clear all precomputed values from the mesh.
        """
        self._precomputed.clear()

    # =======================
    # Properties
    # =======================
    @property
    def natural_coordinates(self) -> numpy.ndarray:
        r"""
        [Get or Set] The natural coordinates of the integration points as a numpy ndarray with shape (:math:`N_p`, :math:`K`),
        where :math:`N_p` is the number of integration points and :math:`K` is the topological dimension of the element.

        If a point is not included in any element, its natural coordinates should be set to NaN.
        
        .. note::
        
            The property is copied-on-write (cow) into :obj:`numpy.ndarray` of type :obj:`numpy.float64`. Accessing it will return an unwritable view to prevent accidental modifications.
            
        Parameters
        ----------
        value: ArrayLike
            An array-like of shape :math:`(N_p, K)` representing the natural coordinates of the integration points. If 1D array is provided, it will be reshaped to :math:`(N_p, 1)`.


        Returns
        -------
        :class:`numpy.ndarray`
            A unwritable NumPy array of shape :math:`(N_p, K)` containing the natural coordinates of the integration points.


        Raises
        ------
        TypeError
            If the input points is not a NumPy array.
        
        ValueError
            If the input array does not have the correct shape :math:`(N_p, K)`.
        """
        view = self._natural_coordinates.view()
        view.flags.writeable = False
        return view
    
    @natural_coordinates.setter
    def natural_coordinates(self, value: ArrayLike) -> None:
        points = numpy.asarray(value, dtype=numpy.float64).copy()
        if self.n_topological_dimensions == 1 and points.ndim == 1:
            points = points.reshape((-1, 1))
        if not (points.ndim == 2 and points.shape[1] == self.n_topological_dimensions):
            raise ValueError(f"Natural coordinates must be a 2D NumPy array with shape (N_p, {self.n_topological_dimensions}).")
        if not points.shape[0] == self.n_points:
            raise ValueError(f"Natural coordinates must have the same number of points N_p={self.n_points}. Consider using remove/add point methods to change the number of points.")
        self._natural_coordinates = points
        

    @property
    def element_indices(self) -> numpy.ndarray:
        r"""
        [Get or Set] The element indices of the integration points as a numpy ndarray with shape (:math:`N_p`,),
        where :math:`N_p` is the number of integration points.

        if a point is not included in any element, its element ID should be set to :obj:`-1`.

        .. note::
        
            The property is copied-on-write (cow) into :obj:`numpy.ndarray` of type :obj:`numpy.int64`. Accessing it will return an unwritable view to prevent accidental modifications.

        Parameters
        ----------
        value: ArrayLike
            An array-like of shape :math:`(N_p,)` representing the element indices of the integration points.

        Returns
        -------
        :class:`numpy.ndarray`
            The element IDs of the integration points as a NumPy array of shape :math:`(N_p,)` and type :obj:`numpy.int64`.
        """
        view = self._element_indices.view()
        view.flags.writeable = False
        return view

    @element_indices.setter
    def element_indices(self, value: ArrayLike) -> None:
        indices = numpy.asarray(value, dtype=int).copy()
        if indices.ndim != 1:
            raise ValueError("element_indices should be a 1D array of shape (N_p,).")
        if not indices.shape[0] == self.n_points:
            raise ValueError(f"element_indices must have the same number of points N_p={self.n_points}. Consider using remove/add point methods to change the number of points.")
        self._element_indices = indices


    @property
    def weights(self) -> numpy.ndarray:
        r"""
        [Get or Set] The weights of the integration points as a numpy ndarray with shape (:math:`N_p`,),
        where :math:`N_p` is the number of integration points.

        If weights are not provided, equal weights of :obj:`1` are assumed for all points.

        .. note::
        
            The property is copied-on-write (cow) into :obj:`numpy.ndarray` of type :obj:`numpy.float64`. Accessing it will return an unwritable view to prevent accidental modifications.
            
        Parameters
        ----------
        value: Optional[ArrayLike]
            The new weights of the integration points as a numpy ndarray with shape (:math:`N_p`,), or None to set equal weights of 1 for all points.

        Returns
        -------
        :class:`numpy.ndarray`
            The weights of the integration points. If not provided, returns an array of ones with shape (:math:`N_p`,).

        """
        if self._weights is None:
            return numpy.ones(self.n_points, dtype=numpy.float64)
        return self._weights
    
    @weights.setter
    def weights(self, value: Optional[ArrayLike]) -> None:
        if value is None:
            self._weights = None
        else:
            value = numpy.asarray(value, dtype=numpy.float64).copy()
            if value.ndim != 1:
                raise ValueError("weights should be a 1D array of shape (N_p,).")
            if not value.shape[0] == self.n_points:
                raise ValueError(f"weights must have the same number of points N_p={self.n_points}. Consider using remove/add point methods to change the number of points.")
            self._weights = value
        self._internal_check_consistency()


    @property
    def n_topological_dimensions(self) -> int:
        r"""
        [Get] The topological dimension of the element.

        .. seealso::
        
            - :meth:`shape` for getting the shape of the natural coordinates array.
            - :meth:`n_points` for getting the number of integration points.
            - :meth:`n_valids` for getting the number of valid integration points (points included in an element).

        Returns
        -------
        :class:`int`
            The topological dimension of the element.
        """
        return self._natural_coordinates.shape[1]

    
    @property
    def n_points(self) -> int:
        r"""
        [Get] The number of integration points.
        
        .. note::

            You can also use `len(integration_points)`.

        .. seealso::

            - :meth:`shape` for getting the shape of the natural coordinates array.
            - :meth:`n_topological_dimensions` for getting the dimension of the natural coordinates array.
            - :meth:`n_valids` for getting the number of valid integration points (points included in an element).
            

        Returns
        -------
        :class:`int`
            The number of integration points :math:`N_p`.
        """
        return self._natural_coordinates.shape[0]
    
    
    def __len__(self) -> int:
        return self.n_points
    
    
    @property
    def n_valids(self) -> int:
        r"""
        [Get] The number of valid integration points (points included in an element).

        A point is considered valid if its element index is not :obj:`-1`.
        
        .. seealso::
        
            - :meth:`shape` for getting the shape of the natural coordinates array.
            - :meth:`n_topological_dimensions` for getting the dimension of the natural coordinates array.
            - :meth:`n_points` for getting the total number of integration points.

        Returns
        -------
        :class:`int`
            The number of valid integration points.
        """
        return numpy.sum(self._element_indices != -1)
    
    
    def shape(self) -> Tuple[int, int]:
        r"""
        [Get] The shape of the integration points data.
        
        .. seealso::
        
            - :meth:`n_topological_dimensions` for getting the dimension of the natural coordinates array.
            - :meth:`n_points` for getting the total number of integration points.
            - :meth:`n_valids` for getting the number of valid integration points (points included in an element).

        Returns
        -------
        Tuple[:class:`int`, :class:`int`]
            A tuple (:math:`N_p`, :math:`K`) where :math:`N_p` is the number of integration points and :math:`K` is the topological dimension of the element.
        """
        return (self.n_points, self.n_topological_dimensions)
    
    def __repr__(self) -> str:
        string = f"IntegrationPoints(n_points={self.n_points}, n_topological_dimensions={self.n_topological_dimensions})"
        for key in self._properties.keys():
            string += f"\n  - Property: '{key}' with shape {self._properties[key].shape}"
        return string
    
    
    # ==========================
    # Manage properties
    # ==========================
    def set_property(
        self,
        key: str,
        values: ArrayLike
    ) -> None:
        r"""
        Set a property for the integration points as a NumPy array of shape :math:`(N_p, A)` where :math:`A` is the number of attributes per point for the given property.

        .. note::
        
            The given array will be copied and stored as type :obj:`numpy.float64`.
            
        .. note::
        
            The property can also be set using dictionary-like syntax:
            
            .. code-block:: python
                :linenos:
                
                integration_points['property_key'] = property_values
                    

        Parameters
        ----------
        key: :class:`str`
            The key/name of the property to set.
        
        values: ArrayLike
            A NumPy array of shape :math:`(N_p, A)` or :math:`(N_p,)` representing the property values for each integration point.
            If 1D array is provided, it will be reshaped to :math:`(N_p, 1)`.


        Raises
        ------
        TypeError
            If the property key is not a string.
            If the input values is not a NumPy array.
            
        ValueError
            If the input array does not have the correct shape :math:`(N_p, A)` or :math:`(N_p,)`.
            
        """
        if not isinstance(key, str):
            raise TypeError("Property key must be a string.")
        values = numpy.asarray(values, dtype=numpy.float64).copy()
        if values.ndim == 1:
            values = values.reshape((-1, 1))
        if not (values.ndim == 2 and values.shape[0] == self.n_points):
            raise ValueError(f"Property values must be a 2D NumPy array with shape (N_p, A) where N_p={self.n_points}.")
        self._properties[key] = values
        
    def __setitem__(self, key: str, values: ArrayLike) -> None:
        self.set_property(key, values)
        
    
    def get_property(
        self,
        key: str
    ) -> numpy.ndarray:
        r"""
        Get a property of the integration points by its key/name.
        
        .. note::
        
            The returned array is a unwritable view to prevent accidental modifications and is of type :obj:`numpy.float64`.
            
        .. note::
        
            The property can also be accessed using dictionary-like syntax:
            
            .. code-block:: python
                :linenos:
                
                property_values = integration_points['property_key']
                

        Parameters
        ----------
        key: :class:`str`
            The key/name of the property to retrieve.


        Returns
        -------
        :class:`numpy.ndarray`
            A NumPy array of shape :math:`(N_p, A)` representing the property values for each integration point, where :math:`A` is the number of attributes per point for the given property.


        Raises
        ------
        KeyError
            If the property with the given key does not exist.

        """
        if key not in self._properties:
            raise KeyError(f"Property '{key}' does not exist in the integration points.")
        view = self._properties[key].view()
        view.flags.writeable = False
        return view
    
    def __getitem__(self, key: str) -> numpy.ndarray:
        return self.get_property(key)
    
    
    def delete_property(
        self,
        key: str
    ) -> None:
        r"""
        Delete a property of the integration points by its key/name.


        Parameters
        ----------
        key: :class:`str`
            The key/name of the property to delete.


        Raises
        ------
        KeyError
            If the property with the given key does not exist.
        """
        if key not in self._properties:
            raise KeyError(f"Property '{key}' does not exist in the integration points.")
        del self._properties[key]
        
    
    def has_property(
        self,
        key: str
    ) -> bool:
        r"""
        Check if the integration points have a property with the given key/name.
        
        .. note::
        
            The property can also be checked using dictionary-like syntax:
            
            .. code-block:: python
                :linenos:
                
                has_property = 'property_key' in integration_points


        Parameters
        ----------
        key: :class:`str`
            The key/name of the property to check.


        Returns
        -------
        :class:`bool`
            :obj:`True` if the property exists, :obj:`False` otherwise.

        """
        return key in self._properties
    
    def __contains__(self, key: str) -> bool:
        return self.has_property(key)
    
    
    def list_properties(self) -> List[str]:
        r"""
        List all property keys/names associated with the integration points.


        Returns
        -------
        List[:class:`str`]
            A list of strings representing the keys/names of all properties in the integration points.

        """
        return list(self._properties.keys())
    
    
    def copy_properties(self) -> Dict[str, numpy.ndarray]:
        r"""
        Return a copy of all properties of the integration points.

        Returns
        -------
        dict[:class:`str`, :class:`numpy.ndarray`]
            A dictionary containing copies of all properties in the integration points.

        """
        properties_copy = {}
        for key, values in self._properties.items():
            properties_copy[key] = values.copy()
        return properties_copy
    
    
    def clear_properties(self) -> None:
        r"""
        Remove all properties from the integration points.
        """
        self._properties.clear()
    
    
    # =======================
    # Precompute properties
    # =======================
    def get_precomputed(self, key: str) -> Any:
        r"""
        Get the value of a precomputed property stored in the mesh by its key.

        Parameters
        ----------
        key: :class:`str`
            The key of the precomputed property to retrieve.

        Returns
        -------
        Any
            The value of the precomputed property associated with the specified key, or :obj:`None` if the key does not exist in the precomputed properties of the mesh.
        
        
        Examples
        --------
        Create a simple :class:`Mesh` instance.

        .. code-block:: python
            :linenos:

            import numpy as np
            from pysdic import Mesh, PointCloud

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3], [1, 2, 4]])
            mesh3d = Mesh(points, connectivity)

        Precompute the vertices path distance matrix from the connectivity and store it in the precomputed properties of the mesh with the key 'vertices_adjacency_matrix'.

        .. code-block:: python
            :linenos:

            distance_matrix = mesh3d.precompute_vertices_adjacency_matrix()

        Get the value of the precomputed vertices adjacency matrix using its key.

        .. code-block:: python
            :linenos:

            retrieved_distance_matrix = mesh3d.get_precomputed('vertices_adjacency_matrix')
            print(retrieved_distance_matrix)

        """
        return self._get_precomputed(key)
    

    def list_precomputed(self) -> List[str]:
        r"""
        List all the keys of the precomputed properties that are currently stored in the mesh.

        Returns
        -------
        List[:class:`str`]
            A list of strings representing the keys of the precomputed properties stored in the mesh.
    
    
        Examples
        --------
        Create a simple :class:`Mesh` instance.

        .. code-block:: python
            :linenos:

            import numpy as np
            from pysdic import Mesh, PointCloud

            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
            connectivity = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3], [1, 2, 4]])
            mesh3d = Mesh(points, connectivity)

        List all the keys of the precomputed properties stored in the mesh.

        .. code-block:: python
            :linenos:

            precomputed_keys = mesh3d.list_precomputed()
            print(precomputed_keys)

        """
        return list(self._precomputed.keys())
    
    
    def clear_precomputed(self) -> None:
        r"""
        Clear all the precomputed properties stored in the mesh, removing all key-value pairs from the precomputed properties dictionary.
        """
        self._clear_precomputed()
        
    
    # ==========================
    # I/O Methods
    # ==========================
    @classmethod
    def from_npz(cls, filepath: str, load_properties: bool = True) -> IntegrationPoints:
        r"""
        Create a :class:`IntegrationPoints` object from a NumPy NPZ file.

        The NPZ file should contain an array named 'natural_coordinates' with shape :math:`(N_p, K)` representing the natural coordinates of the integration points, and an array named 'element_indices' with shape :math:`(N_p,)` representing the element indices of the integration points. Optionally, it can also contain an array named 'weights' with shape :math:`(N_p,)` representing the weights of the integration points.
        Additional properties can be stored as separate arrays in the NPZ file.

        .. seealso::

            - :meth:`to_npz` method for saving the integration points to a NPZ file.

        Parameters
        ----------
        filepath: :class:`str`
            The path to the NPZ file.
            
        load_properties: :class:`bool`, optional
            If :obj:`True`, all arrays in the NPZ file with format ``"_properties/{property_name}"`` will be loaded as properties of the integration points. Default is :obj:`True`.


        Returns
        -------
        :class:`IntegrationPoints`
            A :class:`IntegrationPoints` object containing the natural coordinates, element indices, weights, and properties read from the NPZ file.
        
        """
        path = os.path.expanduser(filepath)
        if not os.path.isfile(path) or not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        data = numpy.load(filepath)
        if 'natural_coordinates' not in data or 'element_indices' not in data:
            raise ValueError("NPZ file must contain arrays named 'natural_coordinates' with shape (N_p, K) and 'element_indices' with shape (N_p,).")
        natural_coordinates = data['natural_coordinates']
        element_indices = data['element_indices']
        weights = data['weights'] if 'weights' in data else None    
        properties = {}
        if load_properties:
            for key in data.files:
                # key as "_properties/{property_name}" is reserved for properties
                if key.startswith("_properties/"):
                    property_name = key[len("_properties/"):]
                    properties[property_name] = data[key].copy()
        return cls(natural_coordinates=natural_coordinates, element_indices=element_indices, weights=weights, properties=properties)
    
    
    def to_npz(self, filepath: str, save_properties: bool = True) -> None:
        r"""
        Save the integration points to a NumPy NPZ file.

        The natural coordinates will be saved as an array named 'natural_coordinates' with shape :math:`(N_p, K)`.
        The element indices will be saved as an array named 'element_indices' with shape :math:`(N_p,)`.
        The weights will be saved as an array named 'weights' with shape :math:`(N_p,)` if they are present.
        Additional properties will be saved as separate arrays in the NPZ file with names in the format ``"_properties/{property_name}"``.

        .. seealso::

            - :meth:`from_npz` method for creating a :class:`IntegrationPoints` object from a NPZ file.

        Parameters
        ----------
        filepath: :class:`str`
            The path to the output NPZ file.

        save_properties: :class:`bool`, optional
            If :obj:`True`, all properties of the integration points will be saved as separate arrays in the NPZ file. Default is :obj:`True`.


        """
        path = os.path.expanduser(filepath)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        data_to_save = {
            'natural_coordinates': self._natural_coordinates.copy(),
            'element_indices': self._element_indices.copy()
        }
        if self._weights is not None:
            data_to_save['weights'] = self._weights.copy()
        if save_properties:
            properties_data = {}
            for key, values in self._properties.items():
                properties_data[f"_properties/{key}"] = values.copy()
            data_to_save.update(properties_data)
            
        numpy.savez(path, **data_to_save)
    

    # =======================
    # Methods
    # =======================
    def validate(self) -> None:
        r"""
        Validate the consistency of the :class:`IntegrationPoints` instance.

        This method checks the internal consistency of the attributes and raises an error if any inconsistency is found.

        Raises
        ------
        ValueError
            If any inconsistency is found in the attributes.
        """
        self._internal_check_consistency()


    def concatenate(self, other: IntegrationPoints, inplace: bool = False) -> IntegrationPoints:
        r"""
        Concatenate two :class:`IntegrationPoints` instances.
        
        .. note::
        
            The properties in common between the two integration points are also concatenated. 
            If a property exists in only one of the integration points, it will be filled with :obj:`numpy.nan` values for the points from the other integration point.
            

        Parameters
        ----------
        other: :class:`IntegrationPoints`
            Another :class:`IntegrationPoints` instance to concatenate with.

        inplace: :class:`bool`, optional
            If :obj:`True`, modify the current instance in place, and return itself. If :obj:`False`, return a new :class:`IntegrationPoints` instance (default is :obj:`False`).


        Returns
        -------
        :class:`IntegrationPoints`
            An new :class:`IntegrationPoints` instance containing the concatenated data or the modified current instance if :obj:`inplace` is :obj:`True`.


        Raises
        ------
        TypeError
            If the other object is not an :class:`IntegrationPoints` instance.
        ValueError
            If the dimensions of the two :class:`IntegrationPoints` instances do not match.

        """
        # Check if other is a IntegrationPoints instance
        if not isinstance(other, IntegrationPoints):
            raise ValueError("Input must be an instance of IntegrationPoints.")
        if self.n_topological_dimensions != other.n_topological_dimensions:
            raise ValueError("IntegrationPoints must have the same number of topological dimensions to concatenate.")
        if not isinstance(inplace, bool):
            raise ValueError("inplace must be a boolean value.")
        
        # Extract the number of points
        n_points_self = self.n_points
        n_points_other = other.n_points
        
        # Concatenate natural coordinates and element indices
        concatenated_natural_coordinates = numpy.vstack((self._natural_coordinates.copy(), other._natural_coordinates.copy()))
        concatenated_element_indices = numpy.concatenate((self._element_indices.copy(), other._element_indices.copy()))
        concatenated_weights = None
        if self._weights is not None or other._weights is not None:
            weights_self = self._weights if self._weights is not None else numpy.ones(n_points_self, dtype=numpy.float64)
            weights_other = other._weights if other._weights is not None else numpy.ones(n_points_other, dtype=numpy.float64)
            concatenated_weights = numpy.concatenate((weights_self.copy(), weights_other.copy()))
        
        # Concatenate properties
        all_keys = set(self._properties.keys()).union(set(other._properties.keys()))
        concatenated_properties = {}
        
        for key in all_keys:
            # Get property values or fill with NaNs if not present
            values_self = self._properties.get(key, None)
            values_other = other._properties.get(key, None)
            
            # Ensure proper shape
            if values_self is not None and values_other is not None:
                if not values_self.ndim == 2 or not values_other.ndim == 2:
                    raise ValueError(f"Property '{key}' must be a 2D NumPy array in both connectivities.")
                if not values_self.shape[0] == n_points_self or not values_other.shape[0] == n_points_other:
                    raise ValueError(f"Property '{key}' must have shape (N_e, A) where N_e matches the number of elements in the respective connectivity.")
                if not values_self.shape[1] == values_other.shape[1]:
                    raise ValueError(f"Property '{key}' must have the same number of attributes (A) in both connectivities.")

            elif values_self is not None:
                if not values_self.ndim == 2:
                    raise ValueError(f"Property '{key}' must be a 2D NumPy array in the first connectivity.")
                if not values_self.shape[0] == n_points_self:
                    raise ValueError(f"Property '{key}' must have shape (N_e, A) where N_e matches the number of elements in the first connectivity.")
                values_other = numpy.full((n_points_other, values_self.shape[1]), numpy.nan)
               
            elif values_other is not None:
                if not values_other.ndim == 2:
                    raise ValueError(f"Property '{key}' must be a 2D NumPy array in the second connectivity.")
                if not values_other.shape[0] == n_points_other:
                    raise ValueError(f"Property '{key}' must have shape (N_e, A) where N_e matches the number of elements in the second connectivity.")
                values_self = numpy.full((n_points_self, values_other.shape[1]), numpy.nan)
                
            else:
                raise ValueError("This should never happen. Both property values are None when concatenating.")              
            
            # Concatenate property values
            concatenated_values = numpy.vstack((values_self.copy(), values_other.copy()))
            concatenated_properties[key] = concatenated_values   
        
        # Return new instance or modify in place
        if inplace:
            self._natural_coordinates = concatenated_natural_coordinates
            self._element_indices = concatenated_element_indices
            self._properties = concatenated_properties
            if concatenated_weights is not None:
                self._weights = concatenated_weights
            return self
        else:
            return self.__class__(natural_coordinates=concatenated_natural_coordinates, element_indices=concatenated_element_indices, weights=concatenated_weights, properties=concatenated_properties)


    def copy(self, copy_properties: bool = True) -> IntegrationPoints:
        r"""
        Create a copy of the current :class:`IntegrationPoints` instance.
        
        Parameters
        ----------
        copy_properties: :class:`bool`, optional
            If :obj:`True`, the properties of the integration points will also be copied to the new instance. Default is :obj:`True`.


        Returns
        -------
        :class:`IntegrationPoints`
            A new :class:`IntegrationPoints` object containing the same integration points as the current instance.

        """
        if not isinstance(copy_properties, bool):
            raise ValueError("copy_properties must be a boolean value.")
        
        obj = self.__class__(
            natural_coordinates=self._natural_coordinates.copy(),
            element_indices=self._element_indices.copy(),
            weights=None if self._weights is None else self._weights.copy(),
            properties={key: values.copy() for key, values in self._properties.items()} if copy_properties else {}
        )
                
        return obj

    def remove_points(self, indices: ArrayLike, inplace: bool = False) -> IntegrationPoints:
        r"""
        Remove specific integration points by their indices.

        Parameters
        ----------
        indices: ArrayLike
            The indices of the integration points to remove as a numpy ndarray with shape :math:`(R,)`,
            where :math:`R` is the number of points to remove.

        inplace: :class:`bool`, optional
            If :obj:`True`, modify the current instance in place, and return itself. If :obj:`False`, return a new :class:`IntegrationPoints` instance (default is :obj:`False`).

        Returns
        -------
        :class:`IntegrationPoints`
            A new :class:`IntegrationPoints` instance with the specified points removed or the modified current instance if :obj:`inplace` is :obj:`True`.


        Raises
        ------
        IndexError
            If any index is out of bounds.

        
        Examples
        --------

        .. code-block:: python
            :linenos:
            
            import numpy
            from pysdic import IntegrationPoints 
            natural_coordinates = numpy.array([[0.5, 0.5], [0.5, 0.0], [0.0, 0.5], [1/3, 1/3]])
            element_indices = numpy.array([0, 0, 0, 1])
            weights = numpy.array([1.0, 1.0, 1.0, 1.0])

            integration_points = IntegrationPoints(natural_coordinates, element_indices, weights)
            print(integration_points.n_points)  # Output: 4

            # Remove points at indices 1 and 2
            new_integration_points = integration_points.remove_points(numpy.array([1, 2]))
            print(new_integration_points.n_points)  # Output: 2

            # Remove points at indices 0 and 1 in place
            integration_points.remove_points(numpy.array([0, 1]), inplace=True)
            print(integration_points.n_points)  # Output: 2

        """
        indices = numpy.asarray(indices, dtype=int)
        if indices.ndim != 1:
            raise ValueError("indices should be a 1D array of shape (R,).")
        if numpy.any(indices < 0) or numpy.any(indices >= self.n_points):
            raise IndexError("Some indices are out of bounds.")
        if not isinstance(inplace, bool):
            raise TypeError("inplace must be a boolean.")
        
        mask = numpy.ones(self.n_points, dtype=bool)
        mask[indices] = False

        if inplace:
            self._natural_coordinates = self.natural_coordinates[mask]
            self._element_indices = self.element_indices[mask]
            if self._weights is not None:
                self._weights = self._weights[mask]
                
            for key in self._properties.keys():
                self._properties[key] = self._properties[key][mask, :]
        
            return self

        out = IntegrationPoints(self.natural_coordinates[mask].copy(), self.element_indices[mask].copy(), None if self._weights is None else self.weights[mask].copy(), self.n_topological_dimensions)
        for key in self._properties.keys():
            out.set_property(key, self._properties[key][mask, :].copy())
        return out


    def remove_invalids(self, inplace: bool = False) -> IntegrationPoints:
        r"""
        Remove all invalid integration points (points not included in any element).

        A point is considered invalid if its element index is :obj:`-1`.

        .. seealso::

            - :meth:`remove_points` to remove specific integration points.

        Parameters
        ----------
        inplace: :class:`bool`, optional
            If :obj:`True`, modify the current instance in place, and return itself. If :obj:`False`, return a new :class:`IntegrationPoints` instance (default is :obj:`False`).

        Returns
        -------
        :class:`IntegrationPoints`
            A new :class:`IntegrationPoints` instance with the invalid points removed or the modified current instance if :obj:`inplace` is :obj:`True`.

        """
        mask = self.element_indices == -1
        return self.remove_points(numpy.where(mask)[0], inplace=inplace)


    def disable_points(self, indices: ArrayLike, inplace: bool = False) -> IntegrationPoints:
        r"""
        Disable specific integration points by their indices without removing them.

        Disabled points will have their element indices set to :obj:`-1` and their natural coordinates set to :obj:`NaN`.

        The inverse operation can be done by re-assigning valid values to the natural_coordinates and element_indices attributes.

        Parameters
        ----------
        indices: ArrayLike
            The indices of the integration points to disable as a numpy ndarray with shape (:math:`R`,),
            where :math:`R` is the number of points to disable.

        inplace: :class:`bool`, optional
            If :obj:`True`, modify the current instance in place. If :obj:`False`, return a new :class:`IntegrationPoints` instance (default is :obj:`False`).

        Returns
        -------
        :class:`IntegrationPoints`
            A new :class:`IntegrationPoints` instance with the specified points disabled or the modified current instance if :obj:`inplace` is :obj:`True`.

        Raises
        ------
        IndexError
            If any index is out of bounds.
        """
        indices = numpy.asarray(indices, dtype=int)
        if indices.ndim != 1:
            raise ValueError("indices should be a 1D array of shape (R,).")
        if numpy.any(indices < 0) or numpy.any(indices >= self.n_points):
            raise IndexError("Some indices are out of bounds.")
        if not isinstance(inplace, bool):
            raise TypeError("inplace must be a boolean.")
    
        if inplace:
            self._natural_coordinates[indices, :] = numpy.nan
            self._element_indices[indices] = -1
            return self

        # Create copies
        new_natural_coordinates = self.natural_coordinates.copy()
        new_element_indices = self.element_indices.copy()
        new_weights = self.weights.copy()
        new_natural_coordinates[indices, :] = numpy.nan
        new_element_indices[indices] = -1

        return IntegrationPoints(new_natural_coordinates, new_element_indices, new_weights, properties=self.copy_properties())