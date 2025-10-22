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

from typing import Optional
import numpy

class IntegratedPoints(object):
    r"""
    Base class to store integrated points for numerical integration over elements and localisation of points into :class:`Mesh3D` objects.

    Integrated points are defined in the reference element using natural coordinates, element indices, and weights.

    The natural coordinates (:math:`\xi, \eta, \zeta, ...`) for the integrated points satisfy:

    - :math:`0 \leq \xi, \eta, \zeta, ... \leq 1`
    - :math:`\xi + \eta + \zeta + ... \leq 1`

    The number of natural coordinates depends on the topological dimension of the element :math:`d` noted ``n_dimensions``:

    .. note::

        If the weights are not provided, equal weights are assumed for all integrated points.
        By default the weights are set to 1 for all integrated points (and not :math:`1/Np`).

    If a specific specific point is not include in any element, it can be identified by setting its element ID to -1 and its natural coordinates to NaN.

    Parameters
    ----------
    natural_coordinates : numpy.ndarray
        The natural coordinates of the integrated points as a numpy ndarray with shape (Np, d),
        where Np is the number of integrated points and d is the topological dimension of the element.

    element_indices : numpy.ndarray
        The element indices of the integrated points as a numpy ndarray with shape (Np,),
        where Np is the number of integrated points.

    weights : Optional[numpy.ndarray], optional
        The weights of the integrated points as a numpy ndarray with shape (Np,),
        where Np is the number of integrated points, by default None means equal weights of 1 for all points.

    n_dimension: Optional[int], optional
        The topological dimension of the element, by default None means it will be inferred from the shape of the natural_coordinates.
        Otherwise, an error will be raised if the provided dimension does not match the shape of the natural_coordinates.

    internal_bypass : bool, optional
        If True, internal checks are bypassed for better performance, by default False.

    Examples
    --------

    Create a IntegratedPoints object with 4 integrated points in a 2D element (triangle - :math:`d=2`):

    .. code-block:: python

        import numpy
        from pysdic import IntegratedPoints

        natural_coordinates = numpy.array([[0.5, 0.5], [0.5, 0.0], [0.0, 0.5], [1/3, 1/3]])
        element_indices = numpy.array([0, 0, 0, 1])
        weights = numpy.array([1.0, 1.0, 1.0, 1.0])

        integrated_points = IntegratedPoints(natural_coordinates, element_indices, weights)
        print(integrated_points.n_points)  # Output: 4
        print(integrated_points.n_dimensions)  # Output: 2

    Operations
    ----------

    - ``len(integrated_points)``: Returns the number of points in the integrated points (see :meth:`n_points`).
    - ``integrated_points1 + integrated_points2``: Concatenates two integrated points (The new integrated points is a copy of original integrated points, not same instance).

    """
    __slots__ = ['_natural_coordinates', '_element_ids', '_weights', '_n_dimensions', '_internal_bypass']

    def __init__(self, natural_coordinates: numpy.ndarray, element_indices: numpy.ndarray, weights: Optional[numpy.ndarray] = None, n_dimensions: Optional[int] = None, internal_bypass: bool = False):
        # Type checks and conversions
        natural_coordinates = numpy.asarray(natural_coordinates, dtype=numpy.float64)
        element_indices = numpy.asarray(element_indices, dtype=int)
        if weights is not None:
            weights = numpy.asarray(weights, dtype=numpy.float64)

        if natural_coordinates.ndim != 2:
            raise ValueError("natural_coordinates should be a 2D array of shape (Np, d).")
        if element_indices.ndim != 1:
            raise ValueError("element_indices should be a 1D array of shape (Np,).")
        if natural_coordinates.shape[0] != element_indices.shape[0]:
            raise ValueError("The number of integrated points Np in natural_coordinates and element_indices should match.")
        if weights is not None and weights.ndim != 1:
            raise ValueError("weights should be a 1D array of shape (Np,).")
        if weights is not None and weights.shape[0] != element_indices.shape[0]:
            raise ValueError("The number of integrated points Np in weights and element_indices should match.")
        if n_dimensions is not None and (not isinstance(n_dimensions, int) or n_dimensions <= 0):
            raise ValueError("n_dimensions should be a positive integer.")
        if n_dimensions is not None and natural_coordinates.shape[1] != n_dimensions:
            raise ValueError("The provided n_dimensions does not match the shape of the natural_coordinates.")
        
        # Set the attributes
        self.internal_bypass = True
        self.natural_coordinates = natural_coordinates
        self.element_indices = element_indices
        self._n_dimensions = natural_coordinates.shape[1] if n_dimensions is None else n_dimensions
        self.weights = weights
        self.internal_bypass = internal_bypass
        self.validate()

    # =======================
    # Internals
    # =======================
    @property
    def internal_bypass(self) -> bool:
        r"""
        Get and set the internal bypass mode status.
        When enabled, internal checks are skipped for better performance.

        This is useful for testing purposes, but should not be used in production code.
        Please ensure that all necessary checks are performed before using this mode.

        Parameters
        ----------
        value : bool
            If True, internal checks are bypassed. If False, internal checks are performed.

        Raises
        --------
        TypeError
            If the value is not a boolean.
        """
        return self._internal_bypass
    
    @internal_bypass.setter
    def internal_bypass(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError(f"Bypass mode must be a boolean, got {type(value)}.")
        self._internal_bypass = value

    def _internal_check_consistency(self, coord: bool, ids: bool, weights: bool) -> None:
        r"""
        Internal method to check the consistency of the attributes.

        All checks are skipped if internal_bypass is True.

        Otherwise, the shape and type of the attributes are always checked and the content of arrays is checked based on the input flags.
        
        Parameters
        ----------
        coord : bool
            If True, check the natural_coordinates.
            
        ids : bool
            If True, check the element_indices.
            
        weights : bool
            If True, check the weights.
            
        """
        if self.internal_bypass:
            return

        if not isinstance(self._n_dimensions, int):
            raise TypeError("dimension should be an integer.")
        if self._n_dimensions <= 0:
            raise ValueError("dimension should be a positive integer.")

        if not isinstance(self._natural_coordinates, numpy.ndarray):
            raise TypeError("natural_coordinates should be a numpy ndarray.")
        if self._natural_coordinates.ndim != 2:
            raise ValueError("natural_coordinates should be a 2D array of shape (Np, d).")
        if not self._natural_coordinates.shape == (self.n_points, self.n_dimensions):
            raise ValueError("natural_coordinates shape does not match (Np, d).")
        if not self._natural_coordinates.dtype == numpy.float64:
            raise TypeError("natural_coordinates should be of type numpy.float64.")
        
        if not isinstance(self._element_ids, numpy.ndarray):
            raise TypeError("element_indices should be a numpy ndarray.")
        if self._element_ids.ndim != 1:
            raise ValueError("element_indices should be a 1D array of shape (Np,).")
        if not self._element_ids.shape[0] == self.n_points:
            raise ValueError("element_indices shape does not match (Np,).")
        if not self._element_ids.dtype == int:
            raise TypeError("element_indices should be of type int.")
        
        if self._weights is not None and not isinstance(self._weights, numpy.ndarray):
            raise TypeError("weights should be a numpy ndarray.")
        if self._weights is not None and self._weights.ndim != 1:
            raise ValueError("weights should be a 1D array of shape (Np,).")
        if self._weights is not None and not self._weights.shape[0] == self.n_points:
            raise ValueError("weights shape does not match (Np,).")
        if self._weights is not None and not self._weights.dtype == numpy.float64:
            raise TypeError("weights should be of type numpy.float64.")

        if coord:
            valid_coordinates = self._natural_coordinates[~numpy.isnan(self._natural_coordinates).all(axis=1)]
            if numpy.any(valid_coordinates < 0) or numpy.any(valid_coordinates > 1):
                raise ValueError("All natural coordinates should be in the range [0, 1].")
            if numpy.any(numpy.sum(valid_coordinates, axis=1) > 1):
                raise ValueError("The sum of natural coordinates for each point should be less than or equal to 1.")
        if ids:
            valid_element_ids = self._element_ids[self._element_ids != -1]
            if numpy.any(valid_element_ids < 0):
                raise ValueError("All element IDs should be non-negative.")
        if weights and self._weights is not None:
            if numpy.any(self._weights < 0):
                raise ValueError("All weights should be non-negative.")
            if numpy.all(self._weights == 0):
                raise ValueError("At least one weight should be positive.")



    # =======================
    # Properties
    # =======================
    @property
    def natural_coordinates(self) -> numpy.ndarray:
        r"""
        Get or set the natural coordinates of the integrated points as a numpy ndarray with shape (Np, d),
        where Np is the number of integrated points and d is the topological dimension of the element.

        If a point is not included in any element, its natural coordinates should be set to NaN.

        .. note::

            If the number of integrated points (Np) is changed, the properties should be updated accordingly.
            To avoid inconsistencies, it is recommended to create a new IntegratedPoints instance with the updated coordinates and ids or use the appropriate methods to modify the points.

            Otherwise, you should set `internal_bypass` to True before modifying the connectivity, and set it back to False afterwards.

            .. code-block:: python

                mesh.internal_bypass = True
                mesh.connectivity = new_connectivity
                mesh.clear_elements_properties()  # Optional: clear properties if they are no longer valid
                mesh.elements_uvmap = new_uvmap  # Optional: set new uvmap if needed or other properties
                mesh.internal_bypass = False
                mesh.validate()  # Optional: ensure the mesh is still valid

        .. seealso::

            - :meth:`remove_points` to remove specific integrated points.
            - :meth:`add_points` to add new integrated points.
            - :meth:`disable_points` to disable specific integrated points without removing them.

        Parameters
        ----------
        value : numpy.ndarray
            The new natural coordinates of the integrated points as a numpy ndarray with shape (Np, d).

        Returns
        -------
        numpy.ndarray
            The natural coordinates of the integrated points.
        """
        return self._natural_coordinates
    
    @natural_coordinates.setter
    def natural_coordinates(self, value: numpy.ndarray) -> None:
        value = numpy.asarray(value, dtype=numpy.float64)
        self._natural_coordinates = value
        self._internal_check_consistency(coord=True, ids=False, weights=False)

    @property
    def element_indices(self) -> numpy.ndarray:
        r"""
        Get or set the element indices of the integrated points as a numpy ndarray with shape (Np,),
        where Np is the number of integrated points.

        if a point is not included in any element, its element ID should be set to -1.

        .. note::

            If the number of integrated points (Np) is changed, the properties should be updated accordingly.
            To avoid inconsistencies, it is recommended to create a new IntegratedPoints instance with the updated coordinates and ids or use the appropriate methods to modify the points.

            Otherwise, you should set `internal_bypass` to True before modifying the connectivity, and set it back to False afterwards.

            .. code-block:: python

                mesh.internal_bypass = True
                mesh.connectivity = new_connectivity
                mesh.clear_elements_properties()  # Optional: clear properties if they are no longer valid
                mesh.elements_uvmap = new_uvmap  # Optional: set new uvmap if needed or other properties
                mesh.internal_bypass = False
                mesh.validate()  # Optional: ensure the mesh is still valid

        .. seealso::

            - :meth:`remove_points` to remove specific integrated points.
            - :meth:`add_points` to add new integrated points.
            - :meth:`disable_points` to disable specific integrated points without removing them.

        Parameters
        ----------
        value : numpy.ndarray
            The new element IDs of the integrated points as a numpy ndarray with shape (Np,).

        Returns
        -------
        numpy.ndarray
            The element IDs of the integrated points.
        """
        return self._element_ids

    @element_indices.setter
    def element_indices(self, value: numpy.ndarray) -> None:
        value = numpy.asarray(value, dtype=int)
        self._element_ids = value
        self._internal_check_consistency(coord=False, ids=True, weights=False)

    @property
    def weights(self) -> numpy.ndarray:
        r"""
        Get or set the weights of the integrated points as a numpy ndarray with shape (Np,),
        where Np is the number of integrated points.

        If weights are not provided, equal weights of 1 are assumed for all points.

        .. note::

            If the number of integrated points (Np) is changed, the properties should be updated accordingly.
            To avoid inconsistencies, it is recommended to create a new IntegratedPoints instance with the updated coordinates and ids or use the appropriate methods to modify the points.

            Otherwise, you should set `internal_bypass` to True before modifying the connectivity, and set it back to False afterwards.

            .. code-block:: python

                mesh.internal_bypass = True
                mesh.connectivity = new_connectivity
                mesh.clear_elements_properties()  # Optional: clear properties if they are no longer valid
                mesh.elements_uvmap = new_uvmap  # Optional: set new uvmap if needed or other properties
                mesh.internal_bypass = False
                mesh.validate()  # Optional: ensure the mesh is still valid
            
        .. seealso::

            - :meth:`remove_points` to remove specific integrated points.
            - :meth:`add_points` to add new integrated points.
            - :meth:`disable_points` to disable specific integrated points without removing them.

        Returns
        -------
        numpy.ndarray
            The weights of the integrated points. If not provided, returns an array of ones with shape (Np,).

        """
        if self._weights is None:
            return numpy.ones(self.n_points, dtype=numpy.float64)
        return self._weights
    
    @weights.setter
    def weights(self, value: Optional[numpy.ndarray]) -> None:
        if value is None:
            self._weights = None
        else:
            value = numpy.asarray(value, dtype=numpy.float64)
            self._weights = value
        self._internal_check_consistency(coord=False, ids=False, weights=True)

    @property
    def n_dimensions(self) -> int:
        r"""
        The topological dimension of the element.

        This is inferred from the shape of the natural_coordinates if not provided during instantiation.

        Returns
        -------
        int
            The topological dimension of the element.
        """
        return self._n_dimensions
    
    @property
    def n_points(self) -> int:
        r"""
        The number of integrated points.

        Returns
        -------
        int
            The number of integrated points Np.
        """
        return self._natural_coordinates.shape[0]
    
    @property
    def n_valids(self) -> int:
        r"""
        The number of valid integrated points (points included in an element).

        A point is considered valid if its element index is not -1.

        Returns
        -------
        int
            The number of valid integrated points.
        """
        return numpy.sum(self._element_ids != -1)
    
    def shape(self) -> tuple[int, int]:
        r"""
        The shape of the integrated points data.

        Returns
        -------
        tuple[int, int]
            A tuple (Np, d) where Np is the number of integrated points and d is the topological dimension of the element.
        """
        return (self.n_points, self.n_dimensions)
    
    # ======================
    # Dunder methods
    # =======================
    def __len__(self) -> int:
        r"""
        Get the number of integrated points.

        Returns
        -------
        int
            The number of integrated points Np.
        """
        return self.n_points
    
    def __add__(self, other: IntegratedPoints) -> IntegratedPoints:
        r"""
        Concatenate two IntegratedPoints instances.

        The new IntegratedPoints instance is a copy of the original instances, not the same instance.

        Parameters
        ----------
        other : IntegratedPoints
            Another IntegratedPoints instance to concatenate with.

        Returns
        -------
        IntegratedPoints
            A new IntegratedPoints instance containing the concatenated data.

        Raises
        ------
        TypeError
            If the other object is not an IntegratedPoints instance.
        ValueError
            If the dimensions of the two IntegratedPoints instances do not match.
        """
        if not isinstance(other, IntegratedPoints):
            raise TypeError("Can only concatenate with another IntegratedPoints instance.")
        if self.n_dimensions != other.n_dimensions:
            raise ValueError("Cannot concatenate IntegratedPoints with different dimensions.")
        
        new_natural_coordinates = numpy.vstack((self.natural_coordinates, other.natural_coordinates))
        new_element_ids = numpy.hstack((self.element_indices, other.element_indices))
        
        if self._weights is None and other._weights is None:
            new_weights = None
        else:
            current_weights = self.weights 
            other_weights = other.weights 
            new_weights = numpy.hstack((current_weights, other_weights))

        return IntegratedPoints(new_natural_coordinates, new_element_ids, new_weights, self.n_dimensions, self.internal_bypass and other.internal_bypass)

    # =======================
    # Methods
    # =======================
    def validate(self) -> None:
        r"""
        Validate the consistency of the IntegratedPoints instance.

        This method checks the internal consistency of the attributes and raises an error if any inconsistency is found.

        It is recommended to call this method after modifying the attributes directly (when `internal_bypass` is True).

        Raises
        ------
        ValueError
            If any inconsistency is found in the attributes.
        """
        self._internal_check_consistency(coord=True, ids=True, weights=True)

    def copy(self) -> IntegratedPoints:
        r"""
        Create a deep copy of the IntegratedPoints instance.

        Returns
        -------
        IntegratedPoints
            A new IntegratedPoints instance with the same attributes as the original.
        """
        return IntegratedPoints(self.natural_coordinates.copy(), self.element_indices.copy(), None if self._weights is None else self._weights.copy(), self.n_dimensions, self.internal_bypass)

    def remove_points(self, indices: numpy.ndarray) -> None:
        r"""
        Remove specific integrated points by their indices.

        Parameters
        ----------
        indices : numpy.ndarray
            The indices of the integrated points to remove as a numpy ndarray with shape (R,),
            where R is the number of points to remove.

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
        
        mask = numpy.ones(self.n_points, dtype=bool)
        mask[indices] = False

        current_bypass = self.internal_bypass
        self.internal_bypass = True
        self.natural_coordinates = self.natural_coordinates[mask]
        self.element_indices = self.element_indices[mask]
        if self._weights is not None:
            self.weights = self.weights[mask]
        self.internal_bypass = current_bypass

    def remove_invalids(self) -> None:
        r"""
        Remove all invalid integrated points (points not included in any element).

        A point is considered invalid if its element index is -1.
        """
        mask = self.element_indices == -1
        self.remove_points(numpy.where(mask)[0])

    def add_points(self, natural_coordinates: numpy.ndarray, element_indices: numpy.ndarray, weights: Optional[numpy.ndarray] = None) -> None:
        r"""
        Add new integrated points.

        Parameters
        ----------
        natural_coordinates : numpy.ndarray
            The natural coordinates of the new integrated points as a numpy ndarray with shape (A, d),
            where A is the number of points to add and d is the topological dimension of the element.

        element_indices : numpy.ndarray
            The element IDs of the new integrated points as a numpy ndarray with shape (A,),
            where A is the number of points to add.

        weights : Optional[numpy.ndarray], optional
            The weights of the new integrated points as a numpy ndarray with shape (A,),
            where A is the number of points to add, by default None means equal weights of 1 for all new points.

        Raises
        ------
        ValueError
            If the shapes of the inputs are inconsistent or if the dimension does not match.
        """
        natural_coordinates = numpy.asarray(natural_coordinates, dtype=numpy.float64)
        element_indices = numpy.asarray(element_indices, dtype=int)
        if weights is not None:
            weights = numpy.asarray(weights, dtype=numpy.float64)

        if natural_coordinates.ndim != 2:
            raise ValueError("natural_coordinates should be a 2D array of shape (A, d).")
        if element_indices.ndim != 1:
            raise ValueError("element_indices should be a 1D array of shape (A,).")
        if natural_coordinates.shape[0] != element_indices.shape[0]:
            raise ValueError("The number of new points A in natural_coordinates and element_indices should match.")
        if weights is not None and weights.ndim != 1:
            raise ValueError("weights should be a 1D array of shape (A,).")
        if weights is not None and weights.shape[0] != element_indices.shape[0]:
            raise ValueError("The number of new points A in weights and element_indices should match.")
        if natural_coordinates.shape[1] != self.n_dimensions:
            raise ValueError("The n_dimensions d of the new natural_coordinates does not match the existing dimension.")

        current_bypass = self.internal_bypass
        self.internal_bypass = True
        self.natural_coordinates = numpy.vstack((self.natural_coordinates, natural_coordinates))
        self.element_indices = numpy.hstack((self.element_indices, element_indices))
        if self._weights is not None or weights is not None:
            current_weights = self.weights
            new_weights = numpy.ones(element_indices.shape[0], dtype=numpy.float64) if weights is None else weights
            self.weights = numpy.hstack((current_weights, new_weights))
        self.internal_bypass = current_bypass
        self.validate()

    def disable_points(self, indices: numpy.ndarray) -> None:
        r"""
        Disable specific integrated points by their indices without removing them.

        Disabled points will have their element indices set to -1 and their natural coordinates set to NaN.

        The inverse operation can be done by re-assigning valid values to the natural_coordinates and element_indices attributes.

        Parameters
        ----------
        indices : numpy.ndarray
            The indices of the integrated points to disable as a numpy ndarray with shape (R,),
            where R is the number of points to disable.

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

        current_bypass = self.internal_bypass
        self.internal_bypass = True
        self.natural_coordinates[indices, :] = numpy.nan
        self.element_indices[indices] = -1
        self.internal_bypass = current_bypass
        
