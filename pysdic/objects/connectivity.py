# Copyright 2026 Artezaru
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

from typing import Optional, Dict, Tuple, List
from numpy.typing import ArrayLike
from numbers import Integral, Real

import numpy
import os


class Connectivity(object):
    r"""
    A class representing the connectivity information of a mesh.

    Connectivity can store any scalar/vector properties associated with each element in the mesh.
    
    If :obj:`element_type` is provided, the mesh will enforce that all elements are of the specified type. 
    This allow to use the predefined shape functions and properties associated with that element type.
    The implemented types are:

    +---------------------+-------------------------------------------------------------------+
    | Element Type        | Description                                                       |
    +=====================+===================================================================+
    | "segment_2"         | 2-node line element                                               |
    +---------------------+-------------------------------------------------------------------+
    | "segment_3"         | 3-node line element                                               |
    +---------------------+-------------------------------------------------------------------+
    | "triangle_3"        | 3-node triangular element                                         |
    +---------------------+-------------------------------------------------------------------+
    | "triangle_6"        | 6-node triangular element                                         |
    +---------------------+-------------------------------------------------------------------+
    | "quadrangle_4"      | 4-node quadrilateral element                                      |
    +---------------------+-------------------------------------------------------------------+
    | "quadrangle_8"      | 8-node quadrilateral element                                      |
    +---------------------+-------------------------------------------------------------------+


    .. note::

        The number of vertices per element of the connectivity is not designed to change after connectivity creation.
        
    .. note::
    
        The connections and the properties of the elements are stored as copy-on-write (cow) NumPy arrays of type :obj:`numpy.int64` (connectivity) and :obj:`numpy.float64` (properties). 
        Accesing these arrays directly will return unwritable views to prevent accidental modifications.
        

    Parameters
    ----------
    elements: ArrayLike
        A NumPy array of shape :math:`(N_{e}, N_{vpe})` representing the connectivity of the mesh, where :math:`N_{e}` is the number of elements and :math:`N_{vpe}` is the number of vertices per element.
        
    properties: Optional[Dict[:class:`str`, ArrayLike]], optional
        A dictionary of additional properties to associate with the connectivity. Keys are property names, and values are NumPy arrays of property data of shape :math:`(N_{e}, A)` where :math:`A` is the number of attributes per element for the given property.
        Default is :obj:`None`, meaning no additional properties are set.
        
    element_type: Optional[:class:`str`], optional
        The expected type of elements in the connectivity. If provided, the connectivity will enforce that all elements are of the specified type.
        Default is :obj:`None`.
                

    Raises  
    ------
    TypeError
        If the input elements is not a NumPy array.
        If the given dimension is not an integer.
        If the properties is not a dictionary of string keys and NumPy array values.
        If the element type is not a string.
        
    ValueError
        If the input array does not have the correct shape.
        If the given dimension is not a strictly positive integer.
        If the properties arrays do not have the correct shape.
        If the element type is not one of the implemented types.

    """    
    __slots__ = [
        "_elements",
        "_properties",
        "_element_type",
    ]
    
    _mapping_element_type_to_properties = {
        "segment_2": {
            "expected_N_vpe": 2,
            "expected_K": 1,
            "meshio_cell_type": "line",
            "vtk_cell_type": 3,
        },
        "segment_3": {
            "expected_N_vpe": 3,
            "expected_K": 1,
            "meshio_cell_type": "line3",
            "vtk_cell_type": 21,
        },
        "triangle_3": {
            "expected_N_vpe": 3,
            "expected_K": 2,
            "meshio_cell_type": "triangle",
            "vtk_cell_type": 5,
        },
        "triangle_6": {
            "expected_N_vpe": 6,
            "expected_K": 2,
            "meshio_cell_type": "triangle6",
            "vtk_cell_type": 22,
        },
        "quadrangle_4": {
            "expected_N_vpe": 4,
            "expected_K": 2,
            "meshio_cell_type": "quad",
            "vtk_cell_type": 9,
        },
        "quadrangle_8": {
            "expected_N_vpe": 8,
            "expected_K": 2,
            "meshio_cell_type": "quad8",
            "vtk_cell_type": 23,
        },
    }
    
    def __init__(
        self, 
        elements: ArrayLike, 
        properties: Optional[Dict[str, ArrayLike]] = None,
        element_type: Optional[str] = None
    ) -> None:
        # Convert points to 2D NumPy array
        self._element_type = None  # Temporary assignment to allow validation in set_element_type
        elements = numpy.asarray(elements, dtype=numpy.int64).copy()
        if elements.ndim == 1:
            elements = elements.reshape((-1, 1))
            
        if not elements.ndim == 2:
            raise ValueError("Elements must be a 2D NumPy array with shape (N_e, N_vpe).")

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
            if not (values.ndim == 2 and values.shape[0] == elements.shape[0]):
                raise ValueError(f"Property values must be a 2D NumPy array with shape (N_e, A) where N_e={elements.shape[0]}.")
           
        # Save elements
        self._elements = elements
        self._properties = {}
        
        # Element type handling
        self.element_type = element_type
        
        # Save properties
        for key, values in properties.items():
            self.set_property(key, values)

    
    
    # ==========================
    # Properties
    # ==========================
    @property
    def elements(self) -> numpy.ndarray:
        r"""
        [Get or Set] An numpy array of shape :math:`(N_e, N_{vpe})` representing the connectivity of the mesh.
        
        .. note::
        
            The property is copied-on-write (cow) with type :obj:`numpy.int64`. Accessing it will return an unwritable view to prevent accidental modifications.
            

        Parameters
        ----------
        value: ArrayLike
            An array-like of shape :math:`(N_e, N_{vpe})` representing the connectivity of the mesh. If 1D array is provided, it will be reshaped to :math:`(N_e, 1)`.


        Returns
        -------
        :class:`numpy.ndarray`
            A unwritable NumPy array of shape :math:`(N_e, N_{vpe})` containing the connectivity of the mesh.


        Raises
        ------
        TypeError
            If the input points is not a NumPy array.
        
        ValueError
            If the input array does not have the correct shape :math:`(N_e, N_{vpe})`.
        """
        view = self._elements.view()
        view.flags.writeable = False
        return view
    
    @elements.setter
    def elements(self, value: ArrayLike) -> None:
        elements = numpy.asarray(value, dtype=numpy.int64).copy()
        if self.n_vertices_per_element == 1 and elements.ndim == 1:
            elements = elements.reshape((-1, 1))
        if not (elements.ndim == 2 and elements.shape[1] == self.n_vertices_per_element):
            raise ValueError(f"Elements must be a 2D NumPy array with shape (N_e, {self.n_vertices_per_element}).")
        if not elements.shape[0] == self.n_elements:
            raise ValueError(f"Elements must have the same number of elements N_e={self.n_elements}. Consider using remove/add element methods to change the number of elements.")
        if any(elements < 0):
            raise ValueError("Vertices indices in elements must be non-negative.")
        self._elements = elements
        
    @property
    def n_elements(self) -> int:
        r"""
        [Get] The number of elements :math:`N_e` in the mesh.

        .. note::

            You can also use `len(connectivity)`.

        .. seealso::

            - :meth:`shape` for getting the shape of the elements array.
            - :meth:`n_vertices_per_element` for getting the number of vertices per element.
            

        Returns
        -------
        :class:`int`
            The number of points in the point cloud.

        """
        return self._elements.shape[0]
    
    def __len__(self) -> int:
        return self.n_elements

    
    @property
    def shape(self) -> Tuple[int, int]:
        r"""
        [Get] The shape of the elements array (:math:`N_e`, :math:`N_{vpe}`).

        .. seealso::

            - :meth:`n_elements` for getting the number of elements in the mesh.
            - :meth:`n_vertices_per_element` for getting the number of vertices per element.


        Returns
        -------
        tuple[:class:`int`, :class:`int`]
            A tuple representing the shape of the elements array (:math:`N_e`, :math:`N_{vpe}`).

        """
        return self._elements.shape
    
    
    @property
    def n_vertices_per_element(self) -> int:
        r"""
        [Get] The number of vertices per element :math:`N_{vpe}` in the mesh.

        .. seealso::

            - :meth:`n_elements` for getting the number of elements in the mesh.
            - :meth:`shape` for getting the shape of the elements array.

        Returns
        -------
        :class:`int`
            The number of vertices per element in the mesh.
            
        """
        return self._elements.shape[1]
    
    
    @property
    def element_type(self) -> Optional[str]:
        r"""
        [Get or Set] The expected type of elements in the connectivity.
        
        +---------------------+-------------------------------------------------------------------+
        | Element Type        | Description                                                       |
        +=====================+===================================================================+
        | "segment_2"         | 2-node line element                                               |
        +---------------------+-------------------------------------------------------------------+
        | "segment_3"         | 3-node line element                                               |
        +---------------------+-------------------------------------------------------------------+
        | "triangle_3"        | 3-node triangular element                                         |
        +---------------------+-------------------------------------------------------------------+
        | "triangle_6"        | 6-node triangular element                                         |
        +---------------------+-------------------------------------------------------------------+
        | "quadrangle_4"      | 4-node quadrilateral element                                      |
        +---------------------+-------------------------------------------------------------------+
        | "quadrangle_8"      | 8-node quadrilateral element                                      |
        +---------------------+-------------------------------------------------------------------+
        
        Parameters
        ----------
        value: Optional[:class:`str`]
            The expected type of elements in the connectivity. If :obj:`None`, no type enforcement is applied.
            

        Returns
        -------
        Optional[:class:`str`]
            The expected type of elements in the connectivity.
            
        
        Raises
        ------
        TypeError
            If the element type is not a string or :obj:`None`.
        
        ValueError
            If the element type is not one of the implemented types.
            If the current elements do not match the expected type.
            
        """
        return self._element_type
    
    @element_type.setter
    def element_type(self, value: Optional[str]) -> None:
        if value is not None:
            if not isinstance(value, str):
                raise TypeError("Element type must be a string or None.")
            if value not in self._mapping_element_type_to_properties:
                raise ValueError(f"Element type '{value}' is not implemented. Supported types are: {list(self._mapping_element_type_to_properties.keys())}.")
            expected_N_vpe = self._mapping_element_type_to_properties[value]["expected_N_vpe"]
            if not self.n_vertices_per_element == expected_N_vpe:
                raise ValueError(f"Current elements have {self.n_vertices_per_element} vertices per element, which does not match the expected {expected_N_vpe} for element type '{value}'.")
        self._element_type = value
    

    @property
    def n_topological_dimensions(self) -> Optional[int]:
        r"""
        [Get] The topological dimension :math:`K` of the elements in the connectivity, if :obj:`element_type` is set.

        Returns
        -------
        Optional[:class:`int`]
            The topological dimension :math:`K` of the elements in the connectivity, or :obj:`None` if :obj:`element_type` is not set.
        """
        if self._element_type is None:
            return None
        return self._mapping_element_type_to_properties[self._element_type]["expected_K"]
    
    @property
    def _meshio_cell_type(self) -> Optional[str]:
        r"""
        [Get] The corresponding meshio cell type for the elements, if :obj:`element_type` is set.

        Returns
        -------
        Optional[:class:`str`]
            The corresponding meshio cell type for the elements, or :obj:`None` if :obj:`element_type` is not set.
        """
        if self._element_type is None:
            return None
        return self._mapping_element_type_to_properties[self._element_type]["meshio_cell_type"]
    
    @property
    def _vtk_cell_type(self) -> Optional[int]:
        r"""
        [Get] The corresponding VTK cell type for the elements, if :obj:`element_type` is set.

        Returns
        -------
        Optional[:class:`int`]
            The corresponding VTK cell type for the elements, or :obj:`None` if :obj:`element_type` is not set.
        """
        if self._element_type is None:
            return None
        return self._mapping_element_type_to_properties[self._element_type]["vtk_cell_type"]
    
    @property
    def _expected_N_vpe(self) -> Optional[int]:
        r"""
        [Get] The expected number of vertices per element :math:`N_{vpe}` for the elements, if :obj:`element_type` is set.

        Returns
        -------
        Optional[:class:`int`]
            The expected number of vertices per element :math:`N_{vpe}` for the elements, or :obj:`None` if :obj:`element_type` is not set.
        """
        if self._element_type is None:
            return None
        return self._mapping_element_type_to_properties[self._element_type]["expected_N_vpe"]
    
    @property
    def _expected_K(self) -> Optional[int]:
        r"""
        [Get] The expected topological dimension :math:`K` for the elements, if :obj:`element_type` is set.

        Returns
        -------
        Optional[:class:`int`]
            The expected topological dimension :math:`K` for the elements, or :obj:`None` if :obj:`element_type` is not set.
        """
        if self._element_type is None:
            return None
        return self._mapping_element_type_to_properties[self._element_type]["expected_K"]    
    
    def __repr__(self) -> str:
        string = f"Connectivity(n_elements={self.n_elements}, n_vertices_per_element={self.n_vertices_per_element})"
        if self._element_type is not None:
            string += f"\n  - Element Type: '{self.element_type}'"
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
        Set a property for the connectivity as a NumPy array of shape :math:`(N_e, A)` where :math:`A` is the number of attributes per element for the given property.

        .. note::
        
            The given array will be copied and stored as type :obj:`numpy.float64`.
            
        .. note::
        
            The property can also be set using dictionary-like syntax:
            
            .. code-block:: python
                :linenos:
                
                connectivity['property_key'] = property_values
                    

        Parameters
        ----------
        key: :class:`str`
            The key/name of the property to set.
        
        values: ArrayLike
            A NumPy array of shape :math:`(N_e, A)` or :math:`(N_e,)` representing the property values for each element in the connectivity.
            If 1D array is provided, it will be reshaped to :math:`(N_e, 1)`.


        Raises
        ------
        TypeError
            If the property key is not a string.
            If the input values is not a NumPy array.
            
        ValueError
            If the input array does not have the correct shape :math:`(N_e, A)` or :math:`(N_e,)`.
            

        Examples
        --------
        Set a property for the connectivity.
        
        .. code-block:: python
            :linenos:
            
            import numpy
            from pysdic import Connectivity
            
            connectivity = Connectivity.from_array(numpy.random.randint(0, 100, size=(50, 3)))
            intensity = numpy.random.rand(50)
            connectivity.set_property('intensity', intensity)
            
        """
        if not isinstance(key, str):
            raise TypeError("Property key must be a string.")
        values = numpy.asarray(values, dtype=numpy.float64).copy()
        if values.ndim == 1:
            values = values.reshape((-1, 1))
        if not (values.ndim == 2 and values.shape[0] == self.n_elements):
            raise ValueError(f"Property values must be a 2D NumPy array with shape (N_e, A) where N_e={self.n_elements}.")
        self._properties[key] = values
        
    def __setitem__(self, key: str, values: ArrayLike) -> None:
        self.set_property(key, values)
        
    
    def get_property(
        self,
        key: str
    ) -> numpy.ndarray:
        r"""
        Get a property of the connectivity by its key/name.
        
        .. note::
        
            The returned array is a unwritable view to prevent accidental modifications and is of type :obj:`numpy.float64`.
            
        .. note::
        
            The property can also be accessed using dictionary-like syntax:
            
            .. code-block:: python
                :linenos:
                
                property_values = connectivity['property_key']
                

        Parameters
        ----------
        key : :class:`str`
            The key/name of the property to retrieve.


        Returns
        -------
        :class:`numpy.ndarray`
            A NumPy array of shape :math:`(N_p, A)` representing the property values for each element in the connectivity.


        Raises
        ------
        KeyError
            If the property with the given key does not exist.
            

        Examples
        --------
        Get a property of the connectivity.
        
        .. code-block:: python
            :linenos:
        
            import numpy
            from pysdic import Connectivity
            
            connectivity = Connectivity.from_array(numpy.random.rand(100, 3))
            intensity = numpy.random.rand(100)
            connectivity.set_property('intensity', intensity)
            retrieved_intensity = connectivity.get_property('intensity')

        """
        if key not in self._properties:
            raise KeyError(f"Property '{key}' does not exist in the connectivity.")
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
        Delete a property of the connectivity by its key/name.


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
            raise KeyError(f"Property '{key}' does not exist in the connectivity.")
        del self._properties[key]
        
    
    def has_property(
        self,
        key: str
    ) -> bool:
        r"""
        Check if the connectivity has a property with the given key/name.
        
        .. note::
        
            The property can also be checked using dictionary-like syntax:
            
            .. code-block:: python
                :linenos:
                
                has_property = 'property_key' in connectivity


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
        List all property keys/names associated with the connectivity.


        Returns
        -------
        List[:class:`str`]
            A list of strings representing the keys/names of all properties in the connectivity.

        """
        return list(self._properties.keys())
    

    def copy_properties(self) -> Dict[str, numpy.ndarray]:
        r"""
        Create a copy of all properties of the connectivity.

        Returns
        -------
        Dict[:class:`str`, :class:`numpy.ndarray`]
            A dictionary containing copies of all properties in the connectivity.

        """
        properties_copy = {}
        for key, values in self._properties.items():
            properties_copy[key] = values.copy()
        return properties_copy
    
    
    def clear_properties(self) -> None:
        r"""
        Remove all properties from the connectivity.

        """
        self._properties.clear()
    
    
    # ==========================
    # Class methods
    # ==========================
    @classmethod
    def from_array(cls, elements: ArrayLike, copy: bool = False, properties: Optional[Dict[str, ArrayLike]] = None, element_type: Optional[str] = None) -> Connectivity:
        r"""
        Create a :class:`Connectivity` object from a NumPy array of shape :math:`(N_e, N_vpe)`.

        .. seealso::

            - :meth:`to_array` method for converting the connectivity back to a NumPy array.


        Parameters
        ----------
        elements: ArrayLike
            A NumPy array of shape :math:`(N_e, N_vpe)` representing the connectivity elements.
            
        copy: :class:`bool`, optional
            If :obj:`True`, a copy of the input array is made. Default is :obj:`False`.
            If the given object is not a type :obj:`numpy.int64` array, a copy will be made regardless of this parameter.

        properties: Optional[Dict[:class:`str`, ArrayLike]], optional
            A dictionary of additional properties to associate with the connectivity. Keys are property names, and values are NumPy arrays of property data of shape :math:`(N_e, A)`.

        element_type: Optional[:class:`str`], optional
            The expected type of elements in the connectivity. If provided, the connectivity will enforce that all elements are of the specified type.
            Default is :obj:`None`.
            

        Returns
        -------
        :class:`Connectivity`
            A :class:`Connectivity` object containing the provided elements.


        Raises
        ------
        TypeError
            If the input elements is not a NumPy array.
            If the properties is not a dictionary.
            If the element type is not a string.
            
        ValueError
            If the input array does not have the correct shape :math:`(N_e, N_vpe)`.
            If the properties arrays do not have the correct shape :math:`(N_e, A)`.
            If the element type is not one of the implemented types.
            

        Examples
        --------
        Creating a :class:`Connectivity` object from a random NumPy array.

        .. code-block:: python
            :linenos:

            import numpy as np
            from pysdic import Connectivity

            # Create a random connectivity with 100 elements
            random_elements = np.random.randint(0, 100, size=(100, 4))  # shape (100, 4)
            connectivity = Connectivity.from_array(random_elements)
            
        Now, ``connectivity`` is a :class:`Connectivity` object containing the elements from the NumPy array.

        """
        if not isinstance(copy, bool):
            raise ValueError("The 'copy' parameter must be a boolean.")
        
        elements = numpy.asarray(elements, dtype=numpy.int64)
        if elements.ndim == 1:
            elements = elements.reshape((-1, 1))
        if copy:
            elements = elements.copy()
        
        return cls(elements, properties=properties, element_type=element_type)
    

    def to_array(self) -> numpy.ndarray:
        r"""
        Convert the connectivity to a NumPy array of shape :math:`(N_e, N_vpe)`.

        .. note::

            The returned array is a copy of the internal elements array. Modifying it will not affect the original connectivity.

        .. seealso::

            - :meth:`elements` property for accessing and modifying the elements of the connectivity.
            - :meth:`from_array` class method for creating a Connectivity object from a NumPy array.
            

        Returns
        -------
        :class:`numpy.ndarray`
            A NumPy array of shape :math:`(N_e, N_vpe)` containing the elements of the connectivity.

            
        Examples
        --------
        Creating a :class:`Connectivity` object from a random NumPy array.

        .. code-block:: python
            :linenos:
            
            import numpy as np
            from pysdic import Connectivity

            # Create a random connectivity with 100 elements
            random_elements = np.random.randint(0, 100, size=(100, 4))  # shape (100, 4)
            connectivity = Connectivity.from_array(random_elements)
            
        Convert back to a NumPy array using the `to_array` method.

        .. code-block:: python
            :linenos:

            # Convert the connectivity back to a NumPy array
            elements_array = connectivity.to_array()
            print(elements_array)
            # Output: A NumPy array of shape (100, 4) containing the elements of the connectivity.
        
        """
        return self._elements.copy()
    
    @classmethod
    def from_npz(cls, filepath: str, load_properties: bool = True) -> Connectivity:
        r"""
        Create a :class:`Connectivity` object from a NumPy NPZ file.

        The NPZ file should contain an array named 'elements' with shape :math:`(N_e, N_{vpe})` representing the elements of the connectivity.
        Additional properties can be stored as separate arrays in the NPZ file.

        .. seealso::

            - :meth:`to_npz` method for saving the connectivity to a NPZ file.

        Parameters
        ----------
        filepath: :class:`str`
            The path to the NPZ file.
            
        load_properties: :class:`bool`, optional
            If :obj:`True`, all arrays in the NPZ file with format ``"_properties/{property_name}"`` will be loaded as properties of the connectivity. Default is :obj:`True`.


        Returns
        -------
        :class:`Connectivity`
            A :class:`Connectivity` object containing the elements and properties read from the NPZ file.

        
        Examples
        --------
        Creating a :class:`Connectivity` object from a NPZ file.
        
        .. code-block:: python
            :linenos:
            
            from pysdic import Connectivity
            # Create a connectivity from a NPZ file
            connectivity = Connectivity.from_npz('path/to/connectivity.npz')
            
        Now, ``connectivity`` is a :class:`Connectivity` object containing the elements and properties read from the specified NPZ file.
        
        """
        path = os.path.expanduser(filepath)
        if not os.path.isfile(path) or not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        data = numpy.load(filepath)
        if 'elements' not in data:
            raise ValueError("NPZ file must contain an array named 'elements' with shape (N_e, N_{vpe}).")
        elements = data['elements']
        properties = {}
        if load_properties:
            for key in data.files:
                # key as "_properties/{property_name}" is reserved for properties
                if key.startswith("_properties/"):
                    property_name = key[len("_properties/"):]
                    properties[property_name] = data[key].copy()
        return cls(elements=elements, properties=properties)
    
    
    def to_npz(self, filepath: str, save_properties: bool = True) -> None:
        r"""
        Save the connectivity to a NumPy NPZ file.

        The elements will be saved as an array named 'elements' with shape :math:`(N_e, N_{vpe})`.
        Additional properties will be saved as separate arrays in the NPZ file with names in the format ``"_properties/{property_name}"``.

        .. seealso::

            - :meth:`from_npz` method for creating a :class:`Connectivity` object from a NPZ file.

        Parameters
        ----------
        filepath: :class:`str`
            The path to the output NPZ file.

        save_properties: :class:`bool`, optional
            If :obj:`True`, all properties of the connectivity will be saved as separate arrays in the NPZ file. Default is :obj:`True`.


        Examples
        --------
        Saving a :class:`Connectivity` object to a NPZ file.

        .. code-block:: python
            :linenos:
            
            from pysdic import PointCloud
            import numpy as np

            # Create a random connectivity with 100 elements
            random_elements = np.random.randint(0, 100, size=(100, 4))  # shape (100, 4)
            connectivity = Connectivity.from_array(random_elements)

            # Save the connectivity to a NPZ file
            connectivity.to_npz('path/to/output_connectivity.npz')
            
        This will save the elements and properties of the connectivity to the specified NPZ file.
        """
        path = os.path.expanduser(filepath)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        data_to_save = {'elements': self._elements.copy()}
        if save_properties:
            properties_data = {}
            for key, values in self._properties.items():
                properties_data[f"_properties/{key}"] = values.copy()
            data_to_save.update(properties_data)
            
        numpy.savez(path, **data_to_save)

 
    # ==========================
    # Methods Geometry Manipulation
    # ==========================
    def concatenate(self, other: Connectivity, inplace: bool = False) -> Connectivity:
        r"""
        Concatenate the current connectivity with another :class:`Connectivity` instance.

        This method combines the elements from both connectivities into a new :class:`Connectivity` object.
        
        .. note::
        
            The properties in common between the two connectivities are also concatenated. 
            If a property exists in only one of the connectivities, it will be filled with :obj:`numpy.nan` values for the points from the other connectivity.


        Parameters
        ----------
        other: :class:`Connectivity`
            Another :class:`Connectivity` instance to concatenate with the current connectivity.

        inplace: :class:`bool`, optional
            If :obj:`True`, modifies the current connectivity in place and returns itself. If :obj:`False`, returns a new :class:`Connectivity` instance (default is :obj:`False`).
            

        Returns
        -------
        :class:`Connectivity`
            A new :class:`Connectivity` object containing the concatenated elements from both connectivities or the modified current instance if :obj:`inplace` is True
            
            
        Raises
        ------
        ValueError
            If the input is not an instance of :class:`Connectivity`.
            If the two connectivities have different dimensions.


        Examples
        --------
        Creating two :class:`Connectivity` objects.

        .. code-block:: python
            :linenos:

            import numpy as np
            from pysdic import Connectivity

            # Create two random NumPy arrays of shape (100, 3)
            random_elements1 = np.random.randint(0, 100, size=(100, 4))  # shape (100, 4)
            random_elements2 = np.random.randint(0, 100, size=(50, 4))   # shape (50, 4)

            connectivity1 = Connectivity.from_array(random_elements1)
            connectivity2 = Connectivity.from_array(random_elements2)

        Concatenate the two connectivities using the :meth:`concatenate` method.

        .. code-block:: python
            :linenos:

            # Concatenate the two connectivities
            concatenated_connectivity = connectivity1.concatenate(connectivity2)

            print(concatenated_connectivity.elements)
            # Output: A NumPy array of shape (150, 4) containing the concatenated elements

        """
        # Check if other is a Connectivity instance
        if not isinstance(other, Connectivity):
            raise ValueError("Input must be an instance of Connectivity.")
        if self.n_vertices_per_element != other.n_vertices_per_element:
            raise ValueError("Connectivities must have the same number of vertices per element to concatenate.")
        if not isinstance(inplace, bool):
            raise ValueError("inplace must be a boolean value.")
        if self.element_type is not None and other.element_type is not None:
            if self.element_type != other.element_type:
                raise ValueError("Both connectivities have different element types. Cannot concatenate.")
        
        # Extract the number of elements
        n_elements_self = self.n_elements
        n_elements_other = other.n_elements
        
        # Concatenate elements
        concatenated_elements = numpy.vstack((self._elements.copy(), other._elements.copy()))
        
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
                if not values_self.shape[0] == n_elements_self or not values_other.shape[0] == n_elements_other:
                    raise ValueError(f"Property '{key}' must have shape (N_e, A) where N_e matches the number of elements in the respective connectivity.")
                if not values_self.shape[1] == values_other.shape[1]:
                    raise ValueError(f"Property '{key}' must have the same number of attributes (A) in both connectivities.")

            elif values_self is not None:
                if not values_self.ndim == 2:
                    raise ValueError(f"Property '{key}' must be a 2D NumPy array in the first connectivity.")
                if not values_self.shape[0] == n_elements_self:
                    raise ValueError(f"Property '{key}' must have shape (N_e, A) where N_e matches the number of elements in the first connectivity.")
                values_other = numpy.full((n_elements_other, values_self.shape[1]), numpy.nan)
               
            elif values_other is not None:
                if not values_other.ndim == 2:
                    raise ValueError(f"Property '{key}' must be a 2D NumPy array in the second connectivity.")
                if not values_other.shape[0] == n_elements_other:
                    raise ValueError(f"Property '{key}' must have shape (N_e, A) where N_e matches the number of elements in the second connectivity.")
                values_self = numpy.full((n_elements_self, values_other.shape[1]), numpy.nan)
                
            else:
                raise ValueError("This should never happen. Both property values are None when concatenating.")              
            
            # Concatenate property values
            concatenated_values = numpy.vstack((values_self.copy(), values_other.copy()))
            concatenated_properties[key] = concatenated_values   
        
        # Return new instance or modify in place
        if inplace:
            self._elements = concatenated_elements
            self._properties = concatenated_properties
            return self
        else:
            return self.__class__(elements=concatenated_elements, properties=concatenated_properties, element_type=self.element_type)


    def copy(self, copy_properties: bool = True) -> Connectivity:
        r"""
        Create a copy of the current :class:`Connectivity` instance.
        
        Parameters
        ----------
        copy_properties: :class:`bool`, optional
            If :obj:`True`, the properties of the connectivity will also be copied to the new instance. Default is :obj:`True`.


        Returns
        -------
        :class:`Connectivity`
            A new :class:`Connectivity` object containing the same elements as the current instance.


        Examples
        --------
        Creating a :class:`Connectivity` from a random NumPy array and making a copy.

        .. code-block:: python
            :linenos:

            import numpy as np
            from pysdic import Connectivity

            # Create a random connectivity with 100 elements
            random_elements = np.random.randint(0, 100, size=(100, 3))  # shape (100, 3)
            connectivity1 = Connectivity.from_array(random_elements)
            
            # Create a copy of the existing Connectivity object
            connectivity2 = connectivity1.copy()

        """
        if not isinstance(copy_properties, bool):
            raise ValueError("copy_properties must be a boolean value.")
        
        obj = self.__class__(elements=self._elements.copy(), properties={key: values.copy() for key, values in self._properties.items()} if copy_properties else {}, element_type=self.element_type)
                
        return obj
    
    
    def filter_elements(self, mask: ArrayLike, inplace: bool = False) -> Connectivity:
        r"""
        Filter elements in the connectivity based on a boolean mask.

        This method returns a new :class:`Connectivity` object containing only the elements where the mask is True.
        The mask should be a 1D boolean NumPy array with the same length as the number of elements in the connectivity.
        All the properties associated with the elements are also filtered accordingly.
        
        .. seealso::

            - :meth:`keep_elements` for keeping elements at specified indices.
            - :meth:`remove_elements` for removing elements at specified indices.


        Parameters
        ----------
        mask: ArrayLike
            A 1D boolean NumPy array of shape :math:`(N_e,)` where :math:`N_e` is the number of elements in the connectivity.
            Each element should be True to keep the corresponding element, or False to remove it.

        inplace: :class:`bool`, optional
            If :obj:`True`, modifies the current connectivity in place and returns itself. If :obj:`False`, returns a new :class:`Connectivity` instance (default is :obj:`False`).


        Returns
        -------
        :class:`Connectivity`
            A new :class:`Connectivity` object containing only the filtered elements or the modified current instance if :obj:`inplace` is True.
            

        Raises
        ------
        ValueError
            If the input mask is not a 1D boolean array of the correct length.


        Examples
        --------
        Create a :class:`Connectivity` from a random NumPy array.

        .. code-block:: python
            :linenos:

            import numpy as np
            from pysdic import Connectivity

            # Create a random connectivity with 100 elements
            random_elements = np.random.randint(0, 100, size=(100, 3))  # shape (100, 3)
            connectivity = Connectivity.from_array(random_elements)
            
        Filtering elements with any vertex index greater than 50.

        .. code-block:: python
            :linenos:

            # Create a boolean mask for elements with any vertex index greater than 50
            mask = (connectivity.elements > 50).any(axis=1)

            # Filter the connectivity using the mask
            filtered_connectivity = connectivity.filter_elements(mask)
            print(filtered_connectivity.elements)
            # Output: A NumPy array containing only the elements with any vertex index greater than 50

        """
        if not isinstance(inplace, bool):
            raise ValueError("inplace must be a boolean value.")
        
        # Load and validate mask
        mask = numpy.asarray(mask, dtype=bool)
        if mask.ndim != 1:
            raise ValueError("Mask must be a 1D array.")
        if mask.shape[0] != self.n_elements:
            raise ValueError(f"Mask length must be equal to the number of elements N_e={self.n_elements}.")
        
        # Return new instance or modify in place
        if inplace:
            self._elements = self._elements[mask]
            for key in self._properties:
                self._properties[key] = self._properties[key][mask]
            return self
        else:
            return self.__class__(elements=self._elements[mask].copy(), properties={key: values[mask].copy() for key, values in self._properties.items()}, element_type=self.element_type)
        

    def keep_elements(self, indices: ArrayLike, inplace: bool = False) -> Connectivity:
        r"""
        Keep only the elements at the specified indices in the connectivity.

        This method returns a new :class:`Connectivity` object containing only the elements at the specified indices.
        The properties associated with the elements are also filtered accordingly.

        .. seealso::

            - :meth:`remove_elements` for removing elements at specified indices.
            - :meth:`filter_elements` for filtering elements based on a boolean mask.


        Parameters
        ----------
        indices: ArrayLike
            A 1D NumPy array of integer indices representing the elements to be kept in the connectivity.

        inplace: :class:`bool`, optional
            If :obj:`True`, modifies the current connectivity in place and returns itself. If :obj:`False`, returns a new Connectivity instance (default is :obj:`False`).
            

        Returns
        -------
        :class:`Connectivity`
            A new :class:`Connectivity` object containing only the elements at the specified indices or the modified current instance if :obj:`inplace` is True.


        Raises
        ------
        ValueError
            If any index is out of bounds or if the input is not a 1D array of integers.


        Examples
        --------
        Create a :class:`Connectivity` from a random NumPy array.

        .. code-block:: python
            :linenos:

            import numpy as np
            from pysdic import Connectivity

            # Create a random connectivity with 100 elements
            random_elements = np.random.randint(0, 100, size=(100, 3))  # shape (100, 3)
            connectivity = Connectivity.from_array(random_elements)
            
        Keeping only the points at indices 0, 2, and 4.

        .. code-block:: python
            :linenos:

            # Keep only elements at indices 0, 2, and 4
            indices_to_keep = np.array([0, 2, 4])
            new_connectivity = connectivity.keep_elements(indices_to_keep)
            print(new_connectivity.elements)
            # Output: A NumPy array of shape (3, 3) containing only the elements at indices 0, 2, and 4

        """
        if not isinstance(inplace, bool):
            raise ValueError("inplace must be a boolean value.")
        
        # Load and validate indices
        indices = numpy.asarray(indices, dtype=numpy.int64)
        if indices.ndim != 1:
            raise ValueError("Indices must be a 1D array.")
        if numpy.any(indices < 0) or numpy.any(indices >= self.n_elements):
            raise ValueError("Indices are out of bounds.")
        
        # Create mask for points to keep
        mask = numpy.zeros(self.n_elements, dtype=bool)
        mask[indices] = True
        
        # Apply mask
        return self.filter_elements(mask, inplace=inplace)
    
        
    def remove_elements(self, indices: ArrayLike, inplace: bool = False) -> Connectivity:
        r"""
        Remove elements from the connectivity based on their indices.

        This method returns a new :class:`Connectivity` object with the elements at the specified indices removed.
        The properties associated with the elements are also updated accordingly.

        .. seealso::

            - :meth:`keep_elements` for keeping elements at specified indices.
            - :meth:`filter_elements` for filtering elements based on a boolean mask.


        Parameters
        ----------
        indices: ArrayLike
            A 1D NumPy array of integer indices representing the elements to be removed from the connectivity.

        inplace: :class:`bool`, optional
            If :obj:`True`, modifies the current connectivity in place and returns itself. If :obj:`False`, returns a new :class:`Connectivity` instance (default is :obj:`False`).
            

        Returns
        -------
        :class:`Connectivity`
            A new :class:`Connectivity` object with the elements at the specified indices removed or the modified current instance if `inplace` is True.


        Raises
        ------
        ValueError
            If any index is out of bounds or if the input is not a 1D array of integers.

            
        Examples
        --------
        Create a :class:`Connectivity` from a random NumPy array.

        .. code-block:: python
            :linenos:

            import numpy as np
            from pysdic import Connectivity

            # Create a random connectivity with 100 elements
            random_elements = np.random.randint(0, 100, size=(100, 3))  # shape (100, 3)
            connectivity = Connectivity.from_array(random_elements)
            
        Removing elements at indices 1 and 3.

        .. code-block:: python
            :linenos:

            # Remove elements at indices 1 and 3
            indices_to_remove = np.array([1, 3])
            new_connectivity = connectivity.remove_elements(indices_to_remove)
            print(new_connectivity.elements)
            # Output: A NumPy array of shape (98, 3) with elements at indices 1 and 3 removed

        """
        if not isinstance(inplace, bool):
            raise ValueError("inplace must be a boolean value.")
        
        # Load and validate indices
        indices = numpy.asarray(indices, dtype=numpy.int64)
        if indices.ndim != 1:
            raise ValueError("Indices must be a 1D array.")
        if numpy.any(indices < 0) or numpy.any(indices >= self.n_elements):
            raise ValueError("Indices are out of bounds.")
       
        # Create mask for elements to remove  
        mask = numpy.ones(self.n_elements, dtype=bool)
        mask[indices] = False
        
        # Apply mask
        return self.filter_elements(mask, inplace=inplace)