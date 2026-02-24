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

from typing import Optional, Tuple, Union, Dict, List
from numbers import Integral, Real
from numpy.typing import ArrayLike

from py3dframe import Frame, FrameTransform

import numpy
import pyvista
import meshio
import os
import matplotlib


class PointCloud(object):
    r"""
    A class representing a :math:`E`-dimensional point cloud, which is a collection of :math:`N_p` points in a :math:`E`-dimensional space.

    Point clouds can store any scalar/vector properties associated with each point in the cloud.

    .. note::

        The dimension :math:`E` of the cloud is not designed to change after point cloud creation.
        
    .. note::
    
        The coordinates and the properties of the points are stored as copy-on-write (cow) NumPy arrays of type :obj:`numpy.float64`. 
        Accesing these arrays directly will return unwritable views to prevent accidental modifications.


    Parameters
    ----------
    points: ArrayLike
        A NumPy array of shape :math:`(N_{p}, E)` representing the coordinates of the points.
        

    n_dimensions: Integral, optional
        The dimension :math:`E` of the point cloud. If not provided, it is inferred from the shape of the :obj:`points` array.
        If :obj:`points` second dimension is less than :obj:`n_dimensions`, the points will be reshaped accordingly and filled with zeros.
        If :obj:`points` second dimension is greater than :obj:`n_dimensions`, a ValueError will be raised.
        Default is :obj:`None`.
        
    properties: Optional[Dict[:class:`str`, ArrayLike]], optional
        A dictionary of additional properties to associate with the point cloud. Keys are property names, and values are NumPy arrays of property data of shape :math:`(N_{p}, A)` where :math:`A` is the number of attributes per point for the given property.
        Default is :obj:`None`, meaning no additional properties are set.
        

    Raises  
    ------
    TypeError
        If the input points is not a NumPy array.
        If the given dimension is not an integer.
        If the properties is not a dictionary of string keys and NumPy array values.
        
    ValueError
        If the input array does not have the correct shape.
        If the given dimension is not a strictly positive integer.
        If the properties arrays do not have the correct shape.

    """
    __slots__ = [
        "_points",
        "_properties",
    ]

    def __init__(
        self, 
        points: ArrayLike, 
        n_dimensions: Optional[Integral] = None, 
        properties: Optional[Dict[str, ArrayLike]] = None
    ) -> None:
        # Convert points to 2D NumPy array
        points = numpy.asarray(points, dtype=numpy.float64).copy()
        if points.ndim == 1:
            points = points.reshape((-1, 1))
            
        if not points.ndim == 2:
            raise ValueError("Points must be a 2D NumPy array with shape (N_p, E).")
        if n_dimensions is None:
            n_dimensions = points.shape[1]
        if not isinstance(n_dimensions, Integral) or n_dimensions <= 0:
            raise ValueError("n_dimensions must be a strictly positive integer.")
        
        if points.shape[1] < n_dimensions:
            padded_points = numpy.zeros((points.shape[0], n_dimensions), dtype=points.dtype)
            padded_points[:, :points.shape[1]] = points
            points = padded_points
        elif points.shape[1] > n_dimensions:
            raise ValueError(f"Points have more dimensions ({points.shape[1]}) than n_dimensions ({n_dimensions}).")
        
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
            if not (values.ndim == 2 and values.shape[0] == points.shape[0]):
                raise ValueError(f"Property values must be a 2D NumPy array with shape (N_p, A) where N_p={points.shape[0]}.")
           
        # Save points
        self._points = points
        self._properties = {}
        
        # Save properties
        for key, values in properties.items():
            self.set_property(key, values)

                
        


    # ==========================
    # Properties
    # ==========================
    @property
    def points(self) -> numpy.ndarray:
        r"""
        [Get or Set] An numpy array of shape :math:`(N_p, E)` representing the coordinates of the points in the cloud.
        
        .. note::

            An alias for ``points`` is ``coordinates``.
            
        .. note::
        
            The property is copied-on-write (cow) into :obj:`numpy.ndarray` of type :obj:`numpy.float64`. Accessing it will return an unwritable view to prevent accidental modifications.
            

        Parameters
        ----------
        value: ArrayLike
            An array-like of shape :math:`(N_p, E)` representing the coordinates of the points in the cloud. If 1D array is provided, it will be reshaped to :math:`(N_p, 1)`.


        Returns
        -------
        :class:`numpy.ndarray`
            A unwritable NumPy array of shape :math:`(N_p, E)` containing the coordinates of the points in the cloud.


        Raises
        ------
        TypeError
            If the input points is not a NumPy array.
        
        ValueError
            If the input array does not have the correct shape :math:`(N_p, E)`.
        """
        view = self._points.view()
        view.flags.writeable = False
        return view
    
    @points.setter
    def points(self, value: ArrayLike) -> None:
        points = numpy.asarray(value, dtype=numpy.float64).copy()
        if self.n_dimensions == 1 and points.ndim == 1:
            points = points.reshape((-1, 1))
        if not (points.ndim == 2 and points.shape[1] == self.n_dimensions):
            raise ValueError(f"Points must be a 2D NumPy array with shape (N_p, {self.n_dimensions}).")
        if not points.shape[0] == self.n_points:
            raise ValueError(f"Points must have the same number of points N_p={self.n_points}. Consider using remove/add point methods to change the number of points.")
        self._points = points

    @property
    def coordinates(self) -> numpy.ndarray:
        r"""
        [Get or Set] Alias for :meth:`points` property.
        """
        return self.points

    @coordinates.setter
    def coordinates(self, value: ArrayLike) -> None:
        self.points = value


    @property
    def n_points(self) -> int:
        r"""
        [Get] The number of points :math:`N_p` in the point cloud.

        .. note::

            You can also use `len(point_cloud)`.

        .. seealso::

            - :meth:`shape` for getting the shape of the points array.
            - :meth:`n_dimensions` for getting the dimension of the point cloud.
            

        Returns
        -------
        :class:`int`
            The number of points in the point cloud.

        """
        return self._points.shape[0]
    
    def __len__(self) -> int:
        return self.n_points
    
    
    @property
    def shape(self) -> Tuple[int, int]:
        r"""
        [Get] The shape of the points array (:math:`N_p`, :math:`E`).

        .. seealso::

            - :meth:`n_points` for getting the number of points in the point cloud.
            - :meth:`n_dimensions` for getting the dimension of the point cloud.


        Returns
        -------
        Tuple[:class:`int`, :class:`int`]
            A tuple representing the shape of the points array (:math:`N_p`, :math:`E`).

        """
        return self._points.shape
    
    
    @property
    def n_dimensions(self) -> int:
        r"""
        [Get] The dimension :math:`E` of the point cloud.

        .. seealso::

            - :meth:`n_points` for getting the number of points in the point cloud.
            - :meth:`shape` for getting the shape of the points array.

        Returns
        -------
        :class:`int`
            The dimension of the point cloud.
            
        """
        return self._points.shape[1]
    
    
    def __repr__(self) -> str:
        string = f"PointCloud(n_points={self.n_points}, n_dimensions={self.n_dimensions})"
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
        Set a property for the point cloud as a NumPy array of shape :math:`(N_p, A)` where :math:`A` is the number of attributes per point for the given property.

        .. note::
        
            The given array will be copied and stored as type :obj:`numpy.float64`.
            
        .. note::
        
            The property can also be set using dictionary-like syntax:
            
            .. code-block:: python
                :linenos:
                
                point_cloud['property_key'] = property_values
                    

        Parameters
        ----------
        key: :class:`str`
            The key/name of the property to set.
        
        values: ArrayLike
            A NumPy array of shape :math:`(N_p, A)` or :math:`(N_p,)` representing the property values for each point in the cloud.
            If 1D array is provided, it will be reshaped to :math:`(N_p, 1)`.


        Raises
        ------
        TypeError
            If the property key is not a string.
            If the input values is not a NumPy array.
            
        ValueError
            If the input array does not have the correct shape :math:`(N_p, A)` or :math:`(N_p,)`.
            

        Examples
        --------
        Set a property for the point cloud.
        
        .. code-block:: python
            :linenos:
            
            import numpy
            from pysdic import PointCloud
            
            pc = PointCloud.from_array(numpy.random.rand(100, 3))
            intensity = numpy.random.rand(100)
            pc.set_property('intensity', intensity)

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
        Get a property of the point cloud by its key/name.
        
        .. note::
        
            The returned array is a unwritable view to prevent accidental modifications and is of type :obj:`numpy.float64`.
            
        .. note::
        
            The property can also be accessed using dictionary-like syntax:
            
            .. code-block:: python
                :linenos:
                
                property_values = point_cloud['property_key']
                

        Parameters
        ----------
        key: :class:`str`
            The key/name of the property to retrieve.


        Returns
        -------
        :class:`numpy.ndarray`
            A NumPy array of shape :math:`(N_p, A)` representing the property values for each point in the cloud.


        Raises
        ------
        KeyError
            If the property with the given key does not exist.
            

        Examples
        --------
        Get a property of the point cloud.
        
        .. code-block:: python
            :linenos:
            
            import numpy
            from pysdic import PointCloud
            
            pc = PointCloud.from_array(numpy.random.rand(100, 3))
            intensity = numpy.random.rand(100)
            pc.set_property('intensity', intensity)
            retrieved_intensity = pc.get_property('intensity')

        """
        if key not in self._properties:
            raise KeyError(f"Property '{key}' does not exist in the point cloud.")
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
        Delete a property of the point cloud by its key/name.


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
            raise KeyError(f"Property '{key}' does not exist in the point cloud.")
        del self._properties[key]
        
    
    def has_property(
        self,
        key: str
    ) -> bool:
        r"""
        Check if the point cloud has a property with the given key/name.
        
        .. note::
        
            The property can also be checked using dictionary-like syntax:
            
            .. code-block:: python
                :linenos:
                
                has_property = 'property_key' in point_cloud


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
        List all property keys/names associated with the point cloud.


        Returns
        -------
        List[:class:`str`]
            A list of strings representing the keys/names of all properties in the point cloud.

        """
        return list(self._properties.keys())
    
    
    def copy_properties(self) -> Dict[str, numpy.ndarray]:
        r"""
        Return a copy of all properties of the point cloud.

        Returns
        -------
        dict[:class:`str`, :class:`numpy.ndarray`]
            A dictionary containing copies of all properties in the point cloud.

        """
        properties_copy = {}
        for key, values in self._properties.items():
            properties_copy[key] = values.copy()
        return properties_copy
    
    
    def clear_properties(self) -> None:
        r"""
        Remove all properties from the point cloud.
        """
        self._properties.clear()
    
    # ==========================
    # Class methods
    # ==========================
    @classmethod
    def from_array(cls, points: ArrayLike, copy: bool = False, properties: Optional[Dict[str, ArrayLike]] = None) -> PointCloud:
        r"""
        Create a :class:`PointCloud` object from a NumPy array of shape :math:`(N_p, E)`.

        .. seealso::

            - :meth:`to_array` method for converting the point cloud back to a NumPy array.


        Parameters
        ----------
        points: ArrayLike
            A NumPy array of shape :math:`(N_p, E)` representing the coordinates of the points.
            
        copy: :class:`bool`, optional
            If :obj:`True`, a copy of the input array is made. Default is :obj:`False`.
            If the given object is not a type :obj:`numpy.float64` array, a copy will be made regardless of this parameter.

        properties: Optional[Dict[:class:`str`, ArrayLike]], optional
            A dictionary of additional properties to associate with the point cloud. Keys are property names, and values are NumPy arrays of property data of shape :math:`(N_p, A)`


        Returns
        -------
        :class:`PointCloud`
            A :class:`PointCloud` object containing the provided points.


        Raises
        ------
        TypeError
            If the input points is not a NumPy array.
            If the properties is not a dictionary.
            
        ValueError
            If the input array does not have the correct shape :math:`(N_p, E)`.
            If the properties arrays do not have the correct shape :math:`(N_p, A)`.


        Examples
        --------
        Creating a :class:`PointCloud` object from a random NumPy array.

        .. code-block:: python
            :linenos:
            
            import numpy as np
            from pysdic import PointCloud

            # Create a random point cloud with 100 points
            random_points = np.random.rand(100, 3)  # shape (100, 3)
            point_cloud = PointCloud.from_array(random_points)

        Now, ``point_cloud`` is a :class:`PointCloud` object containing the points from the NumPy array.

        """
        if not isinstance(copy, bool):
            raise ValueError("The 'copy' parameter must be a boolean.")
        
        points = numpy.asarray(points, dtype=numpy.float64)
        if points.ndim == 1:
            points = points.reshape((-1, 1))
        if copy:
            points = points.copy()
        
        return cls(points, properties=properties)
    

    def to_array(self) -> numpy.ndarray:
        r"""
        Convert the point cloud to a NumPy array of shape :math:`(N_p, E)`.

        .. note::

            The returned array is a copy of the internal points array. Modifying it will not affect the original point cloud.

        .. seealso::

            - :meth:`points` property for accessing and modifying the points of the point cloud.
            - :meth:`from_array` class method for creating a PointCloud object from a NumPy array.
            

        Returns
        -------
        :class:`numpy.ndarray`
            A NumPy array of shape :math:`(N_p, E)` containing the coordinates of the points in the cloud.

            
        Examples
        --------
        Creating a :class:`PointCloud` object from a random NumPy array.

        .. code-block:: python
            :linenos:
            
            import numpy as np
            from pysdic import PointCloud

            # Create a random point cloud with 100 points
            random_points = np.random.rand(100, 3)  # shape (100, 3)
            point_cloud = PointCloud.from_array(random_points)

        Convert back to a NumPy array using the `to_array` method.

        .. code-block:: python
            :linenos:
            
            # Convert the point cloud back to a NumPy array
            points_array = point_cloud.to_array()
            print(points_array)
            # Output: A NumPy array of shape (100, 3) containing the coordinates
        
        """
        return self._points.copy()
 

    @classmethod
    def from_meshio(cls, mesh: meshio.Mesh, load_properties: bool = True) -> PointCloud:
        r"""
        Create a :class:`PointCloud` object from a :class:`meshio.Mesh` object.

        The points are extracted from the mesh vertices and the properties are read from the point data.

        .. seealso::

            - :meth:`to_array` method for converting the point cloud back to a NumPy array.

        .. note::

            Only the vertices of the mesh are used to create the point cloud. Cells are ignored.


        Parameters
        ----------
        mesh: :class:`meshio.Mesh`
            A meshio Mesh object containing the vertices.
            
        load_properties: :class:`bool`, optional
            If :obj:`True`, the point data properties from the mesh will be loaded into the point cloud. Default is :obj:`True`.
            

        Returns
        -------
        :class:`PointCloud`
            A :class:`PointCloud` object containing the points extracted from the mesh vertices.

        Raises
        ------
        ValueError
            If the mesh does not contain any points.


        Examples
        --------
        Creating a :class:`PointCloud` object from a :class:`meshio.Mesh` object.
    
        .. code-block:: python
            :linenos:
            
            import meshio
            from pysdic import PointCloud

            # Load a mesh using meshio
            mesh = meshio.read('path/to/mesh_file.vtk')

            # Create a point cloud from the mesh
            point_cloud = PointCloud.from_meshio(mesh)

        Now, ``point_cloud`` is a :class:`PointCloud` object containing the points extracted from the mesh vertices.

        """
        if not isinstance(mesh, meshio.Mesh):
            raise ValueError("Input must be an instance of meshio.Mesh.")
        if not isinstance(load_properties, bool):
            raise ValueError("The 'load_properties' parameter must be a boolean.")
        
        # Extract points from mesh vertices
        if mesh.points is None:
            raise ValueError("The mesh does not contain any points.")
        vertices = numpy.asarray(mesh.points, dtype=numpy.float64)
        if vertices.shape[0] == 0:
            raise ValueError("The mesh does not contain any points.")
        
        # Extract properties from point data       
        properties = {}
        if load_properties and mesh.point_data is not None:
            for key, values in mesh.point_data.items():
                properties[key] = numpy.asarray(values, dtype=numpy.float64).copy()
                
        return cls(points=vertices, properties=properties)


    def to_meshio(self, save_properties: bool = True) -> meshio.Mesh:
        r"""
        Convert the point cloud to a :class:`meshio.Mesh` object.

        The points are stored as vertices in the mesh and the properties are saved as point data.

        .. seealso::

            - :meth:`from_meshio` class method for creating a :class:`PointCloud` object from a :class:`meshio.Mesh` object.
            
            
        Parameters
        ----------
        save_properties: :class:`bool`, optional
            If :obj:`True`, the point cloud properties will be saved as point data in the mesh. Default is :obj:`True`.
            

        Returns
        -------
        :class:`meshio.Mesh`
            A :class:`meshio.Mesh` object containing the points as vertices.


        Examples
        --------
        Converting a :class:`PointCloud` object to a :class:`meshio.Mesh` object.
    
        .. code-block:: python
            :linenos:
            
            import numpy as np
            from pysdic import PointCloud

            # Create a random point cloud with 100 points
            random_points = np.random.rand(100, 3)  # shape (100, 3)
            point_cloud = PointCloud.from_array(random_points)

            # Convert the point cloud to a meshio Mesh
            mesh = point_cloud.to_meshio()

        Now, ``mesh`` is a :class:`meshio.Mesh` object containing the points of the point cloud as vertices.
        """
        if not isinstance(save_properties, bool):
            raise ValueError("The 'save_properties' parameter must be a boolean.")
        
        point_data = {}
        
        if save_properties and self._properties:
            point_data = self.copy_properties()
                
        return meshio.Mesh(points=self._points.copy(), cells=[], point_data=point_data)


    @classmethod
    def from_vtk(cls, filepath: str, load_properties: bool = True) -> PointCloud:
        r"""
        Create a :class:`PointCloud` object from a VTK file (only for 3D point clouds).

        The VTK file should contain vertex definitions.
        Only vertices are read; faces and other elements are ignored.

        The points are extracted using ``meshio`` library.

        .. seealso::

            - :meth:`to_vtk` method for saving the point cloud to a VTK file.

        .. warning::

            This method only works for 3-dimensional point clouds, as VTK format is primarily used for 3D data.
            

        Parameters
        ----------
        filepath: :class:`str`
            The path to the VTK file.
            
        load_properties: :class:`bool`, optional
            If :obj:`True`, the point data properties from the VTK file will be loaded into the point cloud. Default is :obj:`True`.


        Returns
        -------
        :class:`PointCloud`
            A :class:`PointCloud` object containing the points read from the VTK file.

        
        Examples
        --------
        Creating a :class:`PointCloud` object from a VTK file.
        
        .. code-block:: python
            :linenos:
            
            from pysdic import PointCloud
            # Create a point cloud from a VTK file
            point_cloud = PointCloud.from_vtk('path/to/point_cloud.vtk')

        Now, ``point_cloud`` is a :class:`PointCloud` object containing the points read from the specified VTK file.

        """
        path = os.path.expanduser(filepath)

        if not os.path.isfile(path) or not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        mesh = meshio.read(filepath, file_format='vtk')
        mesh = cls.from_meshio(mesh, load_properties=load_properties)
        if mesh.n_dimensions != 3:
            raise ValueError("VTK format only supports 3D point clouds.")
        return mesh


    def to_vtk(self, filepath: str, binary: bool = False, only_finite: bool = False, save_properties: bool = True) -> None:
        r"""
        Save the point cloud to a VTK file (only for :math:`E=3` point clouds).

        The points are saved using ``meshio`` library.

        The VTK file will contain vertex definitions.

        .. seealso::

            - :meth:`from_vtk` method for creating a :class:`PointCloud` object from a VTK file.

        .. warning::

            VTK cannot handle NaN or infinite values. If the point cloud contains such values, consider using the ``only_finite`` parameter to filter them out before saving.

        .. warning::

            This method only works for 3-dimensional point clouds, as VTK format is primarily used for 3D data.

        Parameters
        ----------
        filepath: :class:`str`
            The path to the output VTK file.

        binary: :class:`bool`, optional
            If :obj:`True`, the VTK file will be saved in binary format. Default is :obj:`False`.

        only_finite: :class:`bool`, optional
            If :obj:`True`, only finite points (excluding NaN and infinite values) will be saved. Default is :obj:`False`.

        save_properties: :class:`bool`, optional
            If :obj:`True`, the point data properties from the point cloud will be saved to the VTK file. Default is :obj:`True`.


        Raises
        ------
        ValueError
            If the :obj:`binary` parameter is not a boolean value.
            If the :obj:`only_finite` parameter is not a boolean value.
            If :obj:`only_finite` is True and there are no finite points to save.

        
        Examples
        --------
        Saving a :class:`PointCloud` object to a VTK file.

        .. code-block:: python
            :linenos:
            
            from pysdic import PointCloud
            import numpy as np

            # Create a random point cloud with 100 points
            random_points = np.random.rand(100, 3)  # shape (100, 3)
            point_cloud = PointCloud.from_array(random_points)

            # Save the point cloud to a VTK file
            point_cloud.to_vtk('path/to/output_point_cloud.vtk')

        This will save the points of the point cloud to the specified VTK file.
        """
        path = os.path.expanduser(filepath)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if not isinstance(binary, bool):
            raise ValueError("The 'binary' parameter must be a boolean value.")
        if not isinstance(only_finite, bool):
            raise ValueError("The 'only_finite' parameter must be a boolean value.")
        
        if self.n_dimensions != 3:
            raise ValueError("VTK format only supports 3D point clouds.")

        # Set a temporary path to force pyvista to save as VTK
        if only_finite:
            finite_mask = numpy.isfinite(self._points).all(axis=1)
            points_to_save = self._points[finite_mask].copy()
            if points_to_save.shape[0] == 0:
                raise ValueError("No finite points to save in the point cloud.")

        else:
            points_to_save = self._points.copy()
            if not numpy.isfinite(points_to_save).all():
                raise ValueError("Point cloud contains NaN or infinite values. Consider using 'only_finite=True' to filter them out before saving.")

        # Load point properties if needed
        if not save_properties:
            properties = None
        else:
            properties = {}
            for key, values in self._properties.items():
                if only_finite:
                    properties[key] = values[finite_mask].copy()
                else:
                    properties[key] = values.copy()
                    
            
        mesh = meshio.Mesh(points=points_to_save, cells=[("vertex", numpy.arange(points_to_save.shape[0]).reshape((-1, 1)))], point_data=properties)
        meshio.write(filepath, mesh, file_format='vtk', binary=binary)


    @classmethod
    def from_npz(cls, filepath: str, load_properties: bool = True) -> PointCloud:
        r"""
        Create a :class:`PointCloud` object from a NumPy NPZ file.

        The NPZ file should contain an array named 'points' with shape :math:`(N_p, E)` representing the coordinates of the points.
        Additional properties can be stored as separate arrays in the NPZ file.

        .. seealso::

            - :meth:`to_npz` method for saving the point cloud to a NPZ file.

        Parameters
        ----------
        filepath: :class:`str`
            The path to the NPZ file.
            
        load_properties: :class:`bool`, optional
            If :obj:`True`, all arrays in the NPZ file with format ``"_properties/{property_name}"`` will be loaded as properties of the point cloud. Default is :obj:`True`.


        Returns
        -------
        :class:`PointCloud`
            A :class:`PointCloud` object containing the points and properties read from the NPZ file.

        
        Examples
        --------
        Creating a :class:`PointCloud` object from a NPZ file.
        
        .. code-block:: python
            :linenos:
            
            from pysdic import PointCloud
            # Create a point cloud from a NPZ file
            point_cloud = PointCloud.from_npz('path/to/point_cloud.npz')
            
        Now, ``point_cloud`` is a :class:`PointCloud` object containing the points and properties read from the specified NPZ file.
        
        """
        path = os.path.expanduser(filepath)
        if not os.path.isfile(path) or not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        data = numpy.load(filepath)
        if 'points' not in data:
            raise ValueError("NPZ file must contain an array named 'points' with shape (N_p, E).")
        points = data['points']
        properties = {}
        if load_properties:
            for key in data.files:
                # key as "_properties/{property_name}" is reserved for properties
                if key.startswith("_properties/"):
                    property_name = key[len("_properties/"):]
                    properties[property_name] = data[key].copy()
        return cls(points=points, properties=properties)
    
    
    def to_npz(self, filepath: str, save_properties: bool = True) -> None:
        r"""
        Save the point cloud to a NumPy NPZ file.

        The points will be saved as an array named 'points' with shape :math:`(N_p, E)`.
        Additional properties will be saved as separate arrays in the NPZ file with names in the format ``"_properties/{property_name}"``.

        .. seealso::

            - :meth:`from_npz` method for creating a :class:`PointCloud` object from a NPZ file.

        Parameters
        ----------
        filepath: :class:`str`
            The path to the output NPZ file.

        save_properties: :class:`bool`, optional
            If :obj:`True`, all properties of the point cloud will be saved as separate arrays in the NPZ file. Default is :obj:`True`.


        Examples
        --------
        Saving a :class:`PointCloud` object to a NPZ file.

        .. code-block:: python
            :linenos:
            
            from pysdic import PointCloud
            import numpy as np

            # Create a random point cloud with 100 points
            random_points = np.random.rand(100, 3)  # shape (100, 3)
            point_cloud = PointCloud.from_array(random_points)

            # Save the point cloud to a NPZ file
            point_cloud.to_npz('path/to/output_point_cloud.npz')
            
        This will save the points and properties of the point cloud to the specified NPZ file.
        """
        path = os.path.expanduser(filepath)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        data_to_save = {'points': self._points.copy()}
        if save_properties:
            properties_data = {}
            for key, values in self._properties.items():
                properties_data[f"_properties/{key}"] = values.copy()
            data_to_save.update(properties_data)
            
        numpy.savez(path, **data_to_save)


    # ==========================
    # Methods Geometry Manipulation
    # ==========================
    def all_close(self, other: Union[PointCloud, ArrayLike], rtol: Real = 1e-05, atol: Real = 1e-08, equal_nan: bool = True, ordered: bool = True, ) -> bool:
        r"""
        Check if all points in the current point cloud are approximately equal to the points in another :class:`PointCloud` instance within a tolerance.

        This method compares the points of the current point cloud with those of another :class:`PointCloud` instance and returns True if all corresponding points are approximately equal within the specified relative and absolute tolerances.

        .. note::

            If the number of points in both point clouds differ, the method will return False.

        .. seealso::

            - :func:`numpy.allclose` for more details on the comparison.

        Parameters
        ----------
        other: Union[:class:`PointCloud`, ArrayLike]
            Another :class:`PointCloud` instance or a NumPy array with shape :math:`(N_p, 3)` to compare with the current point cloud.

        rtol: Real, optional
            The relative tolerance parameter (default is :obj:`1e-05`).

        atol: Real, optional
            The absolute tolerance parameter (default is :obj:`1e-08`).

        equal_nan: :class:`bool`, optional
            If True, NaN values are considered equal (default is :obj:`True`).

        ordered: :class:`bool`, optional
            If True, the points are compared in order. If False, the points are compared without considering the order (default is :obj:`True`).

            
        Returns
        -------
        bool
            True if all points are approximately equal within the specified tolerances, False otherwise.

            
        Raises
        ------
        ValueError
            If the input is not an instance of :class:`PointCloud`.

            
        Examples
        --------
        Creating a :class:`PointCloud` object from a random NumPy array.

        .. code-block:: python
            :linenos:
            
            import numpy as np
            from pysdic import PointCloud

            # Create a random point cloud with 100 points
            random_points = np.random.rand(100, 3)  # shape (100, 3)
            point_cloud1 = PointCloud.from_array(random_points)

        Compare the point cloud with another point cloud that is slightly modified.

        .. code-block:: python
            :linenos:
            
            # Create a second point cloud by adding small noise to the first one
            noise = np.random.normal(scale=1e-8, size=random_points.shape)
            point_cloud2 = PointCloud.from_array(random_points + noise)

            # Check if the two point clouds are approximately equal
            are_close = point_cloud1.all_close(point_cloud2, rtol=1e-5, atol=1e-8)
            print(are_close)
            # Output: True (most likely, depending on the noise)

        Compare with a point cloud that is significantly different.

        .. code-block:: python
            :linenos:
            
            # Create a third point cloud that is significantly different
            different_points = np.random.rand(100, 3) + 1.0  # Shifted by 1.0
            point_cloud3 = PointCloud.from_array(different_points)
            are_close = point_cloud1.all_close(point_cloud3, rtol=1e-5, atol=1e-8)
            print(are_close)
            # Output: False

        """
        if not isinstance(rtol, Real):
            raise ValueError("rtol must be a numeric value.")
        if not rtol >= 0:
            raise ValueError("rtol must be non-negative.")
        if not isinstance(atol, Real):
            raise ValueError("atol must be a numeric value.")
        if not atol >= 0:
            raise ValueError("atol must be non-negative.")
        if not isinstance(equal_nan, bool):
            raise ValueError("equal_nan must be a boolean value.")
        if not isinstance(ordered, bool):
            raise ValueError("ordered must be a boolean value.")

        if isinstance(other, PointCloud):
            other_points = other.points
        else:
            other_points = numpy.asarray(other, dtype=numpy.float64)
            if not other_points.ndim == 2:
                raise ValueError(f" PointCloud.all_close: Input must be a other PointCloud instance or a 2D NumPy array with shape (N, E), got shape {other_points.shape}.")

        if self.points.shape != other_points.shape:
            return False
        
        if ordered:
            return numpy.allclose(self.points, other_points, rtol=rtol, atol=atol, equal_nan=equal_nan)
        else:
            # argsort rows lexicographically
            self_sorted = self.points[numpy.lexsort(self.points.T[::-1])]
            other_sorted = other_points[numpy.lexsort(other_points.T[::-1])]

            # Compare the sorted views
            return numpy.allclose(self_sorted, other_sorted, rtol=rtol, atol=atol, equal_nan=equal_nan)
    

    def all_finite(self) -> bool:
        r"""
        Check if all points in the point cloud are finite (i.e., not NaN or infinite).

        .. seealso::
        
            - :meth:`is_finite` method for getting a boolean mask of finite points.
            - :meth:`is_nan` method for getting a boolean mask of NaN points.


        Returns
        -------
        bool
            True if all points are finite, False otherwise.


        Examples
        --------
        Creating a :class:`PointCloud` object and checking if all points are finite.

        .. code-block:: python
            :linenos:
            
            import numpy as np
            from pysdic import PointCloud

            # Create a point cloud with some finite and non-finite points
            points = np.array([[0.0, 1.0, 2.0],
                               [3.0, 4.0, 5.0],
                               [numpy.nan, 7.0, 8.0],
                               [9.0, numpy.inf, 11.0]])
            point_cloud = PointCloud.from_array(points)

            # Check if all points are finite
            all_finite = point_cloud.all_finite()
            print(all_finite)
            # Output: False

        """
        return numpy.isfinite(self.points).all()
    
    
    def is_finite(self) -> numpy.ndarray:
        r"""
        Check which points in the point cloud are finite (not NaN or infinite).

        This method returns a boolean mask indicating which points in the point cloud are finite.

        .. seealso::
        
            - :meth:`all_finite` method for checking if all points are finite.
            - :meth:`is_nan` method for getting a boolean mask of NaN points.
            

        Returns
        -------
        :class:`numpy.ndarray`
            A 1D boolean NumPy array of shape :math:`(N_p,)` where :math:`N_p` is the number of points in the point cloud.
            Each element is True if the corresponding point is finite, and False otherwise.


        Examples
        --------
        Create a :class:`PointCloud` from a NumPy array containing some NaN and infinite values.

        .. code-block:: python
            :linenos:
            
            import numpy as np
            from pysdic import PointCloud

            # Create a point cloud with some NaN and infinite values
            points = np.array([[0.0, 1.0, 2.0],
                               [np.nan, 1.0, 2.0],
                               [3.0, np.inf, 4.0],
                               [5.0, 6.0, 7.0]])
            point_cloud = PointCloud.from_array(points)

        Check which points are finite using the :meth:`is_finite` method.

        .. code-block:: python
            :linenos:
            
            finite_mask = point_cloud.is_finite()
            print(finite_mask)
            # Output: [ True False False  True]

        """
        return numpy.isfinite(self.points).all(axis=1)
    

    def is_nan(self) -> numpy.ndarray:
        r"""
        Check which points in the point cloud are NaN.

        This method returns a boolean mask indicating which points in the point cloud contain NaN values.
        
        .. seealso::
        
            - :meth:`all_finite` method for checking if all points are finite.
            - :meth:`is_finite` method for getting a boolean mask of finite points.
            

        Returns
        -------
        :class:`numpy.ndarray`
            A 1D boolean NumPy array of shape :math:`(N_p,)` where :math:`N_p` is the number of points in the point cloud.
            Each element is True if the corresponding point contains any NaN value, and False otherwise.

        Examples
        --------
        Create a :class:`PointCloud` from a NumPy array containing some NaN values.

        .. code-block:: python
            :linenos:
            
            import numpy as np
            from pysdic import PointCloud

            # Create a point cloud with some NaN values
            points = np.array([[0.0, 1.0, 2.0],
                               [np.nan, 1.0, 2.0],
                               [3.0, 4.0, 5.0],
                               [6.0, np.nan, 8.0]])
            point_cloud = PointCloud.from_array(points)

        Check which points are NaN using the :meth:`is_nan` method.

        .. code-block:: python
            :linenos:
            
            nan_mask = point_cloud.is_nan()
            print(nan_mask)
            # Output: [False  True False  True]

        """
        return numpy.isnan(self.points).any(axis=1)
    


    def concatenate(self, other: PointCloud, inplace: bool = False) -> PointCloud:
        r"""
        Concatenate the current point cloud with another :class:`PointCloud` instance.

        This method combines the points from both point clouds into a new :class:`PointCloud` object.
        
        .. note::
        
            The properties in common between the two point clouds are also concatenated. 
            If a property exists in only one of the point clouds, it will be filled with :obj:`numpy.nan` values for the points from the other point cloud.


        Parameters
        ----------
        other : :class:`PointCloud`
            Another :class:`PointCloud` instance to concatenate with the current point cloud.

        inplace : :class:`bool`, optional
            If :obj:`True`, modifies the current point cloud in place and returns itself. If :obj:`False`, returns a new :class:`PointCloud` instance (default is :obj:`False`).


        Returns
        -------
        :class:`PointCloud`
            A new :class:`PointCloud` object containing the concatenated points from both point clouds or the modified current instance if :obj:`inplace` is True
            
            
        Raises
        ------
        ValueError
            If the input is not an instance of :class:`PointCloud`.
            If the two point clouds have different dimensions.


        Examples
        --------
        Creating two :class:`PointCloud` objects.

        .. code-block:: python
            :linenos:
            
            import numpy as np
            from pysdic import PointCloud

            # Create two random NumPy arrays of shape (100, 3)
            random_points1 = np.random.rand(100, 3)  # shape (100, 3)
            random_points2 = np.random.rand(50, 3)   # shape (50, 3)

            point_cloud1 = PointCloud.from_array(random_points1)
            point_cloud2 = PointCloud.from_array(random_points2)

        Concatenate the two point clouds using the :meth:`concatenate` method.

        .. code-block:: python
            :linenos:

            # Concatenate the two point clouds
            concatenated_point_cloud = point_cloud1.concatenate(point_cloud2)

            print(concatenated_point_cloud.points)
            # Output: A NumPy array of shape (150, 3) containing the concatenated coordinates

        """
        # Check if other is a PointCloud instance
        if not isinstance(other, PointCloud):
            raise ValueError("Input must be an instance of PointCloud.")
        if self.n_dimensions != other.n_dimensions:
            raise ValueError("Point clouds must have the same dimension to concatenate.")
        if not isinstance(inplace, bool):
            raise ValueError("inplace must be a boolean value.")
        
        # Extract the number of points
        n_points_self = self.n_points
        n_points_other = other.n_points
        
        # Concatenate points
        concatenated_points = numpy.vstack((self._points.copy(), other._points.copy()))
        
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
                    raise ValueError(f"Property '{key}' must be a 2D NumPy array in both point clouds.")
                if not values_self.shape[0] == n_points_self or not values_other.shape[0] == n_points_other:
                    raise ValueError(f"Property '{key}' must have shape (N_p, A) where N_p matches the number of points in the respective point cloud.")
                if not values_self.shape[1] == values_other.shape[1]:
                    raise ValueError(f"Property '{key}' must have the same number of attributes (A) in both point clouds.")

            elif values_self is not None:
                if not values_self.ndim == 2:
                    raise ValueError(f"Property '{key}' must be a 2D NumPy array in the first point cloud.")
                if not values_self.shape[0] == n_points_self:
                    raise ValueError(f"Property '{key}' must have shape (N_p, A) where N_p matches the number of points in the first point cloud.")
                values_other = numpy.full((n_points_other, values_self.shape[1]), numpy.nan)
               
            elif values_other is not None:
                if not values_other.ndim == 2:
                    raise ValueError(f"Property '{key}' must be a 2D NumPy array in the second point cloud.")
                if not values_other.shape[0] == n_points_other:
                    raise ValueError(f"Property '{key}' must have shape (N_p, A) where N_p matches the number of points in the second point cloud.")
                values_self = numpy.full((n_points_self, values_other.shape[1]), numpy.nan)
                
            else:
                raise ValueError("This should never happen. Both property values are None when concatenating.")              
            
            # Concatenate property values
            concatenated_values = numpy.vstack((values_self.copy(), values_other.copy()))
            concatenated_properties[key] = concatenated_values   
        
        # Return new instance or modify in place
        if inplace:
            self._points = concatenated_points
            self._properties = concatenated_properties
            return self
        else:
            return self.__class__(points=concatenated_points, properties=concatenated_properties)


    def copy(self, copy_properties: bool = True) -> PointCloud:
        r"""
        Create a copy of the current :class:`PointCloud` instance.
        
        Parameters
        ----------
        copy_properties: :class:`bool`, optional
            If :obj:`True`, the properties of the point cloud will also be copied to the new instance. Default is :obj:`True`.


        Returns
        -------
        :class:`PointCloud`
            A new :class:`PointCloud` object containing the same points as the current instance.


        Examples
        --------
        Creating a :class:`PointCloud` from a random NumPy array and making a copy.

        .. code-block:: python
            :linenos:
            
            import numpy as np
            from pysdic import PointCloud

            # Create a random point cloud with 100 points
            random_points = np.random.rand(100, 3)  # shape (100, 3)
            point_cloud1 = PointCloud.from_array(random_points)

            # Create a copy of the existing PointCloud object
            point_cloud2 = point_cloud1.copy()

        """
        if not isinstance(copy_properties, bool):
            raise ValueError("copy_properties must be a boolean value.")
        
        obj = self.__class__(points=self._points.copy(), properties={key: values.copy() for key, values in self._properties.items()} if copy_properties else {})
                
        return obj
    

    def frame_transform(self, input_frame: Optional[Frame] = None, output_frame: Optional[Frame] = None, inplace: bool = False) -> PointCloud:
        r"""
        Transform the point cloud from an input frame of reference to an output frame of reference (only for 3D point clouds).

        Assuming the point cloud is defined in the coordinate system of the input frame, this method transforms the points to the coordinate system of the output frame.

        .. seealso::

            - Package `py3dframe <https://pypi.org/project/py3dframe/>`_ for more details on :class:`Frame` and :class:`FrameTransform`.

        .. warning::

            This method only works for 3-dimensional point clouds, as frame transformations are defined in 3D space.

        Parameters
        ----------
        input_frame: Optional[:class:`Frame`], optional
            The input frame representing the current coordinate system of the point cloud. If None, the canonical frame is assumed.

        output_frame: Optional[:class:`Frame`], optional
            The output frame representing the target coordinate system for the point cloud. If None, the canonical frame is assumed.

        inplace: :class:`bool`, optional
            If :obj:`True`, modifies the current point cloud in place and returns itself. If :obj:`False`, returns a new :class:`PointCloud` instance (default is :obj:`False`).

        Returns
        -------
        :class:`PointCloud`
            A new :class:`PointCloud` object containing the transformed points in the output frame or the modified current instance if :obj:`inplace` is :obj:`True`.

        Raises
        ------
        ValueError
            If the input or output frames are not instances of :class:`Frame`.

            
        Examples
        --------
        Create a :class:`PointCloud` from a random NumPy array.

        .. code-block:: python
            :linenos:
            
            import numpy as np
            from pysdic import PointCloud

            # Create a random point cloud with 100 points
            random_points = np.random.rand(100, 3)  # shape (100, 3)
            point_cloud = PointCloud.from_array(random_points)
        
        Lets assume this point cloud is defined in the canonical frame.
        We want to express the point cloud in local frame defined by a : 

        - orgin at (1, 1, 1)
        - x-axis along (0, 1, 0)
        - y-axis along (-1, 0, 0)
        - z-axis along (0, 0, 1)

        We can use the `frame_transform` method to perform this transformation.

        .. code-block:: python
            :linenos:
            
            from py3dframe import Frame

            # Define input and output frames
            input_frame = Frame.canonical()
            output_frame = Frame(origin=[1, 1, 1], x_axis=[0, 1, 0], y_axis=[-1, 0, 0], z_axis=[0, 0, 1])

            # Transform the point cloud from input frame to output frame
            transformed_point_cloud = point_cloud.frame_transform(input_frame=input_frame, output_frame=output_frame)
            print(transformed_point_cloud.points)
            # Output: A NumPy array of shape (100, 3) containing the transformed coordinates

        """
        if self.n_dimensions != 3:
            raise ValueError("Frame transformations are only supported for 3D point clouds.")
        
        # Validate input frames
        if input_frame is not None and not isinstance(input_frame, Frame):
            raise ValueError("Input frame must be an instance of Frame or None.")
        if output_frame is not None and not isinstance(output_frame, Frame):
            raise ValueError("Output frame must be an instance of Frame or None.")
        if not isinstance(inplace, bool):
            raise ValueError("inplace must be a boolean value.")
        
        # Default to canonical frame if None
        if input_frame is None:
            input_frame = Frame.canonical()
        if output_frame is None:
            output_frame = Frame.canonical()

        # Create the frame transform
        transform = FrameTransform(input_frame=input_frame, output_frame=output_frame)

        # Transform the points
        transformed_points = transform.transform(point=self.points.T).T

        # Return new instance or modify in place
        if inplace:
            self._points = transformed_points
            return self
        else:
            return self.__class__(points=transformed_points.copy(), properties={key: values.copy() for key, values in self._properties.items()})


    def filter_points(self, mask: ArrayLike, inplace: bool = False) -> PointCloud:
        r"""
        Filter points in the point cloud based on a boolean mask.

        This method returns a new :class:`PointCloud` object containing only the points where the mask is True.
        The mask should be a 1D boolean NumPy array with the same length as the number of points in the point cloud.
        All the properties associated with the points are also filtered accordingly.

        .. seealso::

            - :meth:`keep_points` for keeping points at specified indices.
            - :meth:`remove_points` for removing points at specified indices.


        Parameters
        ----------
        mask: ArrayLike
            A 1D boolean NumPy array of shape :math:`(N_p,)` where :math:`N_p` is the number of points in the point cloud.
            Each element should be True to keep the corresponding point, or False to remove it.

        inplace: :class:`bool`, optional
            If :obj:`True`, modifies the current point cloud in place and returns itself. If :obj:`False`, returns a new :class:`PointCloud` instance (default is :obj:`False`).


        Returns
        -------
        :class:`PointCloud`
            A new :class:`PointCloud` object containing only the filtered points or the modified current instance if :obj:`inplace` is True.


        Raises
        ------
        ValueError
            If the input mask is not a 1D boolean array of the correct length.


        Examples
        --------
        Create a :class:`PointCloud` from a random NumPy array.

        .. code-block:: python
            :linenos:
            
            import numpy as np
            from pysdic import PointCloud

            # Create a random point cloud with 100 points
            random_points = np.random.rand(100, 3)  # shape (100, 3)
            point_cloud = PointCloud.from_array(random_points)

        Filtering points with x-coordinate greater than 0.5.

        .. code-block:: python
            :linenos:
            
            # Create a boolean mask for points with x > 0.5
            mask = point_cloud.points[:, 0] > 0.5

            # Filter the point cloud using the mask
            filtered_point_cloud = point_cloud.filter_points(mask)
            print(filtered_point_cloud.points)
            # Output: A NumPy array containing only the points with x > 0.5

        """
        if not isinstance(inplace, bool):
            raise ValueError("inplace must be a boolean value.")
        
        # Load and validate mask
        mask = numpy.asarray(mask, dtype=bool)
        if mask.ndim != 1:
            raise ValueError("Mask must be a 1D array.")
        if mask.shape[0] != self.n_points:
            raise ValueError("Mask length must match the number of points in the point cloud.")
        
        # Return new instance or modify in place
        if inplace:
            self._points = self._points[mask]
            for key in self._properties:
                self._properties[key] = self._properties[key][mask]
            return self
        else:
            return self.__class__(points=self._points[mask].copy(), properties={key: values[mask].copy() for key, values in self._properties.items()})
        

    def keep_points(self, indices: ArrayLike, inplace: bool = False) -> PointCloud:
        r"""
        Keep only the points at the specified indices in the point cloud.

        This method returns a new :class:`PointCloud` object containing only the points at the specified indices.
        The properties associated with the points are also filtered accordingly.

        .. seealso::

            - :meth:`remove_points` for removing points at specified indices.
            - :meth:`filter_points` for filtering points based on a boolean mask.


        Parameters
        ----------
        indices: ArrayLike
            A 1D NumPy array of integer indices representing the points to be kept in the point cloud.

        inplace: :class:`bool`, optional
            If :obj:`True`, modifies the current point cloud in place and returns itself. If :obj:`False`, returns a new PointCloud instance (default is :obj:`False`).


        Returns
        -------
        :class:`PointCloud`
            A new :class:`PointCloud` object containing only the points at the specified indices or the modified current instance if :obj:`inplace` is True.


        Raises
        ------
        ValueError
            If any index is out of bounds or if the input is not a 1D array of integers.


        Examples
        --------
        Create a :class:`PointCloud` from a random NumPy array.

        .. code-block:: python
            :linenos:
            
            import numpy as np
            from pysdic import PointCloud

            # Create a random point cloud with 100 points
            random_points = np.random.rand(100, 3)  # shape (100, 3)
            point_cloud = PointCloud.from_array(random_points)

        Keeping only the points at indices 0, 2, and 4.

        .. code-block:: python
            :linenos:
            
            # Keep only points at indices 0, 2, and 4
            indices_to_keep = np.array([0, 2, 4])
            new_point_cloud = point_cloud.keep_points(indices_to_keep)
            print(new_point_cloud.points)
            # Output: A NumPy array of shape (3, 3) containing only the points at indices 0, 2, and 4

        """
        if not isinstance(inplace, bool):
            raise ValueError("inplace must be a boolean value.")
        
        # Load and validate indices
        indices = numpy.asarray(indices, dtype=numpy.int64)
        if indices.ndim != 1:
            raise ValueError("Indices must be a 1D array.")
        if numpy.any(indices < 0) or numpy.any(indices >= self.n_points):
            raise ValueError("Indices are out of bounds.")
        
        # Create mask for points to keep
        mask = numpy.zeros(self.n_points, dtype=bool)
        mask[indices] = True
        
        # Apply mask
        return self.filter_points(mask, inplace=inplace)
    
        
    def remove_points(self, indices: ArrayLike, inplace: bool = False) -> PointCloud:
        r"""
        Remove points from the point cloud based on their indices.

        This method returns a new :class:`PointCloud` object with the points at the specified indices removed.
        The properties associated with the points are also updated accordingly.

        .. seealso::

            - :meth:`keep_points` for keeping points at specified indices.
            - :meth:`filter_points` for filtering points based on a boolean mask.


        Parameters
        ----------
        indices: ArrayLike
            A 1D NumPy array of integer indices representing the points to be removed from the point cloud.

        inplace: :class:`bool`, optional
            If :obj:`True`, modifies the current point cloud in place and returns itself. If :obj:`False`, returns a new :class:`PointCloud` instance (default is :obj:`False`).


        Returns
        -------
        :class:`PointCloud`
            A new :class:`PointCloud` object with the points at the specified indices removed or the modified current instance if `inplace` is True.


        Raises
        ------
        ValueError
            If any index is out of bounds or if the input is not a 1D array of integers.

            
        Examples
        --------
        Create a :class:`PointCloud` from a random NumPy array.

        .. code-block:: python
            :linenos:
            
            import numpy as np
            from pysdic import PointCloud

            # Create a random point cloud with 100 points
            random_points = np.random.rand(100, 3)  # shape (100, 3)
            point_cloud = PointCloud.from_array(random_points)

        Removing points at indices 1 and 3.

        .. code-block:: python
            :linenos:
            
            # Remove points at indices 1 and 3
            indices_to_remove = np.array([1, 3])
            new_point_cloud = point_cloud.remove_points(indices_to_remove)
            print(new_point_cloud.points)
            # Output: A NumPy array of shape (98, 3) with points at indices 1 and 3 removed

        """
        if not isinstance(inplace, bool):
            raise ValueError("inplace must be a boolean value.")
        
        # Load and validate indices
        indices = numpy.asarray(indices, dtype=numpy.int64)
        if indices.ndim != 1:
            raise ValueError("Indices must be a 1D array.")
        if numpy.any(indices < 0) or numpy.any(indices >= self.n_points):
            raise ValueError("Indices are out of bounds.")
       
        # Create mask for points to remove  
        mask = numpy.ones(self.n_points, dtype=bool)
        mask[indices] = False
        
        # Apply mask
        return self.filter_points(mask, inplace=inplace)
        
        
    def remove_not_finite(self, inplace: bool = False) -> PointCloud:
        r"""
        Remove points from the point cloud that contain non-finite values (NaN or Inf).

        This method returns a new PointCloud object with all points containing NaN or Inf values removed.
        The properties associated with the points are also updated accordingly.

        .. seealso::

            - :meth:`is_finite` method for getting a boolean mask of finite points.
            - :meth:`remove_points` method for removing points at specified indices.

        Parameters
        ----------
        inplace: :class:`bool`, optional
            If :obj:`True`, modifies the current point cloud in place and returns itself. If :obj:`False`, returns a new PointCloud instance (default is :obj:`False`).


        Returns
        -------
        :class:`PointCloud`
            A new :class:`PointCloud` object containing only the finite points or the modified current instance if :obj:`inplace` is True.

            
        Examples
        --------
        Create a :class:`PointCloud` from a NumPy array with some non-finite values.

        .. code-block:: python
            :linenos:
            
            import numpy as np
            from pysdic import PointCloud

            # Create a point cloud with some non-finite values
            points_with_nan = np.array([[0.0, 1.0, 2.0],
                                        [np.nan, 1.0, 2.0],
                                        [3.0, np.inf, 4.0],
                                        [5.0, 6.0, 7.0]])

            point_cloud = PointCloud.from_array(points_with_nan)

        Removing non-finite points from the point cloud.

        .. code-block:: python
            :linenos:

            # Remove non-finite points
            finite_point_cloud = point_cloud.remove_not_finite()
            print(finite_point_cloud.points)
            # Output: A NumPy array of shape (2, 3) containing only the finite points
        """
        if not isinstance(inplace, bool):
            raise ValueError("inplace must be a boolean value.")
        
        # Create a mask for finite points
        mask = self.is_finite()
        
        # Apply mask
        return self.filter_points(mask, inplace=inplace)
   


    # ==============
    # Geometric Computations
    # ==============

    def bounding_box(self) -> Tuple[numpy.ndarray, numpy.ndarray]:
        r"""
        Compute the axis-aligned bounding box of the point cloud.

        The bounding box is defined by the minimum and maximum coordinates along each axis :math:`(x, y, z, ...)`.

        .. note::

            The non-finite values (NaN, Inf) are ignored in the computation. If the point cloud is empty or contains only non-finite values, a ValueError is raised.

        Returns
        -------
        :class:`numpy.ndarray`
            The minimum coordinates of the bounding box as a NumPy array of shape :math:`(E,)` representing (min_x, min_y, min_z, ...).

        :class:`numpy.ndarray`
            The maximum coordinates of the bounding box as a NumPy array of shape :math:`(E,)` representing (max_x, max_y, max_z, ...).

        Raises
        ------
        ValueError
            If the point cloud is empty or contains only non-finite values.

        Examples
        --------
        Create a tetrahedron point cloud and compute its bounding box.

        .. code-block:: python
            :linenos:

            import numpy as np
            from pysdic import PointCloud

            # Create a tetrahedron point cloud
            tetrahedron_points = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]])  # shape (4, 3)
            point_cloud = PointCloud.from_array(tetrahedron_points)

        Compute the bounding box using the `bounding_box` method.

        .. code-block:: python
            :linenos:

            # Compute the bounding box of the point cloud
            min_coords, max_coords = point_cloud.bounding_box()
            print("Min coordinates:", min_coords)
            print("Max coordinates:", max_coords)
            # Output:
            # Min coordinates: [0. 0. 0.]
            # Max coordinates: [1. 2. 3.]

        """
        if self.n_points == 0:
            raise ValueError("Cannot compute bounding box of an empty point cloud.")

        mask = self.is_finite()
        if not numpy.any(mask):
            raise ValueError("Point cloud contains only non-finite values; cannot compute bounding box.")
        
        finite_points = self.points[mask, :]
        
        # Compute min and max coordinates
        min_coords = numpy.min(finite_points, axis=0)
        max_coords = numpy.max(finite_points, axis=0)

        return min_coords, max_coords


    def bounding_sphere(self) -> Tuple[numpy.ndarray, float]:
        r"""
        Compute the bounding sphere of the point cloud.

        The bounding sphere is defined by its center and radius, which encompasses all points in the point cloud.

        .. note::

            The non-finite values (NaN, Inf) are ignored in the computation. If the point cloud is empty or contains only non-finite values, a ValueError is raised.


        Returns
        -------
        :class:`numpy.ndarray`
            The center of the bounding sphere as a NumPy array of shape :math:`(E,)`.

        :class:`float`
            The radius of the bounding sphere.


        Raises
        ------
        ValueError
            If the point cloud is empty or contains only non-finite values.

        
        Examples
        --------
        Create a tetrahedron point cloud and compute its bounding sphere.

        .. code-block:: python
            :linenos:

            import numpy as np
            from pysdic import PointCloud

            # Create a tetrahedron point cloud
            tetrahedron_points = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]])  # shape (4, 3)
            point_cloud = PointCloud.from_array(tetrahedron_points)

        Compute the bounding sphere using the `bounding_sphere` method.

        .. code-block:: python
            :linenos:

            # Compute the bounding sphere of the point cloud
            center, radius = point_cloud.bounding_sphere()
            print("Center of bounding sphere:", center)
            print("Radius of bounding sphere:", radius)
            # Output:
            # Center of bounding sphere: [0.25       0.5      0.75     ]
            # Radius of bounding sphere: 2.3184046

        """
        if self.n_points == 0:
            raise ValueError("Cannot compute bounding box of an empty point cloud.")

        mask = self.is_finite()
        if not numpy.any(mask):
            raise ValueError("Point cloud contains only non-finite values; cannot compute bounding box.")
        
        finite_points = self.points[mask, :]

        # Compute center and radius
        center = numpy.mean(finite_points, axis=0) # Shape (:math:`E`,)
        radius = numpy.max(numpy.linalg.norm(finite_points - center, axis=1))   
        
        return center, radius


    # ==============
    # Visualization
    # ==============
    def visualize(
        self, 
        point_color: Optional[Union[str, Tuple[Real, Real, Real]]] = None,
        point_size: Optional[Real] = None,
        point_opacity: Optional[Real] = None,
        point_mask: Optional[ArrayLike] = None,
        
        property: Optional[Union[str, ArrayLike]] = None,
        property_axis: Optional[int] = None,
        property_clim: Optional[Tuple[Real, Real]] = None,
        property_cmap: Optional[Union[str, matplotlib.colors.Colormap]] = None,
        property_below_color: Optional[Union[str, Tuple[Real, Real, Real]]] = None,
        property_above_color: Optional[Union[str, Tuple[Real, Real, Real]]] = None,
        
        bar_title: Optional[str] = None,
        bar_width: Optional[Real] = None,
        bar_height: Optional[Real] = None,
        bar_position_x: Optional[Real] = None,
        bar_position_y: Optional[Real] = None,
        bar_title_font_family: Optional[str] = None,
        bar_title_font_size: Optional[Integral] = None,
        bar_label_font_size: Optional[Integral] = None,
        bar_title_color: Optional[Union[str, Tuple[Real, Real, Real]]] = None,
        bar_title_bold: bool = False,
        bar_vertical: bool = True,
        bar_interactive: bool = False,
        show_bar: bool = True,

        title: Optional[str] = None,
        title_color: Optional[Union[str, Tuple[Real, Real, Real]]] = None,
        title_font_family: Optional[str] = None,
        title_font_size: Optional[Integral] = None,
        show_title: bool = True,
        
        bounds: Optional[Tuple[Real, Real, Real, Real, Real, Real]] = None,
        bounds_grid: Optional[str] = None,
        bounds_ticks: Optional[str] = None,
        bounds_location: Optional[str] = None,
        bounds_color: Optional[Union[str, Tuple[Real, Real, Real]]] = None,
        bounds_font_size: Optional[Integral] = None,
        bounds_font_family: Optional[str] = None,
        bounds_bold: bool = False,
        bounds_fmt: Optional[str] = None,
        bounds_title_x: str = "X Axis",
        bounds_title_y: str = "Y Axis",
        bounds_title_z: str = "Z Axis",
        bounds_n_xlabels: Integral = 5,
        bounds_n_ylabels: Integral = 5,
        bounds_n_zlabels: Integral = 5,
        bounds_show_xaxis: bool = True,
        bounds_show_yaxis: bool = True,
        bounds_show_zaxis: bool = True,
        bounds_show_xlabels: bool = True,
        bounds_show_ylabels: bool = True,
        bounds_show_zlabels: bool = True,
        show_bounds: bool = True,
        
        camera_position: Optional[ArrayLike] = None,
        camera_focal_point: Optional[ArrayLike] = None,
        camera_view_up: Optional[ArrayLike] = None,
        
        screenshot_path: Optional[str] = None,
    ) -> None:
        r"""
        Visualize the point cloud using PyVista (only for :math:`E \leq 3` point clouds).
        
        .. note::
        
            For 1D or 2D point clouds, the points are visualized in a 3D space by adding zero coordinates for the missing dimensions.

        Only finite points (not NaN or Inf) are visualized.  

        .. seealso::

            - `PyVista Documentation <https://docs.pyvista.org>`_ for more details on visualization options.


        Parameters
        ----------
        point_color: Union[:class:`str`, Tuple[Real, Real, Real]], optional
            The color of the points in the visualization. Can be a string representing a color name (e.g., :obj:`"red"`), or an RGB tuple or hex string.
            This parameter is ignored if :obj:`property` is provided.
            
        point_size: Real, optional
            The size of the points in the visualization. Must be a positive value if provided. Default is set by PyVista to :obj:`5.0`.
            
        point_opacity: Real, optional
            The opacity of the points in the visualization. Must be between :obj:`0.0` (fully transparent) and :obj:`1.0` (fully opaque) if provided. Default is set by PyVista to :obj:`1.0`.
            
        point_mask: Optional[ArrayLike], optional
            A boolean NumPy array of shape :math:`(N_p,)` where :math:`N_p` is the number of points in the point cloud. Points corresponding to True values in the mask are visualized, while points corresponding to False values are not visualized. If :obj:`None`, all finite points are visualized.
            
        property: Union[:class:`str`, ArrayLike], optional
            The property to use for coloring the points. Can be a string representing the property key of a property stored in the point cloud, or a NumPy array of shape :math:`(N_p,)` or :math:`(N_p, A)` representing a property to use for coloring the points.
            Note that if a string key is provided, the corresponding property must exist in the point cloud.
            
        property_axis: Optional[Integral], optional
            The axis to use when :obj:`property` contains multiple attributes per point (shape :math:`(N_p, A)`). If :obj:`None`, the magnitude of the property vector is used for coloring.
            By default, :obj:`None`.
            
        property_clim: Optional[Tuple[Real, Real]], optional
            The color limits (min, max) for the property values when :obj:`property` is provided. If :obj:`None`, the color limits are determined automatically.
            Default is :obj:`None`.
            
        property_cmap: Union[:class:`str`, :class:`matplotlib.colors.Colormap`], optional
            The colormap to use for coloring the points based on the property values. Can be a string representing a Matplotlib colormap or a Matplotlib Colormap object.
            If :obj:`None`, the default colormap :obj:`"viridis"` is used by Matplotlib.
            
        property_below_color: Union[:class:`str`, Tuple[Real, Real, Real]], optional
            The color to use for points with property values below the minimum color limit when :obj:`property_clim` is provided.
            Can be a string representing a color name (e.g., :obj:`"blue"`), or an RGB tuple or hex string.
            If :obj:`None`, the default below color is set to the minimum color of the colormap.
            
        property_above_color: Union[:class:`str`, Tuple[Real, Real, Real]], optional
            The color to use for points with property values above the maximum color limit when :obj:`property_clim` is provided.
            Can be a string representing a color name (e.g., :obj:`"red"`), or an RGB tuple or hex string.
            If :obj:`None`, the default above color is set to the maximum color of the colormap.
            
        bar_title: Optional[:class:`str`], optional
            The title of the color bar when :obj:`property` is provided. If :obj:`None`, no title is displayed.
            
        bar_width: Optional[Real], optional
            The width of the color bar as a fraction of the window width. Must be between :obj:`0.0` and :obj:`1.0` if provided.
            
        bar_height: Optional[Real], optional
            The height of the color bar as a fraction of the window height. Must be between :obj:`0.0` and :obj:`1.0` if provided.
            
        bar_position_x  Optional[Real], optional
            The x position of the color bar as a fraction of the window width. Must be between :obj:`0.0` and :obj:`1.0` if provided.
            
        bar_position_y: Optional[Real], optional
            The y position of the color bar as a fraction of the window height. Must be between :obj:`0.0` and :obj:`1.0` if provided.
            
        bar_title_font_family: Optional[:class:`str`], optional
            The font type of the color bar title between :obj:`"arial"`, :obj:`"courier"`, and :obj:`"times"`.
            
        bar_title_font_size: Optional[Integral], optional
            The font size of the color bar title.
            
        bar_label_font_size: Optional[Integral], optional
            The font size of the color bar labels.
            
        bar_title_color: Union[:class:`str`, Tuple[Real, Real, Real]], optional
            The color of the color bar title text. Can be a string representing a color name (e.g., :obj:`"black"`), or an RGB tuple or hex string.
            
        bar_title_bold: :class:`bool`, optional
            Whether to display the color bar title in bold font. Default is :obj:`False`.
            
        bar_vertical: :class:`bool`, optional
            Whether to display the color bar vertically. If :obj:`False`, the color bar is displayed horizontally. Default is :obj:`True`.
            
        bar_interactive: :class:`bool`, optional
            If :obj:`True`, enables interactive mode for the color bar in PyVista. Default is :obj:`False`. 
            
        show_bar: :class:`bool`, optional
            Whether to show the color bar in the visualization. Default is :obj:`True`.           
                    
        title: Optional[:class:`str`], optional
            The title of the visualization window.
            
        title_color: Union[:class:`str`, Tuple[Real, Real, Real]], optional
            The color of the title text. Can be a string representing a color name (e.g., :obj:`"blue"`), or an RGB tuple or hex string.

        title_font_family: Optional[:class:`str`], optional
            The font type of the title between :obj:`"arial"`, :obj:`"courier"`, and :obj:`"times"`.
            
        title_font_size: Optional[Integral], optional
            The font size of the title.
            
        show_title: :class:`bool`, optional
            Whether to show the title in the visualization. Default is :obj:`True`.
            
        bounds: Optional[Tuple[Real, Real, Real, Real, Real, Real]], optional
            The bounds of the axes in the format (:obj:`min_x`, :obj:`max_x`, :obj:`min_y`, :obj:`max_y`, :obj:`min_z`, :obj:`max_z`). If :obj:`None`, the bounds are determined automatically.
            
        bounds_grid: Optional[:class:`str`], optional
            Add grid lines to the backface (True, 'back', or 'backface') or to the frontface ('front', 'frontface') of the axes actor.
            
        bounds_ticks: Optional[:class:`str`], optional
            Set how the ticks are drawn on the axes grid. Options include: 'inside', 'outside', 'both'.
            
        bounds_location: Optional[:class:`str`], optional
            Set how the axes are drawn: either static ('all'), closest triad ('front', 'closest', 'default'), furthest triad ('back', 'furthest'), static closest to the origin ('origin'), or outer edges ('outer') in relation to the camera position.
            
        bounds_color: Union[:class:`str`, Tuple[Real, Real, Real]], optional
            The color of the axes bounds. Can be a string representing a color name (e.g., :obj:`"black"`), or an RGB tuple or hex string.

        bounds_font_size: Optional[Integral], optional
            The font size of the axes bounds.
            
        bounds_font_family: Optional[:class:`str`], optional
            The font type of the axes bounds between :obj:`"arial"`, :obj:`"courier"`, and :obj:`"times"`.
            
        bounds_bold: :class:`bool`, optional
            Whether to display the axes bounds in bold font. Default is :obj:`False`.
            
        bounds_fmt: Optional[:class:`str`], optional
            The format string for the axes bounds labels (e.g., :obj:`"{:.2f}"` for 2 decimal places).
            
        bounds_title_x: :class:`str`, optional
            The title for the x-axis bounds. Default is :obj:`"X Axis"`.
            
        bounds_title_y: :class:`str`, optional
            The title for the y-axis bounds. Default is :obj:`"Y Axis"`.
            
        bounds_title_z: :class:`str`, optional
            The title for the z-axis bounds. Default is :obj:`"Z Axis"`.
            
        bounds_n_xlabels: Integral, optional
            The number of labels to display on the x-axis bounds. Default is :obj:`5`.
            
        bounds_n_ylabels: Integral, optional
            The number of labels to display on the y-axis bounds. Default is :obj:`5`.
            
        bounds_n_zlabels: Integral, optional
            The number of labels to display on the z-axis bounds. Default is :obj:`5`.
            
        bounds_show_xaxis: :class:`bool`, optional
            Whether to show the x-axis in the bounds. Default is :obj:`True`.
            
        bounds_show_yaxis: :class:`bool`, optional
            Whether to show the y-axis in the bounds. Default is :obj:`True`.
            
        bounds_show_zaxis: :class:`bool`, optional
            Whether to show the z-axis in the bounds. Default is :obj:`True`.
            
        bounds_show_xlabels: :class:`bool`, optional
            Whether to show the x-axis labels in the bounds. Default is :obj:`True`.
            
        bounds_show_ylabels: :class:`bool`, optional
            Whether to show the y-axis labels in the bounds. Default is :obj:`True`.
            
        bounds_show_zlabels: :class:`bool`, optional
            Whether to show the z-axis labels in the bounds. Default is :obj:`True`.
            
        show_bounds: :class:`bool`, optional
            Whether to show the axes bounds in the visualization. Default is :obj:`True`.    
        
        camera_position: Optional[ArrayLike], optional
            The position of the camera in the 3D space as a NumPy array of shape :math:`(3,)`.
            
        camera_focal_point: Optional[ArrayLike], optional
            The focal point of the camera in the 3D space as a NumPy array of shape :math:`(3,)`.
            
        camera_view_up: Optional[ArrayLike], optional
            The view up vector of the camera as a NumPy array of shape :math:`(3,)`.
            
        screenshot_path: Optional[:class:`str`], optional
            The file path to save a screenshot of the visualization. If :obj:`None`, no screenshot is saved.
            

        Examples
        --------
        Create a :class:`PointCloud` from a random NumPy array.

        .. code-block:: python
            :linenos:
            
            import numpy as np
            from pysdic import PointCloud

            # Create a random point cloud with 100 points
            points_array = np.random.rand(100, 3)
            point_cloud = PointCloud.from_array(points_array)

        Visualize the point cloud using the `visualize` method.

        .. code-block:: python
            :linenos:
            
            point_cloud.visualize(point_color='red', point_size=10)
            
        """
        # Check empty point cloud
        if self.n_points == 0:
            raise ValueError("Cannot visualize an empty point cloud.")

        # Convert the mesh to 3D if necessary
        if self.n_dimensions < 1 or self.n_dimensions > 3:
            raise ValueError("Visualization is only supported for 1D, 2D, and 3D point clouds.")
        
        if self.n_dimensions == 3:
            coordinates = self.points
        elif self.n_dimensions == 2:
            z_coords = numpy.zeros((self.n_points, 1), dtype=self.points.dtype)
            coordinates = numpy.hstack((self.points, z_coords))
        elif self.n_dimensions == 1:
            y_coords = numpy.zeros((self.n_points, 1), dtype=self.points.dtype)
            z_coords = numpy.zeros((self.n_points, 1), dtype=self.points.dtype)
            coordinates = numpy.hstack((self.points, y_coords, z_coords))
        else:
            raise ValueError("Visualization is only supported for 1D, 2D, and 3D point clouds.")
        
        # Filter with point mask if provided
        if point_mask is not None:
            point_mask = numpy.asarray(point_mask, dtype=bool)
            if point_mask.ndim != 1:
                raise ValueError("point_mask must be a 1D boolean array.")
            if point_mask.shape[0] != self.n_points:
                raise ValueError("point_mask length must match the number of points in the point cloud.")
        else:
            point_mask = numpy.ones(self.n_points, dtype=bool)
            
        # Keep only finite points
        finite_mask = self.is_finite()
        mask = numpy.logical_and(point_mask, finite_mask)

        if numpy.sum(mask) == 0:
            raise ValueError("No finite points available for visualization.")
        
        coordinates = coordinates[mask]
        
        # Extract property array if provided
        _is_display_property = property is not None
        
        if _is_display_property:
            if isinstance(property, str):
                if property not in self._properties:
                    raise ValueError(f"Property '{property}' not found in the point cloud.")
                property_values = self._properties[property][mask]
            else:
                property_values = numpy.asarray(property)
                if property_values.ndim == 1:
                    if property_values.shape[0] != self.n_points:
                        raise ValueError(f"Property array must have shape ({self.n_points},) or ({self.n_points}, A).")
                    property_values = property_values[mask]
                elif property_values.ndim == 2:
                    if property_values.shape[0] != self.n_points:
                        raise ValueError(f"Property array must have shape ({self.n_points},) or ({self.n_points}, A).")
                    property_values = property_values[mask, :]
                else:
                    raise ValueError("Property array must be 1D or 2D.")

            # Reduce property to single attribute if necessary
            if property_values.ndim == 2:
                if property_axis is None:
                    property_values = numpy.linalg.norm(property_values, axis=1)
                else:
                    if property_axis < 0 or property_axis >= property_values.shape[1]:
                        raise ValueError(f"property_axis must be between 0 and {property_values.shape[1]-1}.")
                    property_values = property_values[:, property_axis]
        else:
            property_values = None
                    
        # Display the point cloud using PyVista
        pv_point_cloud = pyvista.PolyData(coordinates)
        plotter = pyvista.Plotter()
        
        plotter.add_mesh(
            pv_point_cloud,
            color=point_color,
            point_size=point_size,
            opacity=point_opacity,
            scalars=property_values,
            clim=property_clim,
            cmap=property_cmap,
            below_color=property_below_color,
            above_color=property_above_color,
            render_points_as_spheres=True,
            show_scalar_bar=False,  # Disable default scalar bar
        )
        
        if show_bounds:
            plotter.show_bounds(
                bounds=bounds,
                grid=bounds_grid,
                ticks=bounds_ticks,
                location=bounds_location,
                color=bounds_color,
                font_size=bounds_font_size,
                bold=bounds_bold,
                font_family=bounds_font_family,
                xtitle=bounds_title_x,
                ytitle=bounds_title_y,
                ztitle=bounds_title_z,
                n_xlabels=bounds_n_xlabels,
                n_ylabels=bounds_n_ylabels,
                n_zlabels=bounds_n_zlabels,
                fmt=bounds_fmt,
                show_xaxis=bounds_show_xaxis,
                show_yaxis=bounds_show_yaxis,
                show_zaxis=bounds_show_zaxis,
                show_xlabels=bounds_show_xlabels,
                show_ylabels=bounds_show_ylabels,
                show_zlabels=bounds_show_zlabels,
            )

        if show_bar and _is_display_property:
            plotter.add_scalar_bar(
                title=bar_title,
                width=bar_width,
                height=bar_height,
                position_x=bar_position_x,
                position_y=bar_position_y,
                font_family=bar_title_font_family,
                title_font_size=bar_title_font_size,
                label_font_size=bar_label_font_size,
                color=bar_title_color,
                bold=bar_title_bold,
                vertical=bar_vertical,
                interactive=bar_interactive,
            )

        if show_title and title is not None:
            plotter.add_title(
                title, 
                font_size=title_font_size,
                color=title_color,
                font=title_font_family,
            )
            
        if camera_position is not None or camera_focal_point is not None or camera_view_up is not None:
            if camera_focal_point is None or camera_view_up is None or camera_position is None:
                raise ValueError("`camera_position`, `camera_focal_point`, and `camera_view_up` must all be provided together.")
            plotter.camera.position = camera_position
            plotter.camera.focal_point = camera_focal_point
            plotter.camera.up = camera_view_up
    
        plotter.show(screenshot=screenshot_path)



