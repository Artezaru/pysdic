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

from typing import Optional, Tuple
from numbers import Number
from py3dframe import Frame, FrameTransform

import numpy
import pyvista
import os


class PointCloud3D(object):
    r"""
    A class representing a 3D point cloud.

    This class is designed to handle 3D point clouds, which are collections of points in a three-dimensional space.

    .. figure:: ../../../pysdic/resources/point_cloud_3d_visualize_example.png
        :width: 600
        :align: center

        Example of a 3D point cloud visualization using the `visualize` method.

    Parameters
    ----------
    points : numpy.ndarray
        A NumPy array of shape (N, 3) representing the coordinates of the points.

    Raises  
    ------
    ValueError
        If the input array does not have the correct shape (N, 3).

    """
    __slots__ = [
        "_points"
    ]

    def __init__(self, points: numpy.ndarray):
        self.points = points

    # ==========================
    # Properties
    # ==========================
    @property
    def points(self) -> numpy.ndarray:
        r"""
        An numpy array of shape (N, 3) representing the coordinates of the points in the cloud.

        .. note::

            This property is settable.

        .. note::

            An alias for ``points`` is ``coordinates``.

        Access and modify the points of the point cloud.

        Parameters
        ----------
        value : numpy.ndarray
            An array-like of shape (N, 3) representing the coordinates of the points in the cloud.

        Returns
        -------
        numpy.ndarray
            A NumPy array of shape (N, 3) containing the coordinates of the points in the cloud.

        Raises
        ------
        ValueError
            If the input array does not have the correct shape (N, 3).

        Examples
        --------
        Lets create a basic point random point cloud.

        .. code-block:: python

            import numpy as np
            from pysdic.geometry import PointCloud3D

            # Create a random point cloud with 100 points
            random_points = np.random.rand(100, 3)  # shape (100, 3)
            point_cloud = PointCloud3D(points=random_points)

        Now, we can access the points of the point cloud using the `points` property.

        .. code-block:: python

            # Access the points of the point cloud
            points = point_cloud.points
            print(points)
            # Output: A NumPy array of shape (100, 3) containing the coordinates of the points

        We can also modify the points of the point cloud by assigning a new array to the `points` property.

        .. code-block:: python

            # Modify the points of the point cloud
            new_points = np.random.rand(50, 3)  # shape (50, 3)
            point_cloud.points = new_points
            print(point_cloud.points)
            # Output: A NumPy array of shape (50, 3) containing the new coordinates of the points
        
        The attribute ``points`` is modifiable in place.

        .. code-block:: python

            # Modify the points of the point cloud in place
            point_cloud.points[0] = [0.0, 0.0, 0.0]
            print(point_cloud.points[0])
            # Output: [0.0, 0.0, 0.0]

        """
        return self._points
    
    @points.setter
    def points(self, value: numpy.ndarray) -> None:
        points = numpy.asarray(value, dtype=numpy.float64)
        if not (points.ndim == 2 and points.shape[1] == 3):
            raise ValueError("Points must be a 2D NumPy array with shape (N, 3).")
        self._points = points

    @property
    def coordinates(self) -> numpy.ndarray:
        r"""
        Alias for :meth:`points` property.
        """
        return self.points

    @coordinates.setter
    def coordinates(self, value: numpy.ndarray) -> None:
        self.points = value


    @property
    def n_points(self) -> int:
        r"""
        The number of points N in the point cloud.

        .. note::

            You can also use `len(point_cloud)`.

        .. seealso::

            - :meth:`shape` for getting the shape of the points array.

        Returns
        -------
        int
            The number of points in the point cloud.

        Examples
        --------
        Lets create a basic point random point cloud.

        .. code-block:: python

            import numpy as np
            from pysdic.geometry import PointCloud3D

            # Create a random point cloud with 100 points
            random_points = np.random.rand(100, 3)  # shape (100, 3)
            point_cloud = PointCloud3D(points=random_points)

        The number of points in the point cloud can be accessed using the `n_points` property.

        .. code-block:: python

            # Access the number of points in the point cloud
            num_points = point_cloud.n_points
            print(num_points)
            # Output: 100

        You can also use the built-in `len` function to get the number of points.

        .. code-block:: python

            # Get the number of points using len()
            num_points_len = len(point_cloud)
            print(num_points_len)
            # Output: 100

        """
        return self.points.shape[0]
    
    @property
    def shape(self) -> tuple[int, int]:
        r"""
        The shape of the points array (N, 3), where N is the number of points.

        .. seealso::

            - :meth:`n_points` for getting the number of points in the point cloud.

        Returns
        -------
        tuple[int, int]
            A tuple representing the shape of the points array (N, 3), where N is the number of points.

        Examples
        --------
        Lets create a basic point random point cloud.

        .. code-block:: python

            import numpy as np
            from pysdic.geometry import PointCloud3D

            # Create a random point cloud with 100 points
            random_points = np.random.rand(100, 3)  # shape (100, 3)
            point_cloud = PointCloud3D(points=random_points)

        The shape of the points array can be accessed using the `shape` property.

        .. code-block:: python

            # Access the shape of the points array
            points_shape = point_cloud.shape
            print(points_shape)
            # Output: (100, 3)

        """
        return self.points.shape
    
    # ==========================
    # Class methods
    # ==========================
    @classmethod
    def from_array(cls, points: numpy.ndarray) -> PointCloud3D:
        r"""
        Create a PointCloud3D object from a NumPy array of shape (N, 3).

        .. seealso::

            - :meth:`to_array` method for converting the point cloud back to a NumPy array.

        Parameters
        ----------
        points : numpy.ndarray
            A NumPy array of shape (N, 3) representing the coordinates of the points.

        Returns
        -------
        PointCloud3D
            A PointCloud3D object containing the provided points.

        Raises
        ------
        ValueError
            If the input array does not have the correct shape (N, 3).

        Examples
        --------
        Creating a PointCloud3D object from a random NumPy array.

        .. code-block:: python

            import numpy as np
            from pysdic.geometry import PointCloud3D

            # Create a random point cloud with 100 points
            random_points = np.random.rand(100, 3)  # shape (100, 3)
            point_cloud = PointCloud3D.from_array(random_points)

        Now, ``point_cloud`` is a PointCloud3D object containing the points from the NumPy array.

        """
        return cls(points=points)
    

    @classmethod
    def from_cls(cls, other: PointCloud3D) -> PointCloud3D:
        r"""
        Create a PointCloud3D object from another PointCloud3D instance.

        .. seealso::

            - :meth:`copy` for creating a copy of the current instance. Same as ``PointCloud3D.from_cls(self)``.

        Parameters
        ----------
        other : PointCloud3D
            Another PointCloud3D instance to copy points from.

        Returns
        -------
        PointCloud3D
            A new PointCloud3D object containing the same points as the input instance.

        Examples
        --------
        Creating a PointCloud3D object from another PointCloud3D instance.

        .. code-block:: python

            import numpy as np
            from pysdic.geometry import PointCloud3D

            # Create a random point cloud with 100 points
            random_points = np.random.rand(100, 3)  # shape (100, 3)
            point_cloud1 = PointCloud3D.from_array(random_points)

            # Create a new PointCloud3D object from the existing one
            point_cloud2 = PointCloud3D.from_cls(point_cloud1)

        Now, ``point_cloud2`` is a new PointCloud3D object containing a copy of the points from `point_cloud1`.

        """
        if not isinstance(other, cls):
            raise ValueError("Input must be an instance of PointCloud3D.")
        return cls(points=other.points.copy())
    

    @classmethod
    def from_empty(cls) -> PointCloud3D:
        r"""
        Create an empty PointCloud3D object with no points, i.e., shape (0, 3).

        Returns
        -------
        PointCloud3D
            An empty PointCloud3D object with no points.

        Examples
        --------
        Creating an empty PointCloud3D object.

        .. code-block:: python

            from pysdic.geometry import PointCloud3D

            # Create an empty point cloud
            empty_point_cloud = PointCloud3D.from_empty()

        Now, ``empty_point_cloud`` is a PointCloud3D object with 0 points.

        """
        return cls(points=numpy.empty((0, 3), dtype=numpy.float64))
    

    @classmethod
    def from_obj(cls, filepath: str) -> PointCloud3D:
        r"""
        Create a PointCloud3D object from a OBJ file.

        The OBJ file should contain vertex definitions starting with the letter 'v'.
        Only vertices are read; faces and other elements are ignored.

        The points are extracted using ``pyvista`` library.

        .. seealso::

            - :meth:`to_obj` method for saving the point cloud to a OBJ file.

        Parameters
        ----------
        filepath : str
            The path to the OBJ file.

        Returns
        -------
        PointCloud3D
            A PointCloud3D object containing the points read from the OBJ file.

        
        Examples
        --------
        Creating a PointCloud3D object from a OBJ file.

        .. code-block:: python

            from pysdic.geometry import PointCloud3D
            # Create a point cloud from a OBJ file
            point_cloud = PointCloud3D.from_obj('path/to/point_cloud.obj')

        Now, ``point_cloud`` is a PointCloud3D object containing the points read from the specified OBJ file.

        """
        path = os.path.expanduser(filepath)
        if not os.path.isfile(path) or not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        mesh = pyvista.read(filepath, force_ext='.obj')
        
        if not hasattr(mesh, 'points'):
            raise ValueError(f"No points found in the OBJ file: {filepath}")
        if not (isinstance(mesh.points, numpy.ndarray) and mesh.points.ndim == 2 and mesh.points.shape[1] == 3):
            raise ValueError(f"Points in the OBJ file must be a 2D NumPy array with shape (N, 3).")
        
        return cls.from_array(mesh.points)
    

    @classmethod
    def from_ply(cls, filepath: str) -> PointCloud3D:
        r"""
        Create a PointCloud3D object from a PLY file.

        The PLY file should contain vertex definitions.
        Only vertices are read; faces and other elements are ignored.

        The points are extracted using ``pyvista`` library.

        .. seealso::

            - :meth:`to_ply` method for saving the point cloud to a PLY file.

        Parameters
        ----------
        filepath : str
            The path to the PLY file.

        Returns
        -------
        PointCloud3D
            A PointCloud3D object containing the points read from the PLY file.


        Examples
        --------
        Creating a PointCloud3D object from a PLY file.

        .. code-block:: python

            from pysdic.geometry import PointCloud3D
            # Create a point cloud from a PLY file
            point_cloud = PointCloud3D.from_ply('path/to/point_cloud.ply')

        Now, ``point_cloud`` is a PointCloud3D object containing the points read from the specified PLY file.
        """
        path = os.path.expanduser(filepath)
        if not os.path.isfile(path) or not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {filepath}")

        mesh = pyvista.read(filepath, force_ext='.ply')
        if not hasattr(mesh, 'points'):
            raise ValueError(f"No points found in the PLY file: {filepath}")
        
        if not (isinstance(mesh.points, numpy.ndarray) and mesh.points.ndim == 2 and mesh.points.shape[1] == 3):
            raise ValueError(f"Points in the PLY file must be a 2D NumPy array with shape (N, 3).")
        
        return cls.from_array(mesh.points)
    

    @classmethod
    def from_vtk(cls, filepath: str) -> PointCloud3D:
        r"""
        Create a PointCloud3D object from a VTK file.

        The VTK file should contain vertex definitions.
        Only vertices are read; faces and other elements are ignored.

        The points are extracted using ``pyvista`` library.

        .. seealso::

            - :meth:`to_vtk` method for saving the point cloud to a VTK file.

        Parameters
        ----------
        filepath : str
            The path to the VTK file.


        Returns
        -------
        PointCloud3D
            A PointCloud3D object containing the points read from the VTK file.

        
        Examples
        --------
        Creating a PointCloud3D object from a VTK file.
        
        .. code-block:: python

            from pysdic.geometry import PointCloud3D
            # Create a point cloud from a VTK file
            point_cloud = PointCloud3D.from_vtk('path/to/point_cloud.vtk')

        Now, ``point_cloud`` is a PointCloud3D object containing the points read from the specified VTK file.

        """
        path = os.path.expanduser(filepath)

        if not os.path.isfile(path) or not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {filepath}")

        mesh = pyvista.read(filepath, force_ext='.vtk')

        if not hasattr(mesh, 'points'):
            raise ValueError(f"No points found in the VTK file: {filepath}")

        if not (isinstance(mesh.points, numpy.ndarray) and mesh.points.ndim == 2 and mesh.points.shape[1] == 3):
            raise ValueError(f"Points in the VTK file must be a 2D NumPy array with shape (N, 3).")

        return cls.from_array(mesh.points)
    

    @classmethod
    def from_xyz(cls, filepath: str, delimiter: str = ' ') -> PointCloud3D:
        r"""
        Create a PointCloud3D object from a XYZ file.

        The points are read using ``numpy.loadtxt``.

        The XYZ file should contain three columns representing the x, y, and z coordinates of the points. 
        Lines starting with `#` or `//` are treated as comments and ignored.

        .. seealso::

            - :meth:`to_xyz` method for saving the point cloud to a XYZ file.

        Parameters
        ----------
        filepath : str
            The path to the XYZ file.
        delimiter : str, optional
            The delimiter used in the XYZ file (default is space).

        Returns
        -------
        PointCloud3D
            A PointCloud3D object containing the points read from the XYZ file.

            
        Examples
        --------
        Creating a PointCloud3D object from a XYZ file.

        .. code-block:: python

            from pysdic.geometry import PointCloud3D

            # Create a point cloud from a XYZ file
            point_cloud = PointCloud3D.from_xyz('path/to/point_cloud.xyz', delimiter=',')

        Now, ``point_cloud`` is a PointCloud3D object containing the points read from the specified XYZ file.

        """
        path = os.path.expanduser(filepath)
        if not os.path.isfile(path) or not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        points = numpy.loadtxt(filepath, delimiter=delimiter, comments=['#', '//'])
        if points.ndim == 1 and points.shape[0] == 3:
            points = points.reshape((1, 3))
        return cls.from_array(points)
    

    def to_array(self) -> numpy.ndarray:
        r"""
        Convert the point cloud to a NumPy array of shape (N, 3).

        Similar to :meth:`as_array` method.

        .. note::

            The returned array is a copy of the internal points array. Modifying it will not affect the original point cloud.

        .. seealso::

            - :meth:`points` property for accessing and modifying the points of the point cloud.
            - :meth:`as_array` method for converting the point cloud to a NumPy array.

        Returns
        -------
        numpy.ndarray
            A NumPy array of shape (N, 3) containing the coordinates of the points in the cloud.

            
        Examples
        --------
        Creating a PointCloud3D object from a random NumPy array.

        .. code-block:: python

            import numpy as np
            from pysdic.geometry import PointCloud3D

            # Create a random point cloud with 100 points
            random_points = np.random.rand(100, 3)  # shape (100, 3)
            point_cloud = PointCloud3D.from_array(random_points)

        Convert back to a NumPy array using the `to_array` method.

        .. code-block:: python

            # Convert the point cloud back to a NumPy array
            points_array = point_cloud.to_array()
            print(points_array)
            # Output: A NumPy array of shape (100, 3) containing the coordinates
        
        """
        return self.as_array()
    

    def as_array(self) -> numpy.ndarray:
        r"""
        Convert the point cloud to a NumPy array of shape (N, 3).

        Similar to :meth:`to_array` method.

        .. note::

            The returned array is a copy of the internal points array. Modifying it will not affect the original point cloud.

        .. seealso::

            - :meth:`points` property for accessing and modifying the points of the point cloud.

        Returns
        -------
        numpy.ndarray
            A NumPy array of shape (N, 3) containing the coordinates of the points in the cloud.

            
        Examples
        --------
        Creating a PointCloud3D object from a random NumPy array.      

        .. code-block:: python

            import numpy as np
            from pysdic.geometry import PointCloud3D

            # Create a random point cloud with 100 points
            random_points = np.random.rand(100, 3)  # shape (100, 3)
            point_cloud = PointCloud3D.from_array(random_points)

        Convert back to a NumPy array using the `as_array` method.

        .. code-block:: python

            # Convert the point cloud back to a NumPy array
            points_array = point_cloud.as_array()
            print(points_array)
            # Output: A NumPy array of shape (100, 3) containing the coordinates
        
        """
        return self.points.copy()
    

    def to_obj(self, filepath: str, binary: bool = False) -> None:
        r"""
        Save the point cloud to a OBJ file.

        The points are saved using ``pyvista`` library.

        The OBJ file will contain vertex definitions starting with the letter 'v'.

        .. seealso::

            - :meth:`from_obj` method for creating a PointCloud3D object from a OBJ file.

        .. note::

            To force `pyvista` to save as OBJ format, an temporary path ``filepath.tmp.obj`` is created internally.

        Parameters
        ----------
        filepath : str
            The path to the output OBJ file.

        binary : bool, optional
            If True, the OBJ file will be saved in binary format. Default is False.

            
        Examples
        --------
        Saving a PointCloud3D object to a OBJ file.

        .. code-block:: python

            from pysdic.geometry import PointCloud3D
            import numpy as np

            # Create a random point cloud with 100 points
            random_points = np.random.rand(100, 3)  # shape (100, 3)
            point_cloud = PointCloud3D.from_array(random_points)

            # Save the point cloud to a OBJ file
            point_cloud.to_obj('path/to/output_point_cloud.obj')

        This will save the points of the point cloud to the specified OBJ file.
        """
        path = os.path.expanduser(filepath)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if not isinstance(binary, bool):
            raise ValueError("The 'binary' parameter must be a boolean value.")
        
        # Set a temporary path to force pyvista to save as OBJ
        tmp_path = filepath + ".tmp.obj"
        
        point_cloud_mesh = pyvista.PolyData(self.points)
        point_cloud_mesh.save(tmp_path, binary=binary)

        # Rename the temporary file to the desired location
        os.rename(tmp_path, path)


    def to_ply(self, filepath: str, binary: bool = False) -> None:
        r"""
        Save the point cloud to a PLY file.

        The points are saved using ``pyvista`` library.

        The PLY file will contain vertex definitions.

        .. seealso::

            - :meth:`from_ply` method for creating a PointCloud3D object from a PLY file.

        .. note::

            To force `pyvista` to save as PLY format, an temporary path ``filepath.tmp.ply`` is created internally.

        Parameters
        ----------
        filepath : str
            The path to the output PLY file.

        binary : bool, optional
            If True, the PLY file will be saved in binary format. Default is False.
 
            
        Examples
        --------
        Saving a PointCloud3D object to a PLY file.

        .. code-block:: python

            from pysdic.geometry import PointCloud3D
            import numpy as np

            # Create a random point cloud with 100 points
            random_points = np.random.rand(100, 3)  # shape (100, 3)
            point_cloud = PointCloud3D.from_array(random_points)

            # Save the point cloud to a PLY file
            point_cloud.to_ply('path/to/output_point_cloud.ply')

        This will save the points of the point cloud to the specified PLY file.
        """
        path = os.path.expanduser(filepath)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if not isinstance(binary, bool):
            raise ValueError("The 'binary' parameter must be a boolean value.")

        # Set a temporary path to force pyvista to save as PLY
        tmp_path = filepath + ".tmp.ply"

        point_cloud_mesh = pyvista.PolyData(self.points)
        point_cloud_mesh.save(tmp_path, binary=binary)

        # Rename the temporary file to the desired location
        os.rename(tmp_path, path)



    def to_vtk(self, filepath: str, binary: bool = False, only_finite: bool = False) -> None:
        r"""
        Save the point cloud to a VTK file.

        The points are saved using ``pyvista`` library.

        The VTK file will contain vertex definitions.

        .. seealso::

            - :meth:`from_vtk` method for creating a PointCloud3D object from a VTK file.

        .. note::

            To force `pyvista` to save as VTK format, an temporary path ``filepath.tmp.vtk`` is created internally.

        .. warning::

            VTK cannot handle NaN or infinite values. If the point cloud contains such values, consider using the ``only_finite`` parameter to filter them out before saving.

        Parameters
        ----------
        filepath : str
            The path to the output VTK file.

        binary : bool, optional
            If True, the VTK file will be saved in binary format. Default is False.

        only_finite : bool, optional
            If True, only finite points (excluding NaN and infinite values) will be saved. Default is False.

        Raises
        ------
        ValueError
            If the 'binary' parameter is not a boolean value.
            If the 'only_finite' parameter is not a boolean value.
            If 'only_finite' is True and there are no finite points to save.

        
        Examples
        --------
        Saving a PointCloud3D object to a VTK file.

        .. code-block:: python

            from pysdic.geometry import PointCloud3D
            import numpy as np

            # Create a random point cloud with 100 points
            random_points = np.random.rand(100, 3)  # shape (100, 3)
            point_cloud = PointCloud3D.from_array(random_points)

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

        # Set a temporary path to force pyvista to save as VTK
        tmp_path = filepath + ".tmp.vtk"

        points_to_save = self.points
        if only_finite:
            finite_mask = numpy.isfinite(self.points).all(axis=1)
            points_to_save = self.points[finite_mask]
            if points_to_save.shape[0] == 0:
                raise ValueError("No finite points to save in the point cloud.")

        else:
            if not numpy.isfinite(self.points).all():
                raise ValueError("Point cloud contains NaN or infinite values. Consider using 'only_finite=True' to filter them out before saving.")

        point_cloud_mesh = pyvista.PolyData(points_to_save)
        point_cloud_mesh.save(tmp_path, binary=binary)

        # Rename the temporary file to the desired location
        os.rename(tmp_path, path)


    def to_xyz(self, filepath: str, delimiter: str = ' ') -> None:
        r"""
        Save the point cloud to a XYZ file.

        The points are saved using ``numpy.savetxt``.

        The XYZ file will contain three columns representing the x, y, and z coordinates of the points.

        .. seealso::

            - :meth:`from_xyz` method for creating a PointCloud3D object from a XYZ file.

        Parameters
        ----------
        filepath : str
            The path to the output XYZ file.
        delimiter : str, optional
            The delimiter to use in the XYZ file (default is space).

        Examples
        --------
        Saving a PointCloud3D object to a XYZ file.

        .. code-block:: python

            from pysdic.geometry import PointCloud3D

            # Create a random point cloud with 100 points
            random_points = np.random.rand(100, 3)  # shape (100, 3)
            point_cloud = PointCloud3D.from_array(random_points)

            # Save the point cloud to a XYZ file
            point_cloud.to_xyz('path/to/output_point_cloud.xyz', delimiter=',')

        This will save the points of the point cloud to the specified XYZ file.

        """
        path = os.path.expanduser(filepath)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        numpy.savetxt(path, self.points, delimiter=delimiter)
    

    # ==========================
    # Operations
    # ==========================
    def __len__(self) -> int:
        return self.n_points

    def __add__(self, other: PointCloud3D) -> PointCloud3D:
        return self.concatenate(other)
    
    def __iadd__(self, other: PointCloud3D) -> PointCloud3D:
        self.concatenate(other, inplace=True)
        return self
    
    # ==========================
    # Methods Geometry Manipulation
    # ==========================
    def allclose(self, other: PointCloud3D, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
        r"""
        Check if all points in the current point cloud are approximately equal to the points in another PointCloud3D instance within a tolerance.

        This method compares the points of the current point cloud with those of another PointCloud3D instance and returns True if all corresponding points are approximately equal within the specified relative and absolute tolerances.

        .. seealso::

            - :meth:`numpy.allclose` for more details on the comparison.

        Nans are considered equal.

        Parameters
        ----------
        other : PointCloud3D
            Another PointCloud3D instance to compare with the current point cloud.

        rtol : float, optional
            The relative tolerance parameter (default is 1e-05).

        atol : float, optional
            The absolute tolerance parameter (default is 1e-08).

        Returns
        -------
        bool
            True if all points are approximately equal within the specified tolerances, False otherwise.

        Raises
        ------
        ValueError
            If the input is not an instance of PointCloud3D or if the point clouds have different shapes.

        Examples
        --------
        Creating a PointCloud3D object from a random NumPy array.

        .. code-block:: python
            
            import numpy as np
            from pysdic.geometry import PointCloud3D

            # Create a random point cloud with 100 points
            random_points = np.random.rand(100, 3)  # shape (100, 3)
            point_cloud1 = PointCloud3D.from_array(random_points)

        Compare the point cloud with another point cloud that is slightly modified.

        .. code-block:: python

            # Create a second point cloud by adding small noise to the first one
            noise = np.random.normal(scale=1e-8, size=random_points.shape)
            point_cloud2 = PointCloud3D.from_array(random_points + noise)

            # Check if the two point clouds are approximately equal
            are_close = point_cloud1.allclose(point_cloud2, rtol=1e-5, atol=1e-8)
            print(are_close)
            # Output: True (most likely, depending on the noise)

        Compare with a point cloud that is significantly different.

        .. code-block:: python

            # Create a third point cloud that is significantly different
            different_points = np.random.rand(100, 3) + 1.0  # Shifted by 1.0
            point_cloud3 = PointCloud3D.from_array(different_points)
            are_close = point_cloud1.allclose(point_cloud3, rtol=1e-5, atol=1e-8)
            print(are_close)
            # Output: False

        """
        if not isinstance(other, PointCloud3D):
            raise ValueError("Input must be an instance of PointCloud3D.")
        if self.shape != other.shape:
            raise ValueError("Point clouds must have the same shape to compare.")
        
        return numpy.allclose(self.points, other.points, rtol=rtol, atol=atol, equal_nan=True)


    def concatenate(self, other: PointCloud3D, inplace: bool = False) -> PointCloud3D:
        r"""
        Concatenate the current point cloud with another PointCloud3D instance.

        This method combines the points from both point clouds into a new PointCloud3D object.

        .. note::

            This method is functionally equivalent to using the `+` operator.

        .. seealso::

            - :meth:`merge` for merging points from another point cloud avoiding duplicates.

        Parameters
        ----------
        other : PointCloud3D
            Another PointCloud3D instance to concatenate with the current point cloud.

        inplace : bool, optional
            If True, modifies the current point cloud in place and returns itself. If False, returns a new PointCloud3D instance (default is False).

        Returns
        -------
        PointCloud3D
            A new PointCloud3D object containing the concatenated points from both point clouds or the modified current instance if `inplace` is True.

        Raises
        ------
        ValueError
            If the input is not an instance of PointCloud3D.

        Examples
        --------
        Creating two PointCloud3D objects.

        .. code-block:: python

            import numpy as np
            from pysdic.geometry import PointCloud3D

            # Create two random NumPy arrays of shape (100, 3)
            random_points1 = np.random.rand(100, 3)  # shape (100, 3)
            random_points2 = np.random.rand(50, 3)   # shape (50, 3)

            point_cloud1 = PointCloud3D.from_array(random_points1)
            point_cloud2 = PointCloud3D.from_array(random_points2)

        Concatenate the two point clouds using the `concatenate` method.

        .. code-block:: python

            # Concatenate the two point clouds
            concatenated_point_cloud = point_cloud1.concatenate(point_cloud2)

            print(concatenated_point_cloud.points)
            # Output: A NumPy array of shape (150, 3) containing the concatenated coordinates

        This is equivalent to using the `+` operator.

        .. code-block:: python

            # Concatenate using the + operator
            concatenated_point_cloud_op = point_cloud1 + point_cloud2
            print(concatenated_point_cloud_op.points)
            # Output: A NumPy array of shape (150, 3) containing the concatenated coordinates

        """
        # Check if other is a PointCloud3D instance
        if not isinstance(other, PointCloud3D):
            raise ValueError("Input must be an instance of PointCloud3D.")
        if not isinstance(inplace, bool):
            raise ValueError("inplace must be a boolean value.")
        
        # Concatenate points
        concatenated_points = numpy.vstack((self.points, other.points))

        # Return new instance or modify in place
        if inplace:
            self.points = concatenated_points
            return self
        else:
            return PointCloud3D.from_array(concatenated_points.copy())


    def copy(self) -> PointCloud3D:
        r"""
        Create a copy of the current PointCloud3D instance.

        .. seealso::

            - :meth:`from_cls` for creating a new PointCloud3D instance from another instance. Same as ``self.copy()``.

        Returns
        -------
        PointCloud3D
            A new PointCloud3D object containing the same points as the current instance.

        Examples
        --------
        Creating a PointCloud3D from a random NumPy array and making a copy.

        .. code-block:: python

            import numpy as np
            from pysdic.geometry import PointCloud3D

            # Create a random point cloud with 100 points
            random_points = np.random.rand(100, 3)  # shape (100, 3)
            point_cloud1 = PointCloud3D.from_array(random_points)

            # Create a copy of the existing PointCloud3D object
            point_cloud2 = point_cloud1.copy()

        """
        return self.from_cls(self)


    def frame_transform(self, input_frame: Optional[Frame] = None, output_frame: Optional[Frame] = None, inplace: bool = False) -> PointCloud3D:
        r"""
        Transform the point cloud from an input frame of reference to an output frame of reference.

        Assuming the point cloud is defined in the coordinate system of the input frame, this method transforms the points to the coordinate system of the output frame.

        .. seealso::

            - Package `py3dframe <https://pypi.org/project/py3dframe/>`_ for more details on Frame and FrameTransform.

        Parameters
        ----------
        input_frame : Optional[Frame], optional
            The input frame representing the current coordinate system of the point cloud. If None, the canonical frame is assumed.

        output_frame : Optional[Frame], optional
            The output frame representing the target coordinate system for the point cloud. If None, the canonical frame is assumed.

        inplace : bool, optional
            If True, modifies the current point cloud in place and returns itself. If False, returns a new PointCloud3D instance (default is False).

        Returns
        -------
        PointCloud3D
            A new PointCloud3D object containing the transformed points in the output frame or the modified current instance if `inplace` is True.

        Raises
        ------
        ValueError
            If the input or output frames are not instances of Frame.

        Examples
        --------
        Create a PointCloud3D from a random NumPy array.

        .. code-block:: python

            import numpy as np
            from pysdic.geometry import PointCloud3D

            # Create a random point cloud with 100 points
            random_points = np.random.rand(100, 3)  # shape (100, 3)
            point_cloud = PointCloud3D.from_array(random_points)
        
        Lets assume this point cloud is defined in the canonical frame.
        We want to express the point cloud in local frame defined by a : 

        - orgin at (1, 1, 1)
        - x-axis along (0, 1, 0)
        - y-axis along (-1, 0, 0)
        - z-axis along (0, 0, 1)

        We can use the `frame_transform` method to perform this transformation.

        .. code-block:: python

            from py3dframe import Frame

            # Define input and output frames
            input_frame = Frame.canonical()
            output_frame = Frame(origin=[1, 1, 1], x_axis=[0, 1, 0], y_axis=[-1, 0, 0], z_axis=[0, 0, 1])

            # Transform the point cloud from input frame to output frame
            transformed_point_cloud = point_cloud.frame_transform(input_frame=input_frame, output_frame=output_frame)
            print(transformed_point_cloud.points)
            # Output: A NumPy array of shape (100, 3) containing the transformed coordinates

        """
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
            self.points = transformed_points
            return self
        else:
            return PointCloud3D.from_array(transformed_points.copy())
    

    def keep_points(self, other: PointCloud3D, inplace: bool = False) -> PointCloud3D:
        r"""
        Keep only the points in the current point cloud that are present in another PointCloud3D instance.

        This method returns a new PointCloud3D object containing only the points that are also present in the provided PointCloud3D instance.

        .. note::

            Points in the `other` point cloud that are not present in the current point cloud are ignored.

        .. seealso::

            - :meth:`remove_points` for removing points that are present in another PointCloud3D instance.
            - :meth:`keep_points_at` for keeping points at specified indices.
            - :meth:`keep_points_inplace` for keeping points in place.

        Parameters
        ----------
        other : PointCloud3D
            Another PointCloud3D instance containing the points to be kept in the current point cloud.

        inplace : bool, optional
            If True, modifies the current point cloud in place and returns itself. If False, returns a new PointCloud3D instance (default is False).

        Returns
        -------
        PointCloud3D
            A new PointCloud3D object containing only the points that are also present in the provided PointCloud3D instance or the modified current instance if `inplace` is True.

        Raises
        ------
        ValueError
            If the input is not an instance of PointCloud3D.

        Examples
        --------
        Create a PointCloud3D from a random NumPy array.

        .. code-block:: python

            import numpy as np
            from pysdic.geometry import PointCloud3D
            # Create a random point cloud with 100 points
            random_points = np.random.rand(100, 3)  # shape (100, 3)
            point_cloud = PointCloud3D.from_array(random_points)

        Create another PointCloud3D with some common points.

        .. code-block:: python

            # Create another point cloud with some common points
            common_points = random_points[[3, 6, 10]]  # shape (3, 3)
            other_point_cloud = PointCloud3D.from_array(common_points)

        Keeping only the points that are present in the other point cloud.

        .. code-block:: python

            # Keep only points that are present in the other point cloud
            new_point_cloud = point_cloud.keep_points(other_point_cloud)
            print(new_point_cloud.points)
            # Output: A NumPy array of shape (3, 3) with points [3, 6, 10] retained

        """
        # Check if other is a PointCloud3D instance
        if not isinstance(other, PointCloud3D):
            raise ValueError("Input must be an instance of PointCloud3D.")
        if not isinstance(inplace, bool):
            raise ValueError("inplace must be a boolean value.")
        
        # Conversion to void type for easy comparison
        dtype = numpy.dtype((numpy.void, self.points.dtype.itemsize * self.points.shape[1]))

        # Create views of the points as 1D arrays of void type
        a = numpy.ascontiguousarray(self.points).view(dtype).ravel()
        b = numpy.ascontiguousarray(other.points).view(dtype).ravel()

        # Create a mask for points in self.points that are also in other.points
        mask = numpy.isin(a, b)
        kept_points = self.points[mask]

        # Return new instance or modify in place
        if inplace:
            self.points = kept_points
            return self
        else:
            return PointCloud3D.from_array(kept_points.copy())


    def keep_points_at(self, indices: numpy.ndarray, inplace: bool = False) -> PointCloud3D:
        r"""
        Keep only the points at the specified indices in the point cloud.

        This method returns a new PointCloud3D object containing only the points at the specified indices.

        .. seealso::

            - :meth:`remove_points_at` for removing points at specified indices.
            - :meth:`keep_points` for keeping points that are present in another PointCloud3D instance.
            - :meth:`keep_points_at_inplace` for keeping points at specified indices in place.

        Parameters
        ----------
        indices : numpy.ndarray
            A 1D NumPy array of integer indices representing the points to be kept in the point cloud.

        inplace : bool, optional
            If True, modifies the current point cloud in place and returns itself. If False, returns a new PointCloud3D instance (default is False).

        Returns
        -------
        PointCloud3D
            A new PointCloud3D object containing only the points at the specified indices or the modified current instance if `inplace` is True.

        Raises
        ------
        ValueError
            If any index is out of bounds or if the input is not a 1D array of integers.

        Examples
        --------
        Create a PointCloud3D from a random NumPy array.

        .. code-block:: python

            import numpy as np
            from pysdic.geometry import PointCloud3D

            # Create a random point cloud with 100 points
            random_points = np.random.rand(100, 3)  # shape (100, 3)
            point_cloud = PointCloud3D.from_array(random_points)

        Keeping only the points at indices 0, 2, and 4.

        .. code-block:: python

            # Keep only points at indices 0, 2, and 4
            indices_to_keep = np.array([0, 2, 4])
            new_point_cloud = point_cloud.keep_points_at(indices_to_keep)
            print(new_point_cloud.points)
            # Output: A NumPy array of shape (3, 3) containing only the points at indices 0, 2, and 4

        """
        if not isinstance(inplace, bool):
            raise ValueError("inplace must be a boolean value.")
        
        # Load and validate indices
        indices = numpy.asarray(indices, dtype=int)
        if indices.ndim != 1:
            raise ValueError("Indices must be a 1D array.")
        if not numpy.issubdtype(indices.dtype, numpy.integer):
            raise ValueError("Indices must be integers.")
        if numpy.any(indices < 0) or numpy.any(indices >= self.n_points):
            raise ValueError("Indices are out of bounds.")
        
        # Return new instance or modify in place
        if inplace:
            self.points = self.points[indices]
            return self
        else:
            return PointCloud3D.from_array(self.points[indices].copy())
        

    def remove_not_finite(self, inplace: bool = False) -> PointCloud3D:
        r"""
        Remove points from the point cloud that contain non-finite values (NaN or Inf).

        This method returns a new PointCloud3D object with all points containing NaN or Inf values removed.

        .. seealso::

            - :meth:`remove_not_finite_inplace` for removing non-finite points in place.

        Parameters
        ----------
        inplace : bool, optional
            If True, modifies the current point cloud in place and returns itself. If False, returns a new PointCloud3D instance (default is False).

        Returns
        -------
        PointCloud3D
            A new PointCloud3D object containing only the finite points or the modified current instance if `inplace` is True.

            
        Examples
        --------
        Create a PointCloud3D from a NumPy array with some non-finite values.

        .. code-block:: python

            import numpy as np
            from pysdic.geometry import PointCloud3D

            # Create a point cloud with some non-finite values
            points_with_nan = np.array([[0.0, 1.0, 2.0],
                                        [np.nan, 1.0, 2.0],
                                        [3.0, np.inf, 4.0],
                                        [5.0, 6.0, 7.0]])

            point_cloud = PointCloud3D.from_array(points_with_nan)

        Removing non-finite points from the point cloud.

        .. code-block:: python

            # Remove non-finite points
            finite_point_cloud = point_cloud.remove_not_finite()
            print(finite_point_cloud.points)
            # Output: A NumPy array of shape (2, 3) containing only the finite points
        """
        if not isinstance(inplace, bool):
            raise ValueError("inplace must be a boolean value.")
        
        # Create a mask for finite points
        mask = numpy.isfinite(self.points).all(axis=1)
        finite_points = self.points[mask]

        # Return new instance or modify in place
        if inplace:
            self.points = finite_points
            return self
        else:
            return PointCloud3D.from_array(finite_points.copy())

        
    def merge(self, other: PointCloud3D, inplace: bool = False) -> PointCloud3D:
        r"""
        Merge points from another PointCloud3D instance with the current point cloud, avoiding duplicates.

        This method returns a new PointCloud3D object containing the points from both point clouds, ensuring that duplicate points are not included.
        This method removes duplicate points from the initial point clouds so the indices of the points may change.

        .. note::

            Points in the `other` point cloud that are already present in the current point cloud are ignored.

        .. seealso::

            - :meth:`concatenate` for concatenating two point clouds, including duplicates.
            - :meth:`merge_inplace` for merging points from another point cloud in place.
            - :meth:`unique` to remove duplicate points within the same point cloud.

        Parameters
        ----------
        other : PointCloud3D
            Another PointCloud3D instance containing the points to be merged with the current point cloud.

        inplace : bool, optional
            If True, modifies the current point cloud in place and returns itself. If False, returns a new PointCloud3D instance (default is False).

        Returns
        -------
        PointCloud3D
            A new PointCloud3D object containing the merged points from both point clouds, excluding duplicates or the modified current instance if `inplace` is True.

        Raises
        ------
        ValueError
            If the input is not an instance of PointCloud3D.

        Examples
        --------
        Create a PointCloud3D from a random NumPy array.

        .. code-block:: python

            import numpy as np
            from pysdic.geometry import PointCloud3D

            # Create a random point cloud with 100 points
            random_points = np.random.rand(100, 3)  # shape (100, 3)
            point_cloud = PointCloud3D.from_array(random_points)

        Create another PointCloud3D with some common points.

        .. code-block:: python

            # Create another point cloud with some common points
            common_points = random_points[[3, 6, 10]]  # shape (3, 3)
            non_common_points = np.random.rand(5, 3) + 10  # shape (5, 3), offset to avoid overlap
            other_point_cloud = PointCloud3D.from_array(np.vstack((common_points, non_common_points)))

        Merging points from the other point cloud.

        .. code-block:: python

            # Merge points from the other point cloud
            new_point_cloud = point_cloud.merge(other_point_cloud)
            print(new_point_cloud.points)
            # Output: A NumPy array of shape (102, 3) with points [3, 6, 10] ignored and 5 new points added

        """
        # Check if other is a PointCloud3D instance
        if not isinstance(other, PointCloud3D):
            raise ValueError("Input must be an instance of PointCloud3D.")
        if not isinstance(inplace, bool):
            raise ValueError("inplace must be a boolean value.")
        
        # Conversion to void type for easy comparison
        dtype = numpy.dtype((numpy.void, self.points.dtype.itemsize * self.points.shape[1]))

        # Create views of the points as 1D arrays of void type
        a = numpy.ascontiguousarray(self.points).view(dtype).ravel()
        b = numpy.ascontiguousarray(other.points).view(dtype).ravel()

        # Find unique points
        _, unique_indices = numpy.unique(a, return_index=True)
        unique_indices.sort()  # Sort indices to maintain original order
        a = a[unique_indices]

        _, unique_indices = numpy.unique(b, return_index=True)
        unique_indices.sort()  # Sort indices to maintain original order
        b = b[unique_indices]

        # Find points in other.points that are not in self.points
        mask_new = ~numpy.isin(b, a)
        unique_points = other.points[mask_new]

        # Merge points
        merged_points = numpy.vstack((self.points, unique_points))

        # Return new instance or modify in place
        if inplace:
            self.points = merged_points
            return self
        else:
            return PointCloud3D.from_array(merged_points.copy())
        

    def remove_points(self, other: PointCloud3D, inplace: bool = False) -> PointCloud3D:
        r"""
        Remove points from the current point cloud that are present in another PointCloud3D instance.

        This method returns a new PointCloud3D object with the points that are also present in the provided PointCloud3D instance removed.

        .. note::

            Points in the `other` point cloud that are not present in the current point cloud are ignored.

        .. seealso::

            - :meth:`keep_points` for keeping points that are present in another PointCloud3D instance.
            - :meth:`remove_points_at` for removing points at specified indices.
            - :meth:`remove_points_inplace` for removing points in place.

        Parameters
        ----------
        other : PointCloud3D
            Another PointCloud3D instance containing the points to be removed from the current point cloud.

        inplace : bool, optional
            If True, modifies the current point cloud in place and returns itself. If False, returns a new PointCloud3D instance (default is False). 

        Returns
        -------
        PointCloud3D
            A new PointCloud3D object with the points that are also present in the provided PointCloud3D instance removed or the modified current instance if `inplace` is True.

        Raises
        ------
        ValueError
            If the input is not an instance of PointCloud3D.

        Examples
        --------
        Create a PointCloud3D from a random NumPy array.

        .. code-block:: python

            import numpy as np
            from pysdic.geometry import PointCloud3D

            # Create a random point cloud with 100 points
            random_points = np.random.rand(100, 3)  # shape (100, 3)
            point_cloud = PointCloud3D.from_array(random_points)

        Create another PointCloud3D with some common points.

        .. code-block:: python

            # Create another point cloud with some common points
            common_points = random_points[[3, 6, 10]]  # shape (3, 3)
            other_point_cloud = PointCloud3D.from_array(common_points)

        Removing points that are present in the other point cloud.

        .. code-block:: python

            # Remove points that are present in the other point cloud
            new_point_cloud = point_cloud.remove_points(other_point_cloud)
            print(new_point_cloud.points)
            # Output: A NumPy array of shape (97, 3) with points [3, 6, 10] removed

        """
        # Check if other is a PointCloud3D instance
        if not isinstance(other, PointCloud3D):
            raise ValueError("Input must be an instance of PointCloud3D.")
        if not isinstance(inplace, bool):
            raise ValueError("inplace must be a boolean value.")
        
        # Conversion to void type for easy comparison
        dtype = numpy.dtype((numpy.void, self.points.dtype.itemsize * self.points.shape[1]))

        # Create views of the points as 1D arrays of void type
        a = numpy.ascontiguousarray(self.points).view(dtype).ravel()
        b = numpy.ascontiguousarray(other.points).view(dtype).ravel()

        # Create a mask for points in self.points that are not in other.points
        mask = ~numpy.isin(a, b)
        remaining_points = self.points[mask]

        # Return new instance or modify in place
        if inplace:
            self.points = remaining_points
            return self
        else:
            return PointCloud3D.from_array(remaining_points.copy())


    def remove_points_at(self, indices: numpy.ndarray, inplace: bool = False) -> PointCloud3D:
        r"""
        Remove points from the point cloud based on their indices.

        This method returns a new PointCloud3D object with the points at the specified indices removed.

        .. seealso::

            - :meth:`keep_points_at` for keeping points at specified indices.
            - :meth:`remove_points` for removing points that are present in another PointCloud3D instance.
            - :meth:`remove_points_at_inplace` for removing points at specified indices in place.

        Parameters
        ----------
        indices : numpy.ndarray
            A 1D NumPy array of integer indices representing the points to be removed from the point cloud.

        inplace : bool, optional
            If True, modifies the current point cloud in place and returns itself. If False, returns a new PointCloud3D instance (default is False).

        Returns
        -------
        PointCloud3D
            A new PointCloud3D object with the points at the specified indices removed or the modified current instance if `inplace` is True.

        Raises
        ------
        ValueError
            If any index is out of bounds or if the input is not a 1D array of integers.

        Examples
        --------
        Create a PointCloud3D from a random NumPy array.

        .. code-block:: python

            import numpy as np
            from pysdic.geometry import PointCloud3D

            # Create a random point cloud with 100 points
            random_points = np.random.rand(100, 3)  # shape (100, 3)
            point_cloud = PointCloud3D.from_array(random_points)

        Removing points at indices 1 and 3.

        .. code-block:: python

            # Remove points at indices 1 and 3
            indices_to_remove = np.array([1, 3])
            new_point_cloud = point_cloud.remove_points_at(indices_to_remove)
            print(new_point_cloud.points)
            # Output: A NumPy array of shape (98, 3) with points at indices 1 and 3 removed

        """
        if not isinstance(inplace, bool):
            raise ValueError("inplace must be a boolean value.")
        
        # Load and validate indices
        indices = numpy.asarray(indices, dtype=int)
        if indices.ndim != 1:
            raise ValueError("Indices must be a 1D array.")
        if not numpy.issubdtype(indices.dtype, numpy.integer):
            raise ValueError("Indices must be integers.")
        if numpy.any(indices < 0) or numpy.any(indices >= self.n_points):
            raise ValueError("Indices are out of bounds.")
        
        # Select points to keep
        mask = numpy.ones(self.n_points, dtype=bool)
        mask[indices] = False
        remaining_points = self.points[mask]

        # Return new instance or modify in place
        if inplace:
            self.points = remaining_points
            return self
        else:
            return PointCloud3D.from_array(remaining_points.copy())


    def unique(self, inplace: bool = False) -> PointCloud3D:
        r"""
        Remove duplicate points from the point cloud.

        This method returns a new PointCloud3D object containing only unique points, with duplicates removed.

        .. seealso::

            - :meth:`merge` for merging two point clouds while avoiding duplicates.
            - :meth:`unique_inplace` for removing duplicate points in place.

        Parameters
        ----------
        inplace : bool, optional
            If True, modifies the current point cloud in place and returns itself. If False, returns a new PointCloud3D instance (default is False).

        Returns
        -------
        PointCloud3D
            A new PointCloud3D object containing only unique points or the modified current instance if `inplace` is True.

        Examples
        --------
        Create a PointCloud3D from a NumPy array with duplicate points.

        .. code-block:: python

            import numpy as np
            from pysdic.geometry import PointCloud3D

            # Create a point cloud with duplicate points
            points_with_duplicates = np.array([[0, 0, 0],
                                               [1, 1, 1],
                                               [0, 0, 0],  # Duplicate
                                               [2, 2, 2],
                                               [1, 1, 1]]) # Duplicate
            point_cloud = PointCloud3D.from_array(points_with_duplicates)

        Removing duplicate points using the `unique` method.

        .. code-block:: python

            unique_point_cloud = point_cloud.unique()
            print(unique_point_cloud.points)
            # Output: A NumPy array of shape (3, 3) with unique points [[0, 0, 0], [1, 1, 1], [2, 2, 2]]

        """
        if not isinstance(inplace, bool):
            raise ValueError("inplace must be a boolean value.")
        
        # Conversion to void type for easy comparison
        dtype = numpy.dtype((numpy.void, self.points.dtype.itemsize * self.points.shape[1]))

        # Create a view of the points as a 1D array of void type
        a = numpy.ascontiguousarray(self.points).view(dtype).ravel()

        # Find unique points
        _, unique_indices = numpy.unique(a, return_index=True)
        unique_indices.sort()  # Sort indices to maintain original order
        unique_points = self.points[unique_indices]

        # Return new instance or modify in place
        if inplace:
            self.points = unique_points
            return self
        else:
            return PointCloud3D.from_array(unique_points.copy())


    # ==============
    # Geometric Computations
    # ==============

    def bounding_box(self) -> Tuple[numpy.ndarray, numpy.ndarray]:
        r"""
        Compute the axis-aligned bounding box of the point cloud.

        The bounding box is defined by the minimum and maximum coordinates along each axis (x, y, z).

        .. note::

            The non-finite values (NaN, Inf) are ignored in the computation. If the point cloud is empty or contains only non-finite values, a ValueError is raised.

        Returns
        -------
        numpy.ndarray
            The minimum coordinates of the bounding box as a NumPy array of shape (3,) representing (min_x, min_y, min_z).

        numpy.ndarray
            The maximum coordinates of the bounding box as a NumPy array of shape (3,) representing (max_x, max_y, max_z).

        Raises
        ------
        ValueError
            If the point cloud is empty or contains only non-finite values.

        Examples
        --------
        Create a tetrahedron point cloud and compute its bounding box.

        .. code-block:: python

            import numpy as np
            from pysdic.geometry import PointCloud3D

            # Create a tetrahedron point cloud
            tetrahedron_points = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]])  # shape (4, 3)
            point_cloud = PointCloud3D.from_array(tetrahedron_points)

        Compute the bounding box using the `bounding_box` method.

        .. code-block:: python

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

        finite_points = self.points[numpy.all(numpy.isfinite(self.points), axis=1), :]
        min_coords = numpy.min(finite_points, axis=0)
        max_coords = numpy.max(finite_points, axis=0)

        if not numpy.all(numpy.isfinite(min_coords)) or not numpy.all(numpy.isfinite(max_coords)):
            raise ValueError("Point cloud contains only non-finite values; cannot compute bounding box.")

        return min_coords, max_coords


    def bounding_sphere(self) -> Tuple[numpy.ndarray, float]:
        r"""
        Compute the bounding sphere of the point cloud.

        The bounding sphere is defined by its center and radius, which encompasses all points in the point cloud.

        .. note::

            The non-finite values (NaN, Inf) are ignored in the computation. If the point cloud is empty or contains only non-finite values, a ValueError is raised.

        Returns
        -------
        numpy.ndarray
            The center of the bounding sphere as a NumPy array of shape (3,).

        float
            The radius of the bounding sphere.

        Raises
        ------
        ValueError
            If the point cloud is empty or contains only non-finite values.

        
        Examples
        --------
        Create a tetrahedron point cloud and compute its bounding sphere.

        .. code-block:: python

            import numpy as np
            from pysdic.geometry import PointCloud3D

            # Create a tetrahedron point cloud
            tetrahedron_points = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]])  # shape (4, 3)
            point_cloud = PointCloud3D.from_array(tetrahedron_points)

        Compute the bounding sphere using the `bounding_sphere` method.

        .. code-block:: python

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
        
        finite_points = self.points[numpy.all(numpy.isfinite(self.points), axis=1), :]

        # Compute center and radius
        center = numpy.mean(finite_points, axis=0) # Shape (3,)
        radius = numpy.max(numpy.linalg.norm(finite_points - center, axis=1))   

        if not numpy.all(numpy.isfinite(center)) or not numpy.isfinite(radius):
            raise ValueError("Point cloud contains only non-finite values; cannot compute bounding sphere.")
        
        return center, radius
    


    # ==============
    # Visualization
    # ==============
    def visualize(
            self, 
            color: str = "black",
            point_size: float = 1.0,
            opacity: float = 1.0,
            title: Optional[str] = None,
            show_axes: bool = True,
            show_grid: bool = True,
            ) -> None:
        r"""
        Visualize the point cloud using PyVista.

        Only finite points (not NaN or Inf) are visualized.  

        .. seealso::

            - `PyVista Documentation <https://docs.pyvista.org>`_ for more details on visualization options.

        Parameters
        ----------
        color : str, optional
            The color of the points in the visualization. Default is "black".

        point_size : float, optional
            The size of the points in the visualization. Default is 1.0.

        opacity : float, optional
            The opacity of the points in the visualization. Default is 1.0 (fully opaque).

        title : Optional[str], optional
            The title of the visualization window. Default is None.

        show_axes : bool, optional
            Whether to display the axes in the visualization. Default is True.
        
        show_grid : bool, optional
            Whether to display the grid in the visualization. Default is True.

        Examples
        --------
        Create a PointCloud3D from a random NumPy array.

        .. code-block:: python

            # Create a random point cloud with 100 points
            points_array = np.random.rand(100, 3)
            point_cloud = PointCloud3D.from_array(points_array)

        Visualize the point cloud using the `visualize` method.

        .. code-block:: python

            point_cloud.visualize(color='red', point_size=10)

        This will open a PyVista window displaying the point cloud with red points of size 10.

        .. figure:: ../../../pysdic/resources/point_cloud_3d_visualize_example.png
            :width: 600
            :align: center

            Example of a 3D point cloud visualization using the `visualize` method.
            
        """
        # Check empty point cloud
        if self.n_points == 0:
            raise ValueError("Cannot visualize an empty point cloud.")
        
        # Validate parameters
        if not isinstance(color, str):
            raise ValueError("Color must be a string.")
        if not (isinstance(point_size, Number) and point_size > 0):
            raise ValueError("Point size must be a positive number.")
        
        if not (isinstance(opacity, Number) and 0.0 <= opacity <= 1.0):
            raise ValueError("Opacity must be a number between 0.0 and 1.0.")
        
        if title is not None and not isinstance(title, str):
            raise ValueError("Title must be a string or None.")
        if not isinstance(show_axes, bool):
            raise ValueError("show_axes must be a boolean value.")
        if not isinstance(show_grid, bool):
            raise ValueError("show_grid must be a boolean value.")

        # Create a PyVista point cloud
        valid_points = self.points[numpy.all(numpy.isfinite(self.points), axis=1)]
        if valid_points.shape[0] == 0:
            raise ValueError("Point cloud contains only non-finite values; cannot visualize.")
        
        pv_point_cloud = pyvista.PolyData(valid_points)
        plotter = pyvista.Plotter()
        plotter.add_mesh(pv_point_cloud, color=color, point_size=float(point_size), render_points_as_spheres=True, opacity=opacity)

        if title is not None:
            plotter.add_title(title)
        if show_axes:
            plotter.show_axes()
        if show_grid:
            plotter.show_grid()
        plotter.show()
