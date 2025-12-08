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

from typing import Optional, Tuple, Union
from numbers import Number, Integral
from py3dframe import Frame, FrameTransform

import numpy
import pyvista
import meshio
import os


class PointCloud(object):
    r"""
    A class representing a :math:`E`-dimensional point cloud.

    This class is designed to handle :math:`E`-dimensional point clouds, which are collections of points in a :math:`E`-dimensional space.

    .. note::

        The dimension :math:`E` of the cloud is not designed to change directly after initialization.

    Parameters
    ----------
    points: :class:`numpy.ndarray`
        A NumPy array of shape (:math:`N_{p}`, :math:`E`) representing the coordinates of the points.

    n_dimensions: :class:`int`, optional
        The dimension :math:`E` of the point cloud. If not provided, it is inferred from the shape of the `points` array.
        If :obj:`points` second dimension is less than :obj:`n_dimensions`, the points will be reshaped accordingly and filled with zeros.
        If :obj:`points` second dimension is greater than :obj:`n_dimensions`, a ValueError will be raised.

    Raises  
    ------
    ValueError
        If the input array does not have the correct shape.

    """
    __slots__ = [
        "_points"
    ]

    def __init__(self, points: numpy.ndarray, n_dimensions: Optional[int] = None) -> None:
        # Deals with points
        points = numpy.asarray(points, dtype=numpy.float64)
        if points.ndim == 1:
            points = points.reshape((-1, 1))
        if not points.ndim == 2:
            raise ValueError("Points must be a 2D NumPy array with shape (N_p, E).")
        if n_dimensions is None:
            n_dimensions = points.shape[1]
        if not isinstance(n_dimensions, Integral) or n_dimensions <= 0:
            raise ValueError("n_dimensions must be a strictly positive integer.")
        if points.shape[1] < n_dimensions:
            # Pad with zeros
            padded_points = numpy.zeros((points.shape[0], n_dimensions), dtype=points.dtype)
            padded_points[:, :points.shape[1]] = points
            points = padded_points
        elif points.shape[1] > n_dimensions:
            raise ValueError("Points have more dimensions than n_dimensions.")

        # Save points
        self._points = points

    # ==========================
    # Properties
    # ==========================
    @property
    def points(self) -> numpy.ndarray:
        r"""
        [Get or Set] An numpy array of shape (:math:`N_p`, :math:`E`) representing the coordinates of the points in the cloud.

        .. note::

            This property is settable.

        .. note::

            An alias for ``points`` is ``coordinates``.

        Access and modify the points of the point cloud.

        Parameters
        ----------
        value : :class:`numpy.ndarray`
            An array-like of shape (:math:`N_p`, :math:`E`) representing the coordinates of the points in the cloud.

        Returns
        -------
        :class:`numpy.ndarray`
            A NumPy array of shape (:math:`N_p`, :math:`E`) containing the coordinates of the points in the cloud.

        Raises
        ------
        ValueError
            If the input array does not have the correct shape (:math:`N_p`, :math:`E`).

        Examples
        --------
        Lets create a basic point random point cloud.

        .. code-block:: python

            import numpy as np
            from pysdic import PointCloud

            # Create a random point cloud with 100 points
            random_points = np.random.rand(100, 3)  # shape (100, 3)
            point_cloud = PointCloud(points=random_points)

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
        if self.n_dimensions == 1 and points.ndim == 1:
            points = points.reshape((-1, 1))
        if not (points.ndim == 2 and points.shape[1] == self.n_dimensions):
            raise ValueError(f"Points must be a 2D NumPy array with shape (N_p, {self.n_dimensions}).")
        self._points = points

    @property
    def coordinates(self) -> numpy.ndarray:
        r"""
        [Get or Set] Alias for :meth:`points` property.
        """
        return self.points

    @coordinates.setter
    def coordinates(self, value: numpy.ndarray) -> None:
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

        Examples
        --------
        Lets create a basic point random point cloud.

        .. code-block:: python

            import numpy as np
            from pysdic import PointCloud

            # Create a random point cloud with 100 points
            random_points = np.random.rand(100, 3)  # shape (100, 3)
            point_cloud = PointCloud(points=random_points)

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
        [Get] The shape of the points array (:math:`N_p`, :math:`E`).

        .. seealso::

            - :meth:`n_points` for getting the number of points in the point cloud.
            - :meth:`n_dimensions` for getting the dimension of the point cloud.

        Returns
        -------
        tuple[:class:`int`, :class:`int`]
            A tuple representing the shape of the points array (:math:`N_p`, :math:`E`).

        Examples
        --------
        Lets create a basic point random point cloud.

        .. code-block:: python

            import numpy as np
            from pysdic import PointCloud

            # Create a random point cloud with 100 points
            random_points = np.random.rand(100, 3)  # shape (100, 3)
            point_cloud = PointCloud(points=random_points)

        The shape of the points array can be accessed using the `shape` property.

        .. code-block:: python

            # Access the shape of the points array
            points_shape = point_cloud.shape
            print(points_shape)
            # Output: (100, 3)

        """
        return self.points.shape
    
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

        Examples
        --------
        Lets create a basic point random point cloud.

        .. code-block:: python

            import numpy as np
            from pysdic import PointCloud

            # Create a random point cloud with 100 points
            random_points = np.random.rand(100, 3)  # shape (100, 3)
            point_cloud = PointCloud(points=random_points)

        The dimension of the point cloud can be accessed using the `n_dimensions` property.

        .. code-block:: python

            # Access the dimension of the point cloud
            dimension = point_cloud.n_dimensions
            print(dimension)
            # Output: 3

        """
        return self.points.shape[1]
    
    # ==========================
    # Class methods
    # ==========================
    @classmethod
    def from_array(cls, points: numpy.ndarray) -> PointCloud:
        r"""
        Create a :class:`PointCloud` object from a NumPy array of shape (:math:`N_p`, :math:`E`).

        .. seealso::

            - :meth:`to_array` method for converting the point cloud back to a NumPy array.

        Parameters
        ----------
        points : :class:`numpy.ndarray`
            A NumPy array of shape (:math:`N_p`, :math:`E`) representing the coordinates of the points.

        Returns
        -------
        :class:`PointCloud`
            A :class:`PointCloud` object containing the provided points.

        Raises
        ------
        ValueError
            If the input array does not have the correct shape (:math:`N_p`, :math:`E`).

        Examples
        --------
        Creating a :class:`PointCloud` object from a random NumPy array.

        .. code-block:: python

            import numpy as np
            from pysdic import PointCloud

            # Create a random point cloud with 100 points
            random_points = np.random.rand(100, 3)  # shape (100, 3)
            point_cloud = PointCloud.from_array(random_points)

        Now, ``point_cloud`` is a :class:`PointCloud` object containing the points from the NumPy array.

        """
        return cls(points=points)
    

    def to_array(self) -> numpy.ndarray:
        r"""
        Convert the point cloud to a NumPy array of shape (:math:`N_p`, :math:`E`).

        .. note::

            The returned array is a copy of the internal points array. Modifying it will not affect the original point cloud.

        .. seealso::

            - :meth:`points` property for accessing and modifying the points of the point cloud.
            - :meth:`from_array` class method for creating a PointCloud object from a NumPy array.

        Returns
        -------
        :class:`numpy.ndarray`
            A NumPy array of shape (:math:`N_p`, :math:`E`) containing the coordinates of the points in the cloud.

            
        Examples
        --------
        Creating a :class:`PointCloud` object from a random NumPy array.

        .. code-block:: python

            import numpy as np
            from pysdic import PointCloud

            # Create a random point cloud with 100 points
            random_points = np.random.rand(100, 3)  # shape (100, 3)
            point_cloud = PointCloud.from_array(random_points)

        Convert back to a NumPy array using the `to_array` method.

        .. code-block:: python

            # Convert the point cloud back to a NumPy array
            points_array = point_cloud.to_array()
            print(points_array)
            # Output: A NumPy array of shape (100, 3) containing the coordinates
        
        """
        return self.points.copy()
 

    @classmethod
    def from_meshio(cls, mesh: meshio.Mesh) -> PointCloud:
        r"""
        Create a :class:`PointCloud` object from a :class:`meshio.Mesh` object.

        The points are extracted from the mesh vertices.

        .. seealso::

            - :meth:`to_array` method for converting the point cloud back to a NumPy array.

        .. note::

            Only the vertices of the mesh are used to create the point cloud. Faces and other elements are ignored.

        Parameters
        ----------
        mesh: :class:`meshio.Mesh`
            A meshio Mesh object containing the vertices.

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
        
        # Extract points from mesh vertices
        if mesh.points is None or mesh.points.shape[0] == 0:
            raise ValueError("The mesh does not contain any points.")
        vertices = mesh.points.copy()
        return cls.from_array(vertices)
    

    def to_meshio(self) -> meshio.Mesh:
        r"""
        Convert the point cloud to a :class:`meshio.Mesh` object.

        The points are stored as vertices in the mesh.

        .. seealso::

            - :meth:`from_meshio` class method for creating a :class:`PointCloud` object from a :class:`meshio.Mesh` object.

        Returns
        -------
        :class:`meshio.Mesh`
            A :class:`meshio.Mesh` object containing the points as vertices.

        Examples
        --------

        Converting a :class:`PointCloud` object to a :class:`meshio.Mesh` object.
    
        .. code-block:: python

            import numpy as np
            from pysdic import PointCloud

            # Create a random point cloud with 100 points
            random_points = np.random.rand(100, 3)  # shape (100, 3)
            point_cloud = PointCloud.from_array(random_points)

            # Convert the point cloud to a meshio Mesh
            mesh = point_cloud.to_meshio()

        Now, ``mesh`` is a :class:`meshio.Mesh` object containing the points of the point cloud as vertices.
        """
        return meshio.Mesh(points=self.points.copy(), cells=[])
    

    @classmethod
    def from_npz(cls, filepath: str) -> PointCloud:
        r"""
        Create a :class:`PointCloud` object from a NPZ file.

        The points are read using ``numpy.load``.

        The NPZ file should contain an array named 'points' with shape (:math:`N_p`, :math:`E`).

        .. seealso::

            - :meth:`to_npz` method for saving the point cloud to a NPZ file.

        Parameters
        ----------
        filepath : :class:`str`
            The path to the NPZ file.

        Returns
        -------
        :class:`PointCloud`
            A :class:`PointCloud` object containing the points read from the NPZ file.

            
        Examples
        --------
        Creating a :class:`PointCloud` object from a NPZ file.

        .. code-block:: python

            from pysdic import PointCloud

            # Create a point cloud from a NPZ file
            point_cloud = PointCloud.from_npz('path/to/point_cloud.npz')

        Now, ``point_cloud`` is a :class:`PointCloud` object containing the points read from the specified NPZ file.

        """
        path = os.path.expanduser(filepath)
        if not os.path.isfile(path) or not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        data = numpy.load(filepath)
        if 'points' not in data:
            raise ValueError("NPZ file must contain an array named 'points'.")
        
        points = data['points']
        return cls.from_array(points)
    

    def to_npz(self, filepath: str) -> None:
        r"""
        Save the point cloud to a NPZ file.

        The points are saved using ``numpy.savez``.

        The NPZ file will contain an array named 'points' with shape (:math:`N_p`, :math:`E`).

        .. seealso::

            - :meth:`from_npz` method for creating a PointCloud object from a NPZ file.

        Parameters
        ----------
        filepath : :class:`str`
            The path to the output NPZ file.

            
        Examples
        --------
        Saving a :class:`PointCloud` object to a NPZ file.

        .. code-block:: python

            from pysdic import PointCloud
            import numpy as np

            # Create a random point cloud with 100 points
            random_points = np.random.rand(100, 3)  # shape (100, 3)
            point_cloud = PointCloud.from_array(random_points)

            # Save the point cloud to a NPZ file
            point_cloud.to_npz('path/to/output_point_cloud.npz')

        This will save the points of the point cloud to the specified NPZ file.

        """
        path = os.path.expanduser(filepath)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        numpy.savez(path, points=self.points)
    

    @classmethod
    def from_ply(cls, filepath: str) -> PointCloud:
        r"""
        Create a :class:`PointCloud` object from a PLY file (only for :math:`E=3` point clouds).

        The PLY file should contain vertex definitions.
        Only vertices are read; faces and other elements are ignored.

        The points are extracted using ``meshio`` library.

        .. seealso::

            - :meth:`to_ply` method for saving the point cloud to a PLY file.

        .. warning::

            This method only works for 3-dimensional point clouds, as PLY format is primarily used for 3D data.

        Parameters
        ----------
        filepath : :class:`str`
            The path to the PLY file.

        Returns
        -------
        :class:`PointCloud`
            A :class:`PointCloud` object containing the points read from the PLY file.


        Examples
        --------
        Creating a :class:`PointCloud` object from a PLY file.

        .. code-block:: python

            from pysdic import PointCloud
            # Create a point cloud from a PLY file
            point_cloud = PointCloud.from_ply('path/to/point_cloud.ply')

        Now, ``point_cloud`` is a :class:`PointCloud` object containing the points read from the specified PLY file.
        """
        path = os.path.expanduser(filepath)
        if not os.path.isfile(path) or not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        mesh = meshio.read(filepath, file_format='ply')
        mesh = cls.from_meshio(mesh)
        if mesh.n_dimensions != 3:
            raise ValueError("PLY format only supports 3D point clouds.")
        return mesh
    

    def to_ply(self, filepath: str, binary: bool = False) -> None:
        r"""
        Save the point cloud to a PLY file (only for :math:`E=3` point clouds).

        The points are saved using ``meshio`` library.

        The PLY file will contain vertex definitions.

        .. seealso::

            - :meth:`from_ply` method for creating a :class:`PointCloud` object from a PLY file.

        .. warning::

            This method only works for 3-dimensional point clouds, as PLY format is primarily used for 3D data.

        Parameters
        ----------
        filepath : :class:`str`
            The path to the output PLY file.

        binary : :class:`bool`, optional
            If :obj:`True`, the PLY file will be saved in binary format. Default is :obj:`False`.
 
            
        Examples
        --------
        Saving a :class:`PointCloud` object to a PLY file.

        .. code-block:: python

            from pysdic import PointCloud
            import numpy as np

            # Create a random point cloud with 100 points
            random_points = np.random.rand(100, 3)  # shape (100, 3)
            point_cloud = PointCloud.from_array(random_points)

            # Save the point cloud to a PLY file
            point_cloud.to_ply('path/to/output_point_cloud.ply')

        This will save the points of the point cloud to the specified PLY file.
        """
        path = os.path.expanduser(filepath)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if not isinstance(binary, bool):
            raise ValueError("The 'binary' parameter must be a boolean value.")
        
        if self.n_dimensions != 3:
            raise ValueError("PLY format only supports 3D point clouds.")

        mesh = self.to_meshio()
        meshio.write(filepath, mesh, file_format='ply', binary=binary)


    @classmethod
    def from_vtk(cls, filepath: str) -> PointCloud:
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
        filepath : :class:`str`
            The path to the VTK file.


        Returns
        -------
        :class:`PointCloud`
            A :class:`PointCloud` object containing the points read from the VTK file.

        
        Examples
        --------
        Creating a :class:`PointCloud` object from a VTK file.
        
        .. code-block:: python

            from pysdic import PointCloud
            # Create a point cloud from a VTK file
            point_cloud = PointCloud.from_vtk('path/to/point_cloud.vtk')

        Now, ``point_cloud`` is a :class:`PointCloud` object containing the points read from the specified VTK file.

        """
        path = os.path.expanduser(filepath)

        if not os.path.isfile(path) or not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        mesh = meshio.read(filepath, file_format='vtk')
        mesh = cls.from_meshio(mesh)
        if mesh.n_dimensions != 3:
            raise ValueError("VTK format only supports 3D point clouds.")
        return mesh


    def to_vtk(self, filepath: str, binary: bool = False, only_finite: bool = False) -> None:
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
        filepath : :class:`str`
            The path to the output VTK file.

        binary : :class:`bool`, optional
            If :obj:`True`, the VTK file will be saved in binary format. Default is :obj:`False`.

        only_finite : :class:`bool`, optional
            If :obj:`True`, only finite points (excluding NaN and infinite values) will be saved. Default is :obj:`False`.

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
        points_to_save = self.points
        if only_finite:
            finite_mask = numpy.isfinite(self.points).all(axis=1)
            points_to_save = self.points[finite_mask]
            if points_to_save.shape[0] == 0:
                raise ValueError("No finite points to save in the point cloud.")

        else:
            if not numpy.isfinite(self.points).all():
                raise ValueError("Point cloud contains NaN or infinite values. Consider using 'only_finite=True' to filter them out before saving.")

        mesh = meshio.Mesh(points=points_to_save, cells=[("vertex", numpy.arange(points_to_save.shape[0]).reshape((-1, 1)))])
        meshio.write(filepath, mesh, file_format='vtk', binary=binary)


    @classmethod
    def from_obj(cls, filepath: str) -> PointCloud:
        r"""
        Create a :class:`PointCloud` object from a OBJ file (only for :math:`E=3` point clouds).

        The OBJ file should contain vertex definitions starting with the letter 'v'.
        Only vertices are read; faces and other elements are ignored.

        The points are extracted using ``meshio`` library.

        .. seealso::

            - :meth:`to_obj` method for saving the point cloud to a OBJ file.

        .. warning::

            This method only works for 3-dimensional point clouds, as OBJ format is primarily used for 3D data.

        Parameters
        ----------
        filepath : :class:`str`
            The path to the OBJ file.

        Returns
        -------
        :class:`PointCloud`
            A :class:`PointCloud` object containing the points read from the OBJ file.

        
        Examples
        --------
        Creating a :class:`PointCloud` object from a OBJ file.

        .. code-block:: python

            from pysdic import PointCloud
            # Create a point cloud from a OBJ file
            point_cloud = PointCloud.from_obj('path/to/point_cloud.obj')

        Now, ``point_cloud`` is a :class:`PointCloud` object containing the points read from the specified OBJ file.
        """
        path = os.path.expanduser(filepath)
        if not os.path.isfile(path) or not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        mesh = meshio.read(filepath, file_format='obj')
        mesh = cls.from_meshio(mesh)
        if mesh.n_dimensions != 3:
            raise ValueError("OBJ format only supports 3D point clouds.")
        return mesh
    

    def to_obj(self, filepath: str) -> None:
        r"""
        Save the point cloud to a OBJ file.

        The points are saved using ``meshio`` library.

        The OBJ file will contain vertex definitions starting with the letter 'v'.

        .. seealso::

            - :meth:`from_obj` method for creating a :class:`PointCloud` object from a OBJ file.

        .. warning::

            This method only works for 3-dimensional point clouds, as OBJ format is primarily used for 3D data.

        Parameters
        ----------
        filepath : :class:`str`
            The path to the output OBJ file.

            
        Examples
        --------
        Saving a :class:`PointCloud` object to a OBJ file.

        .. code-block:: python

            from pysdic import PointCloud
            import numpy as np

            # Create a random point cloud with 100 points
            random_points = np.random.rand(100, 3)  # shape (100, 3)
            point_cloud = PointCloud.from_array(random_points)

            # Save the point cloud to a OBJ file
            point_cloud.to_obj('path/to/output_point_cloud.obj')

        This will save the points of the point cloud to the specified OBJ file.
        """
        path = os.path.expanduser(filepath)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if self.n_dimensions != 3:
            raise ValueError("OBJ format only supports 3D point clouds.")
        
        mesh = self.to_meshio()
        meshio.write(filepath, mesh, file_format='obj')


    @classmethod
    def from_xyz(cls, filepath: str, delimiter: str = ' ') -> PointCloud:
        r"""
        Create a :class:`PointCloud` object from a XYZ file.

        The points are read using ``numpy.loadtxt``.

        The XYZ file should contain :math:`E` columns representing the (x, y, z, ...) coordinates of the points. 
        Lines starting with `#` or `//` are treated as comments and ignored.

        .. seealso::

            - :meth:`to_xyz` method for saving the point cloud to a XYZ file.

        Parameters
        ----------
        filepath : :class:`str`
            The path to the XYZ file.

        delimiter : :class:`str`, optional
            The delimiter used in the XYZ file (default is space).

        Returns
        -------
        :class:`PointCloud`
            A :class:`PointCloud` object containing the points read from the XYZ file.

            
        Examples
        --------
        Creating a :class:`PointCloud` object from a XYZ file.

        .. code-block:: python

            from pysdic import PointCloud

            # Create a point cloud from a XYZ file
            point_cloud = PointCloud.from_xyz('path/to/point_cloud.xyz', delimiter=',')

        Now, ``point_cloud`` is a :class:`PointCloud` object containing the points read from the specified XYZ file.

        """
        path = os.path.expanduser(filepath)
        if not os.path.isfile(path) or not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        points = numpy.loadtxt(filepath, delimiter=delimiter, comments=['#', '//'])
        return cls.from_array(points)
    

    def to_xyz(self, filepath: str, delimiter: str = ' ') -> None:
        r"""
        Save the point cloud to a XYZ file.

        The points are saved using ``numpy.savetxt``.

        The XYZ file will contain three columns representing the (x, y, z, ...) coordinates of the points.

        .. seealso::

            - :meth:`from_xyz` method for creating a PointCloud object from a XYZ file.

        Parameters
        ----------
        filepath : str
            The path to the output XYZ file.
        delimiter : str, optional
            The delimiter to use in the XYZ file (default is space).

        Examples
        --------
        Saving a PointCloud object to a XYZ file.

        .. code-block:: python

            from pysdic import PointCloud

            # Create a random point cloud with 100 points
            random_points = np.random.rand(100, 3)  # shape (100, 3)
            point_cloud = PointCloud.from_array(random_points)

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

    def __add__(self, other: PointCloud) -> PointCloud:
        return self.concatenate(other)
    
    def __iadd__(self, other: PointCloud) -> PointCloud:
        self.concatenate(other, inplace=True)
        return self
    

    # ==========================
    # Methods Geometry Manipulation
    # ==========================
    def all_close(self, other: Union[PointCloud, numpy.ndarray], rtol: float = 1e-05, atol: float = 1e-08, nan_equal: bool = True, ordered: bool = True, ) -> bool:
        r"""
        Check if all points in the current point cloud are approximately equal to the points in another :class:`PointCloud` instance within a tolerance.

        This method compares the points of the current point cloud with those of another :class:`PointCloud` instance and returns True if all corresponding points are approximately equal within the specified relative and absolute tolerances.

        .. note::

            If the number of points in both point clouds differ, the method will return False.

        .. seealso::

            - :func:`numpy.allclose` for more details on the comparison.

        Parameters
        ----------
        other : Union[:class:`PointCloud`, :class:`numpy.ndarray`]
            Another :class:`PointCloud` instance or a NumPy array with shape (n_points, 3) to compare with the current point cloud.

        rtol : :class:`float`, optional
            The relative tolerance parameter (default is :obj:`1e-05`).

        atol : :class:`float`, optional
            The absolute tolerance parameter (default is :obj:`1e-08`).

        nan_equal : :class:`bool`, optional
            If True, NaN values are considered equal (default is :obj:`True`).

        ordered : :class:`bool`, optional
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
            
            import numpy as np
            from pysdic import PointCloud

            # Create a random point cloud with 100 points
            random_points = np.random.rand(100, 3)  # shape (100, 3)
            point_cloud1 = PointCloud.from_array(random_points)

        Compare the point cloud with another point cloud that is slightly modified.

        .. code-block:: python

            # Create a second point cloud by adding small noise to the first one
            noise = np.random.normal(scale=1e-8, size=random_points.shape)
            point_cloud2 = PointCloud.from_array(random_points + noise)

            # Check if the two point clouds are approximately equal
            are_close = point_cloud1.all_close(point_cloud2, rtol=1e-5, atol=1e-8)
            print(are_close)
            # Output: True (most likely, depending on the noise)

        Compare with a point cloud that is significantly different.

        .. code-block:: python

            # Create a third point cloud that is significantly different
            different_points = np.random.rand(100, 3) + 1.0  # Shifted by 1.0
            point_cloud3 = PointCloud.from_array(different_points)
            are_close = point_cloud1.all_close(point_cloud3, rtol=1e-5, atol=1e-8)
            print(are_close)
            # Output: False

        """
        if not isinstance(rtol, Number):
            raise ValueError("rtol must be a numeric value.")
        if not isinstance(atol, Number):
            raise ValueError("atol must be a numeric value.")
        if not isinstance(nan_equal, bool):
            raise ValueError("nan_equal must be a boolean value.")
        if not isinstance(ordered, bool):
            raise ValueError("ordered must be a boolean value.")

        if isinstance(other, PointCloud):
            other_points = other.points
        else:
            other_points = numpy.asarray(other, dtype=numpy.float64)
            if not (other_points.ndim == 2 and other_points.shape[1] == 3):
                raise ValueError(f" PointCloud.all_close: Input must be a other PointCloud instance or a 2D NumPy array with shape (N, 3), got shape {other_points.shape}.")

        if self.points.shape != other_points.shape:
            return False
        
        if ordered:
            return numpy.allclose(self.points, other_points, rtol=rtol, atol=atol, equal_nan=nan_equal)
        else:
            # argsort rows lexicographically
            self_sorted = self.points[numpy.lexsort(self.points.T[::-1])]
            other_sorted = other_points[numpy.lexsort(other_points.T[::-1])]

            # Compare the sorted views
            return numpy.allclose(self_sorted, other_sorted, rtol=rtol, atol=atol, equal_nan=nan_equal)
    

    def all_finite(self) -> bool:
        r"""
        Check if all points in the point cloud are finite (i.e., not NaN or infinite).

        Returns
        -------
        bool
            True if all points are finite, False otherwise.

        Examples
        --------
        Creating a :class:`PointCloud` object and checking if all points are finite.

        .. code-block:: python

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


    def concatenate(self, other: PointCloud, inplace: bool = False) -> PointCloud:
        r"""
        Concatenate the current point cloud with another :class:`PointCloud` instance.

        This method combines the points from both point clouds into a new :class:`PointCloud` object.

        .. note::

            This method is functionally equivalent to using the `+` operator.

        .. seealso::

            - :meth:`merge` for merging points from another point cloud avoiding duplicates.

        Parameters
        ----------
        other : :class:`PointCloud`
            Another :class:`PointCloud` instance to concatenate with the current point cloud.

        inplace : :class:`bool`, optional
            If :obj:`True`, modifies the current point cloud in place and returns itself. If :obj:`False`, returns a new :class:`PointCloud` instance (default is :obj:`False`).

        Returns
        -------
        :class:`PointCloud`
            A new :class:`PointCloud` object containing the concatenated points from both point clouds or the modified current instance if :obj:`inplace` is True.
        Raises
        ------
        ValueError
            If the input is not an instance of :class:`PointCloud`.
            If the two point clouds have different dimensions.

        Examples
        --------
        Creating two :class:`PointCloud` objects.

        .. code-block:: python

            import numpy as np
            from pysdic import PointCloud

            # Create two random NumPy arrays of shape (100, 3)
            random_points1 = np.random.rand(100, 3)  # shape (100, 3)
            random_points2 = np.random.rand(50, 3)   # shape (50, 3)

            point_cloud1 = PointCloud.from_array(random_points1)
            point_cloud2 = PointCloud.from_array(random_points2)

        Concatenate the two point clouds using the :meth:`concatenate` method.

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
        # Check if other is a PointCloud instance
        if not isinstance(other, PointCloud):
            raise ValueError("Input must be an instance of PointCloud.")
        if self.n_dimensions != other.n_dimensions:
            raise ValueError("Point clouds must have the same dimension to concatenate.")
        if not isinstance(inplace, bool):
            raise ValueError("inplace must be a boolean value.")
        
        # Concatenate points
        concatenated_points = numpy.vstack((self.points, other.points))

        # Return new instance or modify in place
        if inplace:
            self.points = concatenated_points
            return self
        else:
            return self.__class__.from_array(concatenated_points.copy())


    def copy(self) -> PointCloud:
        r"""
        Create a copy of the current :class:`PointCloud` instance.

        Returns
        -------
        :class:`PointCloud`
            A new :class:`PointCloud` object containing the same points as the current instance.

        Examples
        --------
        Creating a :class:`PointCloud` from a random NumPy array and making a copy.

        .. code-block:: python

            import numpy as np
            from pysdic import PointCloud

            # Create a random point cloud with 100 points
            random_points = np.random.rand(100, 3)  # shape (100, 3)
            point_cloud1 = PointCloud.from_array(random_points)

            # Create a copy of the existing PointCloud object
            point_cloud2 = point_cloud1.copy()

        """
        return self.__class__.from_array(self.points.copy())
    

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
        input_frame : Optional[:class:`Frame`], optional
            The input frame representing the current coordinate system of the point cloud. If None, the canonical frame is assumed.

        output_frame : Optional[:class:`Frame`], optional
            The output frame representing the target coordinate system for the point cloud. If None, the canonical frame is assumed.

        inplace : :class:`bool`, optional
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
            self.points = transformed_points
            return self
        else:
            return self.__class__.from_array(transformed_points.copy())


    def is_finite(self) -> numpy.ndarray:
        r"""
        Check which points in the point cloud are finite (not NaN or infinite).

        This method returns a boolean mask indicating which points in the point cloud are finite.

        Returns
        -------
        :class:`numpy.ndarray`
            A 1D boolean NumPy array of shape (:math:`N_p`,) where :math:`N_p` is the number of points in the point cloud.
            Each element is True if the corresponding point is finite, and False otherwise.

        Examples
        --------
        Create a :class:`PointCloud` from a NumPy array containing some NaN and infinite values.

        .. code-block:: python

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

            finite_mask = point_cloud.is_finite()
            print(finite_mask)
            # Output: [ True False False  True]

        """
        return numpy.isfinite(self.points).all(axis=1)
    

    def is_nan(self) -> numpy.ndarray:
        r"""
        Check which points in the point cloud are NaN.

        This method returns a boolean mask indicating which points in the point cloud contain NaN values.

        Returns
        -------
        :class:`numpy.ndarray`
            A 1D boolean NumPy array of shape (:math:`N_p`,) where :math:`N_p` is the number of points in the point cloud.
            Each element is True if the corresponding point contains any NaN value, and False otherwise.

        Examples
        --------
        Create a :class:`PointCloud` from a NumPy array containing some NaN values.

        .. code-block:: python

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

            nan_mask = point_cloud.is_nan()
            print(nan_mask)
            # Output: [False  True False  True]

        """
        return numpy.isnan(self.points).any(axis=1)
    

    def keep_points(self, other: PointCloud, inplace: bool = False) -> PointCloud:
        r"""
        Keep only the points in the current point cloud that are present in another :class:`PointCloud` instance.

        This method returns a new :class:`PointCloud` object containing only the points that are also present in the provided :class:`PointCloud` instance.

        .. note::

            Points in the `other` point cloud that are not present in the current point cloud are ignored.

        .. seealso::

            - :meth:`remove_points` for removing points that are present in another PointCloud instance.
            - :meth:`keep_points_at` for keeping points at specified indices.
            - :meth:`keep_points_inplace` for keeping points in place.

        Parameters
        ----------
        other : :class:`PointCloud`
            Another :class:`PointCloud` instance containing the points to be kept in the current point cloud.

        inplace : :class:`bool`, optional
            If :obj:`True`, modifies the current point cloud in place and returns itself. If :obj:`False`, returns a new :class:`PointCloud` instance (default is :obj:`False`).

        Returns
        -------
        :class:`PointCloud`
            A new :class:`PointCloud` object containing only the points that are also present in the provided :class:`PointCloud` instance or the modified current instance if :obj:`inplace` is True.
            
        Raises
        ------
        ValueError
            If the input is not an instance of :class:`PointCloud`.
            If the two point clouds have different point dimensions.


        Examples
        --------
        Create a :class:`PointCloud` from a random NumPy array.

        .. code-block:: python

            import numpy as np
            from pysdic import PointCloud
            # Create a random point cloud with 100 points
            random_points = np.random.rand(100, 3)  # shape (100, 3)
            point_cloud = PointCloud.from_array(random_points)

        Create another :class:`PointCloud` with some common points.

        .. code-block:: python

            # Create another point cloud with some common points
            common_points = random_points[[3, 6, 10]]  # shape (3, 3)
            other_point_cloud = PointCloud.from_array(common_points)

        Keeping only the points that are present in the other point cloud.

        .. code-block:: python

            # Keep only points that are present in the other point cloud
            new_point_cloud = point_cloud.keep_points(other_point_cloud)
            print(new_point_cloud.points)
            # Output: A NumPy array of shape (3, 3) with points [3, 6, 10] retained

        """
        # Check if other is a PointCloud instance
        if not isinstance(other, PointCloud):
            raise ValueError("Input must be an instance of PointCloud.")
        if self.n_dimensions != other.n_dimensions:
            raise ValueError("Point clouds must have the same dimension.")
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
            return self.__class__.from_array(kept_points.copy())


    def keep_points_at(self, indices: numpy.ndarray, inplace: bool = False) -> PointCloud:
        r"""
        Keep only the points at the specified indices in the point cloud.

        This method returns a new :class:`PointCloud` object containing only the points at the specified indices.

        .. seealso::

            - :meth:`remove_points_at` for removing points at specified indices.
            - :meth:`keep_points` for keeping points that are present in another :class:`PointCloud` instance.
            - :meth:`keep_points_at_inplace` for keeping points at specified indices in place.

        Parameters
        ----------
        indices : numpy.ndarray
            A 1D NumPy array of integer indices representing the points to be kept in the point cloud.

        inplace : :class:`bool`, optional
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

            import numpy as np
            from pysdic import PointCloud

            # Create a random point cloud with 100 points
            random_points = np.random.rand(100, 3)  # shape (100, 3)
            point_cloud = PointCloud.from_array(random_points)

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
        indices = numpy.asarray(indices, dtype=numpy.int64)
        if indices.ndim != 1:
            raise ValueError("Indices must be a 1D array.")
        if numpy.any(indices < 0) or numpy.any(indices >= self.n_points):
            raise ValueError("Indices are out of bounds.")
        
        # Return new instance or modify in place
        if inplace:
            self.points = self.points[indices]
            return self
        else:
            return self.__class__.from_array(self.points[indices].copy())
        
        
    def merge(self, other: PointCloud, inplace: bool = False) -> PointCloud:
        r"""
        Merge points from another :class:`PointCloud` instance with the current point cloud, avoiding duplicates.

        This method returns a new :class:`PointCloud` object containing the points from both point clouds, ensuring that duplicate points are not included.
        This method removes duplicate points from the initial point clouds so the indices of the points may change.

        .. note::

            Points in the `other` point cloud that are already present in the current point cloud are ignored.

        .. seealso::

            - :meth:`concatenate` for concatenating two point clouds, including duplicates.
            - :meth:`merge_inplace` for merging points from another point cloud in place.
            - :meth:`unique` to remove duplicate points within the same point cloud.

        Parameters
        ----------
        other : :class:`PointCloud`
            Another :class:`PointCloud` instance containing the points to be merged with the current point cloud.

        inplace : :class:`bool`, optional
            If :obj:`True`, modifies the current point cloud in place and returns itself. If :obj:`False`, returns a new :class:`PointCloud` instance (default is :obj:`False`).

        Returns
        -------
        :class:`PointCloud`
            A new :class:`PointCloud` object containing the merged points from both point clouds, excluding duplicates or the modified current instance if :obj:`inplace` is True.

        Raises
        ------
        ValueError
            If the input is not an instance of :class:`PointCloud`.
            If the two point clouds have different point dimensions.


        Examples
        --------
        Create a :class:`PointCloud` from a random NumPy array.

        .. code-block:: python

            import numpy as np
            from pysdic import PointCloud

            # Create a random point cloud with 100 points
            random_points = np.random.rand(100, 3)  # shape (100, 3)
            point_cloud = PointCloud.from_array(random_points)

        Create another :class:`PointCloud` with some common points.

        .. code-block:: python

            # Create another point cloud with some common points
            common_points = random_points[[3, 6, 10]]  # shape (3, 3)
            non_common_points = np.random.rand(5, 3) + 10  # shape (5, 3), offset to avoid overlap
            other_point_cloud = PointCloud.from_array(np.vstack((common_points, non_common_points)))

        Merging points from the other point cloud.

        .. code-block:: python

            # Merge points from the other point cloud
            new_point_cloud = point_cloud.merge(other_point_cloud)
            print(new_point_cloud.points)
            # Output: A NumPy array of shape (102, 3) with points [3, 6, 10] ignored and 5 new points added

        """
        # Check if other is a PointCloud instance
        if not isinstance(other, PointCloud):
            raise ValueError("Input must be an instance of PointCloud.")
        if self.n_dimensions != other.n_dimensions:
            raise ValueError("Point clouds must have the same dimension.")
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
            return self.__class__.from_array(merged_points.copy())
        
    
    def remove_not_finite(self, inplace: bool = False) -> PointCloud:
        r"""
        Remove points from the point cloud that contain non-finite values (NaN or Inf).

        This method returns a new PointCloud object with all points containing NaN or Inf values removed.

        .. seealso::

            - :meth:`remove_not_finite_inplace` for removing non-finite points in place.

        Parameters
        ----------
        inplace : :class:`bool`, optional
            If :obj:`True`, modifies the current point cloud in place and returns itself. If :obj:`False`, returns a new PointCloud instance (default is :obj:`False`).

        Returns
        -------
        :class:`PointCloud`
            A new :class:`PointCloud` object containing only the finite points or the modified current instance if :obj:`inplace` is True.

            
        Examples
        --------
        Create a :class:`PointCloud` from a NumPy array with some non-finite values.

        .. code-block:: python

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
            return self.__class__.from_array(finite_points.copy())

        
    def remove_points(self, other: PointCloud, inplace: bool = False) -> PointCloud:
        r"""
        Remove points from the current point cloud that are present in another :class:`PointCloud` instance.

        This method returns a new :class:`PointCloud` object with the points that are also present in the provided :class:`PointCloud` instance removed.

        .. note::

            Points in the `other` point cloud that are not present in the current point cloud are ignored.

        .. seealso::

            - :meth:`keep_points` for keeping points that are present in another :class:`PointCloud` instance.
            - :meth:`remove_points_at` for removing points at specified indices.
            - :meth:`remove_points_inplace` for removing points in place.

        Parameters
        ----------
        other : PointCloud
            Another PointCloud instance containing the points to be removed from the current point cloud.

        inplace : :class:`bool`, optional
            If :obj:`True`, modifies the current point cloud in place and returns itself. If :obj:`False`, returns a new :class:`PointCloud` instance (default is :obj:`False`). 

        Returns
        -------
        :class:`PointCloud`
            A new :class:`PointCloud` object with the points that are also present in the provided :class:`PointCloud` instance removed or the modified current instance if `inplace` is True.

        Raises
        ------
        ValueError
            If the input is not an instance of :class:`PointCloud`.
            If the two point clouds have different point dimensions.


        Examples
        --------
        Create a :class:`PointCloud` from a random NumPy array.

        .. code-block:: python

            import numpy as np
            from pysdic import PointCloud

            # Create a random point cloud with 100 points
            random_points = np.random.rand(100, 3)  # shape (100, 3)
            point_cloud = PointCloud.from_array(random_points)

        Create another :class:`PointCloud` with some common points.

        .. code-block:: python

            # Create another point cloud with some common points
            common_points = random_points[[3, 6, 10]]  # shape (3, 3)
            other_point_cloud = PointCloud.from_array(common_points)

        Removing points that are present in the other point cloud.

        .. code-block:: python

            # Remove points that are present in the other point cloud
            new_point_cloud = point_cloud.remove_points(other_point_cloud)
            print(new_point_cloud.points)
            # Output: A NumPy array of shape (97, 3) with points [3, 6, 10] removed

        """
        # Check if other is a PointCloud instance
        if not isinstance(other, PointCloud):
            raise ValueError("Input must be an instance of PointCloud.")
        if self.n_dimensions != other.n_dimensions:
            raise ValueError("Point clouds must have the same dimension.")
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
            return self.__class__.from_array(remaining_points.copy())


    def remove_points_at(self, indices: numpy.ndarray, inplace: bool = False) -> PointCloud:
        r"""
        Remove points from the point cloud based on their indices.

        This method returns a new :class:`PointCloud` object with the points at the specified indices removed.

        .. seealso::

            - :meth:`keep_points_at` for keeping points at specified indices.
            - :meth:`remove_points` for removing points that are present in another :class:`PointCloud` instance.
            - :meth:`remove_points_at_inplace` for removing points at specified indices in place.

        Parameters
        ----------
        indices : numpy.ndarray
            A 1D NumPy array of integer indices representing the points to be removed from the point cloud.

        inplace : :class:`bool`, optional
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

            import numpy as np
            from pysdic import PointCloud

            # Create a random point cloud with 100 points
            random_points = np.random.rand(100, 3)  # shape (100, 3)
            point_cloud = PointCloud.from_array(random_points)

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
        indices = numpy.asarray(indices, dtype=numpy.int64)
        if indices.ndim != 1:
            raise ValueError("Indices must be a 1D array.")
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
            return self.__class__.from_array(remaining_points.copy())


    def unique(self, inplace: bool = False) -> PointCloud:
        r"""
        Remove duplicate points from the point cloud.

        This method returns a new :class:`PointCloud` object containing only unique points, with duplicates removed.

        .. seealso::

            - :meth:`merge` for merging two point clouds while avoiding duplicates.
            - :meth:`unique_inplace` for removing duplicate points in place.

        Parameters
        ----------
        inplace : :class:`bool`, optional
            If :obj:`True`, modifies the current point cloud in place and returns itself. If :obj:`False`, returns a new :class:`PointCloud` instance (default is :obj:`False`).

        Returns
        -------
        :class:`PointCloud`
            A new :class:`PointCloud` object containing only unique points or the modified current instance if `inplace` is True.

            
        Examples
        --------
        Create a :class:`PointCloud` from a NumPy array with duplicate points.

        .. code-block:: python

            import numpy as np
            from pysdic import PointCloud

            # Create a point cloud with duplicate points
            points_with_duplicates = np.array([[0, 0, 0],
                                               [1, 1, 1],
                                               [0, 0, 0],  # Duplicate
                                               [2, 2, 2],
                                               [1, 1, 1]]) # Duplicate
            point_cloud = PointCloud.from_array(points_with_duplicates)

        Removing duplicate points using the :meth:`unique` method.

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
            return self.__class__.from_array(unique_points.copy())


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
            The minimum coordinates of the bounding box as a NumPy array of shape (:math:`E`,) representing (min_x, min_y, min_z, ...).

        :class:`numpy.ndarray`
            The maximum coordinates of the bounding box as a NumPy array of shape (:math:`E`,) representing (max_x, max_y, max_z, ...).

        Raises
        ------
        ValueError
            If the point cloud is empty or contains only non-finite values.

        Examples
        --------
        Create a tetrahedron point cloud and compute its bounding box.

        .. code-block:: python

            import numpy as np
            from pysdic import PointCloud

            # Create a tetrahedron point cloud
            tetrahedron_points = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]])  # shape (4, 3)
            point_cloud = PointCloud.from_array(tetrahedron_points)

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
        :class:`numpy.ndarray`
            The center of the bounding sphere as a NumPy array of shape (:math:`E`,).

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

            import numpy as np
            from pysdic import PointCloud

            # Create a tetrahedron point cloud
            tetrahedron_points = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]])  # shape (4, 3)
            point_cloud = PointCloud.from_array(tetrahedron_points)

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
        center = numpy.mean(finite_points, axis=0) # Shape (:math:`E`,)
        radius = numpy.max(numpy.linalg.norm(finite_points - center, axis=1))   

        if not numpy.all(numpy.isfinite(center)) or not numpy.isfinite(radius):
            raise ValueError("Point cloud contains only non-finite values; cannot compute bounding sphere.")
        
        return center, radius

    # =================
    # Dimension changes
    # =================
    def extend_n_dimensions(self, n_dimensions: int) -> PointCloud:
        r"""
        Extend the point cloud to a higher number of dimensions by adding zero coordinates.

        This method returns a new :class:`PointCloud` object with the specified number of dimensions, where additional dimensions are filled with zeros.

        Parameters
        ----------
        n_dimensions : :class:`int`
            The desired number of dimensions for the extended point cloud. Must be greater than the current number of dimensions.

        Returns
        -------
        :class:`PointCloud`
            A new :class:`PointCloud` object with the specified number of dimensions.

        Raises
        ------
        ValueError
            If the specified number of dimensions is less than the current number of dimensions.

            
        Examples
        --------
        Create a 2D :class:`PointCloud` and extend it to 3D.

        .. code-block:: python

            import numpy as np
            from pysdic import PointCloud

            # Create a 2D point cloud
            points_2d = np.array([[1, 2], [3, 4], [5, 6]])  # shape (3, 2)
            point_cloud_2d = PointCloud(points_2d)

            # Extend to 3D
            point_cloud_3d = point_cloud_2d.extend_n_dimensions(3)
            print(point_cloud_3d.points)
            # Output: A NumPy array of shape (3, 3) with z-coordinates set to 0

        """
        if not isinstance(n_dimensions, Integral) or n_dimensions <= self.n_dimensions:
            raise ValueError("n_dimensions must be an integer greater than the current number of dimensions.")
        
        # Create new points array with additional zero coordinates
        new_points = numpy.zeros((self.n_points, n_dimensions), dtype=self.points.dtype)
        new_points[:, :self.n_dimensions] = self.points

        return PointCloud.from_array(new_points)


    def extend_as_hyperplane(self, axes: numpy.ndarray) -> PointCloud:
        r"""
        Extend the point cloud to a higher number of dimensions by embedding it in a hyperplane defined by specified axes.

        This method returns a new :class:`PointCloud` object embedded in a higher-dimensional space, where the original points are placed along the specified axes and other dimensions are filled with zeros.
        :obj:`axes` should contain the coordinates of the axis (:math:`\vec{e}_i`) of the current point cloud in the higher-dimensional space.

        .. math::

            \text{new\_point}_j = \sum_{i=1}^{E} \text{point}_{j,i} \cdot \vec{e}_i

        .. seealso::

            - :meth:`reduce_to_hyperplane` for reducing a point cloud to a lower-dimensional space using specified axes.

        Parameters
        ----------
        axes : numpy.ndarray
            A 2D NumPy array of shape (:math:`E, D`) where :math:`E` is the current number of dimensions and :math:`D` is the desired number of dimensions in the extended point cloud.

        Returns
        -------
        :class:`PointCloud`
            A new :class:`PointCloud` object embedded in the higher-dimensional space.

        Raises
        ------
        ValueError
            If the shape of `axes` is not (:math:`E, D`) where :math:`E` is the current number of dimensions or if :math:`D` is less than :math:`E`.

            
        Examples
        --------
        Create a 2D :class:`PointCloud` and extend it to 4D using specified axes.

        .. code-block:: python

            import numpy as np
            from pysdic import PointCloud

            # Create a 2D point cloud
            points_2d = np.array([[1, 2], [3, 4], [5, 6]])  # shape (3, 2)
            point_cloud_2d = PointCloud(points_2d)

            # Define axes for embedding in 4D space
            axes = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0]])  # shape (2, 4)

            # Extend to 4D
            point_cloud_4d = point_cloud_2d.extend_as_hyperplane(axes)
            print(point_cloud_4d.points)
            # Output: A NumPy array of shape (3, 4) with points embedded in the specified hyperplane

        """
        axes = numpy.asarray(axes, dtype=numpy.float64)
        if axes.ndim != 2 or axes.shape[0] != self.n_dimensions or axes.shape[1] < self.n_dimensions:
            raise ValueError(f"axes must be a 2D array of shape ({self.n_dimensions}, D) with D >= {self.n_dimensions}.")
        
        # Create new points array in higher-dimensional space
        new_points = self.points @ axes  # Matrix multiplication to embed points

        return PointCloud.from_array(new_points)
    

    def reduce_to_hyperplane(self, axes: numpy.ndarray) -> PointCloud:
        r"""
        Reduce the point cloud to a lower number of dimensions by projecting it onto a hyperplane defined by specified axes.

        This method returns a new :class:`PointCloud` object projected onto a lower-dimensional space, where the original points are represented in terms of the specified axes.

        .. math::

            \text{new\_point}_{j,i} = \text{point}_j \cdot \vec{e}_i

        .. seealso::

            - :meth:`extend_as_hyperplane` for extending a point cloud to a higher-dimensional space using specified axes.

        Parameters
        ----------
        axes : numpy.ndarray
            A 2D NumPy array of shape (:math:`D, E`) where :math:`D` is desired number of dimensions in the reduced point cloud and :math:`E` is the current number of dimensions.

        Returns
        -------
        :class:`PointCloud`
            A new :class:`PointCloud` object projected onto the lower-dimensional space.

        Raises
        ------
        ValueError
            If the shape of `axes` is not (:math:`D, E`) where :math:`E` is the current number of dimensions or if :math:`D` is greater than :math:`E`.


        Examples
        --------
        Create a 4D :class:`PointCloud` and reduce it to 2D using specified axes.

        .. code-block:: python

            import numpy as np
            from pysdic import PointCloud

            # Create a 4D point cloud
            points_4d = np.array([[1, 2, 3, 4],
                                  [5, 6, 7, 8],
                                  [9, 10, 11, 12]])  # shape (3, 4)
            point_cloud_4d = PointCloud(points_4d)

            # Define axes for projection to 2D space
            axes = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0]])  # shape (2, 4)

            # Reduce to 2D
            point_cloud_2d = point_cloud_4d.reduce_on_hyperplane(axes)
            print(point_cloud_2d.points)
            # Output: A NumPy array of shape (3, 2) with points projected onto the specified hyperplane

        """
        axes = numpy.asarray(axes, dtype=numpy.float64)
        if axes.ndim != 2 or axes.shape[1] != self.n_dimensions or axes.shape[0] > self.n_dimensions:
            raise ValueError(f"axes must be a 2D array of shape (D, {self.n_dimensions}) with D <= {self.n_dimensions}.")
        
        # Create new points array in lower-dimensional space
        new_points = self.points @ axes.T  # Matrix multiplication to project points

        return PointCloud.from_array(new_points)
    

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
        Visualize the point cloud using PyVista (only for :math:`E \leq 3` point clouds).

        Only finite points (not NaN or Inf) are visualized.  

        .. seealso::

            - `PyVista Documentation <https://docs.pyvista.org>`_ for more details on visualization options.

        .. warning::
        
            This method only works for 1D, 2D, and 3D point clouds, as they can be represented in a 3D space. Attempting to visualize point clouds with more than 3 dimensions will raise a ValueError.

        Parameters
        ----------
        color : :class:`str`, optional
            The color of the points in the visualization. Default is :obj:`"black"`.

        point_size : :class:`float`, optional
            The size of the points in the visualization. Default is :obj:`1.0`.

        opacity : :class:`float`, optional
            The opacity of the points in the visualization. Default is :obj:`1.0` (fully opaque).

        title : Optional[:class:`str`], optional
            The title of the visualization window. Default is :obj:`None`.

        show_axes : :class:`bool`, optional
            Whether to display the axes in the visualization. Default is :obj:`True`.
        
        show_grid : :class:`bool`, optional
            Whether to display the grid in the visualization. Default is :obj:`True`.

        Examples
        --------
        Create a :class:`PointCloud` from a random NumPy array.

        .. code-block:: python

            import numpy as np
            from pysdic import PointCloud

            # Create a random point cloud with 100 points
            points_array = np.random.rand(100, 3)
            point_cloud = PointCloud.from_array(points_array)

        Visualize the point cloud using the `visualize` method.

        .. code-block:: python

            point_cloud.visualize(color='red', point_size=10)

        This will open a PyVista window displaying the point cloud with red points of size 10.

        .. figure:: /_static/point_cloud/point_cloud_3d_visualize_example.png
            :width: 600
            :align: center

            Example of a 3D point cloud visualization using the `visualize` method.
            
        """
        # Check empty point cloud
        if self.n_points == 0:
            raise ValueError("Cannot visualize an empty point cloud.")

        # Convert the mesh to 3D if necessary
        if self.n_dimensions < 1 or self.n_dimensions > 3:
            raise ValueError("Visualization is only supported for 1D, 2D, and 3D point clouds.")
        
        if self.n_dimensions == 3:
            point_cloud = self
        else:
            point_cloud = self.extend_n_dimensions(3)
        
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
        valid_points = point_cloud.points[numpy.all(numpy.isfinite(point_cloud.points), axis=1)]
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



