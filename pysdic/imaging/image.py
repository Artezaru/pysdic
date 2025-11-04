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
import numpy
import scipy
import cv2

import matplotlib.pyplot as plt

class Image(object):
    r"""
    A class to represent a 2D image to interpolate gray or colors values.

    The image is associated with an interpolation function to evaluate pixel values at arbitrary points in the image.
    The interpolation function is constructed using `scipy.interpolate.RectBivariateSpline` with cubic splines (kx=3, ky=3).

    The coordinates system used for the image is the same as the one used in `pycvcam` (See documentation of `pycvcam` at https://artezaru.github.io/pycvcam/).

    - The ``pixel_coordinates`` are defined in the array coordinate system (u, v) where u is the row index and v is the column index.
    - The ``image_coordinates`` are defined in the camera coordinate system (x, y) where x is the column index and y is the row index.

    Parameters
    ----------
    image : Optional[numpy.ndarray]
        The image that is viewed by the camera. The image must be in an unsigned bit integer format (e.g., `numpy.uint8`) and must have a shape of (height, width) or (height, width, channels). If not provided, the image will be set to None

    """
    __slots__ = [
        "_image",
        "_list_interpolation_functions",
    ]

    def __init__(self, image: Optional[numpy.ndarray] = None) -> None:
        self.image = image  # Calls the setter

    # ===================================================================
    # Properties
    # ===================================================================
    @property
    def image(self) -> Optional[numpy.ndarray]:
        r"""
        Get or set the image viewed by the camera.

        The shape of the image must be (height, width) or (height, width, channels).

        .. warning::

            If you update image coefficients without setting a new image, you must call :meth:`image_update` to inform the view that the image has been updated are update the interpolation functions accordingly.

        Parameters
        ----------
        image : Optional[numpy.ndarray]
            The image that is viewed by the camera. The image must be in a unsigned bit integer format (e.g., `numpy.uint8`) and must have a shape of (height, width) or (height, width, channels). If not provided, the image will be set to None.
        """
        return self._image
    
    @image.setter
    def image(self, image: Optional[numpy.ndarray]):
        if image is not None:
            if not isinstance(image, numpy.ndarray):
                raise TypeError("Image must be a numpy.ndarray.")
            if not numpy.issubdtype(image.dtype, numpy.unsignedinteger):
                raise TypeError("Image must be in an unsigned bit integer format (e.g., numpy.uint8).")
            if image.ndim != 2 and image.ndim != 3:
                raise ValueError("Image must have shape (height, width) with a grayscale image or (height, width, channels) with a color image.")
        self._image = image
        self.image_update()
    
    @property
    def shape(self) -> Optional[Tuple[int, int]]:
        r"""
        Get the shape of the image.

        The image shape is a tuple (height, width) representing the dimensions of the image.

        Returns
        -------
        Optional[Tuple[int, int]]
            The shape of the image as a tuple (height, width), or None if not set.

        """
        if self._image is None:
            return None
        return self._image.shape
    
    @property
    def height(self) -> Optional[int]:
        r"""
        Get the height of the image.

        Returns
        -------
        Optional[int]
            The height of the image, or None if not set.

        """
        if self._image is None:
            return None
        return self._image.shape[0]

    @property
    def width(self) -> Optional[int]:
        r"""
        Get the width of the image.

        Returns
        -------
        Optional[int]
            The width of the image, or None if not set.

        """
        if self._image is None:
            return None
        return self._image.shape[1]
    
    @property
    def is_color(self) -> Optional[bool]:
        r"""
        Check if the image is a color image (3D array).

        .. note::

            3-channel images with only one channel are still considered color images.

        Returns
        -------
        Optional[bool]
            True if the image is a color image, False if it is grayscale, or None if not set.

        """
        if self._image is None:
            return None
        return self._image.ndim == 3
    
    @property
    def is_grayscale(self) -> Optional[bool]:
        r"""
        Check if the image is grayscale (2D array).

        .. note::

            3-channel images with only one channel are still considered color images.

        Returns
        -------
        Optional[bool]
            True if the image is grayscale, False if it is color, or None if not set.

        """
        if self._image is None:
            return None
        return self._image.ndim == 2

    @property
    def n_channels(self) -> Optional[int]:
        r"""
        Get the number of channels in the image.

        Returns
        -------
        Optional[int]
            The number of channels in the image, or None if not set.

        """
        if self._image is None:
            return None
        if self._image.ndim == 2:
            return 1
        return self._image.shape[2]
    
    @property
    def dtype(self) -> Optional[numpy.dtype]:
        r"""
        Get the data type of the image.

        Returns
        -------
        Optional[numpy.dtype]
            The data type of the image, or None if not set.

        """
        if self._image is None:
            return None
        return self._image.dtype
    
    @property
    def ndim(self) -> Optional[int]:
        r"""
        Get the number of dimensions of the image.

        Returns
        -------
        Optional[int]
            The number of dimensions of the image, or None if not set.

        """
        if self._image is None:
            return None
        return self._image.ndim
    
    # ==================================================================
    # Update methods
    # ==================================================================
    def image_update(self):
        r"""
        Indicate that the image has been updated.

        This method should be called whenever the image is updated to ensure that the view reflects the new image state.

        - resets the interpolation function of the image.

        """
        self.construct_interpolation_functions()

    # ===================================================================
    # Class methods
    # ===================================================================
    @classmethod
    def from_array(cls, image_data: numpy.ndarray, copy: bool = False) -> Image:
        r"""
        Create an Image object from raw image data.

        Parameters
        ----------
        image_data : numpy.ndarray
            The raw image data as a numpy array. The image must be in an unsigned bit integer format (e.g., `numpy.uint8`) and must have a shape of (height, width) or (height, width, channels).

        copy : bool, optional
            Whether to copy the image data or use it directly, by default False.

        Returns
        -------
        Image
            An Image object initialized with the provided image data.


        Examples
        --------

        .. code-block:: python

            import numpy
            import cv2
            from pysdic.imaging import Image

            # Load an image using OpenCV
            image_data = cv2.imread("path_to_image.jpg")
            # Create an Image object from the raw data
            image = Image.from_array(image_data)

        """
        if not isinstance(image_data, numpy.ndarray):
            raise TypeError("image_data must be a numpy.ndarray.")
        if not numpy.issubdtype(image_data.dtype, numpy.unsignedinteger):
            raise TypeError("image_data must be in an unsigned bit integer format (e.g., numpy.uint8).")
        if image_data.ndim != 2 and image_data.ndim != 3:
            raise ValueError("image_data must have shape (height, width) with a grayscale image or (height, width, channels) with a color image.")
        if copy:
            image_data = image_data.copy()
        return cls(image=image_data)
    

    @classmethod
    def from_file(cls, file_path: str) -> Image:
        r"""
        Create an Image object by loading image data from a file.

        Parameters
        ----------
        file_path : str
            The path to the image file to be loaded.

        Returns
        -------
        Image
            An Image object initialized with the image data loaded from the specified file.


        Examples
        --------

        .. code-block:: python

            from pysdic.imaging import Image

            # Create an Image object by loading from a file
            image = Image.from_file("path_to_image.jpg")

        """
        image_data = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        return cls(image=image_data)
    

    def to_file(self, file_path: str) -> None:
        r"""
        Save the image to a file using OpenCV write function.

        Parameters
        ----------
        file_path : str
            The path where the image will be saved.

        """
        if self._image is None:
            raise ValueError("No image data to save.")
        cv2.imwrite(file_path, self._image)

    
    def to_array(self, copy: bool = False) -> Optional[numpy.ndarray]:
        r"""
        Get the image data as a numpy array.

        Parameters
        ----------
        copy : bool, optional
            Whether to return a copy of the image data. If False, returns a reference to the internal image data, by default False.

        Returns
        -------
        Optional[numpy.ndarray]
            The image data as a numpy array, or None if no image is set.

        """
        return self._image.copy() if copy else self._image
    
    # ==================================================================
    # Methods
    # ==================================================================
    def construct_interpolation_functions(self):
        r"""
        Construct the interpolation functions for the image.

        The interpolation functions are used to evaluate pixel values at arbitrary points in the image.

        .. warning::

            The interpolation functions take ``pixel_points`` as input, which are a swap of the camera's image points !
            See documentation of ``pycvcam`` (https://github.com/Artezaru/pycvcam) for more details on the name and usage of these points.

        """
        if self._image is None:
            self._list_interpolation_functions = None
            return
        
        if self.is_grayscale:  # Grayscale image
            self._list_interpolation_functions = [
                scipy.interpolate.RectBivariateSpline(numpy.arange(self.height), numpy.arange(self.width), self._image.astype(numpy.float64), kx=3, ky=3)
            ]
        else:  # Color image
            self._list_interpolation_functions = []
            for channel in range(self.n_channels):
                self._list_interpolation_functions.append(
                    scipy.interpolate.RectBivariateSpline(numpy.arange(self.height), numpy.arange(self.width), self._image[:, :, channel].astype(numpy.float64), kx=3, ky=3)
                )


    def copy(self) -> Image:
        r"""
        Create a copy of the Image object.

        Returns
        -------
        Image
            A new Image object that is a copy of the current object.

        """
        if self._image is None:
            return Image()
        return Image.from_array(self._image, copy=True)


    def evaluate_image_at_pixel_points(self, pixel_points: numpy.ndarray) -> numpy.ndarray:
        r"""
        Evaluate the image at given pixel points.

        .. seealso::

            - :meth:`evaluate_image_jacobian_dx_at_pixel_points` for evaluating the Jacobian of the image at pixel points along the x-axis (columns).
            - :meth:`evaluate_image_jacobian_dy_at_pixel_points` for evaluating the Jacobian of the image at pixel points along the y-axis (rows).

        Pixel out of bounds will be masked and the value at these points will be set to numpy.nan.

        .. warning::

            The interpolation function takes ``pixel_points`` as input, which are a swap of the camera's image points !
            See documentation of ``pycvcam`` (https://github.com/Artezaru/pycvcam) for more details on the name and usage of these points.

            You can use :meth:`pysdic.imaging.Camera.image_points_to_pixel_points` to achieve this conversion.

        Parameters
        ----------
        pixel_points : numpy.ndarray
            The pixel points at which to evaluate the image. The shape should be (N, 2) where N is the number of points and each point is represented by its (row, column) coordinates.

        Returns
        -------
        numpy.ndarray
            The evaluated pixel values with shape (N,) or (N, C) if color image in float64 dtype. If a pixel point is out of bounds, the corresponding value will be set to numpy.nan.
        
        """
        if self._image is None:
            raise ValueError("No image data to evaluate. Please set an image first.")
        if self._list_interpolation_functions is None:
            raise ValueError("Interpolation function is not constructed. Please set an image first.")
        
        if not isinstance(pixel_points, numpy.ndarray):
            raise TypeError("Pixel points must be a numpy.ndarray.")
        if not pixel_points.ndim == 2 or pixel_points.shape[1] != 2:
            raise ValueError("Pixel points must have shape (N, 2) where N is the number of points.")
        
        # Mask out-of-bounds points
        valid_mask = (pixel_points[:, 0] >= 0) & (pixel_points[:, 0] < self.height - 1) & (pixel_points[:, 1] >= 0) & (pixel_points[:, 1] < self.width - 1)

        # Create the values array with NaNs
        values = numpy.full((pixel_points.shape[0], self.n_channels), numpy.nan, dtype=numpy.float64)

        # Evaluate only valid points
        for channel in range(self.n_channels):
            values[valid_mask, channel] = self._list_interpolation_functions[channel].ev(pixel_points[valid_mask, 0], pixel_points[valid_mask, 1])
        
        # If grayscale image, return shape (N,)
        if self.is_grayscale:
            values = values[:, 0]

        return values
    

    def evaluate_image_at_image_points(self, image_points: numpy.ndarray) -> numpy.ndarray:
        r"""
        Evaluate the image at given image points.

        This is a convenience method that converts image points to pixel points and then evaluates the image at those pixel points.

        .. seealso::

            - :meth:`evaluate_image_at_pixel_points` for evaluating the image at pixel points.
            - :meth:`image_points_to_pixel_points` for converting image points to pixel points.
    
        Parameters
        ----------
        image_points : numpy.ndarray
            The image points at which to evaluate the image. The shape should be (N, 2) where N is the number of points and each point is represented by its (x, y) coordinates.

        Returns
        -------
        numpy.ndarray
            The evaluated pixel values with shape (N,) or (N, C) if color image in float64 dtype. If a pixel point is out of bounds, the corresponding value will be set to numpy.nan.

        """
        pixel_points = self.image_points_to_pixel_points(image_points)
        return self.evaluate_image_at_pixel_points(pixel_points)
    

    def evaluate_image_jacobian_dx_at_pixel_points(self, pixel_points: numpy.ndarray) -> numpy.ndarray:
        r"""
        Evaluate the Jacobian of the image at given pixel points along the x-axis (columns).

        .. seealso::

            - :meth:`evaluate_image_at_pixel_points` for evaluating the image at pixel points.
            - :meth:`evaluate_image_jacobian_dy_at_pixel_points` for evaluating the Jacobian of the image at pixel points along the y-axis (rows).

        Pixel out of bounds will be masked and the value at these points will be set to numpy.nan.

        .. warning::

            The interpolation function takes ``pixel_points`` as input, which are a swap of the camera's image points !
            See documentation of ``pycvcam`` (https://github.com/Artezaru/pycvcam) for more details on the name and usage of these points.

            You can use :meth:`pysdic.imaging.Camera.image_points_to_pixel_points` to achieve this conversion.

        Parameters
        ----------
        pixel_points : numpy.ndarray
            The pixel points at which to evaluate the Jacobian. The shape should be (N, 2) where N is the number of points and each point is represented by its (row, column) coordinates.

        Returns
        -------
        numpy.ndarray
            The evaluated Jacobian values with shape (N,) or (N, C) if color image in float64 dtype. If a pixel point is out of bounds, the corresponding value will be set to numpy.nan.
        """
        if self._image is None:
            raise ValueError("No image data to evaluate. Please set an image first.")
        if self._list_interpolation_functions is None:
            raise ValueError("Interpolation function is not constructed. Please set an image first.")
        
        if not isinstance(pixel_points, numpy.ndarray):
            raise TypeError("Pixel points must be a numpy.ndarray.")
        if not pixel_points.ndim == 2 or pixel_points.shape[1] != 2:
            raise ValueError("Pixel points must have shape (N, 2) where N is the number of points.")
        
        # Mask out-of-bounds points
        valid_mask = (pixel_points[:, 0] >= 0) & (pixel_points[:, 0] < self.height - 1) & (pixel_points[:, 1] >= 0) & (pixel_points[:, 1] < self.width - 1)

        # Create the values array with NaNs
        values = numpy.full((pixel_points.shape[0], self.n_channels), numpy.nan, dtype=numpy.float64)

        # Evaluate only valid points
        # For scipy dx -> rows, dy -> columns, we need to swap them to have derivative along x-axis (columns)
        for channel in range(self.n_channels):
            values[valid_mask, channel] = self._list_interpolation_functions[channel].ev(pixel_points[valid_mask, 0], pixel_points[valid_mask, 1], dx=0, dy=1)

        # If grayscale image, return shape (N,)
        if self.is_grayscale:
            values = values[:, 0]

        return values
    

    def evaluate_image_jacobian_dv_at_pixel_points(self, pixel_points: numpy.ndarray) -> numpy.ndarray:
        r"""
        Evaluate the Jacobian of the image at given pixel points along the v-axis (columns).

        This is a convenience method of the :meth:`evaluate_image_jacobian_dx_at_pixel_points` method as 'dx = dv' in pixel coordinates.

        .. seealso::

            - :meth:`evaluate_image_jacobian_dx_at_pixel_points` for evaluating the Jacobian of the image at pixel points along the x-axis (columns).

        Parameters
        ----------
        pixel_points : numpy.ndarray
            The pixel points at which to evaluate the Jacobian. The shape should be (N, 2) where N is the number of points and each point is represented by its (row, column) coordinates.

        Returns
        -------
        numpy.ndarray
            The evaluated Jacobian values with shape (N,) or (N, C) if color image in float64 dtype. If a pixel point is out of bounds, the corresponding value will be set to numpy.nan.
        """
        return self.evaluate_image_jacobian_dx_at_pixel_points(pixel_points)
    

    def evaluate_image_jacobian_dx_at_image_points(self, image_points: numpy.ndarray) -> numpy.ndarray:
        r"""
        Evaluate the Jacobian of the image at given image points along the x-axis (columns).

        This is a convenience method that converts image points to pixel points and then evaluates the Jacobian of the image at those pixel points along the x-axis.

        .. seealso::

            - :meth:`evaluate_image_jacobian_dx_at_pixel_points` for evaluating the Jacobian of the image at pixel points along the x-axis.
            - :meth:`image_points_to_pixel_points` for converting image points to pixel points.

        Parameters
        ----------
        image_points : numpy.ndarray
            The image points at which to evaluate the Jacobian. The shape should be (N, 2) where N is the number of points and each point is represented by its (x, y) coordinates.

        Returns
        -------
        numpy.ndarray
            The evaluated Jacobian values with shape (N,) or (N, C) if color image in float64 dtype. If a pixel point is out of bounds, the corresponding value will be set to numpy.nan.
        """
        pixel_points = self.image_points_to_pixel_points(image_points)
        return self.evaluate_image_jacobian_dx_at_pixel_points(pixel_points)
    

    def evaluate_image_jacobian_dv_at_image_points(self, image_points: numpy.ndarray) -> numpy.ndarray:
        r"""
        Evaluate the Jacobian of the image at given image points along the v-axis (columns).

        This is a convenience method of the :meth:`evaluate_image_jacobian_dx_at_image_points` method as 'dx = dv' in pixel coordinates.

        .. seealso::

            - :meth:`evaluate_image_jacobian_dx_at_image_points` for evaluating the Jacobian of the image at image points along the x-axis (columns).

        Parameters
        ----------
        image_points : numpy.ndarray
            The image points at which to evaluate the Jacobian. The shape should be (N, 2) where N is the number of points and each point is represented by its (x, y) coordinates.

        Returns
        -------
        numpy.ndarray
            The evaluated Jacobian values with shape (N,) or (N, C) if color image in float64 dtype. If a pixel point is out of bounds, the corresponding value will be set to numpy.nan.
        """
        return self.evaluate_image_jacobian_dx_at_image_points(image_points)


    def evaluate_image_jacobian_dy_at_pixel_points(self, pixel_points: numpy.ndarray) -> numpy.ndarray:
        r"""
        Evaluate the Jacobian of the image at given pixel points along the y-axis (rows).

        .. seealso::

            - :meth:`evaluate_image_at_pixel_points` for evaluating the image at pixel points.
            - :meth:`evaluate_image_jacobian_dx_at_pixel_points` for evaluating the Jacobian of the image at pixel points along the x-axis (columns).

        Pixel out of bounds will be masked and the value at these points will be set to numpy.nan.

        .. warning::

            The interpolation function takes ``pixel_points`` as input, which are a swap of the camera's image points !
            See documentation of ``pycvcam`` (https://github.com/Artezaru/pycvcam) for more details on the name and usage of these points.

            You can use :meth:`pysdic.imaging.Camera.image_points_to_pixel_points` to achieve this conversion.

        Parameters
        ----------
        pixel_points : numpy.ndarray
            The pixel points at which to evaluate the Jacobian. The shape should be (N, 2) where N is the number of points and each point is represented by its (row, column) coordinates.

        Returns
        -------
        numpy.ndarray
            The evaluated Jacobian values with shape (N,) or (N, C) if color image in float64 dtype. If a pixel point is out of bounds, the corresponding value will be set to numpy.nan.
        """
        if self._image is None:
            raise ValueError("No image data to evaluate. Please set an image first.")
        if self._list_interpolation_functions is None:
            raise ValueError("Interpolation functions are not constructed. Please set an image first.")

        if not isinstance(pixel_points, numpy.ndarray):
            raise TypeError("Pixel points must be a numpy.ndarray.")
        if not pixel_points.ndim == 2 or pixel_points.shape[1] != 2:
            raise ValueError("Pixel points must have shape (N, 2) where N is the number of points.")

        # Mask out-of-bounds points
        valid_mask = (pixel_points[:, 0] >= 0) & (pixel_points[:, 0] < self.height - 1) & (pixel_points[:, 1] >= 0) & (pixel_points[:, 1] < self.width - 1)

        # Create the values array with NaNs
        values = numpy.full((pixel_points.shape[0], self.n_channels), numpy.nan, dtype=numpy.float64)

        # Evaluate only valid points
        # For scipy dx -> rows, dy -> columns, we need to swap them to have derivative along y-axis (rows)
        for channel in range(self.n_channels):
            values[valid_mask, channel] = self._list_interpolation_functions[channel].ev(pixel_points[valid_mask, 0], pixel_points[valid_mask, 1], dx=1, dy=0)

        # If grayscale image, return shape (N,)
        if self.is_grayscale:
            values = values[:, 0]
    
        return values
    

    def evaluate_image_jacobian_du_at_pixel_points(self, pixel_points: numpy.ndarray) -> numpy.ndarray:
        r"""
        Evaluate the Jacobian of the image at given pixel points along the u-axis (rows).

        This is a convenience method of the :meth:`evaluate_image_jacobian_dy_at_pixel_points` method as 'dy = du' in pixel coordinates.

        .. seealso::

            - :meth:`evaluate_image_jacobian_dy_at_pixel_points` for evaluating the Jacobian of the image at pixel points along the y-axis (rows).

        Parameters
        ----------
        pixel_points : numpy.ndarray
            The pixel points at which to evaluate the Jacobian. The shape should be (N, 2) where N is the number of points and each point is represented by its (row, column) coordinates.

        Returns
        -------
        numpy.ndarray
            The evaluated Jacobian values with shape (N,) or (N, C) if color image in float64 dtype. If a pixel point is out of bounds, the corresponding value will be set to numpy.nan.
        """
        return self.evaluate_image_jacobian_dy_at_pixel_points(pixel_points)
    

    def evaluate_image_jacobian_dy_at_image_points(self, image_points: numpy.ndarray) -> numpy.ndarray:
        r"""
        Evaluate the Jacobian of the image at given image points along the y-axis (rows).

        This is a convenience method that converts image points to pixel points and then evaluates the Jacobian of the image at those pixel points along the y-axis.

        .. seealso::

            - :meth:`evaluate_image_jacobian_dy_at_pixel_points` for evaluating the Jacobian of the image at pixel points along the y-axis.
            - :meth:`image_points_to_pixel_points` for converting image points to pixel points.

        Parameters
        ----------
        image_points : numpy.ndarray
            The image points at which to evaluate the Jacobian. The shape should be (N, 2) where N is the number of points and each point is represented by its (x, y) coordinates.
        
        Returns
        -------
        numpy.ndarray
            The evaluated Jacobian values with shape (N,) or (N, C) if color image in float64 dtype. If a pixel point is out of bounds, the corresponding value will be set to numpy.nan
        """
        pixel_points = self.image_points_to_pixel_points(image_points)
        return self.evaluate_image_jacobian_dy_at_pixel_points(pixel_points)


    def evaluate_image_jacobian_du_at_image_points(self, image_points: numpy.ndarray) -> numpy.ndarray:
        r"""
        Evaluate the Jacobian of the image at given image points along the u-axis (rows).

        This is a convenience method of the :meth:`evaluate_image_jacobian_dy_at_image_points` method as 'dy = du' in pixel coordinates.

        .. seealso::

            - :meth:`evaluate_image_jacobian_dy_at_image_points` for evaluating the Jacobian of the image at image points along the y-axis (rows).

        Parameters
        ----------
        image_points : numpy.ndarray
            The image points at which to evaluate the Jacobian. The shape should be (N, 2) where N is the number of points and each point is represented by its (x, y) coordinates.

        Returns
        -------
        numpy.ndarray
            The evaluated Jacobian values with shape (N,) or (N, C) if color image in float64 dtype. If a pixel point is out of bounds, the corresponding value will be set to numpy.nan.
        """
        return self.evaluate_image_jacobian_dy_at_image_points(image_points)
    

    def pixel_points_to_image_points(self, pixel_points: numpy.ndarray) -> numpy.ndarray:
        r"""
        Convert pixel points to image points.

        Only swap the x and y coordinates of the pixel points to convert them to image points.

        .. note::

            - The image points are defined in the image coordinate system (x, y).
            - The pixel points are defined in the pixel coordinate system (u, v).

        Parameters
        ----------
        pixel_points : numpy.ndarray
            A 2D array of shape (..., 2) representing the pixel points in pixel coordinate system (rows, columns).

        Returns
        -------
        numpy.ndarray
            A 2D array of shape (..., 2) representing the image points in image coordinate system (x, y).
        """
        if not isinstance(pixel_points, numpy.ndarray):
            raise TypeError("pixel_points must be a numpy.ndarray.")
        if pixel_points.ndim < 2 or pixel_points.shape[1] != 2:
            raise ValueError("pixel_points must be an array with shape (..., 2).")
        
        return pixel_points[..., [1, 0]]  # Swap x and y coordinates to convert to image points
    

    def image_points_to_pixel_points(self, image_points: numpy.ndarray) -> numpy.ndarray:
        r"""
        Convert image points to pixel points.

        Only swap the x and y coordinates of the image points to convert them to pixel points.

        .. note::

            - The image points are defined in the image coordinate system (x, y).
            - The pixel points are defined in the pixel coordinate system (u, v).

        Parameters
        ----------
        image_points : numpy.ndarray
            A 2D array of shape (..., 2) representing the image points in image coordinate system (x, y).

        Returns
        -------
        numpy.ndarray
            A 2D array of shape (..., 2) representing the pixel points in pixel coordinate system (rows, columns).
        """
        if not isinstance(image_points, numpy.ndarray):
            raise TypeError("image_points must be a numpy.ndarray.")
        if image_points.ndim < 2 or image_points.shape[1] != 2:
            raise ValueError("image_points must be an array with shape (..., 2).")

        return image_points[..., [1, 0]]
    

    def get_image_pixel_points(self, mask: Optional[numpy.ndarray] = None) -> numpy.ndarray:
        r"""
        Create a grid of pixel points covering the entire image.

        The method is equivalent to the following code:

        .. code-block:: python

            import numpy

            pixel_points = numpy.indices((height, width)).reshape(2, -1).T
            pixel_points = pixel_points[mask.flatten()] # If mask is provided
            pixel_points = pixel_points.astype(numpy.float64)

        Parameters
        ----------
        mask : Optional[numpy.ndarray], optional
            A boolean mask to filter the pixel points. If None, all pixel points are returned. Default is None.
            Shape (H*W,) or (H, W) where H is the height and W is the width of the camera sensor.

        Returns
        -------
        numpy.ndarray
            A 2D array of shape (N, 2) where N is the number of pixel points in float64 dtype. Each row represents a pixel point in pixel coordinate system (row, column).

        Raises
        ------
        ValueError
            If the image is not set.

        """
        if self._image is None:
            raise ValueError("Image is not set. Cannot create pixel points.")

        pixel_points = numpy.indices((self.height, self.width), dtype=numpy.float64).reshape(2, -1).T  # Shape (H*W, 2)

        if mask is not None:
            if not isinstance(mask, numpy.ndarray):
                raise TypeError("mask must be a numpy.ndarray.")
            if not numpy.issubdtype(mask.dtype, numpy.bool_):
                raise TypeError("mask must be a boolean numpy.ndarray.")
            if not (mask.ndim == 1 and mask.shape[0] == self.sensor_height * self.sensor_width) and not (mask.ndim == 2 and mask.shape == (self.height, self.width)):
                raise ValueError("mask must be a 1D array of shape (H*W,) or a 2D array of shape (H, W) where H is the height and W is the width of the image.")
            
            pixel_points = pixel_points[mask.flatten(), :]

        return pixel_points
    

    def get_image_image_points(self, mask: Optional[numpy.ndarray] = None) -> numpy.ndarray:
        r"""
        Create a grid of image points covering the entire image.

        The method is equivalent to the following code:

        .. code-block:: python

            import numpy

            pixel_points = numpy.indices((height, width)).reshape(2, -1).T
            pixel_points = pixel_points[mask.flatten()] # If mask is provided
            pixel_points = pixel_points.astype(numpy.float64)
            image_points = pixel_points[:, [1, 0]]  # Swap to get image points (x, y)


        Parameters
        ----------
        mask : Optional[numpy.ndarray], optional
            A boolean mask to filter the image points. If None, all image points are returned. Default is None.
            Shape (H*W,) or (H, W) where H is the height and W is the width of the camera sensor.

        Returns
        -------
        numpy.ndarray
            A 2D array of shape (N, 2) where N is the number of image points in float64 dtype. Each row represents an image point in image coordinate system (x, y).

        Raises
        ------
        ValueError
            If the image is not set.

        """
        pixel_points = self.get_image_pixels_points(mask=mask)
        return self.pixel_points_to_image_points(pixel_points)


    def get_interpolation_function(self, channel: int = 1) -> Optional[scipy.interpolate.RectBivariateSpline]:
        r"""
        Get the scipy interpolation function for the image.

        The interpolation function is an object `scipy.interpolate.RectBivariateSpline` with cubic splines (kx=3, ky=3).

        The interpolation function is used to evaluate pixel values at arbitrary points in the image.

        .. warning::

            The interpolation function takes ``pixel_points`` as input, which are a swap of the camera's image points !
            See documentation of ``pycvcam`` (https://github.com/Artezaru/pycvcam) for more details on the name and usage of these points.

        Parameters
        ----------
        channel : int
            The channel index for color images. Defaults to 1 (the first channel). For a grayscale image, this parameter must be 1.

        Returns
        -------
        Optional[scipy.interpolate.RectBivariateSpline]
            The interpolation function for the image, or None if not set.

        """
        if self._list_interpolation_functions is None:
            return None
        if not isinstance(channel, int):
            raise TypeError("Channel index must be an integer.")
        if channel < 0 or channel >= len(self._list_interpolation_functions):
            raise ValueError(f"Channel index {channel} is out of bounds for image with {len(self._list_interpolation_functions)} channels.")
        return self._list_interpolation_functions[channel]
            
        
    # ==================================================================
    # Visualization methods
    # ==================================================================
    def visualize(
        self,
        title: Optional[str] = None,
        cmap: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 6)
    ) -> None:
        r"""
        Visualize the image using Matplotlib.

        Parameters
        ----------
        title : Optional[str], optional
            The title of the plot, by default None (which uses no title).

        cmap : Optional[str], optional
            The colormap to use for grayscale images, by default None (which uses the default gray colormap).

        figsize : Tuple[int, int], optional
            The size of the figure in inches, by default (8, 6).

        Raises
        ------
        ValueError
            If the image is not set.

        """
        if self._image is None:
            raise ValueError("Image is not set. Cannot visualize.")
        
        if not isinstance(figsize, tuple) or len(figsize) != 2:
            raise TypeError("figsize must be a tuple of two integers.")
        if not all(isinstance(i, int) for i in figsize):
            raise TypeError("figsize must be a tuple of two integers.")
        if figsize[0] <= 0 or figsize[1] <= 0:
            raise ValueError("figsize dimensions must be positive integers.")
        
        if title is not None and not isinstance(title, str):
            raise TypeError("title must be a string.")
        if cmap is not None and not isinstance(cmap, str):
            raise TypeError("cmap must be a string or None.")

        plt.figure(figsize=figsize)

        if self.is_grayscale:
            plt.imshow(self._image, cmap=cmap if cmap is not None else 'gray', vmin=0, vmax=numpy.iinfo(self.dtype).max)
        else:
            # Convert BGR to RGB for visualization
            plt.imshow(cv2.cvtColor(self._image, cv2.COLOR_BGR2RGB), vmin=0, vmax=numpy.iinfo(self.dtype).max)

        if title is not None:
            plt.title(title)

        plt.axis('off')
        plt.show()
