import numpy
import scipy
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pycvcam

from .image_projection_result import ImageProjectionResult
from .projection_result import ProjectionResult
from .camera import Camera
from ..geometry.point_cloud_3d import PointCloud3D
from ..geometry.mesh_3d import Mesh3D

class View(object):
    r"""
    A view is a constructed by a camera and an image acquired by the camera.

    If the sensor size of the camera changes, the image must be updated to reflect the new sensor dimensions.

    .. note::

        If multiple views has the same camera, a change on the camera's parameters will affect all views using that camera.

    .. warning::
        
        If the image is updated directly without using the setter, you must be call the method ``image_update()`` to indicate that the image has been updated in order the class reconstruct the interpolation function of the image.

    Parameters
    ----------
    camera : Camera
        The camera that will be used to view the image.
    
    image : Optional[numpy.ndarray]
        The image that is viewed by the camera. The image must be in a unsigned bit integer format (e.g., `numpy.uint8`) and must have a shape of (height, width). If not provided, the image will be set to None.

    """

    __slots__ = [
        "_camera", 
        "_image", 
        "_interpolation_function",
        "_camera_size", 
    ]

    def __init__(self, camera: Camera, image: Optional[numpy.ndarray] = None):
        self.camera = camera
        self.image = image

    # ===================================================================
    # Properties
    # ===================================================================
    @property
    def camera(self) -> Camera:
        r"""
        Get or set the camera used by the view.

        If the camera's sensor size changes, a new image must be set to reflect the new sensor dimensions.

        Parameters
        ----------
        camera : Camera
            The camera to be used by the view.
        """
        return self._camera
    
    @camera.setter
    def camera(self, camera: Camera):
        if not isinstance(camera, Camera):
            raise TypeError("Camera must be an instance of Camera.")
        self._camera = camera
        # Reset the previous camera size to trigger image update if necessary
        self._camera_size = (camera.sensor_height, camera.sensor_width)
        self.camera_update()


    @property
    def image(self) -> Optional[numpy.ndarray]:
        r"""
        Get or set the image viewed by the camera.

        The shape of the image must be (height, width) with a grayscale image.

        Parameters
        ----------
        image : Optional[numpy.ndarray]
            The image that is viewed by the camera. The image must be in a unsigned bit integer format (e.g., `numpy.uint8`) and must have a shape of (height, width). If not provided, the image will be set to None.
        """
        return self._image
    
    @image.setter
    def image(self, image: Optional[numpy.ndarray]):
        if image is not None:
            if not isinstance(image, numpy.ndarray):
                raise TypeError("Image must be a numpy.ndarray.")
            if not numpy.issubdtype(image.dtype, numpy.unsignedinteger):
                raise TypeError("Image must be in an unsigned bit integer format (e.g., numpy.uint8).")
            if image.ndim != 2:
                raise ValueError("Image must have shape (height, width) with a grayscale image.")
            if not image.shape == (self.camera.sensor_height, self.camera.sensor_width):
                raise ValueError(f"Image shape {image.shape} does not match camera sensor size {self.camera.sensor_height, self.camera.sensor_width}.")
        self._image = image
        self.image_update()

    
    @property
    def interpolation_function(self) -> Optional[scipy.interpolate.RectBivariateSpline]:
        r"""
        Get the interpolation function for the image.
        The interpolation function is an object `scipy.interpolate.RectBivariateSpline`.

        The interpolation function is used to evaluate pixel values at arbitrary points in the image.

        .. warning::

            The interpolation function takes ``pixel_points`` as input, which are a swap of the camera's image points !
            See documentation of ``pycvcam`` (https://github.com/Artezaru/pycvcam) for more details on the name and usage of these points.

        Returns
        -------
        Optional[scipy.interpolate.RectBivariateSpline]
            The interpolation function for the image, or None if not set.

        """
        return self._interpolation_function
    

    @property
    def image_shape(self) -> Optional[Tuple[int, int]]:
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
    def camera_size(self) -> Tuple[int, int]:
        r"""
        Get the size of the camera's sensor.

        Returns
        -------
        Tuple[int, int]
            The size of the camera's sensor as a tuple (height, width).
        """
        return self._camera_size

    # ===================================================================
    # Indicate an update
    # ===================================================================
    def camera_update(self):
        r"""
        Indicate that the camera has been updated.

        This method should be called whenever the camera parameters are updated to ensure that the view reflects the new camera state.

        - resets the ``image`` attribute to None (uncalculated) if the camera's sensor size has changed.

        """
        self.camera_size_change(self.camera.sensor_height, self.camera.sensor_width)

 
    def image_update(self):
        r"""
        Indicate that the image has been updated.

        This method should be called whenever the image is updated to ensure that the view reflects the new image state.

        - resets the interpolation function of the image.

        """
        self.construct_interpolation_function()


    def camera_size_change(self, height: int, width: int):
        r"""
        Callback for when the camera's sensor size changes.

        This method is called when the camera's sensor size is updated, allowing the view to adjust accordingly.

        Parameters
        ----------
        height : int
            The new sensor height.
        width : int
            The new sensor width.
        """
        if not isinstance(height, int) or not isinstance(width, int):
            raise TypeError("Height and width must be integers.")
        if height <= 0 or width <= 0:
            raise ValueError("Height and width must be positive integers.")
        old_size = self._camera_size
        self._camera_size = (height, width)
        if old_size != self._camera_size:
            if self._image is not None:
                print(f"[View] Camera size changed from {old_size} to {self._camera_size}. current image will be removed, please set a new image.")
                self._image = None

    # ===================================================================
    # Processing methods
    # ===================================================================
    def construct_interpolation_function(self):
        r"""
        Construct the interpolation function for the image.

        The interpolation function is used to evaluate pixel values at arbitrary points in the image.

        .. warning::

            The interpolation function takes ``pixel_points`` as input, which are a swap of the camera's image points !
            See documentation of ``pycvcam`` (https://github.com/Artezaru/pycvcam) for more details on the name and usage of these points.

        """
        if self._image is None:
            self._interpolation_function = None
            return
        
        self._interpolation_function = scipy.interpolate.RectBivariateSpline(numpy.arange(self.camera.sensor_height), numpy.arange(self.camera.sensor_width), self._image.astype(numpy.float64), kx=3, ky=3)


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
            The evaluated pixel values with shape (N,). If a pixel point is out of bounds, the corresponding value will be set to numpy.nan.
        """
        if self._interpolation_function is None:
            raise ValueError("Interpolation function is not constructed. Please set an image first.")
        
        if not isinstance(pixel_points, numpy.ndarray):
            raise TypeError("Pixel points must be a numpy.ndarray.")
        if not pixel_points.ndim == 2 or pixel_points.shape[1] != 2:
            raise ValueError("Pixel points must have shape (N, 2) where N is the number of points.")
        
        # Mask out-of-bounds points
        valid_mask = (pixel_points[:, 0] >= 0) & (pixel_points[:, 0] < self.camera.sensor_height - 1) & (pixel_points[:, 1] >= 0) & (pixel_points[:, 1] < self.camera.sensor_width - 1)

        # Create the values array with NaNs
        values = numpy.full(pixel_points.shape[0], numpy.nan, dtype=numpy.float64)

        # Evaluate only valid points
        values[valid_mask] = self._interpolation_function.ev(pixel_points[valid_mask, 0], pixel_points[valid_mask, 1])

        return values
    

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
            The evaluated Jacobian values with shape (N,). If a pixel point is out of bounds, the corresponding value will be set to numpy.nan.
        """
        if self._interpolation_function is None:
            raise ValueError("Interpolation function is not constructed. Please set an image first.")
        
        if not isinstance(pixel_points, numpy.ndarray):
            raise TypeError("Pixel points must be a numpy.ndarray.")
        if not pixel_points.ndim == 2 or pixel_points.shape[1] != 2:
            raise ValueError("Pixel points must have shape (N, 2) where N is the number of points.")
        
        # Mask out-of-bounds points
        valid_mask = (pixel_points[:, 0] >= 0) & (pixel_points[:, 0] < self.camera.sensor_height - 1) & (pixel_points[:, 1] >= 0) & (pixel_points[:, 1] < self.camera.sensor_width - 1)

        # Create the values array with NaNs
        values = numpy.full(pixel_points.shape[0], numpy.nan, dtype=numpy.float64)

        # Evaluate only valid points
        # For scipy dx -> rows, dy -> columns, we need to swap them to have derivative along x-axis (columns)
        values[valid_mask] = self._interpolation_function.ev(pixel_points[valid_mask, 0], pixel_points[valid_mask, 1], dx=0, dy=1)

        return values
    

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
            The evaluated Jacobian values with shape (N,). If a pixel point is out of bounds, the corresponding value will be set to numpy.nan.
        """
        if self._interpolation_function is None:
            raise ValueError("Interpolation function is not constructed. Please set an image first.")

        if not isinstance(pixel_points, numpy.ndarray):
            raise TypeError("Pixel points must be a numpy.ndarray.")
        if not pixel_points.ndim == 2 or pixel_points.shape[1] != 2:
            raise ValueError("Pixel points must have shape (N, 2) where N is the number of points.")

        # Mask out-of-bounds points
        valid_mask = (pixel_points[:, 0] >= 0) & (pixel_points[:, 0] < self.camera.sensor_height - 1) & (pixel_points[:, 1] >= 0) & (pixel_points[:, 1] < self.camera.sensor_width - 1)

        # Create the values array with NaNs
        values = numpy.full(pixel_points.shape[0], numpy.nan, dtype=numpy.float64)

        # Evaluate only valid points
        # For scipy dx -> rows, dy -> columns, we need to swap them to have derivative along y-axis (rows)
        values[valid_mask] = self._interpolation_function.ev(pixel_points[valid_mask, 0], pixel_points[valid_mask, 1], dx=1, dy=0)

        return values
    
    # ===================================================================
    # Application composition methods
    # ===================================================================
    def image_project(self, world_points: numpy.ndarray, dx: bool = False, dintrinsic: bool = False, ddistortion: bool = False, dextrinsic: bool = False) -> ImageProjectionResult:
        r"""
        Project 3D world points to gray level using the camera's intrinsic, extrinsic, distortion parameters and image interpolation function.

        .. seealso::

            - :meth:`pysdic.imaging.Camera.project` for the geometric projection process.
            - :meth:`assembly_image_projection` for the assembly of the image projection result.
            - :class:`ImageProjectionResult` for the structure of the output.

        Parameters
        ----------
        world_points : numpy.ndarray
            The 3D world points to be projected. The shape should be (N, 3) where N is the number of points and each point is represented by its (x, y, z) coordinates.
        
        dx : bool, optional
            If True, the function will also return the jacobian of the gray levels with respect to the world points. Default is False.
        
        dintrinsic : bool, optional
            If True, compute the Jacobian of the gray levels with respect to the intrinsic parameters.
        
        ddistortion : bool, optional
            If True, compute the Jacobian of the gray levels with respect to the distortion parameters.
        
        dextrinsic : bool, optional
            If True, compute the Jacobian of the gray levels with respect to the extrinsic parameters.

        Returns
        -------
        ImageProjectionResult
            An instance of ImageProjectionResult containing:
            
            - `gray_levels`: A 1D array of shape (N, ) representing the image values at the projected pixel points.
            - `jacobian_dx`: (optional) A 3D array of shape (N, 1, 3) representing the jacobian of the normalized points with respect to the world points if `dx` is True.
            - `jacobian_dintrinsic`: (optional) A 3D array of shape (N, 1, Nintrinsic) representing the jacobian of the pixel points with respect to the intrinsic parameters if `dintrinsic` is True.
            - `jacobian_ddistortion`: (optional) A 3D array of shape (N, 1, Ndistortion) representing the jacobian of the pixel points with respect to the distortion parameters if `ddistortion` is True.
            - `jacobian_dextrinsic`: (optional) A 3D array of shape (N, 1, Nextrinsic) representing the jacobian of the pixel points with respect to the extrinsic parameters if `dextrinsic` is True.
        
            
        Examples
        --------

        Lets create a simple view and project some 3D points:

        .. code-block:: python

            import numpy
            from pysdic.imaging import Camera
            from pycvcam import Cv2Extrinsic, Cv2Intrinsic

            rotation_vector = numpy.array([0.1, 0.2, 0.3])
            translation_vector = numpy.array([12.0, 34.0, 56.0])

            extrinsic = Cv2Extrinsic.from_rt(rotation_vector, translation_vector)

            intrinsic = Cv2Intrinsic.from_matrix(
                numpy.array([[1000, 0, 320],
                            [0, 1000, 240],
                            [0, 0, 1]])
            )

            camera = Camera(
                sensor_height=480,
                sensor_width=640,
                intrinsic=intrinsic,
                extrinsic=extrinsic,
            )

            # Create a simple view with a blank image
            image = numpy.zeros((480, 640), dtype=numpy.uint8)
            view = View(camera=camera, image=image)

            # Define some 3D world points
            world_points = numpy.array([
                [0, 0, 1000],
                [100, 0, 1000],
                [0, 100, 1000],
                [100, 100, 1000]
            ])

            # Project the 3D points to 2D image points
            image_projection_result = view.image_project(world_points, dx=True, dintrinsic=True, dextrinsic=True)

        Extracting the projected image points and jacobians:

        .. code-block:: python

            gray_levels = image_projection_result.image_points # The projected 2D image points of shape (4, )
            jacobian_dx = image_projection_result.jacobian_dx # The jacobian with respect to the world points of shape (4, 1, 3)
            jacobian_dintrinsic = image_projection_result.jacobian_dintrinsic # The jacobian with respect to the intrinsic parameters of shape (4, 1, 4)
            jacobian_dextrinsic = image_projection_result.jacobian_dextrinsic # The jacobian with respect to the extrinsic parameters of shape (4, 1, 6)

        
        """
        projection_result = self.camera.project(world_points, dx=dx, dintrinsic=dintrinsic, ddistortion=ddistortion, dextrinsic=dextrinsic)
        image_projection_result = self.assembly_image_projection(projection_result, world_points, dx=dx, dintrinsic=dintrinsic, ddistortion=ddistortion, dextrinsic=dextrinsic)
        return image_projection_result

    def image_project_points(self, world_points: PointCloud3D, dx: bool = False, dintrinsic: bool = False, ddistortion: bool = False, dextrinsic: bool = False) -> ImageProjectionResult:
        r"""
        Project 3D world points to gray level using the camera's intrinsic, extrinsic, distortion parameters and image interpolation function from a PointCloud3D instance.

        This method is a convenience wrapper around :meth:`image_project` that extracts the numpy array from the PointCloud3D instance.

        .. seealso::

            - :meth:`image_project` for the main projection functionality.
            - :class:`pysdic.geometry.PointCloud3D` for the structure of the input.

        .. seealso::

            - :meth:`pysdic.imaging.Camera.project` for the geometric projection process.
            - :meth:`assembly_image_projection` for the assembly of the image projection result.
            - :class:`ImageProjectionResult` for the structure of the output.

        Parameters
        ----------
        world_points : PointCloud3D
            The 3D world points to be projected.
        
        dx : bool, optional
            If True, the function will also return the jacobian of the gray levels with respect to the world points. Default is False.
        
        dintrinsic : bool, optional
            If True, compute the Jacobian of the gray levels with respect to the intrinsic parameters.
        
        ddistortion : bool, optional
            If True, compute the Jacobian of the gray levels with respect to the distortion parameters.
        
        dextrinsic : bool, optional
            If True, compute the Jacobian of the gray levels with respect to the extrinsic parameters.

        Returns
        -------
        ImageProjectionResult
            An instance of ImageProjectionResult containing:
            
            - `gray_levels`: A 1D array of shape (N, ) representing the image values at the projected pixel points.
            - `jacobian_dx`: (optional) A 3D array of shape (N, 1, 3) representing the jacobian of the normalized points with respect to the world points if `dx` is True.
            - `jacobian_dintrinsic`: (optional) A 3D array of shape (N, 1, Nintrinsic) representing the jacobian of the pixel points with respect to the intrinsic parameters if `dintrinsic` is True.
            - `jacobian_ddistortion`: (optional) A 3D array of shape (N, 1, Ndistortion) representing the jacobian of the pixel points with respect to the distortion parameters if `ddistortion` is True.
            - `jacobian_dextrinsic`: (optional) A 3D array of shape (N, 1, Nextrinsic) representing the jacobian of the pixel points with respect to the extrinsic parameters if `dextrinsic` is True.
        
        
        Examples
        --------

        Lets create a simple view and project some 3D points from a PointCloud3D instance:

        .. code-block:: python

            import numpy
            from pysdic.imaging import Camera
            from pysdic.geometry import PointCloud3D
            from pycvcam import Cv2Extrinsic, Cv2Intrinsic

            rotation_vector = numpy.array([0.1, 0.2, 0.3])
            translation_vector = numpy.array([12.0, 34.0, 56.0])

            extrinsic = Cv2Extrinsic.from_rt(rotation_vector, translation_vector)
            intrinsic = Cv2Intrinsic.from_matrix(
                numpy.array([[1000, 0, 320],
                            [0, 1000, 240],
                            [0, 0, 1]])
            )

            camera = Camera(
                sensor_height=480,
                sensor_width=640,
                intrinsic=intrinsic,
                extrinsic=extrinsic,
            )

            # Create a simple view with a blank image
            image = numpy.zeros((480, 640), dtype=numpy.uint8)
            view = View(camera=camera, image=image)

            # Define some 3D world points in a PointCloud3D instance
            world_points_array = numpy.array([
                [0, 0, 1000],
                [100, 0, 1000],
                [0, 100, 1000],
                [100, 100, 1000]
            ])
            world_points = PointCloud3D(world_points_array)

            # Project the 3D points to 2D image points
            image_projection_result = view.image_project_points(world_points, dx=True, dintrinsic=True, dextrinsic=True)

        Extracting the projected image points and jacobians:

        .. code-block:: python

            gray_levels = image_projection_result.image_points # The projected 2D image points of shape (4, )
            jacobian_dx = image_projection_result.jacobian_dx # The jacobian with respect to the world points of shape (4, 1, 3)
            jacobian_dintrinsic = image_projection_result.jacobian_dintrinsic # The jacobian with respect to the intrinsic parameters of shape (4, 1, 4)
            jacobian_dextrinsic = image_projection_result.jacobian_dextrinsic # The jacobian with respect to the extrinsic parameters of shape (4, 1, 6)
           

        """
        projection_result = self.camera.project(world_points, dx=dx, dintrinsic=dintrinsic, ddistortion=ddistortion, dextrinsic=dextrinsic)
        image_projection_result = self.assembly_image_projection(projection_result, world_points, dx=dx, dintrinsic=dintrinsic, ddistortion=ddistortion, dextrinsic=dextrinsic)
        return image_projection_result
    
    

    def assembly_image_projection(self, projection_result: ProjectionResult, dx: bool = False, dintrinsic: bool = False, ddistortion: bool = False, dextrinsic: bool = False) -> ImageProjectionResult:
        r"""
        Assemble the image projection result from the camera's projection result and the world points.

        .. warning::

            - If ``dx`` is True, the projection result must contain the jacobian of the image points with respect to the world points, otherwise it will raise an error.
            - If ``dp`` is True, the projection result must contain the jacobian of the image points with respect to the camera parameters, otherwise it will raise an error.

        The chain rule assembly is done as follows:

        .. math::

            \nabla_{\gamma}[IoP](\vec{X}) = \nabla_{x}[I](P(\vec{X})) \cdot \nabla_{\gamma}[P](\vec{X})

        Parameters
        ----------
        projection_result : pycvcam.core.TransformResult
            The result of the camera's projection method containing the projected pixel points and optionally the jacobians.
        
        dx : bool, optional
            If True, the function will also return the jacobian of the gray levels with respect to the world points. Default is False.
        
        dintrinsic : bool, optional
            If True, compute the Jacobian of the gray levels with respect to the intrinsic parameters.
        
        ddistortion : bool, optional
            If True, compute the Jacobian of the gray levels with respect to the distortion parameters.
        
        dextrinsic : bool, optional
            If True, compute the Jacobian of the gray levels with respect to the extrinsic parameters.

        Returns
        -------
        ImageProjectionResult
            An instance of ImageProjectionResult containing:
            
            - `gray_levels`: A 1D array of shape (N, ) representing the image values at the projected pixel points.
            - `jacobian_dx`: (optional) A 3D array of shape (N, 1, 3) representing the jacobian of the normalized points with respect to the world points if `dx` is True.
            - `jacobian_dintrinsic`: (optional) A 3D array of shape (N, 1, Nintrinsic) representing the jacobian of the pixel points with respect to the intrinsic parameters if `dintrinsic` is True.
            - `jacobian_ddistortion`: (optional) A 3D array of shape (N, 1, Ndistortion) representing the jacobian of the pixel points with respect to the distortion parameters if `ddistortion` is True.
            - `jacobian_dextrinsic`: (optional) A 3D array of shape (N, 1, Nextrinsic) representing the jacobian of the pixel points with respect to the extrinsic parameters if `dextrinsic` is True.
        """
        if not isinstance(projection_result, pycvcam.core.TransformResult):
            raise TypeError("projection_result must be an instance of pycvcam.core.TransformResult.")
        
        # Compute the image values at the projected pixel points
        pixel_points = self.camera.image_points_to_pixel_points(projection_result.image_points)
        gray_levels = self.evaluate_image_at_pixel_points(pixel_points)

        # Initialize jacobians as None
        jacobian_dx = None
        jacobian_dintrinsic = None
        jacobian_ddistortion = None
        jacobian_dextrinsic = None

        # Construct the image jacobian -> \nabla_{x}[I](P(\vec{X}))
        if dx or dintrinsic or ddistortion or dextrinsic:
            image_dx = self.evaluate_image_jacobian_dx_at_pixel_points(pixel_points) # Shape (N,)
            image_dy = self.evaluate_image_jacobian_dy_at_pixel_points(pixel_points) # Shape (N,)
            image_jacobian = numpy.empty((pixel_points.shape[0], 1, 2), dtype=numpy.float64)
            image_jacobian[:, 0, 0] = image_dx
            image_jacobian[:, 0, 1] = image_dy

        # Compute the jacobian with respect to dx if requested
        if dx:
            if projection_result.jacobian_dx is None:
                raise ValueError("Projection result must contain jacobian_dx if dx is True.")
            projection_dx = projection_result.jacobian_dx # shape (N, 2, 3)

            jacobian_dx = numpy.matmul(image_jacobian, projection_dx)  # (N, 1, 2) @ (N, 2, 3) = (N, 1, 3)

        # Compute the jacobian with respect to dintrinsic if requested
        if dintrinsic:
            if projection_result.jacobian_dintrinsic is None:
                raise ValueError("Projection result must contain jacobian_dintrinsic if dintrinsic is True.")
            projection_dintrinsic = projection_result.jacobian_dintrinsic  # shape (N, 2, Nintrinsic)

            jacobian_dintrinsic = numpy.matmul(image_jacobian, projection_dintrinsic)  # (N, 1, 2) @ (N, 2, Nintrinsic) = (N, 1, Nintrinsic)

        # Compute the jacobian with respect to ddistortion if requested
        if ddistortion:
            if projection_result.jacobian_ddistortion is None:
                raise ValueError("Projection result must contain jacobian_ddistortion if ddistortion is True.")
            projection_ddistortion = projection_result.jacobian_ddistortion

            jacobian_ddistortion = numpy.matmul(image_jacobian, projection_ddistortion)  # (N, 1, 2) @ (N, 2, Ndistortion) = (N, 1, Ndistortion)

        # Compute the jacobian with respect to dextrinsic if requested
        if dextrinsic:
            if projection_result.jacobian_dextrinsic is None:
                raise ValueError("Projection result must contain jacobian_dextrinsic if dextrinsic is True.")
            projection_dextrinsic = projection_result.jacobian_dextrinsic

            jacobian_dextrinsic = numpy.matmul(image_jacobian, projection_dextrinsic)  # (N, 1, 2) @ (N, 2, Nextrinsic) = (N, 1, Nextrinsic)

        # Create the ImageProjectionResult instance
        image_projection_result = ImageProjectionResult(
            gray_levels=gray_levels,
            jacobian_dx=jacobian_dx,
            jacobian_dintrinsic=jacobian_dintrinsic,
            jacobian_ddistortion=jacobian_ddistortion,
            jacobian_dextrinsic=jacobian_dextrinsic
        )
        return image_projection_result

    # ===================================================================
    # Visualization methods
    # ===================================================================
    def visualize_image(self):
        r"""
        Visualize the image using matplotlib.

        Raises
        ------
        ValueError
            If the image is not set.
        """
        if self._image is None:
            raise ValueError("Image is not set. Cannot visualize.")
        
        plt.imshow(self._image, cmap='gray', vmin=0, vmax=numpy.iinfo(self._image.dtype).max)
        plt.title("View Image")
        plt.axis('off')
        plt.show()

    def visualize_projected_point_cloud(
        self,
        point_cloud: PointCloud3D,
        point_color: str = "black",
        point_size: int = 5,
        clip_sensor: bool = True,
        show_pixel_grid: bool = False,
    ) -> None:
        r"""
        Visualize the projected 2D points of a :class:`pysdic.geometry.PointCloud3D` on a 2D plot using matplotlib.

        Simply calls the camera's own visualization method with image from the view.
        .. seealso::

            - :meth:`pysdic.imaging.Camera.visualize_projected_point_cloud` for the camera's own visualization method.

        Parameters
        ----------
        point_cloud : PointCloud3D
            An instance of PointCloud3D containing the 3D points in the world coordinate system to be projected and visualized.

        point_color : str, optional
            The color of the projected points in the plot. Default is "black".

        point_size : int, optional
            The size of the projected points in the plot. Default is 5.

        clip_sensor : bool, optional
            If True, only the points that are projected within the camera sensor dimensions are visualized. Default is True.

        show_pixel_grid : bool, optional
            If True, a grid representing the pixel layout of the camera sensor is displayed in the background. Default is False.

        """
        self.camera.visualize_projected_point_cloud(
            point_cloud=point_cloud,
            point_color=point_color,
            point_size=point_size,
            image=self._image,
            clip_sensor=clip_sensor,
            show_pixel_grid=show_pixel_grid,
        )

    def visualize_projected_mesh(
        self,
        mesh: Mesh3D,
        face_color: str = "red",
        face_opacity: float = 0.5,
        edge_color: str = "black",
        edge_width: int = 1,
        point_color: str = "black",
        point_size: int = 5,
        clip_sensor: bool = True,
        show_pixel_grid: bool = False,
        show_edges: bool = True,
        show_faces: bool = True,
    ) -> None:
        r"""
        Visualize the projected 2D mesh of a :class:`pysdic.geometry.Mesh3D` on a 2D plot using matplotlib.

        Simply calls the camera's own visualization method with image from the view.

        .. seealso::

            - :meth:`pysdic.imaging.Camera.visualize_projected_mesh` for the camera's own visualization method.

        Parameters
        ----------
        mesh : Mesh3D
            The 3D mesh to visualize.

        face_color : str, optional
            The color of the mesh faces (default is "red").

        face_opacity : float, optional
            The opacity of the mesh faces (default is 0.5).

        edge_color : str, optional
            The color of the mesh edges (default is "black").

        edge_width : int, optional
            The width of the mesh edges (default is 1).

        point_color : str, optional
            The color of the mesh vertices (default is "black").

        point_size : int, optional
            The size of the mesh vertices (default is 5).

        image : Optional[numpy.ndarray], optional
            An image to display as the background (default is None).

        clip_sensor : bool, optional
            Whether to clip points outside the sensor dimensions (default is True).

        show_pixel_grid : bool, optional
            Whether to show the pixel grid on the image (default is False).

        show_edges : bool, optional
            Whether to show the mesh edges (default is True).

        show_faces : bool, optional
            Whether to show the mesh faces (default is True).
        """
        self.camera.visualize_projected_mesh(
            mesh=mesh,
            face_color=face_color,
            face_opacity=face_opacity,
            edge_color=edge_color,
            edge_width=edge_width,
            point_color=point_color,
            point_size=point_size,
            image=self._image,
            clip_sensor=clip_sensor,
            show_pixel_grid=show_pixel_grid,
            show_edges=show_edges,
            show_faces=show_faces,
        )