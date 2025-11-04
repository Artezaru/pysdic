import numpy
import scipy
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pycvcam

from .image_projection_result import ImageProjectionResult
from .projection_result import ProjectionResult

from .camera import Camera
from .image import Image

from ..geometry.point_cloud_3d import PointCloud3D
from ..geometry.mesh_3d import Mesh3D

class View(object):
    r"""
    A view is a constructed by a camera and an image acquired by the camera.

    If the sensor size of the camera changes, the image must be updated to reflect the new sensor dimensions.

    .. warning::

        - If multiple views share the same camera, a change on the camera's parameters will affect all views using that camera.
        - If multiple views share the same image, a change on the image will affect all views using that image.
        - An error will be raised if the image shape does not match the camera sensor size or if the camera sensor size or image shape change after the view creation.

    .. seealso::

        - :class:`Camera` for camera model and parameters.
        - :class:`Image` for image representation and processing.

    Parameters
    ----------
    camera : Camera
        The camera that will be used to view the image.
    
    image : Image
        The image that is viewed by the camera.

    """

    __slots__ = [
        "_camera", 
        "_image", 
        "_saved_camera_size",
        "_saved_image_shape",
    ]

    def __init__(self, camera: Camera, image: Image):
        self.camera = camera
        self.image = image


    def _check_no_size_updates(self) -> None:
        r"""
        Check that no major updates have occurred to the camera or image.

        This method checks if the camera's sensor size or the image's shape has changed since the last check.
        If any changes are detected, a warning is printed to inform the user.

        """
        if self.camera_size != self._saved_camera_size:
            raise ValueError(f"[View] Warning: Camera size has changed from {self._saved_camera_size} to {self.camera_size} since last check. Create a new View with the updated camera.")

        if self.image_shape != self._saved_image_shape:
            raise ValueError(f"[View] Warning: Image shape has changed from {self._saved_image_shape} to {self.image_shape} since last check. Create a new View with the updated image.")

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
        # Save the current camera size for future checks
        self._saved_camera_size = (camera.sensor_height, camera.sensor_width)


    @property
    def image(self) -> Image:
        r"""
        Get or set the image viewed by the camera.

        Parameters
        ----------
        image : Image
            The image to be viewed by the camera.
        """
        return self._image
    
    @image.setter
    def image(self, image: Image):
        if not isinstance(image, Image):
            raise TypeError("Image must be an instance of Image.")
        if image.image is None:
            raise ValueError("Image data is None. Please set a valid image.")
        if not image.shape == (self.camera.sensor_height, self.camera.sensor_width):
            raise ValueError(f"Image shape {image.shape} does not match camera sensor size {self.camera.sensor_height, self.camera.sensor_width}.")
        self._image = image
        # Save the current image shape for future checks
        self._saved_image_shape = image.shape


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
        return self._camera.sensor_height, self._camera.sensor_width


    # ===================================================================
    # Application composition methods
    # ===================================================================
    def project(self, world_points: numpy.ndarray, dx: bool = False, dintrinsic: bool = False, ddistortion: bool = False, dextrinsic: bool = False) -> ProjectionResult:
        r"""
        Project 3D world points to 2D pixel points using the camera's intrinsic, extrinsic, and distortion parameters.

        This method is a convenience wrapper around the camera's own project method.

        .. seealso::

            - :meth:`pysdic.imaging.Camera.project` for the camera's own projection method.

        Parameters
        ----------
        world_points : numpy.ndarray
            The 3D world points to be projected. The shape should be (N, 3) where N is the number of points and each point is represented by its (x, y, z) coordinates.

        dx : bool, optional
            If True, the function will also return the jacobian of the pixel points with respect to the world points. Default is False.

        dintrinsic : bool, optional
            If True, compute the Jacobian of the pixel points with respect to the intrinsic parameters.

        ddistortion : bool, optional
            If True, compute the Jacobian of the pixel points with respect to the distortion parameters.

        dextrinsic : bool, optional
            If True, compute the Jacobian of the pixel points with respect to the extrinsic parameters.

        Returns
        -------
        ProjectionResult
            A ProjectionResult object containing the projected image points and optionally the jacobians.

            - `image_points`: An array of shape (..., 2) representing the projected images points in the image coordinate system (x, y).
            - `jacobian_dx`: (optional) A 3D array of shape (..., 2, 3) representing the jacobian of the normalized points with respect to the world points if `dx` is True.
            - `jacobian_dintrinsic`: (optional) A 3D array of shape (..., 2, Nintrinsic) representing the jacobian of the pixel points with respect to the intrinsic parameters if `dintrinsic` is True.
            - `jacobian_ddistortion`: (optional) A 3D array of shape (..., 2, Ndistortion) representing the jacobian of the pixel points with respect to the distortion parameters if `ddistortion` is True.
            - `jacobian_dextrinsic`: (optional) A 3D array of shape (..., 2, Nextrinsic) representing the jacobian of the pixel points with respect to the extrinsic parameters if `dextrinsic` is True.

        """
        self._check_no_size_updates()

        projection_result = self.camera.project(world_points, dx=dx, dintrinsic=dintrinsic, ddistortion=ddistortion, dextrinsic=dextrinsic)
        return projection_result
    
    def project_points(self, world_points: PointCloud3D, dx: bool = False, dintrinsic: bool = False, ddistortion: bool = False, dextrinsic: bool = False) -> ProjectionResult:
        r"""
        Project 3D world points to 2D pixel points using the camera's intrinsic, extrinsic, and distortion parameters from a PointCloud3D instance.

        This method is a convenience wrapper around the camera's own project method.

        .. seealso::

            - :class:`pysdic.geometry.PointCloud3D` for the structure of the input.
            - :meth:`pysdic.imaging.Camera.project_points` for the camera's own projection method.

        Parameters
        ----------
        world_points : PointCloud3D
            The 3D world points to be projected.

        dx : bool, optional
            If True, the function will also return the jacobian of the pixel points with respect to the world points. Default is False.

        dintrinsic : bool, optional
            If True, compute the Jacobian of the pixel points with respect to the intrinsic parameters.

        ddistortion : bool, optional
            If True, compute the Jacobian of the pixel points with respect to the distortion parameters.

        dextrinsic : bool, optional
            If True, compute the Jacobian of the pixel points with respect to the extrinsic parameters.

        Returns
        -------
        ProjectionResult
            A ProjectionResult object containing the projected image points and optionally the jacobians.

            - `image_points`: An array of shape (..., 2) representing the projected images points in the image coordinate system (x, y).
            - `jacobian_dx`: (optional) A 3D array of shape (..., 2, 3) representing the jacobian of the normalized points with respect to the world points if `dx` is True.
            - `jacobian_dintrinsic`: (optional) A 3D array of shape (..., 2, Nintrinsic) representing the jacobian of the pixel points with respect to the intrinsic parameters if `dintrinsic` is True.
            - `jacobian_ddistortion`: (optional) A 3D array of shape (..., 2, Ndistortion) representing the jacobian of the pixel points with respect to the distortion parameters if `ddistortion` is True.
            - `jacobian_dextrinsic`: (optional) A 3D array of shape (..., 2, Nextrinsic) representing the jacobian of the pixel points with respect to the extrinsic parameters if `dextrinsic` is True.
        
        """
        self._check_no_size_updates()

        projection_result = self.camera.project_points(world_points, dx=dx, dintrinsic=dintrinsic, ddistortion=ddistortion, dextrinsic=dextrinsic)
        return projection_result


    def image_project(self, world_points: numpy.ndarray, dx: bool = False, dintrinsic: bool = False, ddistortion: bool = False, dextrinsic: bool = False) -> ImageProjectionResult:
        r"""
        Project 3D world points to gray level or image values using the camera's intrinsic, extrinsic, distortion parameters and image interpolation function.

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

            - `gray_levels`: 2D array of shape (N, channels) representing the image values at the projected pixel points.
            - `jacobian_dx`: (optional) A 3D array of shape (N, 1, 3) representing the jacobian of the normalized points with respect to the world points if `dx` is True.
            - `jacobian_dintrinsic`: (optional) A 3D array of shape (N, 1, Nintrinsic) representing the jacobian of the pixel points with respect to the intrinsic parameters if `dintrinsic` is True.
            - `jacobian_ddistortion`: (optional) A 3D array of shape (N, 1, Ndistortion) representing the jacobian of the pixel points with respect to the distortion parameters if `ddistortion` is True.
            - `jacobian_dextrinsic`: (optional) A 3D array of shape (N, 1, Nextrinsic) representing the jacobian of the pixel points with respect to the extrinsic parameters if `dextrinsic` is True.
            
        Examples
        --------

        Lets create a simple view and project some 3D points:

        .. code-block:: python

            import numpy
            from pysdic.imaging import Camera, Image, View
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
            image = Image.from_array(numpy.zeros((480, 640), dtype=numpy.uint8))
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

            gray_levels = image_projection_result.image_points # The projected 2D image points of shape (4, 1)
            jacobian_dx = image_projection_result.jacobian_dx # The jacobian with respect to the world points of shape (4, 1, 3)
            jacobian_dintrinsic = image_projection_result.jacobian_dintrinsic # The jacobian with respect to the intrinsic parameters of shape (4, 1, 4)
            jacobian_dextrinsic = image_projection_result.jacobian_dextrinsic # The jacobian with respect to the extrinsic parameters of shape (4, 1, 6)
        
        """
        self._check_no_size_updates()

        projection_result = self.camera.project(world_points, dx=dx, dintrinsic=dintrinsic, ddistortion=ddistortion, dextrinsic=dextrinsic)
        image_projection_result = self.assembly_image_projection(projection_result, dx=dx, dintrinsic=dintrinsic, ddistortion=ddistortion, dextrinsic=dextrinsic)
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

            - `gray_levels`: A 2D array of shape (N, channels) representing the image values at the projected pixel points.
            - `jacobian_dx`: (optional) A 3D array of shape (N, 1, 3) representing the jacobian of the normalized points with respect to the world points if `dx` is True.
            - `jacobian_dintrinsic`: (optional) A 3D array of shape (N, 1, Nintrinsic) representing the jacobian of the pixel points with respect to the intrinsic parameters if `dintrinsic` is True.
            - `jacobian_ddistortion`: (optional) A 3D array of shape (N, 1, Ndistortion) representing the jacobian of the pixel points with respect to the distortion parameters if `ddistortion` is True.
            - `jacobian_dextrinsic`: (optional) A 3D array of shape (N, 1, Nextrinsic) representing the jacobian of the pixel points with respect to the extrinsic parameters if `dextrinsic` is True.
        
        
        Examples
        --------

        Lets create a simple view and project some 3D points from a PointCloud3D instance:

        .. code-block:: python

            import numpy
            from pysdic.imaging import Camera, Image, View
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
            image = Image.from_array(numpy.zeros((480, 640), dtype=numpy.uint8))
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

            gray_levels = image_projection_result.image_points # The projected 2D image points of shape (4, 1)
            jacobian_dx = image_projection_result.jacobian_dx # The jacobian with respect to the world points of shape (4, 1, 3)
            jacobian_dintrinsic = image_projection_result.jacobian_dintrinsic # The jacobian with respect to the intrinsic parameters of shape (4, 1, 4)
            jacobian_dextrinsic = image_projection_result.jacobian_dextrinsic # The jacobian with respect to the extrinsic parameters of shape (4, 1, 6)
           

        """
        self._check_no_size_updates()

        projection_result = self.camera.project(world_points, dx=dx, dintrinsic=dintrinsic, ddistortion=ddistortion, dextrinsic=dextrinsic)
        image_projection_result = self.assembly_image_projection(projection_result, dx=dx, dintrinsic=dintrinsic, ddistortion=ddistortion, dextrinsic=dextrinsic)
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
        projection_result : ProjectionResult
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

            - `gray_levels`: A 2D array of shape (N, channels) representing the image values at the projected pixel points.
            - `jacobian_dx`: (optional) A 3D array of shape (N, 1, 3) representing the jacobian of the normalized points with respect to the world points if `dx` is True.
            - `jacobian_dintrinsic`: (optional) A 3D array of shape (N, 1, Nintrinsic) representing the jacobian of the pixel points with respect to the intrinsic parameters if `dintrinsic` is True.
            - `jacobian_ddistortion`: (optional) A 3D array of shape (N, 1, Ndistortion) representing the jacobian of the pixel points with respect to the distortion parameters if `ddistortion` is True.
            - `jacobian_dextrinsic`: (optional) A 3D array of shape (N, 1, Nextrinsic) representing the jacobian of the pixel points with respect to the extrinsic parameters if `dextrinsic` is True.
        
        """
        self._check_no_size_updates()

        if not isinstance(projection_result,  ProjectionResult):
            raise TypeError("projection_result must be an instance of ProjectionResult.")

        # Compute the image values at the projected pixel points
        pixel_points = self.camera.image_points_to_pixel_points(projection_result.image_points)
        gray_levels = self.image.evaluate_image_at_pixel_points(pixel_points) # Shape (N,) or (N, channels)
        gray_levels = gray_levels.reshape((pixel_points.shape[0], -1))  # Ensure shape is (N, channels)

        # Initialize jacobians as None
        jacobian_dx = None
        jacobian_dintrinsic = None
        jacobian_ddistortion = None
        jacobian_dextrinsic = None

        # Construct the image jacobian -> \nabla_{x}[I](P(\vec{X}))
        if dx or dintrinsic or ddistortion or dextrinsic:
            image_dx = self.image.evaluate_image_jacobian_dx_at_pixel_points(pixel_points) # Shape (N,) or (N, channels)
            image_dy = self.image.evaluate_image_jacobian_dy_at_pixel_points(pixel_points) # Shape (N,) or (N, channels)
            image_dx = image_dx.reshape((pixel_points.shape[0], -1))  # Ensure shape is (N, channels)
            image_dy = image_dy.reshape((pixel_points.shape[0], -1))  # Ensure shape is (N, channels)
            image_jacobian = numpy.empty((pixel_points.shape[0], self.image.n_channels, 2), dtype=numpy.float64)
            for channel in range(self.image.n_channels):
                image_jacobian[:, channel, 0] = image_dx[:, channel]
                image_jacobian[:, channel, 1] = image_dy[:, channel]

        # Compute the jacobian with respect to dx if requested
        if dx:
            if projection_result.jacobian_dx is None:
                raise ValueError("Projection result must contain jacobian_dx if dx is True.")
            projection_dx = projection_result.jacobian_dx # shape (N, 2, 3)

            jacobian_dx = numpy.matmul(image_jacobian, projection_dx)  # (N, C, 2) @ (N, 2, 3) = (N, C, 3)

        # Compute the jacobian with respect to dintrinsic if requested
        if dintrinsic:
            if projection_result.jacobian_dintrinsic is None:
                raise ValueError("Projection result must contain jacobian_dintrinsic if dintrinsic is True.")
            projection_dintrinsic = projection_result.jacobian_dintrinsic  # shape (N, 2, Nintrinsic)

            jacobian_dintrinsic = numpy.matmul(image_jacobian, projection_dintrinsic)  # (N, C, 2) @ (N, 2, Nintrinsic) = (N, C, Nintrinsic)

        # Compute the jacobian with respect to ddistortion if requested
        if ddistortion:
            if projection_result.jacobian_ddistortion is None:
                raise ValueError("Projection result must contain jacobian_ddistortion if ddistortion is True.")
            projection_ddistortion = projection_result.jacobian_ddistortion

            jacobian_ddistortion = numpy.matmul(image_jacobian, projection_ddistortion)  # (N, C, 2) @ (N, 2, Ndistortion) = (N, C, Ndistortion)

        # Compute the jacobian with respect to dextrinsic if requested
        if dextrinsic:
            if projection_result.jacobian_dextrinsic is None:
                raise ValueError("Projection result must contain jacobian_dextrinsic if dextrinsic is True.")
            projection_dextrinsic = projection_result.jacobian_dextrinsic

            jacobian_dextrinsic = numpy.matmul(image_jacobian, projection_dextrinsic)  # (N, C, 2) @ (N, 2, Nextrinsic) = (N, C, Nextrinsic)

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
        points_color: str = "black",
        points_size: int = 5,
        points_opacity: float = 1.0,
        clip_sensor: bool = True,
        show_pixel_grid: bool = False,
        title: Optional[str] = None,
    ) -> None:
        r"""
        Visualize the projected 2D points of a :class:`pysdic.geometry.PointCloud3D` on a 2D plot using matplotlib.

        Simply calls the camera's own visualization method with image from the view.

        .. seealso::

            - :meth:`pysdic.imaging.Camera.visualize_projected_point_cloud` for the camera's own visualization method.
            - :meth:`pysdic.imaging.View.visualize_projected_mesh` for visualizing projected meshes.
            - :meth:`pysdic.geometry.View.visualize_image` for visualizing the view's image.
            - :meth:`pysdic.geometry.PointCloud3D` for the structure of the input point cloud.

        Parameters
        ----------
        point_cloud : PointCloud3D
            An instance of PointCloud3D containing the 3D points in the world coordinate system to be projected and visualized.

        points_color : str, optional
            The color of the projected points in the plot. Default is "black".

        points_size : int, optional
            The size of the projected points in the plot. Default is 5.

        points_opacity : float, optional
            The opacity of the projected points in the plot. Default is 1.0 (fully opaque).

        clip_sensor : bool, optional
            If True, only the points that are projected within the camera sensor dimensions are visualized. Default is True.

        show_pixel_grid : bool, optional
            If True, a grid representing the pixel layout of the camera sensor is displayed in the background. Default is False.

        title : Optional[str], optional
            An optional title for the plot. If None, no title is displayed. Default is None.

        """
        self.camera.visualize_projected_point_cloud(
            point_cloud=point_cloud,
            points_color=points_color,
            points_size=points_size,
            points_opacity=points_opacity,
            image=self._image,
            clip_sensor=clip_sensor,
            show_pixel_grid=show_pixel_grid,
            title=title,
        )

    def visualize_projected_mesh(
        self,
        mesh: Mesh3D,
        vertices_color: str = "black",
        vertices_size: int = 5,
        vertices_opacity: float = 1.0,
        edges_color: str = "black",
        edges_width: int = 1,
        edges_opacity: float = 1.0,
        faces_color: str = "red",
        faces_opacity: float = 0.5,
        clip_sensor: bool = True,
        show_pixel_grid: bool = False,
        show_vertices: bool = True,
        show_edges: bool = True,
        show_faces: bool = True,
        title: Optional[str] = None,
    ) -> None:
        r"""
        Visualize the projected 2D mesh of a :class:`pysdic.geometry.Mesh3D` on a 2D plot using matplotlib.

        Simply calls the camera's own visualization method with image from the view.

        .. seealso::

            - :meth:`pysdic.imaging.Camera.visualize_projected_mesh` for the camera's own visualization method.
            - :meth:`pysdic.imaging.View.visualize_projected_point_cloud` for visualizing projected point clouds.
            - :meth:`pysdic.geometry.View.visualize_image` for visualizing the view's image.
            - :meth:`pysdic.geometry.Mesh3D` for the structure of the input mesh.

        Parameters
        ----------
        mesh : Mesh3D
            The 3D mesh to visualize.

        vertices_color : str, optional
            The color of the mesh vertices (default is "black").

        vertices_size : int, optional
            The size of the mesh vertices (default is 5).

        vertices_opacity : float, optional
            The opacity of the mesh vertices (default is 1.0).

        edges_color : str, optional
            The color of the mesh edges (default is "black").

        edges_width : int, optional
            The width of the mesh edges (default is 1).

        edges_opacity : float, optional
            The opacity of the mesh edges (default is 1.0).

        faces_color : str, optional
            The color of the mesh faces (default is "red").

        faces_opacity : float, optional
            The opacity of the mesh faces (default is 0.5).

        clip_sensor : bool, optional
            Whether to clip points outside the sensor dimensions (default is True).

        show_pixel_grid : bool, optional
            Whether to show the pixel grid on the image (default is False).

        show_vertices : bool, optional
            Whether to show the mesh vertices (default is True).

        show_edges : bool, optional
            Whether to show the mesh edges (default is True).

        show_faces : bool, optional
            Whether to show the mesh faces (default is True).
        
        title : Optional[str], optional
            An optional title for the plot. If None, no title is displayed. Default is None.

        """
        self.camera.visualize_projected_mesh(
            mesh=mesh,
            vertices_color=vertices_color,
            vertices_size=vertices_size,
            vertices_opacity=vertices_opacity,
            edges_color=edges_color,
            edges_width=edges_width,
            edges_opacity=edges_opacity,
            faces_color=faces_color,
            faces_opacity=faces_opacity,
            image=self._image,
            clip_sensor=clip_sensor,
            show_pixel_grid=show_pixel_grid,
            show_vertices=show_vertices,
            show_edges=show_edges,
            show_faces=show_faces,
            title=title,
        )