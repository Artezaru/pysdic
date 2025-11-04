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

import numpy
from typing import Optional, Tuple, Union
from numbers import Integral

import matplotlib.pyplot as plt
import pycvcam

from .projection_result import ProjectionResult
from ..geometry.point_cloud_3d import PointCloud3D
from ..geometry.mesh_3d import Mesh3D

from .image import Image

class Camera(object):
    r"""
    A camera is a structure with :

    - a sensor size (height and width).
    - an ``Extrinsic`` transformation.
    - an ``Distortion`` transformation.
    - an ``Intrinsic`` transformation.

    .. seealso::

        Package ``pycvcam`` (https://github.com/Artezaru/pycvcam) to learn more about the camera model.

    As described in ``pycvcam``, the height of the image is along the y-axis and the width of the image is along the x-axis.
    As described in the figure below, the package ``pycvcam`` uses the following notation:

    - ``world_points``: The 3-D points :math:`\vec{X}_w` with shape (...,3) expressed in the world coordinate system :math:`(\vec{E}_x, \vec{E}_y, \vec{E}_z)`.
    - ``normalized_points``: The 2-D points :math:`\vec{x}_n` with shape (...,2) expressed in the normalized camera coordinate system :math:`(\vec{I}, \vec{J})` with a unit distance along the optical axis :math:`(\vec{K})`.
    - ``distorted_points``: The distorted 2-D points :math:`\vec{x}_d` with shape (...,2) expressed in the normalized camera coordinate system :math:`(\vec{I}, \vec{J})` with a unit distance along the optical axis :math:`(\vec{K})`.
    - ``image_points``: The 2-D points :math:`\vec{x}_i` with shape (...,2) expressed in the image coordinate system :math:`(\vec{e}_x, \vec{e}_y)` in the sensor plane.
    - ``pixel_points``: The 2-D points :math:`\vec{x}_p` with shape (...,2) expressed in the pixel coordinate system :math:`(u, v)` in the matrix of pixels.

    .. figure:: ../../../pysdic/resources/definition_pycvcam.png
        :align: center
        :width: 60%
        
        Definition of quantities in ``pycvcam``.

    To convert the ``image_points`` to the ``pixel_points``, a simple switch of coordinate system can be performed.
    You can use the methods ``pixel_points_to_image_points`` and ``image_points_to_pixel_points`` to perform these conversions.

    Parameters
    ----------
    sensor_height : Integral
        The height of the camera sensor in pixels.

    sensor_width : Integral
        The width of the camera sensor in pixels.

    intrinsic : Optional[pycvcam.core.Intrinsic]
        The intrinsic transformation of the camera. Default is None meaning no intrinsic transformation (identity transformation).
   
    distortion : Optional[pycvcam.core.Distortion]
        The distortion transformation of the camera. Default is None meaning no distortion transformation (identity transformation).

    extrinsic : Optional[pycvcam.core.Extrinsic]
        The extrinsic transformation of the camera. Default is None meaning no extrinsic transformation (identity transformation).

    """
    __slots__ = [
        '_internal_bypass',
        '_sensor_height', 
        '_sensor_width', 
        '_extrinsic',
        '_extrinsic_parameters_save_back',
        '_extrinsic_constants_save_back',
        '_distortion', 
        '_distortion_parameters_save_back',
        '_distortion_constants_save_back',
        '_intrinsic', 
        '_intrinsic_parameters_save_back',
        '_intrinsic_constants_save_back',
        '_normalized_points', 
    ]

    _attached_views: list = []

    def __init__(
            self,
            sensor_height: Optional[int],
            sensor_width: Optional[int],
            intrinsic: Optional[pycvcam.core.Intrinsic] = None,
            distortion: Optional[pycvcam.core.Distortion] = None,
            extrinsic: Optional[pycvcam.core.Extrinsic] = None
    ):
        # Default values for properties
        self._internal_bypass = False
        self._sensor_height = None
        self._sensor_width = None
        self._extrinsic = None
        self._distortion = None
        self._intrinsic = None
        self._extrinsic_parameters_save_back = None
        self._extrinsic_constants_save_back = None
        self._distortion_parameters_save_back = None
        self._distortion_constants_save_back = None
        self._intrinsic_parameters_save_back = None
        self._intrinsic_constants_save_back = None
        self._normalized_points = None

        # Set the properties with given values
        self.sensor_height = sensor_height
        self.sensor_width = sensor_width
        self.extrinsic = extrinsic
        self.distortion = distortion
        self.intrinsic = intrinsic

    # =======================================================================
    # Internal Methods
    # =======================================================================
    @property
    def internal_bypass(self) -> bool:
        r"""
        Get and set the internal bypass mode status.
        When enabled, internal checks are skipped.

        This is useful for testing purposes, but should not be used in production code.
        If this parameters is set to True, the user must manage the updates manually.

        The updates must be call when an object ``Extrinsic``, ``Distortion`` or ``Intrinsic`` is modified.
        Meaning that the parameters or constants of the transformation has been changed.

        .. code-block::

            camera.internal_bypass = True
            camera.extrinsic.parameters = <parameters>
            camera.extrinsic_update()

        If ``internal_bypass`` is set to False, the class stored a copy of the parameters and constants of the extrinsic, distortion and intrinsic transformations.
        If any changes are detected, the update methods are called automatically.
        This action impacts the memory usage of the class and time computation by comparing the stored values with the current values at each call.

        By default, ``internal_bypass`` is set to False when the camera is created.

        .. seealso::

            The methods to update the camera's transformations to call when ``internal_bypass`` is set to True.

            - :meth:`extrinsic_update`
            - :meth:`distortion_update`
            - :meth:`intrinsic_update`
            - :meth:`update`

            The package ``pycvcam`` (https://github.com/Artezaru/pycvcam) to deal with camera transformations.

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
        self._save_back_extrinsic()
        self._save_back_distortion()
        self._save_back_intrinsic()
        self.update() # Ensure all changes are applied before starting with new internal bypass state


    def _save_back_extrinsic(self) -> None:
        r"""
        Save back the extrinsic parameters and constants.

        If ``internal_bypass`` is set to False, the class stores a copy of the parameters and constants of the extrinsic transformation.
        If any changes are detected, the update methods are called automatically.
        This action impacts the memory usage of the class and time computation by comparing the stored values with the current values at each call.

        .. seealso::

            - The attribute :attr:`internal_bypass` to switch between internal and external update modes.
            - The package ``pycvcam`` (https://github.com/Artezaru/pycvcam) to deal with the extrinsic transformation.

        """
        if self._internal_bypass:
            self._extrinsic_parameters_save_back = None
            self._extrinsic_constants_save_back = None
        else:
            self._extrinsic_parameters_save_back = numpy.copy(self._extrinsic.parameters) if self._extrinsic is not None and self._extrinsic.parameters is not None else None
            self._extrinsic_constants_save_back = numpy.copy(self._extrinsic.constants) if self._extrinsic is not None and self._extrinsic.constants is not None else None


    def _save_back_distortion(self) -> None:
        r"""
        Save back the distortion parameters and constants.

        If ``internal_bypass`` is set to False, the class stores a copy of the parameters and constants of the distortion transformation.
        If any changes are detected, the update methods are called automatically.
        This action impacts the memory usage of the class and time computation by comparing the stored values with the current values at each call.

        .. seealso::

            - The attribute :attr:`internal_bypass` to switch between internal and external update modes.
            - The package ``pycvcam`` (https://github.com/Artezaru/pycvcam) to deal with the distortion transformation.
    
        """
        if self._internal_bypass:
            self._distortion_parameters_save_back = None
            self._distortion_constants_save_back = None
        else:
            self._distortion_parameters_save_back = numpy.copy(self._distortion.parameters) if self._distortion is not None and self._distortion.parameters is not None else None
            self._distortion_constants_save_back = numpy.copy(self._distortion.constants) if self._distortion is not None and self._distortion.constants is not None else None


    def _save_back_intrinsic(self) -> None:
        r"""
        Save back the intrinsic parameters and constants.

        If ``internal_bypass`` is set to False, the class stores a copy of the parameters and constants of the intrinsic transformation.
        If any changes are detected, the update methods are called automatically.
        This action impacts the memory usage of the class and time computation by comparing the stored values with the current values at each call.

        .. seealso::

            - The attribute :attr:`internal_bypass` to switch between internal and external update modes.
            - The package ``pycvcam`` (https://github.com/Artezaru/pycvcam) to deal with the intrinsic transformation.
        """
        if self._internal_bypass:
            self._intrinsic_parameters_save_back = None
            self._intrinsic_constants_save_back = None
        else:
            self._intrinsic_parameters_save_back = numpy.copy(self._intrinsic.parameters) if self._intrinsic is not None and self._intrinsic.parameters is not None else None
            self._intrinsic_constants_save_back = numpy.copy(self._intrinsic.constants) if self._intrinsic is not None and self._intrinsic.constants is not None else None


    def _is_extrinsic_changed(self) -> bool:
        r"""
        This method checks if there are any changes in the extrinsic parameters or constants.

        .. note::

            If ``internal_bypass`` is set to True, this method will always return False.

        Returns
        -------
        bool
            True if there are changes, False otherwise.
        """
        if self._internal_bypass:
            return False
        if self._extrinsic is None:
            return self._extrinsic_parameters_save_back is not None or self._extrinsic_constants_save_back is not None

        parameters_changed = False
        if self._extrinsic.parameters is None:
            parameters_changed = self._extrinsic_parameters_save_back is not None
        else:
            parameters_changed = not numpy.array_equal(self._extrinsic.parameters, self._extrinsic_parameters_save_back) if self._extrinsic_parameters_save_back is not None else True

        constants_changed = False
        if self._extrinsic.constants is None:
            constants_changed = self._extrinsic_constants_save_back is not None
        else:
            constants_changed = not numpy.array_equal(self._extrinsic.constants, self._extrinsic_constants_save_back) if self._extrinsic_constants_save_back is not None else True

        return parameters_changed or constants_changed
    

    def _is_distortion_changed(self) -> bool:
        r"""
        This method checks if there are any changes in the distortion parameters or constants.

        .. note::

            If ``internal_bypass`` is set to True, this method will always return False.

        Returns
        -------
        bool
            True if there are changes, False otherwise.
        """
        if self._internal_bypass:
            return False
        if self._distortion is None:
            return self._distortion_parameters_save_back is not None or self._distortion_constants_save_back is not None

        parameters_changed = False
        if self._distortion.parameters is None:
            parameters_changed = self._distortion_parameters_save_back is not None
        else:
            parameters_changed = not numpy.array_equal(self._distortion.parameters, self._distortion_parameters_save_back) if self._distortion_parameters_save_back is not None else True

        constants_changed = False
        if self._distortion.constants is None:
            constants_changed = self._distortion_constants_save_back is not None
        else:
            constants_changed = not numpy.array_equal(self._distortion.constants, self._distortion_constants_save_back) if self._distortion_constants_save_back is not None else True

        return parameters_changed or constants_changed


    def _is_intrinsic_changed(self) -> bool:
        r"""
        This method checks if there are any changes in the intrinsic parameters or constants.

        .. note::

            If ``internal_bypass`` is set to True, this method will always return False.

        Returns
        -------
        bool
            True if there are changes, False otherwise.
        """
        if self._internal_bypass:
            return False
        if self._intrinsic is None:
            return self._intrinsic_parameters_save_back is not None or self._intrinsic_constants_save_back is not None

        parameters_changed = False
        if self._intrinsic.parameters is None:
            parameters_changed = self._intrinsic_parameters_save_back is not None
        else:
            parameters_changed = not numpy.array_equal(self._intrinsic.parameters, self._intrinsic_parameters_save_back) if self._intrinsic_parameters_save_back is not None else True

        constants_changed = False
        if self._intrinsic.constants is None:
            constants_changed = self._intrinsic_constants_save_back is not None
        else:
            constants_changed = not numpy.array_equal(self._intrinsic.constants, self._intrinsic_constants_save_back) if self._intrinsic_constants_save_back is not None else True

        return parameters_changed or constants_changed


    def _check_and_perform_changes(self):
        r"""
        This method updates the class if a change in the transformations is detected.

        - updating the saved back parameters and constants for Extrinsic, Distortion, and Intrinsic.
        - calling the update methods for Extrinsic, Distortion, and Intrinsic.

        If ``internal_bypass`` is set to True, this method will not perform any updates. The user is responsible for managing the state of the camera.
        """
        if self._internal_bypass:
            return
        
        if self._is_extrinsic_changed():
            self._save_back_extrinsic()
            self.extrinsic_update()

        if self._is_distortion_changed():
            self._save_back_distortion()
            self.distortion_update()

        if self._is_intrinsic_changed():
            self._save_back_intrinsic()
            self.intrinsic_update()



    def _attach_view(self, view: "View") -> None:
        r"""
        Attach a view to the camera.

        This method is called when a view is created to propagate changes on the width and the height of the camera.

        .. seealso::

            - Class :class:`pysdic.imaging.View` to represent a camera view.
            - Method :meth:`_detach_view` to remove a view from the camera.

        .. warning::

            To avoid circular error during importation, the method can't check if the given object is an instance of the View class.
            Please ensure that the view is a valid instance of the View class before attaching it to the camera.

        Parameters
        ----------
        view : View
            The view to be attached to the camera.
        """
        if not view in self._attached_views:
            self._attached_views.append(view)


    def _detach_view(self, view: "View") -> None:
        r"""
        Detach a view from the camera.

        This method is called when a view is destroyed to stop propagating changes on the width and the height of the camera.

        .. seealso::
            
            - Class :class:`pysdic.imaging.View` to represent a camera view.
            - Method :meth:`_attach_view` to add a view from the camera.

        .. warning::

            To avoid circular error during importation, the method can't check if the given object is an instance of the View class.
            Please ensure that the view is a valid instance of the View class before attaching it to the camera.
        """
        if view in self._attached_views:
            self._attached_views.remove(view)


    def _notify_views(self) -> None:
        r"""
        Notify all attached views about a change in the camera parameters.

        This method is called whenever the camera parameters are updated to propagate the changes
        to all attached views. This method calls the ``camera_size_change`` method on each view.

        .. seealso::

            - Class :class:`pysdic.imaging.View` to represent a camera view.
            - Method :meth:`_attach_view` to add a view from the camera.
            - Method :meth:`_detach_view` to remove a view from the camera.
           
        """
        for view in self._attached_views:
            view.camera_size_change(self.camera.sensor_height, self.camera.sensor_width)


    # ===================================================================
    # Properties
    # ===================================================================
    @property
    def sensor_height(self) -> int:
        r"""
        Get or set the height of the camera sensor in pixels.

        As described in ``pycvcam`` (https://github.com/Artezaru/pycvcam), the height of the image is along the y-axis.

        Must be a positive integer.

        This method calls the :meth:`size_update` method to apply the changes.

        Returns
        -------
        int
            The height of the camera sensor in pixels.
        """
        return self._sensor_height
    
    @sensor_height.setter
    def sensor_height(self, value: Optional[Integral]):
        if value is not None:
            if not isinstance(value, Integral) or value <= 0:
                raise ValueError("Sensor height must be a positive integer.")
        self._sensor_height = value
        self.size_update()


    @property
    def sensor_width(self) -> int:
        r"""
        Get or set the width of the camera sensor in pixels.

        As described in ``pycvcam`` (https://github.com/Artezaru/pycvcam), the width of the image is along the x-axis.

        Must be a positive integer.

        This method calls the :meth:`size_update` method to apply the changes.

        Returns
        -------
        int
            The width of the camera sensor in pixels.
        """
        return self._sensor_width
    
    @sensor_width.setter
    def sensor_width(self, value: Optional[Integral]):
        if value is not None:
            if not isinstance(value, Integral) or value <= 0:
                raise ValueError("Sensor width must be a positive integer.")
        self._sensor_width = value
        self.size_update()


    @property
    def extrinsic(self) -> Optional[pycvcam.core.Extrinsic]:
        r"""
        Get or set the extrinsic transformation of the camera.

        The extrinsic transformation describes the position and orientation of the camera in the world.

        The methods :meth:`_save_back_extrinsic` and :meth:`extrinsic_update` are called to apply the changes into the camera.

        .. seealso::

            - Package ``pycvcam`` (https://github.com/Artezaru/pycvcam) to define the extrinsic transformation.

        Returns
        -------
        Optional[pycvcam.core.Extrinsic]
            The extrinsic transformation of the camera.
        """
        return self._extrinsic

    @extrinsic.setter
    def extrinsic(self, value: Optional[pycvcam.core.Extrinsic]):
        if value is not None and not isinstance(value, pycvcam.core.Extrinsic):
            raise TypeError("Extrinsic must be an instance of pycvcam.core.Extrinsic.")
        self._extrinsic = value
        self._save_back_extrinsic()
        self.extrinsic_update()


    @property
    def distortion(self) -> Optional[pycvcam.core.Distortion]:
        r"""
        Get or set the distortion transformation of the camera.

        The distortion transformation describes the optical distortion introduced by the camera lens.

        The methods :meth:`_save_back_distortion` and :meth:`distortion_update` are called to apply the changes into the camera.

        .. seealso::

            - Package ``pycvcam`` (https://github.com/Artezaru/pycvcam) to define the distortion transformation.

        Returns
        -------
        Optional[pycvcam.core.Distortion]
            The distortion transformation of the camera.
        """
        return self._distortion
    
    @distortion.setter
    def distortion(self, value: Optional[pycvcam.core.Distortion]):
        if value is not None and not isinstance(value, pycvcam.core.Distortion):
            raise TypeError("Distortion must be an instance of pycvcam.core.Distortion.")
        self._distortion = value
        self._save_back_distortion()
        self.distortion_update()


    @property
    def intrinsic(self) -> Optional[pycvcam.core.Intrinsic]:
        r"""
        Get or set the intrinsic transformation of the camera.

        The intrinsic transformation describes the internal parameters of the camera, such as focal length and optical center.

        The methods :meth:`_save_back_intrinsic` and :meth:`intrinsic_update` are called to apply the changes into the camera.

        .. seealso::

            - Package ``pycvcam`` (https://github.com/Artezaru/pycvcam) to define the intrinsic transformation.

        Returns
        -------
        Optional[pycvcam.core.Intrinsic]
            The intrinsic transformation of the camera.
        """
        return self._intrinsic
    
    @intrinsic.setter
    def intrinsic(self, value: Optional[pycvcam.core.Intrinsic]):
        if value is not None and not isinstance(value, pycvcam.core.Intrinsic):
            raise TypeError("Intrinsic must be an instance of pycvcam.core.Intrinsic.")
        self._intrinsic = value
        self._save_back_intrinsic()
        self.intrinsic_update()


    # ===================================================================
    # Indicate an update
    # ===================================================================
    def update(self):
        r"""
        Called when the camera parameters have changed.
        
        This method should be called whenever the camera parameters are updated to ensure that the camera's internal state is consistent with the new parameters.
        This method updates all the camera by calling the respective update methods : :meth:`extrinsic_update`, :meth:`distortion_update`, :meth:`intrinsic_update`, and :meth:`size_update`.

        """
        self.extrinsic_update()
        self.distortion_update()
        self.intrinsic_update()
        self.size_update()
    

    def extrinsic_update(self):
        r"""
        Called when the extrinsic transformation has changed.

        This method should be called whenever the extrinsic transformation is updated to ensure that the camera's internal state is consistent with the new extrinsic parameters.

        Currently, this method does not perform any specific actions but is provided for future extensibility.
        """
        pass


    def distortion_update(self):
        r"""
        Called when the distortion transformation has changed.

        This method should be called whenever the distortion transformation is updated to ensure that the camera's internal state is consistent with the new distortion parameters.

        - resets the ``normalized_points`` attribute to None (uncalculated).
        """
        self._normalized_points = None


    def intrinsic_update(self):
        r"""
        Called when the intrinsic transformation has changed.
        
        This method should be called whenever the intrinsic transformation is updated to ensure that the camera's internal state is consistent with the new intrinsic parameters.

        - resets the ``normalized_points`` attribute to None (uncalculated).
        """
        self._normalized_points = None


    def size_update(self):
        r"""
        Called when the sensor size has changed.

        This method should be called whenever the sensor size is updated to ensure that the camera's internal state is consistent with the new sensor dimensions.

        - resets the ``normalized_points`` attribute to None (uncalculated).
        - notifies all attached views about the camera size change.

        """
        self._normalized_points = None
        self._notify_views()


    # ===================================================================
    # Processing methods
    # ===================================================================
    def project(self, world_points: numpy.ndarray, dx: bool = False, dintrinsic: bool = False, ddistortion: bool = False, dextrinsic: bool = False) -> ProjectionResult:
        r"""
        Project 3D world points to 2D images points using the camera's intrinsic, extrinsic, and distortion parameters.

        .. note::

            The output are the ``image_points`` in the image coordinate system (x, y) and not the pixel points (rows, columns).

        .. seealso::

            - :meth:`image_points_to_pixel_points` to convert the image points to pixel points if needed.
            - See the package ``pycvcam`` (https://github.com/Artezaru/pycvcam) for more information on the implementation details.
            - :class:`ProjectionResult` for the structure of the output.

        Parameters
        ----------
        world_points : numpy.ndarray
            An array of shape (..., 3) representing N 3D points in the world coordinate system.

        dx : bool, optional
            If True, the function will also return the jacobian of the image points with respect to the world points. Default is False.

        dintrinsic : bool, optional
            If True, the function will also return the jacobian of the image points with respect to the intrinsic parameters. Default is False.

        ddistortion : bool, optional
            If True, the function will also return the jacobian of the image points with respect to the distortion parameters. Default is False.

        dextrinsic : bool, optional
            If True, the function will also return the jacobian of the image points with respect to the extrinsic parameters. Default is False.

        Returns
        -------
        ProjectionResult
            A ProjectionResult object containing the projected image points and optionally the jacobians.

            - `image_points`: An array of shape (..., 2) representing the projected images points in the image coordinate system (x, y).
            - `jacobian_dx`: (optional) A 3D array of shape (..., 2, 3) representing the jacobian of the normalized points with respect to the world points if `dx` is True.
            - `jacobian_dintrinsic`: (optional) A 3D array of shape (..., 2, Nintrinsic) representing the jacobian of the pixel points with respect to the intrinsic parameters if `dintrinsic` is True.
            - `jacobian_ddistortion`: (optional) A 3D array of shape (..., 2, Ndistortion) representing the jacobian of the pixel points with respect to the distortion parameters if `ddistortion` is True.
            - `jacobian_dextrinsic`: (optional) A 3D array of shape (..., 2, Nextrinsic) representing the jacobian of the pixel points with respect to the extrinsic parameters if `dextrinsic` is True.

    
        Examples
        --------

        Lets create a simple camera and project some 3D points:

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

            # Define some 3D world points
            world_points = numpy.array([
                [0, 0, 1000],
                [100, 0, 1000],
                [0, 100, 1000],
                [100, 100, 1000]
            ])

            # Project the 3D points to 2D image points
            projection_result = camera.project(world_points, dx=True, dintrinsic=True, dextrinsic=True)

        Extracting the projected image points and jacobians:

        .. code-block:: python

            image_points = projection_result.image_points # The projected 2D image points of shape (4, 2)            
            jacobian_dx = projection_result.jacobian_dx # The jacobian with respect to the world points of shape (4, 2, 3)
            jacobian_dintrinsic = projection_result.jacobian_dintrinsic # The jacobian with respect to the intrinsic parameters of shape (4, 2, 4)
            jacobian_dextrinsic = projection_result.jacobian_dextrinsic # The jacobian with respect to the extrinsic parameters of shape (4, 2, 6)

        """
        if not isinstance(world_points, numpy.ndarray):
            raise TypeError("world_points must be a numpy.ndarray.")
        if world_points.ndim < 2 or world_points.shape[1] != 3:
            raise ValueError("world_points must be a array with shape (..., 3).")

        result = pycvcam.project_points(world_points, self.intrinsic, self.distortion, self.extrinsic, dx=dx, dp=dintrinsic or ddistortion or dextrinsic)

        projection_result = ProjectionResult(
            image_points=result.image_points,
            jacobian_dx=result.jacobian_dx if dx else None,
            jacobian_dintrinsic=result.jacobian_dintrinsic if dintrinsic else None,
            jacobian_ddistortion=result.jacobian_ddistortion if ddistortion else None,
            jacobian_dextrinsic=result.jacobian_dextrinsic if dextrinsic else None
        )

        return projection_result
    

    def project_points(self, world_points: PointCloud3D, dx: bool = False, dintrinsic: bool = False, ddistortion: bool = False, dextrinsic: bool = False) -> ProjectionResult:
        r"""
        Project 3D world points to 2D images points using the camera's intrinsic, extrinsic, and distortion parameters from an PointCloud3D instance.

        This method is a convenience wrapper around :meth:`project` that extracts the numpy array from the PointCloud3D instance.

        .. seealso::

            - :meth:`project` for the main projection functionality.
            - :class:`pysdic.geometry.PointCloud3D` for the structure of the input.

        Parameters
        ----------
        world_points : PointCloud3D
            An instance of PointCloud3D containing the 3D points in the world coordinate system.

        dx : bool, optional
            If True, the function will also return the jacobian of the image points with respect to the world points. Default is False.

        dintrinsic : bool, optional
            If True, the function will also return the jacobian of the image points with respect to the intrinsic parameters. Default is False.

        ddistortion : bool, optional
            If True, the function will also return the jacobian of the image points with respect to the distortion parameters. Default is False.

        dextrinsic : bool, optional
            If True, the function will also return the jacobian of the image points with respect to the extrinsic parameters. Default is False.

        Returns
        -------
        ProjectionResult
            A ProjectionResult object containing the projected image points and optionally the jacobians.

            - `image_points`: An array of shape (..., 2) representing the projected images points in the image coordinate system (x, y).
            - `jacobian_dx`: (optional) A 3D array of shape (..., 2, 3) representing the jacobian of the normalized points with respect to the world points if `dx` is True.
            - `jacobian_dintrinsic`: (optional) A 3D array of shape (..., 2, Nintrinsic) representing the jacobian of the pixel points with respect to the intrinsic parameters if `dintrinsic` is True.
            - `jacobian_ddistortion`: (optional) A 3D array of shape (..., 2, Ndistortion) representing the jacobian of the pixel points with respect to the distortion parameters if `ddistortion` is True.
            - `jacobian_dextrinsic`: (optional) A 3D array of shape (..., 2, Nextrinsic) representing the jacobian of the pixel points with respect to the extrinsic parameters if `dextrinsic` is True.

            
        Examples
        --------
        Lets create a simple camera and project some 3D points from a PointCloud3D instance:

        .. code-block:: python

            import numpy
            from pysdic.imaging import Camera
            from pycvcam import Cv2Extrinsic, Cv2Intrinsic
            from pysdic.geometry import PointCloud3D

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

            # Define some 3D world points in a PointCloud3D instance
            world_points_array = numpy.array([
                [0, 0, 1000],
                [100, 0, 1000],
                [0, 100, 1000],
                [100, 100, 1000]
            ])
            world_points = PointCloud3D(points=world_points_array)

            # Project the 3D points to 2D image points
            projection_result = camera.project_points(world_points, dx=True, dintrinsic=True, dextrinsic=True)

        Extracting the projected image points and jacobians:

        .. code-block:: python
        
            image_points = projection_result.image_points # The projected 2D image points of shape (4, 2)            
            jacobian_dx = projection_result.jacobian_dx # The jacobian with respect to the world points of shape (4, 2, 3)
            jacobian_dintrinsic = projection_result.jacobian_dintrinsic # The jacobian with respect to the intrinsic parameters of shape (4, 2, 4)
            jacobian_dextrinsic = projection_result.jacobian_dextrinsic # The jacobian with respect to the extrinsic parameters of shape (4, 2, 6)

        """
        if not isinstance(world_points, PointCloud3D):
            raise TypeError("world_points must be an instance of PointCloud3D.")
        
        return self.project(world_points.points, dx=dx, dintrinsic=dintrinsic, ddistortion=ddistortion, dextrinsic=dextrinsic)
    

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


    def get_camera_pixel_points(self, mask: Optional[numpy.ndarray] = None) -> numpy.ndarray:
        r"""
        Get the ``pixel_points`` of the camera for each pixel.

        If a mask is provided, it filters the pixel points based on the mask.

        The coordinates of the first pixel point are (0, 0) and the coordinates of the last pixel point are (sensor_height - 1, sensor_width - 1).

        Parameters
        ----------
        mask : Optional[numpy.ndarray], optional
            A boolean mask to filter the pixel points. If None, all pixel points are returned. Default is None.
            Shape (H*W,) or (H, W) where H is the height and W is the width of the camera sensor.

        Returns
        -------
        numpy.ndarray
            A 2D array of shape (N, 2) representing the pixel points in pixel coordinate system.
        """
        self._check_and_perform_changes()
        pixel_points = numpy.indices((self.sensor_height, self.sensor_width), dtype=numpy.float64).reshape(2, -1).T

        if mask is not None:
            if not isinstance(mask, numpy.ndarray):
                raise TypeError("mask must be a numpy.ndarray.")
            if not numpy.issubdtype(mask.dtype, numpy.bool_):
                raise TypeError("mask must be a boolean numpy.ndarray.")
            if not (mask.ndim == 1 and mask.shape[0] == self.sensor_height * self.sensor_width) and not (mask.ndim == 2 and mask.shape == (self.sensor_height, self.sensor_width)):
                raise ValueError("mask must be a 1D array of shape (H*W,) or a 2D array of shape (H, W) where H is the height and W is the width of the camera sensor.")
        
            mask = mask.flatten()
            pixel_points = pixel_points[mask, :]

        return pixel_points
    

    def get_camera_normalized_points(self, mask: Optional[numpy.ndarray] = None) -> numpy.ndarray:
        r"""
        Get the ``normalized_points`` of the camera by computing the inverse intrinsic and distortion transformations on the pixel points.

        If a mask is provided, it filters the normalized points based on the mask.

        The first time this method is called after an update, the computing ``normalized_points`` will be saved in the camera object to reduce processing time for subsequent calls.
        The not already computed normalized points are saved as nan values in the array.

        - To reset the cached normalized points, you can call the method ``camera.update()``.

        Parameters
        ----------
        mask : Optional[numpy.ndarray], optional
            A boolean mask to filter the normalized points. If None, all normalized points are returned. Default is None.
            Shape (H*W,) or (H, W) where H is the height and W is the width of the camera sensor.

        Returns
        -------
        numpy.ndarray
            A 2D array of shape (N, 2) representing the normalized points in normalized coordinate system.

        """
        self._check_and_perform_changes()
        if self._normalized_points is None:
            self._normalized_points = numpy.full((self.sensor_height * self.sensor_width, 2), numpy.nan, dtype=numpy.float64)

        if mask is None:
            mask = numpy.ones((self.sensor_height, self.sensor_width), dtype=numpy.bool_)

        if not isinstance(mask, numpy.ndarray):
            raise TypeError("mask must be a numpy.ndarray.")
        if not numpy.issubdtype(mask.dtype, numpy.bool_):
            raise TypeError("mask must be a boolean numpy.ndarray.")
        if not (mask.ndim == 1 and mask.shape[0] == self.sensor_height * self.sensor_width) and not (mask.ndim == 2 and mask.shape == (self.sensor_height, self.sensor_width)):
            raise ValueError("mask must be a 1D array of shape (H*W,) or a 2D array of shape (H, W) where H is the height and W is the width of the camera sensor.")
        
        mask = mask.flatten()
        values = self._normalized_points[mask, :]
        nan_mask = numpy.isnan(values).any(axis=1)
        compute_mask = numpy.zeros((self.sensor_height * self.sensor_width,), dtype=numpy.bool_)
        compute_mask[mask] = nan_mask

        # Compute the transformation only for the non-computed points
        pixel_points = self.get_camera_pixel_points(mask=compute_mask)
        image_points = self.pixel_points_to_image_points(pixel_points)
        normalized_points = pycvcam.undistort_points(image_points, self.intrinsic, self.distortion)
        self._normalized_points[compute_mask, :] = normalized_points

        return self._normalized_points[mask, :]


    def get_camera_rays(self, mask: Optional[numpy.ndarray] = None) -> Tuple[numpy.ndarray, numpy.ndarray]:
        r"""
        Get the camera rays emitted from the camera sensor to scene in the world coordinate system for each pixel.

        The rays are computed from the normalized points of the camera.

        If a mask is provided, it filters the rays based on the mask to select only the rays corresponding to the valid pixels.

        Parameters
        ----------
        mask : Optional[numpy.ndarray], optional
            A boolean mask to filter the rays. If None, all rays are returned. Default is None.
            Shape (H*W,) or (H, W) where H is the height and W is the width of the camera sensor.

        Returns
        -------
        numpy.ndarray
            A 2D array of shape (N, 3) representing the origins of the rays in world coordinate system.

        numpy.ndarray
            A 2D array of shape (N, 3) representing the directions of the rays in world coordinate system.

        """
        self._check_and_perform_changes()
        rays_object = pycvcam.compute_rays(image_points=self.get_camera_normalized_points(mask=mask), # Directly the normalized points because distortion correction already applied.
                                           intrinsic=None, # None because we gives the normalized points directly and not the image points to avoid recomputation.
                                           distortion=None,  # None because we gives the normalized points directly and not the image points to avoid recomputation.
                                           extrinsic=self.extrinsic)
        
        origins = rays_object.origins
        directions = rays_object.directions
        return origins, directions


    # ===================================================================
    # Visualization methods
    # ===================================================================
    def visualize_projected_point_cloud(
        self,
        point_cloud: PointCloud3D,
        points_color: str = "black",
        points_size: int = 5,
        points_opacity: float = 1.0,
        image: Optional[Union[numpy.ndarray, Image]] = None,
        clip_sensor: bool = True,
        show_pixel_grid: bool = False,
        title: Optional[str] = None,
    ) -> None:
        r"""
        Visualize the projected 2D points of a :class:`pysdic.geometry.PointCloud3D` on a 2D plot using matplotlib.

        .. seealso::

            - :meth:`project_points` to project 3D points to 2D image points.
            - :meth:`visualize_projected_mesh` to visualize projected 3D meshes.
            - :class:`pysdic.geometry.PointCloud3D` for the structure of the input.
            - Package `matplotlib <https://matplotlib.org/stable/index.html>`_ for visualization.

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

        image : Optional[Union[numpy.ndarray, Image]], optional
            An optional background image to display behind the projected points. If provided, the image should have dimensions matching the camera sensor size. Default is None.

        clip_sensor : bool, optional
            If True, only the points that are projected within the camera sensor dimensions are visualized. Default is True.

        show_pixel_grid : bool, optional
            If True, a grid representing the pixel layout of the camera sensor is displayed in the background. Default is False.

        title : Optional[str], optional
            An optional title for the plot. Default is None with eefault title 'Projected Points on Image Plane'.

            
        Examples
        --------
        Visualize a projected point cloud using a simple camera:

        .. code-block:: python

            import numpy
            from pysdic.imaging import Camera
            from pycvcam import Cv2Extrinsic, Cv2Intrinsic
            from pysdic.geometry import PointCloud3D

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

            # Define some 3D world points in a PointCloud3D instance
            world_points_array = numpy.array([
                [0, 0, 1000],
                [100, 0, 1000],
                [0, 100, 1000],
                [100, 100, 1000]
            ])
            world_points = PointCloud3D(points=world_points_array)

            # Visualize the projected point cloud
            camera.visualize_projected_point_cloud(point_cloud=world_points, points_color="red", points_size=10, show_pixel_grid=True)

        .. figure:: ../../../pysdic/resources/camera_visualize_projected_point_cloud_example.png
            :align: center
            :width: 400px
            :alt: Visualize Projected Point Cloud Example

            Example of projected point cloud visualization.
            This figure shows the 2D projection of the 3D points onto the image plane.

        """
        # Check input types
        if not isinstance(point_cloud, PointCloud3D):
            raise TypeError("point_cloud must be an instance of PointCloud3D.")
        if not isinstance(points_color, str):
            raise TypeError("points_color must be a string.")
        if not isinstance(points_size, Integral) or points_size <= 0:
            raise ValueError("points_size must be a positive integer.")
        if not isinstance(points_opacity, (float, int)) or not (0.0 <= points_opacity <= 1.0):
            raise ValueError("points_opacity must be a float between 0.0 and 1.0.")
        if not isinstance(clip_sensor, bool):
            raise TypeError("clip_sensor must be a boolean.")
        if title is not None and not isinstance(title, str):
            raise TypeError("title must be a string or None.")

        # Project the 3D points to 2D image points
        image_points = self.project_points(point_cloud).image_points
        
        # Create the plot
        figure = plt.figure()
        ax = figure.add_subplot(1, 1, 1)

        # If an image is provided, display it as the background
        if image is not None:
            if not isinstance(image, (numpy.ndarray, Image)):
                raise TypeError("image must be a numpy.ndarray or Image object.")
            if isinstance(image, Image):
                image = image.to_array()
            if image.ndim != 2 and image.ndim != 3:
                raise ValueError("image must be a 2D (grayscale) or 3D (color) array.")
            if image.shape[0] != self.sensor_height or image.shape[1] != self.sensor_width:
                raise ValueError("image dimensions must match the camera sensor size.")
            ax.imshow(image, extent=(0, self.sensor_width, self.sensor_height, 0))
        else: 
            # Draw a gray background to represent the image sensor
            ax.add_patch(plt.Rectangle((0, 0), self.sensor_width, self.sensor_height, color='lightgray'))

        # If show_pixel_grid is True, draw the pixel grid
        if show_pixel_grid:
            for x in range(self.sensor_width + 1):
                ax.plot([x, x], [0, self.sensor_height], color='gray', linewidth=0.5, alpha=0.5)
            for y in range(self.sensor_height + 1):
                ax.plot([0, self.sensor_width], [y, y], color='gray', linewidth=0.5, alpha=0.5)

        # Display the projected points
        ax.scatter(image_points[:, 0], image_points[:, 1], c=points_color, s=points_size, alpha=points_opacity)

        # Clip points outside the sensor dimensions if clip_sensor is True
        if clip_sensor:
            ax.set_xlim(0, self.sensor_width)
            ax.set_ylim(self.sensor_height, 0)  # Invert y-axis to match image coordinate system

        # Set labels
        ax.set_xlabel('Image X (pixels)')
        ax.set_ylabel('Image Y (pixels)')
        if title is not None:
            ax.set_title(title)
        else:
            ax.set_title('Projected Points on Image Plane')
        ax.set_aspect('equal', adjustable='box')
        plt.show()


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
        image: Optional[Union[numpy.ndarray, Image]] = None,
        clip_sensor: bool = True,
        show_pixel_grid: bool = False,
        show_vertices: bool = True,
        show_edges: bool = True,
        show_faces: bool = True,
        title: Optional[str] = None,
    ) -> None:
        r"""
        Visualize the projected 2D mesh of a :class:`pysdic.geometry.Mesh3D` on a 2D plot using matplotlib.
        
        .. seealso::

            - :meth:`project_points` to project 3D points to 2D image points.
            - :meth:`visualize_projected_point_cloud` to visualize projected 3D point clouds.
            - :class:`pysdic.geometry.Mesh3D` for the structure of the input.
            - Package `matplotlib <https://matplotlib.org/stable/index.html>`_ for visualization.

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

        image : Optional[Union[numpy.ndarray, Image]], optional
            An image to display as the background (default is None).

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
            An optional title for the plot. Default is None with default title 'Projected Mesh on Image Plane'.


        Examples
        --------

        Visualize a projected :class:pysdic.geometry.LinearTriangleMesh3D using a simple camera:

        .. code-block:: python

            import numpy
            from pysdic.imaging import Camera
            from pycvcam import Cv2Extrinsic, Cv2Intrinsic
            from pysdic.geometry import LinearTriangleMesh3D

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

            # Define a simple triangle mesh
            vertices = numpy.array([
                [0, 0, 1000],
                [100, 0, 1000],
                [50, 100, 1000],
                [0, 100, 1000]
            ])
            connectivity = numpy.array([
                [0, 1, 2],
                [0, 2, 3]
            ])
            mesh = LinearTriangleMesh3D(vertices=vertices, connectivity=connectivity)

            # Visualize the projected mesh
            camera.visualize_projected_mesh(mesh=mesh, faces_color="blue", edges_color="black", vertices_color="red")


        .. figure:: ../../../pysdic/resources/camera_visualize_projected_mesh_example.png
            :align: center
            :width: 400px
            :alt: Visualize Projected Mesh Example

            Example of projected mesh visualization.
            This figure shows the 2D projection of the 3D mesh onto the image plane.

        """
        # Check input types
        if not isinstance(mesh, Mesh3D):
            raise TypeError("mesh must be an instance of Mesh3D.")
        if not isinstance(faces_color, str):
            raise TypeError("faces_color must be a string.")
        if not isinstance(faces_opacity, float) or not (0.0 <= faces_opacity <= 1.0):
            raise ValueError("faces_opacity must be a float between 0.0 and 1.0.")
        if not isinstance(edges_color, str):
            raise TypeError("edges_color must be a string.")
        if not isinstance(edges_width, Integral) or edges_width <= 0:
            raise ValueError("edges_width must be a positive integer.")
        if not isinstance(edges_opacity, float) or not (0.0 <= edges_opacity <= 1.0):
            raise ValueError("edges_opacity must be a float between 0.0 and 1.0.")
        if not isinstance(vertices_color, str):
            raise TypeError("vertices_color must be a string.")
        if not isinstance(vertices_size, Integral) or vertices_size <= 0:
            raise ValueError("vertices_size must be a positive integer.")
        if not isinstance(vertices_opacity, float) or not (0.0 <= vertices_opacity <= 1.0):
            raise ValueError("vertices_opacity must be a float between 0.0 and 1.0.")
        if not isinstance(clip_sensor, bool):
            raise TypeError("clip_sensor must be a boolean.")
        if not isinstance(show_pixel_grid, bool):
            raise TypeError("show_pixel_grid must be a boolean.")
        if not isinstance(show_vertices, bool):
            raise TypeError("show_vertices must be a boolean.")
        if not isinstance(show_edges, bool):
            raise TypeError("show_edges must be a boolean.")
        if not isinstance(show_faces, bool):
            raise TypeError("show_faces must be a boolean.")
        if title is not None and not isinstance(title, str):
            raise TypeError("title must be a string or None.")
        
        # Project the 3D vertices to 2D image points
        image_points = self.project_points(mesh.vertices).image_points

        # Create the plot
        figure = plt.figure()
        ax = figure.add_subplot(1, 1, 1)

        # If an image is provided, display it as the background
        if image is not None:
            if not isinstance(image, (numpy.ndarray, Image)):
                raise TypeError("image must be a numpy.ndarray or Image.")
            if isinstance(image, Image):
                image = image.to_array()
            if image.ndim != 2 and image.ndim != 3:
                raise ValueError("image must be a 2D (grayscale) or 3D (color) array.")
            if image.shape[0] != self.sensor_height or image.shape[1] != self.sensor_width:
                raise ValueError("image dimensions must match the camera sensor size.")
            ax.imshow(image, extent=(0, self.sensor_width, self.sensor_height, 0))
        else: 
            # Draw a gray background to represent the image sensor
            ax.add_patch(plt.Rectangle((0, 0), self.sensor_width, self.sensor_height, color='lightgray'))

        # If show_pixel_grid is True, draw the pixel grid
        if show_pixel_grid:
            for x in range(self.sensor_width + 1):
                ax.plot([x, x], [0, self.sensor_height], color='gray', linewidth=0.5, alpha=0.5)
            for y in range(self.sensor_height + 1):
                ax.plot([0, self.sensor_width], [y, y], color='gray', linewidth=0.5, alpha=0.5)

        # Display the mesh faces
        if show_faces:
            for face in mesh.connectivity:
                polygon = plt.Polygon(image_points[face, :], closed=True, facecolor=faces_color, alpha=faces_opacity, edgecolor='none')
                ax.add_patch(polygon)

        # Display the mesh edges
        if show_edges:
            for face in mesh.connectivity:
                polygon = plt.Polygon(image_points[face, :], closed=True, fill=None, edgecolor=edges_color, linewidth=edges_width, alpha=edges_opacity)
                ax.add_patch(polygon)

        # Display the mesh vertices
        if show_vertices:
            ax.scatter(image_points[:, 0], image_points[:, 1], c=vertices_color, s=vertices_size, alpha=vertices_opacity)

        # Clip points outside the sensor dimensions if clip_sensor is True
        if clip_sensor:
            ax.set_xlim(0, self.sensor_width)
            ax.set_ylim(self.sensor_height, 0)  # Invert y-axis to match image coordinate system

        # Set plot labels
        ax.set_xlabel('Image X (pixels)')
        ax.set_ylabel('Image Y (pixels)')
        if title is not None:
            ax.set_title(title)
        else:
            ax.set_title('Projected 2D Mesh on Image Plane')
        ax.set_aspect('equal', adjustable='box')
        plt.show()