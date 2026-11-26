"""
pysdic - Python Stereo-Digital Image Correlation (Computer Vision
Cameras Models, Image Processing, Meshes and tools for SDIC analysis)
Copyright (C) 2026 Artezaru, artezaru.github@proton.me

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union
from numbers import Integral, Real
from numpy.typing import ArrayLike

import numpy

from ..transformations.transformation import ComposedTransformation, Transformation
from ..transformations.rays import Rays
from ..computer_vision.project_points import (
    ProjectionResult,
    compose_projection,
    project_points,
)
from ..computer_vision.compute_rays import compute_rays
from ..computer_vision.distorting_points import undistort_points
from ..geometry.point_cloud import PointCloud
from ..geometry.mesh import Mesh
from .._validation import _validate_bool, _validate_integer, _validate_kwargs

from .image import Image, _as_flat_pixel_mask, _pixel_points_grid


class Camera(ComposedTransformation):
    r"""
    Subclass of :class:`pysdic.ComposedTransformation` representing a camera model.

    A camera is a structure with :

    - a sensor size (height and width), fixed at the creation of the camera.
    - an extrinsic transformation (world points to normalized points), from 3D world
      points or from 2D world points for planar problems (the input dimension :math:`E`
      of the camera, ``camera.input_dim``).
    - a distortion transformation (normalized points to distorted points).
    - an intrinsic transformation (distorted points to image points).

    The camera is the composition
    :math:`T = \text{Intrinsic} \circ \text{Distortion} \circ \text{Extrinsic}` (see
    :func:`pysdic.compose_projection`), so its ``parameters`` are the concatenation of
    the parameters of the extrinsic, distortion and intrinsic transformations (in this
    order).

    .. important::

        The sensor size and the three transformations cannot be changed after the
        creation of the camera. Only the parameters of the transformations can be
        modified (with ``camera.parameters`` or the parameters of each transformation).
        Create a new camera to use other transformations.

    .. note::

        The transformations are not copied: modifying the parameters of
        ``camera.intrinsic`` modifies the parameters of the camera, and conversely.

    As described in the computer vision literature, the height of the image is along
    the y-axis and the width of the image is along the x-axis.
    The following coordinate systems are used in the camera model:

    - ``world_points``: The :math:`E`-D points :math:`\vec{X}_w` with shape (...,E) expressed
      in the **world coordinate system** :math:`(\vec{E}_x, \vec{E}_y, \vec{E}_z)`
      (:math:`E = 3`, or :math:`E = 2` for planar problems).
    - ``normalized_points``: The 2-D points :math:`\vec{x}_n` with shape (...,2)
      expressed in the **normalized camera coordinate system** :math:`(\vec{I}, \vec{J})`
      with a unit distance along the optical axis :math:`(\vec{K})`.
    - ``distorted_points``: The distorted 2-D points :math:`\vec{x}_d` with shape (...,2)
      expressed in the **normalized camera coordinate system** :math:`(\vec{I}, \vec{J})`
      with a unit distance along the optical axis :math:`(\vec{K})`.
    - ``image_points``: The 2-D points :math:`\vec{x}_i` with shape (...,2) expressed in
      the **image coordinate system** :math:`(\vec{e}_x, \vec{e}_y)` in the sensor plane.
    - ``pixel_points``: The 2-D points :math:`\vec{x}_p` with shape (...,2) expressed in
      the **pixel coordinate system** :math:`(u, v)` in the matrix of pixels.

    .. figure:: /_static/cv_definition.png
        :align: center
        :width: 60%

        Definition of quantities in the package.

    To convert the **image points** to the **pixel points**, a simple swap of the
    coordinates is performed with :meth:`pysdic.Image.image_points_to_pixel_points`
    and :meth:`pysdic.Image.pixel_points_to_image_points`.


    Parameters
    ----------
    sensor_height : Integral
        The height of the camera sensor in pixels (:math:`y`-axis), a positive integer.

    sensor_width : Integral
        The width of the camera sensor in pixels (:math:`x`-axis), a positive integer.

    intrinsic : Optional[:class:`pysdic.Transformation`], optional (default: ``None``)
        The intrinsic transformation of the camera (2D to 2D).
        If None, :class:`pysdic.NoIntrinsic` is used (identity).

    distortion : Optional[:class:`pysdic.Transformation`], optional (default: ``None``)
        The distortion transformation of the camera (2D to 2D).
        If None, :class:`pysdic.NoDistortion` is used (identity).

    extrinsic : Optional[:class:`pysdic.Transformation`], optional (default: ``None``)
        The extrinsic transformation of the camera (3D or 2D world points to 2D).
        If None, :class:`pysdic.NoExtrinsic` is used (3D world points).

    parameters : Optional[ArrayLike], optional (default: ``None``)
        A 1D array of shape :math:`(N_e + N_d + N_i,)` to set the parameters of the
        three transformations. If None, the current parameters of the transformations
        are kept.


    Raises
    ------
    TypeError
        If a transformation is not a :class:`pysdic.Transformation`.

    ValueError
        If a transformation has wrong input or output dimensions, or if the sensor size
        is not positive.

    """

    __slots__ = [
        "_sensor_height",
        "_sensor_width",
        "_normalized_points",
        "_normalized_computed",
        "_normalized_save_back",
    ]

    def __init__(
        self,
        sensor_height: Integral,
        sensor_width: Integral,
        *,
        intrinsic: Optional[Transformation] = None,
        distortion: Optional[Transformation] = None,
        extrinsic: Optional[Transformation] = None,
        parameters: Optional[ArrayLike] = None,
    ) -> None:
        self._sensor_height = _validate_integer(sensor_height, "sensor_height", min_value=1)
        self._sensor_width = _validate_integer(sensor_width, "sensor_width", min_value=1)
        self._normalized_points = None
        self._normalized_computed = None
        self._normalized_save_back = None

        projection = compose_projection(intrinsic, distortion, extrinsic, world_dim=3)
        super().__init__(projection.transformations, parameters=parameters)

    # =============================================
    # Serialization and copy
    # =============================================
    def _to_dict(self) -> Dict[str, Any]:
        return {
            "sensor_height": self._sensor_height,
            "sensor_width": self._sensor_width,
            "transformations": [t.to_dict() for t in self._transformations],
        }

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> Camera:
        for key in ("sensor_height", "sensor_width", "transformations"):
            if key not in data:
                raise ValueError(f"Input dictionary must contain a '{key}' key.")
        transformations = data["transformations"]
        if not isinstance(transformations, (tuple, list)) or len(transformations) != 3:
            raise ValueError(
                "'transformations' key in input dictionary must be a list of 3 "
                "transformations (extrinsic, distortion, intrinsic)."
            )
        extrinsic, distortion, intrinsic = (Transformation.from_dict(t) for t in transformations)
        return cls(
            data["sensor_height"],
            data["sensor_width"],
            extrinsic=extrinsic,
            distortion=distortion,
            intrinsic=intrinsic,
            parameters=data.get("parameters", None),
        )

    def _copy(self) -> Camera:
        return self.__class__(
            self._sensor_height,
            self._sensor_width,
            extrinsic=self.extrinsic.copy(),
            distortion=self.distortion.copy(),
            intrinsic=self.intrinsic.copy(),
        )

    @classmethod
    def from_composition(
        cls,
        composition: ComposedTransformation,
        sensor_height: Integral,
        sensor_width: Integral,
    ) -> Camera:
        r"""
        Create a Camera object from a :class:`pysdic.ComposedTransformation` of three
        transformations ``[extrinsic, distortion, intrinsic]`` (for example built with
        :func:`pysdic.compose_projection`).

        The transformations are not copied: the camera and the composition share them.

        Parameters
        ----------
        composition : :class:`pysdic.ComposedTransformation`
            The composition ``[extrinsic, distortion, intrinsic]``.

        sensor_height : Integral
            The height of the camera sensor in pixels.

        sensor_width : Integral
            The width of the camera sensor in pixels.

        Returns
        -------
        :class:`pysdic.Camera`
            A new Camera object using the transformations of the composition.

        Raises
        ------
        TypeError
            If ``composition`` is not a :class:`pysdic.ComposedTransformation`.

        ValueError
            If the composition does not contain exactly 3 transformations.

        """
        if not isinstance(composition, ComposedTransformation):
            raise TypeError(
                f"Input 'composition' must be a ComposedTransformation. Got type {type(composition)}."
            )
        if composition.n_transformations != 3:
            raise ValueError(
                "Input 'composition' must contain 3 transformations (extrinsic, distortion, "
                f"intrinsic). Got {composition.n_transformations}."
            )
        extrinsic, distortion, intrinsic = composition.transformations
        return cls(
            sensor_height,
            sensor_width,
            extrinsic=extrinsic,
            distortion=distortion,
            intrinsic=intrinsic,
        )

    def __repr__(self) -> str:
        return (
            f"Camera(sensor_height={self._sensor_height}, sensor_width={self._sensor_width}, "
            f"extrinsic={self.extrinsic.__class__.__name__}, "
            f"distortion={self.distortion.__class__.__name__}, "
            f"intrinsic={self.intrinsic.__class__.__name__})"
        )

    # ===================================================================
    # Properties
    # ===================================================================
    @property
    def sensor_height(self) -> int:
        r"""
        [Get] The height of the camera sensor in pixels (:math:`y`-axis).

        Returns
        -------
        int
            The height of the camera sensor in pixels.

        """
        return self._sensor_height

    @property
    def sensor_width(self) -> int:
        r"""
        [Get] The width of the camera sensor in pixels (:math:`x`-axis).

        Returns
        -------
        int
            The width of the camera sensor in pixels.

        """
        return self._sensor_width

    @property
    def extrinsic(self) -> Transformation:
        r"""
        [Get] The extrinsic transformation of the camera (world points to normalized
        points), describing the position and orientation of the camera in the world.

        Returns
        -------
        :class:`pysdic.Transformation`
            The extrinsic transformation of the camera (not a copy).

        """
        return self._transformations[0]

    @property
    def distortion(self) -> Transformation:
        r"""
        [Get] The distortion transformation of the camera (normalized points to
        distorted points), describing the optical distortion of the lens.

        Returns
        -------
        :class:`pysdic.Transformation`
            The distortion transformation of the camera (not a copy).

        """
        return self._transformations[1]

    @property
    def intrinsic(self) -> Transformation:
        r"""
        [Get] The intrinsic transformation of the camera (distorted points to image
        points), describing the focal lengths and the principal point.

        Returns
        -------
        :class:`pysdic.Transformation`
            The intrinsic transformation of the camera (not a copy).

        """
        return self._transformations[2]

    # ===================================================================
    # Processing methods
    # ===================================================================
    def project(
        self,
        world_points: Union[ArrayLike, PointCloud, Mesh],
        *,
        dx: bool = False,
        dextrinsic: bool = False,
        ddistortion: bool = False,
        dintrinsic: bool = False,
    ) -> ProjectionResult:
        r"""
        Project world points to 2D images points using the camera's extrinsic,
        distortion and intrinsic transformations.

        This method calls :func:`pysdic.project_points`.

        .. note::

            The output are the **image points** in the **image coordinate system**
            :math:`(x, y)` and not the **pixel points** (rows, columns).

        .. important::

            The Jacobian with respect to the parameters is computed for all the
            transformations at once (``jacobian_dp``). The blocks of the transformations
            that are not requested (e.g. ``jacobian_ddistortion`` if
            ``ddistortion=False`` while ``dextrinsic=True``) are filled with NaN. If no
            Jacobian with respect to the parameters is requested, ``jacobian_dp`` and
            its blocks are None.

        .. seealso::

            - :meth:`pysdic.Image.image_points_to_pixel_points` to convert the image
              points to **pixel points** if needed.
            - :func:`pysdic.project_points` for the underlying projection function.
            - :class:`pysdic.ProjectionResult` for the structure of the output.


        Parameters
        ----------
        world_points : Union[ArrayLike, :class:`pysdic.PointCloud`, :class:`pysdic.Mesh`]
            An array of shape :math:`(..., E)` representing the points in the
            **world coordinate system**, with :math:`E` the input dimension of the
            camera. For a point cloud or a mesh, the coordinates of the points or
            vertices are used.

        dx : bool, optional (default: ``False``)
            If True, compute the jacobian of the image points with respect to the world
            points.

        dextrinsic : bool, optional (default: ``False``)
            If True, compute the jacobian of the image points with respect to the
            extrinsic parameters.

        ddistortion : bool, optional (default: ``False``)
            If True, compute the jacobian of the image points with respect to the
            distortion parameters.

        dintrinsic : bool, optional (default: ``False``)
            If True, compute the jacobian of the image points with respect to the
            intrinsic parameters.


        Returns
        -------
        :class:`pysdic.ProjectionResult`
            The projection result:

            - ``image_points``: shape :math:`(..., 2)`, the image points :math:`(x, y)`.
            - ``jacobian_dx``: shape :math:`(..., 2, E)` if ``dx`` is True.
            - ``jacobian_dextrinsic``: shape :math:`(..., 2, N_e)`.
            - ``jacobian_ddistortion``: shape :math:`(..., 2, N_d)`.
            - ``jacobian_dintrinsic``: shape :math:`(..., 2, N_i)`.
            - ``jacobian_dp``: shape :math:`(..., 2, N_e + N_d + N_i)`, the three
              blocks above in this order.


        Examples
        --------
        Lets create a simple camera and project some 3D points:

        .. code-block:: python

            import numpy
            from pysdic import Camera, PerspectiveExtrinsic, PinholeIntrinsic

            extrinsic = PerspectiveExtrinsic.from_rt(rvec=[0.1, 0.2, 0.3], tvec=[12.0, 34.0, 56.0])
            intrinsic = PinholeIntrinsic.from_matrix([[1000.0, 0.0, 320.0],
                                                      [0.0, 1000.0, 240.0],
                                                      [0.0, 0.0, 1.0]])

            camera = Camera(480, 640, intrinsic=intrinsic, extrinsic=extrinsic)

            world_points = numpy.array([
                [0, 0, 1000],
                [100, 0, 1000],
                [0, 100, 1000],
                [100, 100, 1000]
            ])

            result = camera.project(world_points, dx=True, dintrinsic=True, dextrinsic=True)

            result.image_points         # shape (4, 2)
            result.jacobian_dx          # shape (4, 2, 3)
            result.jacobian_dintrinsic  # shape (4, 2, N_i)
            result.jacobian_dextrinsic  # shape (4, 2, N_e)

        """
        if isinstance(world_points, (PointCloud, Mesh)):
            world_points = world_points.coordinates
        dx = _validate_bool(dx, "dx")
        flags = (
            _validate_bool(dextrinsic, "dextrinsic"),
            _validate_bool(ddistortion, "ddistortion"),
            _validate_bool(dintrinsic, "dintrinsic"),
        )
        dp = any(flags)
        return project_points(
            world_points,
            self.intrinsic,
            self.distortion,
            self.extrinsic,
            dx=dx,
            dp=dp,
            bypass_dp=tuple(not flag for flag in flags) if dp else None,
        )

    def get_camera_pixel_points(self, mask: Optional[ArrayLike] = None) -> numpy.ndarray:
        r"""
        Get the ``pixel_points`` :math:`(u, v)` of the camera for each pixel, in row-major
        order.

        The first pixel point is :math:`(0, 0)` and the last one is
        :math:`(H - 1, W - 1)`.

        Parameters
        ----------
        mask : Optional[ArrayLike], optional (default: ``None``)
            A boolean mask to filter the pixel points, with shape :math:`(H \times W,)`
            or :math:`(H, W)` where :math:`H` and :math:`W` are the height and width of
            the camera sensor. If None, all pixel points are returned.

        Returns
        -------
        numpy.ndarray
            An array of shape :math:`(N_p, 2)` in float64 dtype representing the pixel
            points (row, column).

        Raises
        ------
        TypeError
            If the mask is not a boolean array.

        ValueError
            If the mask has a wrong shape.

        """
        return _pixel_points_grid(self._sensor_height, self._sensor_width, mask)

    def get_camera_normalized_points(
        self,
        mask: Optional[ArrayLike] = None,
        *,
        distortion_kwargs: Optional[Dict[str, Any]] = None,
    ) -> numpy.ndarray:
        r"""
        Get the ``normalized_points`` of the pixels of the camera, by inverting the
        intrinsic and distortion transformations (see :func:`pysdic.undistort_points`).

        The normalized points are cached: each pixel is computed only once, at the first
        call that requests it. The cache is automatically reset when the parameters of the
        intrinsic or distortion transformations, or ``distortion_kwargs``, change (they
        are compared exactly with a saved copy at each call).

        .. note::

            Pixels for which the inversion fails (NaN) are cached as NaN and not computed
            again until the cache is reset.

        Parameters
        ----------
        mask : Optional[ArrayLike], optional (default: ``None``)
            A boolean mask to filter the normalized points, with shape
            :math:`(H \times W,)` or :math:`(H, W)` where :math:`H` and :math:`W` are the
            height and width of the camera sensor. If None, all normalized points are
            returned.

        distortion_kwargs : Optional[Dict[str, Any]], optional (default: ``None``)
            Additional keyword arguments passed to the ``invert`` method of the distortion
            transformation (e.g. the options of the iterative inversion).

        Returns
        -------
        numpy.ndarray
            An array of shape :math:`(N_p, 2)` representing the normalized points
            (a copy of the cache).

        Raises
        ------
        ValueError
            If the parameters of the intrinsic or distortion transformations are not set.

        """
        distortion_kwargs = _validate_kwargs(distortion_kwargs, "distortion_kwargs")
        n_pixels = self._sensor_height * self._sensor_width
        mask = (
            numpy.ones(n_pixels, dtype=bool)
            if mask is None
            else _as_flat_pixel_mask(mask, self._sensor_height, self._sensor_width)
        )
        for name, transformation in (("intrinsic", self.intrinsic), ("distortion", self.distortion)):
            if not transformation.is_setup:
                raise ValueError(
                    f"The parameters of the {name} transformation "
                    f"({transformation.__class__.__name__}) are not set."
                )

        # Reset the cache if the intrinsic, the distortion or the options have changed
        save_back = (
            self.intrinsic.parameters.copy(),
            self.distortion.parameters.copy(),
            distortion_kwargs,
        )
        if self._normalized_points is None or not self._is_same_save_back(save_back):
            self._normalized_points = numpy.full((n_pixels, 2), numpy.nan, dtype=numpy.float64)
            self._normalized_computed = numpy.zeros(n_pixels, dtype=bool)
            self._normalized_save_back = save_back

        # Compute only the requested pixels that are not computed yet
        compute_mask = mask & ~self._normalized_computed
        if numpy.any(compute_mask):
            image_points = self.get_camera_pixel_points(compute_mask)[:, [1, 0]]
            self._normalized_points[compute_mask] = undistort_points(
                image_points,
                self.intrinsic,
                self.distortion,
                distortion_kwargs=distortion_kwargs,
            ).x
            self._normalized_computed[compute_mask] = True

        return self._normalized_points[mask]

    def _is_same_save_back(self, save_back: Tuple) -> bool:
        # Exact comparison of the saved parameters and options with the current ones
        intrinsic, distortion, kwargs = self._normalized_save_back
        if not (
            numpy.array_equal(intrinsic, save_back[0])
            and numpy.array_equal(distortion, save_back[1])
        ):
            return False
        try:
            return bool(kwargs == save_back[2])
        except (ValueError, TypeError):  # e.g. arrays in the options: compare as different
            return False

    def get_camera_rays(
        self,
        mask: Optional[ArrayLike] = None,
        *,
        distortion_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Rays:
        r"""
        Get the camera rays emitted from the camera sensor to the scene in the world
        coordinate system for each pixel.

        The rays are computed from the cached normalized points of the camera (see
        :meth:`get_camera_normalized_points`) with the ``compute_rays`` method of the
        extrinsic transformation.

        Parameters
        ----------
        mask : Optional[ArrayLike], optional (default: ``None``)
            A boolean mask to filter the rays, with shape :math:`(H \times W,)` or
            :math:`(H, W)` where :math:`H` and :math:`W` are the height and width of the
            camera sensor. If None, all rays are returned.

        distortion_kwargs : Optional[Dict[str, Any]], optional (default: ``None``)
            Additional keyword arguments passed to the ``invert`` method of the distortion
            transformation.

        Returns
        -------
        :class:`pysdic.Rays`
            The rays (origins and directions) in the world coordinate system, with
            batch shape :math:`(N_p,)`.

        Raises
        ------
        ValueError
            If the world points of the camera are not 3D.

        """
        if self.input_dim != 3:
            raise ValueError(
                f"The camera rays are only available for 3D world points. Got E={self.input_dim}."
            )
        normalized_points = self.get_camera_normalized_points(
            mask, distortion_kwargs=distortion_kwargs
        )
        # The normalized points are given directly: no intrinsic and no distortion
        return compute_rays(normalized_points, None, None, self.extrinsic)

    # ===================================================================
    # Visualization methods
    # ===================================================================
    def quick_visualize(
        self,
        geometry: Optional[Union[ArrayLike, PointCloud, Mesh]] = None,
        *,
        image: Optional[Union[ArrayLike, Image]] = None,
        figsize: Tuple[Real, Real] = (8, 6),
        title: Optional[str] = None,
        show_vertices_index: bool = False,
        show_elements_index: bool = False,
        index_font_size: Real = 8,
    ) -> None:
        r"""
        Quickly visualize the sensor of the camera (or an image) and the projection of a
        point cloud or a mesh with :class:`pysdic.ImagePlotter`.

        .. seealso::

            For more control over the visualization, use :class:`pysdic.ImagePlotter`.

        Parameters
        ----------
        geometry : Optional[Union[ArrayLike, :class:`pysdic.PointCloud`, :class:`pysdic.Mesh`]], optional
            The world points :math:`(N_p, E)`, point cloud or mesh to project and draw
            (see :meth:`pysdic.ImagePlotter.add_point_cloud` and
            :meth:`pysdic.ImagePlotter.add_mesh`). If None, nothing is drawn on the sensor.

        image : Optional[Union[ArrayLike, :class:`pysdic.Image`]], optional
            A background image matching the sensor size (see
            :meth:`pysdic.ImagePlotter.add_image`). If None, the sensor is drawn.

        figsize : Tuple[Real, Real], optional (default: ``(8, 6)``)
            The size of the figure in inches (width, height).

        title : Optional[str], optional
            The title of the plot.

        show_vertices_index : bool, optional (default: ``False``)
            Whether to write the index of each displayed point (or vertex of the mesh) next
            to it.

        show_elements_index : bool, optional (default: ``False``)
            Whether to write the index of each displayed element of the mesh at its
            projected center (only for a mesh).

        index_font_size : Real, optional (default: ``8``)
            The font size of the indices in typographic points (constant when zooming).

        Raises
        ------
        ValueError
            If ``show_elements_index`` is True and ``geometry`` is not a mesh.

        Examples
        --------
        Visualize the projection of some points with a simple camera:

        .. code-block:: python

            import numpy
            from pysdic import Camera, PerspectiveExtrinsic, PinholeIntrinsic

            extrinsic = PerspectiveExtrinsic.from_rt(rvec=[0.1, 0.2, 0.3], tvec=[12.0, 34.0, 56.0])
            intrinsic = PinholeIntrinsic.from_matrix([[1000.0, 0.0, 320.0],
                                                      [0.0, 1000.0, 240.0],
                                                      [0.0, 0.0, 1.0]])
            camera = Camera(480, 640, intrinsic=intrinsic, extrinsic=extrinsic)

            world_points = numpy.array([[0, 0, 1000], [100, 0, 1000], [0, 100, 1000], [100, 100, 1000]])
            camera.quick_visualize(world_points, title="Projected Points", show_vertices_index=True)

        """
        from .image_plotter import ImagePlotter

        plotter = ImagePlotter(self, figsize=figsize)
        if image is None:
            plotter.add_sensor()
        else:
            plotter.add_image(image)
        if show_elements_index and not isinstance(geometry, Mesh):
            raise ValueError("Input 'show_elements_index' is only available when 'geometry' is a Mesh.")
        if isinstance(geometry, Mesh):
            plotter.add_mesh(
                geometry,
                projected=True,
                show_vertices_index=show_vertices_index,
                show_elements_index=show_elements_index,
                index_font_size=index_font_size,
            )
        elif geometry is not None:
            plotter.add_point_cloud(
                geometry, projected=True, show_points_index=show_vertices_index, index_font_size=index_font_size
            )
        plotter.set_title(title)
        plotter.show()