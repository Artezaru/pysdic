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

from dataclasses import dataclass
from typing import Optional, Tuple, Union
from numbers import Real
from numpy.typing import ArrayLike

import numpy

from ..transformations.transformation import TransformationResult
from ..geometry.point_cloud import PointCloud
from ..geometry.mesh import Mesh

from .camera import Camera
from .image import Image


@dataclass(slots=True, frozen=True)
class ImageProjectionResult(TransformationResult):
    r"""
    A class to represent the result of the projection of world points to the values
    (gray levels or colors) of an image.

    It extends :class:`pysdic.TransformationResult` with the projected image points and
    the number of parameters of each step of the projection, to give access to the blocks
    of the Jacobian with respect to the parameters of the extrinsic, distortion and
    intrinsic transformations (as :class:`pysdic.ProjectionResult`).

    The values always have a channel axis :math:`C` (:math:`C = 1` for a grayscale image).

    The Jacobians are obtained with the chain rule:

    .. math::

        \nabla_{\gamma}[I \circ P](\vec{X}) = \nabla_{x}[I](P(\vec{X})) \cdot \nabla_{\gamma}[P](\vec{X})

    .. seealso::

        - Method :meth:`pysdic.View.image_project` that produces a
          :class:`pysdic.ImageProjectionResult` object.

    Parameters
    ----------
    x : numpy.ndarray
        The image values at the projected points. Shape :math:`(\dots, C)`.

    jacobian_dx : Optional[numpy.ndarray]
        The Jacobian of the values with respect to the world points, or None if not
        computed. Shape :math:`(\dots, C, E)`.

    jacobian_dp : Optional[numpy.ndarray]
        The Jacobian of the values with respect to the parameters of the projection, or
        None if not computed. Shape :math:`(\dots, C, N_e + N_d + N_i)`. The blocks that
        were not requested are filled with NaN.

    image_points : numpy.ndarray
        The projected image points :math:`(x, y)`. Shape :math:`(\dots, 2)`.

    n_extrinsic : int
        The number of parameters :math:`N_e` of the extrinsic transformation.

    n_distortion : int
        The number of parameters :math:`N_d` of the distortion transformation.

    n_intrinsic : int
        The number of parameters :math:`N_i` of the intrinsic transformation.

    """

    image_points: Optional[numpy.ndarray] = None
    n_extrinsic: int = 0
    n_distortion: int = 0
    n_intrinsic: int = 0

    @property
    def gray_levels(self) -> numpy.ndarray:
        r"""
        [Get] the image values at the projected points (alias of ``x``).
        Shape :math:`(\dots, C)`.

        Returns
        -------
        numpy.ndarray
            The image values.
        """
        return self.x

    def _jacobian_block(self, start: int, stop: int) -> Optional[numpy.ndarray]:
        # View of a block of columns of the Jacobian with respect to the parameters
        if self.jacobian_dp is None:
            return None
        return self.jacobian_dp[..., start:stop]

    @property
    def jacobian_dextrinsic(self) -> Optional[numpy.ndarray]:
        r"""
        [Get] the Jacobian of the values with respect to the extrinsic parameters, as a
        view of ``jacobian_dp[..., :N_e]``. Shape :math:`(\dots, C, N_e)`.

        Returns
        -------
        Optional[numpy.ndarray]
            The Jacobian with respect to the extrinsic parameters, or None if not computed.
        """
        return self._jacobian_block(0, self.n_extrinsic)

    @property
    def jacobian_ddistortion(self) -> Optional[numpy.ndarray]:
        r"""
        [Get] the Jacobian of the values with respect to the distortion parameters, as a
        view of ``jacobian_dp[..., N_e:N_e + N_d]``. Shape :math:`(\dots, C, N_d)`.

        Returns
        -------
        Optional[numpy.ndarray]
            The Jacobian with respect to the distortion parameters, or None if not computed.
        """
        start = self.n_extrinsic
        return self._jacobian_block(start, start + self.n_distortion)

    @property
    def jacobian_dintrinsic(self) -> Optional[numpy.ndarray]:
        r"""
        [Get] the Jacobian of the values with respect to the intrinsic parameters, as a
        view of ``jacobian_dp[..., N_e + N_d:]``. Shape :math:`(\dots, C, N_i)`.

        Returns
        -------
        Optional[numpy.ndarray]
            The Jacobian with respect to the intrinsic parameters, or None if not computed.
        """
        start = self.n_extrinsic + self.n_distortion
        return self._jacobian_block(start, start + self.n_intrinsic)


class View(object):
    r"""
    A view is constructed by a camera and an image acquired by the camera.

    Several views can share the same camera (e.g. the same sensor at different times) or
    the same image (e.g. a split-screen setup where several virtual cameras see parts of
    the same image). Sharing an object shares its state: the parameters of a shared
    camera, or the data, ROI and interpolation functions of a shared image.

    .. note::

        The ROI of the image only restricts the zone where the image is interpolated.
        For a split-screen setup, use a ROI covering all the useful zones, no ROI, or two
        :class:`pysdic.Image` objects built from the same array with different ROIs.

    .. important::

        The camera and the image of a view cannot be changed after its creation: create a
        new view instead. The image is immutable (see :class:`pysdic.Image`), so its size
        always matches the sensor size of the camera. Only the parameters of the camera
        can be modified.

    .. seealso::

        - :class:`pysdic.Camera` for camera model and parameters
          (use ``view.camera.project`` for the geometric projection).
        - :class:`pysdic.Image` for image representation and interpolation.

    Parameters
    ----------
    camera : :class:`pysdic.Camera`
        The camera that acquired the image.

    image : :class:`pysdic.Image`
        The image acquired by the camera, with height and width equal to the sensor size
        of the camera (any number of channels).

    Raises
    ------
    TypeError
        If ``camera`` is not a :class:`pysdic.Camera` or ``image`` is not a
        :class:`pysdic.Image`.

    ValueError
        If the image does not match the sensor size of the camera.

    """

    __slots__ = ["_camera", "_image"]

    def __init__(self, camera: Camera, image: Image) -> None:
        if not isinstance(camera, Camera):
            raise TypeError(f"Input 'camera' must be a Camera. Got type {type(camera)}.")
        if not isinstance(image, Image):
            raise TypeError(f"Input 'image' must be an Image. Got type {type(image)}.")
        self._camera = camera
        self._image = image
        self._check_image_shape()

    def _check_image_shape(self) -> None:
        # Checked once at the creation: the image is immutable and the sensor size is fixed
        sensor_size = (self._camera.sensor_height, self._camera.sensor_width)
        if self._image.shape[:2] != sensor_size:
            raise ValueError(
                f"The image (height, width) {self._image.shape[:2]} does not match the "
                f"camera sensor size {sensor_size}."
            )

    def __repr__(self) -> str:
        return f"View(camera={self._camera!r}, image={self._image!r})"

    # ===================================================================
    # Properties
    # ===================================================================
    @property
    def camera(self) -> Camera:
        r"""
        [Get] The camera of the view (not a copy, it can be shared with other views).

        Returns
        -------
        :class:`pysdic.Camera`
            The camera of the view.
        """
        return self._camera

    @property
    def image(self) -> Image:
        r"""
        [Get] The image of the view (not a copy, it can be shared with other views).

        Returns
        -------
        :class:`pysdic.Image`
            The image of the view.
        """
        return self._image

    # ===================================================================
    # Projection methods
    # ===================================================================
    def image_project(
        self,
        world_points: Union[ArrayLike, PointCloud, Mesh],
        *,
        dx: bool = False,
        dextrinsic: bool = False,
        ddistortion: bool = False,
        dintrinsic: bool = False,
    ) -> ImageProjectionResult:
        r"""
        Project world points to the values (gray levels or colors) of the image, using
        the camera's transformations and the interpolation functions of the image.

        The world points are projected with :meth:`pysdic.Camera.project`, then the image
        (and its gradient if a Jacobian is requested) is evaluated at the image points
        with a single call of :meth:`pysdic.Image.evaluate_at_image_points`.

        .. note::

            Points projected out of the ROI of the image get the ``default_value`` of the
            image (NaN by default), as their Jacobians.

        .. important::

            As for :meth:`pysdic.Camera.project`, the blocks of ``jacobian_dp`` that are
            not requested are filled with NaN. If no Jacobian with respect to the
            parameters is requested, ``jacobian_dp`` and its blocks are None.

        Parameters
        ----------
        world_points : Union[ArrayLike, :class:`pysdic.PointCloud`, :class:`pysdic.Mesh`]
            The world points with shape :math:`(\dots, E)`, with :math:`E` the input
            dimension of the camera. For a point cloud or a mesh, the coordinates of the
            points or vertices are used.

        dx : bool, optional (default: ``False``)
            If True, compute the Jacobian of the values with respect to the world points.

        dextrinsic : bool, optional (default: ``False``)
            If True, compute the Jacobian of the values with respect to the extrinsic
            parameters.

        ddistortion : bool, optional (default: ``False``)
            If True, compute the Jacobian of the values with respect to the distortion
            parameters.

        dintrinsic : bool, optional (default: ``False``)
            If True, compute the Jacobian of the values with respect to the intrinsic
            parameters.

        Returns
        -------
        :class:`pysdic.ImageProjectionResult`
            The projection result:

            - ``gray_levels`` (or ``x``): shape :math:`(\dots, C)`, the image values.
            - ``image_points``: shape :math:`(\dots, 2)`, the image points :math:`(x, y)`.
            - ``jacobian_dx``: shape :math:`(\dots, C, E)` if ``dx`` is True.
            - ``jacobian_dextrinsic``: shape :math:`(\dots, C, N_e)`.
            - ``jacobian_ddistortion``: shape :math:`(\dots, C, N_d)`.
            - ``jacobian_dintrinsic``: shape :math:`(\dots, C, N_i)`.
            - ``jacobian_dp``: shape :math:`(\dots, C, N_e + N_d + N_i)`, the three
              blocks above in this order.

        Raises
        ------
        ValueError
            If the interpolation function of the image is not constructed.

        Examples
        --------
        Lets create a simple view and project some 3D points:

        .. code-block:: python

            import numpy
            from pysdic import Camera, Image, View, PerspectiveExtrinsic, PinholeIntrinsic

            extrinsic = PerspectiveExtrinsic.from_rt(rvec=[0.1, 0.2, 0.3], tvec=[12.0, 34.0, 56.0])
            intrinsic = PinholeIntrinsic.from_matrix([[1000.0, 0.0, 320.0],
                                                      [0.0, 1000.0, 240.0],
                                                      [0.0, 0.0, 1.0]])
            camera = Camera(480, 640, intrinsic=intrinsic, extrinsic=extrinsic)

            image = Image.from_array(numpy.zeros((480, 640), dtype=numpy.uint8))
            view = View(camera, image)

            world_points = numpy.array([
                [0, 0, 1000],
                [100, 0, 1000],
                [0, 100, 1000],
                [100, 100, 1000]
            ])

            result = view.image_project(world_points, dx=True, dintrinsic=True, dextrinsic=True)

            result.gray_levels          # shape (4, 1)
            result.jacobian_dx          # shape (4, 1, 3)
            result.jacobian_dintrinsic  # shape (4, 1, N_i)
            result.jacobian_dextrinsic  # shape (4, 1, N_e)

        """
        projection = self._camera.project(
            world_points,
            dx=dx,
            dextrinsic=dextrinsic,
            ddistortion=ddistortion,
            dintrinsic=dintrinsic,
        )
        image_points = projection.image_points  # (..., 2)
        batch_shape = image_points.shape[:-1]
        n_channels = self._image.n_channels
        need_gradient = projection.jacobian_dx is not None or projection.jacobian_dp is not None

        # Values (and gradient (d/dx, d/dy)) in a single evaluation of the image
        derivatives = ((0, 0), (1, 0), (0, 1)) if need_gradient else ((0, 0),)
        evaluations = [
            values.reshape(*batch_shape, n_channels)  # (..., C), with a channel axis for grayscale images too
            for values in self._image.evaluate_at_image_points(image_points.reshape(-1, 2), derivatives)
        ]

        jacobian_dx = None
        jacobian_dp = None
        if need_gradient:
            # Image gradient (..., C, 2) ordered as the rows (x, y) of the projection Jacobians
            image_gradient = numpy.stack((evaluations[1], evaluations[2]), axis=-1)
            if projection.jacobian_dx is not None:
                jacobian_dx = image_gradient @ projection.jacobian_dx  # (..., C, E)
            if projection.jacobian_dp is not None:
                # Chain rule only for the requested blocks (as ComposedTransformation): the product is
                # written directly in its columns, the columns of the blocks not requested are NaN
                n_e, n_d, n_i = projection.n_extrinsic, projection.n_distortion, projection.n_intrinsic
                jacobian_dp = numpy.empty((*batch_shape, n_channels, n_e + n_d + n_i), dtype=numpy.float64)
                for requested, start, stop in (
                    (dextrinsic, 0, n_e),
                    (ddistortion, n_e, n_e + n_d),
                    (dintrinsic, n_e + n_d, n_e + n_d + n_i),
                ):
                    if stop == start:
                        continue
                    if requested:
                        numpy.matmul(
                            image_gradient, projection.jacobian_dp[..., start:stop], out=jacobian_dp[..., start:stop]
                        )  # (..., C, 2) @ (..., 2, n) -> (..., C, n)
                    else:
                        jacobian_dp[..., start:stop] = numpy.nan

        return ImageProjectionResult(
            x=evaluations[0],
            jacobian_dx=jacobian_dx,
            jacobian_dp=jacobian_dp,
            image_points=image_points,
            n_extrinsic=projection.n_extrinsic,
            n_distortion=projection.n_distortion,
            n_intrinsic=projection.n_intrinsic,
        )

    # ===================================================================
    # Visualization methods
    # ===================================================================
    def quick_visualize(
        self,
        geometry: Optional[Union[ArrayLike, PointCloud, Mesh]] = None,
        *,
        figsize: Tuple[Real, Real] = (8, 6),
        title: Optional[str] = None,
        show_vertices_index: bool = False,
        show_elements_index: bool = False,
        index_font_size: Real = 8,
    ) -> None:
        r"""
        Quickly visualize the image of the view and the projection of a point cloud or a
        mesh with :class:`pysdic.ImagePlotter`.

        .. seealso::

            For more control over the visualization, use :class:`pysdic.ImagePlotter`
            built from the view.

        Parameters
        ----------
        geometry : Optional[Union[ArrayLike, :class:`pysdic.PointCloud`, :class:`pysdic.Mesh`]], optional
            The world points :math:`(N_p, E)`, point cloud or mesh to project and draw.
            If None, only the image is drawn.

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

        """
        from .image_plotter import ImagePlotter

        plotter = ImagePlotter(self, figsize=figsize)
        plotter.add_image()
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