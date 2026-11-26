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

from numbers import Integral, Real
from typing import Optional, Tuple, Union

import numpy
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
from matplotlib.path import Path

from .. import fem
from ..geometry.integration_points import IntegrationPoints
from ..geometry.mesh import Mesh
from ..geometry.plotter3d import Plotter3D, _warped_coordinates
from ..geometry.point_cloud import PointCloud
from .._validation import _validate_bool, _validate_integer, _validate_real
from .camera import Camera
from .image import _as_flat_pixel_mask
from .image_plotter import _CONTOUR_NODES, ImagePlotter
from .view import View


# =============================================================================
# Internal helpers
# =============================================================================
def _as_camera(camera: Union[Camera, View]) -> Tuple[Camera, Optional[View]]:
    # The camera and the view (None if a camera is given)
    if isinstance(camera, View):
        return camera.camera, camera
    if isinstance(camera, Camera):
        return camera, None
    raise TypeError(f"Input 'camera' must be a Camera or a View. Got type {type(camera)}.")


def _deformed_mesh(mesh: Mesh, warp_by: Optional[Union[str, ArrayLike]]) -> Mesh:
    # Copy of the mesh with the vertices displaced by warp_by (the mesh itself if None)
    if not isinstance(mesh, Mesh):
        raise TypeError(f"Input 'mesh' must be a Mesh. Got type {type(mesh)}.")
    if warp_by is None:
        return mesh
    deformed = mesh.copy()
    deformed.coordinates = _warped_coordinates(mesh.vertices, warp_by, "mesh vertices")
    return deformed


def _image_plotter(
    camera: Union[Camera, View], figsize: Optional[Tuple[Real, Real]], background: bool = True
) -> ImagePlotter:
    # Plotter on the sensor of the camera, with the image of the view (or the sensor) as background
    plotter = ImagePlotter(camera, figsize=(8, 6) if figsize is None else figsize)
    if background:
        if isinstance(camera, View):
            plotter.add_image(opacity=0.7)
        else:
            plotter.add_sensor()
    return plotter


def _finish(plotter: ImagePlotter, title: Optional[str], show: bool) -> plt.Figure:
    # Title, show and return the figure of an image plotter
    plotter.set_title(title)
    if _validate_bool(show, "show"):
        plotter.show()
    return plotter.figure


def _rasterize_elements(
    image_points: numpy.ndarray, elements: numpy.ndarray, element_type: str, height: int, width: int
) -> numpy.ndarray:
    # Boolean mask (H * W,) of the pixels whose center (x = column, y = row) is inside the
    # contour of a projected element (elements with non-finite image points are skipped)
    mask = numpy.zeros((height, width), dtype=bool)
    contours = image_points[elements[:, _CONTOUR_NODES[element_type]]]  # (N_e, N_c, 2)
    for contour in contours[numpy.isfinite(contours).all(axis=(1, 2))]:
        x_min, y_min = numpy.maximum(numpy.floor(contour.min(axis=0)).astype(int), 0)
        x_max, y_max = numpy.ceil(contour.max(axis=0)).astype(int)
        x_max, y_max = min(x_max, width - 1), min(y_max, height - 1)
        if x_min > x_max or y_min > y_max:
            continue  # Out of the sensor
        ys, xs = numpy.mgrid[y_min : y_max + 1, x_min : x_max + 1]
        inside = Path(contour).contains_points(numpy.stack((xs.ravel(), ys.ravel()), axis=1))
        mask[ys.ravel()[inside], xs.ravel()[inside]] = True
    return mask.ravel()


def _projected_pixel_mask(deformed_mesh: Mesh, camera: Camera, feature: str) -> numpy.ndarray:
    # Check the (deformed) surface mesh against the camera and return the boolean mask
    # (H * W,) of the pixels whose center is inside the projection of an element
    E = camera.input_dim
    deformed_mesh._check_surface_mesh(feature, require_3d=(E == 3))
    if deformed_mesh.n_dimensions != E:
        raise ValueError(
            f"The mesh must have the input dimension of the camera ({E}). Got E={deformed_mesh.n_dimensions}."
        )
    image_points = camera.project(deformed_mesh.coordinates).image_points  # (N_v, 2)
    return _rasterize_elements(
        image_points, deformed_mesh.elements, deformed_mesh.element_type, camera.sensor_height, camera.sensor_width
    )


# =============================================================================
# Projected bounding box
# =============================================================================
def compute_projected_bounding_box(
    mesh: Mesh,
    camera: Union[Camera, View],
    *,
    warp_by: Optional[Union[str, ArrayLike]] = None,
    margin: Real = 0.0,
) -> Optional[Tuple[float, float, float, float]]:
    r"""
    Compute the bounding box of the vertices of a mesh projected on the sensor of a camera.

    The vertices of the (deformed) mesh are projected on the sensor (see
    :meth:`pysdic.Camera.project`), and the bounding box of their finite image points
    is extended by ``margin`` pixels on each side:

    .. math::

        x_{\min} = \min_k x_k - m, \quad y_{\min} = \min_k y_k - m, \quad
        x_{\max} = \max_k x_k + m, \quad y_{\max} = \max_k y_k + m

    with :math:`x` along the columns and :math:`y` along the rows of the image. Only one
    projection of the vertices is needed: it is a fast way to restrict a computation on
    the pixels (e.g. :func:`pysdic.generate_synthetic_image`) to the region of the mesh.

    .. note::

        The box contains the projection of the vertices, not always the projection of the
        whole surface: the distortion and the curved (e.g. quadratic) elements can move a
        point of an element out of the box of its vertices. Use ``margin`` to cover these
        cases. The box is not clipped to the sensor. The projection of the vertices behind
        the camera is not meaningful.

    .. seealso::

        :func:`compute_projected_roi` for the same box in integer pixels, clipped to the
        sensor, in the ROI format of :class:`pysdic.Image`.

    Parameters
    ----------
    mesh : :class:`pysdic.Mesh`
        The mesh, of dimension :math:`E` equal to the input dimension of the camera (the
        element type is not used).

    camera : Union[:class:`pysdic.Camera`, :class:`pysdic.View`]
        The camera (or a view, whose camera is used).

    warp_by : Optional[Union[str, ArrayLike]], optional
        The displacement of the vertices (name of a vertex property or array of shape
        :math:`(N_v, E)`), e.g. the displacement of a frame of an analysis.

    margin : Real, optional (default: ``0.0``)
        The margin in pixels added on each side of the box (:math:`\geq 0`).

    Returns
    -------
    Optional[Tuple[float, float, float, float]]
        The box ``(x_min, y_min, x_max, y_max)`` in image points, or ``None`` if no vertex
        has a finite projection.

    Raises
    ------
    TypeError
        If an input has a wrong type.

    ValueError
        If the dimension of the mesh is not the input dimension of the camera, or if
        ``margin`` is negative or not finite.

    Examples
    --------
    .. code-block:: python
        :linenos:

        from pysdic import compute_projected_bounding_box

        x_min, y_min, x_max, y_max = compute_projected_bounding_box(mesh, camera, margin=10)

    """
    camera, _ = _as_camera(camera)
    if not isinstance(mesh, Mesh):
        raise TypeError(f"Input 'mesh' must be a Mesh. Got type {type(mesh)}.")
    if mesh.n_dimensions != camera.input_dim:
        raise ValueError(
            f"The mesh must have the input dimension of the camera ({camera.input_dim}). "
            f"Got E={mesh.n_dimensions}."
        )
    margin = _validate_real(margin, "margin", non_negative=True, finite=True)
    coordinates = _warped_coordinates(mesh.vertices, warp_by, "mesh vertices")  # (N_v, E)
    image_points = camera.project(coordinates).image_points  # (N_v, 2)
    image_points = image_points[numpy.isfinite(image_points).all(axis=1)]
    if image_points.shape[0] == 0:
        return None
    (x_min, y_min), (x_max, y_max) = image_points.min(axis=0), image_points.max(axis=0)
    return (float(x_min) - margin, float(y_min) - margin, float(x_max) + margin, float(y_max) + margin)


# =============================================================================
# Projected ROI
# =============================================================================
def compute_projected_roi(
    mesh: Mesh,
    camera: Union[Camera, View],
    *,
    warp_by: Optional[Union[str, ArrayLike]] = None,
    margin: Real = 0.0,
) -> Optional[Tuple[int, int, int, int]]:
    r"""
    Compute the region of interest (ROI) of the pixels covering the projected bounding box
    of a mesh (see :func:`compute_projected_bounding_box`), clipped to the sensor.

    The box is rounded outwards (floor of the minimum, ceil of the maximum) and clipped to
    the sensor:

    .. math::

        x = \max(\lfloor x_{\min} \rfloor, 0), \quad
        x + width - 1 = \min(\lceil x_{\max} \rceil, W - 1)

    (and the same along :math:`y` with :math:`H`). All the projected vertices inside the
    sensor are then in :math:`[x, x + width - 1] \times [y, y + height - 1]`, the domain where
    an :class:`pysdic.Image` with this ROI can be interpolated.

    The ROI is given in the format of :class:`pysdic.Image`: it can be used directly as
    ``Image(data, roi=roi)`` or to slice the data, ``data[y : y + height, x : x + width]``.

    .. note::

        The interpolation function of an image is built on its ROI only, so it is less
        accurate close to the borders of the ROI. Use ``margin`` (a few pixels) to keep the
        projected mesh away from these borders. The notes of
        :func:`compute_projected_bounding_box` also apply.

    Parameters
    ----------
    mesh : :class:`pysdic.Mesh`
        The mesh, of dimension :math:`E` equal to the input dimension of the camera (the
        element type is not used).

    camera : Union[:class:`pysdic.Camera`, :class:`pysdic.View`]
        The camera (or a view, whose camera is used).

    warp_by : Optional[Union[str, ArrayLike]], optional
        The displacement of the vertices (name of a vertex property or array of shape
        :math:`(N_v, E)`), e.g. the displacement of a frame of an analysis.

    margin : Real, optional (default: ``0.0``)
        The margin in pixels added on each side of the box before rounding (:math:`\geq 0`).

    Returns
    -------
    Optional[Tuple[int, int, int, int]]
        The ROI ``(x, y, width, height)`` in pixels, or ``None`` if no vertex has a finite
        projection or if the box is out of the sensor.

    Raises
    ------
    TypeError
        If an input has a wrong type.

    ValueError
        If the dimension of the mesh is not the input dimension of the camera, or if
        ``margin`` is negative or not finite.

    Examples
    --------
    .. code-block:: python
        :linenos:

        from pysdic import Image, compute_projected_roi

        roi = compute_projected_roi(mesh, camera, margin=5)
        image = Image.from_array(data, roi=roi)

    """
    camera, _ = _as_camera(camera)
    box = compute_projected_bounding_box(mesh, camera, warp_by=warp_by, margin=margin)
    if box is None:
        return None
    x_min = max(int(numpy.floor(box[0])), 0)
    y_min = max(int(numpy.floor(box[1])), 0)
    x_max = min(int(numpy.ceil(box[2])), camera.sensor_width - 1)
    y_max = min(int(numpy.ceil(box[3])), camera.sensor_height - 1)
    if x_min > x_max or y_min > y_max:
        return None  # Out of the sensor
    return (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)


# =============================================================================
# Projected UV map
# =============================================================================
def compute_projected_uvmap(
    mesh: Mesh,
    camera: Union[Camera, View],
    *,
    warp_by: Optional[Union[str, ArrayLike]] = None,
) -> numpy.ndarray:
    r"""
    Compute the UV map of a mesh projected on the sensor of a camera: the texture
    coordinates that map the image of the camera onto the elements of the mesh.

    1. The vertices of the (deformed) mesh are projected on the sensor (see
       :meth:`pysdic.Camera.project`), giving the image points :math:`(x_k, y_k)` in pixels
       (:math:`x` along the columns, :math:`y` along the rows, the centers of the pixels at
       the integer coordinates).
    2. The image points are converted to UV coordinates with the OpenGL texture mapping
       convention (see https://learnopengl.com/Getting-started/Textures): the whole sensor
       of height :math:`H` and width :math:`W`, from :math:`-0.5` to :math:`W - 0.5` along
       :math:`x` and from :math:`-0.5` to :math:`H - 0.5` along :math:`y`, is mapped to
       :math:`[0, 1]^2`, with :math:`v` pointing upwards:

       .. math::

           u_k = \frac{x_k + 0.5}{W}, \qquad v_k = 1 - \frac{y_k + 0.5}{H}

    3. The UV coordinates of the vertices are gathered element by element.

    The corners of the image are then:

    - lower-left corner, pixel ``image[H - 1, 0]``: :math:`(u, v) = (0, 0)`;
    - upper-left corner, pixel ``image[0, 0]``: :math:`(u, v) = (0, 1)`;
    - lower-right corner, pixel ``image[H - 1, W - 1]``: :math:`(u, v) = (1, 0)`;
    - upper-right corner, pixel ``image[0, W - 1]``: :math:`(u, v) = (1, 1)`.

    This is the convention of the UV maps of ``pysdic`` (see :attr:`pysdic.Mesh.uvmap` and
    :func:`pysdic.interpolate_texture`, :math:`x = u W - 0.5` and :math:`y = (1 - v) H - 0.5`):
    with the image of the camera as texture, each vertex shows the gray level of the pixel
    where it is projected. For example, to paint the image of a view onto the mesh:

    .. code-block:: python

        mesh.uvmap = compute_projected_uvmap(mesh, view)
        mesh.quick_visualize(texture=view.image.image)

    .. note::

        The UV coordinates of a vertex projected out of the image (:math:`(u, v)` out of
        :math:`[0, 1]^2`, i.e. out of :math:`[-0.5, W - 0.5] \times [-0.5, H - 0.5]`) or with a
        non-finite projection are NaN (both :math:`u` and :math:`v`). :attr:`pysdic.Mesh.uvmap`
        only accepts finite values: check ``numpy.isfinite(uvmap).all()`` before setting it,
        or use the UV map of the elements entirely in the image only. The occlusions are not taken into account: a hidden element also gets the
        texture of the pixels where it is projected (see :func:`compute_visibility`). Inside an
        element, the texture coordinates are interpolated linearly between the vertices, which
        approximates the perspective and the distortion over the element.

    Parameters
    ----------
    mesh : :class:`pysdic.Mesh`
        The mesh, of dimension :math:`E` equal to the input dimension of the camera (the
        element type is not used).

    camera : Union[:class:`pysdic.Camera`, :class:`pysdic.View`]
        The camera (or a view, whose camera is used).

    warp_by : Optional[Union[str, ArrayLike]], optional
        The displacement of the vertices (name of a vertex property or array of shape
        :math:`(N_v, E)`), e.g. the displacement of a frame of an analysis.

    Returns
    -------
    numpy.ndarray
        The UV map, an array of shape :math:`(N_e, 2 N_{vpe})` storing, for each element,
        :math:`(u_1, v_1, u_2, v_2, \dots, u_{N_{vpe}}, v_{N_{vpe}})` in the order of the
        vertices of the element. NaN for the vertices projected out of the image.

    Raises
    ------
    TypeError
        If an input has a wrong type.

    ValueError
        If the dimension of the mesh is not the input dimension of the camera.

    Examples
    --------
    .. code-block:: python
        :linenos:

        from pysdic import compute_projected_uvmap

        uvmap = compute_projected_uvmap(mesh, camera)                                    # (N_e, 2 N_vpe)
        uvmap_1 = compute_projected_uvmap(mesh, view, warp_by=analysis.get_displacement(1))  # Frame 1

    """
    camera, _ = _as_camera(camera)
    if not isinstance(mesh, Mesh):
        raise TypeError(f"Input 'mesh' must be a Mesh. Got type {type(mesh)}.")
    if mesh.n_dimensions != camera.input_dim:
        raise ValueError(
            f"The mesh must have the input dimension of the camera ({camera.input_dim}). "
            f"Got E={mesh.n_dimensions}."
        )
    coordinates = _warped_coordinates(mesh.vertices, warp_by, "mesh vertices")  # (N_v, E)
    image_points = camera.project(coordinates).image_points  # (N_v, 2) as (x, y)
    H, W = camera.sensor_height, camera.sensor_width
    uv = numpy.empty_like(image_points, dtype=numpy.float64)  # (N_v, 2)
    uv[:, 0] = (image_points[:, 0] + 0.5) / W
    uv[:, 1] = 1.0 - (image_points[:, 1] + 0.5) / H
    with numpy.errstate(invalid="ignore"):
        outside = ~numpy.all((uv >= 0.0) & (uv <= 1.0), axis=1)  # Out of the image, or not finite
    uv[outside] = numpy.nan
    elements = mesh.elements  # (N_e, N_vpe)
    return uv[elements].reshape(elements.shape[0], 2 * elements.shape[1])  # (u_1, v_1, u_2, v_2, ...)


# =============================================================================
# Projected mask
# =============================================================================
def compute_projected_mesh_mask(
    mesh: Mesh,
    camera: Union[Camera, View],
    *,
    warp_by: Optional[Union[str, ArrayLike]] = None,
    flatten: bool = False,
) -> numpy.ndarray:
    r"""
    Compute the boolean mask of the pixels of a camera covered by the projection of a
    surface mesh.

    The vertices of the (deformed) mesh are projected on the sensor (see
    :meth:`pysdic.Camera.project`), and a pixel is selected if its center is inside the
    contour of at least one projected element. The centers of the pixels are at the integer
    image points: the pixel (row, column) has its center at :math:`x = \text{column}` and
    :math:`y = \text{row}`.

    The contour of an element is the polygon of its projected corner and edge nodes: for a
    curved element (e.g. ``"triangle_6"``), the edges are approximated by the segments
    between these nodes. The elements with a non-finite projected node are skipped.

    .. note::

        The occlusions are not taken into account: a pixel is selected if any element covers
        it, visible or hidden (see :func:`compute_visibility` and
        :func:`compute_pixel_integration_points` for the material points actually seen by
        the pixels). The projection of the points behind the camera is not meaningful: the
        elements crossing the plane of the camera can give a wrong mask.

    Parameters
    ----------
    mesh : :class:`pysdic.Mesh`
        The surface mesh (with an element type), of dimension :math:`E` equal to the input
        dimension of the camera.

    camera : Union[:class:`pysdic.Camera`, :class:`pysdic.View`]
        The camera (or a view, whose camera is used).

    warp_by : Optional[Union[str, ArrayLike]], optional
        The displacement of the vertices (name of a vertex property or array of shape
        :math:`(N_v, E)`), e.g. the displacement of a frame of an analysis.

    flatten : bool, optional (default: ``False``)
        If True, the mask is returned with shape :math:`(H \times W,)` (row-major order of the
        pixels), as expected by :meth:`pysdic.Camera.get_camera_rays`. Otherwise, with shape
        :math:`(H, W)`.

    Returns
    -------
    numpy.ndarray
        A boolean array of shape :math:`(H, W)` (or :math:`(H \times W,)` if ``flatten``), True
        for the pixels whose center is inside the projected mesh.

    Raises
    ------
    TypeError
        If an input has a wrong type.

    ValueError
        If the mesh is not a surface mesh with an element type, or if its dimension is not
        the input dimension of the camera.

    Examples
    --------
    .. code-block:: python
        :linenos:

        from pysdic import compute_projected_mesh_mask

        mask = compute_projected_mesh_mask(mesh, camera)  # (H, W)
        mask_1 = compute_projected_mesh_mask(mesh, camera, warp_by=sdic.get_displacement(1))

        # Restrict the integration points to the pixels seen at both frames
        points, pixels = compute_pixel_integration_points(mesh, camera, mask=mask & mask_1)

    """
    camera, _ = _as_camera(camera)
    flatten = _validate_bool(flatten, "flatten")
    deformed_mesh = _deformed_mesh(mesh, warp_by)
    pixel_mask = _projected_pixel_mask(deformed_mesh, camera, "compute_projected_mesh_mask")
    if flatten:
        return pixel_mask
    return pixel_mask.reshape(camera.sensor_height, camera.sensor_width)


# =============================================================================
# Projected areas
# =============================================================================
def compute_projected_elements_areas(
    mesh: Mesh,
    camera: Union[Camera, View],
    *,
    warp_by: Optional[Union[str, ArrayLike]] = None,
    signed: bool = False,
) -> numpy.ndarray:
    r"""
    Compute the area in pixels² of the projection of each element of a surface mesh
    (:math:`K = 2`) on the sensor of a camera.

    The area is integrated with the Gauss points of the element type:

    .. math::

        A_e = \sum_g w_g \left| \det \left( J_P(\vec{X}_g) \, J_X(\vec{\xi}_g) \right) \right|

    where :math:`J_X = \frac{\partial \vec{X}}{\partial \vec{\xi}}` of shape :math:`(E, 2)`
    is the Jacobian of the element (see :func:`pysdic.fem.compute_jacobian_matrix`) and
    :math:`J_P = \frac{\partial \vec{x}_i}{\partial \vec{X}}` of shape :math:`(2, E)` the
    Jacobian of the projection (see :meth:`pysdic.Camera.project`). The perspective, the
    distortion and the curved elements are taken into account.

    .. note::

        The occlusions are not taken into account: a hidden element has a projected area
        (see :func:`compute_visibility`). Elements projected out of the sensor also have an
        area.

    Parameters
    ----------
    mesh : :class:`pysdic.Mesh`
        A surface mesh with an element type, of dimension :math:`E` equal to the input
        dimension of the camera.

    camera : Union[:class:`pysdic.Camera`, :class:`pysdic.View`]
        The camera (or a view, whose camera is used).

    warp_by : Optional[Union[str, ArrayLike]], optional
        The displacement added to the vertices: the name of a vertex property or an array
        of shape :math:`(N_v, E)`.

    signed : bool, optional (default: ``False``)
        If True, the determinant is not taken in absolute value: the sign gives the
        orientation of the projected element (its sense of rotation in the image compared
        with the reference element), e.g. to detect the elements seen from the back.

    Returns
    -------
    numpy.ndarray
        An array of shape :math:`(N_e,)` with the projected area of each element in pixels²
        (NaN if the projection of a Gauss point is not finite).

    Raises
    ------
    TypeError
        If ``mesh`` or ``camera`` has a wrong type.

    ValueError
        If the mesh is not a surface mesh with an element type, or if its dimension is not
        the input dimension of the camera.

    """
    camera, _ = _as_camera(camera)
    if not isinstance(mesh, Mesh):
        raise TypeError(f"Input 'mesh' must be a Mesh. Got type {type(mesh)}.")
    element_type = mesh._check_surface_mesh("compute_projected_elements_areas")
    if mesh.n_dimensions != camera.input_dim:
        raise ValueError(
            f"The mesh must have the input dimension of the camera ({camera.input_dim}). "
            f"Got E={mesh.n_dimensions}."
        )
    signed = _validate_bool(signed, "signed")
    coordinates = _warped_coordinates(mesh.vertices, warp_by, "mesh vertices")  # (N_v, E)

    gauss_points = IntegrationPoints.from_gauss_points(element_type, mesh.n_elements)
    jacobian_X = fem.compute_jacobian_matrix(
        coordinates, mesh.elements, element_type, gauss_points.natural_coordinates, gauss_points.element_indices
    )  # (N_g, E, 2)
    points = fem.compute_property_interpolation(
        coordinates, mesh.elements, element_type, gauss_points.natural_coordinates, gauss_points.element_indices
    )  # (N_g, E)
    jacobian_P = camera.project(points, dx=True).jacobian_dx  # (N_g, 2, E)
    determinants = numpy.linalg.det(jacobian_P @ jacobian_X)  # (N_g,)
    if not signed:
        determinants = numpy.abs(determinants)
    return numpy.bincount(
        gauss_points.element_indices, weights=gauss_points.weights * determinants, minlength=mesh.n_elements
    )


# =============================================================================
# Visibility
# =============================================================================
def compute_visibility(
    mesh: Mesh,
    integration_points: IntegrationPoints,
    camera: Union[Camera, View],
    *,
    warp_by: Optional[Union[str, ArrayLike]] = None,
    tolerance: Optional[Real] = None,
) -> numpy.ndarray:
    r"""
    Check which integration points of a 3D surface mesh are visible by a camera.

    A point is visible if:

    1. it is located in the mesh and its projection is in the domain of the centers of the
       pixels, :math:`x \in [0, W - 1]` and :math:`y \in [0, H - 1]`, where the image can be
       interpolated (for a :class:`pysdic.View`, its gray level must also be finite, i.e. in
       the ROI of the image);
    2. it is not hidden by the mesh: the first intersection of the ray of the camera through
       the point (computed with the extrinsic transformation only, see below) with the mesh is
       not before the point, up to ``tolerance``.

    The ray is obtained without the distortion and the intrinsic transformation: the point
    is mapped to its normalized point by the extrinsic transformation, and the ray of the
    extrinsic transformation through this normalized point (``extrinsic.compute_rays``)
    passes exactly through the point. The first intersection of the ray with the deformed
    mesh (see :meth:`pysdic.Mesh.cast_rays`) is compared with the point: the point is hidden
    if the intersection is closer to the origin of the ray. A ray missing the mesh (possible
    for a ray exactly on an edge of the triangles used by Open3D) hides nothing. The points
    behind the camera are not visible (their ray does not pass through them).

    .. note::

        This function is a safeguard, not called by the operators of the SDIC: check the
        visibility of the integration points before building a term, and remove the hidden
        ones.

    .. important::

        Requires the optional package ``open3d`` (``pip install open3d``), a 3D surface mesh
        and a camera with 3D world points.

    Parameters
    ----------
    mesh : :class:`pysdic.Mesh`
        The 3D surface mesh (with an element type).

    integration_points : :class:`pysdic.IntegrationPoints`
        The points to check, consistent with the mesh.

    camera : Union[:class:`pysdic.Camera`, :class:`pysdic.View`]
        The camera, or a view (its camera and the ROI of its image).

    warp_by : Optional[Union[str, ArrayLike]], optional
        The displacement of the vertices (name of a vertex property or array of shape
        :math:`(N_v, 3)`), e.g. the displacement of a frame of an analysis.

    tolerance : Optional[Real], optional
        The margin on the distances along the ray: the point is hidden if the first
        intersection is closer than the point by more than ``tolerance``. Also the maximum
        distance between the point and its ray. By default, :math:`10^{-6}` times the
        diagonal of the bounding box of the (deformed) vertices.

    Returns
    -------
    numpy.ndarray
        A boolean array of shape :math:`(N_p,)`, True for the visible points.

    Raises
    ------
    TypeError
        If an input has a wrong type.

    ValueError
        If the mesh is not a 3D surface mesh, the points are not consistent with the mesh,
        or the camera does not use 3D world points.

    """
    camera, view = _as_camera(camera)
    if camera.input_dim != 3:
        raise ValueError(f"The camera must use 3D world points. Got an input dimension {camera.input_dim}.")
    deformed_mesh = _deformed_mesh(mesh, warp_by)
    deformed_mesh._check_surface_mesh("compute_visibility", require_3d=True)
    deformed_mesh.validate_integration_points(integration_points)
    coordinates = deformed_mesh.coordinates  # (N_v, 3)
    if tolerance is None:
        finite = coordinates[numpy.isfinite(coordinates).all(axis=1)]
        diagonal = numpy.linalg.norm(finite.max(axis=0) - finite.min(axis=0)) if finite.size else 0.0
        tolerance = 1e-6 * diagonal
    tolerance = _validate_real(tolerance, "tolerance", non_negative=True, finite=True)

    points = deformed_mesh.compute_property_interpolation(coordinates, integration_points, allow_unlocated_points=True)
    visible = integration_points.is_located() & numpy.isfinite(points).all(axis=1)

    # 1. Projection in the domain of the centers of the pixels (and in the ROI of the image for a view)
    image_points = camera.project(points).image_points
    x, y = image_points[:, 0], image_points[:, 1]
    with numpy.errstate(invalid="ignore"):
        visible &= (x >= 0.0) & (x <= camera.sensor_width - 1.0) & (y >= 0.0) & (y <= camera.sensor_height - 1.0)
    if view is not None:
        visible &= numpy.isfinite(view.image_project(points).gray_levels).all(axis=1)
    if not numpy.any(visible):
        return visible

    # 2. Occlusions: the first intersection of the ray through the point must not be before it
    indices = numpy.flatnonzero(visible)
    rays = camera.extrinsic.compute_rays(camera.extrinsic.apply(points[indices]).x)
    hits = deformed_mesh.cast_rays(rays)
    hit_points = deformed_mesh.compute_property_interpolation(coordinates, hits, allow_unlocated_points=True)
    origins, directions = rays.origins.reshape(-1, 3), rays.directions.reshape(-1, 3)
    offsets = points[indices] - origins
    along = numpy.einsum("ij,ij->i", offsets, directions)  # Distance of the point along its ray
    on_ray = (along > 0) & (numpy.linalg.norm(offsets - along[:, numpy.newaxis] * directions, axis=1) <= tolerance)
    hit_distances = numpy.einsum("ij,ij->i", hit_points - origins, directions)
    with numpy.errstate(invalid="ignore"):
        hidden = hits.is_located() & (hit_distances < along - tolerance)
    visible[indices] = on_ray & ~hidden
    return visible


# =============================================================================
# Pixels -> integration points
# =============================================================================
def compute_pixel_integration_points(
    mesh: Mesh,
    camera: Union[Camera, View],
    *,
    warp_by: Optional[Union[str, ArrayLike]] = None,
    mask: Optional[ArrayLike] = None,
) -> Tuple[IntegrationPoints, numpy.ndarray]:
    r"""
    Compute the material points of a surface mesh seen by the pixels of a camera, as
    integration points of the mesh.

    1. The (deformed) mesh is projected on the sensor, and the pixels whose center is
       inside a projected element are selected (see :func:`compute_projected_mesh_mask`).
    2. The material point seen by each selected pixel is computed, according to the
       dimension :math:`E` of the world points of the camera (the dimension of the mesh):

       - :math:`E = 3` (**3D surface mesh**): the ray of the pixel (see
         :meth:`pysdic.Camera.get_camera_rays`) is intersected with the mesh (see
         :meth:`pysdic.Mesh.cast_rays`). The first intersection gives the material point
         seen by the pixel, so the occlusions are taken into account.
       - :math:`E = 2` (**2D surface mesh**): the camera is inverted to get the world point
         of the pixel, :math:`\vec{x}_w = \text{Extrinsic}^{-1}(\vec{x}_n)` with the
         normalized point :math:`\vec{x}_n` of the pixel (see
         :meth:`pysdic.Camera.get_camera_normalized_points`), and the world point is located
         in the mesh (see :meth:`pysdic.Mesh.locate_points`). This covers a camera of the
         plane (:class:`pysdic.PlanarRigidExtrinsic`,
         :class:`pysdic.PlanarHomographicExtrinsic`) and a 3D camera observing the plane
         :math:`Z_w = 0` (:class:`pysdic.WorldPlanePerspectiveExtrinsic`,
         :class:`pysdic.WorldPlaneOrthographicExtrinsic`), whose inverse is the intersection
         of the ray of the pixel with the plane. The extrinsic must implement ``invert``.

       The pixels that see no point of the mesh are discarded.

    The natural coordinates of the points are valid for any displacement of the mesh (same
    connectivity), e.g. to follow the material points seen at a frame to another frame.

    .. important::

        The case :math:`E = 3` requires the optional package ``open3d``
        (``pip install open3d``). The case :math:`E = 2` has no optional dependency.

    Parameters
    ----------
    mesh : :class:`pysdic.Mesh`
        The surface mesh (with an element type), of dimension :math:`E = 2` or :math:`3`
        equal to the input dimension of the camera.

    camera : Union[:class:`pysdic.Camera`, :class:`pysdic.View`]
        The camera (or a view, whose camera is used).

    warp_by : Optional[Union[str, ArrayLike]], optional
        The displacement of the vertices (name of a vertex property or array of shape
        :math:`(N_v, E)`), e.g. the displacement of a frame of an analysis.

    mask : Optional[ArrayLike], optional
        A boolean mask of shape :math:`(H, W)` or :math:`(H \times W,)` restricting the
        pixels to use (e.g. a region of interest). By default, all the pixels inside the
        projected mesh.

    Returns
    -------
    integration_points : :class:`pysdic.IntegrationPoints`
        One located point per pixel seeing the mesh, in row-major order of the pixels.

    pixel_points : numpy.ndarray
        The pixel indices (row, column) of the points, of shape :math:`(N_p, 2)` and integer
        dtype.

    Raises
    ------
    TypeError
        If an input has a wrong type.

    ValueError
        If the mesh is not a surface mesh of dimension 2 or 3, its dimension is not the
        input dimension of the camera, or the extrinsic of the camera cannot be inverted
        (:math:`E = 2`).

    """
    camera, _ = _as_camera(camera)
    E = camera.input_dim
    if E not in (2, 3):
        raise ValueError(f"The camera must use 2D or 3D world points. Got an input dimension {E}.")
    deformed_mesh = _deformed_mesh(mesh, warp_by)
    if E == 2 and not camera.extrinsic.has_invert:
        raise ValueError(
            f"The extrinsic of the camera ({type(camera.extrinsic).__name__}) must implement 'invert' "
            f"for a 2D mesh."
        )
    H, W = camera.sensor_height, camera.sensor_width

    # 1. Pixels inside the projected elements
    pixel_mask = _projected_pixel_mask(deformed_mesh, camera, "compute_pixel_integration_points")
    if mask is not None:
        pixel_mask &= _as_flat_pixel_mask(mask, H, W)
    pixel_indices = numpy.flatnonzero(pixel_mask)  # Row-major order

    # 2. Material points seen by the selected pixels
    if E == 3:
        integration_points = deformed_mesh.cast_rays(camera.get_camera_rays(pixel_mask))
    else:
        normalized_points = camera.get_camera_normalized_points(pixel_mask)  # (N, 2), NaN if not invertible
        world_points = numpy.full_like(normalized_points, numpy.nan)
        finite = numpy.all(numpy.isfinite(normalized_points), axis=1)
        if numpy.any(finite):
            world_points[finite] = camera.extrinsic.invert(normalized_points[finite]).x
        integration_points = deformed_mesh.locate_points(world_points)
    located = integration_points.is_located()
    integration_points = integration_points.remove_unlocated()
    rows, columns = numpy.divmod(pixel_indices[located], W)
    return integration_points, numpy.stack((rows, columns), axis=1)


# =============================================================================
# Displays
# =============================================================================
def display_projected_bounding_box(
    mesh: Mesh,
    camera: Union[Camera, View],
    *,
    warp_by: Optional[Union[str, ArrayLike]] = None,
    margin: Real = 0.0,
    box_color: str = "red",
    figsize: Optional[Tuple[Real, Real]] = None,
    title: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    r"""
    Display the mesh projected on the sensor of a camera (on the image of a view) with its
    projected bounding box (see :func:`compute_projected_bounding_box`).

    Parameters
    ----------
    mesh : :class:`pysdic.Mesh`
        The mesh (with an element type), of dimension :math:`E` equal to the input dimension
        of the camera.

    camera : Union[:class:`pysdic.Camera`, :class:`pysdic.View`]
        The camera (drawn on its sensor), or a view (drawn on its image).

    warp_by : Optional[Union[str, ArrayLike]], optional
        The displacement of the vertices (name of a vertex property or array of shape
        :math:`(N_v, E)`).

    margin : Real, optional (default: ``0.0``)
        The margin in pixels of the box (see :func:`compute_projected_bounding_box`).

    box_color : str, optional (default: ``"red"``)
        The color of the box.

    figsize : Optional[Tuple[Real, Real]], optional
        The size of the figure in inches (width, height). Default: ``(8, 6)``.

    title : Optional[str], optional
        The title of the figure. By default, the box.

    show : bool, optional (default: ``True``)
        Whether to show the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.

    Raises
    ------
    ValueError
        If no vertex has a finite projection, or see :func:`compute_projected_bounding_box`.

    """
    box = compute_projected_bounding_box(mesh, camera, warp_by=warp_by, margin=margin)
    if box is None:
        raise ValueError("No vertex of the mesh has a finite projection: no bounding box to display.")
    x_min, y_min, x_max, y_max = box
    plotter = _image_plotter(camera, figsize)
    plotter.add_mesh(mesh, warp_by=warp_by, face_opacity=0.3, show_vertices=False)
    plotter.add_rectangle(
        ((x_min + x_max) / 2, (y_min + y_max) / 2), x_max - x_min, y_max - y_min,
        show_face=False, edge_color=box_color, edge_width=2.0,
    )
    if title is None:
        title = f"Projected bounding box: x in [{x_min:.1f}, {x_max:.1f}], y in [{y_min:.1f}, {y_max:.1f}]"
        if margin:
            title += f" (margin {margin:g} pixels)"
    return _finish(plotter, title, show)


def display_projected_roi(
    mesh: Mesh,
    camera: Union[Camera, View],
    *,
    warp_by: Optional[Union[str, ArrayLike]] = None,
    margin: Real = 0.0,
    roi_color: str = "orange",
    figsize: Optional[Tuple[Real, Real]] = None,
    title: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    r"""
    Display the mesh projected on the sensor of a camera (on the image of a view) with its
    projected ROI (see :func:`compute_projected_roi`).

    The ROI is drawn along the borders of its pixels: from :math:`x - 0.5` to
    :math:`x + width - 0.5` and from :math:`y - 0.5` to :math:`y + height - 0.5`.

    Parameters
    ----------
    mesh : :class:`pysdic.Mesh`
        The mesh (with an element type), of dimension :math:`E` equal to the input dimension
        of the camera.

    camera : Union[:class:`pysdic.Camera`, :class:`pysdic.View`]
        The camera (drawn on its sensor), or a view (drawn on its image).

    warp_by : Optional[Union[str, ArrayLike]], optional
        The displacement of the vertices (name of a vertex property or array of shape
        :math:`(N_v, E)`).

    margin : Real, optional (default: ``0.0``)
        The margin in pixels of the ROI (see :func:`compute_projected_roi`).

    roi_color : str, optional (default: ``"orange"``)
        The color of the ROI.

    figsize : Optional[Tuple[Real, Real]], optional
        The size of the figure in inches (width, height). Default: ``(8, 6)``.

    title : Optional[str], optional
        The title of the figure. By default, the ROI.

    show : bool, optional (default: ``True``)
        Whether to show the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.

    Raises
    ------
    ValueError
        If no vertex has a finite projection, if the ROI is out of the sensor, or see
        :func:`compute_projected_roi`.

    """
    roi = compute_projected_roi(mesh, camera, warp_by=warp_by, margin=margin)
    if roi is None:
        raise ValueError(
            "No vertex of the mesh has a finite projection inside the sensor: no ROI to display."
        )
    plotter = _image_plotter(camera, figsize)
    plotter.add_mesh(mesh, warp_by=warp_by, face_opacity=0.3, show_vertices=False)
    plotter.add_roi(roi, edge_color=roi_color)
    if title is None:
        title = f"Projected ROI: (x, y, width, height) = {roi}"
        if margin:
            title += f" (margin {margin:g} pixels)"
    return _finish(plotter, title, show)


def display_projected_mesh_mask(
    mesh: Mesh,
    camera: Union[Camera, View],
    *,
    warp_by: Optional[Union[str, ArrayLike]] = None,
    channel: Integral = 0,
    show_mesh: bool = True,
    edge_color: str = "cyan",
    figsize: Optional[Tuple[Real, Real]] = None,
    title: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    r"""
    Display the pixels covered by the projection of a surface mesh (see
    :func:`compute_projected_mesh_mask`).

    - For a :class:`pysdic.View`: its image with the pixels out of the mask set to NaN (not
      drawn), i.e. the part of the image seen through the projected mesh.
    - For a :class:`pysdic.Camera`: the mask itself (1 in the mask, 0 outside).

    Parameters
    ----------
    mesh : :class:`pysdic.Mesh`
        The surface mesh (with an element type), of dimension :math:`E` equal to the input
        dimension of the camera.

    camera : Union[:class:`pysdic.Camera`, :class:`pysdic.View`]
        The camera, or a view.

    warp_by : Optional[Union[str, ArrayLike]], optional
        The displacement of the vertices (name of a vertex property or array of shape
        :math:`(N_v, E)`).

    channel : Integral, optional (default: ``0``)
        The channel of the image drawn for an image with several channels.

    show_mesh : bool, optional (default: ``True``)
        Whether to draw the edges of the projected mesh.

    edge_color : str, optional (default: ``"cyan"``)
        The color of the edges of the mesh.

    figsize : Optional[Tuple[Real, Real]], optional
        The size of the figure in inches (width, height). Default: ``(8, 6)``.

    title : Optional[str], optional
        The title of the figure. By default, the number of pixels of the mask.

    show : bool, optional (default: ``True``)
        Whether to show the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.

    """
    show_mesh = _validate_bool(show_mesh, "show_mesh")
    mask = compute_projected_mesh_mask(mesh, camera, warp_by=warp_by)  # (H, W)
    plotter = _image_plotter(camera, figsize, background=False)
    if isinstance(camera, View):
        data = numpy.asarray(camera.image.image, dtype=numpy.float64)
        if data.ndim == 3:
            channel = _validate_integer(channel, "channel", min_value=0)
            if channel >= data.shape[2]:
                raise IndexError(f"Input 'channel' must be in [0, {data.shape[2] - 1}]. Got {channel}.")
            data = data[:, :, channel]
        data = data.copy()
        data[~mask] = numpy.nan
        plotter.add_sensor(color="black")
        plotter.add_image(data, cmap="gray")
    else:
        plotter.add_image(mask.astype(numpy.float64), cmap="gray", clim=(0.0, 1.0))
    if show_mesh:
        plotter.add_mesh(mesh, warp_by=warp_by, show_faces=False, show_vertices=False, edge_color=edge_color, edge_opacity=0.6)
    if title is None:
        title = f"Projected mesh mask: {int(mask.sum())} pixels ({100.0 * mask.mean():.1f} % of the sensor)"
    return _finish(plotter, title, show)


def display_projected_elements_areas(
    mesh: Mesh,
    camera: Union[Camera, View],
    *,
    warp_by: Optional[Union[str, ArrayLike]] = None,
    signed: bool = False,
    annotate: bool = False,
    n_points_per_element: Optional[Integral] = None,
    cmap: str = "viridis",
    clim: Optional[Tuple[Real, Real]] = None,
    fontsize: Real = 6,
    figsize: Optional[Tuple[Real, Real]] = None,
    title: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    r"""
    Display the area in pixels² of the projection of each element of a surface mesh (see
    :func:`compute_projected_elements_areas`), on the sensor of a camera or on the image
    of a view.

    With ``annotate=True``, the area of each element is written at the projection of its
    center. With ``n_points_per_element``, the density of integration points of each
    element (points per pixel, :math:`n / A_e`) is also given, to choose the number of
    integration points: about 1 point per pixel on the smallest elements uses all the
    pixels without oversampling.

    Parameters
    ----------
    mesh : :class:`pysdic.Mesh`
        The surface mesh (with an element type), of dimension :math:`E` equal to the input
        dimension of the camera.

    camera : Union[:class:`pysdic.Camera`, :class:`pysdic.View`]
        The camera, or a view (drawn on its image).

    warp_by : Optional[Union[str, ArrayLike]], optional
        The displacement of the vertices (name of a vertex property or array of shape
        :math:`(N_v, E)`).

    signed : bool, optional (default: ``False``)
        Whether to display the signed areas (negative for the elements seen from the back).

    annotate : bool, optional (default: ``False``)
        Whether to write the area (and the density) in each element.

    n_points_per_element : Optional[Integral], optional
        The number of integration points per element: the range of the density is added
        to the title, and to the annotations.

    cmap : str, optional (default: ``"viridis"``)
        The colormap of the areas.

    clim : Optional[Tuple[Real, Real]], optional
        The color limits. By default, set automatically.

    fontsize : Real, optional (default: ``6``)
        The font size of the annotations.

    figsize : Optional[Tuple[Real, Real]], optional
        The size of the figure in inches (width, height). Default: ``(8, 6)``.

    title : Optional[str], optional
        The title of the figure. By default, the range of the areas (and of the density).

    show : bool, optional (default: ``True``)
        Whether to show the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.

    """
    annotate = _validate_bool(annotate, "annotate")
    if n_points_per_element is not None:
        n_points_per_element = _validate_integer(n_points_per_element, "n_points_per_element", min_value=1)
    areas = compute_projected_elements_areas(mesh, camera, warp_by=warp_by, signed=signed)  # (N_e,)
    camera_, _ = _as_camera(camera)

    plotter = _image_plotter(camera, figsize)
    plotter.add_mesh(
        mesh, warp_by=warp_by, element_property=areas, property_cmap=cmap, property_clim=clim,
        face_opacity=0.8, show_vertices=False, show_bar=True,
        bar={"label": "signed projected area (pixels²)" if signed else "projected area (pixels²)"},
    )
    if annotate:
        coordinates = _warped_coordinates(mesh.vertices, warp_by, "mesh vertices")
        centers = fem.compute_elements_centers(coordinates, mesh.elements, mesh.element_type)  # (N_e, E)
        image_points = camera_.project(centers).image_points  # (N_e, 2) as (x, y)
        for (x, y), area in zip(image_points, areas):
            if not (numpy.isfinite(x) and numpy.isfinite(y) and numpy.isfinite(area)):
                continue
            text = f"{area:.0f}"
            if n_points_per_element is not None:
                text += f"\n{n_points_per_element / abs(area):.2f}"
            plotter.ax.text(x, y, text, fontsize=fontsize, ha="center", va="center", color="white", zorder=5)

    if title is None:
        title = f"Projected areas: {numpy.nanmin(areas):.1f} to {numpy.nanmax(areas):.1f} pixels²"
        if n_points_per_element is not None:
            density = n_points_per_element / numpy.abs(areas)
            title += (
                f"\n{n_points_per_element} points/element: "
                f"{numpy.nanmin(density):.2f} to {numpy.nanmax(density):.2f} points/pixel"
            )
    return _finish(plotter, title, show)


def display_visibility(
    mesh: Mesh,
    integration_points: IntegrationPoints,
    camera: Union[Camera, View],
    *,
    warp_by: Optional[Union[str, ArrayLike]] = None,
    tolerance: Optional[Real] = None,
    visible_color: str = "blue",
    hidden_color: str = "red",
    point_size: Real = 5.0,
    show_camera: bool = True,
    title: Optional[str] = None,
    show: bool = True,
) -> Plotter3D:
    r"""
    Display in 3D the visibility of integration points by a camera (see
    :func:`compute_visibility`): the (deformed) mesh, the visible points in
    ``visible_color`` and the other ones in ``hidden_color``.

    .. important::

        Requires the optional packages ``open3d`` (visibility) and ``pyvista`` (display).

    Parameters
    ----------
    mesh : :class:`pysdic.Mesh`
        The 3D surface mesh (with an element type).

    integration_points : :class:`pysdic.IntegrationPoints`
        The points to check, consistent with the mesh.

    camera : Union[:class:`pysdic.Camera`, :class:`pysdic.View`]
        The camera, or a view (its camera and the ROI of its image).

    warp_by : Optional[Union[str, ArrayLike]], optional
        The displacement of the vertices (name of a vertex property or array of shape
        :math:`(N_v, 3)`).

    tolerance : Optional[Real], optional
        See :func:`compute_visibility`.

    visible_color, hidden_color : str, optional (default: ``"blue"``, ``"red"``)
        The colors of the visible and of the hidden points.

    point_size : Real, optional (default: ``5.0``)
        The size of the points.

    show_camera : bool, optional (default: ``True``)
        Whether to draw the center of the camera (origin of the ray of the principal
        normalized point :math:`(0, 0)`) with an arrow along its optical axis, up to the
        center of the mesh.

    title : Optional[str], optional
        The title. By default, the number of visible points.

    show : bool, optional (default: ``True``)
        Whether to show the plotter.

    Returns
    -------
    :class:`pysdic.Plotter3D`
        The plotter.

    """
    show_camera = _validate_bool(show_camera, "show_camera")
    show = _validate_bool(show, "show")
    visible = compute_visibility(mesh, integration_points, camera, warp_by=warp_by, tolerance=tolerance)
    camera_, _ = _as_camera(camera)

    plotter = Plotter3D()
    plotter.add_mesh(mesh, warp_by=warp_by, face_color="lightgray", face_opacity=0.4, show_vertices=False)
    for selection, color in ((visible, visible_color), (~visible & integration_points.is_located(), hidden_color)):
        if numpy.any(selection):
            plotter.add_integration_points(
                integration_points, mesh, warp_by=warp_by, mask=selection, color=color, size=point_size
            )
    if show_camera:
        rays = camera_.extrinsic.compute_rays(numpy.zeros((1, 2)))
        origin, direction = rays.origins.reshape(3), rays.directions.reshape(3)
        coordinates = _warped_coordinates(mesh.vertices, warp_by, "mesh vertices")
        length = float(numpy.linalg.norm(numpy.nanmean(coordinates, axis=0) - origin))
        plotter.add_point_cloud(
            PointCloud(origin[numpy.newaxis]), color="black", size=3 * point_size,
            vectors=direction[numpy.newaxis], vector_scale=length, vector_color="black",
        )
    plotter.set_title(
        f"Visibility: {int(visible.sum())} / {integration_points.n_points} points visible" if title is None else title
    )
    if show:
        plotter.show()
    return plotter


def display_pixel_integration_points(
    mesh: Mesh,
    camera: Union[Camera, View],
    *,
    warp_by: Optional[Union[str, ArrayLike]] = None,
    mask: Optional[ArrayLike] = None,
    stride: Integral = 1,
    color: str = "blue",
    point_size: Real = 2.0,
    channel: Integral = 0,
    title: Optional[str] = None,
    show: bool = True,
) -> Plotter3D:
    r"""
    Display in 3D the material points of a surface mesh seen by the pixels of a camera
    (see :func:`compute_pixel_integration_points`): the intersections of the rays of the
    pixels with the (deformed) mesh.

    For a :class:`pysdic.View`, the points are colored by the gray level of their pixel: the
    image is "painted" on the surface. For a :class:`pysdic.Camera`, they have the color
    ``color``.

    .. important::

        Requires the optional package ``pyvista`` (display), and ``open3d`` for a 3D mesh.

    Parameters
    ----------
    mesh : :class:`pysdic.Mesh`
        The surface mesh (with an element type), of dimension :math:`E = 2` or :math:`3`
        equal to the input dimension of the camera.

    camera : Union[:class:`pysdic.Camera`, :class:`pysdic.View`]
        The camera, or a view.

    warp_by : Optional[Union[str, ArrayLike]], optional
        The displacement of the vertices (name of a vertex property or array of shape
        :math:`(N_v, E)`).

    mask : Optional[ArrayLike], optional
        A boolean mask of the pixels to use, see :func:`compute_pixel_integration_points`.

    stride : Integral, optional (default: ``1``)
        Only one point out of ``stride`` is displayed (in the order of the pixels), to
        lighten the display of large images.

    color : str, optional (default: ``"blue"``)
        The color of the points for a camera.

    point_size : Real, optional (default: ``2.0``)
        The size of the points.

    channel : Integral, optional (default: ``0``)
        The channel of the image used for the colors, for an image with several channels.

    title : Optional[str], optional
        The title. By default, the number of points.

    show : bool, optional (default: ``True``)
        Whether to show the plotter.

    Returns
    -------
    :class:`pysdic.Plotter3D`
        The plotter.

    """
    stride = _validate_integer(stride, "stride", min_value=1)
    show = _validate_bool(show, "show")
    integration_points, pixel_points = compute_pixel_integration_points(mesh, camera, warp_by=warp_by, mask=mask)
    display = numpy.zeros(integration_points.n_points, dtype=bool)
    display[::stride] = True

    plotter = Plotter3D()
    plotter.add_mesh(mesh, warp_by=warp_by, face_color="lightgray", face_opacity=0.3, show_vertices=False)
    if integration_points.n_points > 0:
        options = {"color": color}
        if isinstance(camera, View):
            data = camera.image.image
            if data.ndim == 3:
                channel = _validate_integer(channel, "channel", min_value=0)
                if channel >= data.shape[2]:
                    raise IndexError(f"Input 'channel' must be in [0, {data.shape[2] - 1}]. Got {channel}.")
                data = data[:, :, channel]
            gray_levels = data[pixel_points[:, 0], pixel_points[:, 1]].astype(numpy.float64)
            options = {"property": gray_levels, "property_cmap": "gray", "show_bar": True, "bar": {"title": "gray level"}}
        plotter.add_integration_points(
            integration_points, mesh, warp_by=warp_by, mask=display, size=point_size, **options
        )
    plotter.set_title(
        f"Pixel integration points: {integration_points.n_points} pixels see the mesh"
        + (f" (1 / {stride} displayed)" if stride > 1 else "")
        if title is None else title
    )
    if show:
        plotter.show()
    return plotter