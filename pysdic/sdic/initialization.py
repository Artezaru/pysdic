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
from numbers import Integral
from typing import Optional, Sequence, Tuple

import numpy
from numpy.typing import ArrayLike
from scipy.ndimage import map_coordinates

from ..computer_vision.compute_rays import compute_rays
from ..core.disflow_config import DisflowConfig
from ..computer_vision.intersect_rays import intersect_rays
from ..computer_vision.optical_flow import compute_optical_flow
from ..imaging.camera import Camera
from ..imaging.view import View
from .._validation import _as_array, _validate_integer
from .sdic import SDIC


# =============================================================================
# Result
# =============================================================================
@dataclass(frozen=True)
class OpticalFlowDisplacementResult:
    r"""
    Result of :func:`pysdic.compute_optical_flow_displacement`.

    The arrays are indexed by the vertices of the mesh of the SDIC. A vertex that could
    not be measured (outside an image, flow not computed, rays behind a camera) has NaN
    values from the step where it was lost: for example a vertex lost by the second
    optical flow has a finite ``x_def1`` but NaN ``x_def2``, ``displacement`` and
    ``ray_distances``.

    Attributes
    ----------
    displacement : numpy.ndarray
        The estimated displacement :math:`U_{def}` of the vertices at the deformed frame,
        of shape :math:`(N_v, 3)`, relative to the coordinates of the mesh of the SDIC.

    ray_distances : numpy.ndarray
        For each vertex, the distance between the two rays of the deformed views, of
        shape :math:`(N_v,)`, in the unit of the mesh. It is zero for a perfect matching:
        large values reveal wrong matchings.

    x_ref : numpy.ndarray
        The image points of the vertices in the reference view (projection of the mesh
        displaced by :math:`U_{ref}`), of shape :math:`(N_v, 2)`.

    x_def1 : numpy.ndarray
        The image points of the vertices measured by optical flow in the first deformed
        view, of shape :math:`(N_v, 2)`.

    x_def2 : numpy.ndarray
        The image points of the vertices measured by optical flow in the second deformed
        view, of shape :math:`(N_v, 2)`.

    """

    displacement: numpy.ndarray
    ray_distances: numpy.ndarray
    x_ref: numpy.ndarray
    x_def1: numpy.ndarray
    x_def2: numpy.ndarray


# =============================================================================
# Internal helpers
# =============================================================================
def _uint_image(view: View, channel: int, name: str) -> numpy.ndarray:
    # The channel (H, W) of the image of a view, checked for the optical flow (unsigned integers)
    image = view.image.image
    if not numpy.issubdtype(image.dtype, numpy.unsignedinteger):
        raise TypeError(
            f"The image of the {name} must be of unsigned integer type for the optical flow. "
            f"Got dtype {image.dtype}."
        )
    if image.ndim == 2:
        return image
    if channel >= image.shape[2]:
        raise IndexError(
            f"Input 'channel' is out of the range [0, {image.shape[2]}) of the image of the {name}. Got {channel}."
        )
    return image[:, :, channel]


def _project_in_front(camera: Camera, points: numpy.ndarray) -> numpy.ndarray:
    # Image points (N, 2) of the points (N, 3), NaN for the points that are not finite,
    # behind the camera or outside the sensor
    image_points = numpy.full((points.shape[0], 2), numpy.nan)
    finite = numpy.isfinite(points).all(axis=1)
    if not numpy.any(finite):
        return image_points
    indices = numpy.flatnonzero(finite)
    projected = camera.project(points[indices]).image_points
    # A point behind a perspective camera projects as a mirrored point: compare its
    # position along the ray through its own normalized point
    rays = camera.extrinsic.compute_rays(camera.extrinsic.apply(points[indices]).x)
    along = numpy.einsum("ij,ij->i", points[indices] - rays.origins.reshape(-1, 3), rays.directions.reshape(-1, 3))
    x, y = projected[:, 0], projected[:, 1]
    with numpy.errstate(invalid="ignore"):
        keep = (along > 0) & (x >= -0.5) & (x <= camera.sensor_width - 0.5) & (y >= -0.5) & (y <= camera.sensor_height - 0.5)
    image_points[indices[keep]] = projected[keep]
    return image_points


def _fit_affine(source: numpy.ndarray, target: numpy.ndarray, name: str) -> numpy.ndarray:
    # Least squares affine map target = [source, 1] @ M, M of shape (3, 2), on the
    # points finite in both arrays (at least 3 points not aligned)
    valid = numpy.isfinite(source).all(axis=1) & numpy.isfinite(target).all(axis=1)
    design = numpy.column_stack((source[valid], numpy.ones(int(valid.sum()))))
    if valid.sum() < 3 or numpy.linalg.matrix_rank(design) < 3:
        raise ValueError(
            f"Cannot estimate the affine initialization of the optical flow {name}: at least 3 "
            f"vertices not aligned must be projected in both views. Got {int(valid.sum())} vertices."
        )
    return numpy.linalg.lstsq(design, target[valid], rcond=None)[0]


def _region(points: numpy.ndarray, margin: int, shape: Tuple[int, ...]) -> Tuple[int, int, int, int]:
    # Region (x, y, width, height) of the pixels around the finite points (N, 2), widened by
    # the margin and clipped to the image of shape (H, W, ...)
    finite = points[numpy.isfinite(points).all(axis=1)]
    x_min, y_min = numpy.floor(finite.min(axis=0)).astype(int) - margin
    x_max, y_max = numpy.ceil(finite.max(axis=0)).astype(int) + margin
    x_min, y_min = max(x_min, 0), max(y_min, 0)
    x_max, y_max = min(x_max, shape[1] - 1), min(y_max, shape[0] - 1)
    return int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)


def _match(
    image_1: numpy.ndarray,
    image_2: numpy.ndarray,
    source: numpy.ndarray,
    target_estimate: numpy.ndarray,
    *,
    config: DisflowConfig,
    margin: int,
    name: str,
) -> numpy.ndarray:
    # Image points in image_2 (H, W) of the points 'source' (N, 2) of image_1 (H, W),
    # measured by optical flow after an affine alignment of image_2 on image_1
    if image_1.shape != image_2.shape:
        raise ValueError(
            f"The images of the optical flow {name} must have the same shape. "
            f"Got {image_1.shape} and {image_2.shape}."
        )
    # Affine map x2 = A x1 + b fitted on source -> target_estimate
    affine = _fit_affine(source, target_estimate, name)
    A, b = affine[:2].T, affine[2]

    # Aligned image: aligned(x) = image_2(A x + b), so that the remaining motion between
    # image_1 and the aligned image is small and without rotation or scale (the patches of
    # the optical flow only translate)
    rows, columns = numpy.indices(image_1.shape, dtype=numpy.float64)
    grid = numpy.stack((columns.ravel(), rows.ravel()), axis=1)  # (H W, 2) image points (x, y)
    mapped = grid @ A.T + b
    aligned = map_coordinates(image_2.astype(numpy.float64), mapped[:, ::-1].T, order=1, mode="constant", cval=0.0)
    maximum = numpy.iinfo(image_2.dtype).max
    aligned = numpy.clip(numpy.round(aligned), 0, maximum).astype(image_2.dtype).reshape(image_1.shape)

    # Optical flow on the region around the points: aligned(x + F(x)) = image_1(x)
    region = _region(source, margin, image_1.shape)
    flow_x, flow_y = compute_optical_flow(image_1, aligned, region=region, disflow_config=config)

    # Bilinear interpolation of the flow at the points (pixel centers at integer
    # coordinates, NaN outside the region), then x2 = A (x1 + F(x1)) + b
    valid = numpy.isfinite(source).all(axis=1)
    coordinates = source[valid][:, ::-1].T  # (row = y, column = x)
    flow = numpy.stack(
        (
            map_coordinates(flow_x, coordinates, order=1, mode="constant", cval=numpy.nan),
            map_coordinates(flow_y, coordinates, order=1, mode="constant", cval=numpy.nan),
        ),
        axis=1,
    )
    matched = numpy.full_like(source, numpy.nan)
    matched[valid] = (source[valid] + flow) @ A.T + b
    return matched


# =============================================================================
# Workflow
# =============================================================================
def compute_optical_flow_displacement(
    sdic: SDIC,
    ref_frame: Integral,
    ref_view: Integral,
    def_frame: Integral,
    def_views: Sequence[Integral],
    *,
    disflow_config: Optional[DisflowConfig] = None,
    displacement_estimate: Optional[ArrayLike] = None,
    margin: Optional[Integral] = None,
    channel: Integral = 0,
) -> OpticalFlowDisplacementResult:
    r"""
    Estimate the displacement of the vertices of the mesh at a deformed frame by optical
    flow and stereo triangulation (for example to initialize a SDIC measurement).

    The vertices are followed from one reference view to two views of the deformed frame,
    then triangulated:

    1. The mesh displaced by :math:`U_{ref}` (displacement of ``ref_frame``) is projected
       in the reference view: image points :math:`x_{ref}`, and a rectangular region
       around them is defined (widened by ``margin``).
    2. The optical flow from the reference view to the first deformed view is computed
       on this region (after an affine alignment, see below).
    3. The flow interpolated (bilinearly) at :math:`x_{ref}` gives the image points
       :math:`x_{def,1}` of the vertices in the first deformed view.
    4. 5. 6. The same from the first to the second deformed view, on a region around
       :math:`x_{def,1}`, gives :math:`x_{def,2}`.
    7. 8. The rays of the two deformed views through :math:`x_{def,1}` and
       :math:`x_{def,2}` are intersected (:func:`pysdic.intersect_rays`): 3D positions
       :math:`X_{def}` of the vertices.
    9. The displacement is :math:`U_{def} = X_{def} - X`, with :math:`X` the coordinates of
       the mesh of the SDIC.

    **Affine alignment.** The views can be taken by different cameras (the first deformed
    view is not necessarily the camera of the reference view). For each optical flow
    between images :math:`I_1` and :math:`I_2`, an affine map :math:`x \mapsto A x + b` is
    fitted by least squares between the image points of the vertices in :math:`I_1` and
    their estimated image points in :math:`I_2` (projection of the mesh displaced by
    ``displacement_estimate``). It is the mean motion of the region at first order:
    rotation, scale, shear and translation due to the change of point of view and to the
    estimated displacement. The second image is aligned on the first one,
    :math:`\tilde{I}_2(x) = I_2(A x + b)` (bilinear interpolation, 0 outside the image),
    the optical flow :math:`F` is computed between :math:`I_1` and :math:`\tilde{I}_2`, and
    the image points are :math:`x_2 = A (x_1 + F(x_1)) + b`.

    The alignment is applied to the images and not given as an initial flow, because the
    patches of the DIS optical flow only translate: a rotation of a few degrees between
    two cameras is enough to make large patches fail, even with an exact initial flow.

    .. important::

        The optical flow only corrects a small motion around the affine alignment (a few
        pixels to a few tens of pixels, depending on the texture and the
        ``disflow_config``). For a large displacement, give a ``displacement_estimate``
        (for example the result of the previous frame).

    A vertex gets a NaN displacement if it is not finite, behind a camera or outside an
    image, if the flow is not defined at its image point, or if the triangulated point is
    behind one of the deformed cameras.

    .. important::

        The occlusions are not detected: a vertex hidden by another part of the surface
        gets the flow of the visible surface. Check the ``ray_distances`` of the result,
        or :func:`pysdic.compute_visibility`.

    .. note::

        The images are used as full arrays (``view.image.image``): the ROI of the
        :class:`pysdic.Image` objects is ignored. They must be of unsigned integer type
        (see :func:`pysdic.compute_optical_flow`).

    Parameters
    ----------
    sdic : :class:`pysdic.SDIC`
        The SDIC with the mesh, the views and the displacement of ``ref_frame``. The mesh
        must be 3D (:math:`E = 3`): the triangulation of two rays has no meaning in a
        plane.

    ref_frame : Integral
        The reference frame. Its displacement :math:`U_{ref}` must be set (use
        :meth:`pysdic.SDIC.set_zero_displacement` for the undeformed mesh).

    ref_view : Integral
        The index of the reference view in ``ref_frame``.

    def_frame : Integral
        The deformed frame, different from ``ref_frame``.

    def_views : Sequence[Integral]
        The indices of the two views of ``def_frame`` used for the triangulation,
        ``(view_1, view_2)``. The flow goes from the reference view to ``view_1``, then
        from ``view_1`` to ``view_2``.

    disflow_config : :class:`pysdic.DisflowConfig`, optional
        The parameters of the optical flow. By default, ``DisflowConfig()``.

    displacement_estimate : ArrayLike, optional
        An estimation of the displacement at ``def_frame``, of shape :math:`(N_v, 3)`,
        used only to initialize the optical flows. NaN values are accepted: these vertices
        are not used for the initialization (for example the ``displacement`` of a
        previous result).

        If None (default), the estimation is :math:`U_{ref}`: the surface **has not moved**
        since the reference frame. This is not a zero displacement, which would mean that
        the surface is back to the coordinates of the mesh.

    margin : Integral, optional
        Number of pixels added around the image points of the vertices to define the
        regions of the optical flows. By default, the ``patch_size`` of the
        configuration, so that the patches of the border contain the points. Must be
        non-negative.

    channel : Integral, optional (default: ``0``)
        The channel of the images used for the optical flow (multi-channel images).

    Returns
    -------
    :class:`pysdic.OpticalFlowDisplacementResult`
        The estimated displacement, the distances between the rays and the image points
        of the vertices in the three views.

    Raises
    ------
    TypeError
        If an input has a wrong type, or an image is not of unsigned integer type.

    ValueError
        If the mesh is not 3D, ``def_frame`` equals ``ref_frame``, ``def_views`` does not
        contain two different views, the images of an optical flow have different shapes,
        or less than 3 vertices can be used for an affine initialization.

    IndexError
        If a frame or a view index is out of range.

    Examples
    --------
    .. code-block:: python

        from pysdic import DisflowConfig, compute_optical_flow_displacement

        sdic.set_zero_displacement(0)
        result = compute_optical_flow_displacement(
            sdic, ref_frame=0, ref_view=0, def_frame=1, def_views=(0, 1),
            disflow_config=DisflowConfig(patch_size=30, patch_stride=6),
        )
        print(numpy.nanmax(result.ray_distances))
        sdic.set_displacement(1, numpy.nan_to_num(result.displacement))

    """
    if not isinstance(sdic, SDIC):
        raise TypeError(f"Input 'sdic' must be a SDIC. Got type {type(sdic)}.")
    mesh = sdic.mesh
    if mesh.n_dimensions != 3:
        raise ValueError(
            f"compute_optical_flow_displacement requires a 3D mesh (E = 3): the vertices are "
            f"triangulated from the rays of two views. Got a mesh of dimension {mesh.n_dimensions}."
        )
    if not isinstance(def_views, (tuple, list)) or len(def_views) != 2:
        raise TypeError(f"Input 'def_views' must be a tuple (view_1, view_2). Got {def_views!r}.")

    reference_displacement = sdic.get_displacement(ref_frame)  # Checks ref_frame, error if not set
    reference = sdic.get_view(ref_frame, ref_view)
    view_1 = sdic.get_view(def_frame, def_views[0])
    view_2 = sdic.get_view(def_frame, def_views[1])
    if _validate_integer(def_frame, "def_frame") == _validate_integer(ref_frame, "ref_frame"):
        raise ValueError(f"Input 'def_frame' must be different from 'ref_frame'. Got {def_frame} for both.")
    if view_1 is view_2 or _validate_integer(def_views[0], "def_views[0]") == _validate_integer(def_views[1], "def_views[1]"):
        raise ValueError(f"Input 'def_views' must contain two different views. Got {tuple(def_views)}.")

    if disflow_config is None:
        disflow_config = DisflowConfig()
    elif not isinstance(disflow_config, DisflowConfig):
        raise TypeError(f"Input 'disflow_config' must be a DisflowConfig or None. Got type {type(disflow_config)}.")
    margin = disflow_config.patch_size if margin is None else _validate_integer(margin, "margin", min_value=0)
    channel = _validate_integer(channel, "channel", min_value=0)

    coordinates = mesh.coordinates  # (N_v, 3)
    if displacement_estimate is None:
        displacement_estimate = reference_displacement
    else:
        displacement_estimate = _as_array(
            displacement_estimate,
            "displacement_estimate",
            numpy.float64,
            shape=(mesh.n_vertices, 3),
            shape_hint="(N_v, 3)",
            finite=False,
        )
    estimated_points = coordinates + displacement_estimate

    image_ref = _uint_image(reference, channel, "reference view")
    image_1 = _uint_image(view_1, channel, "first deformed view")
    image_2 = _uint_image(view_2, channel, "second deformed view")

    # 1. to 3. Reference view -> first deformed view
    x_ref = _project_in_front(reference.camera, coordinates + reference_displacement)
    x_def1 = _match(
        image_ref,
        image_1,
        x_ref,
        _project_in_front(view_1.camera, estimated_points),
        config=disflow_config,
        margin=margin,
        name="from the reference view to the first deformed view",
    )
    # 4. to 6. First deformed view -> second deformed view
    x_def2 = _match(
        image_1,
        image_2,
        x_def1,
        _project_in_front(view_2.camera, estimated_points),
        config=disflow_config,
        margin=margin,
        name="from the first to the second deformed view",
    )

    # 7. to 9. Triangulation of the vertices matched in both deformed views
    displacement = numpy.full((mesh.n_vertices, 3), numpy.nan)
    ray_distances = numpy.full(mesh.n_vertices, numpy.nan)
    valid = numpy.isfinite(x_def1).all(axis=1) & numpy.isfinite(x_def2).all(axis=1)
    if numpy.any(valid):
        rays_1 = compute_rays(x_def1[valid], view_1.camera.intrinsic, view_1.camera.distortion, view_1.camera.extrinsic)
        rays_2 = compute_rays(x_def2[valid], view_2.camera.intrinsic, view_2.camera.distortion, view_2.camera.extrinsic)
        origins_1, directions_1 = rays_1.origins.reshape(-1, 3), rays_1.directions.reshape(-1, 3)
        origins_2, directions_2 = rays_2.origins.reshape(-1, 3), rays_2.directions.reshape(-1, 3)
        points = intersect_rays([rays_1, rays_2])  # (N, 3), NaN for parallel rays
        normals = numpy.cross(directions_1, directions_2)
        norms = numpy.linalg.norm(normals, axis=1)
        with numpy.errstate(invalid="ignore", divide="ignore"):
            distances = numpy.abs(numpy.einsum("ij,ij->i", origins_2 - origins_1, normals)) / norms
            in_front = (numpy.einsum("ij,ij->i", points - origins_1, directions_1) > 0) & (
                numpy.einsum("ij,ij->i", points - origins_2, directions_2) > 0
            )
        indices = numpy.flatnonzero(valid)[in_front]
        displacement[indices] = points[in_front] - coordinates[indices]
        ray_distances[indices] = distances[in_front]

    return OpticalFlowDisplacementResult(
        displacement=displacement,
        ray_distances=ray_distances,
        x_ref=x_ref,
        x_def1=x_def1,
        x_def2=x_def2,
    )