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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy
from numpy.typing import ArrayLike
from scipy.ndimage import map_coordinates

from ..computer_vision.compute_rays import compute_rays
from ..computer_vision.intersect_rays import intersect_rays
from ..computer_vision.optical_flow import compute_optical_flow
from ..core.disflow_config import DisflowConfig
from ..imaging.camera import Camera
from ..imaging.view import View
from .._validation import _as_array, _validate_integer, _validate_kwargs
from ..sdic.sdic_analysis import SDICAnalysis


# =============================================================================
# Internal helpers
# =============================================================================
def _flow_channel(view: View, channel: int, name: str) -> numpy.ndarray:
    # The channel (H, W) of the image data of a view, of unsigned integer type for the optical flow
    data = view.image.image  # Read-only, the image is immutable
    if not numpy.issubdtype(data.dtype, numpy.unsignedinteger):
        raise TypeError(
            f"The image of the {name} must be of unsigned integer type for the optical flow. Got dtype {data.dtype}."
        )
    if data.ndim == 2:
        return data
    if channel >= data.shape[2]:
        raise IndexError(f"Input 'channel' must be in [0, {data.shape[2] - 1}] for the image of the {name}. Got {channel}.")
    return data[:, :, channel]


def _project_in_front(camera: Camera, points: numpy.ndarray) -> numpy.ndarray:
    # Image points (N, 2) of the points (N, 3), NaN for the points that are not finite, behind the
    # camera or outside the sensor
    image_points = numpy.full((points.shape[0], 2), numpy.nan)
    indices = numpy.flatnonzero(numpy.isfinite(points).all(axis=1))
    if indices.size == 0:
        return image_points
    projected = camera.project(points[indices]).image_points
    # A point behind a perspective camera projects as a mirrored point: compare its position
    # along the ray of the extrinsic through its own normalized point
    rays = camera.extrinsic.compute_rays(camera.extrinsic.apply(points[indices]).x)
    along = numpy.einsum("ij,ij->i", points[indices] - rays.origins.reshape(-1, 3), rays.directions.reshape(-1, 3))
    x, y = projected[:, 0], projected[:, 1]
    with numpy.errstate(invalid="ignore"):
        keep = (along > 0) & (x >= -0.5) & (x <= camera.sensor_width - 0.5) & (y >= -0.5) & (y <= camera.sensor_height - 0.5)
    image_points[indices[keep]] = projected[keep]
    return image_points


def _points_roi(points: numpy.ndarray, margin: int, height: int, width: int) -> Tuple[int, int, int, int]:
    # ROI (x, y, width, height) of the pixels around the finite image points (N, 2), widened by the
    # margin, rounded outwards and clipped to the image (as pysdic.compute_projected_roi)
    finite = points[numpy.isfinite(points).all(axis=1)]
    x_min, y_min = numpy.floor(finite.min(axis=0)).astype(int) - margin
    x_max, y_max = numpy.ceil(finite.max(axis=0)).astype(int) + margin
    x_min, y_min = max(int(x_min), 0), max(int(y_min), 0)
    x_max, y_max = min(int(x_max), width - 1), min(int(y_max), height - 1)
    return x_min, y_min, x_max - x_min + 1, y_max - y_min + 1


def _fit_affine(source: numpy.ndarray, target: numpy.ndarray, name: str) -> numpy.ndarray:
    # Least squares affine map target = [source, 1] @ M, M of shape (3, 2), on the points finite in
    # both arrays (at least 3 points not aligned)
    valid = numpy.isfinite(source).all(axis=1) & numpy.isfinite(target).all(axis=1)
    design = numpy.column_stack((source[valid], numpy.ones(int(valid.sum()))))
    if valid.sum() < 3 or numpy.linalg.matrix_rank(design) < 3:
        raise ValueError(
            f"Cannot estimate the affine alignment of the optical flow {name}: at least 3 vertices not "
            f"aligned must be projected in both views. Got {int(valid.sum())} vertices."
        )
    return numpy.linalg.lstsq(design, target[valid], rcond=None)[0]


def _match(
    view_1: View,
    view_2: View,
    source: numpy.ndarray,
    target_estimate: numpy.ndarray,
    *,
    channel: int,
    config: DisflowConfig,
    margin: int,
    name: str,
) -> numpy.ndarray:
    # Image points in view_2 of the image points 'source' (N, 2) of view_1, measured by optical flow
    # after an affine alignment of the image of view_2 on the image of view_1
    image_1 = _flow_channel(view_1, channel, f"first view of the optical flow {name}")
    image_2 = _flow_channel(view_2, channel, f"second view of the optical flow {name}")
    if image_1.shape != image_2.shape:
        raise ValueError(
            f"The images of the optical flow {name} must have the same shape. Got {image_1.shape} and {image_2.shape}."
        )
    height, width = image_1.shape

    # Affine map x2 = A x1 + b fitted on source -> target_estimate
    affine = _fit_affine(source, target_estimate, name)
    A, b = affine[:2].T, affine[2]

    # Aligned image: aligned(x) = I_2(A x + b), interpolated with the interpolation function of the
    # Image of view_2 (its ROI and spline degrees; 0 outside its ROI), so that the remaining motion
    # is small and without rotation or scale (the patches of the DIS optical flow only translate)
    x_roi, y_roi, w_roi, h_roi = _points_roi(source, margin, height, width)
    rows, columns = numpy.mgrid[y_roi : y_roi + h_roi, x_roi : x_roi + w_roi]
    pixels = numpy.stack((columns.ravel(), rows.ravel()), axis=1).astype(numpy.float64)  # (P, 2) image points
    (values,) = view_2.image.evaluate_at_image_points(pixels @ A.T + b)
    if values.ndim == 2:
        values = values[:, channel]
    maximum = numpy.iinfo(image_2.dtype).max
    aligned = numpy.zeros_like(image_1)
    aligned[rows.ravel(), columns.ravel()] = numpy.clip(numpy.round(numpy.nan_to_num(values, nan=0.0)), 0, maximum)

    # Optical flow on the ROI of the points: aligned(x + F(x)) = I_1(x)
    flow_x, flow_y = compute_optical_flow(image_1, aligned, region=(x_roi, y_roi, w_roi, h_roi), disflow_config=config)

    # Bilinear interpolation of the flow at the points (NaN outside the region), then x2 = A (x1 + F(x1)) + b
    valid = numpy.isfinite(source).all(axis=1)
    coordinates = source[valid][:, ::-1].T  # (row = y, column = x)
    flow = numpy.stack(
        [map_coordinates(component, coordinates, order=1, mode="constant", cval=numpy.nan) for component in (flow_x, flow_y)],
        axis=1,
    )
    matched = numpy.full_like(source, numpy.nan)
    matched[valid] = (source[valid] + flow) @ A.T + b
    return matched


# =============================================================================
# Result
# =============================================================================
@dataclass(frozen=True)
class OpticalFlowDisplacementResult:
    r"""
    Result of :func:`pysdic.workflows.compute_optical_flow_displacement`.

    The arrays are indexed by the vertices of the mesh of the analysis. A vertex that could not
    be measured in all the deformed views (outside an image, flow not defined, behind a camera)
    has a NaN ``displacement`` and ``ray_distances``, and NaN image points in the views from the
    one where it was lost.

    Attributes
    ----------
    displacement : numpy.ndarray
        The estimated displacement of the vertices at the deformed frame, of shape
        :math:`(N_v, 3)`, relative to the coordinates of the mesh.

    ray_distances : numpy.ndarray
        For each vertex, the largest distance between the triangulated point and the rays of the
        deformed views, of shape :math:`(N_v,)`, in the unit of the mesh. Zero for a perfect
        matching: large values reveal wrong matchings.

    image_points_ref : numpy.ndarray
        The image points of the vertices in the reference view (projection of the mesh displaced
        by the displacement of the reference frame), of shape :math:`(N_v, 2)`.

    image_points_def : numpy.ndarray
        The image points of the vertices measured by optical flow in each deformed view, of shape
        :math:`(K, N_v, 2)`, in the order of ``def_views``.

    """

    displacement: numpy.ndarray
    ray_distances: numpy.ndarray
    image_points_ref: numpy.ndarray
    image_points_def: numpy.ndarray

    @property
    def measured(self) -> numpy.ndarray:
        r"""[Get] The boolean mask :math:`(N_v,)` of the vertices with a finite displacement."""
        return numpy.isfinite(self.displacement).all(axis=1)


# =============================================================================
# Workflow
# =============================================================================
def compute_optical_flow_displacement(
    analysis: SDICAnalysis,
    ref_frame: Integral,
    ref_view: Integral,
    def_frame: Integral,
    def_views: Sequence[Integral],
    *,
    disflow_config: Optional[DisflowConfig] = None,
    displacement_estimate: Optional[ArrayLike] = None,
    margin: Optional[Integral] = None,
    channel: Integral = 0,
    distortion_kwargs: Optional[Dict[str, Any]] = None,
) -> OpticalFlowDisplacementResult:
    r"""
    Estimate the displacement of the vertices of the mesh at a deformed frame by optical flow
    and triangulation, for example to initialize a :class:`pysdic.SDICCorrelationTerm`.

    The vertices are followed from one reference view to :math:`K \geq 2` views of the deformed
    frame, then triangulated:

    1. The mesh displaced by :math:`U_{ref}` (displacement of ``ref_frame``) is projected in the
       reference view: image points :math:`x_{ref}`.
    2. The optical flow from the reference view to the first deformed view ``def_views[0]``
       (usually the same camera, a temporal motion) gives the image points :math:`x_{def,0}`.
    3. The optical flow from the first deformed view to each other deformed view
       ``def_views[k]`` (usually other cameras at the same frame, a stereo motion) gives the
       image points :math:`x_{def,k}`.
    4. The rays of the :math:`K` deformed views through these image points are intersected
       (:func:`pysdic.intersect_rays`, least squares point of the :math:`K` rays): 3D positions
       :math:`X_{def}`, and :math:`U_{def} = X_{def} - X`.

    Each optical flow is computed on the ROI of the source image points (widened by
    ``margin``), and the flow is interpolated bilinearly at the points.

    **Affine alignment.** For each optical flow between the views 1 and 2, an affine map
    :math:`x \mapsto A x + b` is fitted by least squares between the image points of the
    vertices in the view 1 and their estimated image points in the view 2 (projection of the
    mesh displaced by ``displacement_estimate``): the mean motion of the region at first order
    (rotation, scale, shear and translation of the change of point of view and of the estimated
    displacement). The image of the view 2 is aligned on the view 1,
    :math:`\tilde{I}_2(x) = I_2(A x + b)`, with the interpolation function of its
    :class:`pysdic.Image` (0 outside its ROI); the optical flow :math:`F` is computed between
    :math:`I_1` and :math:`\tilde{I}_2`, and :math:`x_2 = A (x_1 + F(x_1)) + b`. The alignment
    is applied to the image, not given as an initial flow, because the patches of the DIS
    optical flow only translate: a rotation of a few degrees between two cameras makes large
    patches fail.

    .. important::

        The optical flow only corrects a small motion around the affine alignment (a few
        pixels to a few tens of pixels, depending on the texture and on ``disflow_config``).
        For a large displacement, give a ``displacement_estimate`` (e.g. the result of the
        previous frame).

    A vertex gets a NaN displacement if it is lost in one of the views (not finite, behind a
    camera or outside an image, flow not defined) or if the triangulated point is behind one of
    the deformed cameras. Replace these values before :meth:`pysdic.SDICAnalysis.set_displacement`,
    which needs finite displacements (see the example).

    .. important::

        The occlusions are not detected: a vertex hidden by another part of the surface gets
        the flow of the visible surface. Check the ``ray_distances`` of the result, or use
        :func:`pysdic.compute_visibility`.

    .. note::

        The source images of the optical flows are the full image data (``view.image.image``),
        the aligned images are interpolated in the ROI of their :class:`pysdic.Image`. The
        images must be of unsigned integer type (see :func:`pysdic.compute_optical_flow`).

    Parameters
    ----------
    analysis : :class:`pysdic.SDICAnalysis`
        The analysis, with a 3D mesh (:math:`E = 3`) and the displacement of ``ref_frame`` set.

    ref_frame : Integral
        The reference frame. Its displacement :math:`U_{ref}` must be set (see
        :meth:`pysdic.SDICAnalysis.set_zero_displacement`).

    ref_view : Integral
        The index of the reference view in ``ref_frame``.

    def_frame : Integral
        The deformed frame, different from ``ref_frame``.

    def_views : Sequence[Integral]
        The indices of at least two distinct views of ``def_frame`` used for the triangulation.
        The flow goes from the reference view to ``def_views[0]``, then from ``def_views[0]`` to
        each other view.

    disflow_config : Optional[:class:`pysdic.DisflowConfig`], optional
        The parameters of the optical flow. By default, ``DisflowConfig()``.

    displacement_estimate : Optional[ArrayLike], optional
        An estimation of the displacement at ``def_frame``, of shape :math:`(N_v, 3)`, only
        used for the affine alignments. NaN values are accepted (these vertices are not used
        for the alignments), e.g. the ``displacement`` of a previous result. By default,
        :math:`U_{ref}`: the surface **has not moved** since the reference frame (not a zero
        displacement, which would bring the surface back to the coordinates of the mesh).

    margin : Optional[Integral], optional
        The number of pixels added around the image points to define the ROI of the optical
        flows (non-negative). By default, the ``patch_size`` of the configuration, so that the
        patches of the border contain the points.

    channel : Integral, optional (default: ``0``)
        The channel of the images used for the optical flow (images with several channels).

    distortion_kwargs : Optional[Dict[str, Any]], optional
        The options of the inversion of the distortion for the rays (see
        :func:`pysdic.compute_rays`).

    Returns
    -------
    :class:`pysdic.workflows.OpticalFlowDisplacementResult`
        The estimated displacement, the distances to the rays and the image points of the
        vertices in the views.

    Raises
    ------
    TypeError
        If an input has a wrong type, or an image is not of unsigned integer type.

    ValueError
        If the mesh is not 3D, ``def_frame`` is ``ref_frame``, ``def_views`` does not contain
        at least two distinct views, the images of an optical flow have different shapes, or
        less than 3 vertices can be used for an affine alignment.

    IndexError
        If a frame or a view index is out of range.

    Examples
    --------
    Initialize the frame 1 of a stereo analysis, the vertices not measured keeping the
    displacement of the reference frame:

    .. code-block:: python

        from pysdic import DisflowConfig
        from pysdic.workflows import compute_optical_flow_displacement

        analysis.set_zero_displacement(0)
        result = compute_optical_flow_displacement(
            analysis, ref_frame=0, ref_view=0, def_frame=1, def_views=(0, 1),
            disflow_config=DisflowConfig(patch_size=30, patch_stride=6),
        )
        print(f"{result.measured.sum()} vertices measured, max distance {numpy.nanmax(result.ray_distances):.2e}")
        analysis.set_displacement(1, numpy.where(result.measured[:, None], result.displacement, analysis.get_displacement(0)))

    """
    if not isinstance(analysis, SDICAnalysis):
        raise TypeError(f"Input 'analysis' must be a SDICAnalysis. Got type {type(analysis)}.")
    mesh = analysis.mesh
    if mesh.n_dimensions != 3:
        raise ValueError(
            f"compute_optical_flow_displacement requires a 3D mesh (E = 3): the vertices are triangulated "
            f"from the rays of several views. Got a mesh of dimension {mesh.n_dimensions}."
        )
    ref_frame = _validate_integer(ref_frame, "ref_frame", min_value=0)
    def_frame = _validate_integer(def_frame, "def_frame", min_value=0)
    if def_frame == ref_frame:
        raise ValueError(f"Input 'def_frame' must be different from 'ref_frame'. Got {def_frame} for both.")
    if not isinstance(def_views, (tuple, list)) or len(def_views) < 2:
        raise TypeError(f"Input 'def_views' must be a list or a tuple of at least 2 view indices. Got {def_views!r}.")
    def_views = [_validate_integer(index, f"def_views[{k}]", min_value=0) for k, index in enumerate(def_views)]
    if len(set(def_views)) != len(def_views):
        raise ValueError(f"Input 'def_views' must contain distinct views. Got {def_views}.")

    reference_displacement = analysis.get_displacement(ref_frame)  # Error if not set
    reference = analysis.get_view(ref_frame, ref_view)
    deformed = [analysis.get_view(def_frame, index) for index in def_views]

    if disflow_config is None:
        disflow_config = DisflowConfig()
    elif not isinstance(disflow_config, DisflowConfig):
        raise TypeError(f"Input 'disflow_config' must be a DisflowConfig or None. Got type {type(disflow_config)}.")
    margin = disflow_config.patch_size if margin is None else _validate_integer(margin, "margin", min_value=0)
    channel = _validate_integer(channel, "channel", min_value=0)
    distortion_kwargs = _validate_kwargs(distortion_kwargs, "distortion_kwargs")

    coordinates = mesh.coordinates  # (N_v, 3)
    if displacement_estimate is None:
        displacement_estimate = reference_displacement
    else:
        displacement_estimate = _as_array(
            displacement_estimate, "displacement_estimate", numpy.float64,
            shape=(mesh.n_vertices, 3), shape_hint="(N_v, 3)", finite=False,
        )
    estimated_points = coordinates + displacement_estimate

    # 1. and 2. Reference view -> first deformed view
    image_points_ref = _project_in_front(reference.camera, coordinates + reference_displacement)
    image_points_def: List[numpy.ndarray] = [
        _match(
            reference, deformed[0], image_points_ref, _project_in_front(deformed[0].camera, estimated_points),
            channel=channel, config=disflow_config, margin=margin,
            name=f"from the view ({ref_frame}, {ref_view}) to the view ({def_frame}, {def_views[0]})",
        )
    ]
    # 3. First deformed view -> each other deformed view
    for index, view in zip(def_views[1:], deformed[1:]):
        image_points_def.append(
            _match(
                deformed[0], view, image_points_def[0], _project_in_front(view.camera, estimated_points),
                channel=channel, config=disflow_config, margin=margin,
                name=f"from the view ({def_frame}, {def_views[0]}) to the view ({def_frame}, {index})",
            )
        )

    # 4. Triangulation of the vertices matched in all the deformed views
    displacement = numpy.full((mesh.n_vertices, 3), numpy.nan)
    ray_distances = numpy.full(mesh.n_vertices, numpy.nan)
    valid = numpy.all([numpy.isfinite(points).all(axis=1) for points in image_points_def], axis=0)
    if numpy.any(valid):
        rays = [
            compute_rays(
                points[valid], view.camera.intrinsic, view.camera.distortion, view.camera.extrinsic,
                distortion_kwargs=distortion_kwargs,
            )
            for points, view in zip(image_points_def, deformed)
        ]
        points = intersect_rays(rays)  # (N, 3), least squares point of the K rays
        in_front = numpy.isfinite(points).all(axis=1)
        distances = numpy.zeros(points.shape[0])
        for ray in rays:
            origins, directions = ray.origins.reshape(-1, 3), ray.directions.reshape(-1, 3)
            offsets = points - origins
            with numpy.errstate(invalid="ignore"):
                along = numpy.einsum("ij,ij->i", offsets, directions)
                in_front &= along > 0
                distances = numpy.maximum(distances, numpy.linalg.norm(offsets - along[:, numpy.newaxis] * directions, axis=1))
        indices = numpy.flatnonzero(valid)[in_front]
        displacement[indices] = points[in_front] - coordinates[indices]
        ray_distances[indices] = distances[in_front]

    return OpticalFlowDisplacementResult(
        displacement=displacement,
        ray_distances=ray_distances,
        image_points_ref=image_points_ref,
        image_points_def=numpy.stack(image_points_def),
    )