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

from typing import List, Optional, Sequence, Union

from numpy.typing import ArrayLike

from ..geometry.mesh import Mesh
from ..imaging.camera import Camera
from ..imaging.image import Image
from ..imaging.view import View
from ..transformations.transformation import Transformation
from .sdic_analysis import SDICAnalysis


# =============================================================================
# Internal helpers
# =============================================================================
def _as_sequence(values: Sequence, name: str, item_type: type, item_name: str) -> List:
    # A non-empty list or tuple whose items are all of the given type
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"Input '{name}' must be a list or a tuple of {item_name}. Got type {type(values)}.")
    if len(values) == 0:
        raise ValueError(f"Input '{name}' must contain at least one {item_name}.")
    for index, value in enumerate(values):
        if not isinstance(value, item_type):
            raise TypeError(f"Input '{name}[{index}]' must be of type {item_name}. Got type {type(value)}.")
    return list(values)


def _create_analysis(mesh: Mesh, views: List[List[View]], times: Optional[ArrayLike]) -> SDICAnalysis:
    # Analysis whose frame f contains the views views[f], in this order
    return SDICAnalysis(mesh, views, times=times)


# =============================================================================
# Setups
# =============================================================================
def create_multicamera_analysis(
    mesh: Mesh,
    cameras: Sequence[Camera],
    images: Sequence[Sequence[Image]],
    *,
    times: Optional[ArrayLike] = None,
) -> SDICAnalysis:
    r"""
    Create an analysis (:class:`pysdic.SDICAnalysis`) for :math:`N` cameras that each acquire their own series of images.

    The frame :math:`f` contains :math:`N` views, the view :math:`i` being
    ``View(cameras[i], images[i][f])``. Each camera is shared by its views of all the
    frames (fixed calibration). The number of frames is the length of the series of
    images, which must be the same for all the cameras.

    The displacements of the frames are not set.

    Parameters
    ----------
    mesh : :class:`pysdic.Mesh`
        The mesh of the observed surface (not copied: it is locked by the analysis).

    cameras : Sequence[:class:`pysdic.Camera`]
        The :math:`N` cameras.

    images : Sequence[Sequence[:class:`pysdic.Image`]]
        The :math:`N` series of images, ``images[i][f]`` being the image of the camera
        :math:`i` at the frame :math:`f`. The images are used as given (ROI, spline
        degrees, ...) and must have the size of the sensor of their camera.

    times : Optional[ArrayLike], optional (default: ``None``)
        The times of acquisition of the frames, of shape :math:`(N_f,)`, strictly increasing
        (see :class:`pysdic.SDICAnalysis`). If None, the times are not defined.

    Returns
    -------
    :class:`pysdic.SDICAnalysis`
        The analysis with :math:`N` views per frame.

    Raises
    ------
    TypeError
        If an input has a wrong type.

    ValueError
        If the number of series of images is not :math:`N`, the series have different
        lengths, or an image does not match the sensor of its camera.

    Examples
    --------
    .. code-block:: python

        from pysdic import Image, create_multicamera_analysis

        images = [[Image.from_file(f"cam{i}_{f:03d}.tif") for f in range(10)] for i in range(3)]
        analysis = create_multicamera_analysis(mesh, [camera_0, camera_1, camera_2], images)
        analysis.n_frames, analysis.n_views(0)  # 10, 3

    """
    cameras = _as_sequence(cameras, "cameras", Camera, "Camera")
    if not isinstance(images, (list, tuple)) or len(images) != len(cameras):
        raise ValueError(
            f"Input 'images' must contain one series of images per camera ({len(cameras)} series)."
        )
    series = [_as_sequence(series, f"images[{i}]", Image, "Image") for i, series in enumerate(images)]
    n_frames = len(series[0])
    for i, images_i in enumerate(series):
        if len(images_i) != n_frames:
            raise ValueError(
                f"All the series of images must have the same length (the number of frames). "
                f"Got {n_frames} images for the camera 0 and {len(images_i)} for the camera {i}."
            )
    views = [[View(camera, images_i[frame]) for camera, images_i in zip(cameras, series)] for frame in range(n_frames)]
    return _create_analysis(mesh, views, times)


def create_bicamera_analysis(
    mesh: Mesh,
    camera_1: Camera,
    camera_2: Camera,
    images_1: Sequence[Image],
    images_2: Sequence[Image],
    *,
    times: Optional[ArrayLike] = None,
) -> SDICAnalysis:
    r"""
    Create an analysis (:class:`pysdic.SDICAnalysis`) for a stereo setup of two cameras that each acquire their own series of
    images.

    The frame :math:`f` contains two views: ``View(camera_1, images_1[f])`` (view 0) and
    ``View(camera_2, images_2[f])`` (view 1). See :func:`pysdic.create_multicamera_analysis`.

    The displacements of the frames are not set.

    Parameters
    ----------
    mesh : :class:`pysdic.Mesh`
        The mesh of the observed surface (not copied: it is locked by the analysis).

    camera_1, camera_2 : :class:`pysdic.Camera`
        The two cameras.

    images_1, images_2 : Sequence[:class:`pysdic.Image`]
        The series of images of each camera, of the same length (the number of frames).

    times : Optional[ArrayLike], optional (default: ``None``)
        The times of acquisition of the frames, of shape :math:`(N_f,)`, strictly increasing
        (see :class:`pysdic.SDICAnalysis`). If None, the times are not defined.

    Returns
    -------
    :class:`pysdic.SDICAnalysis`
        The analysis with 2 views per frame.

    Examples
    --------
    .. code-block:: python

        from pysdic import Pairing, create_bicamera_analysis

        analysis = create_bicamera_analysis(mesh, left, right, left_images, right_images)
        pairing = Pairing.stereo(0, 0, 1)  # Left / right at the frame 0

    """
    return create_multicamera_analysis(mesh, [camera_1, camera_2], [images_1, images_2], times=times)


def create_splitscreen_analysis(
    mesh: Mesh,
    cameras: Sequence[Camera],
    images: Sequence[Image],
    *,
    times: Optional[ArrayLike] = None,
) -> SDICAnalysis:
    r"""
    Create an analysis (:class:`pysdic.SDICAnalysis`) for a split-screen setup: :math:`N` virtual cameras (often 2, through a
    stereo mirror adapter) share one sensor and one series of images.

    The frame :math:`f` contains :math:`N` views, the view :math:`i` being
    ``View(cameras[i], images[f])``: the **same** :class:`pysdic.Image` object is used by
    all the views of a frame. Each camera is shared by its views of all the frames.

    .. note::

        The parts of the image seen by each camera are defined by the cameras (their
        projection), not by the image: the ROI of the image, if any, is common to all
        the views.

    The displacements of the frames are not set.

    Parameters
    ----------
    mesh : :class:`pysdic.Mesh`
        The mesh of the observed surface (not copied: it is locked by the analysis).

    cameras : Sequence[:class:`pysdic.Camera`]
        The :math:`N` virtual cameras, with the size of the sensor.

    images : Sequence[:class:`pysdic.Image`]
        The series of images, one per frame.

    times : Optional[ArrayLike], optional (default: ``None``)
        The times of acquisition of the frames, of shape :math:`(N_f,)`, strictly increasing
        (see :class:`pysdic.SDICAnalysis`). If None, the times are not defined.

    Returns
    -------
    :class:`pysdic.SDICAnalysis`
        The analysis with :math:`N` views per frame.

    Raises
    ------
    TypeError
        If an input has a wrong type.

    ValueError
        If an image does not match the sensor of a camera.

    Examples
    --------
    .. code-block:: python

        from pysdic import create_splitscreen_analysis

        analysis = create_splitscreen_analysis(mesh, [left, right], images)
        analysis.display_structure("image")  # One image per frame, shared by the 2 views

    """
    cameras = _as_sequence(cameras, "cameras", Camera, "Camera")
    images = _as_sequence(images, "images", Image, "Image")
    views = [[View(camera, image) for camera in cameras] for image in images]
    return _create_analysis(mesh, views, times)


def create_cordin_analysis(
    mesh: Mesh,
    extrinsics: Sequence[Transformation],
    intrinsic: Transformation,
    distortions: Union[Transformation, Sequence[Transformation]],
    images: Sequence[Image],
    *,
    times: Optional[ArrayLike] = None,
) -> SDICAnalysis:
    r"""
    Create an analysis (:class:`pysdic.SDICAnalysis`) for a Cordin-like setup: a high-speed camera with one sensor seen
    through :math:`N` points of view (stereo mirror adapter) and a distortion that
    changes with the frame (rotating mirror).

    The cameras are built from shared transformations:

    .. math::

        \text{Camera}[i][f] = \text{Intrinsic} \circ \text{Distortion}[f] \circ \text{Extrinsic}[i]

    and the frame :math:`f` contains :math:`N` views, the view :math:`i` being
    ``View(Camera[i][f], images[f])``: the **same** :class:`pysdic.Image` object is used
    by all the views of a frame.

    The transformations are not copied: all the cameras share the same ``intrinsic``
    object, the cameras of a frame share the same distortion and the cameras of a point
    of view share the same extrinsic. Changing the parameters of one transformation
    changes all the cameras that use it.

    The size of the sensor is the size of the images, which must all have the same
    height and width. The displacements of the frames are not set.

    Parameters
    ----------
    mesh : :class:`pysdic.Mesh`
        The mesh of the observed surface (not copied: it is locked by the analysis).

    extrinsics : Sequence[:class:`pysdic.Transformation`]
        The :math:`N` extrinsic transformations, one per point of view.

    intrinsic : :class:`pysdic.Transformation`
        The intrinsic transformation of the sensor.

    distortions : Union[:class:`pysdic.Transformation`, Sequence[:class:`pysdic.Transformation`]]
        The distortion of each frame (a sequence of length :math:`N_f`), or one
        distortion shared by all the frames. In the latter case, the :math:`N` cameras
        are the same for all the frames, as in :func:`pysdic.create_splitscreen_analysis`.

    images : Sequence[:class:`pysdic.Image`]
        The series of images, one per frame.

    times : Optional[ArrayLike], optional (default: ``None``)
        The times of acquisition of the frames, of shape :math:`(N_f,)`, strictly increasing
        (see :class:`pysdic.SDICAnalysis`). If None, the times are not defined.

    Returns
    -------
    :class:`pysdic.SDICAnalysis`
        The analysis with :math:`N` views per frame.

    Raises
    ------
    TypeError
        If an input has a wrong type.

    ValueError
        If the number of distortions is not the number of images, or the images do not
        all have the same height and width.

    Examples
    --------
    .. code-block:: python

        from pysdic import PinholeIntrinsic, create_cordin_analysis

        analysis = create_cordin_analysis(
            mesh,
            extrinsics=[left_extrinsic, right_extrinsic],
            intrinsic=PinholeIntrinsic.from_matrix(K),
            distortions=[BrownDistortion(d) for d in frame_distortions],
            images=images,
        )
        analysis.display_structure("camera")  # 2 cameras per frame, different at each frame

    """
    extrinsics = _as_sequence(extrinsics, "extrinsics", Transformation, "Transformation")
    if not isinstance(intrinsic, Transformation):
        raise TypeError(f"Input 'intrinsic' must be a Transformation. Got type {type(intrinsic)}.")
    images = _as_sequence(images, "images", Image, "Image")
    if isinstance(distortions, Transformation):
        distortions = [distortions] * len(images)
        shared = True
    else:
        distortions = _as_sequence(distortions, "distortions", Transformation, "Transformation")
        shared = False
        if len(distortions) != len(images):
            raise ValueError(
                f"Input 'distortions' must contain one distortion per frame ({len(images)} images). "
                f"Got {len(distortions)} distortions."
            )

    for frame, image in enumerate(images):
        if image.image is None:
            raise ValueError(f"The image of the frame {frame} has no data.")
        if image.shape[:2] != images[0].shape[:2]:
            raise ValueError(
                f"All the images must have the same height and width (the size of the sensor). "
                f"Got {images[0].shape[:2]} for the frame 0 and {image.shape[:2]} for the frame {frame}."
            )
    height, width = images[0].shape[:2]

    # One camera per point of view and per distortion (per frame, or shared by all the frames)
    cameras = [
        [Camera(height, width, intrinsic=intrinsic, distortion=distortion, extrinsic=extrinsic) for extrinsic in extrinsics]
        for distortion in (distortions[:1] if shared else distortions)
    ]
    views = [[View(camera, image) for camera in cameras[0 if shared else frame]] for frame, image in enumerate(images)]
    return _create_analysis(mesh, views, times)