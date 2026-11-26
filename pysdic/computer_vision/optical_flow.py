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
from typing import Optional, Tuple
from numbers import Integral, Real
from numpy.typing import ArrayLike

import numpy
import matplotlib.pyplot as plt

from .._validation import (
    _validate_integer,
    _validate_real,
    _validate_str,
    _as_array,
    _check_finite,
)
from ..core.disflow_config import DisflowConfig


# ======================================================================
# Internal helpers
# ======================================================================
def _as_uint_image(image: ArrayLike, name: str) -> numpy.ndarray:
    # Check an image with shape (H, W) or (H, W, C) and an unsigned integer type
    image = numpy.asarray(image)
    if not numpy.issubdtype(image.dtype, numpy.unsignedinteger):
        raise TypeError(
            f"Input '{name}' must be an image of unsigned integer type. Got dtype {image.dtype}."
        )
    if image.ndim not in (2, 3):
        raise ValueError(
            f"Input '{name}' must be an image of shape (H, W) or (H, W, C). Got shape {image.shape}."
        )
    return image


def _validate_channel(channel: Integral, image: numpy.ndarray) -> int:
    # Check the channel index of an image (only used for the images with shape (H, W, C))
    channel = _validate_integer(channel, "channel")
    if image.ndim == 3 and not 0 <= channel < image.shape[2]:
        raise IndexError(
            f"Input 'channel' is out of the range [0, {image.shape[2]}). Got {channel}."
        )
    return channel


def _region_slices(
    region: Optional[Tuple[Integral, Integral, Integral, Integral]],
    shape: Tuple[int, ...],
    name: str,
) -> Tuple[slice, slice]:
    # Return the slices (rows, columns) of a region (x, y, width, height) inside an image of shape (H, W, ...)
    if region is None:
        return slice(0, shape[0]), slice(0, shape[1])
    if not isinstance(region, (tuple, list)) or len(region) != 4:
        raise TypeError(
            f"Input '{name}' must be a tuple (x, y, width, height) of 4 integers or None. Got {region!r}."
        )
    x = _validate_integer(region[0], f"{name}[0] (x)", min_value=0)
    y = _validate_integer(region[1], f"{name}[1] (y)", min_value=0)
    w = _validate_integer(region[2], f"{name}[2] (width)", min_value=1)
    h = _validate_integer(region[3], f"{name}[3] (height)", min_value=1)
    if x + w > shape[1] or y + h > shape[0]:
        raise ValueError(
            f"Input '{name}' = {(x, y, w, h)} is out of the image bounds (height={shape[0]}, width={shape[1]})."
        )
    return slice(y, y + h), slice(x, x + w)


def _as_flow_component(flow: ArrayLike, name: str, shape: Tuple[int, ...]) -> numpy.ndarray:
    # Check a component of the optical flow with shape (H, W) (NaN allowed: outside the region)
    return _as_array(
        flow, name, numpy.float64, shape=shape[:2], shape_hint=f"(H, W) = {shape[:2]}", finite=False
    )


def _validate_alpha(alpha: Real) -> float:
    # A blending value in [0, 1]
    alpha = _validate_real(alpha, "alpha", finite=True)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"Input 'alpha' must be in [0, 1]. Got {alpha}.")
    return alpha


# ======================================================================
# Public functions
# ======================================================================
def compute_optical_flow(
    image1: ArrayLike,
    image2: ArrayLike,
    *,
    channel: Integral = 0,
    estimate_flow_x: Optional[ArrayLike] = None,
    estimate_flow_y: Optional[ArrayLike] = None,
    region: Optional[Tuple[Integral, Integral, Integral, Integral]] = None,
    disflow_config: Optional[DisflowConfig] = None,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    r"""
    Compute the optical flow between two images using the DIS method of ``OpenCV``.

    The optical flow is computed between two images using the Dense Inverse Search (DIS)
    algorithm implemented in OpenCV. The function allows selecting a specific channel
    from multi-channel images and the parameters of the DIS algorithm are given by a
    :class:`pysdic.DisflowConfig`.

    Lets consider two images :math:`I_1` and :math:`I_2` of shape :math:`(H, W)`. The
    optical flow is represented as two 2D arrays :math:`F_x` and :math:`F_y`, each of
    shape :math:`(H, W)`, where :math:`F_x[i, j]` and :math:`F_y[i, j]` denote the
    horizontal and vertical displacements (in pixels) of the pixel located at
    :math:`(i, j)` in the first image to its corresponding position in the second image.

    .. math::

        I_2(i + F_y[i, j], j + F_x[i, j]) \approx I_1(i, j)

    With image array indices :math:`i` and :math:`j` representing the row (vertical)
    and column (horizontal) indices, respectively.

    .. note::

        - The method convert the images data to `uint8` before processing, as required by OpenCV's DIS optical flow implementation.
        - The method convert the estimated flow to `float32`, as required by OpenCV's DIS optical flow implementation.

    .. warning::

        ``x`` and ``y`` components of the optical flow are defined in image coordinates
        and not in array indices. This means that the x-component corresponds to
        the horizontal displacement (columns) and the y-component corresponds to the
        vertical displacement (rows).

        The optical flow is compute with :obj:`numpy.uint8` images, so the input images
        will be scaled to fit into the range of :obj:`numpy.uint8` if they are of a
        different unsigned integer type. The output flow will be in pixels and can be
        positive or negative depending on the direction of motion.


    Parameters
    ----------
    image1: ArrayLike
        The first image with shape :math:`(H, W)` or :math:`(H, W, C)` where
        :math:`C` is the number of channels.
        The image must be unsigned integer type.

    image2: ArrayLike
        The second image with shape :math:`(H, W)` or :math:`(H, W, C)` where
        :math:`C` is the number of channels.
        The image must be unsigned integer type.

    channel: Integral, optional
        The channel of the images to use for optical flow computation. Default is 0.

    estimate_flow_x : Optional[ArrayLike], optional
        An initial estimate for the x-component of the optical flow with shape
        :math:`(H, W)`. Default is None.

    estimate_flow_y : Optional[ArrayLike], optional
        An initial estimate for the y-component of the optical flow with shape
        :math:`(H, W)`. Default is None.

    region: Optional[Tuple[Integral, Integral, Integral, Integral]], optional
        A tuple specifying the region of interest in the format (x, y, width, height).
        If None, the entire image is computed. Default is None.

    disflow_config: Optional[DisflowConfig], optional
        Parameters of the DIS optical flow algorithm. If None, the default
        :class:`pysdic.DisflowConfig` is used. Default is None.


    Returns
    -------
    flow_x: :class:`numpy.ndarray`
        The x-component of the optical flow (horizontal displacement) in pixels with
        shape :math:`(H, W)`.
        The pixels outside the specified region (if any) will be set to :obj:`numpy.nan`.

    flow_y: :class:`numpy.ndarray`
        The y-component of the optical flow (vertical displacement) in pixels with
        shape :math:`(H, W)`.
        The pixels outside the specified region (if any) will be set to :obj:`numpy.nan`.


    See Also
    --------
    pysdic.DisflowConfig
        Parameters of the DIS optical flow algorithm.

    pysdic.display_optical_flow
        Display the optical flow overlaid on the given image using Matplotlib.


    Examples
    --------
    Create two example images and compute the optical flow between them.

    .. figure:: /_static/textures/lena_texture.png
        :align: center
        :width: 50%

        Lena image used for the example.

    .. code-block:: python
        :linenos:

        import numpy
        from pysdic import compute_optical_flow
        from pysdic import get_lena_image

        # Create two example images
        image1 = get_lena_image() # numpy array of shape (474, 474)

        # Shift the image to create a second image
        image2 = numpy.roll(image1, shift=5, axis=1) # Shift right by 5 pixels
        image2 = numpy.roll(image2, shift=3, axis=0) # Shift down by 3 pixels

        # Compute optical flow
        flow_x, flow_y = compute_optical_flow(image1, image2)

        # Check if all flow values in the center of the image are approximately (5, 3)
        valid_x = numpy.isclose(flow_x[100:300, 100:300], 5, atol=0.5)
        valid_y = numpy.isclose(flow_y[100:300, 100:300], 3, atol=0.5)
        valid = valid_x & valid_y
        assert valid.all(), "Flow values in the center of the image should be approximately (5, 3)"

    Use smaller patches with a custom configuration:

    .. code-block:: python

        from pysdic import DisflowConfig

        config = DisflowConfig(patch_size=20, patch_stride=5)
        flow_x, flow_y = compute_optical_flow(image1, image2, disflow_config=config)

    """
    # Input validation
    image1 = _as_uint_image(image1, "image1")
    image2 = _as_uint_image(image2, "image2")
    if image1.shape != image2.shape:
        raise ValueError(
            f"Inputs 'image1' and 'image2' must have the same shape. "
            f"Got {image1.shape} and {image2.shape}."
        )
    h_slice, w_slice = _region_slices(region, image1.shape, "region")
    channel = _validate_channel(channel, image1)

    if disflow_config is None:
        disflow_config = DisflowConfig()
    elif not isinstance(disflow_config, DisflowConfig):
        raise TypeError(
            f"Input 'disflow_config' must be a DisflowConfig or None. Got type {type(disflow_config)}."
        )

    if (estimate_flow_x is None) != (estimate_flow_y is None):
        raise ValueError(
            "Both 'estimate_flow_x' and 'estimate_flow_y' must be provided to use an initial "
            "flow estimate."
        )
    estimate_flow = None
    if estimate_flow_x is not None:
        estimate_flow_x = _as_flow_component(estimate_flow_x, "estimate_flow_x", image1.shape)
        estimate_flow_y = _as_flow_component(estimate_flow_y, "estimate_flow_y", image1.shape)
        # The estimate must be finite inside the region (it can be NaN outside, as a computed flow)
        _check_finite(estimate_flow_x[h_slice, w_slice], "estimate_flow_x (inside the region)")
        _check_finite(estimate_flow_y[h_slice, w_slice], "estimate_flow_y (inside the region)")
        # OpenCV uses an initial flow given as one (H, W, 2) float32 array with the size of the images
        estimate_flow = numpy.ascontiguousarray(
            numpy.stack(
                (estimate_flow_x[h_slice, w_slice], estimate_flow_y[h_slice, w_slice]), axis=-1
            ),
            dtype=numpy.float32,
        )

    dis_optical_flow = disflow_config.create_dis_optical_flow()

    # Extract the specified channel
    if image1.ndim == 3:
        img1_channel = image1[h_slice, w_slice, channel]
        img2_channel = image2[h_slice, w_slice, channel]
    else:
        img1_channel = image1[h_slice, w_slice]
        img2_channel = image2[h_slice, w_slice]

    # Convert images to uint8
    image1_range = numpy.iinfo(img1_channel.dtype).max
    image2_range = numpy.iinfo(img2_channel.dtype).max
    uint8_range = numpy.iinfo(numpy.uint8).max

    # Scale images to uint8
    alpha1 = uint8_range / image1_range if image1_range > uint8_range else 1.0
    alpha2 = uint8_range / image2_range if image2_range > uint8_range else 1.0

    img1_float64 = img1_channel.astype(numpy.float64) * alpha1
    img2_float64 = img2_channel.astype(numpy.float64) * alpha2
    img1_uint8 = numpy.round(numpy.clip(img1_float64, 0, uint8_range)).astype(
        numpy.uint8
    )
    img2_uint8 = numpy.round(numpy.clip(img2_float64, 0, uint8_range)).astype(
        numpy.uint8
    )

    # Compute optical flow (the initial flow, if any, is used and overwritten by OpenCV)
    flow = dis_optical_flow.calc(img1_uint8, img2_uint8, estimate_flow).astype(
        numpy.float64
    )

    flow_x = numpy.full(image1.shape[:2], numpy.nan, dtype=numpy.float64)
    flow_y = numpy.full(image1.shape[:2], numpy.nan, dtype=numpy.float64)
    flow_x[h_slice, w_slice] = flow[:, :, 0]
    flow_y[h_slice, w_slice] = flow[:, :, 1]
    return flow_x, flow_y


def display_optical_flow(
    image: ArrayLike,
    flow_x: ArrayLike,
    flow_y: ArrayLike,
    *,
    region: Optional[Tuple[Integral, Integral, Integral, Integral]] = None,
    display_region: Optional[Tuple[Integral, Integral, Integral, Integral]] = None,
    alpha: Real = 0.5,
    channel: Integral = 0,
    norm_cmap: str = "inferno",
    comp_cmap: str = "bwr",
    norm_vmin: Optional[Real] = None,
    norm_vmax: Optional[Real] = None,
    comp_vmin: Optional[Real] = None,
    comp_vmax: Optional[Real] = None,
) -> None:
    """
    Display the optical flow overlaid on the given image using Matplotlib.

    Parameters
    ----------
    image: ArrayLike
        The background image with shape :math:`(H, W)` or :math:`(H, W, C)`.

    flow_x: ArrayLike
        The x-component of the optical flow (horizontal displacement) in pixels with
        shape :math:`(H, W)`.

    flow_y: ArrayLike
        The y-component of the optical flow (vertical displacement) in pixels with
        shape :math:`(H, W)`.

    region: Optional[Tuple[Integral, Integral, Integral, Integral]], optional
        A tuple specifying the region of interest in the format (x, y, width, height).
        Crop the image to this region before displaying.
        If None, the entire image is displayed. Default is None.

    display_region: Optional[Tuple[Integral, Integral, Integral, Integral]], optional
        A tuple specifying the region of the flow to display in the format
        (x, y, width, height). Crop the flow to this region before displaying.
        If None, the entire flow is displayed. Default is None.

    alpha: Real, optional
        The alpha blending value for overlaying the optical flow on the image.
        Default is 0.5.

    channel: Integral, optional
        The channel of the image to display if it is multi-channel. Default is 0.

    norm_cmap: :class:`str`, optional
        The colormap for the flow magnitude. Default is 'inferno'.

    comp_cmap: :class:`str`, optional
        The colormap for the flow components. Default is 'bwr'.

    norm_vmin: Optional[Real], optional
        Minimum value for flow magnitude colormap normalization. Default is None.

    norm_vmax: Optional[Real], optional
        Maximum value for flow magnitude colormap normalization. Default is None.

    comp_vmin: Optional[Real], optional
        Minimum value for flow component colormap normalization. Default is None.

    comp_vmax: Optional[Real], optional
        Maximum value for flow component colormap normalization. Default is None.


    See Also
    --------
    pysdic.compute_optical_flow
        Compute the optical flow between two images using the DIS method of OpenCV.

    pysdic.display_optical_flow_quiver
        Display the optical flow as a quiver plot overlaid on the given
        images using Matplotlib.


    Examples
    --------
    Create two example images and compute the optical flow between them.

    .. figure:: /_static/textures/lena_texture.png
        :align: center
        :width: 50%

        Lena image used for the example.

    .. code-block:: python
        :linenos:

        import numpy
        import cv2
        from pysdic import compute_optical_flow, display_optical_flow
        from pysdic import get_lena_image

        # Create two example images
        image1 = get_lena_image() # numpy array of shape (474, 474)

        # cv2 distortion to create a second image
        image2 = cv2.undistort(image1, cameraMatrix=numpy.array(
            [[300, 0, 237], [0, 300, 237], [0, 0, 1]]
        ), distCoeffs=numpy.array([-0.2, 0.1, 0, 0]))

        # Compute optical flow
        flow_x, flow_y = compute_optical_flow(image1, image2)

        # Select a region of interest (optional)
        display_region = (10, 10, image1.shape[1]-20, image1.shape[0]-20) # (x, y, width, height)

        # Display the optical flow
        display_optical_flow(image1, flow_x, flow_y, display_region=display_region)

    .. figure:: /_static/optical_flow/display_optical_flow.png
        :align: center

        Example of optical flow visualization using display_optical_flow.

    """
    # Input validation
    image = _as_uint_image(image, "image")
    shape = image.shape[:2]
    flow_x = _as_flow_component(flow_x, "flow_x", shape)
    flow_y = _as_flow_component(flow_y, "flow_y", shape)
    channel = _validate_channel(channel, image)
    h_slice, w_slice = _region_slices(region, shape, "region")
    if display_region is not None:
        fh_slice, fw_slice = _region_slices(display_region, shape, "display_region")
        fw_slice = slice(max(fw_slice.start, w_slice.start), min(fw_slice.stop, w_slice.stop))
        fh_slice = slice(max(fh_slice.start, h_slice.start), min(fh_slice.stop, h_slice.stop))
    else:
        fw_slice = w_slice
        fh_slice = h_slice
    alpha = _validate_alpha(alpha)
    norm_cmap = _validate_str(norm_cmap, "norm_cmap")
    comp_cmap = _validate_str(comp_cmap, "comp_cmap")
    norm_vmin = _validate_real(norm_vmin, "norm_vmin", finite=True, allow_none=True)
    norm_vmax = _validate_real(norm_vmax, "norm_vmax", finite=True, allow_none=True)
    comp_vmin = _validate_real(comp_vmin, "comp_vmin", finite=True, allow_none=True)
    comp_vmax = _validate_real(comp_vmax, "comp_vmax", finite=True, allow_none=True)

    img_range = numpy.iinfo(image.dtype).max
    image = image.astype(numpy.float64) / img_range

    # Extract the useful channel
    if image.ndim == 3:
        img_channel = image[h_slice, w_slice, channel]
    else:
        img_channel = image[h_slice, w_slice]

    # Set the flow to nan outside the specified flow region
    flow_x_display = numpy.full_like(flow_x, numpy.nan)
    flow_y_display = numpy.full_like(flow_y, numpy.nan)
    flow_x_display[fh_slice, fw_slice] = flow_x[fh_slice, fw_slice]
    flow_y_display[fh_slice, fw_slice] = flow_y[fh_slice, fw_slice]

    # Extract the flow in the specified region
    flow_x_region = flow_x_display[h_slice, w_slice]
    flow_y_region = flow_y_display[h_slice, w_slice]

    # Compute flow magnitude
    flow_magnitude = numpy.sqrt(flow_x_region**2 + flow_y_region**2)

    # Plotting
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(img_channel, cmap="gray")
    fplot = ax.imshow(
        flow_magnitude, cmap=norm_cmap, alpha=alpha, vmin=norm_vmin, vmax=norm_vmax
    )
    ax.set_title(r"Flow Magnitude $\|\mathbf{F}\|$ (pixels)")
    fig.colorbar(fplot, ax=ax)

    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(img_channel, cmap="gray")
    fxplot = ax.imshow(
        flow_x_region, cmap=comp_cmap, alpha=alpha, vmin=comp_vmin, vmax=comp_vmax
    )
    ax.set_title(r"Flow X Component $F_x$ (pixels)")
    fig.colorbar(fxplot, ax=ax)

    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(img_channel, cmap="gray")
    fyplot = ax.imshow(
        flow_y_region, cmap=comp_cmap, alpha=alpha, vmin=comp_vmin, vmax=comp_vmax
    )
    ax.set_title(r"Flow Y Component $F_y$ (pixels)")
    fig.colorbar(fyplot, ax=ax)

    plt.tight_layout()
    plt.show()


def display_optical_flow_quiver(
    image1: ArrayLike,
    image2: ArrayLike,
    flow_x: ArrayLike,
    flow_y: ArrayLike,
    *,
    region: Optional[Tuple[Integral, Integral, Integral, Integral]] = None,
    display_region: Optional[Tuple[Integral, Integral, Integral, Integral]] = None,
    step: Integral = 10,
    alpha: Real = 0.8,
    color: str = "black",
    width: Real = 0.002,
    channel: Integral = 0,
) -> None:
    r"""
    Display the optical flow as a quiver plot overlaid on the given images using Matplotlib.

    The inuput image is displayed in background blued and the output image is displayed
    in background reddish. The optical flow is displayed as arrows (quiver) overlaid
    on the images, where the direction of the arrows represents the direction of motion
    and the length of the arrows represents the magnitude of motion.

    Parameters
    ----------
    image1: ArrayLike
        The first image with shape :math:`(H, W)` or :math:`(H, W, C)`.

    image2: ArrayLike
        The second image with shape :math:`(H, W)` or :math:`(H, W, C)`.

    flow_x: ArrayLike
        The x-component of the optical flow (horizontal displacement) in pixels with
        shape :math:`(H, W)`.

    flow_y: ArrayLike
        The y-component of the optical flow (vertical displacement) in pixels with
        shape :math:`(H, W)`.

    region: Optional[Tuple[Integral, Integral, Integral, Integral]], optional
        A tuple specifying the region of interest in the format (x, y, width, height).
        Crop the image to this region before displaying.
        If None, the entire image is displayed. Default is None.

    display_region: Optional[Tuple[Integral, Integral, Integral, Integral]], optional
        A tuple specifying the region of the flow to display in the format
        (x, y, width, height). Crop the flow to this region before displaying.
        If None, the entire flow is displayed. Default is None.

    step: Integral, optional
        The step size for sampling the flow vectors to display. Default is 10.

    alpha: Real, optional
        The alpha blending value for overlaying the optical flow on the images.
        Default is 0.8.

    color: str, optional
        The color of the quiver arrows. Default is 'black'.

    width: Real, optional
        The width of the quiver arrows. Default is 0.002.

    channel: Integral, optional
        The channel of the images to display if they are multi-channel. Default is 0.


    See Also
    --------
    pysdic.compute_optical_flow
        Compute the optical flow between two images using the DIS method of OpenCV.

    pysdic.display_optical_flow
        Display the optical flow overlaid on the given image using Matplotlib.


    Examples
    --------
    Create two example images and compute the optical flow between them.

    .. figure:: /_static/textures/lena_texture.png
        :align: center
        :width: 50%

        Lena image used for the example.


    .. code-block:: python

        import numpy
        import cv2
        from pysdic import compute_optical_flow, display_optical_flow_quiver
        from pysdic import get_lena_image

        # Create two example images
        image1 = get_lena_image() # numpy array of shape (474, 474)

        # cv2 distortion to create a second image
        image2 = cv2.undistort(image1, cameraMatrix=numpy.array(
            [[300, 0, 237], [0, 300, 237], [0, 0, 1]]
        ), distCoeffs=numpy.array([-0.2, 0.1, 0, 0]))

        # Compute optical flow
        flow_x, flow_y = compute_optical_flow(image1, image2)

        # Select a region of interest (optional)
        display_region = (10, 10, image1.shape[1]-20, image1.shape[0]-20) # (x, y, width, height)

        # Display the optical flow as quiver plot
        display_optical_flow_quiver(image1, image2, flow_x, flow_y, display_region=display_region)

    .. figure:: /_static/optical_flow/display_optical_flow_quiver.png
        :align: center

        Example of optical flow quiver plot visualization using display_optical_flow_quiver.

    """
    # Input validation
    image1 = _as_uint_image(image1, "image1")
    image2 = _as_uint_image(image2, "image2")
    if image1.shape != image2.shape:
        raise ValueError(
            f"Inputs 'image1' and 'image2' must have the same shape. "
            f"Got {image1.shape} and {image2.shape}."
        )
    shape = image1.shape[:2]
    flow_x = _as_flow_component(flow_x, "flow_x", shape)
    flow_y = _as_flow_component(flow_y, "flow_y", shape)
    channel = _validate_channel(channel, image1)
    h_slice, w_slice = _region_slices(region, shape, "region")
    if display_region is not None:
        fh_slice, fw_slice = _region_slices(display_region, shape, "display_region")
        fw_slice = slice(max(fw_slice.start, w_slice.start), min(fw_slice.stop, w_slice.stop))
        fh_slice = slice(max(fh_slice.start, h_slice.start), min(fh_slice.stop, h_slice.stop))
    else:
        fw_slice = w_slice
        fh_slice = h_slice
    step = _validate_integer(step, "step", min_value=1)
    alpha = _validate_alpha(alpha)
    color = _validate_str(color, "color")
    width = _validate_real(width, "width", positive=True, finite=True)

    img1_range = numpy.iinfo(image1.dtype).max
    img2_range = numpy.iinfo(image2.dtype).max
    image1 = image1.astype(numpy.float64) / img1_range
    image2 = image2.astype(numpy.float64) / img2_range

    fw_slice = slice(fw_slice.start, fw_slice.stop, step)
    fh_slice = slice(fh_slice.start, fh_slice.stop, step)

    # Extract the useful channel
    if image1.ndim == 3:
        img1_channel = image1[h_slice, w_slice, channel]
        img2_channel = image2[h_slice, w_slice, channel]
    else:
        img1_channel = image1[h_slice, w_slice]
        img2_channel = image2[h_slice, w_slice]
    crop_shape = img1_channel.shape

    img_rgb = numpy.zeros((crop_shape[0], crop_shape[1], 3), dtype=numpy.float64)
    img_rgb[..., 0] = img2_channel
    img_rgb[..., 2] = img1_channel

    # Set the flow to nan outside the specified flow region
    flow_x_display = numpy.full_like(flow_x, numpy.nan)
    flow_y_display = numpy.full_like(flow_y, numpy.nan)
    flow_x_display[fh_slice, fw_slice] = flow_x[fh_slice, fw_slice]
    flow_y_display[fh_slice, fw_slice] = flow_y[fh_slice, fw_slice]

    # Extract the flow in the specified region
    flow_x_region = flow_x_display[h_slice, w_slice]
    flow_y_region = flow_y_display[h_slice, w_slice]

    # Plotting
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img_rgb, alpha=alpha)
    ax.quiver(
        numpy.arange(flow_x_region.shape[1]),
        numpy.arange(flow_x_region.shape[0]),
        flow_x_region,
        flow_y_region,
        angles="xy",
        scale_units="xy",
        scale=1,
        color=color,
        width=width,
    )
    ax.set_title("Optical Flow F (quiver)")
    plt.tight_layout()
    plt.show()