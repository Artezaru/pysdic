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
from typing import Optional, Union

import numpy
from numpy.typing import ArrayLike

from .. import fem
from ..computer_vision.compute_rays import compute_rays
from ..computer_vision.distorting_points import undistort_points
from ..geometry.integration_points import IntegrationPoints
from ..geometry.mesh import Mesh
from ..imaging.camera import Camera
from ..imaging.image import Image, _as_flat_pixel_mask
from ..imaging.mesh_projection import _deformed_mesh, compute_projected_bounding_box
from .._validation import _validate_integer, _validate_real


# Maximum number of samples (pixels x samples per pixel) processed at once
_MAX_SAMPLES = 500_000

# Default margin in pixels around the bounding box of the projected vertices
_DEFAULT_MARGIN = 10.0


# =============================================================================
# Internal helpers
# =============================================================================
def _as_texture(texture: Union[Image, ArrayLike]) -> Image:
    # The texture as an Image with its interpolation function (an array is converted)
    if isinstance(texture, Image):
        return texture  # An Image always has data
    texture = numpy.asarray(texture)
    if texture.ndim not in (2, 3):
        raise ValueError(
            f"Input 'texture' must be an Image or an array of shape (H, W) or (H, W, C). Got shape {texture.shape}."
        )
    return Image.from_array(texture)


def _bounding_box_pixel_mask(
    box: Optional[tuple], height: int, width: int
) -> Optional[numpy.ndarray]:
    # Boolean mask (H * W,) of the pixels whose square [x - 0.5, x + 0.5] x [y - 0.5, y + 0.5]
    # intersects the box (x_min, y_min, x_max, y_max) in image points (None: no restriction)
    if box is None:
        return None
    x_min, y_min, x_max, y_max = box
    column_min, column_max = max(int(numpy.ceil(x_min - 0.5)), 0), min(int(numpy.floor(x_max + 0.5)), width - 1)
    row_min, row_max = max(int(numpy.ceil(y_min - 0.5)), 0), min(int(numpy.floor(y_max + 0.5)), height - 1)
    mask = numpy.zeros((height, width), dtype=bool)
    if column_min <= column_max and row_min <= row_max:
        mask[row_min : row_max + 1, column_min : column_max + 1] = True
    return mask.ravel()


def _check_textured_mesh(mesh: Mesh, feature: str) -> str:
    # Surface mesh (K = 2) with an element type and a UV map. Returns the element type.
    if not isinstance(mesh, Mesh):
        raise TypeError(f"Input 'mesh' must be a Mesh. Got type {type(mesh)}.")
    element_type = mesh._check_surface_mesh(feature)
    if mesh.uvmap is None:
        raise ValueError(f"'{feature}' requires a mesh with a UV map ('mesh.uvmap').")
    return element_type


def _texture_values(mesh: Mesh, element_type: str, integration_points: IntegrationPoints, texture: Image) -> numpy.ndarray:
    # Values (N_p, C) of the texture at the UV coordinates of the points (NaN if unlocated)
    N_p, C = integration_points.n_points, texture.n_channels
    values = numpy.full((N_p, C), numpy.nan, dtype=numpy.float64)
    located = numpy.flatnonzero(integration_points.is_located())
    if located.size == 0:
        return values

    # UV coordinates interpolated in each element (the seams of the UV map are respected)
    elements = integration_points.element_indices[located]
    shape_functions = fem.compute_shape_functions(integration_points.natural_coordinates[located], element_type)
    element_uv = mesh.uvmap.reshape((mesh.n_elements, -1, 2))[elements]  # (M, N_vpe, 2)
    uv = numpy.einsum("mv,mvc->mc", shape_functions, element_uv)  # (M, 2)

    # OpenGL convention: (u, v) in [0, 1]^2 covers the whole texture, v upward
    H_t, W_t = texture.height, texture.width
    x = numpy.clip(uv[:, 0] * W_t - 0.5, 0.0, W_t - 1.0)  # Clamped to the centers of the edge pixels
    y = numpy.clip((1.0 - uv[:, 1]) * H_t - 0.5, 0.0, H_t - 1.0)
    (evaluated,) = texture.evaluate_at_image_points(numpy.stack((x, y), axis=1))  # Values only: one order
    values[located] = evaluated.reshape((located.size, C))
    return values


# =============================================================================
# Texture
# =============================================================================
def interpolate_texture(
    mesh: Mesh,
    integration_points: IntegrationPoints,
    texture: Union[Image, ArrayLike],
) -> numpy.ndarray:
    r"""
    Evaluate the texture of a surface mesh at integration points, through its UV map.

    The UV coordinates of a point :math:`p` of natural coordinates :math:`\Xi_p` in the
    element :math:`e` are interpolated from the UV coordinates of the vertices of the
    element (see :attr:`pysdic.Mesh.uvmap`):

    .. math::

        (u_p, v_p) = \sum_{k=1}^{N_{vpe}} N_k(\Xi_p) \, (u_{e,k}, v_{e,k})

    The UV map is defined per element, so the seams of the UV map are respected. The
    texture is then evaluated (with the interpolation function of the
    :class:`pysdic.Image`) at the image point:

    .. math::

        x = u \, W_t - 0.5, \qquad y = (1 - v) \, H_t - 0.5

    following the OpenGL convention of the UV maps of ``pysdic``: :math:`(u, v) \in [0, 1]^2`
    covers the whole texture of height :math:`H_t` and width :math:`W_t`, with :math:`v`
    pointing upward. The image points are clamped to the centers of the edge pixels of
    the texture (clamp to edge).

    The natural coordinates do not depend on the displacement of the mesh: the texture is
    attached to the material.

    Parameters
    ----------
    mesh : :class:`pysdic.Mesh`
        The surface mesh (:math:`K = 2`, any dimension :math:`E`), with an element type and
        a UV map.

    integration_points : :class:`pysdic.IntegrationPoints`
        The points, consistent with the mesh. The unlocated points get NaN values.

    texture : Union[:class:`pysdic.Image`, ArrayLike]
        The texture, an :class:`pysdic.Image` or an array of shape :math:`(H_t, W_t)` or
        :math:`(H_t, W_t, C)` (converted with :meth:`pysdic.Image.from_array`, cubic
        splines). With an :class:`pysdic.Image`, its interpolation degrees, region of
        interest and default value are used (points out of the ROI get the default value).

    Returns
    -------
    numpy.ndarray
        The values of the texture, a float array of shape :math:`(N_p, C)`, with
        :math:`C = 1` for a grayscale texture.

    Raises
    ------
    TypeError
        If an input has a wrong type.

    ValueError
        If the mesh is not a surface mesh with a UV map, the points are not consistent
        with the mesh, or the texture has a wrong shape.

    Examples
    --------
    Gray levels of a speckle texture at the Gauss points of a mesh:

    .. code-block:: python

        import numpy
        from pysdic import IntegrationPoints, compute_wave_speckle, interpolate_texture

        u, v = numpy.meshgrid(numpy.linspace(0, 1, 512), numpy.linspace(0, 1, 512))
        texture = compute_wave_speckle(numpy.stack((u, v), axis=-1), wavelength=0.02)

        values = interpolate_texture(mesh, integration_points, texture)  # (N_p, 1)

    """
    element_type = _check_textured_mesh(mesh, "interpolate_texture")
    if not isinstance(integration_points, IntegrationPoints):
        raise TypeError(
            f"Input 'integration_points' must be an IntegrationPoints. Got type {type(integration_points)}."
        )
    mesh.validate_integration_points(integration_points)
    return _texture_values(mesh, element_type, integration_points, _as_texture(texture))


# =============================================================================
# Synthetic images
# =============================================================================
def generate_synthetic_image(
    camera: Camera,
    mesh: Mesh,
    texture: Union[Image, ArrayLike],
    *,
    displacement: Optional[Union[str, ArrayLike]] = None,
    n_samples: Integral = 1,
    background: Real = 0.0,
    mask: Optional[ArrayLike] = None,
    margin: Optional[Real] = _DEFAULT_MARGIN,
) -> numpy.ndarray:
    r"""
    Generate the synthetic image of a textured surface mesh seen by a camera.

    Each pixel is sampled by :math:`n \times n` points of the sensor (``n_samples``). The
    material point seen by each sample is found in the (deformed) mesh, and gets the value
    of the texture at its UV coordinates (see :func:`pysdic.interpolate_texture`). The
    samples that see no point of the mesh get the ``background`` value. The value of a
    pixel is the mean of its samples, which smooths the edges of the mesh (anti-aliasing).

    The case is chosen from the dimension :math:`E` of the world points of the camera,
    which must be the dimension of the mesh:

    - :math:`E = 2` (**2D surface mesh**): the camera is inverted to get the world point of
      each sample, :math:`\vec{x}_w = \text{Extrinsic}^{-1}(\text{Distortion}^{-1}(\text{Intrinsic}^{-1}(\vec{x}_i)))`
      (see :func:`pysdic.undistort_points`), then the point is located in the mesh (see
      :meth:`pysdic.Mesh.locate_points`). This covers a camera of the plane
      (:class:`pysdic.PlanarRigidExtrinsic`, :class:`pysdic.PlanarHomographicExtrinsic`) and
      a 3D camera observing the plane :math:`Z_w = 0`
      (:class:`pysdic.WorldPlanePerspectiveExtrinsic`,
      :class:`pysdic.WorldPlaneOrthographicExtrinsic`), whose inverse is the intersection of
      the ray of the sample with the plane. The extrinsic must implement ``invert``.
    - :math:`E = 3` (**3D surface mesh**): the ray of each sample is computed (see
      :func:`pysdic.compute_rays`) and intersected with the mesh (see
      :meth:`pysdic.Mesh.cast_rays`). The first intersection is the point seen by the sample,
      so the occlusions are taken into account. The surfaces seen from the back are rendered
      as the others (the orientation of the surfaces is a convention of the user). Requires
      the optional package ``open3d``.

    The sample :math:`(i, j)` of the pixel :math:`(u, v)` (row, column) is the image point:

    .. math::

        x = v + \frac{j + 0.5}{n} - 0.5, \qquad y = u + \frac{i + 0.5}{n} - 0.5

    The points are located in the mesh displaced by ``displacement``, and the texture is
    evaluated with their natural coordinates: the texture is attached to the material and
    follows the displacement.

    To save time, only the pixels around the projected mesh are computed: the vertices of
    the deformed mesh are projected once, and the samples are restricted to the pixels
    intersecting their bounding box extended by ``margin`` pixels (see
    :func:`pysdic.compute_projected_bounding_box`). The other pixels get the ``background``
    value. The margin covers the parts of the surface projected out of the box of the
    vertices (distortion, curved elements): increase it for a strong distortion, or set
    ``margin=None`` to compute all the pixels.

    .. note::

        The image has no blur, no noise and is not quantized: use
        :func:`pysdic.add_gaussian_blur`, :func:`pysdic.add_poisson_noise`,
        :func:`pysdic.add_gaussian_noise`, :func:`pysdic.add_salt_and_pepper_noise` and
        :func:`pysdic.quantize_image` to get a realistic acquisition.

    Parameters
    ----------
    camera : :class:`pysdic.Camera`
        The camera, with world points of dimension :math:`E = 2` or :math:`3` (the dimension
        of the mesh). The size of the image is the size of its sensor.

    mesh : :class:`pysdic.Mesh`
        The surface mesh (:math:`K = 2`) in the reference configuration, with an element type
        and a UV map (see :attr:`pysdic.Mesh.uvmap`).

    texture : Union[:class:`pysdic.Image`, ArrayLike]
        The texture of the mesh, an :class:`pysdic.Image` or an array of shape
        :math:`(H_t, W_t)` or :math:`(H_t, W_t, C)` (see :func:`pysdic.interpolate_texture`).

    displacement : Optional[Union[str, ArrayLike]], optional
        The displacement of the vertices of the mesh, the name of a vertex property or an
        array of shape :math:`(N_v, E)`. By default, the mesh is not displaced.

    n_samples : Integral, optional (default: ``1``)
        The number :math:`n \geq 1` of samples per pixel along each axis
        (:math:`n^2` samples per pixel).

    background : Real, optional (default: ``0.0``)
        The value of the samples that see no point of the mesh.

    mask : Optional[ArrayLike], optional
        A boolean mask of shape :math:`(H, W)` or :math:`(H \times W,)` of the pixels to
        compute. The other pixels get the ``background`` value. By default, all the pixels
        (restricted by ``margin``).

    margin : Optional[Real], optional (default: ``10.0``)
        The margin in pixels (:math:`\geq 0`) around the bounding box of the projected
        vertices of the deformed mesh: only the pixels intersecting the extended box are
        computed (combined with ``mask``). If ``None``, all the pixels are computed (the
        behavior without restriction, e.g. if a part of the surface is projected far from
        its vertices). If no vertex has a finite projection, all the pixels are computed.

    Returns
    -------
    numpy.ndarray
        The image, a float array of shape :math:`(H, W, C)` with :math:`C` the number of
        channels of the texture (:math:`C = 1` for a grayscale texture).

    Raises
    ------
    TypeError
        If an input has a wrong type.

    ValueError
        If the mesh is not a surface mesh with a UV map, the dimension of the camera is not
        the dimension of the mesh, the extrinsic cannot be inverted (:math:`E = 2`), or an
        input has a wrong shape or value.

    Examples
    --------
    Images of a speckled plate before and after a uniform stretching, with a camera
    observing the plane :math:`Z_w = 0`:

    .. code-block:: python

        import numpy
        from pysdic import Camera, add_gaussian_noise, compute_wave_speckle, generate_synthetic_image, quantize_image

        u, v = numpy.meshgrid(numpy.linspace(0, 1, 1024), numpy.linspace(0, 1, 1024))
        texture = compute_wave_speckle(numpy.stack((u, v), axis=-1), wavelength=0.01)

        camera = Camera(1024, 1280, extrinsic=extrinsic, distortion=distortion, intrinsic=intrinsic)
        reference = generate_synthetic_image(camera, mesh, texture, n_samples=3)

        displacement = 0.01 * mesh.coordinates * [1.0, 0.0]  # 1 % along x
        deformed = generate_synthetic_image(camera, mesh, texture, displacement=displacement, n_samples=3)

        images = [
            quantize_image(add_gaussian_noise(image[..., 0], std=1.0, seed=k), bit_depth=8)
            for k, image in enumerate((reference, deformed))
        ]

    """
    if not isinstance(camera, Camera):
        raise TypeError(f"Input 'camera' must be a Camera. Got type {type(camera)}.")
    element_type = _check_textured_mesh(mesh, "generate_synthetic_image")
    E = mesh.n_dimensions
    if E not in (2, 3):
        raise ValueError(f"The mesh must be a 2D or 3D surface mesh. Got E={E}.")
    if camera.input_dim != E:
        raise ValueError(
            f"The dimension of the world points of the camera must be the dimension of the mesh. "
            f"Got {camera.input_dim} for the camera and {E} for the mesh."
        )
    if E == 2 and not camera.extrinsic.has_invert:
        raise ValueError(
            f"The extrinsic of the camera ({type(camera.extrinsic).__name__}) must implement 'invert' "
            f"to generate the image of a 2D mesh."
        )
    texture = _as_texture(texture)
    n_samples = _validate_integer(n_samples, "n_samples", min_value=1)
    background = _validate_real(background, "background")
    if margin is not None:
        margin = _validate_real(margin, "margin", non_negative=True, finite=True)
    deformed_mesh = _deformed_mesh(mesh, displacement)

    H, W, C = camera.sensor_height, camera.sensor_width, texture.n_channels
    pixel_mask = numpy.ones(H * W, dtype=bool) if mask is None else _as_flat_pixel_mask(mask, H, W)
    if margin is not None:
        # Only the pixels around the projected vertices (one projection of the vertices)
        box = compute_projected_bounding_box(deformed_mesh, camera, margin=margin)
        box_mask = _bounding_box_pixel_mask(box, H, W)
        if box_mask is not None:
            pixel_mask = pixel_mask & box_mask  # Not in place: may be the array of the user
    image = numpy.full((H * W, C), background, dtype=numpy.float64)

    # Offsets of the samples in a pixel, in image points: (n^2, 2) as (x, y)
    offsets = (numpy.arange(n_samples) + 0.5) / n_samples - 0.5
    offset_y, offset_x = numpy.meshgrid(offsets, offsets, indexing="ij")
    offsets = numpy.stack((offset_x.ravel(), offset_y.ravel()), axis=1)
    S = offsets.shape[0]

    pixels = numpy.flatnonzero(pixel_mask)
    chunk_size = max(1, _MAX_SAMPLES // S)
    for start in range(0, pixels.size, chunk_size):
        chunk = pixels[start : start + chunk_size]
        rows, columns = numpy.divmod(chunk, W)
        centers = numpy.stack((columns, rows), axis=1).astype(numpy.float64)  # (P, 2) as (x, y)
        image_points = (centers[:, numpy.newaxis, :] + offsets[numpy.newaxis]).reshape((-1, 2))  # (P S, 2)

        # Normalized points of the samples (NaN if the inversion fails)
        normalized_points = undistort_points(image_points, camera.intrinsic, camera.distortion).x
        finite = numpy.flatnonzero(numpy.all(numpy.isfinite(normalized_points), axis=1))

        if finite.size == 0:
            continue  # All the pixels of the chunk see the background

        # Material points seen by the samples
        if E == 2:
            world_points = camera.extrinsic.invert(normalized_points[finite]).x
            integration_points = deformed_mesh.locate_points(world_points)
        else:
            rays = compute_rays(normalized_points[finite], None, None, camera.extrinsic)
            integration_points = deformed_mesh.cast_rays(rays)

        values = numpy.full((image_points.shape[0], C), background, dtype=numpy.float64)
        seen = integration_points.is_located()
        texture_values = _texture_values(mesh, element_type, integration_points, texture)
        values[finite[seen]] = texture_values[seen]
        image[chunk] = values.reshape((chunk.size, S, C)).mean(axis=1)

    return image.reshape((H, W, C))