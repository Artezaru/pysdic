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

from __future__ import annotations

import numpy

def compute_bouguer_law(
    surface_points: numpy.ndarray,
    surface_normals: numpy.ndarray,
    light_position: numpy.ndarray,
) -> numpy.ndarray:
    r"""
    Compute the Bouguer law ratio between irradiance :math:`E` at source intensity :math:`I`.

    .. math::

        \frac{E}{I} = \frac{\cos \theta}{r^2}

    where :math:`\theta` is the angle between the surface normal and the light direction,

    .. note ::

        - If :obj:`light_position` is mulpti-dimensional, the function returns an array with the last dimension corresponding to each light source.
        - If :obj:`surface_normals` is a 1D array, it is treated as a single surface point and the corresponding dimension is omitted in the output.

    Parameters
    ----------
    surface_points : :class:`numpy.ndarray`
        An array of shape :math:`(N, E)` representing the coordinates of :math:`N` surface points in :math:`E`-dimensional space.

    surface_normals : :class:`numpy.ndarray`
        An array of shape :math:`(N, E)` representing the normal vectors at each surface point.

    light_position : :class:`numpy.ndarray`
        An array of shape :math:`(E,)` or :math:`(M, E)` representing the position(s) of the light source(s).
        If a 1D array is provided, it is treated as a single light source and the second dimension is omitted in the output.

    Returns
    -------
    :class:`numpy.ndarray`
        An array of shape :math:`(N,)` or :math:`(N, M)` representing the Bouguer law ratio :math:`\frac{E}{I}` at each surface point for each light source.
    
        
    Raises
    ------
    ValueError
        If the dimensions of the input arrays are inconsistent.

    
    Examples
    --------
    
    >>> import numpy
    >>> from pysdic.photometric import compute_bouguer_law
    >>> surface_points = numpy.array([[0.0, 0.0, 0.0],
    ...                               [1.0, 0.0, 0.0],
    ...                               [0.0, 1.0, 0.0]])
    >>> surface_normals = numpy.array([[0.0, 0.0, 1.0],
    ...                                 [0.0, 0.0, 1.0],
    ...                                 [0.0, 0.0, 1.0]])
    >>> light_position = numpy.array([0.0, 0.0, 10.0])
    >>> bouguer_ratios = compute_bouguer_law(surface_points, surface_normals, light_position)
    >>> print(bouguer_ratios)
    [0.01       0.00990099 0.00990099]

    """
    skip_light_dim = False

    # Input validation
    surface_points = numpy.asarray(surface_points)
    if not surface_points.ndim == 2:
        raise ValueError("surface_points must be a 2D array with shape (N_p, E).")
    if not numpy.issubdtype(surface_points.dtype, numpy.floating):
        surface_points = surface_points.astype(numpy.float64)
    
    surface_normals = numpy.asarray(surface_normals)
    if not surface_normals.ndim == 2:
        raise ValueError("surface_normals must be a 2D array with shape (N_p, E).")
    if not numpy.issubdtype(surface_normals.dtype, numpy.floating):
        surface_normals = surface_normals.astype(numpy.float64)
    if not surface_points.shape[0] == surface_normals.shape[0]:
        raise ValueError("surface_points and surface_normals must have the same number of points (N_p).")
    if not surface_points.shape[1] == surface_normals.shape[1]:
        raise ValueError("surface_points and surface_normals must have the same number of spatial dimensions (E).")

    light_position = numpy.asarray(light_position)
    if light_position.ndim == 1:
        light_position = light_position[numpy.newaxis, :]  # Shape: (1, E)
        skip_light_dim = True
    if not light_position.ndim == 2 or light_position.shape[1] != surface_points.shape[1]:
        raise ValueError("light_position must be a 1D array with shape (E,) or a 2D array with shape (N_l, E).")
    if not numpy.issubdtype(light_position.dtype, numpy.floating):
        light_position = light_position.astype(numpy.float64)
    
    # Compute vectors from surface points to light sources
    vectors_to_light = light_position[numpy.newaxis, :, :] - surface_points[:, numpy.newaxis, :]  # Shape: (N_p, N_l, E)

    # Compute distances to light sources
    distances = numpy.linalg.norm(vectors_to_light, axis=-1)  # Shape: (N_p, N_l)

    # Normalize vectors to light sources
    directions_to_light = vectors_to_light / distances[:, :, numpy.newaxis]  # Shape: (N_p, N_l, E)

    # Compute cos(theta) using dot product between surface normals and light directions
    cos_theta = numpy.einsum('ne,nle->nl', surface_normals, directions_to_light)  # Shape: (N_p, N_l)

    # Compute Bouguer law ratio E/I
    bouguer_ratio = cos_theta / (distances ** 2)  # Shape: (N_p, N_l)

    if skip_light_dim:
        bouguer_ratio = bouguer_ratio[:, 0]  # Shape: (N_p,)
    return bouguer_ratio


