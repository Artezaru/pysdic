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

from typing import Optional, Union
from numbers import Number

import numpy

def compute_BRDF_ward(
    surface_points: numpy.ndarray,
    surface_normals: numpy.ndarray,
    light_positions: numpy.ndarray,
    observer_positions: numpy.ndarray,
    diffuse_coefficient: Optional[float] = 0.5,
    specular_coefficient: Optional[float] = 0.5,
    roughness: Optional[float] = 0.2,
) -> Union[float, numpy.ndarray]:
    r"""
    Compute the Bidirectional Reflectance Distribution Function (BRDF) using Ward's model.

    The BRDF describes how light is reflected at an opaque surface. Ward's model accounts for both diffuse and specular reflection components.
    The equation for the BRDF in Ward's model is given by:

    .. math::

        \text{BRDF}(\theta_i, \theta_o) = \frac{\rho_d}{\pi} + \frac{\rho_s}{4 \pi \sigma^2 \sqrt{\cos(\theta_i) \cos(\theta_o)}} \exp\left(-\frac{\tan^2(\delta)}{\sigma^2}\right)

    where:

    - :math:`\theta_i = \arccos(\mathbf{I} \cdot \mathbf{N})` is the angle between the input light direction and the surface normal.
    - :math:`\theta_o = \arccos(\mathbf{O} \cdot \mathbf{N})` is the angle between the observer direction and the surface normal.
    - :math:`\delta = \arccos(\mathbf{H} \cdot \mathbf{N})` is the angle between the surface normal and the half-vector (the bisector between the input and observer directions).
    - :math:`\rho_d` is the diffuse reflection coefficient.
    - :math:`\rho_s` is the specular reflection coefficient.
    - :math:`\sigma` is the surface roughness parameter.

    .. math::

        \delta = \arccos(\mathbf{H} \cdot \mathbf{N}) \quad \text{where} \quad \mathbf{H} = \frac{\mathbf{I} + \mathbf{O}}{||\mathbf{I} + \mathbf{O}||}

    Note that :math:`(A \cdot B)` denotes the positive dot product between vectors :math:`A` and :math:`B` given by :math:`\max(0, A^T B)`.
    If the input light direction :math:`\mathbf{I}` or the observer direction :math:`\mathbf{O}` is below the surface (i.e., :math:`\theta_i > \frac{\pi}{2}` or :math:`\theta_o > \frac{\pi}{2}`), the BRDF is defined to be zero.

    .. note ::

        By default, the functions return a 3D array of shape (:math:`N_p`, :math:`N_l`, :math:`N_o`) when the inputs are 2D arrays of shapes (:math:`N_p`, :math:`E`), (:math:`N_l`, :math:`E`), and (:math:`N_o`, :math:`E`).

        If any of the input arrays are 1D arrays of shape (:math:`E`,), they are treated as single positions and the corresponding output array dimensions are omitted.

        - If :obj:`surface_points` and :obj:`surface_normals` are 1D arrays, the first dimension is omitted (shape (:math:`N_l`, :math:`N_o`)).
        - If :obj:`light_positions` is a 1D array, the second dimension is omitted (shape (:math:`N_p`, :math:`N_o`)).
        - If :obj:`observer_positions` is a 1D array, the third dimension is omitted (shape (:math:`N_p`, :math:`N_l`)).
        - If :obj:`light_positions` and :obj:`observer_positions` are 1D arrays, both the second and third dimensions are omitted (shape (:math:`N_p`,)).
        - ...
        - If all inputs are 1D arrays, the output is a scalar value.


    Parameters
    ----------
    surface_points : :class:`numpy.ndarray`
        Array of shape (:math:`N_p`, :math:`E`) or (:math:`E`,) representing the coordinates of surface points where the BRDF is evaluated where :math:`N_p` is the number of points and :math:`E` is the embedding dimension.
        If a 1D array is provided, it is treated as a single surface point and the first dimension is omitted in the output.

    surface_normals : :class:`numpy.ndarray`
        Array of shape (:math:`N_p`, :math:`E`) or (:math:`E`,) representing the normal vectors at the surface points.
        Must have the same shape as :obj:`surface_points`.

    light_positions : :class:`numpy.ndarray`
        Array of shape (:math:`E`) or (:math:`N_l`, :math:`E`) representing the positions of light sources where :math:`N_l` is the number of light sources.
        If a 1D array is provided, it is treated as a single light source and the second dimension is omitted in the output.

    observer_positions: :class:`numpy.ndarray`
        Array of shape (:math:`E`) or (:math:`N_o`, :math:`E`) representing the positions of observers where :math:`N_o` is the number of observers.
        If a 1D array is provided, it is treated as a single observer and the last dimension is omitted in the output.

    diffuse_coefficient : :class:`float`, optional
        The diffuse reflection coefficient :math:`\rho_d`. Default is 0.5.

    specular_coefficient : :class:`float`, optional
        The specular reflection coefficient :math:`\rho_s`. Default is 0.5.

    roughness : :class:`float`, optional
        The surface roughness parameter :math:`\sigma`. Default is 0.2.

    Returns
    -------
    Union[:class:`numpy.ndarray`, :class:`float`]
        Array of shape (:math:`N_p`, :math:`N_l`, :math:`N_o`) containing the computed BRDF values for each combination of surface point, light source, and observer.
        The shape of the output array is adjusted based on whether :obj:`surface_points`, :obj:`light_positions`, or :obj:`observer_positions` are provided as 1D arrays.

    
    Raises
    ------
    ValueError
        If input arrays do not have the correct dimensions or if coefficients are not valid numbers.

    
    Examples
    --------

    >>> import numpy as np
    >>> from pysdic import compute_BRDF_ward
    >>> surface_point = np.array([0.0, 0.0, 0.0])
    >>> surface_normal = np.array([0.0, 0.0, 1.0])
    >>> light_positions = np.array([10.0, 0.0, 10.0])
    >>> observer_positions = np.array([0.0, 0.0, 10.0])
    >>> brdf_values = compute_BRDF_ward(
        surface_point, # 1D
        surface_normal, # 1D
        light_positions, # 1D
        observer_positions, # 1D
        diffuse_coefficient=0.5,
        specular_coefficient=0.5,
        roughness=0.2
    )
    >>> print(brdf_values) # float value
    0.1753778555153093

    Deeling with multiple observers:

    >>> surface_point = np.array([0.0, 0.0, 0.0])
    >>> surface_normal = np.array([0.0, 0.0, 1.0])
    >>> light_positions = np.array([10.0, 0.0, 10.0])
    >>> observer_positions = np.array([[0.0, 0.0, 10.0], [10.0, 0.0, 10.0]])
    >>> brdf_values = compute_BRDF_ward(
        surface_point, # 1D
        surface_normal, # 1D
        light_positions, # 1D
        observer_positions, # 2D
        diffuse_coefficient=0.5,
        specular_coefficient=0.5,
        roughness=0.2
    )
    >>> print(brdf_values) # Array of shape (N_o,)
    [0.17537786 0.15915494]

    Deeling with multiple light sources:

    >>> surface_point = np.array([0.0, 0.0, 0.0])
    >>> surface_normal = np.array([0.0, 0.0, 1.0])
    >>> light_positions = np.array([[10.0, 0.0, 10.0], [0.0, 10.0, 10.0]])
    >>> observer_positions = np.array([0.0, 0.0, 10.0])
    >>> brdf_values = compute_BRDF_ward(
        surface_point, # 1D
        surface_normal, # 1D
        light_positions, # 2D
        observer_positions, # 1D
        diffuse_coefficient=0.5,
        specular_coefficient=0.5,
        roughness=0.2
    )
    >>> print(brdf_values) # Array of shape (N_l,)
    [0.17537786 0.17537786]

    Deeling with multiple points, light sources and observers:

    >>> surface_point = np.array([0.0, 0.0, 0.0])
    >>> surface_normal = np.array([0.0, 0.0, 1.0])
    >>> light_positions = np.array([[10.0, 0.0, 10.0], [0.0, 10.0, 10.0]])
    >>> observer_positions = np.array([[0.0, 0.0, 10.0], [10.0, 0.0, 10.0]])
    >>> brdf_values = compute_BRDF_ward(
        surface_point.reshape(1, -1), # 2D
        surface_normal.reshape(1, -1), # 2D
        light_positions, # 2D
        observer_positions, # 2D
        diffuse_coefficient=0.5,
        specular_coefficient=0.5,
        roughness=0.2
    ) 
    >>> print(brdf_values) # Array of shape (N_p, N_l, N_o)
    [[[0.17537786 0.15915494]
      [0.17537786 0.15916019]]]

    """
    skip_point_dim = False
    skip_light_dim = False
    skip_observer_dim = False

    # Input validation
    surface_points = numpy.asarray(surface_points)
    surface_normals = numpy.asarray(surface_normals)
    if surface_points.shape != surface_normals.shape:
        raise ValueError("surface_points and surface_normals must have the same shape (N_p, E).")
    
    if surface_points.ndim == 1:
        surface_points = surface_points[numpy.newaxis, :]  # Shape: (1, E)
        surface_normals = surface_normals[numpy.newaxis, :]  # Shape: (1, E)
        skip_point_dim = True
    if not surface_points.ndim == 2:
        raise ValueError("surface_points must be a 2D array with shape (N_p, E).")
    if not numpy.issubdtype(surface_points.dtype, numpy.floating):
        surface_points = surface_points.astype(numpy.float64)
    if not numpy.issubdtype(surface_normals.dtype, numpy.floating):
        surface_normals = surface_normals.astype(numpy.float64)

    light_positions = numpy.asarray(light_positions)
    if light_positions.ndim == 1:
        light_positions = light_positions[numpy.newaxis, :]  # Shape: (1, E)
        skip_light_dim = True
    if not light_positions.ndim == 2 or light_positions.shape[1] != surface_points.shape[1]:
        raise ValueError("light_positions must be a 1D array with shape (E,) or a 2D array with shape (N_l, E).")
    if not numpy.issubdtype(light_positions.dtype, numpy.floating):
        light_positions = light_positions.astype(numpy.float64)
    
    observer_positions = numpy.asarray(observer_positions)
    if observer_positions.ndim == 1:
        observer_positions = observer_positions[numpy.newaxis, :]  # Shape: (1, E)
        skip_observer_dim = True
    if not observer_positions.ndim == 2 or observer_positions.shape[1] != surface_points.shape[1]:
        raise ValueError("observer_positions must be a 1D array with shape (E,) or a 2D array with shape (N_o, E).")
    if not numpy.issubdtype(observer_positions.dtype, numpy.floating):
        observer_positions = observer_positions.astype(numpy.float64)

    if not isinstance(diffuse_coefficient, Number) or diffuse_coefficient < 0:
        raise ValueError("diffuse_coefficient must be a non-negative number.")
    
    if not isinstance(specular_coefficient, Number) or specular_coefficient < 0:
        raise ValueError("specular_coefficient must be a non-negative number.")
    
    if not isinstance(roughness, Number) or roughness <= 0:
        raise ValueError("roughness must be a positive number.")
    
    # Compute the BRDF using Ward's model
    N_p, E = surface_points.shape
    N_l = light_positions.shape[0]
    N_o = observer_positions.shape[0]

    # Compute the input and observer direction vectors
    I = light_positions[numpy.newaxis, :, :] - surface_points[:, numpy.newaxis, :]  # Shape: (N_p, N_l, E)
    I = I / numpy.linalg.norm(I, axis=2, keepdims=True) # Normalize (N_p, N_l, E)
    I = numpy.broadcast_to(I[:,:,numpy.newaxis, :], (N_p, N_l, N_o, E))  # Shape: (N_p, N_l, N_o, E)

    O = observer_positions[numpy.newaxis, :, :] - surface_points[:, numpy.newaxis, :]  # Shape: (N_p, N_o, E)
    O = O / numpy.linalg.norm(O, axis=2, keepdims=True) # Normalize (N_p, N_o, E)
    O = numpy.broadcast_to(O[:, numpy.newaxis, :, :], (N_p, N_l, N_o, E))  # Shape: (N_p, N_l, N_o, E)

    # Compute the half-vector
    H = I + O  # Shape: (N_p, N_l, N_o, E)
    H = H / numpy.linalg.norm(H, axis=3, keepdims=True) # Normalize (N_p, N_l, N_o, E)

    # Compute the angles input and ouput angles
    cos_theta_i = numpy.einsum('plne,pe->pln', I, surface_normals)  # Shape: (N_p, N_l, N_o)
    cos_theta_i = numpy.maximum(0, cos_theta_i)  # Ensure non-negative
    cos_theta_0 = numpy.einsum('plne,pe->pln', O, surface_normals)  # Shape: (N_p, N_l, N_o)
    cos_theta_0 = numpy.maximum(0, cos_theta_0)  # Ensure non-negative
    cos_delta = numpy.einsum('plne,pe->pln', H, surface_normals)  # Shape: (N_p, N_l, N_o)
    cos_delta = numpy.maximum(0, cos_delta)  # Ensure non-negative

    # Create the zero BRDF mask where either cos_theta_i or cos_theta_0 is zero
    zero_brdf_mask = (cos_theta_i < 1e-10) | (cos_theta_0 < 1e-10) | (cos_delta < 1e-10)
    valid_mask = ~zero_brdf_mask

    # Initialize BRDF array
    brdf = numpy.zeros((N_p, N_l, N_o), dtype=numpy.float64)

    # Reshape arrays for valid computations
    cos_theta_i = cos_theta_i[valid_mask]
    cos_theta_0 = cos_theta_0[valid_mask]
    cos_delta = cos_delta[valid_mask]

    # Compute tan^2(delta)
    tan_2_delta = (1 - cos_delta**2) / cos_delta**2

    # Compute the BRDF using Ward's model
    brdf[valid_mask] = (diffuse_coefficient / numpy.pi) + \
           (specular_coefficient / (numpy.maximum(4 * numpy.pi * roughness**2 * numpy.sqrt(cos_theta_i * cos_theta_0), 1e-10)) * \
           numpy.exp(- tan_2_delta / (numpy.maximum(roughness**2, 1e-10))))

    # Remove singleton dimensions if necessary
    if skip_light_dim and skip_observer_dim and skip_point_dim:
        brdf = brdf[0, 0, 0]  # Scalar
    elif skip_point_dim and skip_light_dim:
        brdf = brdf[0, 0, :]  # Shape: (N_o,)
    elif skip_point_dim and skip_observer_dim:
        brdf = brdf[0, :, 0]  # Shape: (N_l,)
    elif skip_light_dim and skip_observer_dim:
        brdf = brdf[:, 0, 0]  # Shape: (N_p,)
    elif skip_point_dim:
        brdf = brdf[0, :, :]  # Shape: (N_l, N_o)
    elif skip_light_dim:
        brdf = brdf[:, 0, :]  # Shape: (N_p, N_o)
    elif skip_observer_dim:
        brdf = brdf[:, :, 0]  # Shape: (N_p, N_l)
    return brdf




def compute_BRDF_beckmann(
    surface_points: numpy.ndarray,
    surface_normals: numpy.ndarray,
    light_positions: numpy.ndarray,
    observer_positions: numpy.ndarray,
    diffuse_coefficient: Optional[float] = 0.5,
    specular_coefficient: Optional[float] = 0.5,
    rms: Optional[float] = 0.2,
) -> numpy.ndarray:
    r"""
    Compute the Bidirectional Reflectance Distribution Function (BRDF) using Beckmann's model.

    The BRDF describes how light is reflected at an opaque surface. Beckmann's model accounts for both diffuse and specular reflection components.
    The equation for the BRDF in Beckmann's model is given by:

    .. math::

        \text{BRDF}(\theta_i, \theta_o) = \frac{\rho_d}{\pi} + \frac{\rho_s}{\pi m^2 \cos(\delta)^4} \exp\left(-\frac{\tan^2(\delta)}{m^2}\right)

    where:

    - :math:`\theta_i = \arccos(\mathbf{I} \cdot \mathbf{N})` is the angle between the input light direction and the surface normal.
    - :math:`\theta_o = \arccos(\mathbf{O} \cdot \mathbf{N})` is the angle between the observer direction and the surface normal.
    - :math:`\delta = \arccos(\mathbf{H} \cdot \mathbf{N})` is the angle between the surface normal and the half-vector (the bisector between the input and observer directions).
    - :math:`\rho_d` is the diffuse reflection coefficient.
    - :math:`\rho_s` is the specular reflection coefficient.
    - :math:`m` is the root mean square (RMS) slope of the surface.

    .. math::

        \delta = \arccos(\mathbf{H} \cdot \mathbf{N}) \quad \text{where} \quad \mathbf{H} = \frac{\mathbf{I} + \mathbf{O}}{||\mathbf{I} + \mathbf{O}||}

    Note that :math:`(A \cdot B)` denotes the positive dot product between vectors :math:`A` and :math:`B` given by :math:`\max(0, A^T B)`.
    If the input light direction :math:`\mathbf{I}` or the observer direction :math:`\mathbf{O}` is below the surface (i.e., :math:`\theta_i > \frac{\pi}{2}` or :math:`\theta_o > \frac{\pi}{2}`), the BRDF is defined to be zero.

    .. note ::

        By default, the functions return a 3D array of shape (:math:`N_p`, :math:`N_l`, :math:`N_o`) when the inputs are 2D arrays of shapes (:math:`N_p`, :math:`E`), (:math:`N_l`, :math:`E`), and (:math:`N_o`, :math:`E`).

        If any of the input arrays are 1D arrays of shape (:math:`E`,), they are treated as single positions and the corresponding output array dimensions are omitted.

        - If :obj:`surface_points` and :obj:`surface_normals` are 1D arrays, the first dimension is omitted (shape (:math:`N_l`, :math:`N_o`)).
        - If :obj:`light_positions` is a 1D array, the second dimension is omitted (shape (:math:`N_p`, :math:`N_o`)).
        - If :obj:`observer_positions` is a 1D array, the third dimension is omitted (shape (:math:`N_p`, :math:`N_l`)).
        - If :obj:`light_positions` and :obj:`observer_positions` are 1D arrays, both the second and third dimensions are omitted (shape (:math:`N_p`,)).
        - ...
        - If all inputs are 1D arrays, the output is a scalar value.


    Parameters
    ----------
    surface_points : :class:`numpy.ndarray`
        Array of shape (:math:`N_p`, :math:`E`) or (:math:`E`,) representing the coordinates of surface points where the BRDF is evaluated where :math:`N_p` is the number of points and :math:`E` is the embedding dimension.
        If a 1D array is provided, it is treated as a single surface point and the first dimension is omitted in the output.

    surface_normals : :class:`numpy.ndarray`
        Array of shape (:math:`N_p`, :math:`E`) or (:math:`E`,) representing the normal vectors at the surface points.
        Must have the same shape as :obj:`surface_points`.

    light_positions : :class:`numpy.ndarray`
        Array of shape (:math:`E`) or (:math:`N_l`, :math:`E`) representing the positions of light sources where :math:`N_l` is the number of light sources.
        If a 1D array is provided, it is treated as a single light source and the second dimension is omitted in the output.

    observer_positions : :class:`numpy.ndarray`
        Array of shape (:math:`E`) or (:math:`N_o`, :math:`E`) representing the positions of observers where :math:`N_o` is the number of observers.
        If a 1D array is provided, it is treated as a single observer and the last dimension is omitted in the output.

    diffuse_coefficient : :class:`float`, optional
        The diffuse reflection coefficient :math:`\rho_d`. Default is 0.5.

    specular_coefficient : :class:`float`, optional
        The specular reflection coefficient :math:`\rho_s`. Default is 0.5.

    rms : :class:`float`, optional
        The root mean square (RMS) slope of the surface :math:`m`. Default is 0.2.

    Returns
    -------
    Union[:class:`numpy.ndarray`, :class:`float`]
        Array of shape (:math:`N_p`, :math:`N_l`, :math:`N_o`) containing the computed BRDF values for each combination of surface point, light source, and observer.
        The shape of the output array is adjusted based on whether :obj:`surface_points`, :obj:`light_positions`, or :obj:`observer_positions` are provided as 1D arrays.


    Raises
    ------
    ValueError
        If input arrays do not have the correct dimensions or if coefficients are not valid numbers.

    
    Examples
    --------
    
    >>> import numpy as np
    >>> from pysdic import compute_BRDF_beckmann
    >>> surface_point = np.array([0.0, 0.0, 0.0])
    >>> surface_normal = np.array([0.0, 0.0, 1.0])
    >>> light_positions = np.array([10.0, 0.0, 10.0])
    >>> observer_positions = np.array([0.0, 0.0, 10.0])
    >>> brdf_values = compute_BRDF_beckmann(
        surface_point, # 1D
        surface_normal, # 1D
        light_positions, # 1D
        observer_positions, # 1D
        diffuse_coefficient=0.5,
        specular_coefficient=0.5,
        rms=0.2
    )
    >>> print(brdf_values) # float value
    0.23405289334762058

    Deeling with multiple observers:

    >>> surface_point = np.array([0.0, 0.0, 0.0])
    >>> surface_normal = np.array([0.0, 0.0, 1.0])
    >>> light_positions = np.array([10.0, 0.0, 10.0])
    >>> observer_positions = np.array([[0.0, 0.0, 10.0], [10.0, 0.0, 10.0]])
    >>> brdf_values = compute_BRDF_beckmann(
        surface_point, # 1D
        surface_normal, # 1D
        light_positions, # 1D
        observer_positions, # 2D
        diffuse_coefficient=0.5,
        specular_coefficient=0.5,
        rms=0.2
    )
    >>> print(brdf_values) # Array of shape (N_o,)
    [0.23405289 0.15915494]

    Deeling with multiple light sources:

    >>> surface_point = np.array([0.0, 0.0, 0.0])
    >>> surface_normal = np.array([0.0, 0.0, 1.0])
    >>> light_positions = np.array([[10.0, 0.0, 10.0], [0.0, 10.0, 10.0]])
    >>> observer_positions = np.array([0.0, 0.0, 10.0])
    >>> brdf_values = compute_BRDF_beckmann(
        surface_point, # 1D
        surface_normal, # 1D
        light_positions, # 2D
        observer_positions, # 1D
        diffuse_coefficient=0.5,
        specular_coefficient=0.5,
        rms=0.2
    )
    >>> print(brdf_values) # Array of shape (N_l,)
    [0.23405289 0.23405289]

    Deeling with multiple points, light sources and observers:

    >>> surface_point = np.array([0.0, 0.0, 0.0])
    >>> surface_normal = np.array([0.0, 0.0, 1.0])
    >>> light_positions = np.array([[10.0, 0.0, 10.0], [0.0, 10.0, 10.0]])
    >>> observer_positions = np.array([[0.0, 0.0, 10.0], [10.0, 0.0, 10.0]])
    >>> brdf_values = compute_BRDF_beckmann(
        surface_point.reshape(1, -1), # 2D
        surface_normal.reshape(1, -1), # 2D
        light_positions, # 2D
        observer_positions, # 2D
        diffuse_coefficient=0.5,
        specular_coefficient=0.5,
        rms=0.2
    ) 
    >>> print(brdf_values) # Array of shape (N_p, N_l, N_o)
    [[[0.23405289 0.15915494]
      [0.23405289 0.15918831]]]

    """
    skip_point_dim = False
    skip_light_dim = False
    skip_observer_dim = False

    # Input validation
    surface_points = numpy.asarray(surface_points)
    surface_normals = numpy.asarray(surface_normals)
    if surface_points.shape != surface_normals.shape:
        raise ValueError("surface_points and surface_normals must have the same shape (N_p, E).")
    
    if surface_points.ndim == 1:
        surface_points = surface_points[numpy.newaxis, :]  # Shape: (1, E)
        surface_normals = surface_normals[numpy.newaxis, :]  # Shape: (1, E)
        skip_point_dim = True
    if not surface_points.ndim == 2:
        raise ValueError("surface_points must be a 2D array with shape (N_p, E).")
    if not numpy.issubdtype(surface_points.dtype, numpy.floating):
        surface_points = surface_points.astype(numpy.float64)
    if not numpy.issubdtype(surface_normals.dtype, numpy.floating):
        surface_normals = surface_normals.astype(numpy.float64)

    light_positions = numpy.asarray(light_positions)
    if light_positions.ndim == 1:
        light_positions = light_positions[numpy.newaxis, :]  # Shape: (1, E)
        skip_light_dim = True
    if not light_positions.ndim == 2 or light_positions.shape[1] != surface_points.shape[1]:
        raise ValueError("light_positions must be a 1D array with shape (E,) or a 2D array with shape (N_l, E).")
    if not numpy.issubdtype(light_positions.dtype, numpy.floating):
        light_positions = light_positions.astype(numpy.float64)
    
    observer_positions = numpy.asarray(observer_positions)
    if observer_positions.ndim == 1:
        observer_positions = observer_positions[numpy.newaxis, :]  # Shape: (1, E)
        skip_observer_dim = True
    if not observer_positions.ndim == 2 or observer_positions.shape[1] != surface_points.shape[1]:
        raise ValueError("observer_positions must be a 1D array with shape (E,) or a 2D array with shape (N_o, E).")
    if not numpy.issubdtype(observer_positions.dtype, numpy.floating):
        observer_positions = observer_positions.astype(numpy.float64)

    if not isinstance(diffuse_coefficient, Number) or diffuse_coefficient < 0:
        raise ValueError("diffuse_coefficient must be a non-negative number.")
    
    if not isinstance(specular_coefficient, Number) or specular_coefficient < 0:
        raise ValueError("specular_coefficient must be a non-negative number.")
    
    if not isinstance(rms, Number) or rms <= 0:
        raise ValueError("rms must be a positive number.")
    
    # Compute the BRDF using Ward's model
    N_p, E = surface_points.shape
    N_l = light_positions.shape[0]
    N_o = observer_positions.shape[0]

    # Compute the input and observer direction vectors
    I = light_positions[numpy.newaxis, :, :] - surface_points[:, numpy.newaxis, :]  # Shape: (N_p, N_l, E)
    I = I / numpy.linalg.norm(I, axis=2, keepdims=True) # Normalize (N_p, N_l, E)
    I = numpy.broadcast_to(I[:,:,numpy.newaxis, :], (N_p, N_l, N_o, E))  # Shape: (N_p, N_l, N_o, E)

    O = observer_positions[numpy.newaxis, :, :] - surface_points[:, numpy.newaxis, :]  # Shape: (N_p, N_o, E)
    O = O / numpy.linalg.norm(O, axis=2, keepdims=True) # Normalize (N_p, N_o, E)
    O = numpy.broadcast_to(O[:, numpy.newaxis, :, :], (N_p, N_l, N_o, E))  # Shape: (N_p, N_l, N_o, E)

    # Compute the half-vector
    H = I + O  # Shape: (N_p, N_l, N_o, E)
    H = H / numpy.linalg.norm(H, axis=3, keepdims=True) # Normalize (N_p, N_l, N_o, E)

    # Compute the angles input and ouput angles
    cos_theta_i = numpy.einsum('plne,pe->pln', I, surface_normals)  # Shape: (N_p, N_l, N_o)
    cos_theta_i = numpy.maximum(0, cos_theta_i)  # Ensure non-negative
    cos_theta_0 = numpy.einsum('plne,pe->pln', O, surface_normals)  # Shape: (N_p, N_l, N_o)
    cos_theta_0 = numpy.maximum(0, cos_theta_0)  # Ensure non-negative
    cos_delta = numpy.einsum('plne,pe->pln', H, surface_normals)  # Shape: (N_p, N_l, N_o)
    cos_delta = numpy.maximum(0, cos_delta)  # Ensure non-negative

    # Create the zero BRDF mask where either cos_theta_i or cos_theta_0 is zero
    zero_brdf_mask = (cos_theta_i < 1e-10) | (cos_theta_0 < 1e-10)
    valid_mask = ~zero_brdf_mask

    # Initialize BRDF array
    brdf = numpy.zeros((N_p, N_l, N_o), dtype=numpy.float64)

    # Reshape arrays for valid computations
    cos_theta_i = cos_theta_i[valid_mask]
    cos_theta_0 = cos_theta_0[valid_mask]
    cos_delta = cos_delta[valid_mask]

    # Approximate tan^2(delta) using cos(delta)
    tan_2_delta = (1 - cos_delta**2) / (numpy.maximum(cos_delta**2, 1e-10))

    # Compute the BRDF using Beckmann's model
    brdf[valid_mask] = (diffuse_coefficient / numpy.pi) + \
           (specular_coefficient / (numpy.maximum(numpy.pi * rms**2 * cos_delta**4, 1e-10)) * \
           numpy.exp(- tan_2_delta / (numpy.maximum(rms**2, 1e-10))))  # Shape: (N_p, N_l, N_o)
    
    # Remove singleton dimensions if necessary
    if skip_light_dim and skip_observer_dim and skip_point_dim:
        brdf = brdf[0, 0, 0]  # Scalar
    elif skip_point_dim and skip_light_dim:
        brdf = brdf[0, 0, :]  # Shape: (N_o,)
    elif skip_point_dim and skip_observer_dim:
        brdf = brdf[0, :, 0]  # Shape: (N_l,)
    elif skip_light_dim and skip_observer_dim:
        brdf = brdf[:, 0, 0]  # Shape: (N_p,)
    elif skip_point_dim:
        brdf = brdf[0, :, :]  # Shape: (N_l, N_o)
    elif skip_light_dim:
        brdf = brdf[:, 0, :]  # Shape: (N_p, N_o)
    elif skip_observer_dim:
        brdf = brdf[:, :, 0]  # Shape: (N_p, N_l)
    return brdf

