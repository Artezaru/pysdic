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
from numbers import Real
from numpy.typing import ArrayLike

import numpy

def compute_bouguer_law(
    surface_points: ArrayLike,
    surface_normals: ArrayLike,
    light_positions: ArrayLike,
) -> numpy.ndarray:
    r"""
    Compute the Bouguer law ratio between irradiance :math:`E` at source intensity :math:`I` for a set of :math:`N_p` surface points and a set of :math:`N_l` light source positions.
    
    .. math::

        \frac{E}{I} = \frac{\cos \theta}{r^2}

    where :math:`\theta` is the angle between the surface normal and the light direction computed as the positive dot product between the surface normal and the direction vector.
    Note that :math:`(A \cdot B)` denotes the positive dot product between vectors :math:`A` and :math:`B` given by :math:`\max(0, A^T B)`.
        
    .. note::
    
        The inputs arrays will be converted to :obj:`numpy.float64` for computation.
        The output array will be also of type :obj:`numpy.float64`.


    Parameters
    ----------
    surface_points: ArrayLike
        An array of shape :math:`(N_p, 3)` or :math:`(3,)` representing the coordinates of :math:`N_p` surface points in 3-dimensional space.
        If a 1D array is provided, it is treated as a single surface point :math:`(1, 3)`.

    surface_normals: ArrayLike
        An array of shape :math:`(N_p, 3)` or :math:`(3,)` representing the normal vectors at each surface point.
        Must have the same shape as :obj:`surface_points`.

    light_positions: ArrayLike
        An array of shape :math:`(N_l, 3)` or :math:`(3,)` representing the position(s) of the light source(s).        
        If a 1D array is provided, it is treated as a single light source :math:`(1, 3)`.
        

    Returns
    -------
    :class:`numpy.ndarray`
        An array of :math:`(N_p, N_l)` representing the Bouguer law ratio :math:`\frac{E}{I}` at each surface point for each light source.
    
        
    Raises
    ------
    TypeError
        If the inputs cannot be converted to floating-point numpy arrays.
        
    ValueError
        If the dimensions of the input arrays are inconsistent.

    
    Examples
    --------
    Lets compute the Bouguer law ratio for a set of surface points and normals with a single light source:
    
    .. code-block:: python
        :linenos:
        
        import numpy
        from pysdic import compute_bouguer_law
        
        surface_points = numpy.array(
            [[0.0, 0.0, 0.0],
             [1.0, 0.0, 0.0],
             [0.0, 1.0, 0.0]]
        )
        surface_normals = numpy.array(
            [[0.0, 0.0, 1.0],
             [0.0, 0.0, 1.0],
             [0.0, 0.0, 1.0]]
        )
        light_positions = numpy.array([0.0, 0.0, 10.0])
        
        bouguer_ratios = compute_bouguer_law(
            surface_points,
            surface_normals,
            light_positions
        )
        
        print(f"Bouguer ratios (Shape: {bouguer_ratios.shape}):")
        print(bouguer_ratios)
    
    .. code-block:: console
    
        Bouguer ratios (Shape: (3, 1)):
        [[0.01      ]
         [0.00985185]
         [0.00985185]]
                

    """
    # Input validation
    surface_points = numpy.asarray(surface_points, dtype=numpy.float64)
    surface_normals = numpy.asarray(surface_normals, dtype=numpy.float64)
    if not surface_points.shape == surface_normals.shape:
        raise ValueError("surface_points and surface_normals must have the same shape (N_p, 3) or (3,).")
    
    if surface_points.ndim == 1:
        surface_points = surface_points[numpy.newaxis, :]  # Shape: (1, 3)
        surface_normals = surface_normals[numpy.newaxis, :]  # Shape: (1, 3)
    if not surface_points.ndim == 2 or surface_points.shape[1] != 3:
        raise ValueError("surface_points must be a 2D array with shape (N_p, 3).")
    
    light_positions = numpy.asarray(light_positions, dtype=numpy.float64)
    if light_positions.ndim == 1:
        light_positions = light_positions[numpy.newaxis, :]  # Shape: (1, 3)
    if not light_positions.ndim == 2 or light_positions.shape[1] != 3:
        raise ValueError("light_positions must be a 1D array with shape (3,) or a 2D array with shape (N_l, 3).")
    
    # Compute vectors from surface points to light sources
    vectors_to_light = light_positions[numpy.newaxis, :, :] - surface_points[:, numpy.newaxis, :]  # Shape: (N_p, N_l, 3)

    # Compute distances to light sources
    distances = numpy.linalg.norm(vectors_to_light, axis=-1)  # Shape: (N_p, N_l)

    # Normalize vectors to light sources
    directions_to_light = vectors_to_light / distances[:, :, numpy.newaxis]  # Shape: (N_p, N_l, 3)

    # Compute cos(theta) using dot product between surface normals and light directions
    cos_theta = numpy.einsum('ne,nle->nl', surface_normals, directions_to_light)  # Shape: (N_p, N_l)
    cos_theta = numpy.maximum(0, cos_theta)  # Ensure non-negative values for cos(theta)

    # Compute Bouguer law ratio E/I
    bouguer_ratio = cos_theta / (distances ** 2)  # Shape: (N_p, N_l)

    return bouguer_ratio





def compute_brdf_ward(
    surface_points: ArrayLike,
    surface_normals: ArrayLike,
    light_positions: ArrayLike,
    observer_positions: ArrayLike,
    parameters: ArrayLike,
    *,
    default: Real = 0.0,
) -> numpy.ndarray:
    r"""
    Compute the Bidirectional Reflectance Distribution Function (BRDF) using Ward's model for given parameters :math:`(\rho_d, \rho_s, \sigma)` (ie. diffuse, specular, and roughness).

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

    .. note::

        The inputs arrays will be converted to :obj:`numpy.float64` for computation.
        The output array will be also of type :obj:`numpy.float64`.
        

    Parameters
    ----------
    surface_points: ArrayLike
        Array of shape :math:`(N_p, 3)` or :math:`(3,)` representing the coordinates of :math:`N_p` surface points in 3-dimensional space.
        If a 1D array is provided, it is treated as a single surface point :math:`(1, 3)`.
        
    surface_normals: ArrayLike
        Array of shape :math:`(N_p, 3)` or :math:`(3,)` representing the normal vectors at each surface point.
        Must have the same shape as :obj:`surface_points`.
        
    light_positions: ArrayLike
        Array of shape :math:`(N_l, 3)` or :math:`(3,)` representing the position(s) of the light source(s).        
        If a 1D array is provided, it is treated as a single light source :math:`(1, 3)`.
        
    observer_positions: ArrayLike
        Array of shape :math:`(N_o, 3)` or :math:`(3,)` representing the position(s) of the observer(s).        
        If a 1D array is provided, it is treated as a single observer :math:`(1, 3)`.
        
    parameters: ArrayLike
        Array of shape :math:`(N_{\text{models}}, 3)` or :math:`(3,)` representing the BRDF parameters :math:`(\rho_d, \rho_s, \sigma)` for each model.
        Diffuse coefficient :math:`\rho_d` must be non-negative.
        Specular coefficient :math:`\rho_s` must be non-negative.
        Roughness :math:`\sigma` must be positive.
        If a 1D array is provided, it is treated as a single set of parameters :math:`(1, 3)`.
        
    default: Real, optional
        Default value to use for invalid BRDF computations (e.g., when input or observer directions are below the surface).
        By default, :obj:`0.0` is used.


    Returns
    -------
    brdf: :class:`numpy.ndarray`
        Array of shape :math:`(N_p, N_l, N_o, N_{\text{models}})` representing the BRDF values at each :math:`(N_p)` surface point for each :math:`(N_l)` light source and each :math:`(N_o)` observer position for each :math:`(N_{\text{models}})` set of parameters.


    Raises
    ------
    TypeError
        If the inputs cannot be converted to floating-point numpy arrays.
        
    ValueError
        If input arrays do not have the correct dimensions or if coefficients are not valid numbers.


    Examples
    --------
    Basic usage with single point, light source, and observer:
    
    .. code-block:: python
        :linenos:
    
        import numpy
        form pysdic import compute_brdf_ward
        
        surface_points = numpy.array([
            [0.0, 0.0, 0.0]
            [1.0, 0.0, 0.0]
        ])
        
        surface_normals = numpy.array([
            [0.0, 0.0, 1.0]
            [0.0, 0.0, 1.0]
        ])
        
        light_positions = numpy.array([10.0, 0.0, 10.0])
        
        observer_positions = numpy.array([0.0, 0.0, 10.0])
        
        brdf_values = compute_brdf_ward(
            surface_points,
            surface_normals,
            light_positions,
            observer_positions,
            parameters=numpy.array([0.5, 0.5, 0.2])
        )

        print(f"BRDF values (Shape: {brdf_values.shape}):")
        print(brdf_values)
    
    .. code-block:: console
    
        BRDF values (Shape: (2, 1, 1, 1)):
        [[[0.06366198]]
         [[0.06366198]]]

    """
    # Input validation
    surface_points = numpy.asarray(surface_points, dtype=numpy.float64)
    surface_normals = numpy.asarray(surface_normals, dtype=numpy.float64)
    if not surface_points.shape == surface_normals.shape:
        raise ValueError("surface_points and surface_normals must have the same shape (N_p, 3) or (3,).")
    
    if surface_points.ndim == 1:
        surface_points = surface_points[numpy.newaxis, :]  # Shape: (1, 3)
        surface_normals = surface_normals[numpy.newaxis, :]  # Shape: (1, 3)
    if not surface_points.ndim == 2 or surface_points.shape[1] != 3:
        raise ValueError("surface_points must be a 2D array with shape (N_p, 3).")
    
    light_positions = numpy.asarray(light_positions, dtype=numpy.float64)
    if light_positions.ndim == 1:
        light_positions = light_positions[numpy.newaxis, :]  # Shape: (1, 3)
    if not light_positions.ndim == 2 or light_positions.shape[1] != 3:
        raise ValueError("light_positions must be a 1D array with shape (3,) or a 2D array with shape (N_l, 3).")
    
    observer_positions = numpy.asarray(observer_positions, dtype=numpy.float64)
    if observer_positions.ndim == 1:
        observer_positions = observer_positions[numpy.newaxis, :]  # Shape: (1, 3)
    if not observer_positions.ndim == 2 or observer_positions.shape[1] != 3:
        raise ValueError("observer_positions must be a 1D array with shape (3,) or a 2D array with shape (N_o, 3).")
    
    parameters = numpy.asarray(parameters, dtype=numpy.float64)
    if parameters.ndim == 1:
        parameters = parameters[numpy.newaxis, :]  # Shape: (1, 3)
    if not parameters.ndim == 2 or parameters.shape[1] != 3:
        raise ValueError("parameters must be a 1D array with shape (3,) or a 2D array with shape (N_models, 3).")

    if not isinstance(default, Real):
        raise ValueError("default must be a real number.")

    # Unpack parameters
    rho_d = parameters[:, 0]
    rho_s = parameters[:, 1]
    sigma = parameters[:, 2]
    
    if numpy.any(rho_d < 0):
        raise ValueError("Diffuse coefficient (rho_d) must be non-negative.")
    if numpy.any(rho_s < 0):
        raise ValueError("Specular coefficient (rho_s) must be non-negative.")
    if numpy.any(sigma <= 0):
        raise ValueError("Roughness (sigma) must be positive.")
    
    # Extract sizes
    N_p = surface_points.shape[0]
    N_l = light_positions.shape[0]
    N_o = observer_positions.shape[0]
    N_m = parameters.shape[0]
    
    # Direction vectors
    I = light_positions[numpy.newaxis, :, :] - surface_points[:, numpy.newaxis, :]  # (N_p, N_l, 3)
    I /= numpy.linalg.norm(I, axis=2, keepdims=True)
    assert I.shape == (N_p, N_l, 3)
    O = observer_positions[numpy.newaxis, :, :] - surface_points[:, numpy.newaxis, :]  # (N_p, N_o, 3)
    O /= numpy.linalg.norm(O, axis=2, keepdims=True)
    assert O.shape == (N_p, N_o, 3)

    # Compute cos(theta_i) and cos(theta_o)
    cos_theta_i = numpy.einsum('plk,pk->pl', I, surface_normals)  # (N_p, N_l)
    assert cos_theta_i.shape == (N_p, N_l)
    cos_theta_0 = numpy.einsum('pok,pk->po', O, surface_normals)  # (N_p, N_o)
    assert cos_theta_0.shape == (N_p, N_o)
    cos_theta_i = numpy.maximum(0, cos_theta_i)[:, :, numpy.newaxis]  # (N_p, N_l, 1)
    assert cos_theta_i.shape == (N_p, N_l, 1)
    cos_theta_0 = numpy.maximum(0, cos_theta_0)[:, numpy.newaxis, :]  # (N_p, 1, N_o)
    assert cos_theta_0.shape == (N_p, 1, N_o)
        
    # Broadcast to compute half-vector H
    I_full = I[:, :, numpy.newaxis, :]  # (N_p, N_l, 1, 3)
    assert I_full.shape == (N_p, N_l, 1, 3)
    O_full = O[:, numpy.newaxis, :, :]  # (N_p, 1, N_o, 3)
    assert O_full.shape == (N_p,1, N_o, 3)
    H = I_full + O_full
    assert H.shape == (N_p, N_l, N_o, 3)
    H /= numpy.linalg.norm(H, axis=3, keepdims=True)  # (N_p, N_l, N_o, 3)
    assert H.shape == (N_p, N_l, N_o, 3)

    # Compute cos(delta)
    cos_delta = numpy.einsum('plok,pk->plo', H, surface_normals)
    cos_delta = numpy.maximum(0, cos_delta)  # (N_p, N_l, N_o)
    assert cos_delta.shape == (N_p, N_l, N_o)

    # Mask for invalid BRDF (light/observer below surface)
    zero_brdf_mask = (cos_theta_i < 1e-10) | (cos_theta_0 < 1e-10) | (cos_delta < 1e-10)
    assert zero_brdf_mask.shape == (N_p, N_l, N_o)
    zero_brdf_mask = numpy.broadcast_to(zero_brdf_mask[:, :, :, numpy.newaxis], (N_p, N_l, N_o, N_m))
    assert zero_brdf_mask.shape == (N_p, N_l, N_o, N_m)

    # Broadcast angles and parameters
    cos_theta_i = numpy.broadcast_to(cos_theta_i[:, :, :, numpy.newaxis], (N_p, N_l, N_o, N_m))
    assert cos_theta_i.shape == (N_p, N_l, N_o, N_m)
    cos_theta_0 = numpy.broadcast_to(cos_theta_0[:, :, :, numpy.newaxis], (N_p, N_l, N_o, N_m))
    assert cos_theta_0.shape == (N_p, N_l, N_o, N_m)
    cos_delta = numpy.broadcast_to(cos_delta[:, :, :, numpy.newaxis], (N_p, N_l, N_o, N_m))
    assert cos_delta.shape == (N_p, N_l, N_o, N_m)
    tan_2_delta = (1 - cos_delta**2) / (cos_delta**2 + 1e-20)  # Add epsilon to avoid div by zero
    assert tan_2_delta.shape == (N_p, N_l, N_o, N_m)

    rho_d = rho_d[numpy.newaxis, numpy.newaxis, numpy.newaxis, :]
    assert rho_d.shape == (1, 1, 1, N_m)
    rho_d = numpy.broadcast_to(rho_d, (N_p, N_l, N_o, N_m))
    assert rho_d.shape == (N_p, N_l, N_o, N_m)
    rho_s = rho_s[numpy.newaxis, numpy.newaxis, numpy.newaxis, :]
    assert rho_s.shape == (1, 1, 1, N_m)
    rho_s = numpy.broadcast_to(rho_s, (N_p, N_l, N_o, N_m))
    assert rho_s.shape == (N_p, N_l, N_o, N_m)
    sigma = sigma[numpy.newaxis, numpy.newaxis, numpy.newaxis, :]
    assert sigma.shape == (1, 1, 1, N_m)
    sigma = numpy.broadcast_to(sigma, (N_p, N_l, N_o, N_m))
    assert sigma.shape == (N_p, N_l, N_o, N_m)

    # Ward BRDF
    brdf = (rho_d / numpy.pi) + (rho_s / (4 * numpy.pi * sigma**2 * numpy.sqrt(cos_theta_i * cos_theta_0))) * \
           numpy.exp(-tan_2_delta / sigma**2)
    assert brdf.shape == (N_p, N_l, N_o, N_m)

    # Apply zero mask
    brdf[zero_brdf_mask] = default
    assert brdf.shape == (N_p, N_l, N_o, N_m)

    return brdf




def compute_brdf_beckmann(
    surface_points: ArrayLike,
    surface_normals: ArrayLike,
    light_positions: ArrayLike,
    observer_positions: ArrayLike,
    parameters: ArrayLike,
    *,
    default: Real = 0.0,
) -> numpy.ndarray:
    r"""
    Compute the Bidirectional Reflectance Distribution Function (BRDF) using Beckmann's model for given parameters :math:`(\rho_d, \rho_s, \text{rms})` (ie. diffuse, specular, and RMS slope).

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

    .. note::

        The inputs arrays will be converted to :obj:`numpy.float64` for computation.
        The output array will be also of type :obj:`numpy.float64`.
        
    
    Parameters
    ----------
    surface_points: ArrayLike
        Array of shape :math:`(N_p, 3)` or :math:`(3,)` representing the coordinates of :math:`N_p` surface points in 3-dimensional space.
        If a 1D array is provided, it is treated as a single surface point :math:`(1, 3)`.
        
    surface_normals: ArrayLike
        Array of shape :math:`(N_p, 3)` or :math:`(3,)` representing the normal vectors at each surface point.
        Must have the same shape as :obj:`surface_points`.
        
    light_positions: ArrayLike
        Array of shape :math:`(N_l, 3)` or :math:`(3,)` representing the position(s) of the light source(s).        
        If a 1D array is provided, it is treated as a single light source :math:`(1, 3)`.
        
    observer_positions: ArrayLike
        Array of shape :math:`(N_o, 3)` or :math:`(3,)` representing the position(s) of the observer(s).        
        If a 1D array is provided, it is treated as a single observer :math:`(1, 3)`.
        
    parameters: ArrayLike
        Array of shape :math:`(N_{\text{models}}, 3)` or :math:`(3,)` representing the BRDF parameters :math:`(\rho_d, \rho_s, \text{rms})` for each model.
        Diffuse coefficient :math:`\rho_d` must be non-negative.
        Specular coefficient :math:`\rho_s` must be non-negative.
        RMS slope :math:`\text{rms}` must be positive.
        If a 1D array is provided, it is treated as a single set of parameters :math:`(1, 3)`.
        
    default: Real, optional
        Default value to use for invalid BRDF computations (e.g., when input or observer directions are below the surface).
        By default, :obj:`0.0` is used.
        
    
    Returns
    -------
    brdf: :class:`numpy.ndarray`
        Array of shape :math:`(N_p, N_l, N_o, N_{\text{models}})` representing the BRDF values at each :math:`(N_p)` surface point for each :math:`(N_l)` light source and each :math:`(N_o)` observer position for each :math:`(N_{\text{models}})` set of parameters.
        

    Raises
    ------
    TypeError
        If the inputs cannot be converted to floating-point numpy arrays.
        
    ValueError
        If input arrays do not have the correct dimensions or if coefficients are not valid numbers.
        
    
    Examples
    --------
    Basic usage with single point, light source, and observer:
    
    .. code-block:: python
        :linenos:
    
        import numpy
        form pysdic import compute_brdf_beckmann
        
        surface_points = numpy.array([
            [0.0, 0.0, 0.0]
            [1.0, 0.0, 0.0]
        ])
        
        surface_normals = numpy.array([
            [0.0, 0.0, 1.0]
            [0.0, 0.0, 1.0]
        ])
        
        light_positions = numpy.array([10.0, 0.0, 10.0])
        
        observer_positions = numpy.array([0.0, 0.0, 10.0])
        
        brdf_values = compute_brdf_beckmann(
            surface_points,
            surface_normals,
            light_positions,
            observer_positions,
            parameters=numpy.array([0.5, 0.5, 0.2])
        )

        print(f"BRDF values (Shape: {brdf_values.shape}):")
        print(brdf_values)

    .. code-block:: console
    
        BRDF values (Shape: (2, 1, 1, 1)):
        [[[0.07957747]]
         [[0.07957747]]]

    """
    # Input validation
    surface_points = numpy.asarray(surface_points, dtype=numpy.float64)
    surface_normals = numpy.asarray(surface_normals, dtype=numpy.float64)
    if not surface_points.shape == surface_normals.shape:
        raise ValueError("surface_points and surface_normals must have the same shape (N_p, 3) or (3,).")
    
    if surface_points.ndim == 1:
        surface_points = surface_points[numpy.newaxis, :]  # Shape: (1, 3)
        surface_normals = surface_normals[numpy.newaxis, :]  # Shape: (1, 3)
    if not surface_points.ndim == 2 or surface_points.shape[1] != 3:
        raise ValueError("surface_points must be a 2D array with shape (N_p, 3).")
    
    light_positions = numpy.asarray(light_positions, dtype=numpy.float64)
    if light_positions.ndim == 1:
        light_positions = light_positions[numpy.newaxis, :]  # Shape: (1, 3)
    if not light_positions.ndim == 2 or light_positions.shape[1] != 3:
        raise ValueError("light_positions must be a 1D array with shape (3,) or a 2D array with shape (N_l, 3).")
    
    observer_positions = numpy.asarray(observer_positions, dtype=numpy.float64)
    if observer_positions.ndim == 1:
        observer_positions = observer_positions[numpy.newaxis, :]  # Shape: (1, 3)
    if not observer_positions.ndim == 2 or observer_positions.shape[1] != 3:
        raise ValueError("observer_positions must be a 1D array with shape (3,) or a 2D array with shape (N_o, 3).")
    
    parameters = numpy.asarray(parameters, dtype=numpy.float64)
    if parameters.ndim == 1:
        parameters = parameters[numpy.newaxis, :]  # Shape: (1, 3)
    if not parameters.ndim == 2 or parameters.shape[1] != 3:
        raise ValueError("parameters must be a 1D array with shape (3,) or a 2D array with shape (N_models, 3).")

    if not isinstance(default, Real):
        raise ValueError("default must be a real number.")
    
    # Unpack parameters
    rho_d = parameters[:, 0]
    rho_s = parameters[:, 1]
    rms = parameters[:, 2]
    
    if numpy.any(rho_d < 0):
        raise ValueError("Diffuse coefficient (rho_d) must be non-negative.")
    if numpy.any(rho_s < 0):
        raise ValueError("Specular coefficient (rho_s) must be non-negative.")
    if numpy.any(rms <= 0):
        raise ValueError("RMS slope (rms) must be positive.")
    
    # Extract sizes
    N_p = surface_points.shape[0]
    N_l = light_positions.shape[0]
    N_o = observer_positions.shape[0]
    N_m = parameters.shape[0]
    
    # Direction vectors
    I = light_positions[numpy.newaxis, :, :] - surface_points[:, numpy.newaxis, :]  # (N_p, N_l, 3)
    I /= numpy.linalg.norm(I, axis=2, keepdims=True)
    O = observer_positions[numpy.newaxis, :, :] - surface_points[:, numpy.newaxis, :]  # (N_p, N_o, 3)
    O /= numpy.linalg.norm(O, axis=2, keepdims=True)

    # Compute cos(theta_i) and cos(theta_o)
    cos_theta_i = numpy.einsum('plk,pk->pl', I, surface_normals)  # (N_p, N_l)
    cos_theta_0 = numpy.einsum('pok,pk->po', O, surface_normals)  # (N_p, N_o)
    cos_theta_i = numpy.maximum(0, cos_theta_i)[:, :, numpy.newaxis]  # (N_p, N_l, 1)
    cos_theta_0 = numpy.maximum(0, cos_theta_0)[:, numpy.newaxis, :]  # (N_p, 1, N_o)
        
    # Broadcast to compute half-vector H
    I_full = I[:, :, numpy.newaxis, :]  # (N_p, N_l, 1, 3)
    O_full = O[:, numpy.newaxis, :, :]  # (N_p, 1, N_o, 3)
    H = I_full + O_full
    H /= numpy.linalg.norm(H, axis=3, keepdims=True)  # (N_p, N_l, N_o, 3)

    # Compute cos(delta)
    cos_delta = numpy.einsum('plok,pk->plo', H, surface_normals)
    cos_delta = numpy.maximum(0, cos_delta)  # (N_p, N_l, N_o)

    # Mask for invalid BRDF (light/observer below surface)
    zero_brdf_mask = (cos_theta_i < 1e-10) | (cos_theta_0 < 1e-10) | (cos_delta < 1e-10)
    zero_brdf_mask = numpy.broadcast_to(zero_brdf_mask[:, :, :, numpy.newaxis], (N_p, N_l, N_o, N_m))

    # Broadcast angles and parameters
    cos_theta_i = numpy.broadcast_to(cos_theta_i[:, :, :, numpy.newaxis], (N_p, N_l, N_o, N_m))
    cos_theta_0 = numpy.broadcast_to(cos_theta_0[:, :, :, numpy.newaxis], (N_p, N_l, N_o, N_m))
    cos_delta = numpy.broadcast_to(cos_delta[:, :, :, numpy.newaxis], (N_p, N_l, N_o, N_m))
    tan_2_delta = (1 - cos_delta**2) / (cos_delta**2 + 1e-20)  # Add epsilon to avoid div by zero

    rho_d = rho_d[numpy.newaxis, numpy.newaxis, numpy.newaxis, :]
    rho_d = numpy.broadcast_to(rho_d, (N_p, N_l, N_o, N_m))
    rho_s = rho_s[numpy.newaxis, numpy.newaxis, numpy.newaxis, :]
    rho_s = numpy.broadcast_to(rho_s, (N_p, N_l, N_o, N_m))
    rms = rms[numpy.newaxis, numpy.newaxis, numpy.newaxis, :]
    rms = numpy.broadcast_to(rms, (N_p, N_l, N_o, N_m))
    
    # Beckmann BRDF
    brdf = (rho_d / numpy.pi) + \
        (rho_s / (numpy.pi * rms**2 * cos_delta**4)) * \
        numpy.exp(-tan_2_delta / (rms**2))  # Shape: (N_p, N_l, N_o, N_m)

    # Apply the zero BRDF mask
    brdf[zero_brdf_mask] = default
    
    return brdf

    