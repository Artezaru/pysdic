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

from numbers import Integral, Real
from typing import Optional, Tuple, Union

import numpy
from numpy.typing import ArrayLike

from .._validation import _as_array, _as_batch_array, _validate_bool, _validate_integer, _validate_real


def compute_wave_speckle(
    points: ArrayLike,
    *,
    wavelength: Real = 0.25,
    n_waves: Integral = 40,
    gray_levels: Tuple[Real, Real] = (20.0, 235.0),
    seed: Optional[Integral] = 0,
    gradient: bool = False,
) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
    r"""
    Compute the gray levels of a synthetic speckle pattern at given points.

    The pattern is a sum of :math:`M` plane waves of the same wavelength
    :math:`\lambda`, random directions and random phases, saturated by a hyperbolic
    tangent to give dark and bright spots:

    .. math::

        S(\vec{x}) = \sum_{j=1}^{M} \cos\bigl(\vec{k}_j \cdot \vec{x} + \varphi_j\bigr),
        \qquad \|\vec{k}_j\| = \frac{2\pi}{\lambda}

    .. math::

        g(\vec{x}) = \frac{g_{min} + g_{max}}{2}
        + \frac{g_{max} - g_{min}}{2} \tanh\left(\frac{S(\vec{x})}{\sqrt{M}}\right)

    The division by :math:`\sqrt{M}` keeps the standard deviation of :math:`S` close to
    1 whatever the number of waves. The gray levels stay strictly between
    :math:`g_{min}` and :math:`g_{max}`.

    The pattern is a smooth function of the coordinates: evaluated at the reference
    coordinates of material points, it gives a texture attached to the material that
    can be rendered in any camera (synthetic images for DIC tests).

    With ``gradient=True``, the exact gradient of the gray levels is also returned (e.g.
    as a reference to check the interpolation of an image):

    .. math::

        \nabla g(\vec{x}) = -\frac{g_{max} - g_{min}}{2 \sqrt{M}}
        \left(1 - \tanh^2\left(\frac{S(\vec{x})}{\sqrt{M}}\right)\right)
        \sum_{j=1}^{M} \sin\bigl(\vec{k}_j \cdot \vec{x} + \varphi_j\bigr) \, \vec{k}_j

    .. note::

        The speckle spots have a size of about :math:`\lambda / 2`. For DIC, aim at 3 to 5
        pixels per spot in the images, i.e. a wavelength of about 6 to 10 pixels once
        projected: with points in material units seen with a magnification of :math:`m`
        pixels per unit, choose :math:`\lambda \approx 6 / m` to :math:`10 / m`. Smaller
        spots are aliased by the pixels (see the ``n_samples`` of
        :func:`pysdic.generate_synthetic_image`), larger spots reduce the spatial
        resolution of the measurement.

    .. note::

        The waves are drawn from ``seed`` at each call: the same ``seed``, ``n_waves``
        and point dimension :math:`D` always give the same pattern. Use ``seed=None``
        for a different pattern at each call.

    .. note::

        The directions of the waves are isotropic in :math:`\mathbb{R}^D`. For a
        surface lying in a plane :math:`z = c`, give only the in-plane coordinates
        (``points[..., :2]``): with 3D points, the waves with a component along
        :math:`z` have a longer apparent wavelength in the plane.

    Parameters
    ----------
    points : ArrayLike
        Coordinates of the points, an array of shape :math:`(\dots, D)` with
        :math:`D \in \{1, 2, 3\}`. The values must be finite.

    wavelength : Real, optional (default: ``0.25``)
        Wavelength :math:`\lambda` of the waves, in the unit of ``points``. It sets the
        size of the speckle spots (about :math:`\lambda / 2`). Must be finite and
        positive.

    n_waves : Integral, optional (default: ``40``)
        Number of waves :math:`M`, at least 1. More waves give a more irregular
        pattern.

    gray_levels : Tuple[Real, Real], optional (default: ``(20.0, 235.0)``)
        Bounds :math:`(g_{min}, g_{max})` of the gray levels, finite with
        :math:`g_{min} < g_{max}`.

    seed : Integral, optional (default: ``0``)
        Seed of the random generator (``numpy.random.default_rng``). If None, the
        pattern is different at each call.

    gradient : bool, optional (default: ``False``)
        Whether to also return the gradient of the gray levels with respect to the
        coordinates of the points.

    Returns
    -------
    gray_levels : numpy.ndarray
        The gray levels, a float array of shape :math:`(\dots)`.

    gradient : numpy.ndarray
        Only with ``gradient=True``: the gradient of the gray levels, a float array of
        shape :math:`(\dots, D)` (in gray levels per unit of ``points``).

    Raises
    ------
    TypeError
        If an input has a wrong type.

    ValueError
        If ``points`` has a wrong shape or non-finite values, or if a parameter has
        a wrong value.

    Examples
    --------
    Texture attached to the reference coordinates of the vertices of a plane mesh:

    .. code-block:: python

        import numpy
        from pysdic import compute_wave_speckle

        x, y = numpy.meshgrid(numpy.linspace(0, 1, 200), numpy.linspace(0, 1, 200))
        image = compute_wave_speckle(numpy.stack((x, y), axis=-1), wavelength=0.05)
        image.shape  # (200, 200)

        image, image_gradient = compute_wave_speckle(numpy.stack((x, y), axis=-1), wavelength=0.05, gradient=True)
        image_gradient.shape  # (200, 200, 2), derivatives along x and y

    """
    points = _as_batch_array(points, "points", (1, 2, 3), numpy.float64, finite=True)
    wavelength = _validate_real(wavelength, "wavelength", positive=True, finite=True)
    n_waves = _validate_integer(n_waves, "n_waves", min_value=1)
    if seed is not None:
        seed = _validate_integer(seed, "seed", min_value=0)
    gradient = _validate_bool(gradient, "gradient")
    g_min, g_max = (float(value) for value in _as_array(gray_levels, "gray_levels", numpy.float64, shape=(2,), finite=True))
    if g_min >= g_max:
        raise ValueError(
            f"Input 'gray_levels' must satisfy g_min < g_max. Got {(g_min, g_max)}."
        )

    # Random directions (isotropic) of norm 2 pi / wavelength, random phases
    rng = numpy.random.default_rng(seed)
    wave_vectors = rng.normal(size=(n_waves, points.shape[-1]))
    norms = numpy.linalg.norm(wave_vectors, axis=1, keepdims=True)
    while numpy.any(norms == 0.0):  # Null direction (only possible for D = 1): redraw it
        zero = norms[:, 0] == 0.0
        wave_vectors[zero] = rng.normal(size=(int(zero.sum()), points.shape[-1]))
        norms = numpy.linalg.norm(wave_vectors, axis=1, keepdims=True)
    wave_vectors *= 2.0 * numpy.pi / wavelength / norms
    phases = rng.uniform(0.0, 2.0 * numpy.pi, n_waves)

    arguments = points @ wave_vectors.T + phases  # (..., M)
    saturated = numpy.tanh(numpy.cos(arguments).sum(axis=-1) / numpy.sqrt(n_waves))  # (...)
    gray_levels = 0.5 * (g_min + g_max) + 0.5 * (g_max - g_min) * saturated
    if not gradient:
        return gray_levels
    scale = -0.5 * (g_max - g_min) / numpy.sqrt(n_waves) * (1.0 - saturated**2)  # (...)
    return gray_levels, scale[..., numpy.newaxis] * (numpy.sin(arguments) @ wave_vectors)  # (..., D)