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
from typing import Optional, Tuple

import numpy
from numpy.typing import ArrayLike
from scipy.ndimage import gaussian_filter

from .._validation import _as_array, _validate_integer, _validate_real


# =============================================================================
# Internal helpers
# =============================================================================
def _as_finite_image(image: ArrayLike, feature: str) -> numpy.ndarray:
    # Float64 copy of an image with finite values
    image = numpy.array(image, dtype=numpy.float64)  # always a copy
    if not numpy.all(numpy.isfinite(image)):
        raise ValueError(f"The image must have finite values for '{feature}'.")
    return image


def _rng(seed: Optional[Integral]) -> numpy.random.Generator:
    # Random generator from a non-negative seed (None: different at each call)
    if seed is not None:
        seed = _validate_integer(seed, "seed", min_value=0)
    return numpy.random.default_rng(seed)


# =============================================================================
# Noises
# =============================================================================
def add_gaussian_noise(
    image: ArrayLike,
    *,
    std: Real,
    seed: Optional[Integral] = None,
) -> numpy.ndarray:
    r"""
    Add a white Gaussian noise to an image (read noise of a sensor).

    .. math::

        g' = g + \sigma \, \mathcal{N}(0, 1)

    The noise is drawn independently for each value of the image (each pixel and channel).

    .. seealso::

        The functions of the simulation of an acquisition, in the usual order:
        :func:`add_gaussian_blur`, :func:`add_poisson_noise`, :func:`add_gaussian_noise`,
        :func:`add_salt_and_pepper_noise` and :func:`quantize_image`.

    Parameters
    ----------
    image : ArrayLike
        The image, an array of any shape (e.g. :math:`(H, W)` or :math:`(H, W, C)`) with
        finite values.

    std : Real
        The standard deviation :math:`\sigma \geq 0` of the noise, in gray levels.

    seed : Optional[Integral], optional (default: ``None``)
        The seed of the random generator (``numpy.random.default_rng``), non-negative. If
        None, the noise is different at each call.

    Returns
    -------
    numpy.ndarray
        The noisy image, a float64 array with the shape of ``image``.

    Raises
    ------
    ValueError
        If the image has non-finite values or an input has a wrong value.

    """
    image = _as_finite_image(image, "add_gaussian_noise")
    std = _validate_real(std, "std", non_negative=True, finite=True)
    rng = _rng(seed)
    if std > 0:
        image += std * rng.standard_normal(image.shape)
    return image


def add_poisson_noise(
    image: ArrayLike,
    *,
    gain: Real,
    seed: Optional[Integral] = None,
) -> numpy.ndarray:
    r"""
    Add a Poisson noise to an image (photon noise of a sensor): the variance of the noise
    grows with the gray level.

    The gray levels are converted to a number of electrons with the gain :math:`k` (gray
    levels per electron), a Poisson count is drawn, and converted back:

    .. math::

        g' = k \, \mathcal{P}\left(\frac{g}{k}\right),
        \qquad \mathbb{E}[g'] = g, \qquad \text{Var}[g'] = k \, g

    A smaller gain gives more electrons per gray level, hence a smaller relative noise.

    .. seealso::

        :func:`add_gaussian_noise` for the usual order of the simulation of an acquisition.

    Parameters
    ----------
    image : ArrayLike
        The image, an array of any shape with finite and non-negative values.

    gain : Real
        The gain :math:`k > 0` of the sensor, in gray levels per electron.

    seed : Optional[Integral], optional (default: ``None``)
        The seed of the random generator (``numpy.random.default_rng``), non-negative. If
        None, the noise is different at each call.

    Returns
    -------
    numpy.ndarray
        The noisy image, a float64 array with the shape of ``image`` (multiples of
        :math:`k`, not rounded).

    Raises
    ------
    ValueError
        If the image has non-finite or negative values, or an input has a wrong value.

    """
    image = _as_finite_image(image, "add_poisson_noise")
    if numpy.any(image < 0):
        raise ValueError(f"The image must have non-negative values for a Poisson noise. Got a minimum of {image.min()}.")
    gain = _validate_real(gain, "gain", positive=True, finite=True)
    return gain * _rng(seed).poisson(image / gain).astype(numpy.float64)


def add_salt_and_pepper_noise(
    image: ArrayLike,
    *,
    fraction: Real,
    values: Tuple[Real, Real] = (0.0, 255.0),
    seed: Optional[Integral] = None,
) -> numpy.ndarray:
    r"""
    Add a salt and pepper noise to an image (dead and hot pixels of a sensor).

    Each value of the image (each pixel and channel) is replaced, with the probability
    ``fraction``, by the low value (pepper) or the high value (salt), with the same
    probability.

    .. seealso::

        :func:`add_gaussian_noise` for the usual order of the simulation of an acquisition.

    Parameters
    ----------
    image : ArrayLike
        The image, an array of any shape with finite values.

    fraction : Real
        The probability :math:`p \in [0, 1]` that a value is replaced.

    values : Tuple[Real, Real], optional (default: ``(0.0, 255.0)``)
        The values (pepper, salt), finite. For example ``(0, 2**b - 1)`` for an image of
        :math:`b` bits.

    seed : Optional[Integral], optional (default: ``None``)
        The seed of the random generator (``numpy.random.default_rng``), non-negative. If
        None, the noise is different at each call.

    Returns
    -------
    numpy.ndarray
        The noisy image, a float64 array with the shape of ``image``.

    Raises
    ------
    ValueError
        If the image has non-finite values or an input has a wrong value.

    """
    image = _as_finite_image(image, "add_salt_and_pepper_noise")
    fraction = _validate_real(fraction, "fraction", non_negative=True, finite=True)
    if fraction > 1.0:
        raise ValueError(f"Input 'fraction' must be between 0.0 and 1.0. Got {fraction}.")
    pepper, salt = _as_array(values, "values", numpy.float64, shape=(2,), finite=True)
    rng = _rng(seed)
    replaced = rng.random(image.shape) < fraction
    is_salt = rng.random(image.shape) < 0.5
    image[replaced & is_salt] = salt
    image[replaced & ~is_salt] = pepper
    return image


# =============================================================================
# Blur
# =============================================================================
def add_gaussian_blur(
    image: ArrayLike,
    *,
    sigma: Real,
) -> numpy.ndarray:
    r"""
    Blur an image with a Gaussian kernel (defocus or optical blur).

    The image is convolved with a Gaussian kernel of standard deviation :math:`\sigma`
    along the rows and the columns (not along the channels), with
    ``scipy.ndimage.gaussian_filter``. The edges are extended with their nearest values.

    .. seealso::

        :func:`add_gaussian_noise` for the usual order of the simulation of an acquisition.

    Parameters
    ----------
    image : ArrayLike
        The image, an array of shape :math:`(H, W)` or :math:`(H, W, C)` with finite values.

    sigma : Real
        The standard deviation :math:`\sigma \geq 0` of the kernel, in pixels.

    Returns
    -------
    numpy.ndarray
        The blurred image, a float64 array with the shape of ``image``.

    Raises
    ------
    ValueError
        If the image has a wrong shape or non-finite values, or an input has a wrong value.

    """
    image = _as_finite_image(image, "add_gaussian_blur")
    if image.ndim not in (2, 3):
        raise ValueError(f"The image must have shape (H, W) or (H, W, C). Got shape {image.shape}.")
    sigma = _validate_real(sigma, "sigma", non_negative=True, finite=True)
    if sigma == 0:
        return image
    return gaussian_filter(image, sigma=(sigma, sigma, 0.0)[: image.ndim], mode="nearest")


# =============================================================================
# Quantization
# =============================================================================
def quantize_image(
    image: ArrayLike,
    *,
    bit_depth: Integral,
) -> numpy.ndarray:
    r"""
    Quantize the gray levels of an image (analog-to-digital conversion of a sensor).

    .. math::

        g' = \text{clip}\left(\text{round}(g), \, 0, \, 2^{b} - 1\right)

    The values are rounded to the nearest integer and clipped to :math:`[0, 2^b - 1]`
    (saturation of the sensor). The image is returned with an unsigned integer dtype:
    ``uint8`` for :math:`b \leq 8`, ``uint16`` otherwise.

    .. seealso::

        :func:`add_gaussian_noise` for the usual order of the simulation of an acquisition
        (the quantization is the last step).

    Parameters
    ----------
    image : ArrayLike
        The image, an array of any shape with finite values.

    bit_depth : Integral
        The number of bits :math:`b \in [1, 16]` of the quantization.

    Returns
    -------
    numpy.ndarray
        The quantized image, a ``uint8`` or ``uint16`` array with the shape of ``image``.

    Raises
    ------
    ValueError
        If the image has non-finite values or ``bit_depth`` is out of :math:`[1, 16]`.

    Examples
    --------
    Simulation of the acquisition of a synthetic image (see
    :func:`pysdic.generate_synthetic_image`):

    .. code-block:: python

        from pysdic import add_gaussian_blur, add_gaussian_noise, add_poisson_noise, quantize_image

        acquired = add_gaussian_blur(image[..., 0], sigma=0.5)
        acquired = add_poisson_noise(acquired, gain=0.1, seed=0)
        acquired = add_gaussian_noise(acquired, std=1.0, seed=1)
        acquired = quantize_image(acquired, bit_depth=8)  # uint8 (H, W)

    """
    image = _as_finite_image(image, "quantize_image")
    bit_depth = _validate_integer(bit_depth, "bit_depth", min_value=1)
    if bit_depth > 16:
        raise ValueError(f"Input 'bit_depth' must be between 1 and 16. Got {bit_depth}.")
    dtype = numpy.uint8 if bit_depth <= 8 else numpy.uint16
    return numpy.clip(numpy.round(image), 0, 2**bit_depth - 1).astype(dtype)