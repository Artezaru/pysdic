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
from numbers import Integral, Real
from typing import Any

import cv2

from .._optionals import import_optional
from .._validation import _validate_bool, _validate_integer, _validate_real


@dataclass(frozen=True)
class DisflowConfig:
    r"""
    Parameters of the Dense Inverse Search (DIS) optical flow of OpenCV
    (``cv2.DISOpticalFlow``).

    The DIS algorithm estimates the flow from coarse to fine on a Gaussian pyramid of
    the images. At each scale:

    1. **Patch inverse search**: the image is covered by square patches of
       ``patch_size`` pixels, spaced by ``patch_stride`` pixels. The displacement of
       each patch is found by an inverse compositional gradient descent
       (``gradient_descent_iterations``), optionally improved by the displacements of
       the neighbor patches (``use_spatial_propagation``).
    2. **Densification**: the dense flow at each pixel is a weighted average of the
       displacements of the patches that contain it.
    3. **Variational refinement**: the dense flow is smoothed by minimizing an energy
       made of a data term (``variational_refinement_delta``), a gradient term
       (``variational_refinement_gamma``) and a smoothness term
       (``variational_refinement_alpha``), with ``variational_refinement_iterations``
       fixed point iterations.

    The configuration is immutable and its values are validated at creation. Use
    :func:`dataclasses.replace` to derive a new configuration:

    .. code-block:: python

        from dataclasses import replace
        from pysdic import DisflowConfig

        config = DisflowConfig(patch_size=20, patch_stride=5)
        smooth_config = replace(config, variational_refinement_alpha=5.0)

    .. note::

        The default values are the ones of :func:`pysdic.compute_optical_flow`. They
        differ from the OpenCV presets: the patches are larger (50 pixels instead of 8)
        and the gradient descent longer (500 iterations), which suits the speckle
        patterns of DIC better than natural images.

    Parameters
    ----------
    finest_scale : Integral, optional (default: ``0``)
        Level of the Gaussian pyramid on which the computation stops. ``0`` is the
        original resolution, ``1`` half the resolution, etc. For a value
        :math:`s > 0` the flow is computed on images downsampled by :math:`2^s` and
        then upscaled bilinearly: faster but less accurate. Must be non-negative.

    patch_size : Integral, optional (default: ``50``)
        Size (in pixels) of the square patches matched between the images, at each
        scale. Larger patches are more robust (more texture in each patch) but blur
        the flow where it varies quickly. The patches must fit in the images at the
        finest scale (OpenCV raises an error otherwise). Must be at least 1.

    patch_stride : Integral, optional (default: ``10``)
        Distance (in pixels) between the corners of two neighbor patches. A smaller
        stride gives more overlapping patches: a better flow but a slower
        computation. Must be between 1 and ``patch_size`` (a larger stride leaves
        pixels outside every patch and crashes OpenCV).

    gradient_descent_iterations : Integral, optional (default: ``500``)
        Maximum number of iterations of the gradient descent that finds the
        displacement of each patch. More iterations can improve the flow for large or
        slow-converging displacements. ``0`` disables the search (the patches keep
        their initial displacements). Must be non-negative.

    use_spatial_propagation : bool, optional (default: ``True``)
        If True, the displacement of a patch is replaced by the one of a neighbor
        patch when it matches better. This helps to recover from the errors of the
        coarse scales; disabling it can give a slightly smoother flow.

    use_mean_normalization : bool, optional (default: ``True``)
        If True, the mean gray level of each patch is subtracted before comparing the
        patches, which makes the matching robust to illumination changes between the
        images. Disable it only if the illumination is constant.

    variational_refinement_iterations : Integral, optional (default: ``10``)
        Number of fixed point iterations of the variational refinement at each scale.
        ``0`` disables the variational refinement. More iterations give a smoother
        flow. Must be non-negative.

    variational_refinement_alpha : Real, optional (default: ``1.0``)
        Weight of the smoothness term of the variational refinement: a larger value
        gives a smoother flow. Must be finite and non-negative (a negative weight
        makes the refinement diverge).

    variational_refinement_delta : Real, optional (default: ``1.0``)
        Weight of the data term (gray level constancy :math:`I_2(x + F) = I_1(x)`) of
        the variational refinement. Must be finite and non-negative.

    variational_refinement_gamma : Real, optional (default: ``1.0``)
        Weight of the gradient term (gradient constancy
        :math:`\nabla I_2(x + F) = \nabla I_1(x)`) of the variational refinement,
        which is robust to illumination changes. Must be finite and non-negative.

    variational_refinement_epsilon : Real, optional (default: ``0.02``)
        Shift :math:`\varepsilon` of the robust (Charbonnier) penalty
        :math:`\sqrt{r^2 + \varepsilon^2}` used by the variational refinement: a
        smaller value is more robust to outliers but less smooth. Must be finite and
        strictly positive (``0`` crashes OpenCV).

    Raises
    ------
    TypeError
        If a parameter has a wrong type (booleans are rejected for the numbers).

    ValueError
        If a parameter has a wrong value.

    """

    finest_scale: Integral = 0
    patch_size: Integral = 50
    patch_stride: Integral = 10
    gradient_descent_iterations: Integral = 500
    use_spatial_propagation: bool = True
    use_mean_normalization: bool = True
    variational_refinement_iterations: Integral = 10
    variational_refinement_alpha: Real = 1.0
    variational_refinement_delta: Real = 1.0
    variational_refinement_gamma: Real = 1.0
    variational_refinement_epsilon: Real = 0.02

    def __post_init__(self) -> None:
        # Validate and normalize the values to python types (the dataclass is frozen)
        values = {
            "finest_scale": _validate_integer(self.finest_scale, "finest_scale", min_value=0),
            "patch_size": _validate_integer(self.patch_size, "patch_size", min_value=1),
            "patch_stride": _validate_integer(self.patch_stride, "patch_stride", min_value=1),
            "gradient_descent_iterations": _validate_integer(
                self.gradient_descent_iterations, "gradient_descent_iterations", min_value=0
            ),
            "use_spatial_propagation": _validate_bool(
                self.use_spatial_propagation, "use_spatial_propagation"
            ),
            "use_mean_normalization": _validate_bool(
                self.use_mean_normalization, "use_mean_normalization"
            ),
            "variational_refinement_iterations": _validate_integer(
                self.variational_refinement_iterations,
                "variational_refinement_iterations",
                min_value=0,
            ),
            "variational_refinement_alpha": _validate_real(
                self.variational_refinement_alpha,
                "variational_refinement_alpha",
                non_negative=True,
                finite=True,
            ),
            "variational_refinement_delta": _validate_real(
                self.variational_refinement_delta,
                "variational_refinement_delta",
                non_negative=True,
                finite=True,
            ),
            "variational_refinement_gamma": _validate_real(
                self.variational_refinement_gamma,
                "variational_refinement_gamma",
                non_negative=True,
                finite=True,
            ),
            "variational_refinement_epsilon": _validate_real(
                self.variational_refinement_epsilon,
                "variational_refinement_epsilon",
                positive=True,
                finite=True,
            ),
        }
        if values["patch_stride"] > values["patch_size"]:
            raise ValueError(
                f"Input 'patch_stride' must be lower than or equal to 'patch_size'. "
                f"Got patch_stride={values['patch_stride']} and patch_size={values['patch_size']}."
            )
        for name, value in values.items():
            object.__setattr__(self, name, value)

    def create_dis_optical_flow(self) -> Any:
        r"""
        Create a ``cv2.DISOpticalFlow`` object with this configuration.

        Returns
        -------
        cv2.DISOpticalFlow
            A new OpenCV DIS optical flow object, ready for ``calc``.

        Raises
        ------
        ImportError
            If ``cv2`` (``opencv-python``) is not installed.

        """
        # The preset is only a starting point: all its parameters are overwritten below
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        dis.setFinestScale(self.finest_scale)
        dis.setPatchSize(self.patch_size)
        dis.setPatchStride(self.patch_stride)
        dis.setGradientDescentIterations(self.gradient_descent_iterations)
        dis.setUseSpatialPropagation(self.use_spatial_propagation)
        dis.setUseMeanNormalization(self.use_mean_normalization)
        dis.setVariationalRefinementIterations(self.variational_refinement_iterations)
        dis.setVariationalRefinementAlpha(self.variational_refinement_alpha)
        dis.setVariationalRefinementDelta(self.variational_refinement_delta)
        dis.setVariationalRefinementGamma(self.variational_refinement_gamma)
        dis.setVariationalRefinementEpsilon(self.variational_refinement_epsilon)
        return dis