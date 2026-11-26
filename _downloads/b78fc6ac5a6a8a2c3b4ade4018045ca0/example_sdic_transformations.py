r"""
.. _example_sdic_transformations:

Optimizing the cameras with the SDIC
====================================

.. contents:: Table of Contents
   :local:
   :depth: 1
   :backlinks: top

The unknowns of a :class:`pysdic.SDICCorrelationTerm` (a :class:`pysdic.SDICUnknowns`
created by the user) are displacements of frames and/or parameters of transformations of
the cameras. This example calibrates the extrinsic of the second
camera of a stereo rig on the reference frame, then measures a displacement while refining
it.

.. important::

    Requires the optional package ``open3d`` (ray casting of the 3D surface).

.. seealso::

    :ref:`example_sdic_displacement` for the measurement of a displacement with fixed cameras.

"""

# %%
# Synthetic stereo images
# -----------------------
#
# The same setup as in :ref:`example_sdic_displacement`: two cameras look at a speckled
# plate, which moves with an affine displacement field between the frames 0 and 1 (here
# without change of lighting).

import numpy
from pysdic import (
    Camera, Image, PerspectiveExtrinsic, PinholeIntrinsic, add_gaussian_noise, compute_projected_roi,
    compute_wave_speckle, create_3d_quadrangle_4_heightmap, generate_synthetic_image, quantize_image,
)

height, width = 180, 240
cameras = [
    Camera(
        height, width,
        extrinsic=PerspectiveExtrinsic.from_rt(rvec=[numpy.pi, side * 0.15, 0.0], tvec=[side * 0.6, 0.0, 5.0]),
        intrinsic=PinholeIntrinsic.from_matrix([[300.0, 0.0, 120.0], [0.0, 300.0, 90.0], [0.0, 0.0, 1.0]]),
    )
    for side in (1, -1)
]

plate = create_3d_quadrangle_4_heightmap(x_bounds=(-2.5, 2.5), y_bounds=(-2.0, 2.0), n_x=41, n_y=33)
texture = compute_wave_speckle(
    numpy.stack(numpy.meshgrid(numpy.linspace(-2.5, 2.5, 500), numpy.linspace(2.0, -2.0, 400)), axis=-1), wavelength=0.15
)


def true_displacement(points: numpy.ndarray) -> numpy.ndarray:
    x, y = points[:, 0], points[:, 1]
    return numpy.stack((0.04 + 0.02 * y, -0.03 + 0.01 * x, 0.05 + 0.02 * x), axis=1)


mesh = create_3d_quadrangle_4_heightmap(x_bounds=(-1.0, 1.0), y_bounds=(-0.6, 0.6), n_x=7, n_y=5)
truth = true_displacement(mesh.coordinates)


def acquire(camera: Camera, frame: int, seed: int) -> Image:
    rendered = generate_synthetic_image(camera, plate, texture, displacement=frame * true_displacement(plate.coordinates), n_samples=2)
    data = quantize_image(add_gaussian_noise(rendered[..., 0], std=1.0, seed=seed), bit_depth=8)
    return Image.from_array(data, roi=compute_projected_roi(mesh, camera, warp_by=frame * truth, margin=10))


# %%
# The analysis and its transformations
# ------------------------------------
#
# A transformation of the cameras is identified by ``(kind, index)``, its index in the
# structure of its kind: here each camera has its own extrinsic (``("extrinsic", 0)`` and
# ``("extrinsic", 1)``), shared by its views of both frames. Changing its parameters with
# :meth:`pysdic.SDICAnalysis.set_transformation_parameters` changes all these views.
#
# The images are acquired with the true cameras, then the extrinsic of the second camera is
# perturbed, as an inaccurate calibration would be.

from pysdic import create_bicamera_analysis

images = [[acquire(camera, frame, seed=2 * frame + k) for frame in range(2)] for k, camera in enumerate(cameras)]
analysis = create_bicamera_analysis(mesh, cameras[0], cameras[1], images[0], images[1])
analysis.set_zero_displacement(0)
analysis.display_structure("extrinsic")

true_extrinsic = analysis.get_transformation_parameters("extrinsic", 1)
perturbed_extrinsic = true_extrinsic + numpy.random.default_rng(0).normal(scale=0.005, size=true_extrinsic.size)
analysis.set_transformation_parameters("extrinsic", 1, perturbed_extrinsic)
print(f"Perturbation of the extrinsic 1: {numpy.abs(perturbed_extrinsic - true_extrinsic).max():.1e}")

# %%
# The material points are a regular grid of :math:`6 \times 6` points per element, and the
# gray levels are compared with a ZNSSD correction.

from pysdic import IntegrationPoints, Pairing, SDICCorrelationTerm, SDICUnknowns

grid = numpy.linspace(-0.9, 0.9, 6)
natural = numpy.stack(numpy.meshgrid(grid, grid), axis=-1).reshape(-1, 2)
integration_points = IntegrationPoints(
    numpy.tile(natural, (mesh.n_elements, 1)), numpy.repeat(numpy.arange(mesh.n_elements), natural.shape[0])
)
gl_correction = {"mode": "ZNSSD", "scale": 255.0}


def rms_residuals(correlation: SDICCorrelationTerm) -> str:
    # RMS of the residuals of each pairing of the term, for the current state of the analysis
    return ", ".join(f"{numpy.sqrt(numpy.nanmean(r.residuals**2)):.1f}" for r in correlation.compute_residuals())


# %%
# Calibration on the reference frame
# ----------------------------------
#
# With the known shape of the reference frame (zero displacement), the stereo pairing of
# the frame 0 determines the extrinsic of the second camera relatively to the first one.
# The unknowns are only this transformation (``SDICUnknowns([], [("extrinsic", 1)])``).
# With few unknowns, a dense Jacobian (``sparse=False``) is faster.

import pysolvegn

unknowns = SDICUnknowns([], [("extrinsic", 1)])
calibration = SDICCorrelationTerm(
    analysis, unknowns, integration_points, [Pairing.stereo(0, 0, 1)], gl_correction=gl_correction, sparse=False
)
print(f"RMS residual before: {rms_residuals(calibration)} gray levels")

result = pysolvegn.solve(terms=[calibration.term], p0=analysis.pack_unknowns(unknowns), max_iteration=30)
analysis.apply_unknowns(unknowns, result.parameters)  # Updates all the views of the second camera
print(f"RMS residual after:  {rms_residuals(calibration)} gray levels ({result.n_iterations} iterations)")
print(f"Error on the extrinsic 1: {numpy.abs(analysis.get_transformation_parameters('extrinsic', 1) - true_extrinsic).max():.1e}")

# %%
# Joint optimization
# ------------------
#
# The displacement of the frame 1 and the extrinsic of the second camera are optimized
# together, from the perturbed extrinsic. The vector of unknowns is
# :math:`\vec{p} = [\vec{p}_U, \vec{p}_T]` (see :class:`pysdic.SDICUnknowns`):
# :meth:`pysdic.SDICAnalysis.unpack_unknowns` splits it into the displacements and the parameters of
# the transformations. New unknowns need a new term (a term cannot change its unknowns).
#
# The stereo pairings of both frames determine the extrinsic, and the temporal pairings the
# displacement.
#
# .. warning::
#
#     Displacements and transformations are not always separable: a rigid motion of the
#     surface is equivalent to a rigid motion of all the cameras. Here the extrinsic of the
#     first camera is fixed, which removes this ambiguity.

analysis.set_transformation_parameters("extrinsic", 1, perturbed_extrinsic)
analysis.set_displacement(1, 0.7 * truth)

pairings = [Pairing.temporal(0, 0, 1), Pairing.temporal(1, 0, 1), Pairing.stereo(0, 0, 1), Pairing.stereo(1, 0, 1)]
unknowns = SDICUnknowns([1], [("extrinsic", 1)])
joint = SDICCorrelationTerm(analysis, unknowns, integration_points, pairings, gl_correction=gl_correction, use_cache=True)
print(unknowns)

result = pysolvegn.solve(terms=[joint.term], p0=analysis.pack_unknowns(unknowns), max_iteration=30)
displacements, transformations = analysis.unpack_unknowns(unknowns, result.parameters)
print(f"Error on the displacement of the frame 1: {numpy.abs(displacements[1] - truth).max():.1e}")
print(f"Error on the extrinsic 1: {numpy.abs(transformations[('extrinsic', 1)] - true_extrinsic).max():.1e}")

analysis.apply_unknowns(unknowns, result.parameters)  # Writes both in the analysis
print(f"RMS residuals after: {rms_residuals(joint)} gray levels")