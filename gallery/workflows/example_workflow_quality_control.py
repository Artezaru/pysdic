r"""
.. _example_workflows_quality_control:

Controlling the quality of a measure
====================================

.. contents:: Table of Contents
   :local:
   :depth: 1
   :backlinks: top

After a measure, :mod:`pysdic.workflows` gives two controls:

- :func:`pysdic.workflows.compute_elements_residual_rms`: the RMS of the residuals per element
  and per pairing, and the badly correlated elements (reflections, occlusions);
- :func:`pysdic.workflows.compute_uncertainty`: the covariance of the displacements from the
  Jacobian of the term and the noise of the images.

.. important::

    Requires the optional package ``open3d`` (ray casting of the 3D surface).

"""

# %%
# A measure with a reflection
# ---------------------------
#
# The plate of :ref:`example_sdic_displacement` moves between the frames 0 and 1, with a noise
# of 1 gray level. In the image of the first camera at the frame 1, a reflection saturates a
# disk of 10 pixels around the point :math:`(0.5, 0.25, 0)` of the plate.

import numpy
from pysdic import (
    Camera, Image, PerspectiveExtrinsic, PinholeIntrinsic, add_gaussian_noise, compute_projected_roi,
    compute_wave_speckle, create_3d_quadrangle_4_heightmap, create_bicamera_analysis, generate_synthetic_image,
    quantize_image,
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
x, y = numpy.linspace(-2.5, 2.5, 500), numpy.linspace(2.0, -2.0, 400)
texture = compute_wave_speckle(numpy.stack(numpy.meshgrid(x, y), axis=-1), wavelength=0.15)


def true_displacement(points: numpy.ndarray) -> numpy.ndarray:
    x, y = points[:, 0], points[:, 1]
    return numpy.stack((0.04 + 0.02 * y, -0.03 + 0.01 * x, 0.05 + 0.02 * x), axis=1)


mesh = create_3d_quadrangle_4_heightmap(x_bounds=(-1.0, 1.0), y_bounds=(-0.6, 0.6), n_x=7, n_y=5)
truth = true_displacement(mesh.coordinates)
reflection = numpy.array([[0.5, 0.25, 0.0]])


def acquire(camera: Camera, frame: int, seed: int, reflect: bool = False) -> Image:
    rendered = generate_synthetic_image(camera, plate, texture, displacement=frame * true_displacement(plate.coordinates), n_samples=2)
    data = add_gaussian_noise(rendered[..., 0], std=1.0, seed=seed)
    if reflect:
        cx, cy = camera.project(reflection + true_displacement(reflection)).image_points[0]
        rows, columns = numpy.mgrid[0:height, 0:width]
        data[(columns - cx) ** 2 + (rows - cy) ** 2 <= 10.0**2] = 255.0
    data = quantize_image(data, bit_depth=8)
    return Image.from_array(data, roi=compute_projected_roi(mesh, camera, warp_by=frame * truth, margin=10))


images = [
    [acquire(camera, frame, seed=2 * frame + k, reflect=(k == 0 and frame == 1)) for frame in range(2)]
    for k, camera in enumerate(cameras)
]
analysis = create_bicamera_analysis(mesh, cameras[0], cameras[1], images[0], images[1])
analysis.set_zero_displacement(0)

# %%
# The displacement of the frame 1 is measured with the temporal pairings of both cameras and
# the stereo pairing of the frame 1, without gray level correction (same lighting).

import pysolvegn
from pysdic import Pairing, SDICCorrelationTerm, SDICUnknowns
from pysdic.workflows import create_grid_integration_points

pairings = [Pairing.temporal(0, 0, 1), Pairing.temporal(1, 0, 1), Pairing.stereo(1, 0, 1)]
analysis.set_displacement(1, 0.7 * truth)
unknowns = SDICUnknowns([1], [])


def measure(correlation: SDICCorrelationTerm) -> SDICCorrelationTerm:
    analysis.set_displacement(1, 0.7 * truth)  # Same initial guess for each measure
    result = pysolvegn.solve(terms=[correlation.term], p0=analysis.pack_unknowns(unknowns), max_iteration=30)
    analysis.apply_unknowns(unknowns, result.parameters)
    print(f"Maximum error on the displacement: {numpy.abs(analysis.get_displacement(1) - truth).max():.2e}")
    return correlation


correlation = measure(SDICCorrelationTerm(analysis, unknowns, create_grid_integration_points(mesh, 6), pairings))

# %%
# Badly correlated elements
# -------------------------
#
# The RMS of the residuals of each element is compared with the other elements of the
# pairing (median and robust spread): the elements of the reflection stand out in the two
# pairings that use the view ``(1, 0)``.

import matplotlib.pyplot as plt
from pysdic import ImagePlotter
from pysdic.workflows import compute_elements_residual_rms

quality = compute_elements_residual_rms(analysis, correlation.integration_points, correlation.compute_residuals())
for pairing, rms, outliers in zip(quality.pairings, quality.pairing_rms, quality.outliers):
    print(f"{pairing}: RMS {rms:.1f} gray levels, outliers {numpy.flatnonzero(outliers)}")
print(f"Badly correlated elements: {quality.bad_elements}")

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, pairing, rms in zip(axes, quality.pairings, quality.rms):
    plotter = ImagePlotter(analysis.get_view(*pairing.view_2), ax=ax)
    plotter.add_image()
    plotter.add_mesh(
        mesh, warp_by=analysis.get_displacement(1), element_property=rms, property_cmap="viridis",
        face_opacity=0.7, show_vertices=False, show_bar=True,
    )
    plotter.set_title(f"RMS per element, {pairing}")
plt.show()

# %%
# The integration points of the bad elements are removed and the frame is measured again,
# with a new term on the same unknowns (:meth:`pysdic.SDICCorrelationTerm.with_integration_points`).

good_elements = numpy.setdiff1d(numpy.arange(mesh.n_elements), quality.bad_elements)
correlation = measure(correlation.with_integration_points(create_grid_integration_points(mesh, 6, elements=good_elements)))

# %%
# Uncertainty of the displacements
# --------------------------------
#
# The covariance :math:`s^2 H^{-1} (J^T W J) H^{-1}` of the displacement of each vertex comes
# from the Jacobian of the term at convergence. The variance :math:`s^2` of the residuals is
# estimated from the residuals (default), or given by the noise of the images
# (:math:`s^2 = 2 \sigma^2`, here :math:`\sigma \approx 1` gray level with the quantization):
# both agree when the residuals only come from the noise.

from pysdic.workflows import compute_uncertainty

estimated = compute_uncertainty(correlation)
from_noise = compute_uncertainty(correlation, image_noise_std=1.0)
print(f"Residual variance: {estimated.residual_variance:.2f} (residuals), {from_noise.residual_variance:.2f} (noise)")

std = estimated.standard_deviations[1]  # (N_v, 3)
error = numpy.abs(analysis.get_displacement(1) - truth)
for c, name in enumerate("XYZ"):
    print(f"{name}: median standard deviation {numpy.nanmedian(std[:, c]):.1e}, RMS error {numpy.sqrt(numpy.mean(error[:, c]**2)):.1e}")

# %%
# The standard deviation of :math:`U_Z`, interpolated on the faces from the vertices: the out-of-plane component is the least precise (it
# only comes from the stereo angle), and the vertices on the border of the mesh, supported by
# fewer points, are less precise than the inner ones.

plotter = ImagePlotter(analysis.get_view(1, 1))
plotter.add_image()
plotter.add_mesh(
    mesh, warp_by=analysis.get_displacement(1), vertex_property=std[:, 2], property_cmap="magma",
    face_opacity=0.8, show_vertices=False, show_bar=True,
)
plotter.set_title("Standard deviation of $U_Z$")
plotter.show()