r"""
.. _example_workflows_displacement_tools:

Interpolating and filtering displacements
=========================================

.. contents:: Table of Contents
   :local:
   :depth: 1
   :backlinks: top

Two tools on the displacements of an analysis:

- :func:`pysdic.workflows.compute_interpolated_displacements` fills the frames that are not
  measured (no image, broken calibration) by interpolation in time;
- :func:`pysdic.workflows.compute_median_filtered_displacements` removes the isolated outliers
  of a frame with a median over the neighborhood of each vertex.

Both return the new displacements without modifying the analysis. This example needs no image:
the analysis has no view, only measured displacements.

"""

# %%
# Measured displacements
# ----------------------
#
# A plate vibrates with :math:`U_Z(X, t) = A \sin(2 \pi t / T) (1 - X^2)`. It is "measured"
# at 25 frames, with a noise of 0.002 and an outlier at the frame 12. The frames 8 to 10 and
# 18 are missing (no displacement).

import numpy
from pysdic import SDICAnalysis, create_3d_quadrangle_4_heightmap

n_x, n_y = 21, 11  # Vertex (i_x, i_y) = mesh.coordinates[i_y * n_x + i_x]
mesh = create_3d_quadrangle_4_heightmap(x_bounds=(-1.0, 1.0), y_bounds=(-0.5, 0.5), n_x=n_x, n_y=n_y)
times = numpy.linspace(0.0, 2.4, 25)  # Period T = 1


def true_displacement(t: float) -> numpy.ndarray:
    displacement = numpy.zeros((mesh.n_vertices, 3))
    displacement[:, 2] = 0.1 * numpy.sin(2.0 * numpy.pi * t) * (1.0 - mesh.coordinates[:, 0] ** 2)
    return displacement


rng = numpy.random.default_rng(0)
missing = [8, 9, 10, 18]
measured = [frame for frame in range(times.size) if frame not in missing]
displacements = {frame: true_displacement(times[frame]) + rng.normal(scale=0.002, size=(mesh.n_vertices, 3)) for frame in measured}
outlier = 5 * n_x + 7  # A badly measured vertex (e.g. on a reflection)
displacements[12][outlier, 2] += 0.08

analysis = SDICAnalysis(mesh, [[] for _ in times], times=times, displacements=displacements)

# %%
# Interpolation of the missing frames
# -----------------------------------
#
# The measured frames give the displacements at the times of the missing ones, linearly or
# with a cubic spline. Here, the missing block of three frames is in a strongly curved part of
# the vibration: the spline follows it, the linear interpolation cuts it.

from pysdic.workflows import compute_interpolated_displacements

linear = compute_interpolated_displacements(analysis, missing, measured)
cubic = compute_interpolated_displacements(analysis, missing, measured, method="cubic")

center = 5 * n_x + 10  # Vertex at X = 0
for frame in missing:
    print(f"Frame {frame}: true {true_displacement(times[frame])[center, 2]:+.4f}, "
          f"linear {linear[frame][center, 2]:+.4f}, cubic {cubic[frame][center, 2]:+.4f}")

for frame, displacement in cubic.items():
    analysis.set_displacement(frame, displacement)

# %%
# The displacement :math:`U_Z` of the center vertex over time.

import matplotlib.pyplot as plt

fine = numpy.linspace(times[0], times[-1], 300)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(fine, [true_displacement(t)[center, 2] for t in fine], "k-", lw=0.8, label="true")
ax.plot(times[measured], [displacements[f][center, 2] for f in measured], "o", label="measured")
ax.plot(times[missing], [linear[f][center, 2] for f in missing], "s", label="linear")
ax.plot(times[missing], [cubic[f][center, 2] for f in missing], "^", label="cubic")
ax.set_xlabel("time")
ax.set_ylabel("$U_Z$ at X = 0")
ax.legend()
plt.show()

# %%
# Median filter
# -------------
#
# The neighborhood of a vertex is given by a topological distance (``max_distance=1``: the
# vertices sharing an element, here 9 vertices) or by a radius in the unit of the mesh
# (``radius=0.25``: 21 vertices here, the spacing being 0.1). The outlier of the frame 12 disappears
# while the shape of the field is kept.

from pysdic.workflows import compute_median_filtered_displacements

by_distance = compute_median_filtered_displacements(analysis, [12], max_distance=1)[12]
by_radius = compute_median_filtered_displacements(analysis, [12], radius=0.25)[12]

truth = true_displacement(times[12])
for name, displacement in (("measured", displacements[12]), ("max_distance=1", by_distance), ("radius=0.25", by_radius)):
    print(f"{name:<15} max error {numpy.abs(displacement - truth).max():.4f}")

fig, axes = plt.subplots(1, 3, figsize=(13, 3))
for ax, (name, displacement) in zip(axes, (("measured", displacements[12]), ("max_distance=1", by_distance), ("radius=0.25", by_radius))):
    image = ax.imshow((displacement - truth)[:, 2].reshape(n_y, n_x), cmap="coolwarm", vmin=-0.01, vmax=0.01, origin="lower")
    ax.set_title(f"Error of $U_Z$, {name}")
fig.colorbar(image, ax=axes, shrink=0.8)
plt.show()