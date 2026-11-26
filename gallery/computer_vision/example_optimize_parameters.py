r"""
.. _example_optimize_parameters:

Optimizing the parameters of transformations
============================================

.. contents:: Table of Contents
   :local:
   :depth: 1
   :backlinks: top

The parameters of transformations are estimated from pairs of input and output points,
from a single transformation to the calibration of a camera observing a target in several
positions. The least squares problems are solved with ``pysolvegn``.

"""

# %%
# Single transformation
# ---------------------
#
# A homography :math:`H` is estimated from noisy pairs of points :math:`Y_k \approx H(X_k)`:
#
# .. math::
#
#     \min_{\vec{p}} \frac{1}{2} \sum_k \| H(X_k; \vec{p}) - Y_k \|^2
#
# :func:`pysdic.optimize_transformation_parameters` does not modify the transformation: it
# returns the optimized parameters and the ``pysolvegn.SolveResult``. Its keyword arguments
# (``max_iteration``, ``xtol``, ``verbosity``, ...) are given to ``pysolvegn.solve``.

import numpy
from pysdic import PlanarHomographicExtrinsic, optimize_transformation_parameters

rng = numpy.random.default_rng(0)

true_homography = PlanarHomographicExtrinsic([1.1, 0.05, 3.0, -0.02, 0.95, -2.0, 1e-4, 2e-4])
input_points = rng.uniform(-100.0, 100.0, (200, 2))
output_points = true_homography.apply(input_points).x + rng.normal(scale=0.1, size=(200, 2))

homography = PlanarHomographicExtrinsic([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])  # Initial guess: identity
parameters, result = optimize_transformation_parameters(homography, input_points, output_points, max_iteration=20, xtol=1e-10)
print(f"Success: {result.success} in {result.n_iterations} iterations")
print("True parameters:     ", numpy.round(true_homography.parameters, 5))
print("Optimized parameters:", numpy.round(parameters, 5))

# %%
# Camera calibration
# ------------------
#
# A camera observes a planar target (a grid on :math:`Z = 0`) in 6 positions. Each view
# :math:`v` is a chain of transformations
#
# .. code-block:: text
#
#    target points -> extrinsic_v -> distortion -> intrinsic -> image points of the view v
#
# where the extrinsics are specific to each view, and the distortion and the intrinsic are
# **shared**: they are estimated with all the views at once. The data are synthetic, with
# 0.2 pixel of noise, and the initial guess is perturbed.

from pysdic import BrownDistortion, PerspectiveExtrinsic, PinholeIntrinsic, project_points

n_views = 6
grid_x, grid_y = numpy.meshgrid(numpy.linspace(-0.2, 0.2, 9), numpy.linspace(-0.15, 0.15, 7))
target_points = numpy.stack((grid_x.ravel(), grid_y.ravel(), numpy.zeros(grid_x.size)), axis=-1)  # (63, 3)

true_intrinsic = PinholeIntrinsic([1200.0, 1210.0, 640.0, 480.0])  # fx, fy, cx, cy
true_distortion = BrownDistortion([-0.15, 0.08, 0.0, 0.0, 0.0])  # k1, k2, p1, p2, k3
true_extrinsics = [
    PerspectiveExtrinsic.from_rt(rng.normal(0.0, 0.2, 3), [rng.normal(0.0, 0.05), rng.normal(0.0, 0.05), 1.0])
    for _ in range(n_views)
]
image_points = [
    project_points(target_points, true_intrinsic, true_distortion, extrinsic).image_points
    + rng.normal(scale=0.2, size=(target_points.shape[0], 2))
    for extrinsic in true_extrinsics
]

intrinsic = PinholeIntrinsic([1100.0, 1100.0, 620.0, 500.0])
distortion = BrownDistortion([0.0, 0.0, 0.0, 0.0, 0.0])
extrinsics = [PerspectiveExtrinsic(extrinsic.parameters + rng.normal(0.0, 0.02, 6)) for extrinsic in true_extrinsics]
initial_parameters = [t.parameters.copy() for t in extrinsics + [distortion, intrinsic]]  # For the next sections

# %%
# The transformations are gathered in one sequence, and each chain lists the indices of its
# transformations in the order of application (the shared ones appear in several chains).
# :func:`pysdic.optimize_chains_transformation_parameters` returns the parameters of each
# transformation, in the order of the sequence.

from pysdic import optimize_chains_transformation_parameters

transformations = extrinsics + [distortion, intrinsic]
index_distortion, index_intrinsic = n_views, n_views + 1
chains = [(v, index_distortion, index_intrinsic) for v in range(n_views)]
inputs = [target_points] * n_views

optimized_parameters, result = optimize_chains_transformation_parameters(
    transformations, chains, inputs, image_points, max_iteration=50, xtol=1e-10
)
for transformation, parameters in zip(transformations, optimized_parameters):
    transformation.parameters = parameters

print("True intrinsic:      ", numpy.round(true_intrinsic.parameters, 2))
print("Optimized intrinsic: ", numpy.round(intrinsic.parameters, 2))
print("True distortion:     ", numpy.round(true_distortion.parameters, 4))
print("Optimized distortion:", numpy.round(distortion.parameters, 4))

errors = numpy.concatenate([
    project_points(target_points, intrinsic, distortion, extrinsic).image_points - points
    for extrinsic, points in zip(extrinsics, image_points)
])
print(f"RMS reprojection error: {numpy.sqrt(numpy.mean(numpy.sum(errors**2, axis=1))):.3f} pixels")

# %%
# Optimizing only some parameters
# -------------------------------
#
# With a flat target at a single distance, :math:`k_2` and :math:`k_3` are strongly
# correlated: the full distortion model is poorly determined (see :math:`k_3` above), even
# with a small reprojection error. A :class:`pysdic.ReparametrizedTransformation` with a
# :class:`pysdic.SubsetParametrization` optimizes only :math:`\vec{q} = (k_1, k_2)`, the
# other coefficients being fixed to 0:
#
# .. math::
#
#     \vec{p} = (k_1, k_2, 0, 0, 0) = P(\vec{q})
#
# The full parameters :math:`\vec{p}` are those of its ``transformation`` attribute.

from pysdic import ReparametrizedTransformation, SubsetParametrization

radial_distortion = ReparametrizedTransformation(
    BrownDistortion([0.0, 0.0, 0.0, 0.0, 0.0]), SubsetParametrization([True, True, False, False, False])
)

seq_parameters = initial_parameters[:n_views] + [None, initial_parameters[index_intrinsic]]  # None: current parameters
optimized_parameters, result = optimize_chains_transformation_parameters(
    extrinsics + [radial_distortion, intrinsic], chains, inputs, image_points,
    seq_parameters=seq_parameters, max_iteration=50, xtol=1e-10,
)
radial_distortion.parameters = optimized_parameters[index_distortion]
print("Optimized q:", numpy.round(radial_distortion.parameters, 4))
print("Optimized p:", numpy.round(radial_distortion.transformation.parameters, 4))

# %%
# Modes of the problem
# --------------------
#
# - ``mode="rJ"`` (default): the residuals and the Jacobian of all the equations are given
#   to the solver. Robust losses (``loss="soft_l1"``, ``"cauchy"``, ...) reduce the
#   influence of outliers.
# - ``mode="gH"``: the gradient :math:`J^T \vec{r}` and the Hessian :math:`J^T J` are
#   assembled chain by chain, without storing the Jacobian (less memory, linear loss only).
#
# Both modes give the same solution with the linear loss.

optimized_parameters, result = optimize_chains_transformation_parameters(
    extrinsics + [distortion, intrinsic], chains, inputs, image_points,
    seq_parameters=initial_parameters, mode="gH", max_iteration=50, xtol=1e-10,
)
print("Intrinsic with mode='gH':", numpy.round(optimized_parameters[index_intrinsic], 2))

# %%
# Combining terms
# ---------------
#
# :func:`pysdic.build_chains_transformation_term` builds the problem as a ``pysolvegn.Term``,
# to combine it with other terms (a regularization, groups of chains with other weights or
# losses) before ``pysolvegn.solve``. All the terms of a problem must be built with the
# **same** sequence of transformations, so that they share the vector of parameters. Here
# the first view gets a lower weight.

import pysolvegn
from pysdic import build_chains_transformation_term

transformations = extrinsics + [distortion, intrinsic]
term_first_view, p0 = build_chains_transformation_term(
    transformations, chains[:1], inputs[:1], image_points[:1], initial_parameters, weight=0.1
)
term_other_views, _ = build_chains_transformation_term(
    transformations, chains[1:], inputs[1:], image_points[1:], initial_parameters
)
result = pysolvegn.solve([term_first_view, term_other_views], p0, max_iteration=50, xtol=1e-10)
print("Intrinsic with weighted views:", numpy.round(result.parameters[-4:], 2))  # The intrinsic is the last transformation