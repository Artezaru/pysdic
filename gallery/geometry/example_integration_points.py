r"""
.. _example_integration_points:

Integration points: interpolation, derivatives and projection
=============================================================

.. contents:: Table of Contents
   :local:
   :depth: 1
   :backlinks: top

An :class:`pysdic.IntegrationPoints` object locates points in the elements of a mesh by
their element index and natural coordinates :math:`\Xi`. The methods of
:class:`pysdic.Mesh` use them to interpolate a property given at the vertices with the
shape functions,

.. math::

    u(\Xi) = \sum_{i=1}^{N_{vpe}} N_i(\Xi) \, u_i,

to compute its derivatives and strains, and to project values given at the points back to
the vertices.

.. seealso::

    :mod:`pysdic.fem` for the same operations on raw arrays.

"""

# %%
# Mesh and Gauss points
# ---------------------
#
# :meth:`pysdic.IntegrationPoints.from_gauss_points` creates the Gauss points of all the
# elements (4 per ``quadrangle_4``), and
# :meth:`pysdic.Mesh.compute_integration_points_coordinates` gives their global
# coordinates.

import numpy
from pysdic import IntegrationPoints, Plotter3D, create_2d_quadrangle_4_rectangle

mesh = create_2d_quadrangle_4_rectangle(x_bounds=(0.0, 2.0), y_bounds=(0.0, 1.0), n_x=9, n_y=5)
gauss_points = IntegrationPoints.from_gauss_points(mesh.element_type, mesh.n_elements)
print(gauss_points)

coordinates = mesh.compute_integration_points_coordinates(gauss_points).coordinates  # (N_p, 2)
print(f"First Gauss point (element {gauss_points.element_indices[0]}): {coordinates[0]}")

plotter = Plotter3D()
plotter.add_mesh(mesh, face_color="lightgray", show_vertices=False)
plotter.add_integration_points(gauss_points, mesh, color="red", size=6)
plotter.set_title("Gauss points of a quadrangle_4 mesh")
plotter.plotter.view_xy()
plotter.show()

# %%
# Interpolation of a property
# ---------------------------
#
# A property of the vertices (here a displacement field) is interpolated at the points with
# :meth:`pysdic.Mesh.compute_property_interpolation`. The property is given by its name or
# as an array, and the result always has shape :math:`(N_p, P)`.

x, y = mesh.coordinates[:, 0], mesh.coordinates[:, 1]
mesh.vertices["displacement"] = numpy.stack((0.05 * x * y, 0.02 * x**2), axis=1)  # (N_v, 2)
gauss_points["displacement"] = mesh.compute_property_interpolation("displacement", gauss_points)  # (N_p, 2)

plotter = Plotter3D()
plotter.add_mesh(mesh, warp_by="displacement", face_color="lightgray", show_vertices=False)
plotter.add_integration_points(
    gauss_points, mesh, warp_by="displacement", property="displacement", size=8,
    show_bar=True, bar={"title": "|u|"}, vectors="displacement", vector_scale=3.0,
)
plotter.set_title("Displacement at the Gauss points (arrows x3), on the deformed mesh")
plotter.plotter.view_xy()
plotter.show()

# %%
# The interpolation is linear in the values of the vertices: it is the product with the
# shape function matrix :math:`\mathcal{N}` of shape :math:`(N_p, N_v)`.

N = mesh.compute_shape_function_matrix(gauss_points, sparse=True)
difference = numpy.abs(N @ mesh.vertices["displacement"] - gauss_points["displacement"]).max()
print(f"Shape function matrix {N.shape}: max difference with the interpolation {difference:.1e}")

# %%
# Derivatives and strains
# -----------------------
#
# :meth:`pysdic.Mesh.compute_strains` gives the linear strain tensor
# :math:`\varepsilon = \frac{1}{2}(\nabla u + \nabla u^T)` at the points, of shape
# :math:`(N_p, 2, 2)` (see :meth:`pysdic.Mesh.compute_property_derivative` for the
# gradient itself). Here :math:`u_x = 0.05\,xy` is bilinear, so
# :math:`\varepsilon_{xx} = 0.05\,y` is exact.

strains = mesh.compute_strains("displacement", gauss_points)
gauss_points["strain_xx"] = strains[:, 0, 0]
print(f"Max difference with 0.05 y: {numpy.abs(strains[:, 0, 0] - 0.05 * coordinates[:, 1]).max():.1e}")

plotter = Plotter3D()
plotter.add_mesh(mesh, face_color="lightgray", show_vertices=False)
plotter.add_integration_points(gauss_points, mesh, property="strain_xx", size=10, show_bar=True, bar={"title": "eps_xx"})
plotter.set_title("Strain eps_xx at the Gauss points")
plotter.plotter.view_xy()
plotter.show()

# %%
# Projection to the vertices
# --------------------------
#
# :meth:`pysdic.Mesh.compute_property_projection` does the reverse operation: the values at
# the vertices are fitted by least squares on the values at the points (weighted by
# ``integration_points.weights``). A property interpolated from the vertices is recovered,
# as long as the points are enough to determine the values of the vertices (here 4 points
# per element).

projected = mesh.compute_property_projection("displacement", gauss_points)  # (N_v, 2)
print(f"Max projection error: {numpy.abs(projected - mesh.vertices['displacement']).max():.1e}")

# %%
# Neighborhoods
# -------------
#
# :meth:`pysdic.Mesh.compute_integration_points_neighborhood` gives, for each element, the
# points located in the elements at a distance :math:`\leq` ``max_distance`` (two elements
# are at distance 1 if they share a vertex).

neighborhood = mesh.compute_integration_points_neighborhood(gauss_points, max_distance=1)  # (N_e, N_p)
element = 13
selected = numpy.zeros(gauss_points.n_points, dtype=bool)
selected[neighborhood[element].indices] = True
print(f"Points in the neighborhood of element {element}: {selected.sum()}")

plotter = Plotter3D()
plotter.add_mesh(mesh, face_color="lightgray", show_vertices=False)
plotter.add_integration_points(gauss_points, mesh, mask=~selected, color="black", size=5)
plotter.add_integration_points(gauss_points, mesh, mask=selected, color="red", size=9)
plotter.set_title(f"Gauss points in the neighborhood of element {element}")
plotter.plotter.view_xy()
plotter.show()

# %%
# Removing elements
# -----------------
#
# When elements are removed from a mesh, :meth:`pysdic.IntegrationPoints.filter_elements`
# updates the points with the same mask: the points of the removed elements are unlocated
# (element index ``-1``, not displayed) and the other element indices are renumbered.

keep = mesh.compute_elements_centers()[:, 0] < 1.0  # The elements with x < 1
filtered_mesh = mesh.filter_connectivity(keep).remove_unused_vertices()
filtered_points = gauss_points.filter_elements(keep)
filtered_mesh.validate_integration_points(filtered_points)
print(f"Located points: {filtered_points.n_located} / {filtered_points.n_points}")

plotter = Plotter3D()
plotter.add_mesh(filtered_mesh, face_color="lightgray", show_vertices=False)
plotter.add_integration_points(filtered_points, filtered_mesh, color="red", size=6)
plotter.set_title("Gauss points after removing the elements with x > 1")
plotter.plotter.view_xy()
plotter.show()