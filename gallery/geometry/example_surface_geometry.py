r"""
.. _example_surface_geometry:

Areas, centers and normals of surface meshes
============================================

.. contents:: Table of Contents
   :local:
   :depth: 1
   :backlinks: top

The methods of :class:`pysdic.Mesh` compute the geometry of 3D surface meshes: the areas,
centers and normals of the elements, the normals at the vertices and at integration
points. They are computed from the Jacobian matrices of the elements, so they work for
all the surface element types (``triangle_3``, ``triangle_6``, ``quadrangle_4``,
``quadrangle_8``).

The orientation of the normals follows the order of the vertices of the elements (the
``direct`` argument of the ``create_3d_*`` functions).

"""

# %%
# Areas of the elements
# ---------------------
#
# The area of an element is integrated with the Gauss points of its type. For a cylinder of
# radius 1 and height 2, the total area tends to :math:`4\pi` when the mesh is refined.

import numpy
from pysdic import Plotter3D, create_3d_quadrangle_4_cylindrical

n_t = 20
cylinder = create_3d_quadrangle_4_cylindrical(
    t_bounds=(0.0, 2 * numpy.pi * (1 - 1 / n_t)), h_bounds=(0.0, 2.0), n_t=n_t, n_h=6, close=True
)
areas = cylinder.compute_elements_areas()  # (N_e,)
print(f"Total area: {areas.sum():.4f} (cylinder: {4 * numpy.pi:.4f})")

plotter = Plotter3D()
plotter.add_mesh(cylinder, element_property=areas, show_bar=True, bar={"title": "Area"}, show_vertices=False)
plotter.set_title("Areas of the elements")
plotter.show()

# %%
# Normals of the elements
# -----------------------
#
# The normal of an element is computed at its center (see
# :meth:`pysdic.Mesh.compute_elements_centers`). Here it points outward, along the radial
# direction.

centers = cylinder.compute_elements_centers()  # (N_e, 3)
normals = cylinder.compute_elements_normals()  # (N_e, 3)
radial = centers[:, :2] / numpy.linalg.norm(centers[:, :2], axis=1, keepdims=True)
print(f"Max deviation from the radial direction: {numpy.abs(normals[:, :2] - radial).max():.1e}")

plotter = Plotter3D()
plotter.add_mesh(cylinder, face_color="lightgray", show_vertices=False, element_vectors=normals, vector_scale=0.3, vector_color="red")
plotter.set_title("Normals of the elements")
plotter.show()

# %%
# Normals at the vertices
# -----------------------
#
# The normal at a vertex is the mean of the normals of its elements, weighted by their
# areas. With ``direct=False``, the normals of the sphere point outward.

from pysdic import create_3d_triangle_3_spherical

n_p = 24
sphere = create_3d_triangle_3_spherical(
    t_bounds=(0.15, numpy.pi - 0.15), p_bounds=(0.0, 2 * numpy.pi * (1 - 1 / n_p)), n_t=12, n_p=n_p,
    close=True, direct=False,
)
vertex_normals = sphere.compute_vertices_normals()  # (N_v, 3)
radial = sphere.coordinates / numpy.linalg.norm(sphere.coordinates, axis=1, keepdims=True)
angles = numpy.degrees(numpy.arccos(numpy.clip((vertex_normals * radial).sum(axis=1), -1.0, 1.0)))
print(f"Mean angle with the radial direction: {angles.mean():.2f} deg")

plotter = Plotter3D()
plotter.add_mesh(sphere, face_color="lightblue", vertex_size=3, vertex_vectors=vertex_normals, vector_scale=0.2, vector_color="navy")
plotter.set_title("Normals at the vertices")
plotter.show()

# %%
# Normals at integration points
# -----------------------------
#
# On a curved element (here the bilinear quadrangles of the surface :math:`z = 0.4\,xy`),
# the normal varies inside the element: it can be computed at any integration point.

from pysdic import IntegrationPoints, create_3d_quadrangle_4_heightmap

heightmap = create_3d_quadrangle_4_heightmap(lambda x, y: 0.4 * x * y, x_bounds=(-1.0, 1.0), y_bounds=(-1.0, 1.0), n_x=4, n_y=4)
gauss_points = IntegrationPoints.from_gauss_points(heightmap.element_type, heightmap.n_elements)
gauss_points["normal"] = heightmap.compute_integration_points_normals(gauss_points)  # (N_p, 3)
print(f"Normals at the Gauss points of element 0:\n{gauss_points['normal'][gauss_points.element_indices == 0]}")

plotter = Plotter3D()
plotter.add_mesh(heightmap, face_color="wheat", show_vertices=False)
plotter.add_integration_points(gauss_points, heightmap, color="black", size=6, vectors="normal", vector_scale=0.25, vector_color="darkgreen")
plotter.set_title("Normals at the Gauss points")
plotter.show()