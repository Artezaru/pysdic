r"""
.. _example_cast_rays:

Intersection of rays with surface meshes
========================================

.. contents:: Table of Contents
   :local:
   :depth: 1
   :backlinks: top

:meth:`pysdic.Mesh.cast_rays` computes the first intersection of rays with a 3D surface
mesh. The intersections are returned as :class:`pysdic.IntegrationPoints`, one point per
ray (unlocated if the ray misses the mesh), so all the methods of :class:`pysdic.Mesh`
can use them.

- For ``triangle_3`` meshes, the intersections are computed by Open3D.
- For the other elements (e.g. ``quadrangle_4``), Open3D gives a first approximation on a
  triangulation of the elements, refined by Newton iterations on their exact surface.

.. important::

    Requires the optional package ``open3d`` (``pip install open3d``).

.. seealso::

    :meth:`pysdic.Camera.get_camera_rays` for the rays of the pixels of a camera.

"""

# %%
# Rays from a point
# -----------------
#
# A :class:`pysdic.Rays` stores the origins and the unit directions of the rays
# :math:`(o_x, o_y, o_z, d_x, d_y, d_z)`, with any batch shape: here a grid of
# :math:`20 \times 30` rays from a point above the meshes, as from a pinhole camera.

import numpy
from pysdic import Rays

origin = numpy.array([0.0, 0.0, 4.0])
u, v = numpy.meshgrid(numpy.linspace(-0.35, 0.35, 30), numpy.linspace(-0.25, 0.25, 20))  # (20, 30)
directions = numpy.stack((u, v, -numpy.ones_like(u)), axis=-1)
directions /= numpy.linalg.norm(directions, axis=-1, keepdims=True)

rays = Rays(numpy.concatenate((numpy.broadcast_to(origin, directions.shape), directions), axis=-1))
print(f"Rays: batch shape {rays.batch_shape}")

# %%
# Triangle mesh
# -------------
#
# The rays are flattened in C order: the outputs are reshaped with ``rays.batch_shape``.
# Only the first intersection is kept, so all the points are on the upper half of the
# sphere (:math:`z > 0`).

from pysdic import Plotter3D, PointCloud, create_3d_triangle_3_spherical

n_p = 30
sphere = create_3d_triangle_3_spherical(
    t_bounds=(0.1, numpy.pi - 0.1), p_bounds=(0.0, 2 * numpy.pi * (1 - 1 / n_p)), n_t=15, n_p=n_p, close=True
)
hits = sphere.cast_rays(rays)
hit_grid = hits.is_located().reshape(rays.batch_shape)  # (20, 30)
print(f"Rays hitting the sphere: {hit_grid.sum()} / {hit_grid.size}")

hit_points = sphere.compute_integration_points_coordinates(hits, allow_unlocated_points=True).coordinates
print(f"Minimum z of the intersections: {numpy.nanmin(hit_points[:, 2]):.3f}")

plotter = Plotter3D()
plotter.add_mesh(sphere, face_color="lightblue", face_opacity=0.5, show_vertices=False)
plotter.add_integration_points(hits, sphere, color="red", size=5)  # The unlocated points are not drawn
plotter.add_point_cloud(PointCloud([origin]), color="black", size=12)
plotter.set_title("Intersections with a triangle_3 sphere (origin of the rays in black)")
plotter.add_axes()
plotter.show()

# %%
# Quadrangle mesh
# ---------------
#
# On a warped ``quadrangle_4`` height map, the intersections are on the exact bilinear
# surface of the quadrangles: the points of the surface computed from the natural
# coordinates are on their rays, up to the tolerance of the Newton iterations.

from pysdic import create_3d_quadrangle_4_heightmap

heightmap = create_3d_quadrangle_4_heightmap(lambda x, y: 0.5 * x * y, x_bounds=(-1.5, 1.5), y_bounds=(-1.0, 1.0), n_x=4, n_y=3)
hits = heightmap.cast_rays(rays)
located = hits.is_located()  # (N_rays,), in the C order of the rays
hits = hits.remove_unlocated()
hit_points = heightmap.compute_integration_points_coordinates(hits).coordinates  # (N_p, 3)

# Distance of each point to its ray: |(X - o) x d|
distances = numpy.linalg.norm(numpy.cross(hit_points - origin, directions.reshape((-1, 3))[located]), axis=1)
print(f"Located rays: {hits.n_points}, max distance to the rays: {distances.max():.1e}")

# %%
# Normals at the intersections
# ----------------------------
#
# The intersections are integration points: here the normals of the surface are computed
# at the intersections (one out of 7 is drawn, for readability).

hits["normal"] = heightmap.compute_integration_points_normals(hits)  # (N_p, 3)

plotter = Plotter3D()
plotter.add_mesh(heightmap, face_color="wheat", show_vertices=False)
plotter.add_integration_points(
    hits, heightmap, mask=numpy.arange(hits.n_points) % 7 == 0, color="red", size=5,
    vectors="normal", vector_scale=0.2, vector_color="darkgreen",
)
plotter.set_title("Normals at the intersections with a quadrangle_4 height map")
plotter.add_axes()
plotter.show()