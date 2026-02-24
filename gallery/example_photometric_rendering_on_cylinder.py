"""
.. _example_photometric_rendering_on_cylinder:

Rendering Photometric Properties on a Cylinder
=====================================================================================

.. contents:: Table of Contents
   :local:
   :depth: 1
   :backlinks: top

This example demonstrates how to use the Bouguer law and Ward's BRDF model
from the ``pysdic`` library to render photometric properties on a cylindrical mesh.

.. seealso::

    - :func:`pysdic.compute_bouguer_law` - Official documentation for the Bouguer law function.
    - :func:`pysdic.compute_brdf_ward` - Official documentation for Ward's BRDF function.

.. note::

    Other implemented BRDF models can be use in a similar way, such as Beckmann's model
    using the :func:`pysdic.compute_brdf_beckmann` function.
    
"""

# %%
# Define some points on a cylinder surface
# --------------------------------------------------------------------------------
#
# Construct some points on a cylinder surface using cylindrical coordinates.
# The cylinder is defined by its radius and height range.

import numpy

theta_min = -numpy.pi
theta_max = numpy.pi
height_min = -1.0
height_max = 1.0
radius = 1.0
nt = 1000  # Number of points along the circumference
nh = 100  # Number of points along the height

theta = numpy.linspace(theta_min, theta_max, nt)
height = numpy.linspace(height_min, height_max, nh)
theta_grid, height_grid = numpy.meshgrid(theta, height)

# points
x = radius * numpy.cos(theta_grid)
y = radius * numpy.sin(theta_grid)
z = height_grid
points_coordinates = numpy.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)  # (N_points, 3)

# normals
normals_x = numpy.cos(theta_grid)
normals_y = numpy.sin(theta_grid)
normals_z = numpy.zeros_like(normals_x)
points_normals = numpy.stack([normals_x.flatten(), normals_y.flatten(), normals_z.flatten()], axis=1)  # (N_points, 3)

# %%
# Define light source and view directions
# --------------------------------------------------------------------------------
#
# Define the light source direction and observer position for the photometric rendering.
#
# For this example, we will use two light sources positioned at different locations.
# We also define the observer position and the objective is the render the appearance of the cylinder
# as seen from this position.

light_positions = numpy.array([
    [1.0, 1.0, 0.0],
    [1.0, -1.0, 0.0]
])  # (N_light_sources, 3)

observer_positions = numpy.array([5.0, 0.0, 0.0])  # (3,)

# %%
# Compute photometric properties using Bouguer law and Ward's BRDF model
# --------------------------------------------------------------------------------
#
# Use the ``compute_bouguer_law`` and ``compute_brdf_ward`` functions to compute
# the photometric properties at the defined points on the cylinder surface.
#
# The first one is the ratio between irradiance :math:`E` at source intensity :math:`I`, the second one is the BRDF ratio between outgoing radiance :math:`L_o` and incoming irradiance :math:`E`,
# using 3 parameters: diffuse reflectance :math:`\rho_d`, specular reflectance :math:`\rho_s`, and surface roughness :math:`\sigma` (See the official documentation for more details).
#
# Assuming a given intensity for the light sources :math:`I = 1.0`, the image brightness is proportional to the incoming radiance :math:`L_o` on the camera sensor.

from pysdic import compute_bouguer_law, compute_brdf_ward

bouger_law_ratio = compute_bouguer_law(
    surface_points=points_coordinates,
    surface_normals=points_normals,
    light_positions=light_positions,
) # (N_points, N_light_sources)
print(f"Bouguer law ratio shape: {bouger_law_ratio.shape}, min: {bouger_law_ratio.min()}, max: {bouger_law_ratio.max()}")

brdf_ward_values = compute_brdf_ward(
    surface_points=points_coordinates,
    surface_normals=points_normals,
    light_positions=light_positions,
    observer_positions=observer_positions,
    parameters = [0.1, 0.1, 0.5]  # rho_d, rho_s, sigma
) # (N_points, N_light_sources, N_observers, N_models)

# Here N_observers = 1 and N_models = 1, so we can squeeze these dimensions
brdf_ward_values = brdf_ward_values[:, :, 0, 0]  # (N_points, N_light_sources)
print(f"BRDF Ward values shape: {brdf_ward_values.shape}, min: {brdf_ward_values.min()}, max: {brdf_ward_values.max()}")

# Compute the radiance
radiance = bouger_law_ratio * brdf_ward_values  # (N_points, N_light_sources)

# Sum the contributions from all light sources
total_radiance = numpy.sum(radiance, axis=1)  # (N_points,)
print(f"Total radiance shape: {total_radiance.shape}, min: {total_radiance.min()}, max: {total_radiance.max()}")

# %%
# Visualize the rendered photometric properties on the cylinder surface
# --------------------------------------------------------------------------------
#
# Lets build a Point Cloud object to visualize the rendered photometric properties
# on the cylinder surface using pyvista.

from pysdic import PointCloud

pc = PointCloud(points_coordinates)
pc['radiance'] = total_radiance
pc.visualize(
    property='radiance', 
    property_cmap='plasma', 
    point_size=5,
    camera_position=[10.0, 0.0, 0.0],
    camera_view_up=[0, 0, 1],
    camera_focal_point=[0, 0, 0],
    title="Photometric Rendering on Cylinder Surface"
)

