from pysdic.geometry import create_linear_triangle_heightmap
import numpy as np
from pysdic.visualizer import visualize_qt_pyvista_linear_triangle_mesh_3d

surface_mesh = create_linear_triangle_heightmap(
    height_function=lambda x, y: 0.5 * np.sin(np.pi * x) * np.cos(np.pi * y),
    x_bounds=(-1.0, 1.0),
    y_bounds=(-1.0, 1.0),
    n_x=50,
    n_y=50,
)

# Create some property arrays
height = surface_mesh.vertices.points[:, 2].reshape(-1, 1)  # Use the z-coordinate as a property
surface_mesh.set_vertices_property("Height", height) # Store in the mesh

displacement = np.zeros_like(surface_mesh.vertices.points)
displacement[:, 0] = 0.1 * np.sin(2 * np.pi * surface_mesh.vertices.points[:, 0])
displacement[:, 1] = 0.1 * np.cos(2 * np.pi * surface_mesh.vertices.points[:, 1])
displacement[:, 2] = 0.1 * np.sin(2 * np.pi * surface_mesh.vertices.points[:, 0]) * np.cos(2 * np.pi * surface_mesh.vertices.points[:, 1])
properties = {
    "Displacement": displacement,  # Vectorial property stored as a separate array
}

# Create some rays to cast
ray_origins = np.random.uniform(-1, 1, (100, 3))
ray_origins[:, 2] = 3.0  # Start above the surface
ray_directions = np.tile(np.array([[0, 0, -1]]), (100, 1))  # Pointing downwards

intersection_points = surface_mesh.cast_rays(ray_origins, ray_directions)
integration_points = {
    "Ray Intersections": intersection_points,
}

# Create some textures
u = np.linspace(0, 1, 50)
v = np.linspace(0, 1, 50)
U, V = np.meshgrid(u, v)

texture_image_1 = np.round(255/2 + 255/2 * np.sin(U * 4 * np.pi)).astype(np.uint8)  # Example texture image with shape (50, 50)
texture_image_2 = np.round(255/2 + 255/2 * np.sin(V * 4 * np.pi)).astype(np.uint8)  # Another example texture image with shape (50, 50)
textures = {
    "Sine Texture": texture_image_1,
    "Cosine Texture": texture_image_2,
    "Coconut Texture": np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8),  # Random RGB texture with shape (16, 16, 3)
}


# Visualize using the Qt PyVista viewer
visualize_qt_pyvista_linear_triangle_mesh_3d(
    mesh=surface_mesh,
    property_arrays=properties,
    integration_points=integration_points,
    textures=textures,
)