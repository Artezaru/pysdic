from typing import Optional
from numbers import Number

import numpy

def compute_BRDF_ward_model(
    surface_point: numpy.ndarray,
    surface_normal: numpy.ndarray,
    light_position: numpy.ndarray,
    observer_position: numpy.ndarray,
    diffuse_coefficient: Optional[float] = 1.0,
    specular_coefficient: Optional[float] = 1.0,
    roughness: Optional[float] = 1.0,
) -> numpy.ndarray:
    # Input validation
    surface_point = numpy.asarray(surface_point, dtype=float)
    if surface_point.ndim != 2:
        raise ValueError("surface_point must be a 2D array with shape (N_p, E).")
    
    surface_normal = numpy.asarray(surface_normal, dtype=float)
    if surface_normal.ndim != 2:
        raise ValueError("surface_normal must be a 2D array with shape (N_p, E).")

    light_position = numpy.asarray(light_position, dtype=float)
    if light_position.ndim != 2:
        raise ValueError("light_position must be a 2D array with shape (N_l, E).")
    
    observer_position = numpy.asarray(observer_position, dtype=float)
    if observer_position.ndim != 2:
        raise ValueError("observer_position must be a 2D array with shape (N_o, E).")

    if not surface_point.shape[1] == surface_normal.shape[1] == light_position.shape[1] == observer_position.shape[1]:
        raise ValueError("All input arrays must have the same number of spatial dimensions (E).")
    if not surface_point.shape[0] == surface_normal.shape[0]:
        raise ValueError("surface_point and surface_normal must have the same number of points (N_p).")
    
    if not isinstance(diffuse_coefficient, Number) or diffuse_coefficient < 0:
        raise ValueError("diffuse_coefficient must be a non-negative number.")
    
    if not isinstance(specular_coefficient, Number) or specular_coefficient < 0:
        raise ValueError("specular_coefficient must be a non-negative number.")
    
    if not isinstance(roughness, Number) or roughness <= 0:
        raise ValueError("roughness must be a positive number.")

    N_p, E = surface_point.shape
    N_l = light_position.shape[0]
    N_o = observer_position.shape[0]

    # Compute the input, reflected, and observer direction vectors
    # -------------------
    # Input : Ligth to surface
    # Observer : Surface to observer
    # Reflected : reflection of input direction about surface normal
    # Half-vector : bisector between input and observer directions
    # -------------------
    I = surface_point[:, numpy.newaxis, :] - light_position[numpy.newaxis, :, :] # Shape: (N_p, N_l, E)
    I = I / numpy.linalg.norm(I, axis=2, keepdims=True)  # Normalize
    
    O = observer_position[numpy.newaxis, :, :] - surface_point[:, numpy.newaxis, :] # Shape: (N_p, N_o, E)
    O = O / numpy.linalg.norm(O, axis=2, keepdims=True)  # Normalize

    R = I - 2 * numpy.einsum('ple,pe->pl', I, surface_normal)[:, :, numpy.newaxis] * surface_normal[:, numpy.newaxis, :]  # Shape: (N_p, N_l, E)
    R = R / numpy.linalg.norm(R, axis=2, keepdims=True) # Normalize

    H = (I[:, :, numpy.newaxis, :] + O[:, numpy.newaxis, :, :]) / 2  # Shape: (N_p, N_l, N_o, E)
    H = H / numpy.linalg.norm(H, axis=3, keepdims=True)  # Normalize

    # Compute the angles 
    # -------------------
    # theta_i : angle between input direction and surface normal
    # theta_0 : angle between observer direction and surface normal
    # delta : angle between the normal and the half-vector
    # -------------------

    cos_theta_i = numpy.einsum('ple,pe->pl', -I, surface_normal)  # Shape: (N_p, N_l)
    cos_theta_0 = numpy.einsum('poe,pe->po', O, surface_normal)  # Shape: (N_p, N_o)
    cos_delta = numpy.einsum('ploe,pe->plo', H, surface_normal)  # Shape: (N_p, N_l, N_o)
    tan_2_delta = (1 - cos_delta**2) / (cos_delta**2 + 1e-10)  # Shape: (N_p, N_l, N_o) # Avoid division by zero

    # print(f"cos_theta_i shape: {cos_theta_i.shape}")
    # print(f"cos_theta_i values: {cos_theta_i}")

    # print(f"cos_theta_0 shape: {cos_theta_0.shape}")
    # print(f"cos_theta_0 values: {cos_theta_0}")

    # print(f"tan_2_delta shape: {tan_2_delta.shape}")
    # print(f"tan_2_delta values: {tan_2_delta}")

    # Compute the BRDF using Ward's model
    # -------------------
    #
    # BRDF(theta_i, theta_0) = (diffuse_coefficient / pi) + (specular_coefficient / (4 * pi * roughness^2 * sqrt(cos(theta_i) * cos(theta_0)))) * exp(- (tan(delta)^2) / roughness^2)
    #

    brdf = (diffuse_coefficient / numpy.pi) + \
           (specular_coefficient / (4 * numpy.pi * roughness**2 * numpy.sqrt(cos_theta_i[:, :, numpy.newaxis] * cos_theta_0[:, numpy.newaxis, :]) + 1e-10)) * \
           numpy.exp(- tan_2_delta / (roughness**2 + 1e-10))  # Shape: (N_p, N_l, N_o)
    
    return brdf



if __name__ == "__main__":

    # Example usage
    surface_point = numpy.array([[0.0, 0.0, 0.0]])
    surface_normal = numpy.array([[0.0, 0.0, 1.0]])
    light_position = numpy.array([[10.0, 0.0, 10.0]])
    observer_position = numpy.array([[0.0, 0.0, 10.0]])

    brdf_values = compute_BRDF_ward_model(
        surface_point,
        surface_normal,
        light_position,
        observer_position,
        diffuse_coefficient=0.5,
        specular_coefficient=0.5,
        roughness=0.2
    )

    print("BRDF Values:\n", brdf_values)


    # Mapping observer positions
    thetas = numpy.linspace(0, numpy.pi/2, 5)
    phis = numpy.linspace(0, 2*numpy.pi, 10)
    observer_positions = []
    for theta in thetas:
        for phi in phis:
            x = 10.0 * numpy.sin(theta) * numpy.cos(phi)
            y = 10.0 * numpy.sin(theta) * numpy.sin(phi)
            z = 10.0 * numpy.cos(theta)
            observer_positions.append([x, y, z])
    observer_positions = numpy.array(observer_positions)

    brdf_values = compute_BRDF_ward_model(
        surface_point,
        surface_normal,
        light_position,
        observer_positions,
        diffuse_coefficient=0.5,
        specular_coefficient=0.5,
        roughness=0.2
    )

    print("BRDF Values for multiple observer positions:\n", brdf_values)

    # Display 

