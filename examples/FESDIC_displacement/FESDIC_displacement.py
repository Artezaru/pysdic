
import numpy
import pycvcam 
import os
import copy
import sys
import time
import scipy
import matplotlib.pyplot as plt
import multiprocessing
import tqdm

from pysdic.geometry import LinearTriangleMesh3D, IntegrationPoints, PointCloud3D, integration_points
from pysdic.imaging import Image, Camera, View
from pysdic.assemblers import assembly_FESDIC_displacement, assembly_SDIC_distortion
from pysdic.visualizer import visualize_qt_pyvista_linear_triangle_mesh_3d

work_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(work_dir))

from create_polygon_mask import create_polygon_mask

DISPLAY = False
DEFINE_MASK = False
COMPUTE_DISPLACEMENT = False
REDUCE_FACTOR = 2 # Factor to reduce the number of rays for integration points creation (square root of the downsampling factor)

# ==============================================================
# Convert distortion models between image and normalized coordinates
# ==============================================================

__doc__ = """

In this script the distortion files are stored in image coordinates and not in normalized coordinates.

This allows to interpret the distortion parameters as pixels displacements on the image sensor.
Converting between the two representations is done using the intrinsic matrix of the camera.

"""

def zernike_image_to_normalized(distortion, intrinsic_matrix):
    """
    Converts a Zernike image distortion object to a normalized distortion model.

    A new object is returned with the parameters adjusted according to the intrinsic matrix.
    """
    output_distortion = copy.deepcopy(distortion)
    output_distortion.radius_x = distortion.radius_x / intrinsic_matrix[0, 0]
    output_distortion.radius_y = distortion.radius_y / intrinsic_matrix[1, 1]
    output_distortion.center_x = (distortion.center[0] - intrinsic_matrix[0, 2]) / intrinsic_matrix[0, 0]
    output_distortion.center_y = (distortion.center[1] - intrinsic_matrix[1, 2]) / intrinsic_matrix[1, 1]
    output_distortion.parameters_x = distortion.parameters_x / intrinsic_matrix[0, 0]
    output_distortion.parameters_y = distortion.parameters_y / intrinsic_matrix[1, 1]
    return output_distortion

def zernike_normalized_to_image(distortion, intrinsic_matrix):
    """
    Converts a normalized distortion model to a Zernike image distortion object.

    A new object is returned with the parameters adjusted according to the intrinsic matrix.
    """
    output_distortion = copy.deepcopy(distortion)
    output_distortion.radius_x = distortion.radius_x * intrinsic_matrix[0, 0]
    output_distortion.radius_y = distortion.radius_y * intrinsic_matrix[1, 1]
    output_distortion.center_x = distortion.center_x * intrinsic_matrix[0, 0] + intrinsic_matrix[0, 2]
    output_distortion.center_y = distortion.center_y * intrinsic_matrix[1, 1] + intrinsic_matrix[1, 2]
    output_distortion.parameters_x = distortion.parameters_x * intrinsic_matrix[0, 0]
    output_distortion.parameters_y = distortion.parameters_y * intrinsic_matrix[1, 1]
    return output_distortion


# ==============================================================
# Load Data
# ==============================================================

__doc__ = """

Example : FESDIC displacement field computation and visualization.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Lest consider a cylinder in expansion.

We have two meshes of the cylinder: a reference mesh and a deformed mesh.

This expansion is captured by a single camera with a split-screen setup (mirrors are used to have two views in the same image).
This setup can be represented by two cameras with the same intrinsic parameters but different extrinsic parameters.
Furthermore, we assume that the distortion of the cameras changes between the reference and deformed images due to some vibration of the setup.

- Camera 1 (left): Intrinsic, Left-Extrinsic, Distortion
- Camera 2 (right): Intrinsic, Right-Extrinsic, Distortion

Lets assume an error in the distortion parameters estimation or in the mesh deformation, leading to some residual error in the image correlation.
We want to be able to compute the displacement field that minimizes this residual error using FESDIC.

This example shows how to setup the problem and visualize the results using PySDIC.

"""

# Create the reference and deformed meshes
reference_mesh = LinearTriangleMesh3D.from_vtk(os.path.join(work_dir, "Meshes", "reference_mesh.vtk"))
deformed_mesh = LinearTriangleMesh3D.from_vtk(os.path.join(work_dir, "Meshes", "deformed_mesh.vtk"))

if DISPLAY:
    reference_mesh.visualize(title="Reference Mesh")
    deformed_mesh.visualize(title="Deformed Mesh")

# Load the images acquired by the split-screen camera
reference_image = Image.from_file(os.path.join(work_dir, "Images", "reference_image.tiff"))
deformed_image = Image.from_file(os.path.join(work_dir, "Images", "deformed_image.tiff"))

if DISPLAY:
    reference_image.visualize(title="Reference Image")
    deformed_image.visualize(title="Deformed Image")

# Create the Cameras
intrinsic = pycvcam.read_transform(os.path.join(work_dir, "Camera_parameters", "camera_intrinsic.json"), pycvcam.Cv2Intrinsic)
left_extrinsic = pycvcam.read_transform(os.path.join(work_dir, "Camera_parameters", "camera_left_extrinsic.json"), pycvcam.Cv2Extrinsic)
right_extrinsic = pycvcam.read_transform(os.path.join(work_dir, "Camera_parameters", "camera_right_extrinsic.json"), pycvcam.Cv2Extrinsic)
reference_distortion = pycvcam.read_transform(os.path.join(work_dir, "Camera_parameters", "camera_reference_distortion.json"), pycvcam.ZernikeDistortion)
reference_distortion = zernike_image_to_normalized(reference_distortion, intrinsic.intrinsic_matrix)
deformed_distortion = pycvcam.read_transform(os.path.join(work_dir, "Camera_parameters", "camera_deformed_distortion.json"), pycvcam.ZernikeDistortion)
deformed_distortion = zernike_image_to_normalized(deformed_distortion, intrinsic.intrinsic_matrix)

reference_left_camera = Camera(reference_image.height, reference_image.width, intrinsic, reference_distortion, left_extrinsic)
reference_right_camera = Camera(reference_image.height, reference_image.width, intrinsic, reference_distortion, right_extrinsic)
deformed_left_camera = Camera(deformed_image.height, deformed_image.width, intrinsic, deformed_distortion, left_extrinsic)
deformed_right_camera = Camera(deformed_image.height, deformed_image.width, intrinsic, deformed_distortion, right_extrinsic)

# Create the Views
reference_left_view = View(reference_left_camera, reference_image)
reference_right_view = View(reference_right_camera, reference_image)
deformed_left_view = View(deformed_left_camera, deformed_image)
deformed_right_view = View(deformed_right_camera, deformed_image)

if DISPLAY:
    reference_left_view.visualize_projected_mesh(reference_mesh, show_edges=True, show_faces=False, title="Reference Left View Mesh Projection")
    reference_right_view.visualize_projected_mesh(reference_mesh, show_edges=True, show_faces=False, title="Reference Right View Mesh Projection")
    deformed_left_view.visualize_projected_mesh(deformed_mesh, show_edges=True, show_faces=False, title="Deformed Left View Mesh Projection")
    deformed_right_view.visualize_projected_mesh(deformed_mesh, show_edges=True, show_faces=False, title="Deformed Right View Mesh Projection")




# ==============================================================
# Create a set of integrated points for FESDIC
# ==============================================================

__doc__ = """

Throw rays from the camera (Reference Left Camera) to the reference mesh to create a set of integrated points on the mesh.
These points will be used for FESDIC displacement field computation.

"""

# Select a region on the image sensor (left view)
if not os.path.exists(os.path.join(work_dir, "Masks", "left_rays_mask.npy")) or DEFINE_MASK:
    left_mask = create_polygon_mask(reference_left_view.image.to_array())
    os.makedirs(os.path.join(work_dir, "Masks"), exist_ok=True)
    numpy.save(os.path.join(work_dir, "Masks", "left_rays_mask.npy"), left_mask)
else:
    left_mask = numpy.load(os.path.join(work_dir, "Masks", "left_rays_mask.npy"))

if not os.path.exists(os.path.join(work_dir, "Masks", "right_rays_mask.npy")) or DEFINE_MASK:
    right_mask = create_polygon_mask(reference_right_view.image.to_array())
    os.makedirs(os.path.join(work_dir, "Masks"), exist_ok=True)
    numpy.save(os.path.join(work_dir, "Masks", "right_rays_mask.npy"), right_mask)
else:
    right_mask = numpy.load(os.path.join(work_dir, "Masks", "right_rays_mask.npy"))

image_mask = numpy.zeros((reference_image.height, reference_image.width), dtype=bool).flatten()
image_mask[::REDUCE_FACTOR] = True
image_mask = image_mask.reshape((reference_image.height, reference_image.width))
left_mask = left_mask & image_mask
right_mask = right_mask & image_mask

# Create integration points from rays casted through the masked region
if not os.path.exists(os.path.join(work_dir, "Integration_Points", "integration_points.npz")) or DEFINE_MASK:
    # Cast rays from both cameras
    left_rays_origin, left_rays_direction = reference_left_camera.get_camera_rays(mask=left_mask.astype(bool))
    right_rays_origin, right_rays_direction = reference_right_camera.get_camera_rays(mask=right_mask.astype(bool))

    integration_points_left = reference_mesh.cast_rays(left_rays_origin, left_rays_direction, nan_open3d_errors=True)
    integration_points_right = reference_mesh.cast_rays(right_rays_origin, right_rays_direction, nan_open3d_errors=True)
    integration_points_left.remove_invalids(inplace=True)
    integration_points_right.remove_invalids(inplace=True)

    # Keep only elements that are seen by both cameras and recover at least 80 % of the max number of points per element
    unique_indices_left, counts_left = numpy.unique(integration_points_left.element_indices, return_counts=True)
    unique_indices_right, counts_right = numpy.unique(integration_points_right.element_indices, return_counts=True)

    indices_enough_points_left = unique_indices_left[counts_left >= 0.75 * numpy.max(counts_left)] # 80 % of the max number of points
    indices_enough_points_right = unique_indices_right[counts_right >= 0.75 * numpy.max(counts_right)] # 80 % of the max number of points

    # Keep only indices common to both cameras
    indices_common = numpy.intersect1d(indices_enough_points_left, indices_enough_points_right)

    integration_points = integration_points_left + integration_points_right
    points_indices = numpy.arange(len(integration_points))
    points_indices_to_remove = points_indices[~numpy.isin(integration_points.element_indices, indices_common)]
    integration_points.remove_points(points_indices_to_remove, inplace=True)
    unique, count = numpy.unique(integration_points.element_indices, return_counts=True)
    print(f"Number of integration points per element after filtering: min={numpy.min(count)}, max={numpy.max(count)}, mean={numpy.mean(count):.2f}")

    os.makedirs(os.path.join(work_dir, "Integration_Points"), exist_ok=True)
    integration_points.to_npz(os.path.join(work_dir, "Integration_Points", "integration_points.npz"))

else:

    integration_points = IntegrationPoints.from_npz(os.path.join(work_dir, "Integration_Points", "integration_points.npz"))

usefull_vertices = reference_mesh.connectivity[numpy.unique(integration_points.element_indices)]
usefull_vertices = numpy.unique(usefull_vertices)

if DISPLAY or True:
    reference_mesh.visualize_integration_points(integration_points, title="Integration Points on Reference Mesh")





# ==============================================================
# Compute the residual for the perfect deformation
# ==============================================================

__doc__ = """

Project the integration points to both views in the reference and deformed configurations to compute the residual error for the perfect deformation with 5 links

- Reference Left View - Deformed Left View
- Reference Left View - Deformed Right View
- Reference Right View - Deformed Left View
- Reference Right View - Deformed Right View
- Deformed Left View - Deformed Right View

"""

start_time = time.time()
# Convert integration points to 3D points in space for both meshes
reference_points_3d = reference_mesh.interpolate_property_at_integration_points(integration_points, vertices_coordinates=True)
deformed_points_3d = deformed_mesh.interpolate_property_at_integration_points(integration_points, vertices_coordinates=True)

# Project to both views and compute residuals (mean per integration point)
projected_gray_level_reference_left = reference_left_view.image_project(reference_points_3d).gray_levels.reshape((-1,))
projected_gray_level_reference_right = reference_right_view.image_project(reference_points_3d).gray_levels.reshape((-1,))
projected_gray_level_deformed_left = deformed_left_view.image_project(deformed_points_3d).gray_levels.reshape((-1,))
projected_gray_level_deformed_right = deformed_right_view.image_project(deformed_points_3d).gray_levels.reshape((-1,))

residuals = [
    projected_gray_level_reference_left - projected_gray_level_deformed_left,
    projected_gray_level_reference_left - projected_gray_level_deformed_right,
    projected_gray_level_reference_right - projected_gray_level_deformed_left,
    projected_gray_level_reference_right - projected_gray_level_deformed_right,
    projected_gray_level_deformed_left - projected_gray_level_deformed_right
]

residuals = numpy.hstack(residuals)  # Shape (total_Np,)
perfect_residual = numpy.linalg.norm(residuals, axis=0) / residuals.shape[0]  # Mean residual per integration point per view link

end_time = time.time()
print(f"Time to compute the residual for the perfect deformation: {end_time - start_time:.2f} seconds")
print(f"Residual for the perfect deformation: {perfect_residual:.4f} gray levels per integration point per view link")




# ==============================================================
# Realize displacement measurement using FESDIC
# ==============================================================

__doc__ = """

Realize displacement measurement using FESDIC to minimize the residual error between the reference and deformed images.

"""


if not os.path.exists(os.path.join(work_dir, "Results FESDIC Displacement", "DIC_measured_mesh.vtk")) or COMPUTE_DISPLACEMENT:

    def multiprocessing_full_assembly(task):
        Residual, Jacobian = assembly_FESDIC_displacement(*task)
        return Residual, Jacobian
    
    def multiprocessing_image_project(task):
        return task[0].image_project(task[1], dx=task[2])
    
    def multiprocessing_last_assembly(task):
        Residual, Jacobian = assembly_FESDIC_displacement(*task[:-2], bypass_image_projection_result_1=task[-2], bypass_image_projection_result_2=task[-1])
        return Residual, Jacobian

    DIC_mesh = reference_mesh.copy()
    DIC_mesh.set_vertices_property("displacement", 1.0 * (deformed_mesh.vertices.points - reference_mesh.vertices.points))  # Initial guess with 90 % of the real displacement

    Max_iterations = 20
    Delta_U_magnitude_threshold = 1e-2  # in mesh units (millimeters here)

    Norm_Rs = []
    Norm_JRs = []
    Norm_dUs = []

    start_time = time.time()

    for iteration in tqdm.tqdm(range(1, Max_iterations + 1), desc="FESDIC Displacement Measurement"):

        # Assemble the system
        assemble_start_time = time.time()

        sparse = True  # Use sparse matrices for the Jacobian assembly

        # task = (Mesh, Integration_Points, Displacement_Name_1, Displacement_Name_2, View_1, View_2, is_operative_1, is_operative_2, sparse)
        Tasks = [(DIC_mesh, integration_points, None, "displacement", reference_left_view, deformed_left_view, False, True, sparse),
                 (DIC_mesh, integration_points, None, "displacement", reference_left_view, deformed_right_view, False, True, sparse),
                 (DIC_mesh, integration_points, None, "displacement", reference_right_view, deformed_left_view, False, True, sparse),
                 (DIC_mesh, integration_points, None, "displacement", reference_right_view, deformed_right_view, False, True, sparse),
                 (DIC_mesh, integration_points, "displacement", "displacement", deformed_left_view, deformed_right_view, True, True, sparse)]
 
        outputs = []
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            for i in pool.imap(multiprocessing_full_assembly, Tasks):
                outputs.append(i)

        Residual = numpy.hstack([o[0] for o in outputs]) # Shape (total_Np,)
        if sparse:
            Jacobian = scipy.sparse.vstack([o[1] for o in outputs]) # Shape (total_Np, Nn * 3)
        else:
            Jacobian = numpy.vstack([o[1] for o in outputs]) # Shape (total_Np, Nn * 3)

        # Create the J.T @ J and J.T @ R matrices
        # Cost function to solve: |J @ dU - R|^2
        JTJ = Jacobian.T @ Jacobian # Shape (Nn * 3, Nn * 3)
        JTR = Jacobian.T @ Residual # Shape (Nn * 3,)

        # Search lines with only zeros in JTJ and remove them
        if sparse:
            non_zero_rows = numpy.where(JTJ.getnnz(axis=1) != 0)[0]
            JTJ_reduced = JTJ[non_zero_rows, :][:, non_zero_rows]
            JTR_reduced = JTR[non_zero_rows]    
        else:
            non_zero_rows = numpy.where(numpy.any(JTJ != 0, axis=1))[0]
            JTJ_reduced = JTJ[non_zero_rows, :][:, non_zero_rows]
            JTR_reduced = JTR[non_zero_rows]

        # Solve the linear system
        if sparse:
            Delta_U_reduced = scipy.sparse.linalg.spsolve(JTJ_reduced, JTR_reduced)  # Shape (reduced_Nn * 3,)
        else:
            Delta_U_reduced = numpy.linalg.solve(JTJ_reduced, JTR_reduced)  # Shape (reduced_Nn * 3,)

        # Rebuild the full Delta_U vector
        Delta_U = numpy.zeros(DIC_mesh.n_vertices * 3, dtype=numpy.float64)
        Delta_U[non_zero_rows] = Delta_U_reduced

        # Update the displacement field
        DIC_mesh.set_vertices_property("displacement", DIC_mesh.get_vertices_property("displacement") + Delta_U.reshape((-1, 3)))

        # Time logging
        assemble_end_time = time.time()
        print(f"Time to assemble the system: {assemble_end_time - assemble_start_time:.2f} seconds")

        # Save the displacement at the current iteration
        DIC_mesh.set_vertices_property(f"displacement_iteration_{iteration}", DIC_mesh.get_vertices_property("displacement").copy())
        DIC_mesh.set_vertices_property(f"delta_displacement_iteration_{iteration}", Delta_U.reshape((-1, 3)).copy())

        # Compute norms for monitoring
        norm_R = numpy.linalg.norm(Residual, axis=0) / Residual.shape[0]
        norm_JR = numpy.linalg.norm(JTR, axis=0) / Residual.shape[0]
        Norm_Rs.append(norm_R)
        Norm_JRs.append(norm_JR)

        # Check convergence
        Delta_U_magnitude = numpy.linalg.norm(Delta_U, axis=0)
        Norm_dUs.append(Delta_U_magnitude)

        tqdm.tqdm.write(f"{iteration}: ||R||_mean = {norm_R:.6f}, ||J.T@R||_mean = {norm_JR:.6f}, ||Delta_U|| = {Delta_U_magnitude:.6f}")

        if Delta_U_magnitude < Delta_U_magnitude_threshold:
            tqdm.tqdm.write(f"Converged after {iteration} iterations.")
            break

    end_time = time.time()
    print(f"Total time for FESDIC Displacement Measurement: {end_time - start_time:.2f} seconds")

    # Save the final displacement field
    DIC_mesh.set_vertices_property("displacement_final", DIC_mesh.get_vertices_property("displacement").copy())

    # Save the results
    os.makedirs(os.path.join(work_dir, "Results FESDIC Displacement"), exist_ok=True)
    DIC_mesh.to_vtk(os.path.join(work_dir, "Results FESDIC Displacement", "DIC_measured_mesh.vtk"), save_properties=True)
    numpy.savetxt(os.path.join(work_dir, "Results FESDIC Displacement", "norm_Rs.txt"), numpy.array(Norm_Rs))
    numpy.savetxt(os.path.join(work_dir, "Results FESDIC Displacement", "norm_JRs.txt"), numpy.array(Norm_JRs))
    numpy.savetxt(os.path.join(work_dir, "Results FESDIC Displacement", "norm_dUs.txt"), numpy.array(Norm_dUs))

else:

    DIC_mesh = LinearTriangleMesh3D.from_vtk(os.path.join(work_dir, "Results FESDIC Displacement", "DIC_measured_mesh.vtk"), load_properties=True)
    Norm_Rs = numpy.loadtxt(os.path.join(work_dir, "Results FESDIC Displacement", "norm_Rs.txt")).tolist()
    Norm_JRs = numpy.loadtxt(os.path.join(work_dir, "Results FESDIC Displacement", "norm_JRs.txt")).tolist()
    Norm_dUs = numpy.loadtxt(os.path.join(work_dir, "Results FESDIC Displacement", "norm_dUs.txt")).tolist()

# Display the residuals curves and the errors in the displacement field along X, Y and Z
if DISPLAY or True:
    errors = []
    relative_errors = []
    for iteration in range(1, len(Norm_Rs) + 1):
        measured_displacement = DIC_mesh.get_vertices_property(f"displacement_iteration_{iteration}")
        error = measured_displacement - (deformed_mesh.vertices.points - reference_mesh.vertices.points) # Shape (Nn, 3)
        error = error[usefull_vertices, :]  # Keep only usefull vertices
        errors.append(error)
        relative_error = 100 * error / (numpy.abs((deformed_mesh.vertices.points - reference_mesh.vertices.points)[usefull_vertices, :]) + 1e-6)  # Shape (Nn, 3)
        relative_errors.append(relative_error)

    mean_error_x = [numpy.abs(numpy.mean(errors[i][:, 0])) for i in range(len(errors))]
    mean_error_y = [numpy.abs(numpy.mean(errors[i][:, 1])) for i in range(len(errors))]
    mean_error_z = [numpy.abs(numpy.mean(errors[i][:, 2])) for i in range(len(errors))]
    std_error_x = [numpy.std(numpy.abs(errors[i][:, 0])) for i in range(len(errors))]
    std_error_y = [numpy.std(numpy.abs(errors[i][:, 1])) for i in range(len(errors))]
    std_error_z = [numpy.std(numpy.abs(errors[i][:, 2])) for i in range(len(errors))]
    mean_relative_error_x = [numpy.mean(numpy.abs(relative_errors[i][:, 0])) for i in range(len(relative_errors))]
    mean_relative_error_y = [numpy.mean(numpy.abs(relative_errors[i][:, 1])) for i in range(len(relative_errors))]
    mean_relative_error_z = [numpy.mean(numpy.abs(relative_errors[i][:, 2])) for i in range(len(relative_errors))]
    std_relative_error_x = [numpy.std(numpy.abs(relative_errors[i][:, 0])) for i in range(len(relative_errors))]
    std_relative_error_y = [numpy.std(numpy.abs(relative_errors[i][:, 1])) for i in range(len(relative_errors))]
    std_relative_error_z = [numpy.std(numpy.abs(relative_errors[i][:, 2])) for i in range(len(relative_errors))]

    figure = plt.figure(figsize=(12, 8))
    ax_error_X = figure.add_subplot(2, 3, 1)
    ax_error_Y = figure.add_subplot(2, 3, 2)
    ax_error_Z = figure.add_subplot(2, 3, 3)
    ax_norm_R = figure.add_subplot(2, 3, 4)
    ax_norm_JR = figure.add_subplot(2, 3, 5)
    ax_norm_dU = figure.add_subplot(2, 3, 6)

    ax_error_X.plot(range(1, len(Norm_Rs) + 1), mean_error_x, marker='o', color='b')
    ax_error_X.fill_between(range(1, len(Norm_Rs) + 1), numpy.array(mean_error_x) - numpy.array(std_error_x), numpy.array(mean_error_x) + numpy.array(std_error_x), alpha=0.2, color='b')
    ax_relative_error_X = ax_error_X.twinx()
    ax_relative_error_X.plot(range(1, len(Norm_Rs) + 1), mean_relative_error_x, marker='o', color='r')
    ax_relative_error_X.fill_between(range(1, len(Norm_Rs) + 1), numpy.array(mean_relative_error_x) - numpy.array(std_relative_error_x), numpy.array(mean_relative_error_x) + numpy.array(std_relative_error_x), alpha=0.2, color='r')
    ax_error_X.set_title("Error in Displacement X")
    ax_error_X.set_xlabel("Iteration")
    ax_error_X.set_ylabel("Error Magnitude [mm]", color='b')
    ax_relative_error_X.set_ylabel("Relative Error [%]", color='r')

    ax_error_Y.plot(range(1, len(Norm_Rs) + 1), mean_error_y, marker='o', color='b')
    ax_error_Y.fill_between(range(1, len(Norm_Rs) + 1), numpy.array(mean_error_y) - numpy.array(std_error_y), numpy.array(mean_error_y) + numpy.array(std_error_y), alpha=0.2, color='b')
    ax_relative_error_Y = ax_error_Y.twinx()
    ax_relative_error_Y.plot(range(1, len(Norm_Rs) + 1), mean_relative_error_y, marker='o', color='r')
    ax_relative_error_Y.fill_between(range(1, len(Norm_Rs) + 1), numpy.array(mean_relative_error_y) - numpy.array(std_relative_error_y), numpy.array(mean_relative_error_y) + numpy.array(std_relative_error_y), alpha=0.2, color='r')
    ax_error_Y.set_title("Error in Displacement Y")
    ax_error_Y.set_xlabel("Iteration")
    ax_error_Y.set_ylabel("Error Magnitude [mm]", color='b')
    ax_relative_error_Y.set_ylabel("Relative Error [%]", color='r')

    ax_error_Z.plot(range(1, len(Norm_Rs) + 1), mean_error_z, marker='o', color='b')
    ax_error_Z.fill_between(range(1, len(Norm_Rs) + 1), numpy.array(mean_error_z) - numpy.array(std_error_z), numpy.array(mean_error_z) + numpy.array(std_error_z), alpha=0.2, color='b')
    ax_relative_error_Z = ax_error_Z.twinx()
    ax_relative_error_Z.plot(range(1, len(Norm_Rs) + 1), mean_relative_error_z, marker='o', color='r')
    ax_relative_error_Z.fill_between(range(1, len(Norm_Rs) + 1), numpy.array(mean_relative_error_z) - numpy.array(std_relative_error_z), numpy.array(mean_relative_error_z) + numpy.array(std_relative_error_z), alpha=0.2, color='r')
    ax_error_Z.set_title("Error in Displacement Z")
    ax_error_Z.set_xlabel("Iteration")
    ax_error_Z.set_ylabel("Error Magnitude [mm]", color='b')
    ax_relative_error_Z.set_ylabel("Relative Error [%]", color='r')

    ax_norm_R.semilogy(range(1, len(Norm_Rs) + 1), Norm_Rs, marker='o')
    ax_norm_R.axhline(y=perfect_residual, color='r', linestyle='--', label='Perfect Deformation Residual')
    ax_norm_R.legend()
    ax_norm_R.set_title("Norm of Residual ||R||")
    ax_norm_R.set_xlabel("Iteration")
    ax_norm_R.set_ylabel("||R|| [gray levels / integration point / view link]")

    ax_norm_JR.semilogy(range(1, len(Norm_JRs) + 1), Norm_JRs, marker='o')
    ax_norm_JR.set_title("Norm of J.T @ R ||J.T @ R||")
    ax_norm_JR.set_xlabel("Iteration")
    ax_norm_JR.set_ylabel("||J.T @ R|| [ / integration point / view link]")

    ax_norm_dU.semilogy(range(1, len(Norm_dUs) + 1), Norm_dUs, marker='o')
    ax_norm_dU.set_title("Norm of Delta U ||Delta U||")
    ax_norm_dU.set_xlabel("Iteration")
    ax_norm_dU.set_ylabel("||Delta U|| [mm]")

    plt.tight_layout()
    plt.savefig(os.path.join(work_dir, "Results FESDIC Displacement", "FESDIC_Displacement_Convergence.png"))
    plt.show()

visualize_qt_pyvista_linear_triangle_mesh_3d(DIC_mesh, integration_points={"integration_points": integration_points})
