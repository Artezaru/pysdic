Workflows (``pysdic.workflows``)
================================

.. currentmodule:: pysdic.workflows

.. contents:: Table of Contents
   :local:
   :depth: 1

The submodule ``pysdic.workflows`` is a bank of procedures built on the API of ``pysdic``: each workflow chains the
base objects and functions (analyses, terms, cameras, meshes, ...) to solve a recurring task of a SDIC analysis. The
workflows are not part of the base API: they are called from their namespace,

.. code-block:: python

   from pysdic.workflows import create_grid_integration_points, measure_reference_shape

so that the base API stays small while new workflows can be added freely.

Conventions of a workflow:

- it only uses the public API of ``pysdic``: it can be read as an example of the API;
- it **does not modify its inputs** (in particular the analysis), except when it is its purpose and it is documented:
  the displacements computed by a workflow are returned, and written by the user with
  :meth:`pysdic.SDICAnalysis.set_displacement`;
- its results are returned as a frozen dataclass named ``<Name>Result`` when there are several outputs.

.. seealso::

   The gallery section of the workflows gives an example of each family of workflows.


Integration points
------------------

Prepare the material points of a correlation: a regular grid of points in each element, sized from the number of
pixels per element, and the selection of the points visible in all the views.

- :doc:`create_grid_integration_points <workflows/create_grid_integration_points>`: Regular grid of :math:`n \times n` integration points (midpoints of the cells) in each element (or in some elements), with the weights of the cells.
- :doc:`compute_grid_size <workflows/compute_grid_size>`: The size :math:`n` of the grid giving a target density of points per pixel, from the projected areas of the elements.
- :doc:`select_visible_integration_points <workflows/select_visible_integration_points>`: Keep the points visible in all the views of some frames (projection in the image, not hidden by the mesh).
- :doc:`VisibleIntegrationPointsResult <workflows/VisibleIntegrationPointsResult>`: Result of :func:`select_visible_integration_points` (selected points, mask and visibility per view).

.. toctree::
   :maxdepth: 1
   :hidden:

   ./workflows/create_grid_integration_points.rst
   ./workflows/compute_grid_size.rst
   ./workflows/select_visible_integration_points.rst
   ./workflows/VisibleIntegrationPointsResult.rst


Initialization of the displacements
-----------------------------------

Estimate the displacement of a frame before the correlation, which needs an initial guess close to the solution.

- :doc:`compute_optical_flow_displacement <workflows/compute_optical_flow_displacement>`: Displacement of the vertices of a frame from the optical flow between the views and the triangulation of the matched points.
- :doc:`OpticalFlowDisplacementResult <workflows/OpticalFlowDisplacementResult>`: Result of :func:`compute_optical_flow_displacement`.

.. toctree::
   :maxdepth: 1
   :hidden:

   ./workflows/compute_optical_flow_displacement.rst
   ./workflows/OpticalFlowDisplacementResult.rst


Shape at the reference frame
----------------------------

The mesh of an analysis (e.g. from a CAD model) is never exactly at the position of the specimen: its shape at the
reference frame is measured with the stereo pairings of this frame, before any measure of displacement.

- :doc:`measure_reference_shape <workflows/measure_reference_shape>`: Displacement bringing the mesh onto the observed surface, along the normals of the vertices by default (the tangential displacement is not observable with stereo pairings only).
- :doc:`ReferenceShapeResult <workflows/ReferenceShapeResult>`: Result of :func:`measure_reference_shape`.

.. toctree::
   :maxdepth: 1
   :hidden:

   ./workflows/measure_reference_shape.rst
   ./workflows/ReferenceShapeResult.rst


Tools on the displacements
--------------------------

Process the displacements of the frames of an analysis. The functions return a dictionary ``{frame: displacement}``,
the analysis is not modified.

- :doc:`compute_interpolated_displacements <workflows/compute_interpolated_displacements>`: Temporal interpolation (linear or cubic) of the displacements of frames from known frames, with a constant extrapolation.
- :doc:`compute_median_filtered_displacements <workflows/compute_median_filtered_displacements>`: Median filter of the displacements over a neighborhood of the vertices (given, by graph distance or by radius).

.. code-block:: python

   from pysdic.workflows import compute_interpolated_displacements

   interpolated = compute_interpolated_displacements(analysis, missing_frames, measured_frames)
   for frame, displacement in interpolated.items():
       analysis.set_displacement(frame, displacement)

.. toctree::
   :maxdepth: 1
   :hidden:

   ./workflows/compute_interpolated_displacements.rst
   ./workflows/compute_median_filtered_displacements.rst


Quality control
---------------

Check a measure after its convergence: the residuals per element (reflections, occlusions, damaged speckle) and the
uncertainty of the displacements.

- :doc:`compute_elements_residual_rms <workflows/compute_elements_residual_rms>`: RMS of the residuals per element and per pairing, and the badly correlated elements (robust threshold on the median and the MAD).
- :doc:`ElementsResidualRMSResult <workflows/ElementsResidualRMSResult>`: Result of :func:`compute_elements_residual_rms`.
- :doc:`compute_uncertainty <workflows/compute_uncertainty>`: Covariance of the unknowns of a correlation term, :math:`s^2 H^{-1} (J^T W J) H^{-1}`, from its Jacobian at convergence and the noise of the images.
- :doc:`UncertaintyResult <workflows/UncertaintyResult>`: Result of :func:`compute_uncertainty` (covariances and standard deviations of the vertices and of the transformations).

.. toctree::
   :maxdepth: 1
   :hidden:

   ./workflows/compute_elements_residual_rms.rst
   ./workflows/ElementsResidualRMSResult.rst
   ./workflows/compute_uncertainty.rst
   ./workflows/UncertaintyResult.rst