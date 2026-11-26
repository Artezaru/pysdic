"""
pysdic - Python Stereo-Digital Image Correlation (Computer Vision
Cameras Models, Image Processing, Meshes and tools for SDIC analysis)
Copyright (C) 2026 Artezaru, artezaru.github@proton.me

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

r"""
Workflows of ``pysdic``: a bank of procedures built on the API of the package.

A workflow chains the base objects and functions of ``pysdic`` (analyses, terms, cameras,
meshes, ...) to solve a recurring task, e.g. the initialization of a displacement by optical
flow. The workflows are not part of the base API: they are called from this namespace,

.. code-block:: python

    from pysdic.workflows import compute_optical_flow_displacement

    result = compute_optical_flow_displacement(analysis, 0, 0, 1, (0, 1))

so that the base API stays small while new workflows can be added freely.

Conventions of a workflow:

- one module per workflow (or per family of workflows) in ``pysdic/workflows/``, exported
  below and in ``__all__``;
- it only uses the public API of ``pysdic`` (and the shared validation helpers of
  ``pysdic._validation``), not the private functions of the other modules, so that it does
  not break when their internals change;
- it does not modify its inputs, except when it is its purpose and it is documented (e.g. a
  workflow writing displacements in an analysis);
- its results are returned as a frozen dataclass named ``<Name>Result`` when there are
  several outputs.
"""

from .integration_points import (
    VisibleIntegrationPointsResult,
    compute_grid_size,
    create_grid_integration_points,
    select_visible_integration_points,
)
from .initialisation import (
    OpticalFlowDisplacementResult,
    compute_optical_flow_displacement,
)
from .reference_shape import (
    ReferenceShapeResult,
    measure_reference_shape,
)
from .displacement_tools import (
    compute_interpolated_displacements,
    compute_median_filtered_displacements,
)
from .quality import (
    ElementsResidualRMSResult,
    UncertaintyResult,
    compute_elements_residual_rms,
    compute_uncertainty,
)

__all__ = [
    # Integration points
    "VisibleIntegrationPointsResult",
    "compute_grid_size",
    "create_grid_integration_points",
    "select_visible_integration_points",
    # Initialization of the displacements
    "OpticalFlowDisplacementResult",
    "compute_optical_flow_displacement",
    # Shape at the reference frame
    "ReferenceShapeResult",
    "measure_reference_shape",
    # Tools on the displacements
    "compute_interpolated_displacements",
    "compute_median_filtered_displacements",
    # Quality control
    "ElementsResidualRMSResult",
    "UncertaintyResult",
    "compute_elements_residual_rms",
    "compute_uncertainty",
]