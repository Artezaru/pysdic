# Architecture of the `pysdic` package

This file gives the content of each sub-folder of the package, to put a new file in the
right place. The details of the classes and functions are in the documentation.

## Sub-folders

| Folder | Content |
|---|---|
| `core` | Internal building blocks: serialization, lock, property container. |
| `resources` | Data shipped with the package (example textures). |
| `parametrizations` | Maps between parameter vectors, with their Jacobians. |
| `transformations` | Camera models (extrinsic, distortion, intrinsic), composition and reparametrization. |
| `transformation_optimizations` | Fit of the parameters or input points of transformations with `pysolvegn`. |
| `computer_vision` | Functions on raw arrays: projection, rays, distortion, optical flow. |
| `fem` | Finite element computations on raw arrays (shape functions, adjacency, ...). |
| `numeric` | Internal numerical tools: finite differences, polynomial fit. |
| `geometry` | Point clouds, meshes, integration points, mesh creation, 3D display. |
| `imaging` | Images, cameras, views, 2D display, and the functions combining a mesh and a camera. |
| `synthetic` | Speckle textures, synthetic images and noise. |
| `sdic` | The SDIC analysis, its setups, the unknowns, the terms and the residuals. |
| `workflows` | Bank of procedures built on the API (`pysdic.workflows.<function>`). |
| `cordin` | Reading of the CORDIN high-speed camera captures (`pysdic.cordin.<name>`). |

The files `_validation.py` (validation of the inputs) and `_optionals.py` (lazy import of
the optional dependencies) are shared by all the modules.