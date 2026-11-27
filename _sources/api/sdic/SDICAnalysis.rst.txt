pysdic.SDICAnalysis
===================

.. currentmodule:: pysdic

.. contents:: Table of Contents
   :local:
   :depth: 2

The :class:`pysdic.SDICAnalysis` stores the data of a stereo digital image correlation analysis: a reference mesh,
the views (camera and image) acquired at each frame, the displacement of the vertices at each frame and,
optionally, the acquisition times of the frames.

.. code-block:: text

   SDICAnalysis
   ├── mesh                     (reference configuration, not copied but locked)
   ├── times                    (N_f,) strictly increasing, or None
   ├── frame 0
   │   ├── displacement         (N_v, E), read-only (not set by default)
   │   └── views                [View 0, View 1, ...] (possibly empty)
   ├── frame 1
   │   └── ...
   └── frame N_f - 1

The views are linked by :class:`pysdic.Pairing` objects, used by the workflows and the terms of the analysis
(see :class:`pysdic.SDICCorrelationTerm`). The unknowns of an optimization (displacements of some frames, parameters
of some transformations) are described by a :class:`pysdic.SDICUnknowns`, validated by the analysis, which builds the
vector of unknowns (see `Unknowns`_): its displacements are always **total** displacements from the reference mesh.


SDICAnalysis class
------------------

.. autoclass:: SDICAnalysis
   :no-members:
   :show-inheritance:


Instantiation
-------------

The :class:`SDICAnalysis` class is instantiated with its constructor
``SDICAnalysis(mesh, views, *, times=None, displacements=None)``:

- ``mesh``: the reference mesh, with an element type. It is **not copied but locked** (see below).
- ``views``: the views of each frame, ``views[f]`` being the list of the views of the frame ``f`` (a frame can
  have no view). The number of frames is ``len(views)``.
- ``times``: the acquisition times of the frames, strictly increasing (required by
  :class:`pysdic.SDICTemporalRegularizationTerm`), or None.
- ``displacements``: the initial displacements, a dictionary ``{frame: (N_v, E) array}``. The other frames have
  no displacement.

.. code-block:: python

   from pysdic import SDICAnalysis, View

   views = [[View(left, images[f, 0]), View(right, images[f, 1])] for f in range(2)]
   analysis = SDICAnalysis(mesh, views, times=[0.0, 0.1])
   analysis.set_zero_displacement(0)

The usual acquisition setups can be built directly with the constructors functions
(see :doc:`SDIC analysis <../sdic>`):

- :func:`pysdic.create_multicamera_analysis` and :func:`pysdic.create_bicamera_analysis`: one series of images per camera.
- :func:`pysdic.create_splitscreen_analysis`: several virtual cameras sharing one sensor and one series of images.
- :func:`pysdic.create_cordin_analysis`: several points of view on one sensor with a distortion changing with the frame.

.. important::

   The structure of the analysis (mesh, views, times) is fixed at its creation: the structures of the shared
   objects are computed once and checked by :meth:`SDICAnalysis.check_consistency`.

   - The mesh is locked: its vertices and its connectivity cannot be modified (its properties can). Work on a copy
     (``mesh.copy()``) and use :meth:`SDICAnalysis.with_mesh` to analyse another mesh. A mesh already locked (e.g. by
     another analysis) is accepted and shared.
   - The ``with_*`` methods create a new analysis sharing everything else (images, cameras, displacements).
   - The displacements and the parameters of the transformations remain modifiable.
   - The terms and parametrizations built on an analysis keep this analysis: an analysis created with a ``with_*``
     method needs new terms (the same :class:`pysdic.SDICUnknowns` can be reused).


Implemented methods and attributes
----------------------------------

The following sections provide an overview of the attributes and methods defined
in the :class:`pysdic.SDICAnalysis` class.


Attributes
^^^^^^^^^^

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   SDICAnalysis.mesh
   SDICAnalysis.n_frames
   SDICAnalysis.times


New analyses
^^^^^^^^^^^^

A new analysis with another mesh, other views or other times. The other objects are shared (not copied), and the
displacements are kept when it is possible (``keep_displacements``).

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   SDICAnalysis.with_mesh
   SDICAnalysis.with_views
   SDICAnalysis.with_frame_views
   SDICAnalysis.with_times


Displacements
^^^^^^^^^^^^^

The displacements are stored in the analysis (not in the mesh), one read-only array of shape :math:`(N_v, E)` per frame.
A frame without displacement raises an error when its displacement is used, and a displacement cannot be removed
once set (``set_displacement(frame, None)`` raises an error).

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   SDICAnalysis.set_displacement
   SDICAnalysis.set_zero_displacement
   SDICAnalysis.has_displacement
   SDICAnalysis.get_displacement
   SDICAnalysis.get_deformed_coordinates
   SDICAnalysis.get_deformed_mesh


Views
^^^^^

The views of a frame are indexed in the order of the construction. The views are not copied, and the input
dimension of their camera must be the dimension of the mesh.

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   SDICAnalysis.get_view
   SDICAnalysis.get_views
   SDICAnalysis.n_views


Pairings
^^^^^^^^

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   SDICAnalysis.validate_pairing


Unknowns
^^^^^^^^

The unknowns of an optimization are described by a :class:`pysdic.SDICUnknowns`, which does not know the analysis
(as a :class:`pysdic.Pairing`). The analysis validates it and builds the vector of unknowns
:math:`\vec{p} = [\vec{p}_U, \vec{p}_T]` from its current state: the **total** displacements of the optimized frames
(Fortran order) and the parameters of the optimized transformations.

.. code-block:: python

   unknowns = SDICUnknowns([1], [("extrinsic", 1)])
   analysis.validate_unknowns(unknowns)

   p0 = analysis.pack_unknowns(unknowns)                                 # Current state -> p
   displacements, parameters = analysis.unpack_unknowns(unknowns, p)     # p -> copies
   analysis.apply_unknowns(unknowns, result.parameters)                  # p -> analysis (replaced, not added)

.. warning::

   With a parametrization, the solver works on a reduced vector :math:`\vec{q}`: use
   ``parametrization.reparametrized_pack/unpack/apply`` (see :class:`pysdic.SDICFixedParametrization`).

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   SDICAnalysis.validate_unknowns
   SDICAnalysis.n_unknowns
   SDICAnalysis.pack_unknowns
   SDICAnalysis.unpack_unknowns
   SDICAnalysis.apply_unknowns
   SDICAnalysis.get_unknowns_frame_slice
   SDICAnalysis.get_unknowns_transformation_slice
   SDICAnalysis.compute_unknowns_indices


Structure of the analysis
^^^^^^^^^^^^^^^^^^^^^^^^^

Several views can share the same image, camera or transformation of camera. The structure of a kind of object
(``"view"``, ``"image"``, ``"camera"``, ``"extrinsic"``, ``"distortion"`` or ``"intrinsic"``) gives, for each view,
the index of its object. The objects are compared by identity and numbered in the order of their first occurrence
along the frames and the views.

.. code-block:: python

   analysis.display_structure("extrinsic")
   # Structure of the extrinsics (index of the shared extrinsic of each view):
   #   Frame 0  : [0, 1]
   #   Frame 1  : [0, 1]
   p = analysis.get_transformation_parameters("extrinsic", 1)
   analysis.set_transformation_parameters("extrinsic", 1, p)  # All the views using the extrinsic 1

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   SDICAnalysis.compute_structure
   SDICAnalysis.display_structure
   SDICAnalysis.get_transformation
   SDICAnalysis.get_transformation_parameters
   SDICAnalysis.set_transformation_parameters
   SDICAnalysis.check_consistency

.. note::

   Each transformation ``(kind, index)`` of the structure owns its own parameters. :meth:`SDICAnalysis.check_consistency`
   (called at the creation) checks that:

   - the extrinsics, the distortions and the intrinsics are disjoint sets (an object cannot be used as two kinds);
   - each unit transformation appears only once: the composed and reparametrized transformations are explored down
     to their unit transformations, and an error is raised if two entries (or two steps of the same entry) share
     a unit transformation (their parameters would be coupled without being declared as one unknown).


Save and load
^^^^^^^^^^^^^

An analysis is saved in a folder (or a zip archive of the same files): the mesh, each image and each transformation
once (even if shared), the displacements, the times and the structure of the analysis, with a SHA-256 manifest. No
``pickle`` is used. The writing is atomic (temporary folder renamed at the end), and ``overwrite=True`` only replaces
a previous save of an analysis. At the loading, the integrity of the files is checked (``check_integrity=True``),
then the summary of the metadata (numbers of frames, vertices, elements, views, images, cameras and transformations,
frames with a displacement, times) and the structures of the loaded analysis are compared with the saved ones
("Inconsistent loading" error if they differ).

.. code-block:: text

   folder/
   ├── metadata.json        format, pysdic version, date, title, description, summary
   ├── README.txt           the same information, readable by a human
   ├── manifest.json        SHA-256 of each file
   ├── structure.json       cameras, views, frames, structures
   ├── mesh/mesh.npz        (+ mesh.vtk with export_vtk=True, never read)
   ├── images/image_00.npz ...
   ├── extrinsics/, distortions/, intrinsics/   one JSON file per transformation
   ├── displacements.npz    frame_00, ... (only the frames with a displacement)
   └── times.npz            (only if the times are set)

.. code-block:: python

   analysis.to_folder("results/test_42", title="Tensile test 42")
   same_analysis = SDICAnalysis.from_folder("results/test_42")

.. autosummary::
   :toctree: ./_generated/
   :signatures: long

   SDICAnalysis.to_folder
   SDICAnalysis.from_folder
   SDICAnalysis.to_archive
   SDICAnalysis.from_archive

.. note::

   The integration points and the pairings are not part of the analysis: save them separately
   (e.g. :meth:`pysdic.IntegrationPoints.to_npz_file`).

Saves of previous versions
""""""""""""""""""""""""""

The compatibility of a save is given by the version of pysdic that wrote it (``pysdic_version`` in
``metadata.json``):

- **2.1.0 and later**: the current format, loaded directly.
- **0.0.1 to 2.1.0** (including the saves of the former ``pysdic.SDIC`` class): converted to the current format
  while loading, with a :class:`pysdic.PreviousSaveWarning` displayed in the console. The files are not modified:
  save the loaded analysis again (``to_folder`` or ``to_archive``) to update the save and remove the warning.
- **Before 0.0.1**: deprecated, a :class:`pysdic.DeprecatedError` is raised (a subclass of ``ValueError``).

.. code-block:: python

   import warnings
   from pysdic import PreviousSaveWarning, SDICAnalysis

   analysis = SDICAnalysis.from_folder("results/old_test")   # PreviousSaveWarning
   analysis.to_folder("results/old_test", overwrite=True)     # Updated: no more warning

   warnings.simplefilter("error", PreviousSaveWarning)        # Or: refuse the saves of previous versions

.. autosummary::
   :toctree: ./_generated/

   PreviousSaveWarning
   DeprecatedError