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

from __future__ import annotations

import datetime
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy
from numpy.typing import ArrayLike

from ..__version__ import __version__
from ..geometry.mesh import Mesh
from ..imaging.camera import Camera
from ..imaging.image import Image
from ..imaging.view import View
from ..transformations.transformation import ComposedTransformation, Transformation
from ..transformations.reparametrized_transformation import ReparametrizedTransformation
from .._validation import _as_array, _validate_bool, _validate_integer, _validate_path
from ._load_previous_sdic_analysis import PREVIOUS_FORMAT_NAMES, load_previous_save, parse_version
from .pairing import Pairing
from .sdic_unknowns import SDICUnknowns, _as_component_selection, _as_vertex_selection


# Objects whose sharing between the views can be displayed (see SDICAnalysis.compute_structure)
_STRUCTURE_KINDS = {
    "view": lambda view: view,
    "image": lambda view: view.image,
    "camera": lambda view: view.camera,
    "extrinsic": lambda view: view.camera.extrinsic,
    "distortion": lambda view: view.camera.distortion,
    "intrinsic": lambda view: view.camera.intrinsic,
}
_TRANSFORMATION_KINDS = ("extrinsic", "distortion", "intrinsic")

# Reason of the lock of the mesh of an SDIC analysis (added after "Cannot ...: this Mesh is locked.")
_MESH_LOCK_MESSAGE = (
    "It is used by an SDIC analysis, whose operators (shape functions at the integration "
    "points, neighborhoods of the gray level corrections, unknowns of the displacements) "
    "assume that its vertices and elements never change. The displacements of the analysis "
    "are stored in the analysis (analysis.set_displacement), not in the mesh: the mesh is the "
    "reference configuration. To use a modified mesh, work on a copy (mesh.copy()) and create "
    "a new analysis with it (analysis.with_mesh(new_mesh)). The properties of the mesh "
    "(mesh.vertices['name'] = ...) can still be modified."
)


# Sentinel of SDICAnalysis.with_views: keep the current times
_KEEP = object()

# Format of the saves written by SDICAnalysis.to_folder (see SDICAnalysis.from_folder). The saves
# written by the versions in [_MIN_VERSION, _MAX_VERSION) (None: no bound) have the current format;
# the older ones are converted while loading (see _load_previous_sdic_analysis.py, whose last
# migration must end at _MIN_VERSION).
_FORMAT_NAME = "pysdic.SDICAnalysis"
_MIN_VERSION = "2.1.0"
_MAX_VERSION: Optional[str] = None

# Files of the atoms of a save: kind -> (folder, extension)
_ATOM_FILES = {
    "image": ("images", "npz"),
    "extrinsic": ("extrinsics", "json"),
    "distortion": ("distortions", "json"),
    "intrinsic": ("intrinsics", "json"),
}


def _index_width(n: int) -> int:
    # Number of digits of the largest index n - 1 (at least 1): 1543 frames -> 4 digits
    return max(1, len(str(n - 1)))


def _write_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_arrays(path: Path) -> Dict[str, numpy.ndarray]:
    # Numeric arrays of an NPZ file, without pickle
    with numpy.load(path, allow_pickle=False) as npz:
        return {key: npz[key] for key in npz.files}


def _read_metadata(folder: Path) -> Optional[Dict[str, Any]]:
    # The metadata of a SDIC save (current or previous format), or None if the folder is not a SDIC save
    try:
        metadata = _read_json(folder / "metadata.json")
    except (OSError, ValueError):
        return None
    if isinstance(metadata, dict) and metadata.get("format") in (_FORMAT_NAME, *PREVIOUS_FORMAT_NAMES):
        return metadata
    return None


def _check_version(folder: Path, metadata: Dict[str, Any]) -> Tuple[int, ...]:
    # The parsed version of pysdic that wrote the save, which must be before _MAX_VERSION
    version = metadata.get("pysdic_version")
    try:
        saved = parse_version(version)
    except ValueError:
        raise ValueError(f"The SDIC save '{folder}' has an invalid pysdic version {version!r}.") from None
    if _MAX_VERSION is not None and saved >= parse_version(_MAX_VERSION):
        raise ValueError(
            f"The SDIC save '{folder}' was written by pysdic {version}, but this version of pysdic "
            f"({__version__}) loads the saves written by the versions before {_MAX_VERSION}."
        )
    return saved


def _sha256(path: Path) -> str:
    # SHA-256 of a file, read by chunks
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_manifest(folder: Path) -> None:
    # Check the SHA-256 of all the files listed in the manifest
    manifest_path = folder / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(
            f"The SDIC save '{folder}' has no manifest.json: its integrity cannot be checked. "
            f"Use check_integrity=False to load it anyway."
        )
    missing, modified = [], []
    for relative, expected in _read_json(manifest_path).get("files", {}).items():
        path = folder / relative
        if not path.is_file():
            missing.append(relative)
        elif _sha256(path) != expected:
            modified.append(relative)
    if missing or modified:
        raise ValueError(
            f"The SDIC save '{folder}' is corrupted or was modified after its creation: "
            f"missing files {missing}, modified files {modified} (SHA-256 of manifest.json). "
            f"Use check_integrity=False to load it anyway, at your own risk."
        )


def _leaf_transformations(transformation: Transformation) -> List[Transformation]:
    # The leaf transformations of a transformation, recursively: the transformations of a
    # composition, the transformation reparametrized by a ReparametrizedTransformation, or the
    # transformation itself otherwise
    if isinstance(transformation, ComposedTransformation):
        return [leaf for inner in transformation.transformations for leaf in _leaf_transformations(inner)]
    if isinstance(transformation, ReparametrizedTransformation):
        return _leaf_transformations(transformation.transformation)
    return [transformation]


class SDICAnalysis(object):
    r"""
    The data of a stereo digital image correlation (SDIC) analysis: a reference mesh,
    the views (camera and image) acquired at each frame and the displacement of the
    vertices of the mesh at each frame.

    - The **mesh** is the reference (undeformed) configuration. It is not copied but
      **locked** (see :class:`pysdic.core.lockable.LockableMixin`): its vertices and
      elements cannot be modified while it is used by the analysis. A mesh that is already
      locked (for example by another analysis) is accepted and shared.
    - The **views** of each frame are given at the creation, in their index order. Several
      views can share the same camera or the same image (see :meth:`display_structure`).
      A frame can have no view (for example a frame only used by a temporal
      regularization).
    - The **displacements** are stored in the analysis (not in the mesh), one array of shape
      :math:`(N_v, E)` per frame. A frame without displacement raises an error when its
      displacement is used. Once set, a displacement can be replaced but not removed.
    - The **times** of acquisition of the frames (optional) are strictly increasing with the
      frame index: the frames are in chronological order. They are needed by the temporal
      operators (see :class:`pysdic.SDICTemporalRegularizationTerm`).

    .. important::

        The **structure** of the analysis (mesh, number of frames, views) is fixed at the
        creation: the views, cameras and images are immutable, and the mesh is locked.
        Only the displacements (:meth:`set_displacement`) and the parameters of the
        transformations of the cameras (:meth:`set_transformation_parameters`) can be
        modified. Use :meth:`with_mesh`, :meth:`with_views`, :meth:`with_frame_views` and
        :meth:`with_times` to get a new analysis with another mesh, other views or other
        times: the new analysis shares the other data (no copy).

    The views are linked by :class:`pysdic.Pairing` objects, used by the workflows and the
    operators of the analysis. The unknowns of an optimization are described by a
    :class:`pysdic.SDICUnknowns` (which does not know the analysis): the analysis validates it
    (:meth:`validate_unknowns`) and builds the vector of unknowns from its current state
    (:meth:`pack_unknowns`, :meth:`unpack_unknowns`, :meth:`apply_unknowns`).

    Parameters
    ----------
    mesh : :class:`pysdic.Mesh`
        The reference mesh, with an element type. It is not copied: it is locked.

    views : Sequence[Sequence[:class:`pysdic.View`]]
        The views of each frame, ``views[f]`` being the views of the frame :math:`f` in
        their index order (possibly empty). The number of frames :math:`N_f` is
        ``len(views)`` (at least 1). The views are not copied (they are immutable). Their
        cameras must have the dimension of the mesh as input dimension.

    times : Optional[ArrayLike], optional (default: ``None``)
        The times of acquisition of the frames, of shape :math:`(N_f,)`, finite and strictly
        increasing (copied), in any unit. If None, the times are not defined (the temporal
        operators raise an error).

    displacements : Optional[Mapping[Integral, ArrayLike]], optional (default: ``None``)
        The initial displacements, as a dictionary ``{frame: displacement}`` with arrays of
        shape :math:`(N_v, E)` (copied). The other frames have no displacement. Equivalent
        to calling :meth:`set_displacement` for each frame.

    Raises
    ------
    TypeError
        If an input has a wrong type.

    ValueError
        If the mesh has no element type, there is no frame, the input dimension of a
        camera is not the dimension of the mesh, the times are invalid, a displacement is
        invalid, or the transformations of the cameras are not independent (see
        :meth:`check_consistency`).

    IndexError
        If a frame of ``displacements`` is out of range.

    Examples
    --------
    .. code-block:: python

        from pysdic import SDICAnalysis, View

        views = [
            [View(left_camera, left_images[f]), View(right_camera, right_images[f])]
            for f in range(n_frames)
        ]
        analysis = SDICAnalysis(mesh, views, displacements={0: numpy.zeros((mesh.n_vertices, 3))})

        mesh.coordinates = ...   # RuntimeError: the mesh is used by the analysis

    """

    __slots__ = ("_mesh", "_n_frames", "_displacements", "_views", "_structures", "_times")

    def __init__(
        self,
        mesh: Mesh,
        views: Sequence[Sequence[View]],
        *,
        times: Optional[ArrayLike] = None,
        displacements: Optional[Mapping[Integral, ArrayLike]] = None,
    ) -> None:
        mesh = self._validate_mesh(mesh)
        views = self._as_views(views, mesh)
        n_frames = len(views)
        times = self._as_times(times, n_frames)

        frame_displacements: List[Optional[numpy.ndarray]] = [None] * n_frames
        if displacements is not None:
            if not isinstance(displacements, Mapping):
                raise TypeError(
                    f"Input 'displacements' must be a dictionary {{frame: displacement}} or None. "
                    f"Got type {type(displacements)}."
                )
            for frame, displacement in displacements.items():
                frame = _validate_integer(frame, "frame of 'displacements'", min_value=0)
                if frame >= n_frames:
                    raise IndexError(
                        f"The frames of 'displacements' must be in [0, {n_frames - 1}]. Got {frame}."
                    )
                frame_displacements[frame] = self._as_displacement(displacement, mesh)

        self._init_shared(mesh, views, frame_displacements, times)

    def _init_shared(
        self,
        mesh: Mesh,
        views: Tuple[Tuple[View, ...], ...],
        displacements: List[Optional[numpy.ndarray]],
        times: Optional[numpy.ndarray],
    ) -> None:
        # Set the validated data (the displacements and the times are read-only arrays, possibly
        # shared with another analysis), compute the structures and check their consistency. The
        # mesh is locked last: a failed creation leaves it unchanged.
        self._mesh = mesh
        self._views = views
        self._n_frames = len(views)
        self._displacements = list(displacements)
        self._times = times
        self._structures = {kind: self._compute_structure(kind) for kind in _STRUCTURE_KINDS}
        self.check_consistency()
        if not mesh.is_locked:
            mesh._set_locked(True, message=_MESH_LOCK_MESSAGE)

    @classmethod
    def _from_shared(
        cls,
        mesh: Mesh,
        views: Tuple[Tuple[View, ...], ...],
        displacements: List[Optional[numpy.ndarray]],
        times: Optional[numpy.ndarray],
    ) -> SDICAnalysis:
        # New analysis from validated data, shared with another analysis without copy (the
        # mesh is locked, the views are immutable, the displacements and times are read-only)
        new = cls.__new__(cls)
        new._init_shared(mesh, views, displacements, times)
        return new

    def __repr__(self) -> str:
        return (
            f"SDICAnalysis(n_frames={self._n_frames}, n_vertices={self._mesh.n_vertices}, "
            f"n_elements={self._mesh.n_elements}, n_views={[len(views) for views in self._views]})"
        )

    # ===================================================================
    # Internal checks
    # ===================================================================
    @staticmethod
    def _validate_mesh(mesh: Mesh) -> Mesh:
        if not isinstance(mesh, Mesh):
            raise TypeError(f"Input 'mesh' must be a Mesh. Got type {type(mesh)}.")
        if mesh.element_type is None:
            raise ValueError(
                "The mesh of a SDIC analysis must have an element type (the shape functions "
                "of the elements are needed). Set it with mesh.element_type = ... before "
                "creating the analysis."
            )
        return mesh

    @staticmethod
    def _as_frame_views(views: Sequence[View], mesh: Mesh, name: str) -> Tuple[View, ...]:
        # The views of one frame (possibly empty), checked against the mesh
        if not isinstance(views, (list, tuple)):
            raise TypeError(f"Input '{name}' must be a list or a tuple of View. Got type {type(views)}.")
        for index, view in enumerate(views):
            if not isinstance(view, View):
                raise TypeError(f"Input '{name}[{index}]' must be a View. Got type {type(view)}.")
            if view.camera.input_dim != mesh.n_dimensions:
                raise ValueError(
                    f"The camera of the view '{name}[{index}]' must have the dimension of the "
                    f"mesh as input dimension ({mesh.n_dimensions}). Got {view.camera.input_dim}."
                )
        return tuple(views)

    @classmethod
    def _as_views(cls, views: Sequence[Sequence[View]], mesh: Mesh) -> Tuple[Tuple[View, ...], ...]:
        # The views of all the frames (at least one frame)
        if not isinstance(views, (list, tuple)):
            raise TypeError(
                f"Input 'views' must be a list or a tuple with the views of each frame. "
                f"Got type {type(views)}."
            )
        if len(views) == 0:
            raise ValueError("Input 'views' must contain at least one frame (possibly without views).")
        return tuple(
            cls._as_frame_views(frame_views, mesh, f"views[{frame}]")
            for frame, frame_views in enumerate(views)
        )

    @staticmethod
    def _as_times(times: Optional[ArrayLike], n_frames: int) -> Optional[numpy.ndarray]:
        # Read-only copy of finite and strictly increasing times of shape (N_f,), or None
        if times is None:
            return None
        times = _as_array(
            times, "times", numpy.float64, shape=(n_frames,), shape_hint=f"({n_frames},)",
            finite=True, copy=True,
        )
        decreasing = numpy.flatnonzero(numpy.diff(times) <= 0)
        if decreasing.size > 0:
            frame = int(decreasing[0])
            raise ValueError(
                f"Input 'times' must be strictly increasing (the frames are in chronological "
                f"order). Got times[{frame}] = {times[frame]} and times[{frame + 1}] = "
                f"{times[frame + 1]}."
            )
        times.flags.writeable = False
        return times

    @staticmethod
    def _as_displacement(displacement: ArrayLike, mesh: Mesh) -> numpy.ndarray:
        # Read-only copy of a displacement of shape (N_v, E)
        if displacement is None:
            raise TypeError(
                "A displacement must be an array of shape (N_v, E). Got None: the displacement "
                "of a frame cannot be removed once set (the operators built on the analysis "
                "rely on it). Use set_zero_displacement for a zero displacement."
            )
        displacement = _as_array(
            displacement,
            "displacement",
            numpy.float64,
            shape=(mesh.n_vertices, mesh.n_dimensions),
            finite=True,
            copy=True,
        )
        displacement.flags.writeable = False
        return displacement

    def _validate_frame(self, frame: Integral, name: str = "frame") -> int:
        frame = _validate_integer(frame, name, min_value=0)
        if frame >= self._n_frames:
            raise IndexError(f"Input '{name}' must be in [0, {self._n_frames - 1}]. Got {frame}.")
        return frame

    def _validate_view_index(self, frame: int, index: Integral, name: str = "index") -> int:
        index = _validate_integer(index, name, min_value=0)
        if index >= len(self._views[frame]):
            raise IndexError(
                f"Input '{name}' must be in [0, {len(self._views[frame]) - 1}] for the frame {frame} "
                f"({len(self._views[frame])} views). Got {index}."
            )
        return index

    # ===================================================================
    # New analyses sharing the data
    # ===================================================================
    def with_mesh(self, mesh: Mesh, *, keep_displacements: bool = True) -> SDICAnalysis:
        r"""
        Create a new analysis with another mesh and the same views.

        The new analysis shares the views and (if kept) the displacements with this
        analysis, without copy. The new mesh is locked; the mesh of this analysis stays
        locked (it is still used by this analysis).

        Parameters
        ----------
        mesh : :class:`pysdic.Mesh`
            The new reference mesh, with an element type (not copied: it is locked).

        keep_displacements : bool, optional (default: ``True``)
            If True, the displacements of this analysis are kept: the new mesh must then
            have the same number of vertices and dimension, and the vertex :math:`i` of
            the new mesh gets the displacement of the vertex :math:`i` of the current one.
            If False, the new analysis has no displacement.

        Returns
        -------
        :class:`pysdic.SDICAnalysis`
            The new analysis.

        Raises
        ------
        ValueError
            If the new mesh has no element type, if its dimension does not match the
            cameras, or if the displacements are kept and the shape :math:`(N_v, E)` of the
            new mesh differs from the current one while a displacement is set.

        Examples
        --------
        Refine the analysis with a modified mesh (the current mesh is locked):

        .. code-block:: python

            new_mesh = analysis.mesh.copy()
            new_mesh.coordinates = smoothed_coordinates
            new_sdic = analysis.with_mesh(new_mesh)

        """
        mesh = self._validate_mesh(mesh)
        keep_displacements = _validate_bool(keep_displacements, "keep_displacements")
        views = self._as_views(self._views, mesh)  # Check the dimension of the cameras

        if not keep_displacements:
            return self._from_shared(mesh, views, [None] * self._n_frames, self._times)

        new_shape = (mesh.n_vertices, mesh.n_dimensions)
        old_shape = (self._mesh.n_vertices, self._mesh.n_dimensions)
        set_frames = [frame for frame, u in enumerate(self._displacements) if u is not None]
        if set_frames and new_shape != old_shape:
            raise ValueError(
                f"Cannot keep the displacements of the frames {set_frames}: they have the shape "
                f"(N_v, E) = {old_shape} of the current mesh, but the new mesh has the shape "
                f"{new_shape}. Use keep_displacements=False and set the displacements of the new "
                f"analysis (for example by interpolating the current ones on the new vertices)."
            )
        return self._from_shared(mesh, views, self._displacements, self._times)

    def with_views(
        self,
        views: Sequence[Sequence[View]],
        *,
        times: Optional[ArrayLike] = _KEEP,
        keep_displacements: bool = True,
    ) -> SDICAnalysis:
        r"""
        Create a new analysis with the same mesh and other views for all the frames.

        The new analysis shares the mesh and (if kept) the displacements with this
        analysis, without copy.

        Parameters
        ----------
        views : Sequence[Sequence[:class:`pysdic.View`]]
            The views of each frame of the new analysis (see :class:`pysdic.SDICAnalysis`). The
            number of frames can differ from this analysis if ``keep_displacements`` is
            False.

        times : Optional[ArrayLike], optional
            The times of the frames of the new analysis (see :class:`pysdic.SDICAnalysis`), or None
            for no times. By default, the times of this analysis are kept: the number of
            frames must then be the same if the times are defined.

        keep_displacements : bool, optional (default: ``True``)
            If True, the displacements of this analysis are kept (the frame :math:`f` of
            the new analysis gets the displacement of the frame :math:`f`): the number of
            frames must then be the same. If False, the new analysis has no displacement.

        Returns
        -------
        :class:`pysdic.SDICAnalysis`
            The new analysis.

        Raises
        ------
        ValueError
            If the views or the times are invalid (see :class:`pysdic.SDICAnalysis`), or if the
            displacements or the times are kept and the number of frames differs.

        """
        keep_displacements = _validate_bool(keep_displacements, "keep_displacements")
        views = self._as_views(views, self._mesh)
        if times is _KEEP:
            if self._times is not None and len(views) != self._n_frames:
                raise ValueError(
                    f"Cannot keep the times: the new views have {len(views)} frames while the "
                    f"analysis has {self._n_frames} frames. Give the times of the new frames "
                    f"with times=..., or times=None."
                )
            times = self._times
        else:
            times = self._as_times(times, len(views))
        if not keep_displacements:
            return self._from_shared(self._mesh, views, [None] * len(views), times)
        if len(views) != self._n_frames:
            raise ValueError(
                f"Cannot keep the displacements: the new views have {len(views)} frames while "
                f"the analysis has {self._n_frames} frames. Give the views of {self._n_frames} "
                f"frames, or use keep_displacements=False."
            )
        return self._from_shared(self._mesh, views, self._displacements, times)

    def with_frame_views(self, frame: Integral, views: Sequence[View]) -> SDICAnalysis:
        r"""
        Create a new analysis where the views of one frame are replaced.

        The new analysis shares the mesh, the views of the other frames and the
        displacements with this analysis, without copy.

        Parameters
        ----------
        frame : Integral
            The index of the frame, in :math:`[0, N_f)`.

        views : Sequence[:class:`pysdic.View`]
            The new views of the frame, in their index order (possibly empty).

        Returns
        -------
        :class:`pysdic.SDICAnalysis`
            The new analysis.

        Raises
        ------
        IndexError
            If ``frame`` is out of range.

        ValueError
            If a view is invalid (see :class:`pysdic.SDICAnalysis`).

        Examples
        --------
        Replace the image of the second view of the frame 3:

        .. code-block:: python

            views = list(analysis.get_views(3))
            views[1] = View(views[1].camera, new_image)
            new_sdic = analysis.with_frame_views(3, views)

        """
        frame = self._validate_frame(frame)
        frame_views = self._as_frame_views(views, self._mesh, "views")
        all_views = self._views[:frame] + (frame_views,) + self._views[frame + 1 :]
        return self._from_shared(self._mesh, all_views, self._displacements, self._times)

    def with_times(self, times: Optional[ArrayLike]) -> SDICAnalysis:
        r"""
        Create a new analysis with other times of acquisition of the frames.

        The new analysis shares the mesh, the views and the displacements with this
        analysis, without copy.

        Parameters
        ----------
        times : Optional[ArrayLike]
            The times of the frames, of shape :math:`(N_f,)`, finite and strictly
            increasing, or None for no times.

        Returns
        -------
        :class:`pysdic.SDICAnalysis`
            The new analysis.

        Raises
        ------
        ValueError
            If the times are invalid.

        """
        times = self._as_times(times, self._n_frames)
        return self._from_shared(self._mesh, self._views, self._displacements, times)

    # ===================================================================
    # Mesh and displacements
    # ===================================================================
    @property
    def mesh(self) -> Mesh:
        r"""
        [Get] The reference (undeformed) mesh of the analysis.

        .. note::

            The mesh is not a copy: it is **locked** while it is used by the analysis, so
            its vertices and elements cannot be modified (its properties can). Use
            :meth:`with_mesh` to get an analysis with another mesh.

        Returns
        -------
        :class:`pysdic.Mesh`
            The reference mesh.
        """
        return self._mesh

    @property
    def times(self) -> Optional[numpy.ndarray]:
        r"""
        [Get] The times of acquisition of the frames, strictly increasing.

        Use :meth:`with_times` to get an analysis with other times.

        Returns
        -------
        Optional[numpy.ndarray]
            A read-only array of shape :math:`(N_f,)`, or None if the times are not defined.
        """
        return self._times

    @property
    def n_frames(self) -> int:
        r"""
        [Get] The number of frames of the analysis.

        Returns
        -------
        int
            The number of frames.
        """
        return self._n_frames

    def set_displacement(self, frame: Integral, displacement: ArrayLike) -> None:
        r"""
        Set (or replace) the displacement of the vertices of the mesh at a frame.

        .. note::

            A displacement cannot be removed once set: the operators built on the analysis
            (data terms, regularizations, parametrizations) rely on the frames with a
            displacement.

        Parameters
        ----------
        frame : Integral
            The index of the frame, in :math:`[0, N_f)`.

        displacement : ArrayLike
            The displacement of the vertices, of shape :math:`(N_v, E)` (copied).

        Raises
        ------
        IndexError
            If ``frame`` is out of range.

        TypeError
            If ``displacement`` is None.

        ValueError
            If the displacement has a wrong shape or non-finite values.

        """
        frame = self._validate_frame(frame)
        self._displacements[frame] = self._as_displacement(displacement, self._mesh)

    def set_zero_displacement(self, frame: Integral) -> None:
        r"""
        Set a zero displacement of shape :math:`(N_v, E)` at a frame: the vertices are at
        the coordinates of the mesh.

        Parameters
        ----------
        frame : Integral
            The index of the frame, in :math:`[0, N_f)`.

        Raises
        ------
        IndexError
            If ``frame`` is out of range.

        """
        self.set_displacement(frame, numpy.zeros((self._mesh.n_vertices, self._mesh.n_dimensions)))

    def has_displacement(self, frame: Integral) -> bool:
        r"""
        Check if the displacement of a frame is set.

        Parameters
        ----------
        frame : Integral
            The index of the frame, in :math:`[0, N_f)`.

        Returns
        -------
        bool
            True if the displacement of the frame is set.
        """
        return self._displacements[self._validate_frame(frame)] is not None

    def get_displacement(self, frame: Integral) -> numpy.ndarray:
        r"""
        Get the displacement of the vertices of the mesh at a frame.

        Parameters
        ----------
        frame : Integral
            The index of the frame, in :math:`[0, N_f)`.

        Returns
        -------
        numpy.ndarray
            A read-only array of shape :math:`(N_v, E)`.

        Raises
        ------
        ValueError
            If the displacement of the frame is not set.

        """
        frame = self._validate_frame(frame)
        if self._displacements[frame] is None:
            raise ValueError(
                f"The displacement of the frame {frame} is not set. Use set_displacement or "
                f"set_zero_displacement first."
            )
        return self._displacements[frame]

    def get_deformed_coordinates(self, frame: Integral) -> numpy.ndarray:
        r"""
        Get the coordinates of the vertices of the mesh deformed at a frame:
        :math:`\vec{X} + \vec{u}(\text{frame})`.

        Parameters
        ----------
        frame : Integral
            The index of the frame, in :math:`[0, N_f)`.

        Returns
        -------
        numpy.ndarray
            An array of shape :math:`(N_v, E)`.

        Raises
        ------
        ValueError
            If the displacement of the frame is not set.

        """
        return self._mesh.coordinates + self.get_displacement(frame)

    def get_deformed_mesh(self, frame: Integral) -> Mesh:
        r"""
        Get a copy of the mesh deformed at a frame (same connectivity and properties).

        The copy is not locked: it can be modified freely.

        Parameters
        ----------
        frame : Integral
            The index of the frame, in :math:`[0, N_f)`.

        Returns
        -------
        :class:`pysdic.Mesh`
            The deformed mesh.

        Raises
        ------
        ValueError
            If the displacement of the frame is not set.

        """
        deformed = self._mesh.copy()
        deformed.coordinates = self.get_deformed_coordinates(frame)
        return deformed

    # ===================================================================
    # Views
    # ===================================================================
    def get_view(self, frame: Integral, index: Integral) -> View:
        r"""
        Get a view of a frame.

        Parameters
        ----------
        frame : Integral
            The index of the frame, in :math:`[0, N_f)`.

        index : Integral
            The index of the view in the frame.

        Returns
        -------
        :class:`pysdic.View`
            The view (not a copy, it is immutable).

        Raises
        ------
        IndexError
            If ``frame`` or ``index`` is out of range.

        """
        frame = self._validate_frame(frame)
        return self._views[frame][self._validate_view_index(frame, index)]

    def get_views(self, frame: Integral) -> Tuple[View, ...]:
        r"""
        Get the views of a frame.

        Parameters
        ----------
        frame : Integral
            The index of the frame, in :math:`[0, N_f)`.

        Returns
        -------
        Tuple[:class:`pysdic.View`, ...]
            The views of the frame, in their index order.
        """
        return self._views[self._validate_frame(frame)]

    def n_views(self, frame: Integral) -> int:
        r"""
        Get the number of views of a frame.

        Parameters
        ----------
        frame : Integral
            The index of the frame, in :math:`[0, N_f)`.

        Returns
        -------
        int
            The number of views of the frame.
        """
        return len(self._views[self._validate_frame(frame)])

    # ===================================================================
    # Pairings
    # ===================================================================
    def validate_pairing(self, pairing: Pairing) -> None:
        r"""
        Check that the two views of a pairing exist in the analysis and that their images
        have the same number of channels.

        Parameters
        ----------
        pairing : :class:`pysdic.Pairing`
            The pairing to check.

        Raises
        ------
        TypeError
            If ``pairing`` is not a :class:`pysdic.Pairing`.

        IndexError
            If a frame or a view of the pairing does not exist.

        ValueError
            If the images of the two views have different numbers of channels.

        """
        if not isinstance(pairing, Pairing):
            raise TypeError(f"Input 'pairing' must be a Pairing. Got type {type(pairing)}.")
        view_1 = self.get_view(pairing.frame_index_1, pairing.view_index_1)
        view_2 = self.get_view(pairing.frame_index_2, pairing.view_index_2)
        if view_1.image.n_channels != view_2.image.n_channels:
            raise ValueError(
                f"The images of the views of {pairing} must have the same number of channels. "
                f"Got {view_1.image.n_channels} and {view_2.image.n_channels}."
            )

    # ===================================================================
    # Unknowns
    # ===================================================================
    def _unknowns_layout(self, unknowns: SDICUnknowns) -> Tuple[int, List[slice], int]:
        # (number of displacement unknowns, slices of the transformations, number of unknowns)
        self.validate_unknowns(unknowns)
        n_U = unknowns.n_frames * self._mesh.n_vertices * self._mesh.n_dimensions
        slices, start = [], n_U
        for kind, index in unknowns.optimized_transformations:
            n_p = self.get_transformation(kind, index).n_parameters
            slices.append(slice(start, start + n_p))
            start += n_p
        return n_U, slices, start

    def validate_unknowns(self, unknowns: SDICUnknowns) -> None:
        r"""
        Check that a :class:`pysdic.SDICUnknowns` can be used with the analysis: its optimized
        frames exist, its optimized transformations exist, are distinct objects, have
        parameters and are not locked.

        Called at the creation of the objects of a problem (:class:`pysdic.SDICCorrelationTerm`,
        :class:`pysdic.SDICTemporalRegularizationTerm`, :class:`pysdic.SDICFixedParametrization`)
        and by the methods on the vector of unknowns (:meth:`pack_unknowns`, ...).

        .. note::

            The displacements of the optimized frames are not required here (they are the
            initial guess, needed by :meth:`pack_unknowns`).

        Parameters
        ----------
        unknowns : :class:`pysdic.SDICUnknowns`
            The unknowns to check.

        Raises
        ------
        TypeError
            If ``unknowns`` is not a :class:`pysdic.SDICUnknowns`.

        IndexError
            If a frame or a transformation does not exist.

        ValueError
            If a transformation has no parameters, is locked, or two entries are the same object.

        """
        if not isinstance(unknowns, SDICUnknowns):
            raise TypeError(f"Input 'unknowns' must be a SDICUnknowns. Got type {type(unknowns)}.")
        for k, frame in enumerate(unknowns.optimized_frames):
            self._validate_frame(frame, f"unknowns.optimized_frames[{k}]")
        objects = []
        for entry in unknowns.optimized_transformations:
            transformation = self.get_transformation(*entry)
            if any(transformation is other for other in objects):
                raise ValueError(f"The transformation {entry} of the unknowns is the same object as another entry.")
            if transformation.n_parameters == 0 or transformation.parameters is None:
                raise ValueError(f"The transformation {entry} ({type(transformation).__name__}) has no parameters to optimize.")
            if transformation.is_locked:
                raise ValueError(f"The parameters of the transformation {entry} ({type(transformation).__name__}) are locked.")
            objects.append(transformation)

    def n_unknowns(self, unknowns: SDICUnknowns) -> int:
        r"""
        The size of the vector of unknowns :math:`N_f N_v E + \sum n_p` of a
        :class:`pysdic.SDICUnknowns` in the analysis.
        """
        return self._unknowns_layout(unknowns)[2]

    def get_unknowns_frame_slice(self, unknowns: SDICUnknowns, frame: Integral) -> slice:
        r"""
        The slice of the displacement :math:`(N_v E,)` (Fortran order) of an optimized frame in
        the vector of unknowns.

        Raises
        ------
        ValueError
            If the frame is not optimized.

        """
        self.validate_unknowns(unknowns)
        if frame not in unknowns.optimized_frames:
            raise ValueError(f"The frame {frame} is not optimized. Optimized frames: {list(unknowns.optimized_frames)}.")
        k = unknowns.optimized_frames.index(frame)
        size = self._mesh.n_vertices * self._mesh.n_dimensions
        return slice(k * size, (k + 1) * size)

    def get_unknowns_transformation_slice(self, unknowns: SDICUnknowns, kind: str, index: Integral) -> slice:
        r"""
        The slice of the parameters of an optimized transformation in the vector of unknowns.

        Raises
        ------
        ValueError
            If the transformation is not optimized.

        """
        _, slices, _ = self._unknowns_layout(unknowns)
        entry = (kind, int(index))
        if entry not in unknowns.optimized_transformations:
            raise ValueError(
                f"The transformation {entry} is not optimized. "
                f"Optimized transformations: {list(unknowns.optimized_transformations)}."
            )
        return slices[unknowns.optimized_transformations.index(entry)]

    def compute_unknowns_indices(
        self,
        unknowns: SDICUnknowns,
        *,
        vertices: Optional[ArrayLike] = None,
        components: Optional[Sequence[Integral]] = None,
    ) -> numpy.ndarray:
        r"""
        The indices in the vector of unknowns of the displacements of selected vertices and
        components of all the optimized frames, followed by the indices of all the parameters of
        the optimized transformations.

        The displacement indices are returned frame by frame, component by component and vertex
        by vertex (increasing order).

        Parameters
        ----------
        unknowns : :class:`pysdic.SDICUnknowns`
            The unknowns.

        vertices : Optional[ArrayLike], optional
            The selected vertices: indices or a boolean mask of shape :math:`(N_v,)` (see
            :func:`pysdic.compute_supported_vertices`). By default, all the vertices.

        components : Optional[Sequence[Integral]], optional
            The selected components of the displacement (e.g. ``[0, 1]`` for :math:`x` and
            :math:`y`). By default, all the components.

        Returns
        -------
        numpy.ndarray
            The sorted indices (``numpy.int64``).

        """
        n_U, _, n = self._unknowns_layout(unknowns)
        N_v, E = self._mesh.n_vertices, self._mesh.n_dimensions
        vertices = _as_vertex_selection(vertices, N_v)
        components = _as_component_selection(components, E)
        block = (components[:, numpy.newaxis] * N_v + vertices[numpy.newaxis, :]).ravel()  # Indices in one frame
        displacement_indices = (numpy.arange(unknowns.n_frames)[:, numpy.newaxis] * (N_v * E) + block).ravel()
        return numpy.concatenate((displacement_indices, numpy.arange(n_U, n))).astype(numpy.int64)

    def pack_unknowns(self, unknowns: SDICUnknowns) -> numpy.ndarray:
        r"""
        The complete vector of unknowns :math:`\vec{p}` of the current state of the analysis
        (e.g. the initial parameters ``p0`` without parametrization): the **total** displacements
        of the optimized frames, then the parameters of the optimized transformations.

        .. code-block:: python

            p = numpy.concatenate(
                [analysis.get_displacement(frame).ravel(order="F") for frame in unknowns.optimized_frames]
                + [analysis.get_transformation_parameters(kind, index) for kind, index in unknowns.optimized_transformations]
            )

        .. warning::

            With a parametrization, use ``parametrization.reparametrized_pack()`` (see
            :class:`pysdic.SDICFixedParametrization`).

        Returns
        -------
        numpy.ndarray
            The vector of unknowns, of shape :math:`(N_f N_v E + \sum n_p,)` (:meth:`n_unknowns`).

        Raises
        ------
        ValueError
            If the unknowns are not valid or the displacement of an optimized frame is not set.

        """
        self.validate_unknowns(unknowns)
        return numpy.concatenate(
            [self.get_displacement(frame).ravel(order="F") for frame in unknowns.optimized_frames]
            + [
                numpy.asarray(self.get_transformation(kind, index).parameters, dtype=numpy.float64)
                for kind, index in unknowns.optimized_transformations
            ]
            + [numpy.zeros(0)]
        )

    def unpack_unknowns(
        self, unknowns: SDICUnknowns, parameters: ArrayLike
    ) -> Tuple[Dict[int, numpy.ndarray], Dict[Tuple[str, int], numpy.ndarray]]:
        r"""
        Split a complete vector of unknowns (inverse of :meth:`pack_unknowns`). The analysis is
        not modified (see :meth:`apply_unknowns`).

        .. warning::

            With a parametrization, use ``parametrization.reparametrized_unpack(q)`` (see
            :class:`pysdic.SDICFixedParametrization`).

        Parameters
        ----------
        unknowns : :class:`pysdic.SDICUnknowns`
            The unknowns.

        parameters : ArrayLike
            The vector of unknowns, of shape :math:`(N_f N_v E + \sum n_p,)` (:meth:`n_unknowns`).

        Returns
        -------
        displacements : Dict[int, numpy.ndarray]
            For each optimized frame, its total displacement of shape :math:`(N_v, E)` (a copy).

        transformations : Dict[Tuple[str, int], numpy.ndarray]
            For each optimized transformation ``(kind, index)``, its parameters (a copy).

        """
        n_U, slices, n = self._unknowns_layout(unknowns)
        parameters = _as_array(parameters, "parameters", numpy.float64, shape=(n,), finite=False)
        N_v, E = self._mesh.n_vertices, self._mesh.n_dimensions
        displacements = {
            frame: parameters[k * N_v * E : (k + 1) * N_v * E].reshape((N_v, E), order="F").copy()
            for k, frame in enumerate(unknowns.optimized_frames)
        }
        transformations = {
            entry: parameters[columns].copy() for entry, columns in zip(unknowns.optimized_transformations, slices)
        }
        return displacements, transformations

    def apply_unknowns(self, unknowns: SDICUnknowns, parameters: ArrayLike) -> None:
        r"""
        Write a complete vector of unknowns (e.g. ``result.parameters`` without parametrization)
        in the analysis: the displacements of the optimized frames (:meth:`set_displacement`,
        replaced, not added) and the parameters of the optimized transformations
        (:meth:`set_transformation_parameters`, all the cameras using them are updated).

        .. warning::

            With a parametrization, use ``parametrization.reparametrized_apply(q)`` (see
            :class:`pysdic.SDICFixedParametrization`).

        Parameters
        ----------
        unknowns : :class:`pysdic.SDICUnknowns`
            The unknowns.

        parameters : ArrayLike
            The vector of unknowns, of shape :math:`(N_f N_v E + \sum n_p,)` (:meth:`n_unknowns`).

        """
        displacements, transformations = self.unpack_unknowns(unknowns, parameters)
        for frame, displacement in displacements.items():
            self.set_displacement(frame, displacement)
        for (kind, index), values in transformations.items():
            self.set_transformation_parameters(kind, index, values)

    # ===================================================================
    # Structure
    # ===================================================================
    def _compute_structure(self, kind: str) -> Tuple[List[List[int]], List[Any]]:
        # The structure of 'kind' (index of the object of each view) and the distinct
        # objects, in the order of their first occurrence along the frames and the views
        getter = _STRUCTURE_KINDS[kind]
        indices = {}  # id(object) -> index
        objects = []  # Keep the objects alive while their ids are used
        structure = []
        for views in self._views:
            frame_indices = []
            for view in views:
                obj = getter(view)
                if id(obj) not in indices:
                    indices[id(obj)] = len(objects)
                    objects.append(obj)
                frame_indices.append(indices[id(obj)])
            structure.append(frame_indices)
        return structure, objects

    def _structure(self, kind: str) -> Tuple[List[List[int]], List[Any]]:
        # The structure of 'kind', computed once at the creation (the views are fixed)
        if kind not in _STRUCTURE_KINDS:
            raise ValueError(f"Input 'kind' must be one of {tuple(_STRUCTURE_KINDS)}. Got {kind!r}.")
        return self._structures[kind]

    def compute_structure(self, kind: str) -> List[List[int]]:
        r"""
        Compute the sharing of an object between the views of the analysis.

        Each distinct object (compared by identity, not by value) gets an index in the order
        of its first occurrence along the frames and the views. Two views with the same index
        share the same object: for example, the same camera at all the frames, or the same
        distortion for two cameras.

        .. note::

            The structures are computed once at the creation of the analysis (its views
            cannot change): this method returns a copy of the stored structure.

        Parameters
        ----------
        kind : str
            The shared object: ``"view"``, ``"image"``, ``"camera"``, ``"extrinsic"``,
            ``"distortion"`` or ``"intrinsic"``.

        Returns
        -------
        List[List[int]]
            For each frame, the index of the object of each of its views.

        Raises
        ------
        ValueError
            If ``kind`` is not valid.

        Examples
        --------
        Two cameras observing 3 frames, with a new image per view and frame:

        .. code-block:: python

            analysis.compute_structure("camera")  # [[0, 1], [0, 1], [0, 1]]
            analysis.compute_structure("image")   # [[0, 1], [2, 3], [4, 5]]

        """
        return [list(indices) for indices in self._structure(kind)[0]]

    def display_structure(self, kind: str) -> None:
        r"""
        Print the sharing of an object between the views of the analysis (see
        :meth:`compute_structure`).

        Parameters
        ----------
        kind : str
            The shared object: ``"view"``, ``"image"``, ``"camera"``, ``"extrinsic"``,
            ``"distortion"`` or ``"intrinsic"``.

        """
        structure = self.compute_structure(kind)
        print(f"Structure of the {kind}s (index of the shared {kind} of each view):")
        for frame, indices in enumerate(structure):
            print(f"  Frame {frame:<3}: {indices}")

    def get_transformation(self, kind: str, index: Integral) -> Transformation:
        r"""
        Get a transformation of the cameras of the analysis by its index in the structure
        of its kind (see :meth:`compute_structure` and :meth:`display_structure`).

        The indices are numbered separately for each kind: ``("extrinsic", 0)`` and
        ``("distortion", 0)`` are two different transformations.

        Parameters
        ----------
        kind : str
            The kind of the transformation: ``"extrinsic"``, ``"distortion"`` or
            ``"intrinsic"``.

        index : Integral
            The index of the transformation in the structure of ``kind``.

        Returns
        -------
        :class:`pysdic.Transformation`
            The transformation (not a copy: it is shared by the cameras that use it).

        Raises
        ------
        ValueError
            If ``kind`` is not valid.

        IndexError
            If ``index`` is out of range.

        Examples
        --------
        .. code-block:: python

            analysis.display_structure("extrinsic")
            # Frame 0  : [0, 1]
            # Frame 1  : [0, 1]
            right_extrinsic = analysis.get_transformation("extrinsic", 1)

        """
        if kind not in _TRANSFORMATION_KINDS:
            raise ValueError(f"Input 'kind' must be one of {_TRANSFORMATION_KINDS}. Got {kind!r}.")
        objects = self._structure(kind)[1]
        index = _validate_integer(index, "index", min_value=0)
        if index >= len(objects):
            raise IndexError(
                f"Input 'index' must be in [0, {len(objects) - 1}] for the kind {kind!r} "
                f"({len(objects)} distinct transformations). Got {index}."
            )
        return objects[index]

    def get_transformation_parameters(self, kind: str, index: Integral) -> Optional[numpy.ndarray]:
        r"""
        Get a copy of the parameters of a transformation of the cameras (see
        :meth:`get_transformation`).

        Parameters
        ----------
        kind : str
            The kind of the transformation: ``"extrinsic"``, ``"distortion"`` or
            ``"intrinsic"``.

        index : Integral
            The index of the transformation in the structure of ``kind``.

        Returns
        -------
        Optional[numpy.ndarray]
            A copy of the parameters, of shape :math:`(n_p,)`, or None if they are not set.

        Raises
        ------
        ValueError
            If ``kind`` is not valid.

        IndexError
            If ``index`` is out of range.

        """
        parameters = self.get_transformation(kind, index).parameters
        return None if parameters is None else parameters.copy()

    def set_transformation_parameters(
        self, kind: str, index: Integral, parameters: Optional[ArrayLike]
    ) -> None:
        r"""
        Set the parameters of a transformation of the cameras (see
        :meth:`get_transformation`).

        The transformation is shared by all the cameras that use it: their projections
        change together. The checks and the lock of the transformation are respected (for
        example, the transformation used inside a :class:`pysdic.ReparametrizedTransformation`
        is locked: set the parameters of the reparametrized transformation instead).

        Parameters
        ----------
        kind : str
            The kind of the transformation: ``"extrinsic"``, ``"distortion"`` or
            ``"intrinsic"``.

        index : Integral
            The index of the transformation in the structure of ``kind``.

        parameters : Optional[ArrayLike]
            The parameters, of shape :math:`(n_p,)` (copied), or None to unset them.

        Raises
        ------
        ValueError
            If ``kind`` is not valid or the parameters are invalid for the transformation.

        IndexError
            If ``index`` is out of range.

        RuntimeError
            If the transformation is locked (see its ``locked_message``).

        """
        self.get_transformation(kind, index).parameters = parameters

    def check_consistency(self) -> None:
        r"""
        Check that the transformations of the cameras of the analysis are independent of
        each other.

        The transformations are identified by ``(kind, index)`` (see
        :meth:`get_transformation`), each one owning its own parameters. Two rules are
        checked:

        1. The sets of the distinct extrinsics :math:`\mathcal{E}`, distortions
           :math:`\mathcal{D}` and intrinsics :math:`\mathcal{I}` (objects compared by
           identity) are disjoint: an object cannot be used in two roles. The same object
           can be used by several cameras in the same role (it is one entry).
        2. The leaf transformations of all the entries of
           :math:`\mathcal{E} \cup \mathcal{D} \cup \mathcal{I}` are all distinct. The
           leaves are found recursively: the transformations of a
           :class:`pysdic.ComposedTransformation`, the transformation reparametrized by a
           :class:`pysdic.ReparametrizedTransformation`, or the transformation itself
           otherwise. A leaf used twice, in two entries or twice in the same entry, would
           couple parameters without declaring them as one unknown (and would not be
           restored as one object after a save).

        .. note::

            This check is done at the creation of the analysis: since its views cannot
            change, the analysis always remains consistent.

        Raises
        ------
        ValueError
            If an object is used in two roles, or if a leaf transformation is used twice.

        """
        roles = {}  # id(entry object) -> (kind, index)
        owners = {}  # id(leaf transformation) -> (kind, index)
        alive = []  # Keep the objects alive while their ids are used
        for kind in _TRANSFORMATION_KINDS:
            for index, transformation in enumerate(self._structure(kind)[1]):
                alive.append(transformation)
                role = roles.setdefault(id(transformation), (kind, index))
                if role != (kind, index):
                    raise ValueError(
                        f"The same {type(transformation).__name__} object is used as {role} and "
                        f"as {(kind, index)}: the extrinsics, distortions and intrinsics of the "
                        f"cameras must be distinct objects. Use a copy (transformation.copy()) for "
                        f"one of the roles."
                    )
                for leaf in _leaf_transformations(transformation):
                    alive.append(leaf)
                    if id(leaf) not in owners:
                        owners[id(leaf)] = (kind, index)
                        continue
                    owner = owners[id(leaf)]
                    where = f"twice inside {owner}" if owner == (kind, index) else f"in {owner} and in {(kind, index)}"
                    raise ValueError(
                        f"The leaf transformation {type(leaf).__name__} is used {where}: the leaf "
                        f"transformations of the cameras (inside the composed and reparametrized "
                        f"transformations) must all be distinct objects, otherwise their parameters "
                        f"are coupled without being declared as one unknown. Use copies "
                        f"(transformation.copy())."
                    )

    # ===================================================================
    # Save and load
    # ===================================================================
    def _summary(self) -> Dict[str, Any]:
        # Summary of the analysis, written in metadata.json and compared at the loading
        return {
            "n_frames": self._n_frames,
            "n_vertices": self._mesh.n_vertices,
            "n_elements": self._mesh.n_elements,
            "n_dimensions": self._mesh.n_dimensions,
            "n_views": [len(frame_views) for frame_views in self._views],
            "n_images": len(self._structure("image")[1]),
            "n_cameras": len(self._structure("camera")[1]),
            "n_transformations": {kind: len(self._structure(kind)[1]) for kind in _TRANSFORMATION_KINDS},
            "frames_with_displacement": [frame for frame, u in enumerate(self._displacements) if u is not None],
            "has_times": self._times is not None,
        }

    def _write_folder(
        self, folder: Path, title: Optional[str], description: Optional[str], export_vtk: bool
    ) -> None:
        # Write the content of a save in an existing empty folder (see to_folder)
        objects = {kind: self._structure(kind)[1] for kind in _STRUCTURE_KINDS}
        index = {kind: {id(obj): k for k, obj in enumerate(objs)} for kind, objs in objects.items()}

        # Mesh (exact NPZ, optional VTK for the visualization only)
        self._mesh.to_npz_file(folder / "mesh" / "mesh.npz", make_dirs=True)
        if export_vtk:
            self._mesh.to_vtk_file(folder / "mesh" / "mesh.vtk", make_dirs=True, binary=True)

        # Atoms: each distinct image and transformation is saved once
        paths = {}
        for kind, (subfolder, extension) in _ATOM_FILES.items():
            width = _index_width(len(objects[kind]))
            paths[kind] = [f"{subfolder}/{kind}_{k:0{width}d}.{extension}" for k in range(len(objects[kind]))]
            for obj, relative in zip(objects[kind], paths[kind]):
                save = obj.to_npz_file if extension == "npz" else obj.to_json_file
                save(folder / relative, make_dirs=True)

        # Displacements (only the frames with a displacement) and times
        width = _index_width(self._n_frames)
        displacements = {
            str(frame): f"frame_{frame:0{width}d}"
            for frame, u in enumerate(self._displacements)
            if u is not None
        }
        if displacements:
            numpy.savez(
                folder / "displacements.npz",
                **{key: self._displacements[int(frame)] for frame, key in displacements.items()},
            )
        if self._times is not None:
            numpy.savez(folder / "times.npz", times=self._times)

        # Cameras, views and frames as indices of the atoms
        _write_json(folder / "structure.json", {
            "mesh": "mesh/mesh.npz",
            "images": paths["image"],
            "transformations": {kind: paths[kind] for kind in _TRANSFORMATION_KINDS},
            "cameras": [
                {
                    "sensor_height": camera.sensor_height,
                    "sensor_width": camera.sensor_width,
                    **{kind: index[kind][id(getattr(camera, kind))] for kind in _TRANSFORMATION_KINDS},
                }
                for camera in objects["camera"]
            ],
            "views": [
                {"camera": index["camera"][id(view.camera)], "image": index["image"][id(view.image)]}
                for view in objects["view"]
            ],
            "frames": [[index["view"][id(view)] for view in frame_views] for frame_views in self._views],
            "displacements": displacements,
            "times": self._times is not None,
            "structures": {kind: self.compute_structure(kind) for kind in _STRUCTURE_KINDS},
        })

        # Metadata, and the same information readable by a human in README.txt
        created = datetime.datetime.now(datetime.timezone.utc).isoformat()
        summary = self._summary()
        _write_json(folder / "metadata.json", {
            "format": _FORMAT_NAME,
            "pysdic_version": __version__,
            "created_utc": created,
            "title": title,
            "description": description,
            "summary": summary,
        })
        lines = [f"pysdic SDIC save (pysdic {__version__})", f"Created (UTC): {created}"]
        lines += [f"{label}: {text}" for label, text in (("Title", title), ("Description", description)) if text is not None]
        lines += ["", f"element_type: {self._mesh.element_type}"]
        lines += [f"{key}: {value}" for key, value in summary.items()]
        lines += [
            "",
            "Load with: pysdic.SDICAnalysis.from_folder(path)",
            "Do not modify the files: their SHA-256 are checked at the loading (manifest.json).",
        ]
        with open(folder / "README.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        # Manifest: SHA-256 of all the other files, written last
        manifest_path = folder / "manifest.json"
        files = {
            path.relative_to(folder).as_posix(): _sha256(path)
            for path in sorted(folder.rglob("*"))
            if path.is_file() and path != manifest_path
        }
        _write_json(manifest_path, {"algorithm": "sha256", "files": files})

    def _check_loaded(self, folder: Path, metadata: Dict[str, Any], structure: Dict[str, Any]) -> None:
        # The loaded analysis must have the saved summary (metadata.json) and structures (structure.json).
        # Only the keys of the saved summary are compared (a previous format may lack some of them).
        saved_summary = metadata.get("summary")
        saved_summary = saved_summary if isinstance(saved_summary, dict) else {}
        saved_structures = structure.get("structures")
        saved_structures = saved_structures if isinstance(saved_structures, dict) else {}
        mismatches = [
            f"{key}: saved {saved_summary[key]!r}, loaded {value!r}"
            for key, value in self._summary().items()
            if key in saved_summary and saved_summary[key] != value
        ]
        for kind in _STRUCTURE_KINDS:
            loaded = self.compute_structure(kind)
            if saved_structures.get(kind) != loaded:
                mismatches.append(f"structure of the {kind}s: saved {saved_structures.get(kind)!r}, loaded {loaded!r}")
        if mismatches:
            raise ValueError(
                f"Inconsistent loading of '{folder}': the loaded analysis differs from the save "
                f"({'; '.join(mismatches)}). The save may have been modified or written by an "
                f"incompatible version."
            )

    @staticmethod
    def _validate_text(value: Optional[str], name: str) -> Optional[str]:
        if value is not None and not isinstance(value, str):
            raise TypeError(f"Input '{name}' must be a string or None. Got type {type(value)}.")
        return value

    @staticmethod
    def _prepare_target(path: Path, *, overwrite: bool, make_dirs: bool, is_folder: bool) -> None:
        # Check that a save (folder or archive) can be written at 'path' and create its parent
        if path.exists():
            if not overwrite:
                raise FileExistsError(f"'{path}' already exists. Use overwrite=True to replace it.")
            if is_folder and _read_metadata(path) is None:
                raise FileExistsError(
                    f"'{path}' exists but is not a SDIC save (no metadata.json of the format "
                    f"{_FORMAT_NAME!r}): it is not replaced, to avoid deleting unrelated data."
                )
            if not is_folder and path.is_dir():
                raise FileExistsError(f"'{path}' is a folder: it cannot be replaced by an archive.")
        if not path.parent.is_dir():
            if not make_dirs:
                raise FileNotFoundError(f"The parent folder '{path.parent}' does not exist. Use make_dirs=True to create it.")
            path.parent.mkdir(parents=True, exist_ok=True)

    def to_folder(
        self,
        folder: Union[str, os.PathLike],
        *,
        overwrite: bool = False,
        make_dirs: bool = False,
        title: Optional[str] = None,
        description: Optional[str] = None,
        export_vtk: bool = False,
    ) -> None:
        r"""
        Save the analysis in a folder, one file per atom, to reload it exactly with
        :meth:`from_folder`.

        The folder contains:

        .. code-block:: text

            folder/
            ├── metadata.json        format, pysdic version, date, title, description, summary
            ├── README.txt           the same information, readable by a human
            ├── manifest.json        SHA-256 of all the files (checked at the loading)
            ├── structure.json       cameras, views and frames as indices of the atoms,
            │                        and the structures of the analysis
            ├── mesh/mesh.npz        the reference mesh (+ mesh.vtk if export_vtk=True)
            ├── images/image_0000.npz          one per distinct image (data and settings)
            ├── extrinsics/extrinsic_0000.json one per distinct transformation of each kind
            ├── distortions/distortion_0000.json
            ├── intrinsics/intrinsic_0000.json
            ├── displacements.npz    keys frame_0000, ... (only the frames with a displacement)
            └── times.npz            key "times" (only if the times are defined)

        Each distinct object (image, transformation) is saved **once**, even if it is shared
        by several views or cameras: :meth:`from_folder` restores the same sharing. The
        indices of the file names have the number of digits of the largest index (for
        example 4 digits for 1543 frames).

        The save is safe: the content is written in a temporary folder next to ``folder``,
        which is renamed at the end. If an error occurs, the temporary folder is removed and
        an existing ``folder`` is unchanged.

        .. note::

            The integration points and the pairings are not part of the analysis: save them
            separately (for example with ``integration_points.to_npz_file``).

        Parameters
        ----------
        folder : Union[str, os.PathLike]
            The folder to create.

        overwrite : bool, optional (default: ``False``)
            If True, an existing folder is replaced, only if it is a SDIC save (it has a
            ``metadata.json`` of this format): a folder with other content is never
            deleted. It is removed only after the new save is complete.

        make_dirs : bool, optional (default: ``False``)
            If True, the missing parent folders are created.

        title : Optional[str], optional
            A title written in the metadata.

        description : Optional[str], optional
            A description written in the metadata.

        export_vtk : bool, optional (default: ``False``)
            If True, the mesh is also exported to ``mesh/mesh.vtk`` (binary, with its
            properties) to be visualized, for example in ParaView. This file is not used by
            :meth:`from_folder`. Requires the VTK dependencies of :meth:`pysdic.Mesh.to_vtk_file`.

        Raises
        ------
        FileExistsError
            If ``folder`` exists and ``overwrite`` is False, or if it exists but is not a
            SDIC save.

        FileNotFoundError
            If the parent folder does not exist and ``make_dirs`` is False.

        Examples
        --------
        .. code-block:: python

            analysis.to_folder("results/test_42", title="Tensile test 42")
            same_sdic = SDICAnalysis.from_folder("results/test_42")

        """
        folder = _validate_path(folder, "folder")
        overwrite = _validate_bool(overwrite, "overwrite")
        make_dirs = _validate_bool(make_dirs, "make_dirs")
        title = self._validate_text(title, "title")
        description = self._validate_text(description, "description")
        export_vtk = _validate_bool(export_vtk, "export_vtk")
        self._prepare_target(folder, overwrite=overwrite, make_dirs=make_dirs, is_folder=True)

        # Write in a temporary folder next to 'folder', then swap it with the existing save (if any)
        temporary = Path(tempfile.mkdtemp(prefix=f".{folder.name}.tmp-", dir=folder.parent))
        try:
            self._write_folder(temporary, title, description, export_vtk)
            if not folder.exists():
                os.replace(temporary, folder)
                return
            old = temporary.with_name(temporary.name + ".old")
            os.replace(folder, old)
            try:
                os.replace(temporary, folder)
            except BaseException:
                os.replace(old, folder)  # Restore the previous save
                raise
            shutil.rmtree(old)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)

    @classmethod
    def from_folder(
        cls,
        folder: Union[str, os.PathLike],
        *,
        check_integrity: bool = True,
        auto_construct: bool = True,
    ) -> SDICAnalysis:
        r"""
        Load an analysis saved with :meth:`to_folder`.

        A save written by a previous version of pysdic (before 2.1.0) is converted to the
        current format while loading, with a :class:`PreviousSaveWarning` (the files are not
        modified: save the analysis again to update the save). The atoms are loaded first
        (mesh, images, extrinsics, distortions, intrinsics), then the cameras and the views
        are rebuilt with the same sharing as the saved analysis, and the analysis is created
        with its times and displacements. Finally, the summary (``metadata.json``) and the
        structures (:meth:`compute_structure`) of the loaded analysis are compared with the
        saved ones.

        Parameters
        ----------
        folder : Union[str, os.PathLike]
            The folder of the save.

        check_integrity : bool, optional (default: ``True``)
            If True, the SHA-256 of all the files are checked with ``manifest.json`` before
            the loading: a missing or modified file raises an error.

        auto_construct : bool, optional (default: ``True``)
            Whether to construct the interpolation functions of the images.

        Returns
        -------
        :class:`pysdic.SDICAnalysis`
            The loaded analysis (its mesh is locked).

        Raises
        ------
        FileNotFoundError
            If the folder does not exist.

        DeprecatedError
            If the save was written by a version of pysdic whose format is deprecated.

        ValueError
            If the folder is not a SDIC save, it was written by an unsupported version of
            pysdic, a file is missing or modified (with ``check_integrity``), or the loaded
            analysis differs from the saved summary and structures (inconsistent loading).

        """
        folder = _validate_path(folder, "folder", must_be_dir=True)
        check_integrity = _validate_bool(check_integrity, "check_integrity")
        auto_construct = _validate_bool(auto_construct, "auto_construct")
        metadata = _read_metadata(folder)
        if metadata is None:
            raise ValueError(f"'{folder}' is not a SDIC save (no metadata.json of the format {_FORMAT_NAME!r}).")
        version = _check_version(folder, metadata)
        if check_integrity:
            _check_manifest(folder)
        structure = _read_json(folder / "structure.json")
        if version < parse_version(_MIN_VERSION):
            metadata, structure = load_previous_save(folder, metadata, structure, version)

        # Atoms
        mesh = Mesh.from_npz_file(folder / structure["mesh"])
        images = [Image.from_npz_file(folder / relative, auto_construct=auto_construct) for relative in structure["images"]]
        transformations = {
            kind: [Transformation.from_json_file(folder / relative) for relative in structure["transformations"][kind]]
            for kind in _TRANSFORMATION_KINDS
        }

        # Cameras, views and frames with the saved sharing
        cameras = [
            Camera(
                camera["sensor_height"],
                camera["sensor_width"],
                **{kind: transformations[kind][camera[kind]] for kind in _TRANSFORMATION_KINDS},
            )
            for camera in structure["cameras"]
        ]
        views = [View(cameras[view["camera"]], images[view["image"]]) for view in structure["views"]]
        frames = [[views[index] for index in frame_views] for frame_views in structure["frames"]]

        times = _read_arrays(folder / "times.npz")["times"] if structure["times"] else None
        displacements = {}
        if structure["displacements"]:
            arrays = _read_arrays(folder / "displacements.npz")
            displacements = {int(frame): arrays[key] for frame, key in structure["displacements"].items()}

        analysis = cls(mesh, frames, times=times, displacements=displacements)
        analysis._check_loaded(folder, metadata, structure)
        return analysis

    def to_archive(
        self,
        file_path: Union[str, os.PathLike],
        *,
        overwrite: bool = False,
        make_dirs: bool = False,
        compress: bool = False,
        title: Optional[str] = None,
        description: Optional[str] = None,
        export_vtk: bool = False,
    ) -> None:
        r"""
        Save the analysis in a single ZIP archive, to reload it exactly with
        :meth:`from_archive`.

        The archive contains exactly the files of :meth:`to_folder` (at its root), including
        ``manifest.json``: the integrity of the content is checked at the loading. It is
        written in a temporary folder next to ``file_path`` and renamed at the end, so an
        existing archive is unchanged if an error occurs.

        Parameters
        ----------
        file_path : Union[str, os.PathLike]
            The archive to create (for example ``"test_42.analysis.zip"``).

        overwrite : bool, optional (default: ``False``)
            If True, an existing file is replaced (a folder is never replaced).

        make_dirs : bool, optional (default: ``False``)
            If True, the missing parent folders are created.

        compress : bool, optional (default: ``False``)
            If True, the files are compressed (``ZIP_DEFLATED``): smaller but slower. By
            default they are stored (``ZIP_STORED``).

        title, description, export_vtk
            See :meth:`to_folder`.

        Raises
        ------
        FileExistsError
            If ``file_path`` exists and ``overwrite`` is False, or if it is a folder.

        FileNotFoundError
            If the parent folder does not exist and ``make_dirs`` is False.

        """
        file_path = _validate_path(file_path, "file_path")
        overwrite = _validate_bool(overwrite, "overwrite")
        make_dirs = _validate_bool(make_dirs, "make_dirs")
        compress = _validate_bool(compress, "compress")
        title = self._validate_text(title, "title")
        description = self._validate_text(description, "description")
        export_vtk = _validate_bool(export_vtk, "export_vtk")
        self._prepare_target(file_path, overwrite=overwrite, make_dirs=make_dirs, is_folder=False)

        # Write the save and its archive in a temporary folder next to 'file_path' (same file
        # system: the final rename is atomic), removed in all cases
        with tempfile.TemporaryDirectory(prefix=f".{file_path.name}.tmp-", dir=file_path.parent) as temporary:
            content = Path(temporary) / "content"
            content.mkdir()
            self._write_folder(content, title, description, export_vtk)
            archive_path = Path(temporary) / "archive.zip"
            mode = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
            with zipfile.ZipFile(archive_path, "w", compression=mode) as archive:
                for path in sorted(content.rglob("*")):
                    if path.is_file():
                        archive.write(path, path.relative_to(content).as_posix())
            os.replace(archive_path, file_path)

    @classmethod
    def from_archive(
        cls,
        file_path: Union[str, os.PathLike],
        *,
        check_integrity: bool = True,
        auto_construct: bool = True,
    ) -> SDICAnalysis:
        r"""
        Load an analysis saved with :meth:`to_archive`.

        The archive is extracted in a temporary folder, loaded with :meth:`from_folder`
        (same checks), then the temporary folder is removed.

        Parameters
        ----------
        file_path : Union[str, os.PathLike]
            The archive to load.

        check_integrity : bool, optional (default: ``True``)
            If True, the SHA-256 of all the files are checked (see :meth:`from_folder`).

        auto_construct : bool, optional (default: ``True``)
            Whether to construct the interpolation functions of the images.

        Returns
        -------
        :class:`pysdic.SDICAnalysis`
            The loaded analysis.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.

        ValueError
            If the file is not a valid ZIP archive, contains unsafe paths, or cannot be
            loaded (see :meth:`from_folder`).

        """
        file_path = _validate_path(file_path, "file_path", must_exist=True)
        if not zipfile.is_zipfile(file_path):
            raise ValueError(f"'{file_path}' is not a ZIP archive of a SDIC save.")
        with zipfile.ZipFile(file_path, "r") as archive:
            for name in archive.namelist():
                if Path(name).is_absolute() or ".." in Path(name).parts or name.startswith(("/", "\\")):
                    raise ValueError(f"The archive '{file_path}' contains an unsafe path: {name!r}.")
            with tempfile.TemporaryDirectory() as temporary:
                archive.extractall(temporary)
                return cls.from_folder(temporary, check_integrity=check_integrity, auto_construct=auto_construct)