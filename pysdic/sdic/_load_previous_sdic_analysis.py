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

# Loading of the SDIC saves written by the previous versions of pysdic (see SDICAnalysis.from_folder).
#
# A save written by a previous version is converted in memory, before its loading, to the current
# format by a chain of migrations: each migration converts the content (metadata.json and
# structure.json) of the saves written by the versions in [min_version, max_version) to the
# format of max_version. The files of the save are never modified, and a warning
# (PreviousSaveWarning) is displayed: save the analysis again to update the save.
#
# The migrations are frozen: they describe the formats of the past, so they use literal values
# and never the constants of the current format.
#
# To add a format (version X.Y.Z): add a migration load_from_<previous min>_to_X_Y_Z at the end of
# _MIGRATIONS and set _MIN_VERSION = "X.Y.Z" in sdic_analysis.py.
#
# To deprecate a format that can no longer be converted: replace the body of its migration by
# _raise_deprecated(...) (see load_from_0_0_0_to_0_0_1). The more recent migrations are kept: the
# saves of the more recent versions are still loaded.

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, NoReturn, Optional, Tuple

Metadata = Dict[str, Any]
Structure = Dict[str, Any]

# Names of the formats of the previous versions (the current name "pysdic.SDICAnalysis" was also used by 2.0.0)
PREVIOUS_FORMAT_NAMES = ("pysdic.SDIC",)


class DeprecatedError(ValueError):
    r"""
    The SDIC save was written by a version of pysdic whose format can no longer be loaded.
    """


class PreviousSaveWarning(UserWarning):
    r"""
    The SDIC save was written by a previous version of pysdic: it is converted to the current
    format while loading. Save the analysis again (:meth:`pysdic.SDICAnalysis.to_folder`) to update it.
    """


def parse_version(version: Any) -> Tuple[int, ...]:
    r"""
    The ``(major, minor, patch)`` of a version ``"X.Y.Z..."`` (the suffixes such as ``"rc1"`` or
    ``".dev0"`` are ignored).

    Raises
    ------
    ValueError
        If the version is not a string starting with ``"X.Y.Z"``.
    """
    match = re.match(r"(\d+)\.(\d+)\.(\d+)", version) if isinstance(version, str) else None
    if match is None:
        raise ValueError(f"Invalid version {version!r}: 'major.minor.patch' expected.")
    return tuple(int(number) for number in match.groups())


# ===================================================================
# Helpers of the migrations
# ===================================================================
def _check_format(folder: Path, metadata: Metadata, name: str, versions: str) -> None:
    # The metadata of a previous save must have the format of the versions that wrote it
    format_version = metadata.get("format_version")
    if metadata.get("format") != name or type(format_version) is not int or format_version != 1:
        raise ValueError(
            f"The SDIC save '{folder}' was written by pysdic {metadata.get('pysdic_version')}: the saves "
            f"of the versions {versions} have the format {name!r} version 1. Got the format "
            f"{metadata.get('format')!r} version {format_version!r}."
        )


def _raise_deprecated(folder: Path, metadata: Metadata, versions: str, last_reader: Optional[str]) -> NoReturn:
    # Refuse a save of a deprecated format. 'last_reader': a version of pysdic that can still load it
    advice = (
        f"Load it with pysdic {last_reader} and save it again (to_folder) to convert it to a supported format."
        if last_reader is not None
        else "It cannot be converted anymore."
    )
    raise DeprecatedError(
        f"The SDIC save '{folder}' was written by pysdic {metadata.get('pysdic_version')}: the saves of "
        f"the versions {versions} are deprecated and can no longer be loaded. {advice}"
    )


# ===================================================================
# Migrations, from the oldest versions
# ===================================================================
def load_from_0_0_0_to_0_0_1(folder: Path, metadata: Metadata, structure: Structure) -> Tuple[Metadata, Structure]:
    r"""
    Saves written by pysdic in [0.0.0, 0.0.1): **deprecated**, they can no longer be loaded.

    This is the template of a deprecated format. To deprecate another format later, replace the
    body of its migration by :func:`_raise_deprecated`, for example:

    .. code-block:: python

        def load_from_0_0_1_to_2_0_0(folder, metadata, structure):
            _raise_deprecated(folder, metadata, "[0.0.1, 2.0.0)", last_reader="2.1.0")

    Raises
    ------
    DeprecatedError
        Always.
    """
    _raise_deprecated(folder, metadata, "[0.0.0, 0.0.1)", last_reader=None)


def load_from_0_0_1_to_2_0_0(folder: Path, metadata: Metadata, structure: Structure) -> Tuple[Metadata, Structure]:
    r"""
    Convert a save written by pysdic in [0.0.1, 2.0.0) to the format of pysdic 2.0.0.

    The analysis class was named ``SDIC``: the format is ``"pysdic.SDIC"`` (format version 1). The
    files and ``structure.json`` are the same as in 2.0.0: only the name of the format changes.
    """
    _check_format(folder, metadata, "pysdic.SDIC", "[0.0.1, 2.0.0)")
    return {**metadata, "format": "pysdic.SDICAnalysis"}, structure


def load_from_2_0_0_to_2_1_0(folder: Path, metadata: Metadata, structure: Structure) -> Tuple[Metadata, Structure]:
    r"""
    Convert a save written by pysdic in [2.0.0, 2.1.0) to the format of pysdic 2.1.0.

    - ``metadata.json``: the format version is removed (the compatibility is given by
      ``pysdic_version``) and the summary, at the root of the metadata, is moved to ``"summary"``.
      ``n_transformations`` is computed from ``structure.json``; ``n_dimensions`` was not saved
      (it is not checked at the loading).
    - ``structure.json``: ``"n_frames"`` is removed (it is the number of frames).
    """
    _check_format(folder, metadata, "pysdic.SDICAnalysis", "[2.0.0, 2.1.0)")
    structure = dict(structure)
    n_frames = structure.pop("n_frames", None)
    if n_frames != len(structure.get("frames", ())):
        raise ValueError(
            f"Inconsistent SDIC save '{folder}': structure.json has n_frames = {n_frames!r} but "
            f"{len(structure.get('frames', ()))} frames."
        )
    summary_keys = (
        "n_frames", "n_vertices", "n_elements", "n_views", "n_images", "n_cameras",
        "frames_with_displacement", "has_times",
    )
    summary = {key: metadata[key] for key in summary_keys if key in metadata}
    summary["n_transformations"] = {kind: len(paths) for kind, paths in structure.get("transformations", {}).items()}
    new_metadata = {key: metadata.get(key) for key in ("format", "pysdic_version", "created_utc", "title", "description")}
    new_metadata["summary"] = summary
    return new_metadata, structure


# (min_version, max_version, migration), from the oldest versions, contiguous. The last
# max_version is the first version of the current format (_MIN_VERSION of sdic_analysis.py).
_MIGRATIONS: Tuple[Tuple[str, str, Callable[[Path, Metadata, Structure], Tuple[Metadata, Structure]]], ...] = (
    ("0.0.0", "0.0.1", load_from_0_0_0_to_0_0_1),
    ("0.0.1", "2.0.0", load_from_0_0_1_to_2_0_0),
    ("2.0.0", "2.1.0", load_from_2_0_0_to_2_1_0),
)


def load_previous_save(
    folder: Path, metadata: Metadata, structure: Structure, version: Tuple[int, ...]
) -> Tuple[Metadata, Structure]:
    r"""
    Convert the content of a save written by a previous version of pysdic to the current format,
    by applying the migrations from its version, and display a :class:`PreviousSaveWarning`.

    Parameters
    ----------
    folder : Path
        The folder of the save (for the messages).

    metadata, structure : dict
        The content of ``metadata.json`` and ``structure.json`` (not modified).

    version : Tuple[int, ...]
        The parsed version of pysdic that wrote the save (see :func:`parse_version`).

    Returns
    -------
    metadata, structure : dict
        The content converted to the current format.

    Raises
    ------
    DeprecatedError
        If the format of the save is deprecated.

    ValueError
        If the content does not have the format of the version that wrote it.
    """
    for _, max_version, migration in _MIGRATIONS:
        if version < parse_version(max_version):
            metadata, structure = migration(folder, metadata, structure)
    warnings.warn(
        f"The SDIC save '{folder}' was written by pysdic {metadata.get('pysdic_version')} in a previous "
        f"format: it is converted while loading (the files are not modified). Save the analysis again "
        f"with to_folder to update the save.",
        PreviousSaveWarning,
        stacklevel=3,  # The call of SDICAnalysis.from_folder
    )
    return metadata, structure