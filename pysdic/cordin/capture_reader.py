"""
CORDIN XML READER
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

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union


def load_capture_info(xml_file: Union[str, Path]) -> ET.Element:
    r"""
    Load a capture XML file.

    This function parses a CaptureInfo XML file and returns its root element.
    The returned element can then be passed to the other helper functions of
    this module.

    Parameters
    ----------
    xml_file : str or pathlib.Path
        Path to the XML file.

    Returns
    -------
    xml.etree.ElementTree.Element
        Root element corresponding to the ``CaptureInfo`` node.

    Examples
    --------
    >>> root = load_capture_info("CaptureInfo.xml")
    >>> print(root.tag)
    CaptureInfo
    """
    tree = ET.parse(xml_file)
    return tree.getroot()


def get_capture_parameters(root: ET.Element) -> Dict[str, str]:
    r"""
    Return the global capture parameters.

    The root ``CaptureInfo`` element stores all acquisition parameters as XML
    attributes. This function returns these attributes as a dictionary.

    Parameters
    ----------
    root : xml.etree.ElementTree.Element
        Root element returned by :func:`load_capture_info`.

    Returns
    -------
    dict
        Dictionary containing the capture parameters.

    Examples
    --------
    >>> params = get_capture_parameters(root)
    >>> params["Timestamp"]
    '2026-07-18T22:39:47.829176+02:00'
    """
    return dict(root.attrib)


def get_camera_details(root: ET.Element) -> Dict[str, str]:
    r"""
    Return the camera description.

    This function extracts the attributes of the ``CameraDetails`` element.

    Parameters
    ----------
    root : xml.etree.ElementTree.Element
        Root element returned by :func:`load_capture_info`.

    Returns
    -------
    dict
        Dictionary containing the camera information such as serial number,
        sensor type, module count and camera name.

    Examples
    --------
    >>> camera = get_camera_details(root)
    >>> camera["SerialNumber"]
    'C508'
    """
    camera = root.find("CameraDetails")
    return {} if camera is None else dict(camera.attrib)


def convert_value(value: str) -> Any:
    r"""
    Convert an XML attribute value to an appropriate Python type.

    The conversion rules are

    - ``"true"`` and ``"false"`` → ``bool``
    - integer strings → ``int``
    - floating-point strings → ``float``
    - all remaining values are returned unchanged.

    Parameters
    ----------
    value : str
        XML attribute value.

    Returns
    -------
    Any
        Converted value.
    """
    if value.lower() == "true":
        return True

    if value.lower() == "false":
        return False

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    return value


def get_frames(
    root: ET.Element,
    key: Literal["frame", "module"] = "frame",
    convert_types: bool = True,
) -> Dict[int, Dict[str, Any]]:
    r"""
    Return the frame acquisition information.

    Each ``FrameInfo`` element is converted into a dictionary containing both
    its XML attributes and the associated raw image filename.

    The returned dictionary can be indexed either by frame number or by module
    number.

    Parameters
    ----------
    root : xml.etree.ElementTree.Element
        Root element returned by :func:`load_capture_info`.

    key : {"frame", "module"}, optional (default: ``"frame"``)
        Dictionary indexing strategy.

        - ``"frame"``: keys correspond to ``FrameNumber``.
        - ``"module"``: keys correspond to ``Module``.

    convert_types : bool, optional (default: ``True``)
        If True, XML attribute values are automatically converted to native
        Python types (``bool``, ``int`` and ``float`` whenever possible).
        Otherwise, all values remain strings.

    Returns
    -------
    dict
        Dictionary whose keys are frame numbers or module numbers and whose
        values are dictionaries containing the acquisition metadata.

    Notes
    -----
    Each frame dictionary additionally contains the ``FileName`` entry, which
    corresponds to the text contained in the ``FileName`` XML child node.

    Examples
    --------
    Index by frame number

    >>> frames = get_frames(root)
    >>> frames[12]["ExposureTime"]

    Index by module number

    >>> modules = get_frames(root, key="module")
    >>> modules[60]["FileName"]
    """
    frames = {}

    frames_node = root.find("Frames")

    if frames_node is None:
        return frames

    for frame in frames_node.findall("FrameInfo"):

        if convert_types:
            data = {k: convert_value(v) for k, v in frame.attrib.items()}
        else:
            data = dict(frame.attrib)

        data["FileName"] = frame.findtext("FileName", "")

        index = data["Module"] if key == "module" else data["FrameNumber"]

        frames[index] = data

    return frames


class CORDIN:
    r"""
    Acquisition information of a CORDIN capture, read from its ``CaptureInfo`` XML file.

    Each image of the capture has a frame number (``FrameNumber``, from 1 to the number of
    frames, in the order of acquisition) and a module number (``Module``, the module of the
    camera that took it). This class gives the information of an image from either number,
    and converts one into the other.

    .. important::

        The frame numbers start at 1 (not 0), as in the XML file.

    Parameters
    ----------
    xml_file : str or pathlib.Path
        Path to the ``CaptureInfo`` XML file.

    convert_types : bool, optional (default: ``True``)
        If True, the values of the attributes are converted to ``bool``, ``int`` or ``float``
        when possible (see :func:`convert_value`).

    Examples
    --------
    >>> calibration = CORDIN("calibration/capture.xml")
    >>> measure = CORDIN("measure/capture.xml")
    >>> measure.get_infos(frame=12)["FileName"]
    >>> measure.get_module(frame=12)
    >>> measure.get_frame(module=60)
    """

    def __init__(self, xml_file: Union[str, Path], convert_types: bool = True) -> None:
        self._xml_file = Path(xml_file)
        root = load_capture_info(self._xml_file)
        self._parameters = get_capture_parameters(root)
        self._camera = get_camera_details(root)
        self._by_frame = get_frames(root, key="frame", convert_types=convert_types)
        self._frame_by_module = {}
        for frame, infos in self._by_frame.items():
            module = infos["Module"]
            if module in self._frame_by_module:
                raise ValueError(
                    f"The module {module} is used by the frames {self._frame_by_module[module]} and {frame} "
                    f"in {self._xml_file}."
                )
            self._frame_by_module[module] = frame

    def __repr__(self) -> str:
        return f"CORDIN('{self._xml_file}', {self.n_frames} frames)"

    # ------------------------------------------------------------------ Properties
    @property
    def xml_file(self) -> Path:
        r"""The path of the XML file."""
        return self._xml_file

    @property
    def parameters(self) -> Dict[str, str]:
        r"""The global capture parameters (attributes of ``CaptureInfo``), a copy."""
        return dict(self._parameters)

    @property
    def camera(self) -> Dict[str, str]:
        r"""The camera description (attributes of ``CameraDetails``), a copy."""
        return dict(self._camera)

    @property
    def frames(self) -> List[int]:
        r"""The frame numbers, sorted (from 1)."""
        return sorted(self._by_frame)

    @property
    def modules(self) -> List[int]:
        r"""The module numbers, sorted."""
        return sorted(self._frame_by_module)

    @property
    def n_frames(self) -> int:
        r"""The number of frames."""
        return len(self._by_frame)

    # ------------------------------------------------------------------ Access
    def _check_frame(self, frame: Any) -> int:
        if frame not in self._by_frame:
            raise KeyError(
                f"Unknown frame {frame!r} in {self._xml_file}: the frames go from {min(self._by_frame)} "
                f"to {max(self._by_frame)} (the frame numbers start at 1)."
            )
        return frame

    def _check_module(self, module: Any) -> int:
        if module not in self._frame_by_module:
            raise KeyError(f"Unknown module {module!r} in {self._xml_file}. Available: {self.modules}.")
        return module

    def get_infos(self, frame: Optional[int] = None, module: Optional[int] = None) -> Dict[str, Any]:
        r"""
        Return the information of an image, given either its frame number or its module
        number (exactly one of them).

        Parameters
        ----------
        frame : int, optional
            The frame number (from 1).

        module : int, optional
            The module number.

        Returns
        -------
        dict
            A copy of the attributes of the ``FrameInfo`` element, with its ``FileName``.

        Raises
        ------
        ValueError
            If both or none of ``frame`` and ``module`` are given.

        KeyError
            If the frame or the module does not exist.
        """
        if (frame is None) == (module is None):
            raise ValueError("Give exactly one of 'frame' and 'module'.")
        if frame is None:
            frame = self._frame_by_module[self._check_module(module)]
        return dict(self._by_frame[self._check_frame(frame)])

    def get_frame(self, module: int) -> int:
        r"""
        Return the frame number (from 1) of the image taken by a module.

        Raises
        ------
        KeyError
            If the module does not exist.
        """
        return self._frame_by_module[self._check_module(module)]

    def get_module(self, frame: int) -> int:
        r"""
        Return the module number of the image of a frame (from 1).

        Raises
        ------
        KeyError
            If the frame does not exist.
        """
        return self._by_frame[self._check_frame(frame)]["Module"]