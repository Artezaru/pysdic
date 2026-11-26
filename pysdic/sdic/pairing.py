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

from dataclasses import dataclass
from numbers import Integral
from typing import Tuple

from .._validation import _validate_integer


@dataclass(frozen=True, slots=True)
class Pairing:
    r"""
    A link between two views of a :class:`pysdic.SDICAnalysis` analysis: the view
    ``view_index_1`` of the frame ``frame_index_1`` and the view ``view_index_2`` of the
    frame ``frame_index_2``.

    The gray levels of the material points are compared between the two views, with the
    residual:

    .. math::

        r(\vec{X}) = [I_2 \circ P_2](\vec{X} + \vec{u}_2(\vec{X})) - [I_1 \circ P_1](\vec{X} + \vec{u}_1(\vec{X}))

    where :math:`\vec{u}_1` and :math:`\vec{u}_2` are the displacements of the frames 1
    and 2, :math:`P_1`, :math:`P_2` the projections of the cameras and :math:`I_1`,
    :math:`I_2` the images of the views.

    - Two views of the same frame (``frame_index_1 == frame_index_2``) give a stereo
      pairing (see :meth:`stereo`).
    - The same view index at two frames gives a temporal pairing (see :meth:`temporal`).

    The pairing does not know the analysis: use :meth:`pysdic.SDICAnalysis.validate_pairing` to
    check that its views exist.

    Parameters
    ----------
    frame_index_1 : Integral
        The index of the frame of the first view (non-negative).

    view_index_1 : Integral
        The index of the first view in its frame (non-negative).

    frame_index_2 : Integral
        The index of the frame of the second view (non-negative).

    view_index_2 : Integral
        The index of the second view in its frame (non-negative).

    Raises
    ------
    TypeError
        If an index is not an integer.

    ValueError
        If an index is negative or if the two views are the same.

    Examples
    --------
    Compare the first view of the frame 0 with the first view of the frame 3:

    .. code-block:: python

        from pysdic import Pairing

        pairing = Pairing(0, 0, 3, 0)  # Same as Pairing.temporal(0, 0, 3)
        print(pairing)  # Pairing((frame 0, view 0) -> (frame 3, view 0))

    """

    frame_index_1: int
    view_index_1: int
    frame_index_2: int
    view_index_2: int

    def __post_init__(self) -> None:
        for name in ("frame_index_1", "view_index_1", "frame_index_2", "view_index_2"):
            object.__setattr__(self, name, _validate_integer(getattr(self, name), name, min_value=0))
        if self.view_1 == self.view_2:
            raise ValueError(f"The two views of a pairing must be different. Got {self.view_1} twice.")

    @classmethod
    def stereo(cls, frame_index: Integral, view_index_1: Integral, view_index_2: Integral) -> Pairing:
        r"""
        Create a pairing between two views of the same frame.

        Parameters
        ----------
        frame_index : Integral
            The index of the frame.

        view_index_1 : Integral
            The index of the first view in the frame.

        view_index_2 : Integral
            The index of the second view in the frame.

        Returns
        -------
        :class:`pysdic.Pairing`
            The pairing ``(frame_index, view_index_1) -> (frame_index, view_index_2)``.
        """
        return cls(frame_index, view_index_1, frame_index, view_index_2)

    @classmethod
    def temporal(cls, view_index: Integral, frame_index_1: Integral, frame_index_2: Integral) -> Pairing:
        r"""
        Create a pairing between the views of same index of two frames.

        Parameters
        ----------
        view_index : Integral
            The index of the view in both frames.

        frame_index_1 : Integral
            The index of the first frame.

        frame_index_2 : Integral
            The index of the second frame.

        Returns
        -------
        :class:`pysdic.Pairing`
            The pairing ``(frame_index_1, view_index) -> (frame_index_2, view_index)``.
        """
        return cls(frame_index_1, view_index, frame_index_2, view_index)

    @property
    def view_1(self) -> Tuple[int, int]:
        r"""
        [Get] The first view as a tuple ``(frame_index_1, view_index_1)``.

        Returns
        -------
        Tuple[int, int]
            The frame and view indices of the first view.
        """
        return (self.frame_index_1, self.view_index_1)

    @property
    def view_2(self) -> Tuple[int, int]:
        r"""
        [Get] The second view as a tuple ``(frame_index_2, view_index_2)``.

        Returns
        -------
        Tuple[int, int]
            The frame and view indices of the second view.
        """
        return (self.frame_index_2, self.view_index_2)

    @property
    def is_stereo(self) -> bool:
        r"""
        [Get] Whether the two views belong to the same frame.

        Returns
        -------
        bool
            True if ``frame_index_1 == frame_index_2``.
        """
        return self.frame_index_1 == self.frame_index_2

    def reversed(self) -> Pairing:
        r"""
        Return the pairing with the two views swapped (opposite residual).

        Returns
        -------
        :class:`pysdic.Pairing`
            The pairing ``(frame_index_2, view_index_2) -> (frame_index_1, view_index_1)``.
        """
        return Pairing(self.frame_index_2, self.view_index_2, self.frame_index_1, self.view_index_1)

    def __str__(self) -> str:
        return (
            f"Pairing((frame {self.frame_index_1}, view {self.view_index_1}) -> "
            f"(frame {self.frame_index_2}, view {self.view_index_2}))"
        )