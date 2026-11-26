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

# Consistency of the objects of a pysolvegn problem on a SDIC analysis.

from __future__ import annotations

from itertools import combinations
from typing import Any, Sequence

from .sdic_analysis import SDICAnalysis
from .sdic_unknowns import SDICUnknowns


def check_same_problem(objects: Sequence[Any]) -> None:
    r"""
    Check that the objects of a ``pysolvegn`` problem (terms and parametrizations) work on the
    same analysis and the same unknowns, before giving them to ``pysolvegn.solve``.

    The objects are compared two by two, **by identity**: for each pair, ``a.analysis is
    b.analysis`` and ``a.unknowns is b.unknowns``, with an analysis of type
    :class:`pysdic.SDICAnalysis` and unknowns of type :class:`pysdic.SDICUnknowns`. Two
    different (even equivalent) objects are rejected: build all the objects of a problem with
    the same analysis object and the same unknowns object.

    ``pysolvegn`` only sees functions: two terms built on different unknowns of the same size
    (e.g. the frame 1 and the frame 2) would be added without any error. This check makes this
    mistake explicit.

    Parameters
    ----------
    objects : Sequence
        The objects of the problem, each one with the properties ``analysis`` and ``unknowns``:
        :class:`pysdic.SDICCorrelationTerm`, :class:`pysdic.SDICTemporalRegularizationTerm`,
        :class:`pysdic.SDICFixedParametrization`.

    Raises
    ------
    TypeError
        If ``objects`` is not a sequence, an object has no ``analysis`` or ``unknowns``, or they
        have a wrong type.

    ValueError
        If ``objects`` is empty, or two objects have different analyses or different unknowns.

    Examples
    --------
    .. code-block:: python

        from pysdic import check_same_problem

        check_same_problem([correlation, regularization, parametrization])
        result = pysolvegn.solve([correlation.term, regularization.term], parametrization.reparametrized_pack(),
                                 parametrization=parametrization.parametrization)

    """
    if not isinstance(objects, (list, tuple)):
        raise TypeError(f"Input 'objects' must be a list or a tuple. Got type {type(objects)}.")
    if len(objects) == 0:
        raise ValueError("Input 'objects' must contain at least one object.")
    for k, item in enumerate(objects):
        for name, kind in (("analysis", SDICAnalysis), ("unknowns", SDICUnknowns)):
            if not hasattr(item, name):
                raise TypeError(f"The object {k} ({type(item).__name__}) has no property '{name}'.")
            if not isinstance(getattr(item, name), kind):
                raise TypeError(
                    f"The '{name}' of the object {k} ({type(item).__name__}) must be a {kind.__name__}. "
                    f"Got type {type(getattr(item, name))}."
                )
    for (i, a), (j, b) in combinations(enumerate(objects), 2):
        if a.analysis is not b.analysis:
            raise ValueError(
                f"The objects {i} ({type(a).__name__}) and {j} ({type(b).__name__}) have different analyses: "
                "build all the objects of a problem with the same SDICAnalysis object."
            )
        if a.unknowns is not b.unknowns:
            raise ValueError(
                f"The objects {i} ({type(a).__name__}, {a.unknowns!r}) and {j} ({type(b).__name__}, {b.unknowns!r}) "
                "have different unknowns: build all the objects of a problem with the same SDICUnknowns object."
            )