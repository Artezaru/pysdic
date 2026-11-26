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

from typing import Optional, Tuple, Union
from numpy.typing import ArrayLike
from numbers import Integral, Real

import numpy

from .._validation import _validate_bool, _validate_real
from .mesh import Mesh
from .mesh_conversion import quadrangle_4_to_triangle_3
from .create_3D_surface_meshes import _grid_axis, create_uniform_quadrangle_4_mesh


# =============================================================================
# Internal helpers
# =============================================================================
def _rectangle_vertices(x_bounds, y_bounds, x_array, y_array, n_x, n_y) -> numpy.ndarray:
    # Vertices (n_x, n_y, 2) of a rectangular grid
    x = _grid_axis(x_bounds, x_array, n_x, ("x_bounds", "x_array", "n_x"))
    y = _grid_axis(y_bounds, y_array, n_y, ("y_bounds", "y_array", "n_y"))
    x_grid, y_grid = numpy.meshgrid(x, y, indexing="ij")  # (n_x, n_y)
    return numpy.stack((x_grid, y_grid), axis=2)


def _polar_vertices(r_bounds, t_bounds, r_array, t_array, n_r, n_t) -> Tuple[numpy.ndarray, numpy.ndarray]:
    # Vertices (n_r, n_t, 2) of a polar grid and its radii (n_r,). The radius is the first axis
    # of the grid, so that the elements are counterclockwise in the plane (positive Jacobian)
    r = _grid_axis(r_bounds, r_array, n_r, ("r_bounds", "r_array", "n_r"))
    t = _grid_axis(t_bounds, t_array, n_t, ("t_bounds", "t_array", "n_t"))
    if numpy.any(r < 0):
        raise ValueError(f"The radii (r_bounds or r_array) must be non-negative. Got a minimum of {r.min()}.")
    r_grid, t_grid = numpy.meshgrid(r, t, indexing="ij")  # (n_r, n_t)
    return numpy.stack((r_grid * numpy.cos(t_grid), r_grid * numpy.sin(t_grid)), axis=2), r


# =============================================================================
# Rectangles
# =============================================================================
def create_2d_quadrangle_4_rectangle(
    *,
    x_bounds: Optional[Tuple[Real, Real]] = None,
    y_bounds: Optional[Tuple[Real, Real]] = None,
    x_array: Optional[ArrayLike] = None,
    y_array: Optional[ArrayLike] = None,
    n_x: Optional[Integral] = None,
    n_y: Optional[Integral] = None,
    direct: bool = True,
    uv_layout: Integral = 0,
) -> Mesh:
    r"""
    Create a 2D ``quadrangle_4`` mesh (:math:`E = 2`) of a rectangular grid.

    The grid of vertices is given, for each axis, either by bounds and a number of
    vertices (uniform grid), or by an array of coordinates.

    .. important::

        ``n_x`` and ``n_y`` are numbers of **vertices**, not elements.

    The vertex at the indices ``i_x`` along :math:`x` and ``i_y`` along :math:`y` (both
    starting from 0) is ``mesh.coordinates[i_y * n_x + i_x]``.

    .. seealso::

        :func:`pysdic.create_3d_quadrangle_4_heightmap` for a 3D surface :math:`z = f(x, y)`.

    Parameters
    ----------
    x_bounds, y_bounds: Optional[Tuple[Real, Real]], optional
        The bounds of the uniform grid along :math:`x` and :math:`y` (with ``n_x`` and
        ``n_y``).

    x_array, y_array: Optional[ArrayLike], optional
        The coordinates of the grid along :math:`x` and :math:`y` (instead of the
        bounds), with at least 2 values.

    n_x, n_y: Optional[Integral], optional
        The numbers of vertices along :math:`x` and :math:`y` (:math:`\geq 2`).
        Required with the bounds, checked against the arrays if given.

    direct: bool, optional (default: ``True``)
        If True, the vertices of the quadrangles are ordered counterclockwise in the
        plane (for increasing coordinates), otherwise clockwise.

    uv_layout: Integral, optional (default: ``0``)
        The UV mapping strategy, between 0 and 7 (see
        :func:`pysdic.create_uniform_quadrangle_4_mesh`).

    Returns
    -------
    :class:`pysdic.Mesh`
        The 2D ``quadrangle_4`` mesh, with a UV map.

    Examples
    --------
    .. code-block:: python
        :linenos:

        from pysdic import create_2d_quadrangle_4_rectangle

        mesh = create_2d_quadrangle_4_rectangle(x_bounds=(0, 2), y_bounds=(0, 1), n_x=5, n_y=3)
        print(mesh.n_dimensions, mesh.n_vertices, mesh.n_elements)  # 2 15 8

    """
    vertices = _rectangle_vertices(x_bounds, y_bounds, x_array, y_array, n_x, n_y)
    return create_uniform_quadrangle_4_mesh(vertices, direct=direct, uv_layout=uv_layout)


def create_2d_triangle_3_rectangle(
    *,
    x_bounds: Optional[Tuple[Real, Real]] = None,
    y_bounds: Optional[Tuple[Real, Real]] = None,
    x_array: Optional[ArrayLike] = None,
    y_array: Optional[ArrayLike] = None,
    n_x: Optional[Integral] = None,
    n_y: Optional[Integral] = None,
    direct: bool = True,
    diagonal: str = "first",
    seed: Optional[Union[Integral, numpy.random.Generator]] = None,
    uv_layout: Integral = 0,
) -> Mesh:
    r"""
    Create a 2D ``triangle_3`` mesh (:math:`E = 2`) of a rectangular grid.

    The ``quadrangle_4`` mesh of :func:`pysdic.create_2d_quadrangle_4_rectangle` is split
    into triangles with :func:`pysdic.quadrangle_4_to_triangle_3` (the quadrangle
    :math:`q` gives the triangles :math:`2q` and :math:`2q + 1`). The vertices are the
    same.

    Parameters
    ----------
    x_bounds, y_bounds, x_array, y_array, n_x, n_y, direct, uv_layout:
        See :func:`pysdic.create_2d_quadrangle_4_rectangle`.

    diagonal: str, optional (default: ``"first"``)
        The diagonal used to split the quadrangles: ``"first"``, ``"second"``,
        ``"alternate"`` or ``"random"``.

    seed: Optional[Union[Integral, numpy.random.Generator]], optional (default: ``None``)
        The seed of ``diagonal="random"``.

    Returns
    -------
    :class:`pysdic.Mesh`
        The 2D ``triangle_3`` mesh, with a UV map.

    """
    mesh = create_2d_quadrangle_4_rectangle(
        x_bounds=x_bounds, y_bounds=y_bounds, x_array=x_array, y_array=y_array,
        n_x=n_x, n_y=n_y, direct=direct, uv_layout=uv_layout,
    )
    return quadrangle_4_to_triangle_3(mesh, diagonal, seed=seed)


# =============================================================================
# Polar grids (annuli, disks and sectors)
# =============================================================================
def create_2d_quadrangle_4_polar(
    *,
    r_bounds: Optional[Tuple[Real, Real]] = None,
    t_bounds: Optional[Tuple[Real, Real]] = None,
    r_array: Optional[ArrayLike] = None,
    t_array: Optional[ArrayLike] = None,
    n_r: Optional[Integral] = None,
    n_t: Optional[Integral] = None,
    direct: bool = True,
    close: bool = False,
    uv_layout: Integral = 0,
) -> Mesh:
    r"""
    Create a 2D ``quadrangle_4`` mesh (:math:`E = 2`) of a polar grid: an annulus or a
    sector of an annulus.

    The vertex at the radius :math:`r` and the angle :math:`\theta` is
    :math:`(r \cos\theta, r \sin\theta)`.

    .. important::

        ``n_r`` and ``n_t`` are numbers of **vertices**, not elements.

    The vertex at the indices ``i_r`` along :math:`r` and ``i_t`` along :math:`\theta`
    (both starting from 0) is ``mesh.coordinates[i_t * n_r + i_r]``.

    If ``close`` is True, the last and first angles are connected by one more ring of
    elements. For a uniform closed grid, use ``t_bounds=(0, 2 * pi * (1 - 1 / n_t))``.

    The radii must be strictly positive: at :math:`r = 0` the quadrangles would be
    degenerated. Use :func:`pysdic.create_2d_triangle_3_polar` with ``stick=True`` for a
    disk.

    .. seealso::

        :func:`pysdic.create_3d_quadrangle_4_cylindrical` for a 3D cylindrical surface.

    Parameters
    ----------
    r_bounds, t_bounds: Optional[Tuple[Real, Real]], optional
        The bounds of the uniform grid along :math:`r` and :math:`\theta` (with ``n_r``
        and ``n_t``).

    r_array, t_array: Optional[ArrayLike], optional
        The coordinates of the grid along :math:`r` and :math:`\theta` (instead of the
        bounds), with at least 2 values.

    n_r, n_t: Optional[Integral], optional
        The numbers of vertices along :math:`r` and :math:`\theta` (:math:`\geq 2`).
        Required with the bounds, checked against the arrays if given.

    direct: bool, optional (default: ``True``)
        If True, the vertices of the quadrangles are ordered counterclockwise in the
        plane (for increasing radii and angles), otherwise clockwise.

    close: bool, optional (default: ``False``)
        If True, the mesh is closed along :math:`\theta`.

    uv_layout: Integral, optional (default: ``0``)
        The UV mapping strategy, between 0 and 7 (see
        :func:`pysdic.create_uniform_quadrangle_4_mesh`), with :math:`r` as the first
        axis of the grid and :math:`\theta` as the second one.

    Returns
    -------
    :class:`pysdic.Mesh`
        The 2D ``quadrangle_4`` mesh, with a UV map.

    Raises
    ------
    ValueError
        If a radius is not strictly positive, or if an input has a wrong value.

    Examples
    --------
    A closed annulus of radii 1 and 2.

    .. code-block:: python
        :linenos:

        import numpy
        from pysdic import create_2d_quadrangle_4_polar

        n_t = 24
        mesh = create_2d_quadrangle_4_polar(
            r_bounds=(1, 2), n_r=4, t_bounds=(0, 2 * numpy.pi * (1 - 1 / n_t)), n_t=n_t, close=True,
        )
        print(mesh.n_vertices, mesh.n_elements)  # 96 72

    """
    close = _validate_bool(close, "close")
    vertices, r = _polar_vertices(r_bounds, t_bounds, r_array, t_array, n_r, n_t)
    if numpy.any(r == 0.0):
        raise ValueError(
            "The radii must be strictly positive for a quadrangle_4 polar mesh. Use "
            "'create_2d_triangle_3_polar' with stick=True for a disk."
        )
    return create_uniform_quadrangle_4_mesh(vertices, y_connect=close, direct=direct, uv_layout=uv_layout)


def create_2d_triangle_3_polar(
    *,
    r_bounds: Optional[Tuple[Real, Real]] = None,
    t_bounds: Optional[Tuple[Real, Real]] = None,
    r_array: Optional[ArrayLike] = None,
    t_array: Optional[ArrayLike] = None,
    n_r: Optional[Integral] = None,
    n_t: Optional[Integral] = None,
    direct: bool = True,
    close: bool = False,
    stick: bool = False,
    stick_eps: Real = 1e-6,
    diagonal: str = "first",
    seed: Optional[Union[Integral, numpy.random.Generator]] = None,
    uv_layout: Integral = 0,
) -> Mesh:
    r"""
    Create a 2D ``triangle_3`` mesh (:math:`E = 2`) of a polar grid: an annulus, a disk or
    a sector.

    The ``quadrangle_4`` polar grid (see :func:`pysdic.create_2d_quadrangle_4_polar`) is
    split into triangles with :func:`pysdic.quadrangle_4_to_triangle_3`.

    If ``stick`` is True, the vertices at the center (radius lower than ``stick_eps``) are
    merged into a single vertex at the origin. The triangles that become degenerated are
    removed, and the unused vertices too, so the indexing of the vertices of the polar
    grid is no longer valid. Use it for a disk (``r_bounds=(0, R)``) or a sector of a
    disk.

    .. seealso::

        :func:`pysdic.create_3d_triangle_3_cylindrical` for a 3D cylindrical surface.

    Parameters
    ----------
    r_bounds, t_bounds, r_array, t_array, n_r, n_t, direct, close, uv_layout:
        See :func:`pysdic.create_2d_quadrangle_4_polar`. Here, the radius :math:`r = 0`
        is accepted.

    stick: bool, optional (default: ``False``)
        If True, the vertices at the center are merged (see above).

    stick_eps: Real, optional (default: ``1e-6``)
        The radius below which a vertex is at the center (non-negative).

    diagonal: str, optional (default: ``"first"``)
        The diagonal used to split the quadrangles: ``"first"``, ``"second"``,
        ``"alternate"`` or ``"random"``.

    seed: Optional[Union[Integral, numpy.random.Generator]], optional (default: ``None``)
        The seed of ``diagonal="random"``.

    Returns
    -------
    :class:`pysdic.Mesh`
        The 2D ``triangle_3`` mesh, with a UV map.

    Examples
    --------
    A closed unit disk.

    .. code-block:: python
        :linenos:

        import numpy
        from pysdic import create_2d_triangle_3_polar

        n_t = 24
        mesh = create_2d_triangle_3_polar(
            r_bounds=(0, 1), n_r=5, t_bounds=(0, 2 * numpy.pi * (1 - 1 / n_t)), n_t=n_t,
            close=True, stick=True,
        )
        print(mesh.n_vertices, mesh.n_elements)  # 97 168

    """
    close = _validate_bool(close, "close")
    stick = _validate_bool(stick, "stick")
    stick_eps = _validate_real(stick_eps, "stick_eps", non_negative=True, finite=True)
    vertices, _ = _polar_vertices(r_bounds, t_bounds, r_array, t_array, n_r, n_t)
    mesh = create_uniform_quadrangle_4_mesh(vertices, y_connect=close, direct=direct, uv_layout=uv_layout)
    mesh = quadrangle_4_to_triangle_3(mesh, diagonal, seed=seed)

    if stick:
        # Merge all the vertices at the center into the first one, at the origin
        coordinates = numpy.array(mesh.coordinates, copy=True)
        at_center = numpy.flatnonzero(numpy.linalg.norm(coordinates, axis=1) < stick_eps)
        if at_center.size > 0:
            mapping = numpy.arange(mesh.n_vertices)
            mapping[at_center] = at_center[0]
            coordinates[at_center] = 0.0
            mesh.coordinates = coordinates
            mesh.elements = mapping[mesh.elements]
            mesh = mesh.remove_degenerated_elements()  # The methods return a new mesh
            mesh = mesh.remove_unused_vertices()
    return mesh