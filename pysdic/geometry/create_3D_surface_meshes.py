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

from typing import TYPE_CHECKING, Callable, Optional, Tuple, Union
from numpy.typing import ArrayLike
from numbers import Integral, Real

import numpy

from .._optionals import import_optional
from .._validation import (
    _validate_bool,
    _validate_integer,
    _validate_real,
    _as_array,
)
from .mesh import Mesh
from .mesh_conversion import quadrangle_4_to_triangle_3

if TYPE_CHECKING:
    import py3dframe


# =============================================================================
# Internal helpers
# =============================================================================
def _validate_uv_layout(uv_layout: Integral) -> int:
    # UV mapping strategy: an integer in [0, 7]
    uv_layout = _validate_integer(uv_layout, "uv_layout", min_value=0)
    if uv_layout > 7:
        raise ValueError(f"Input 'uv_layout' must be an integer between 0 and 7. Got {uv_layout}.")
    return uv_layout


_UV_MODES = ("index", "parametric")
_UV_MODES_ANGLE = ("index", "parametric", "parametric-angle")


def _validate_uv_mode(uv_mode: str, allowed: Tuple[str, ...]) -> str:
    # UV mode: one of the allowed strings
    if not isinstance(uv_mode, str) or uv_mode not in allowed:
        raise ValueError(f"Input 'uv_mode' must be one of {allowed}. Got {uv_mode!r}.")
    return uv_mode


def _index_uv_fractions(n: int, connect: bool) -> numpy.ndarray:
    # UV fractions of the "index" mode along one axis: (n,), or (n + 1,) with the fictive vertex of a connected axis
    m = n + 1 if connect else n
    return numpy.arange(m, dtype=numpy.float64) / (m - 1)


def _parametric_uv_fractions(values: numpy.ndarray, name: str, period: Optional[float] = None) -> numpy.ndarray:
    # UV fractions of the "parametric" mode along one axis: the coordinates (n,) normalized between the first
    # and the last vertices, or, for a closed axis, (n + 1,) normalized by the period (fictive vertex at
    # values[0] +/- period, fraction 1)
    delta = numpy.diff(values)
    if not (numpy.all(delta > 0) or numpy.all(delta < 0)):
        raise ValueError(
            f"The coordinates along {name} must be strictly monotonic for a parametric UV map "
            f"(use uv_mode='index' for any coordinates)."
        )
    if period is None:
        return (values - values[0]) / (values[-1] - values[0])
    span = abs(values[-1] - values[0])
    if span >= period:
        raise ValueError(
            f"The coordinates along {name} must span less than {period} to close the mesh with a parametric "
            f"UV map. Got a span of {span}."
        )
    return numpy.append((values - values[0]) / (numpy.sign(delta[0]) * period), 1.0)


def _uv_fractions(
    uv_mode: str,
    values: numpy.ndarray,
    connect: bool,
    name: str,
    period: Optional[float] = None,
    conformal: Optional[Callable[[numpy.ndarray], numpy.ndarray]] = None,
) -> numpy.ndarray:
    # UV fractions along one axis of a grid with the parametric coordinates values (n,). conformal is the
    # change of variable of the "parametric-angle" mode along this axis (None: same as "parametric")
    if uv_mode == "index":
        return _index_uv_fractions(values.size, connect)
    if uv_mode == "parametric-angle" and conformal is not None:
        values = conformal(values)
    if connect and period is None:
        raise ValueError(f"A connected axis ({name}) needs a period for a parametric UV map.")
    return _parametric_uv_fractions(values, name, period if connect else None)


def _isometric_latitude(t: numpy.ndarray) -> numpy.ndarray:
    # Isometric latitude psi = ln(tan(theta / 2)) of the polar angles theta (strictly between 0 and pi):
    # (phi, psi) is a conformal parametrization of the unit sphere (Mercator projection), its metric is
    # sin(theta)**2 * (dphi**2 + dpsi**2)
    if numpy.any(t <= 0) or numpy.any(t >= numpy.pi):
        raise ValueError(
            "The polar angles (t_bounds or t_array) must be strictly between 0 and pi for "
            "uv_mode='parametric-angle' (the Mercator projection is infinite at the poles)."
        )
    return numpy.log(numpy.tan(0.5 * t))


def _grid_axis(
    bounds: Optional[Tuple[Real, Real]],
    array: Optional[ArrayLike],
    n: Optional[Integral],
    names: Tuple[str, str, str],
) -> numpy.ndarray:
    # Coordinates (n,) along one axis of a grid, from (bounds, n) or from an array (n checked if given)
    bounds_name, array_name, n_name = names
    if (bounds is None) == (array is None):
        raise ValueError(f"Exactly one of '{bounds_name}' and '{array_name}' must be given.")
    n = _validate_integer(n, n_name, min_value=2, allow_none=True)
    if bounds is not None:
        if n is None:
            raise ValueError(f"Input '{n_name}' must be given with '{bounds_name}'.")
        bounds = _as_array(bounds, bounds_name, numpy.float64, shape=(2,), finite=True)
        if bounds[0] == bounds[1]:
            raise ValueError(f"Input '{bounds_name}' must contain two different values. Got {bounds.tolist()}.")
        return numpy.linspace(bounds[0], bounds[1], n)
    array = _as_array(array, array_name, numpy.float64, shape=(None,), shape_hint=f"({n_name},)", finite=True)
    if array.size < 2:
        raise ValueError(f"Input '{array_name}' must contain at least 2 values. Got {array.size}.")
    if n is not None and array.size != n:
        raise ValueError(f"Input '{array_name}' must contain {n_name}={n} values. Got {array.size}.")
    return array


def _evaluate(function: Optional[Callable], default: float, a: numpy.ndarray, b: numpy.ndarray, name: str) -> numpy.ndarray:
    # Evaluate a surface function f(a, b) on a grid (a scalar result is broadcast), or a constant
    if function is None:
        return numpy.full(a.shape, default, dtype=numpy.float64)
    if not callable(function):
        raise TypeError(f"Input '{name}' must be a callable or None. Got type {type(function)}.")
    values = numpy.asarray(function(a, b), dtype=numpy.float64)
    try:
        return numpy.broadcast_to(values, a.shape).copy()
    except ValueError:
        raise ValueError(
            f"Input '{name}' must return a scalar or an array of shape {a.shape} (the shape "
            f"of the grid). Got shape {values.shape}."
        ) from None


def _to_canonical(points: numpy.ndarray, frame: Optional[py3dframe.Frame], method: str) -> numpy.ndarray:
    # Express points (N, 3) given in a local frame in the canonical frame (no change if frame is None)
    if frame is None:
        return points
    py3dframe = import_optional("py3dframe", method)
    if not isinstance(frame, py3dframe.Frame):
        raise TypeError(f"Input 'frame' must be a py3dframe.Frame or None. Got type {type(frame)}.")
    transform = py3dframe.FrameTransform(input_frame=frame, output_frame=py3dframe.Frame.canonical())
    return numpy.array(transform.transform(point=points.T).T, dtype=numpy.float64)


def _local_frame_mesh(mesh: Mesh, frame: Optional[py3dframe.Frame], method: str) -> Mesh:
    # Move the vertices of a mesh built in the local frame to the canonical frame (in place)
    if frame is not None:
        mesh.coordinates = _to_canonical(mesh.vertices._coordinates, frame, method)
    return mesh


def _create_vertices_uvmap(fx: numpy.ndarray, fy: numpy.ndarray, uv_layout: int) -> numpy.ndarray:
    r"""
    Create the UV map of a grid of vertices from the UV fractions along its two axes.

    Parameters
    ----------
    fx : numpy.ndarray
        The fractions (n_x,) of the vertices along the x direction, from 0 (first vertex) to 1 (last vertex).

    fy : numpy.ndarray
        The fractions (n_y,) of the vertices along the y direction, from 0 (first vertex) to 1 (last vertex).

    uv_layout : int
        The UV mapping strategy. Must be an integer between 0 and 7.

    Returns
    -------
    numpy.ndarray
        An array of shape (n_x, n_y, 2) containing the UV coordinates for each vertex.

    UV Mapping
    ----------

    The UV coordinates are given by the fractions along the two axes of the grid in the range [0, 1] for the ``OpenGL texture mapping convention`` (See https://learnopengl.com/Getting-started/Textures).
    Several UV mapping strategies are available and synthesized in the ``uv_layout`` parameter.

    An image is defined by its four corners:

    - Lower-left corner : pixel with array coordinates ``image[height-1, 0]`` but OpenGL convention is (0,0) at lower-left
    - Upper-left corner : pixel with array coordinates ``image[0, 0]`` but OpenGL convention is (0,1) at upper-left
    - Lower-right corner : pixel with array coordinates ``image[height-1, width-1]`` but OpenGL convention is (1,0) at lower-right
    - Upper-right corner : pixel with array coordinates ``image[0, width-1]`` but OpenGL convention is (1,1) at upper-right

    The following options are available for ``uv_layout`` and their corresponding vertex mapping:

    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | uv_layout       | Vertex lower-left corner| Vertex upper-left corner| Vertex lower-right corner| Vertex upper-right corner|
    +=================+=========================+=========================+==========================+==========================+
    | 0               | (0, 0)                  | (0, n_y-1)              | (n_x-1, 0)               | (n_x-1, n_y-1)           |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | 1               | (0, 0)                  | (n_x-1, 0)              | (0, n_y-1)               | (n_x-1, n_y-1)           |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | 2               | (n_x-1, 0)              | (0, 0)                  | (n_x-1, n_y-1)           | (0, n_y-1)               |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | 3               | (0, n_y-1)              | (0, 0)                  | (n_x-1, n_y-1)           | (n_x-1, 0)               |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | 4               | (0, n_y-1)              | (n_x-1, n_y-1)          | (0, 0)                   | (n_x-1, 0)               |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | 5               | (n_x-1, 0)              | (n_x-1, n_y-1)          | (0, 0)                   | (0, n_y-1)               |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | 6               | (n_x-1, n_y-1)          | (0, n_y-1)              | (n_x-1, 0)               | (0, 0)                   |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | 7               | (n_x-1, n_y-1)          | (n_x-1, 0)              | (0, n_y-1)               | (0, 0)                   |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+

    The table above gives for the 4 corners of a image the corresponding vertex in the mesh.

    """
    # Set the UV mapping strategy (list of 3D points -> [(0,0) ; (Nx,0) ; (0,Ny) ; (Nx,Ny)])
    lower_left = numpy.array([0.0, 0.0])
    upper_left = numpy.array([0.0, 1.0])
    lower_right = numpy.array([1.0, 0.0])
    upper_right = numpy.array([1.0, 1.0])

    if uv_layout == 0:
        uv_mapping = [lower_left, lower_right, upper_left, upper_right]
    elif uv_layout == 1:
        uv_mapping = [lower_left, upper_left, lower_right, upper_right]
    elif uv_layout == 2:
        uv_mapping = [upper_left, lower_left, upper_right, lower_right]
    elif uv_layout == 3:
        uv_mapping = [upper_left, upper_right, lower_left, lower_right]
    elif uv_layout == 4:
        uv_mapping = [lower_right, upper_right, lower_left, upper_left]
    elif uv_layout == 5:
        uv_mapping = [lower_right, lower_left, upper_right, upper_left]
    elif uv_layout == 6:
        uv_mapping = [upper_right, lower_right, upper_left, lower_left]
    elif uv_layout == 7:
        uv_mapping = [upper_right, upper_left, lower_right, lower_left]

    # Compute the UV mapping coordinates
    fx = numpy.asarray(fx, dtype=numpy.float64)  # (n_x,)
    fy = numpy.asarray(fy, dtype=numpy.float64)  # (n_y,)
    uv_fx = fx[:, numpy.newaxis].repeat(fy.size, axis=1)  # (n_x, n_y)
    uv_fy = fy[numpy.newaxis, :].repeat(fx.size, axis=0)  # (n_x, n_y)

    U = (
        uv_mapping[0][0]
        + uv_fx * (uv_mapping[1][0] - uv_mapping[0][0])
        + uv_fy * (uv_mapping[2][0] - uv_mapping[0][0])
    )  # (n_x, n_y)
    V = (
        uv_mapping[0][1]
        + uv_fx * (uv_mapping[1][1] - uv_mapping[0][1])
        + uv_fy * (uv_mapping[2][1] - uv_mapping[0][1])
    )  # (n_x, n_y)
    UV = numpy.stack((U, V), axis=2)  # (n_x, n_y, 2)
    return UV


def _create_grid_quadrangle_4_mesh(
    vertices: ArrayLike,
    *,
    x_connect: bool = False,
    y_connect: bool = False,
    x_swap: bool = False,
    y_swap: bool = False,
    direct: bool = True,
    uv_layout: Integral = 0,
    uv_fx: Optional[numpy.ndarray] = None,
    uv_fy: Optional[numpy.ndarray] = None,
) -> Mesh:
    # Implementation of create_uniform_quadrangle_4_mesh with the UV fractions along the two axes of the grid
    # (n_x,) and (n_y,), or (n_x + 1,) and (n_y + 1,) for the connected axes (fictive vertex, fraction 1).
    # None: the "index" mode (uniform fractions)
    vertices = _as_array(vertices, "vertices", numpy.float64, shape=(None, None, None), shape_hint="(n_x, n_y, E)", finite=False)
    if vertices.shape[0] < 2 or vertices.shape[1] < 2 or vertices.shape[2] < 1:
        raise ValueError(
            f"Input 'vertices' must have a shape (n_x, n_y, E) with n_x, n_y >= 2 and E >= 1. "
            f"Got shape {vertices.shape}."
        )
    x_connect = _validate_bool(x_connect, "x_connect")
    y_connect = _validate_bool(y_connect, "y_connect")
    x_swap = _validate_bool(x_swap, "x_swap")
    y_swap = _validate_bool(y_swap, "y_swap")
    direct = _validate_bool(direct, "direct")
    uv_layout = _validate_uv_layout(uv_layout)
    if x_swap and not x_connect:
        raise ValueError("Input 'x_swap' can only be True if 'x_connect' is True.")
    if y_swap and not y_connect:
        raise ValueError("Input 'y_swap' can only be True if 'y_connect' is True.")

    # Initialize the connectivity and properties
    n_x, n_y, _ = vertices.shape
    n_x_quads = n_x - 1
    n_y_quads = n_y - 1
    if x_connect:
        n_x_quads += 1
    if y_connect:
        n_y_quads += 1
    quads = numpy.zeros((n_x_quads, n_y_quads, 4), dtype=numpy.int64)
    quads_uv = numpy.zeros((n_x_quads, n_y_quads, 4, 2), dtype=numpy.float64)

    # Create the uv mapping for the vertices (with the fictive column/row of the connected axes)
    uv_n_x = n_x + 1 if x_connect else n_x
    uv_n_y = n_y + 1 if y_connect else n_y
    uv_fx = _index_uv_fractions(n_x, x_connect) if uv_fx is None else uv_fx
    uv_fy = _index_uv_fractions(n_y, y_connect) if uv_fy is None else uv_fy
    if uv_fx.shape != (uv_n_x,) or uv_fy.shape != (uv_n_y,):
        raise ValueError(
            f"The UV fractions must have the shapes ({uv_n_x},) and ({uv_n_y},). "
            f"Got {uv_fx.shape} and {uv_fy.shape}."
        )
    extended_vertices_uv_map = _create_vertices_uvmap(uv_fx, uv_fy, uv_layout)  # (uv_n_x, uv_n_y, 2)
    vertices_uv_map = extended_vertices_uv_map[:n_x, :n_y]  # (n_x, n_y, 2)

    # Vertex index function
    #
    # Vertices indexing for nx = 5 and ny = 3:
    #
    # 10 -- 11 -- 12 -- 13 -- 14
    # |     |     |     |     |
    # 05 -- 06 -- 07 -- 08 -- 09
    # |     |     |     |     |
    # 00 -- 01 -- 02 -- 03 -- 04
    #
    # Element indexing for nx = 5 and ny = 3 without connectivity:
    #
    # O -- O -- O -- O -- O
    # | 05 | 06 | 07 | 08 |
    # O -- O -- O -- O -- O
    # | 01 | 02 | 03 | 04 |
    # O -- O -- O -- O -- O
    #
    # Then with x_connect = True:
    # Adding next 2 elements connecting the last column to the first column.
    # Then with y_connect = True:
    # Adding next 4 elements connecting the last row to the first row.
    # Then with x_connect = True and y_connect = True:
    # Adding next 1 element connecting the block.
    #

    index = lambda i_X, i_Y: i_Y * n_x + i_X
    idx = numpy.fromfunction(index, (n_x, n_y), dtype=int)  # (n_x, n_y)

    # Create the quadrilateral connectivity (no connect option)
    bottom_left_idx = idx[:-1, :-1]  # (n_x-1, n_y-1)
    bottom_right_idx = idx[1:, :-1]  # (n_x-1, n_y-1)
    top_left_idx = idx[:-1, 1:]  # (n_x-1, n_y-1)
    top_right_idx = idx[1:, 1:]  # (n_x-1, n_y-1)

    bottom_left_uv = vertices_uv_map[:-1, :-1]  # (n_x-1, n_y-1, 2)
    bottom_right_uv = vertices_uv_map[1:, :-1]  # (n_x-1, n_y-1, 2)
    top_left_uv = vertices_uv_map[:-1, 1:]  # (n_x-1, n_y-1, 2)
    top_right_uv = vertices_uv_map[1:, 1:]  # (n_x-1, n_y-1, 2)

    # All the quadrangles are built counterclockwise, then reversed at the end if not direct
    quads[0 : n_x - 1, 0 : n_y - 1] = numpy.stack(
        (bottom_left_idx, bottom_right_idx, top_right_idx, top_left_idx), axis=2
    )  # (n_x-1, n_y-1, 4)
    quads_uv[0 : n_x - 1, 0 : n_y - 1] = numpy.stack(
        (bottom_left_uv, bottom_right_uv, top_right_uv, top_left_uv), axis=2
    )  # (n_x-1, n_y-1, 4, 2)

    # Create the quadrilateral connectivity for the connect options
    if x_connect:
        bottom_left_idx = idx[-1, :-1]  # (n_y-1,)
        top_left_idx = idx[-1, 1:]  # (n_y-1)

        # Warging: uvmap -> use extended for fictive values
        bottom_left_uv = extended_vertices_uv_map[-2, : n_y - 1]  # (n_y-1, 2)
        top_left_uv = extended_vertices_uv_map[-2, 1:n_y]  # (n_y-1, 2)

        if not x_swap:
            bottom_right_idx = idx[0, :-1]  # (n_y-1)
            top_right_idx = idx[0, 1:]  # (n_y-1)
            bottom_right_uv = extended_vertices_uv_map[-1, : n_y - 1]  # (n_y-1, 2)
            top_right_uv = extended_vertices_uv_map[-1, 1:n_y]  # (n_y-1, 2)

        else:
            bottom_right_idx = idx[0, ::-1][:-1]  # (n_y-1)
            top_right_idx = idx[0, ::-1][1:]  # (n_y-1)
            bottom_right_uv = extended_vertices_uv_map[-1, ::-1][
                : n_y - 1
            ]  # (n_y-1, 2)
            top_right_uv = extended_vertices_uv_map[-1, ::-1][1:n_y]  # (n_y-1, 2)

        quads[n_x - 1, 0 : n_y - 1] = numpy.stack(
            (bottom_left_idx, bottom_right_idx, top_right_idx, top_left_idx), axis=1
        )  # (n_y-1, 4)
        quads_uv[n_x - 1, 0 : n_y - 1] = numpy.stack(
            (bottom_left_uv, bottom_right_uv, top_right_uv, top_left_uv), axis=1
        )  # (n_y-1, 4, 2)

    if y_connect:
        bottom_left_idx = idx[:-1, -1]  # (n_x-1,)
        bottom_right_idx = idx[1:, -1]  # (n_x-1)

        # Warging: uvmap -> use extended for fictive values
        bottom_left_uv = extended_vertices_uv_map[: n_x - 1, -2]  # (n_x-1, 2)
        bottom_right_uv = extended_vertices_uv_map[1:n_x, -2]  # (n_x-1, 2)

        if not y_swap:
            top_left_idx = idx[:-1, 0]  # (n_x-1)
            top_right_idx = idx[1:, 0]  # (n_x-1)
            top_left_uv = extended_vertices_uv_map[: n_x - 1, -1]  # (n_x-1, 2)
            top_right_uv = extended_vertices_uv_map[1:n_x, -1]  # (n_x-1, 2)

        else:
            top_left_idx = idx[::-1, 0][:-1]  # (n_x-1)
            top_right_idx = idx[::-1, 0][1:]  # (n_x-1)
            top_left_uv = extended_vertices_uv_map[::-1, -1][: n_x - 1]  # (n_x-1, 2)
            top_right_uv = extended_vertices_uv_map[::-1, -1][1:n_x]  # (n_x-1, 2)

        quads[0 : n_x - 1, n_y - 1] = numpy.stack(
            (bottom_left_idx, bottom_right_idx, top_right_idx, top_left_idx), axis=1
        )  # (n_x-1, 4)
        quads_uv[0 : n_x - 1, n_y - 1] = numpy.stack(
            (bottom_left_uv, bottom_right_uv, top_right_uv, top_left_uv), axis=1
        )  # (n_x-1, 4, 2)

    if x_connect and y_connect:
        bottom_left_idx = idx[-1, -1]  # ()
        bottom_right_idx = idx[0, -1]  # ()
        top_left_idx = idx[-1, 0]  # ()
        top_right_idx = idx[0, 0]  # ()

        # Warging: uvmap -> use extended for fictive values
        bottom_left_uv = extended_vertices_uv_map[-2, -2]  # (2,)
        bottom_right_uv = extended_vertices_uv_map[-1, -2]  # (2,)
        top_left_uv = extended_vertices_uv_map[-2, -1]  # (2,)
        top_right_uv = extended_vertices_uv_map[-1, -1]  # (2,)

        quads[n_x - 1, n_y - 1] = numpy.array(
            [bottom_left_idx, bottom_right_idx, top_right_idx, top_left_idx]
        )  # (4,)
        quads_uv[n_x - 1, n_y - 1] = numpy.array(
            [bottom_left_uv, bottom_right_uv, top_right_uv, top_left_uv]
        )  # (4, 2)

    # Clockwise order (bottom-left, top-left, top-right, bottom-right), also for the connect elements
    if not direct:
        quads = quads[..., [0, 3, 2, 1]]
        quads_uv = quads_uv[..., [0, 3, 2, 1], :]

    # Reshape the connectivity and properties
    quads = quads.reshape(-1, 4, order="F")  # (n_quads, 4)
    quads_uv = quads_uv.reshape(-1, 4, 2, order="F")  # (n_quads, 4, 2)
    quads_uv = quads_uv.reshape(-1, 8)  # (n_quads, 8) -> (u1,v1,u2,v2,u3,v3,...)
    vertices = vertices.reshape(-1, vertices.shape[2], order="F")  # (n_x * n_y, E)
    mesh = Mesh(vertices, quads, element_type="quadrangle_4")
    mesh.uvmap = quads_uv
    return mesh


def _heightmap_mesh(
    function: Optional[Callable],
    x_bounds, y_bounds, x_array, y_array, n_x, n_y, direct, uv_layout, uv_mode,
) -> Mesh:
    # quadrangle_4 mesh of a height surface z = f(x, y) (grid (n_x, n_y)), in the local frame
    uv_mode = _validate_uv_mode(uv_mode, _UV_MODES)
    x = _grid_axis(x_bounds, x_array, n_x, ("x_bounds", "x_array", "n_x"))
    y = _grid_axis(y_bounds, y_array, n_y, ("y_bounds", "y_array", "n_y"))
    x_grid, y_grid = numpy.meshgrid(x, y, indexing="ij")  # (n_x, n_y)
    z_grid = _evaluate(function, 0.0, x_grid, y_grid, "function")
    uv_fx = _uv_fractions(uv_mode, x, False, "x (x_bounds or x_array)")
    uv_fy = _uv_fractions(uv_mode, y, False, "y (y_bounds or y_array)")
    return _create_grid_quadrangle_4_mesh(
        numpy.stack((x_grid, y_grid, z_grid), axis=2), direct=direct, uv_layout=uv_layout,
        uv_fx=uv_fx, uv_fy=uv_fy,
    )


def _cylindrical_mesh(
    function: Optional[Callable],
    t_bounds, h_bounds, t_array, h_array, n_t, n_h, direct, close, uv_layout, uv_mode,
) -> Tuple[Mesh, int, int]:
    # quadrangle_4 mesh of a cylindrical surface r = f(h, theta) (grid (n_t, n_h)) in the local frame, n_t, n_h
    close = _validate_bool(close, "close")
    uv_mode = _validate_uv_mode(uv_mode, _UV_MODES)
    t = _grid_axis(t_bounds, t_array, n_t, ("t_bounds", "t_array", "n_t"))
    h = _grid_axis(h_bounds, h_array, n_h, ("h_bounds", "h_array", "n_h"))
    t_grid, h_grid = numpy.meshgrid(t, h, indexing="ij")  # (n_t, n_h)
    r_grid = _evaluate(function, 1.0, h_grid, t_grid, "function")
    vertices = numpy.stack((r_grid * numpy.cos(t_grid), r_grid * numpy.sin(t_grid), h_grid), axis=2)
    uv_fx = _uv_fractions(uv_mode, t, close, "theta (t_bounds or t_array)", period=2 * numpy.pi)
    uv_fy = _uv_fractions(uv_mode, h, False, "h (h_bounds or h_array)")
    mesh = _create_grid_quadrangle_4_mesh(
        vertices, x_connect=close, direct=direct, uv_layout=uv_layout, uv_fx=uv_fx, uv_fy=uv_fy,
    )
    return mesh, t.size, h.size


def _spherical_mesh(
    function: Optional[Callable],
    t_bounds, p_bounds, t_array, p_array, n_t, n_p, direct, close, uv_layout, uv_mode,
) -> Tuple[Mesh, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    # quadrangle_4 mesh of a spherical surface r = f(theta, phi) (grid (n_p, n_t)) in the local frame, the
    # polar angles t (n_t,), the azimuthal angles p (n_p,) and the UV map of the grid (n_p (+1), n_t, 2)
    close = _validate_bool(close, "close")
    uv_mode = _validate_uv_mode(uv_mode, _UV_MODES_ANGLE)
    t = _grid_axis(t_bounds, t_array, n_t, ("t_bounds", "t_array", "n_t"))
    p = _grid_axis(p_bounds, p_array, n_p, ("p_bounds", "p_array", "n_p"))
    if numpy.any(p < 0) or numpy.any(p > 2 * numpy.pi):
        raise ValueError("The azimuthal angles (p_bounds or p_array) must be between 0 and 2 pi.")
    p_grid, t_grid = numpy.meshgrid(p, t, indexing="ij")  # (n_p, n_t)
    r_grid = _evaluate(function, 1.0, t_grid, p_grid, "function")
    vertices = numpy.stack(
        (
            r_grid * numpy.sin(t_grid) * numpy.cos(p_grid),
            r_grid * numpy.sin(t_grid) * numpy.sin(p_grid),
            r_grid * numpy.cos(t_grid),
        ),
        axis=2,
    )
    # parametric-angle: (phi, psi) with the isometric latitude psi, conformal on the unit sphere (radius ignored)
    uv_fx = _uv_fractions(uv_mode, p, close, "phi (p_bounds or p_array)", period=2 * numpy.pi)
    uv_fy = _uv_fractions(uv_mode, t, False, "theta (t_bounds or t_array)", conformal=_isometric_latitude)
    mesh = _create_grid_quadrangle_4_mesh(
        vertices, x_connect=close, direct=direct, uv_layout=uv_layout, uv_fx=uv_fx, uv_fy=uv_fy,
    )
    return mesh, t, p, _create_vertices_uvmap(uv_fx, uv_fy, uv_layout)


def create_uniform_quadrangle_4_mesh(
    vertices: ArrayLike,
    x_connect: bool = False,
    y_connect: bool = False,
    x_swap: bool = False,
    y_swap: bool = False,
    direct: bool = True,
    uv_layout: Integral = 0,
) -> Mesh:
    r"""
    Create the quadrilateral connectivity and properties for a uniform grid mesh.

    The mesh is defined in a local :math:`(n_x, n_y)` grid of vertices,
    where :math:`n_x` is the number of vertices along the x direction
    and :math:`n_y` is the number of vertices along the y direction.

    If ``x_connect`` is True, the mesh is connected in the x direction (cylindrical topology)
    by adding a fictive column of vertices at :math:`x = n_x` that is equal
    to the first column of vertices at :math:`x = 0` and connecting them with quadrilaterals.
    If ``x_swap`` is True, the order of the vertices in the x direction is reversed
    during the ``x_connect`` operation, so the vertex at :math:`x = n_x, y = i`
    is connected to the vertex at :math:`x = 0, y = n_y - 1 - i` instead of :math:`x = 0, y = i`.

    The same applies for the y direction with the ``y_connect`` and ``y_swap`` options.

    .. figure:: /_static/meshes/uniform_surface_connectivity.png
        :width: 400
        :align: center

        :math:`(n_x, n_y)` grid of vertices and the corresponding quadrilateral connectivity
        and selected ``x_connect``, ``y_connect``, ``x_swap`` and ``y_swap`` options.

    If ``direct`` is True, the quadrilaterals are defined in counterclockwise order for
    an observer looking from above, otherwise they are defined in clockwise order.

    - counterclockwise order for ``direct=True``: :math:`(i_X, i_Y), (i_X + 1, i_Y), (i_X + 1, i_Y + 1), (i_X, i_Y + 1)`
    - clockwise order for ``direct=False``: :math:`(i_X, i_Y), (i_X, i_Y + 1), (i_X + 1, i_Y + 1), (i_X + 1, i_Y)`

    Vertices are ordered first along the x direction, then along the y direction.
    So the vertex at y index ``i_Y`` and x index ``i_X`` (both starting from 0) is located at:

    .. code-block:: python

        mesh.vertices[i_Y * n_x + i_X, :]


    Parameters
    ----------
    vertices : ArrayLike
        An array of shape :math:`(n_x, n_y, E)` containing the vertex positions in E-dimensional space.

    x_connect : bool, optional (default: ``False``)
        If True, connect the mesh in the x direction by adding a fictive column of vertices
        at :math:`x = n_x` that is equal to the first column of vertices at :math:`x = 0` and connecting them with quadrilaterals.

    y_connect : bool, optional (default: ``False``)
        If True, connect the mesh in the y direction by adding a fictive row of vertices
        at :math:`y = n_y` that is equal to the first row of vertices at :math:`y = 0` and connecting them with quadrilaterals.

    x_swap : bool, optional (default: ``False``)
        If True, reverse the order of the vertices in the x direction during the ``x_connect`` operation,
        so the vertex at :math:`x = n_x, y = i` is connected to the vertex at :math:`x = 0, y = n_y - 1 - i` instead of :math:`x = 0, y = i`.

    y_swap : bool, optional (default: ``False``)
        If True, reverse the order of the vertices in the y direction during the ``y_connect`` operation,
        so the vertex at :math:`x = i, y = n_y` is connected to the vertex at :math:`x = n_x - 1 - i, y = 0` instead of :math:`x = i, y = 0`.

    direct : bool, optional (default: ``True``)
        If True, the quadrilaterals are defined in counterclockwise order for an observer looking from above, otherwise they are defined in clockwise order.

    uv_layout : Integral, optional (default: ``0``)
        The UV mapping strategy. Must be an integer between 0 and 7.


    Returns
    -------
    :class:`pysdic.Mesh`
        The generated quadrilateral mesh as a :class:`pysdic.Mesh` object with the specified connectivity and UV mapping.


    UV Mapping
    ----------

    The UV coordinates are generated based on the vertex indices in the grid and uniformly distributed in the range [0, 1] for the ``OpenGL texture mapping convention`` (See https://learnopengl.com/Getting-started/Textures).
    They do not depend on the positions of the vertices: this is the ``uv_mode="index"`` of the mesh generators
    (for example :func:`pysdic.create_3d_quadrangle_4_heightmap`), whose default ``uv_mode="parametric"`` follows
    the coordinates of the grid instead.
    Several UV mapping strategies are available and synthesized in the ``uv_layout`` parameter.

    An image is defined by its four corners:

    - Lower-left corner : pixel with array coordinates ``image[height-1, 0]`` but OpenGL convention is (0,0) at lower-left
    - Upper-left corner : pixel with array coordinates ``image[0, 0]`` but OpenGL convention is (0,1) at upper-left
    - Lower-right corner : pixel with array coordinates ``image[height-1, width-1]`` but OpenGL convention is (1,0) at lower-right
    - Upper-right corner : pixel with array coordinates ``image[0, width-1]`` but OpenGL convention is (1,1) at upper-right

    The following options are available for ``uv_layout`` and their corresponding vertex mapping:

    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | uv_layout       | Vertex lower-left corner| Vertex upper-left corner| Vertex lower-right corner| Vertex upper-right corner|
    +=================+=========================+=========================+==========================+==========================+
    | 0               | (0, 0)                  | (0, n_y-1)              | (n_x-1, 0)               | (n_x-1, n_y-1)           |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | 1               | (0, 0)                  | (n_x-1, 0)              | (0, n_y-1)               | (n_x-1, n_y-1)           |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | 2               | (n_x-1, 0)              | (0, 0)                  | (n_x-1, n_y-1)           | (0, n_y-1)               |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | 3               | (0, n_y-1)              | (0, 0)                  | (n_x-1, n_y-1)           | (n_x-1, 0)               |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | 4               | (0, n_y-1)              | (n_x-1, n_y-1)          | (0, 0)                   | (n_x-1, 0)               |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | 5               | (n_x-1, 0)              | (n_x-1, n_y-1)          | (0, 0)                   | (0, n_y-1)               |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | 6               | (n_x-1, n_y-1)          | (0, n_y-1)              | (n_x-1, 0)               | (0, 0)                   |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | 7               | (n_x-1, n_y-1)          | (n_x-1, 0)              | (0, n_y-1)               | (0, 0)                   |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+

    The table above gives for the 4 corners of a image the corresponding vertex in the mesh.

    With ``uv_layout`` in 0, 3, 5 and 7, the x axis of the grid is along :math:`u` (image width) and the y axis
    along :math:`v` (image height). With ``uv_layout`` in 1, 2, 4 and 6, the axes are swapped.

    """
    return _create_grid_quadrangle_4_mesh(
        vertices, x_connect=x_connect, y_connect=y_connect, x_swap=x_swap, y_swap=y_swap,
        direct=direct, uv_layout=uv_layout,
    )


def create_3d_quadrangle_4_heightmap(
    function: Optional[Callable[[numpy.ndarray, numpy.ndarray], ArrayLike]] = None,
    *,
    frame: Optional[py3dframe.Frame] = None,
    x_bounds: Optional[Tuple[Real, Real]] = None,
    y_bounds: Optional[Tuple[Real, Real]] = None,
    x_array: Optional[ArrayLike] = None,
    y_array: Optional[ArrayLike] = None,
    n_x: Optional[Integral] = None,
    n_y: Optional[Integral] = None,
    direct: bool = True,
    uv_layout: Integral = 0,
    uv_mode: str = "parametric",
) -> Mesh:
    r"""
    Create a 3D ``quadrangle_4`` mesh of a height surface :math:`z = f(x, y)`.

    The grid of vertices is given, for each axis, either by bounds and a number of
    vertices (uniform grid), or by an array of coordinates.

    .. important::

        ``n_x`` and ``n_y`` are numbers of **vertices**, not elements.

    The vertex at the indices ``i_x`` along :math:`x` and ``i_y`` along :math:`y` (both
    starting from 0) is ``mesh.coordinates[i_y * n_x + i_x]``.

    Parameters
    ----------
    function: Optional[Callable[[numpy.ndarray, numpy.ndarray], ArrayLike]], optional
        The height :math:`f(x, y)`, called once with the 2D grids of :math:`x` and
        :math:`y` (shape :math:`(n_x, n_y)`). It must return a scalar or an array of
        the same shape. By default, :math:`f = 0` (flat mesh).

    frame: Optional[:class:`py3dframe.Frame`], optional (default: ``None``)
        The local frame of the surface: the vertices are computed in this frame and
        expressed in the canonical frame. By default, the canonical frame. Requires the
        optional package ``py3dframe``.

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
        If True, the vertices of the quadrangles are ordered counterclockwise seen from
        above (:math:`+z`), otherwise clockwise.

    uv_layout: Integral, optional (default: ``0``)
        The UV mapping strategy, between 0 and 7 (see
        :func:`pysdic.create_uniform_quadrangle_4_mesh`).

    uv_mode: str, optional (default: ``"parametric"``)
        How the UV coordinates are distributed between the corners of the grid:

        - ``"parametric"``: proportional to the coordinates :math:`x` and :math:`y` of the
          vertices (projection of the surface on the plane :math:`z = 0`, independent of
          :math:`f`). The texture does not depend on the discretization of the grid. The
          coordinates of the grid must be strictly monotonic.
        - ``"index"``: proportional to the indices of the vertices (uniform in the grid,
          whatever its coordinates).

        Both modes give the same UV map for a uniform grid.

    Returns
    -------
    :class:`pysdic.Mesh`
        The ``quadrangle_4`` mesh, with a UV map.

    Examples
    --------
    .. code-block:: python
        :linenos:

        import numpy
        from pysdic import create_3d_quadrangle_4_heightmap

        mesh = create_3d_quadrangle_4_heightmap(
            lambda x, y: numpy.sin(x) * numpy.cos(y),
            x_bounds=(0, numpy.pi), y_bounds=(0, numpy.pi), n_x=10, n_y=20,
        )
        print(mesh.n_vertices, mesh.n_elements)  # 200 171

    """
    mesh = _heightmap_mesh(function, x_bounds, y_bounds, x_array, y_array, n_x, n_y, direct, uv_layout, uv_mode)
    return _local_frame_mesh(mesh, frame, "create_3d_quadrangle_4_heightmap")


def create_3d_triangle_3_heightmap(
    function: Optional[Callable[[numpy.ndarray, numpy.ndarray], ArrayLike]] = None,
    *,
    frame: Optional[py3dframe.Frame] = None,
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
    uv_mode: str = "parametric",
) -> Mesh:
    r"""
    Create a 3D ``triangle_3`` mesh of a height surface :math:`z = f(x, y)`.

    The ``quadrangle_4`` mesh of :func:`pysdic.create_3d_quadrangle_4_heightmap` is split
    into triangles with :func:`pysdic.quadrangle_4_to_triangle_3` (the quadrangle
    :math:`q` gives the triangles :math:`2q` and :math:`2q + 1`). The vertices are the
    same.

    Parameters
    ----------
    function, frame, x_bounds, y_bounds, x_array, y_array, n_x, n_y, direct, uv_layout, uv_mode:
        See :func:`pysdic.create_3d_quadrangle_4_heightmap`.

    diagonal: str, optional (default: ``"first"``)
        The diagonal used to split the quadrangles: ``"first"``, ``"second"``,
        ``"alternate"`` or ``"random"``.

    seed: Optional[Union[Integral, numpy.random.Generator]], optional (default: ``None``)
        The seed of ``diagonal="random"``.

    Returns
    -------
    :class:`pysdic.Mesh`
        The ``triangle_3`` mesh, with a UV map.

    """
    mesh = create_3d_quadrangle_4_heightmap(
        function, frame=frame, x_bounds=x_bounds, y_bounds=y_bounds, x_array=x_array,
        y_array=y_array, n_x=n_x, n_y=n_y, direct=direct, uv_layout=uv_layout, uv_mode=uv_mode,
    )
    return quadrangle_4_to_triangle_3(mesh, diagonal, seed=seed)


def create_3d_quadrangle_4_cylindrical(
    function: Optional[Callable[[numpy.ndarray, numpy.ndarray], ArrayLike]] = None,
    *,
    frame: Optional[py3dframe.Frame] = None,
    t_bounds: Optional[Tuple[Real, Real]] = None,
    h_bounds: Optional[Tuple[Real, Real]] = None,
    t_array: Optional[ArrayLike] = None,
    h_array: Optional[ArrayLike] = None,
    n_t: Optional[Integral] = None,
    n_h: Optional[Integral] = None,
    direct: bool = True,
    close: bool = False,
    uv_layout: Integral = 0,
    uv_mode: str = "parametric",
) -> Mesh:
    r"""
    Create a 3D ``quadrangle_4`` mesh of a cylindrical surface with radius
    :math:`r = f(h, \theta)`.

    The axis of the cylinder is the :math:`z`-axis of the frame and the angle
    :math:`\theta` is measured from its :math:`x`-axis: the vertex at height :math:`h`
    and angle :math:`\theta` is :math:`(r \cos\theta, r \sin\theta, h)`.

    The grid of vertices is given, for each axis, either by bounds and a number of
    vertices (uniform grid), or by an array of coordinates.

    .. important::

        ``n_t`` and ``n_h`` are numbers of **vertices**, not elements.

    The vertex at the indices ``i_t`` along :math:`\theta` and ``i_h`` along :math:`h`
    (both starting from 0) is ``mesh.coordinates[i_h * n_t + i_t]``.

    If ``close`` is True, the last and first angles are connected by one more column of
    elements. For a uniform closed grid, use ``t_bounds=(0, 2 * pi * (1 - 1 / n_t))``.

    Parameters
    ----------
    function: Optional[Callable[[numpy.ndarray, numpy.ndarray], ArrayLike]], optional
        The radius :math:`f(h, \theta)`, called once with the 2D grids of :math:`h` and
        :math:`\theta` (shape :math:`(n_t, n_h)`). It must return a scalar or an array
        of the same shape. For an axisymmetric surface, ignore the second argument
        (e.g. ``lambda h, t: 1 + h**2``). By default, :math:`f = 1`.

    frame: Optional[:class:`py3dframe.Frame`], optional (default: ``None``)
        The local frame of the surface (see above). By default, the canonical frame.
        Requires the optional package ``py3dframe``.

    t_bounds, h_bounds: Optional[Tuple[Real, Real]], optional
        The bounds of the uniform grid along :math:`\theta` and :math:`h` (with ``n_t``
        and ``n_h``).

    t_array, h_array: Optional[ArrayLike], optional
        The coordinates of the grid along :math:`\theta` and :math:`h` (instead of the
        bounds), with at least 2 values.

    n_t, n_h: Optional[Integral], optional
        The numbers of vertices along :math:`\theta` and :math:`h` (:math:`\geq 2`).
        Required with the bounds, checked against the arrays if given.

    direct: bool, optional (default: ``True``)
        If True, the vertices of the quadrangles are ordered counterclockwise in the
        :math:`(\theta, h)` plane, otherwise clockwise.

    close: bool, optional (default: ``False``)
        If True, the mesh is closed along :math:`\theta`.

    uv_layout: Integral, optional (default: ``0``)
        The UV mapping strategy, between 0 and 7 (see
        :func:`pysdic.create_uniform_quadrangle_4_mesh`).

    uv_mode: str, optional (default: ``"parametric"``)
        How the UV coordinates are distributed between the corners of the grid:

        - ``"parametric"``: proportional to :math:`\theta` and :math:`h` (independent of
          the radius :math:`f`). The texture does not depend on the discretization of the
          grid. The coordinates of the grid must be strictly monotonic. If ``close`` is
          True, the angles must span less than :math:`2\pi` and the UV coordinate along
          :math:`\theta` is :math:`(\theta - \theta_0) / (\pm 2\pi)`: the closing column of
          elements goes up to the UV coordinate 1.
        - ``"index"``: proportional to the indices of the vertices (uniform in the grid,
          whatever its coordinates).

        Both modes give the same UV map for a uniform grid (closed with
        ``t_bounds=(0, 2 * pi * (1 - 1 / n_t))``). For a unit radius, ``"parametric"``
        preserves the lengths up to the scale of each axis.

    Returns
    -------
    :class:`pysdic.Mesh`
        The ``quadrangle_4`` mesh, with a UV map.

    """
    mesh, _, _ = _cylindrical_mesh(
        function, t_bounds, h_bounds, t_array, h_array, n_t, n_h, direct, close, uv_layout, uv_mode,
    )
    return _local_frame_mesh(mesh, frame, "create_3d_quadrangle_4_cylindrical")


def create_3d_triangle_3_cylindrical(
    function: Optional[Callable[[numpy.ndarray, numpy.ndarray], ArrayLike]] = None,
    *,
    frame: Optional[py3dframe.Frame] = None,
    t_bounds: Optional[Tuple[Real, Real]] = None,
    h_bounds: Optional[Tuple[Real, Real]] = None,
    t_array: Optional[ArrayLike] = None,
    h_array: Optional[ArrayLike] = None,
    n_t: Optional[Integral] = None,
    n_h: Optional[Integral] = None,
    direct: bool = True,
    close: bool = False,
    stick: bool = False,
    stick_eps: Real = 1e-6,
    diagonal: str = "first",
    seed: Optional[Union[Integral, numpy.random.Generator]] = None,
    uv_layout: Integral = 0,
    uv_mode: str = "parametric",
) -> Mesh:
    r"""
    Create a 3D ``triangle_3`` mesh of a cylindrical surface with radius
    :math:`r = f(h, \theta)`.

    The ``quadrangle_4`` mesh of :func:`pysdic.create_3d_quadrangle_4_cylindrical` is split
    into triangles with :func:`pysdic.quadrangle_4_to_triangle_3`.

    If ``stick`` is True, the vertices on the axis (radius lower than ``stick_eps``)
    are merged: at each height, they are replaced by a single vertex on the axis. The
    triangles that become degenerated are removed, and the unused vertices too, so the
    indexing of the vertices of :func:`pysdic.create_3d_quadrangle_4_cylindrical` is no
    longer valid. Use it to close the ends of a surface of revolution (e.g. a sphere
    with :math:`f(h) = \sqrt{1 - h^2}` and :math:`h \in [-1, 1]`).

    Parameters
    ----------
    function, frame, t_bounds, h_bounds, t_array, h_array, n_t, n_h, direct, close, uv_layout, uv_mode:
        See :func:`pysdic.create_3d_quadrangle_4_cylindrical`.

    stick: bool, optional (default: ``False``)
        If True, the vertices on the axis are merged (see above).

    stick_eps: Real, optional (default: ``1e-6``)
        The radius below which a vertex is on the axis (non-negative).

    diagonal: str, optional (default: ``"first"``)
        The diagonal used to split the quadrangles: ``"first"``, ``"second"``,
        ``"alternate"`` or ``"random"``.

    seed: Optional[Union[Integral, numpy.random.Generator]], optional (default: ``None``)
        The seed of ``diagonal="random"``.

    Returns
    -------
    :class:`pysdic.Mesh`
        The ``triangle_3`` mesh, with a UV map.

    """
    stick = _validate_bool(stick, "stick")
    stick_eps = _validate_real(stick_eps, "stick_eps", non_negative=True, finite=True)
    mesh, n_t, n_h = _cylindrical_mesh(
        function, t_bounds, h_bounds, t_array, h_array, n_t, n_h, direct, close, uv_layout, uv_mode,
    )
    mesh = quadrangle_4_to_triangle_3(mesh, diagonal, seed=seed)

    if stick:
        # Vertex index i_h * n_t + i_t (local frame): merge the vertices on the axis at each height
        coordinates = mesh.vertices._coordinates.copy()
        on_axis = numpy.linalg.norm(coordinates[:, :2], axis=1) < stick_eps  # (N_v,)
        on_axis_grid = on_axis.reshape((n_h, n_t))
        mapping = numpy.arange(mesh.n_vertices)
        for i_h in numpy.flatnonzero(on_axis_grid.any(axis=1)):
            merged = i_h * n_t + numpy.flatnonzero(on_axis_grid[i_h])
            mapping[merged] = merged[0]
        coordinates[on_axis, :2] = 0.0

        mesh.coordinates = coordinates
        mesh.elements = mapping[mesh.connectivity._elements]
        mesh = mesh.remove_degenerated_elements()  # The methods return a new mesh
        mesh = mesh.remove_unused_vertices()

    return _local_frame_mesh(mesh, frame, "create_3d_triangle_3_cylindrical")


def create_3d_quadrangle_4_spherical(
    function: Optional[Callable[[numpy.ndarray, numpy.ndarray], ArrayLike]] = None,
    *,
    frame: Optional[py3dframe.Frame] = None,
    t_bounds: Optional[Tuple[Real, Real]] = None,
    p_bounds: Optional[Tuple[Real, Real]] = None,
    t_array: Optional[ArrayLike] = None,
    p_array: Optional[ArrayLike] = None,
    n_t: Optional[Integral] = None,
    n_p: Optional[Integral] = None,
    direct: bool = True,
    close: bool = False,
    uv_layout: Integral = 0,
    uv_mode: str = "parametric",
) -> Mesh:
    r"""
    Create a 3D ``quadrangle_4`` mesh of a spherical surface with radius
    :math:`r = f(\theta, \phi)`.

    :math:`\theta` is the polar angle, measured from the :math:`z`-axis of the frame
    (between 0 and :math:`\pi`), and :math:`\phi` is the azimuthal angle, measured from
    its :math:`x`-axis (between 0 and :math:`2\pi`): the vertex is
    :math:`(r \sin\theta \cos\phi, r \sin\theta \sin\phi, r \cos\theta)`.

    .. important::

        ``n_t`` and ``n_p`` are numbers of **vertices**, not elements.

    The vertex at the indices ``i_t`` along :math:`\theta` and ``i_p`` along
    :math:`\phi` (both starting from 0) is ``mesh.coordinates[i_t * n_p + i_p]``.

    If ``close`` is True, the last and first azimuthal angles are connected by one more
    column of elements. For a uniform closed grid, use
    ``p_bounds=(0, 2 * pi * (1 - 1 / n_p))``. The mesh is not closed at the poles: use
    :func:`pysdic.create_3d_triangle_3_spherical` for a closed sphere.

    The UV coordinates depend on ``uv_mode``:

    - ``"parametric"`` (default): proportional to :math:`\phi` and :math:`\theta`
      (equirectangular projection). The texture does not depend on the discretization of
      the grid, but it is stretched along :math:`\phi` by :math:`1 / \sin\theta` near the
      poles.
    - ``"parametric-angle"``: proportional to :math:`\phi` and to the isometric latitude
      :math:`\psi = \ln \tan(\theta / 2)` (Mercator projection), which preserves the angles
      on the unit sphere. The local scale :math:`\sin\theta` is the same along both axes,
      so the texture is not stretched in one direction, only scaled (smaller near the
      poles). The polar angles must be strictly between 0 and :math:`\pi`.
    - ``"index"``: proportional to the indices of the vertices (uniform in the grid,
      whatever its coordinates).

    With ``"parametric"`` and ``"parametric-angle"``, the coordinates of the grid must be
    strictly monotonic, and if ``close`` is True, the azimuthal angles must span less than
    :math:`2\pi` and the UV coordinate along :math:`\phi` is
    :math:`(\phi - \phi_0) / (\pm 2\pi)` (the closing column of elements goes up to the
    UV coordinate 1). The three modes give the same UV map for a uniform grid, except
    ``"parametric-angle"`` along :math:`\theta`.

    .. important::

        The UV square :math:`[0, 1]^2` is stretched onto the whole texture image, so with
        ``"parametric-angle"`` the angles are exactly preserved only if the image has the
        aspect ratio of the projected grid:
        :math:`\text{width} / \text{height} = L_\phi / L_\psi` with
        :math:`L_\phi = 2\pi` if ``close`` else :math:`|\phi_{n_p - 1} - \phi_0|`, and
        :math:`L_\psi = |\psi(\theta_{n_t - 1}) - \psi(\theta_0)|` (with ``uv_layout`` in
        0, 3, 5, 7; inverse ratio with ``uv_layout`` in 1, 2, 4, 6). Otherwise, the
        distortion is a uniform stretch of the whole texture along one axis.

    Parameters
    ----------
    function: Optional[Callable[[numpy.ndarray, numpy.ndarray], ArrayLike]], optional
        The radius :math:`f(\theta, \phi)`, called once with the 2D grids of
        :math:`\theta` and :math:`\phi` (shape :math:`(n_p, n_t)`). It must return a
        scalar or an array of the same shape. By default, :math:`f = 1`.

    frame: Optional[:class:`py3dframe.Frame`], optional (default: ``None``)
        The local frame of the surface (see above). By default, the canonical frame.
        Requires the optional package ``py3dframe``.

    t_bounds, p_bounds: Optional[Tuple[Real, Real]], optional
        The bounds of the uniform grid along :math:`\theta` and :math:`\phi` (with
        ``n_t`` and ``n_p``).

    t_array, p_array: Optional[ArrayLike], optional
        The coordinates of the grid along :math:`\theta` and :math:`\phi` (instead of
        the bounds), with at least 2 values.

    n_t, n_p: Optional[Integral], optional
        The numbers of vertices along :math:`\theta` and :math:`\phi` (:math:`\geq 2`).
        Required with the bounds, checked against the arrays if given.

    direct: bool, optional (default: ``True``)
        If True, the vertices of the quadrangles are ordered counterclockwise in the
        :math:`(\phi, \theta)` plane, otherwise clockwise.

    close: bool, optional (default: ``False``)
        If True, the mesh is closed along :math:`\phi`.

    uv_layout: Integral, optional (default: ``0``)
        The UV mapping strategy, between 0 and 7 (see
        :func:`pysdic.create_uniform_quadrangle_4_mesh`).

    uv_mode: str, optional (default: ``"parametric"``)
        How the UV coordinates are distributed between the corners of the grid:
        ``"parametric"``, ``"parametric-angle"`` or ``"index"`` (see above). The radius
        :math:`f` is never used: a change of radius is a deformation of the unit sphere
        that the texture follows.

    Returns
    -------
    :class:`pysdic.Mesh`
        The ``quadrangle_4`` mesh, with a UV map.

    """
    mesh, _, _, _ = _spherical_mesh(
        function, t_bounds, p_bounds, t_array, p_array, n_t, n_p, direct, close, uv_layout, uv_mode,
    )
    return _local_frame_mesh(mesh, frame, "create_3d_quadrangle_4_spherical")


def create_3d_triangle_3_spherical(
    function: Optional[Callable[[numpy.ndarray, numpy.ndarray], ArrayLike]] = None,
    *,
    frame: Optional[py3dframe.Frame] = None,
    t_bounds: Optional[Tuple[Real, Real]] = None,
    p_bounds: Optional[Tuple[Real, Real]] = None,
    t_array: Optional[ArrayLike] = None,
    p_array: Optional[ArrayLike] = None,
    n_t: Optional[Integral] = None,
    n_p: Optional[Integral] = None,
    direct: bool = True,
    close: bool = False,
    add_north: bool = True,
    add_south: bool = True,
    diagonal: str = "first",
    seed: Optional[Union[Integral, numpy.random.Generator]] = None,
    uv_layout: Integral = 0,
    uv_mode: str = "parametric",
) -> Mesh:
    r"""
    Create a 3D ``triangle_3`` mesh of a spherical surface with radius
    :math:`r = f(\theta, \phi)`, optionally closed at the poles.

    The ``quadrangle_4`` mesh of :func:`pysdic.create_3d_quadrangle_4_spherical` is split
    into triangles with :func:`pysdic.quadrangle_4_to_triangle_3` (the quadrangle
    :math:`q` gives the triangles :math:`2q` and :math:`2q + 1`). Then:

    - If ``add_north`` is True, a vertex is added at the north pole
      (:math:`\theta = 0`), with the radius :math:`f(0, \phi)` averaged over the
      azimuthal angles, and connected to the first ring of vertices (first polar angle)
      by a fan of triangles.
    - If ``add_south`` is True, a vertex is added at the south pole
      (:math:`\theta = \pi`) in the same way, and connected to the last ring of
      vertices.

    The pole triangles have the same orientation as the other triangles, and are added
    after them (one per azimuthal interval, :math:`n_p - 1`, or :math:`n_p` if ``close``
    is True). The vertices of the grid keep their indices
    (``mesh.coordinates[i_t * n_p + i_p]``), then come the north and the south poles.

    .. note::

        In the UV map, the pole vertex of each pole triangle takes the middle of the UV
        coordinates of the two other vertices: the texture is not stretched over the
        pole caps.

    Parameters
    ----------
    function, frame, t_bounds, p_bounds, t_array, p_array, n_t, n_p, direct, close, uv_layout, uv_mode:
        See :func:`pysdic.create_3d_quadrangle_4_spherical`. For a closed sphere, use polar
        angles strictly between 0 and :math:`\pi`.

    add_north: bool, optional (default: ``True``)
        If True, the north pole is added. The first polar angle must be greater than 0.

    add_south: bool, optional (default: ``True``)
        If True, the south pole is added. The last polar angle must be lower than
        :math:`\pi`.

    diagonal: str, optional (default: ``"first"``)
        The diagonal used to split the quadrangles: ``"first"``, ``"second"``,
        ``"alternate"`` or ``"random"``.

    seed: Optional[Union[Integral, numpy.random.Generator]], optional (default: ``None``)
        The seed of ``diagonal="random"``.

    Returns
    -------
    :class:`pysdic.Mesh`
        The ``triangle_3`` mesh, with a UV map.

    Examples
    --------
    A closed unit sphere.

    .. code-block:: python
        :linenos:

        import numpy
        from pysdic import create_3d_triangle_3_spherical

        n_p = 20
        mesh = create_3d_triangle_3_spherical(
            t_bounds=(0.1, numpy.pi - 0.1), n_t=10,
            p_bounds=(0, 2 * numpy.pi * (1 - 1 / n_p)), n_p=n_p, close=True,
        )
        print(mesh.n_vertices, mesh.n_elements)  # 202 400

    """
    close = _validate_bool(close, "close")
    direct = _validate_bool(direct, "direct")
    add_north = _validate_bool(add_north, "add_north")
    add_south = _validate_bool(add_south, "add_south")
    quadrangle_mesh, t, p, vertices_uv = _spherical_mesh(
        function, t_bounds, p_bounds, t_array, p_array, n_t, n_p, direct, close, uv_layout, uv_mode,
    )  # vertices_uv: (n_p (+1 if close), n_t, 2), the closing edge uses the fictive column n_p
    if add_north and not t[0] > 0:
        raise ValueError(f"The first polar angle must be greater than 0 to add the north pole. Got {t[0]}.")
    if add_south and not t[-1] < numpy.pi:
        raise ValueError(f"The last polar angle must be lower than pi to add the south pole. Got {t[-1]}.")

    mesh = quadrangle_4_to_triangle_3(quadrangle_mesh, diagonal, seed=seed)

    n_p, n_t = p.size, t.size
    coordinates = [mesh.vertices._coordinates]
    triangles = [mesh.connectivity._elements]
    uvmaps = [mesh.uvmap]

    # Ring edges (a, b) of the fans, with the UV coordinates of a and b
    i_p = numpy.arange(n_p if close else n_p - 1)
    next_p = (i_p + 1) % n_p
    next_uv = i_p + 1  # The closing edge uses the fictive column n_p of the UV grid
    poles = []
    if add_north:
        poles.append((0, 0.0))
    if add_south:
        poles.append((n_t - 1, numpy.pi))
    for i_t, theta in poles:
        radius = _evaluate(function, 1.0, numpy.full(n_p, theta), p, "function").mean()
        pole_index = mesh.n_vertices + len(coordinates) - 1
        coordinates.append(numpy.array([[0.0, 0.0, radius * numpy.cos(theta)]]))

        a, b = i_t * n_p + i_p, i_t * n_p + next_p
        uv_a, uv_b = vertices_uv[i_p, i_t], vertices_uv[next_uv, i_t]  # (n_fan, 2)
        uv_pole = 0.5 * (uv_a + uv_b)
        pole = numpy.full(i_p.size, pole_index)
        # Orientation of the neighboring quadrangles: the ring edge a -> b is used as
        # a -> b by the quadrangles of the first ring if direct, and of the last ring if
        # not direct, so the fan must use it as b -> a (and conversely).
        reverse = direct if i_t == 0 else not direct
        if reverse:
            triangles.append(numpy.stack((pole, b, a), axis=1))
            uvmaps.append(numpy.concatenate((uv_pole, uv_b, uv_a), axis=1))
        else:
            triangles.append(numpy.stack((pole, a, b), axis=1))
            uvmaps.append(numpy.concatenate((uv_pole, uv_a, uv_b), axis=1))

    mesh = Mesh(numpy.concatenate(coordinates), numpy.concatenate(triangles), element_type="triangle_3")
    mesh.uvmap = numpy.concatenate(uvmaps)
    return _local_frame_mesh(mesh, frame, "create_3d_triangle_3_spherical")