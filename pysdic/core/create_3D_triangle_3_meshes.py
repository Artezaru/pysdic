# Copyright 2025 Artezaru
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Callable, Optional, Tuple
from numbers import Integral

import numpy
from py3dframe import Frame, FrameTransform

from ..objects.mesh import Mesh


def create_triangle_3_heightmap(
    height_function: Callable[[float, float], float] = lambda x, y: 0.0,
    frame: Optional[Frame] = None,
    x_bounds: Tuple[float, float] = (-1.0, 1.0),
    y_bounds: Tuple[float, float] = (-1.0, 1.0),
    n_x: int = 10,
    n_y: int = 10,
    first_diagonal: bool = True,
    direct: bool = True,
    uv_layout: int = 0,
) -> Mesh:
    r"""
    Create a 3D :class:`Mesh` XY-plane mesh with variable height defined by a surface function.

    The surface is defined by a function that takes two arguments (x and y) and returns a scalar height z.
    The returned value is interpreted as the vertical position of the surface at that point.

    The :obj:`frame` parameter defines the orientation and the position of the mesh in 3D space.
    The (x, y) grid is centered on the frame origin, lying in the local XY plane.
    The height (z) is applied along the local Z-axis of the frame.

    The :obj:`x_bounds` and :obj:`y_bounds` parameters define the rectangular domain over which the mesh is generated.
    :obj:`n_x` and :obj:`n_y` determine the number of vertices along the x and y directions, respectively.
    Nodes are uniformly distributed across both directions.

    .. note::

        - :obj:`n_x` and :obj:`n_y` refer to the number of **vertices**, not segments.
        - The height function must return a finite scalar value for every (x, y) in the specified domain.

    For example, the following code generates a sinusoidal surface mesh centered on the world origin:

    .. code-block:: python

        from pysdic import create_triangle_3_heightmap
        import numpy as np

        surface_mesh = create_triangle_3_heightmap(
            height_function=lambda x, y: 0.5 * np.sin(np.pi * x) * np.cos(np.pi * y),
            x_bounds=(-1.0, 1.0),
            y_bounds=(-1.0, 1.0),
            n_x=50,
            n_y=50,
        )

        surface_mesh.visualize()

    .. figure:: /_static/meshes/create_triangle_3_heightmap_example.png
        :width: 400
        :align: center

        Sinusoidal height map over a square domain centered at the origin.

    Nodes are ordered first along the x direction, then along the y direction.
    So the vertex at y index ``i_Y`` and x index ``i_X`` (both starting from 0) is located at:

    .. code-block:: python

        mesh.vertices[i_Y * n_x + i_X, :]

    Each quadrilateral face is defined by the vertices:

    - :math:`(i_X, i_Y)`
    - :math:`(i_X + 1, i_Y)`
    - :math:`(i_X + 1, i_Y + 1)`
    - :math:`(i_X, i_Y + 1)`

    This quadrilateral is split into two triangles.

    .. seealso::

        - :class:`Mesh` for more information on how to visualize and manipulate the mesh.
        - https://github.com/Artezaru/py3dframe for details on the ``Frame`` class.

    Parameters
    ----------
    height_function : Callable[[float, float], float]
        A function that takes x and y coordinates and returns the corresponding height (z).

    frame : :class:`Frame`, optional
        The reference frame for the mesh. Defaults to the canonical frame.

    x_bounds : Tuple[:class:`float`, :class:`float`], optional
        The lower and upper bounds for the x coordinate. Default is :obj:`(-1.0, 1.0)`.

    y_bounds : Tuple[:class:`float`, :class:`float`], optional
        The lower and upper bounds for the y coordinate. Default is :obj:`(-1.0, 1.0)`.

    n_x : :class:`int`, optional
        Number of vertices along the x direction. Must be more than 1. Default is :obj:`10`.

    n_y : :class:`int`, optional
        Number of vertices along the y direction. Must be more than 1. Default is :obj:`10`.

    first_diagonal : :class:`bool`, optional
        If :obj:`True`, the quad is split along the first diagonal (bottom-left to top-right). Default is :obj:`True`.

    direct : :class:`bool`, optional
        If :obj:`True`, triangle vertices are ordered counterclockwise for an observer looking from above. Default is :obj:`True`.

    uv_layout : :class:`int`, optional
        The UV mapping strategy. Default is :obj:`0`.

    Returns
    -------
    :class:`Mesh`
        The generated surface mesh as a :class:`Mesh` object.


    Geometry of the mesh
    ------------------------

    Diagonal selection
    ~~~~~~~~~~~~~~~~~~~

    According to the :obj:`first_diagonal` parameter, each quadrilateral face is split into two triangles as follows:

    - If :obj:`first_diagonal` is :obj:`True`:

        - Triangle 1: :math:`(i_X, i_Y)`, :math:`(i_X + 1, i_Y)`, :math:`(i_X + 1, i_Y + 1)`
        - Triangle 2: :math:`(i_X, i_Y)`, :math:`(i_X + 1, i_Y + 1)`, :math:`(i_X, i_Y + 1)`

    - If :obj:`first_diagonal` is :obj:`False`:

        - Triangle 1: :math:`(i_X, i_Y)`, :math:`(i_X + 1, i_Y)`, :math:`(i_X, i_Y + 1)`
        - Triangle 2: :math:`(i_X, i_Y + 1)`, :math:`(i_X + 1, i_Y)`, :math:`(i_X + 1, i_Y + 1)`

    .. figure:: /_static/meshes/create_triangle_3_heightmap_diagonal.png
        :width: 400
        :align: center

        Diagonal selection for splitting quadrilaterals into triangles.

    Triangle orientation
    ~~~~~~~~~~~~~~~~~~~~~~

    - If :obj:`direct` is :obj:`True`, triangles are oriented counterclockwise for an observer looking from above (See the Diagonal selection section).
    - If :obj:`direct` is :obj:`False`, triangles are oriented clockwise for an observer looking from above. Switch the order of the last two vertices in each triangle defined above.

    UV Mapping
    ~~~~~~~~~~

    The UV coordinates are generated based on the vertex positions in the mesh and uniformly distributed in the range [0, 1] for the ``OpenGL texture mapping convention`` (See https://learnopengl.com/Getting-started/Textures).
    Several UV mapping strategies are available and synthesized in the :obj:`uv_layout` parameter.

    An image is defined by its four corners:

    - Lower-left corner : pixel with array coordinates ``image[height-1, 0]`` but OpenGL convention is (0,0) at lower-left
    - Upper-left corner : pixel with array coordinates ``image[0, 0]`` but OpenGL convention is (0,1) at upper-left
    - Lower-right corner : pixel with array coordinates ``image[height-1, width-1]`` but OpenGL convention is (1,0) at lower-right
    - Upper-right corner : pixel with array coordinates ``image[0, width-1]`` but OpenGL convention is (1,1) at upper-right

    The following options are available for :obj:`uv_layout` and their corresponding vertex mapping:

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

    .. figure:: /_static/meshes/create_triangle_3_heightmap_uv_layout.png
        :width: 400
        :align: center

        UV mapping strategies for different :obj:`uv_layout` options.

    To check the UV mapping, you can use the following code:

    .. code-block:: python

        import numpy as np
        from pysdic import create_triangle_3_heightmap
        import cv2

        surface_mesh = create_triangle_3_heightmap(
            height_function=lambda x, y: 0.5 * np.sin(np.pi * x) * np.cos(np.pi * y),
            x_bounds=(-1.0, 1.0),
            y_bounds=(-1.0, 1.0),
            n_x=50,
            n_y=50,
            uv_layout=3,
        )

        image = cv2.imread("lena_image.png")
        surface_mesh.visualize_texture(image)

    .. figure:: /_static/meshes/create_triangle_3_heightmap_lena_texture.png
        :width: 400
        :align: center

        UV mapping of the Lena image on a sinusoidal height map.

    """
    # Check the input parameters
    if frame is None:
        frame = Frame.canonical()
    if not isinstance(frame, Frame):
        raise TypeError("frames must be a Frame object")

    if not isinstance(height_function, Callable):
        raise TypeError("height_function must be a callable function")

    x_bounds = numpy.array(x_bounds, dtype=numpy.float64).flatten()
    if x_bounds.shape != (2,):
        raise ValueError("x_bounds must be a 2D array of shape (2,)")
    if x_bounds[0] == x_bounds[1]:
        raise ValueError("x_bounds must be different")

    y_bounds = numpy.array(y_bounds, dtype=numpy.float64).flatten()
    if y_bounds.shape != (2,):
        raise ValueError("y_bounds must be a 2D array of shape (2,)")
    if y_bounds[0] == y_bounds[1]:
        raise ValueError("y_bounds must be different")

    if not isinstance(n_x, Integral) or n_x < 2:
        raise ValueError("n_x must be an integer greater than 1")
    n_x = int(n_x)

    if not isinstance(n_y, Integral) or n_y < 2:
        raise ValueError("n_y must be an integer greater than 1")
    n_y = int(n_y)

    if not isinstance(first_diagonal, bool):
        raise TypeError("first_diagonal must be a boolean")

    if not isinstance(direct, bool):
        raise TypeError("direct must be a boolean")

    if not isinstance(uv_layout, Integral) or uv_layout < 0 or uv_layout > 7:
        raise ValueError("uv_layout must be an integer between 0 and 7")
    uv_layout = int(uv_layout)

    # Generate the transform
    transform = FrameTransform(input_frame=frame, output_frame=Frame.canonical())

    # Extract the parameters
    x_min = x_bounds[0]
    x_max = x_bounds[1]
    y_min = y_bounds[0]
    y_max = y_bounds[1]

    # Get the indices of the vertices in the array
    index = lambda ix, iy: iy * n_x + ix

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

    # Create the grid of points in local coordinates
    ix = numpy.arange(n_x)  # (n_x,)
    iy = numpy.arange(n_y)  # (n_y,)
    fx = ix / (n_x - 1)  # (n_x,)
    fy = iy / (n_y - 1)  # (n_y,)

    # Compute the local coordinates of each vertex
    X = x_min + (x_max - x_min) * fx  # (n_x, )
    Y = y_min + (y_max - y_min) * fy  # (n_y, )
    print(f"X: {X.shape}, Y: {Y.shape}")
    X = X[numpy.newaxis, :].repeat(n_y, axis=0)  # (n_y, n_x)
    Y = Y[:, numpy.newaxis].repeat(n_x, axis=1)  # (n_y, n_x)
    print(f"X: {X.shape}, Y: {Y.shape}")
    Z = height_function(X, Y)  # (n_y, n_x)
    print(f"Z: {Z.shape}")

    # Convert the local points to vertices coordinates using the frame transform
    local_coordinates = numpy.stack((X, Y, Z), axis=0)  # (3, n_y, n_x)
    local_coordinates = local_coordinates.reshape((3, n_x * n_y))  # (3, n_y*n_x)
    vertices = transform.transform(point=local_coordinates)  # (3, n_x*n_y)
    vertices = vertices.T  # (n_x*n_y, 3)

    # Compute the UV mapping coordinates
    uv_fx = fx[numpy.newaxis, :].repeat(n_y, axis=0)  # (n_y, n_x)
    uv_fy = fy[:, numpy.newaxis].repeat(n_x, axis=1)  # (n_y, n_x)

    U = (
        uv_mapping[0][0]
        + uv_fx * (uv_mapping[1][0] - uv_mapping[0][0])
        + uv_fy * (uv_mapping[2][0] - uv_mapping[0][0])
    )  # (n_y, n_x)
    V = (
        uv_mapping[0][1]
        + uv_fx * (uv_mapping[1][1] - uv_mapping[0][1])
        + uv_fy * (uv_mapping[2][1] - uv_mapping[0][1])
    )  # (n_y, n_x)
    vertices_uvmap = numpy.stack((U, V), axis=-1).reshape(
        (n_x * n_y, 2)
    )  # (n_x*n_y, 2)

    # Initialize the triangles and uvmap arrays
    triangles = numpy.zeros((2 * (n_x - 1) * (n_y - 1), 3), dtype=numpy.int64)
    triangles_uvmap = numpy.zeros(
        (2 * (n_x - 1) * (n_y - 1), 3, 2), dtype=numpy.float32
    )

    # Assemble the indexes in a vectorized way
    idx = iy[:, None] * n_x + ix[None, :]  # (n_y, n_x)

    # For iy in range(n_y - 1) and ix in range(n_x - 1): -> Build the triangles in the right order
    top_left_idx = idx[:-1, :-1]  # (n_y-1, n_x-1)
    top_right_idx = idx[:-1, 1:]  # (n_y-1, n_x-1)
    bottom_left_idx = idx[1:, :-1]  # (n_y-1, n_x-1)
    bottom_right_idx = idx[1:, 1:]  # (n_y-1, n_x-1)

    # Select the nodes index according the diagonal (n_y-1, n_x-1)
    if first_diagonal:
        n1, n2, n3 = top_left_idx, top_right_idx, bottom_right_idx
        n4, n5, n6 = top_left_idx, bottom_right_idx, bottom_left_idx
    else:
        n1, n2, n3 = top_left_idx, top_right_idx, bottom_left_idx
        n4, n5, n6 = bottom_left_idx, top_right_idx, bottom_right_idx

    # Build the triangles
    triangle_a = numpy.stack([n1, n2, n3], axis=-1)  # (n_y-1, n_x-1, 3)
    triangle_b = numpy.stack([n4, n5, n6], axis=-1)  # (n_y-1, n_x-1, 3)
    triangle_uv_a = numpy.stack(
        [vertices_uvmap[n1, :], vertices_uvmap[n2, :], vertices_uvmap[n3, :]], axis=-2
    )  # (n_y-1, n_x-1, 3, 2)
    triangle_uv_b = numpy.stack(
        [vertices_uvmap[n4, :], vertices_uvmap[n5, :], vertices_uvmap[n6, :]], axis=-2
    )  # (n_y-1, n_x-1, 3, 2)

    # Adjust orientation if needed
    if not direct:
        triangle_a = triangle_a[..., [0, 2, 1]]
        triangle_b = triangle_b[..., [0, 2, 1]]
        triangle_uv_a = triangle_uv_a[..., [0, 2, 1], :]
        triangle_uv_b = triangle_uv_b[..., [0, 2, 1], :]

    # Organize the final triangles array
    triangle_idx_a = 2 * (n1 - numpy.arange(n_y - 1)[:, None])  # (n_y-1, n_x-1)
    triangle_idx_b = 2 * (n1 - numpy.arange(n_y - 1)[:, None]) + 1  # (n_y-1, n_x-1)

    triangles[triangle_idx_a, :] = triangle_a
    triangles[triangle_idx_b, :] = triangle_b
    triangles_uvmap[triangle_idx_a, :, :] = triangle_uv_a
    triangles_uvmap[triangle_idx_b, :, :] = triangle_uv_b
    triangles_uvmap = triangles_uvmap.reshape(
        (triangles_uvmap.shape[0], 6)
    )  # (Ntriangles, 6) - (u1,v1,u2,v2,u3,v3)

    # Create the mesh
    mesh = Mesh(vertices=vertices, connectivity=triangles, element_type="triangle_3")

    # Set the UV map
    mesh.uvmap = triangles_uvmap

    return mesh


def create_triangle_3_axisymmetric(
    profile_curve: Callable[[float], float] = lambda _: 1.0,
    frame: Optional[Frame] = None,
    height_bounds: Tuple[float, float] = (0.0, 1.0),
    theta_bounds: Tuple[float, float] = (0.0, 2.0 * numpy.pi),
    n_height: int = 10,
    n_theta: int = 10,
    closed: bool = False,
    first_diagonal: bool = True,
    direct: bool = True,
    uv_layout: int = 0,
) -> Mesh:
    r"""
    Create a 3D axisymmetric mesh :class:`Mesh` using a given profile curve.

    The profile curve is a function that takes a single argument (height) and returns the radius at that height.
    The returned radius must be strictly positive for all z in the range defined by :obj:`height_bounds`.

    The :obj:`frame` parameter defines the orientation and the position of the mesh in 3D space.
    The axis of symmetry is aligned with the z-axis of the frame, and z=0 corresponds to the origin of the frame.
    The x-axis of the frame defines the direction of :math:`\theta=0`, and the y-axis defines the direction of :math:`\theta=\pi/2`.

    The :obj:`height_bounds` parameter defines the vertical extent of the mesh, and :obj:`theta_bounds` defines the angular sweep around the axis.
    :obj:`n_height` and :obj:`n_theta` determine the number of vertices in the height and angular directions, respectively.
    Nodes are uniformly distributed along both directions.

    .. note::

        - :obj:`n_height` and :obj:`n_theta` refer to the number of **vertices**, not segments.

    For example, the following code generates a mesh of a half-cylinder whose flat face is centered on the world x-axis:

    .. code-block:: python

        from pysdic import create_triangle_3_axisymmetric
        import numpy as np

        cylinder_mesh = create_triangle_3_axisymmetric(
            profile_curve=lambda z: 1.0,
            height_bounds=(-1.0, 1.0),
            theta_bounds=(-np.pi/4, np.pi/4),
            n_height=10,
            n_theta=20,
        )

        cylinder_mesh.visualize()

    .. figure:: /_static/meshes/create_triangle_3_axisymmetric_example.png
        :width: 400
        :align: center

        Demi-cylinder mesh with the face centered on the world x-axis.

    Nodes are ordered first in theta (indexed by ``i_T``) and then in height (indexed by ``i_H``).
    So the vertex at theta index ``i_T`` and height index ``i_H`` (both starting from 0) is located at:

    .. code-block:: python

        mesh.vertices[i_H * n_theta + i_T, :]

    Each quadrilateral element is defined by the vertices:

    - :math:`(i_T, i_H)`
    - :math:`(i_T + 1, i_H)`
    - :math:`(i_T + 1, i_H + 1)`
    - :math:`(i_T, i_H + 1)`

    This quadrilateral is split into two triangles.

    If :obj:`closed` is :obj:`True`, the mesh is closed in the angular direction by adding one more set of elements connecting the last and first theta positions.
    In that case, :obj:`theta_bounds` are ignored and the theta starts from 0 to :math:`2\pi(1 - 1/n_{theta})`.

    To generate a closed full cylinder:

    .. code-block:: python

        cylinder_mesh = create_triangle_3_axisymmetric(
            profile_curve=lambda z: 1.0,
            height_bounds=(-1.0, 1.0),
            n_height=10,
            n_theta=50,
            closed=True,
        )

    .. figure:: /_static/meshes/create_triangle_3_axisymmetric_example_closed.png
        :width: 400
        :align: center

        Closed cylinder mesh.

    .. seealso::

        - :class:`Mesh` for more information on how to visualize and manipulate the mesh.
        - https://github.com/Artezaru/py3dframe for details on the ``Frame`` class.

    Parameters
    ----------
    profile_curve : Callable[[float], float], optional
        A function that takes a single height coordinate z and returns a strictly positive radius.
        The default is a function that returns 1.0 for all z.

    frame : :class:`Frame`, optional
        The reference frame for the mesh. Defaults to the identity frame.

    height_bounds : Tuple[:class:`float`, :class:`float`], optional
        The lower and upper bounds for the height coordinate. Defaults to :obj:`(0.0, 1.0)`.
        The order determines the direction of vertex placement.

    theta_bounds : Tuple[:class:`float`, :class:`float`], optional
        The angular sweep in radians. Defaults to :obj:`(0.0, 2.0 * numpy.pi)`.
        The order determines the angular direction of vertex placement. If :obj:`closed` is :obj:`True`, this parameter is ignored.

    n_height : :class:`int`, optional
        Number of vertices along the height direction. Must be more than 1. Default is :obj:`10`.

    n_theta : :class:`int`, optional
        Number of vertices along the angular direction. Must be more than 1. Default is :obj:`10`.

    closed : :class:`bool`, optional
        If :obj:`True`, the mesh is closed in the angular direction. Default is :obj:`False`.

    first_diagonal : :class:`bool`, optional
        If :obj:`True`, the quad is split along the first diagonal (bottom-left to top-right).
        Default is :obj:`True`.

    direct : :class:`bool`, optional
        If :obj:`True`, triangle vertices are ordered counterclockwise for an observer looking from outside the radius. Default is :obj:`True`.

    uv_layout : :class:`int`, optional
        The UV mapping strategy. Default is :obj:`0`.

    Returns
    -------
    :class:`Mesh`
        The generated axisymmetric mesh as a :class:`Mesh` object.


    Geometry of the mesh
    ------------------------

    Diagonal selection
    ~~~~~~~~~~~~~~~~~~

    According to the :obj:`first_diagonal` parameter, each quadrilateral face is split into two triangles as follows:

    - If :obj:`first_diagonal` is :obj:`True`:

        - Triangle 1: :math:`(i_T, i_H)`, :math:`(i_T + 1, i_H)`, :math:`(i_T + 1, i_H + 1)`
        - Triangle 2: :math:`(i_T, i_H)`, :math:`(i_T + 1, i_H + 1)`, :math:`(i_T, i_H + 1)`

    - If :obj:`first_diagonal` is :obj:`False`:

        - Triangle 1: :math:`(i_T, i_H)`, :math:`(i_T + 1, i_H)`, :math:`(i_T, i_H + 1)`
        - Triangle 2: :math:`(i_T, i_H + 1)`, :math:`(i_T + 1, i_H)`, :math:`(i_T + 1, i_H + 1)`

    .. figure:: /_static/meshes/create_triangle_3_axisymmetric_diagonal.png
        :width: 400
        :align: center

        Diagonal selection for splitting quadrilaterals into triangles.

    Notice that if the mesh is closed, the last column of quadrilaterals connecting the last and first theta positions are added at the end of the connectivity list.

    Triangle orientation
    ~~~~~~~~~~~~~~~~~~~~~~

    - If :obj:`direct` is :obj:`True`, triangles are oriented counterclockwise for an observer looking from outside the radius (See the Diagonal selection section).
    - If :obj:`direct` is :obj:`False`, triangles are oriented clockwise for an observer looking from outside the radius. Switch the order of the last two vertices in each triangle defined above.

    UV Mapping
    ~~~~~~~~~~

    The UV coordinates are generated based on the vertex positions in the mesh and uniformly distributed in the range [0, 1] for the ``OpenGL texture mapping convention`` (See https://learnopengl.com/Getting-started/Textures).
    Several UV mapping strategies are available and synthesized in the :obj:`uv_layout` parameter.

    An image is defined by its four corners:

    - Lower-left corner : pitel with array coordinates ``image[height-1, 0]`` but OpenGL convention is (0,0) at lower-left
    - Upper-left corner : pitel with array coordinates ``image[0, 0]`` but OpenGL convention is (0,1) at upper-left
    - Lower-right corner : pitel with array coordinates ``image[height-1, width-1]`` but OpenGL convention is (1,0) at lower-right
    - Upper-right corner : pitel with array coordinates ``image[0, width-1]`` but OpenGL convention is (1,1) at upper-right

    The following options are available for :obj:`uv_layout` and their corresponding vertex mapping:

    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | uv_layout       | Vertex lower-left corner| Vertex upper-left corner| Vertex lower-right corner| Vertex upper-right corner|
    +=================+=========================+=========================+==========================+==========================+
    | 0               | (0, 0)                  | (0, n_h-1)              | (n_t-1, 0)               | (n_t-1, n_h-1)           |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | 1               | (0, 0)                  | (n_t-1, 0)              | (0, n_h-1)               | (n_t-1, n_h-1)           |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | 2               | (n_t-1, 0)              | (0, 0)                  | (n_t-1, n_h-1)           | (0, n_h-1)               |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | 3               | (0, n_h-1)              | (0, 0)                  | (n_t-1, n_h-1)           | (n_t-1, 0)               |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | 4               | (0, n_h-1)              | (n_t-1, n_h-1)          | (0, 0)                   | (n_t-1, 0)               |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | 5               | (n_t-1, 0)              | (n_t-1, n_h-1)          | (0, 0)                   | (0, n_h-1)               |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | 6               | (n_t-1, n_h-1)          | (0, n_h-1)              | (n_t-1, 0)               | (0, 0)                   |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | 7               | (n_t-1, n_h-1)          | (n_t-1, 0)              | (0, n_h-1)               | (0, 0)                   |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+

    Notice that for a closed mesh, the :obj:`n_t - 1` becames :obj:`n_t` in the table above, since the a 'virtal' last vertex is the same as the first one.

    The table above gives for the 4 corners of a image the corresponding vertex in the mesh.

    .. figure:: /_static/meshes/create_triangle_3_axisymmetric_uv_layout.png
        :width: 400
        :align: center

        UV mapping strategies for different :obj:`uv_layout` options.

    To check the UV mapping, you can use the following code:

    .. code-block:: python

        import numpy as np
        from pysdic import create_triangle_3_axisymmetric
        import cv2

        cylinder_mesh = create_triangle_3_axisymmetric(
            profile_curve=lambda z: 1.0,
            height_bounds=(-1.0, 1.0),
            theta_bounds=(0, np.pi/2),
            n_height=10,
            n_theta=20,
            uv_layout=3,
        )

        image = cv2.imread("lena_image.png")
        cylinder_mesh.visualize_texture(image)

    .. figure:: /_static/meshes/create_triangle_3_axisymmetric_lena_texture.png
        :width: 400
        :align: center

        UV mapping of the Lena image on a demi-cylinder mesh with ``uv_layout=3``.

    """
    # Check the input parameters
    if frame is None:
        frame = Frame.canonical()
    if not isinstance(frame, Frame):
        raise TypeError("frames must be a Frame object")

    if not isinstance(profile_curve, Callable):
        raise TypeError("profile_curve must be a callable function")

    if not isinstance(n_theta, Integral) or n_theta < 2:
        raise ValueError("n_theta must be an integer greater than 1")
    n_theta = int(n_theta)

    if not isinstance(n_height, Integral) or n_height < 2:
        raise ValueError("n_height must be an integer greater than 1")
    n_height = int(n_height)

    if not isinstance(closed, bool):
        raise TypeError("closed must be a boolean")
    if closed:
        theta_bounds = (0.0, 2.0 * numpy.pi * (1.0 - 1.0 / n_theta))

    theta_bounds = numpy.array(theta_bounds, dtype=numpy.float64).flatten()
    if theta_bounds.shape != (2,):
        raise ValueError("theta_bounds must be a 2D array of shape (2,)")
    if theta_bounds[0] == theta_bounds[1]:
        raise ValueError("theta_bounds must be different")

    height_bounds = numpy.array(height_bounds, dtype=numpy.float64).flatten()
    if height_bounds.shape != (2,):
        raise ValueError("height_bounds must be a 2D array of shape (2,)")
    if height_bounds[0] == height_bounds[1]:
        raise ValueError("height_bounds must be different")

    if not isinstance(first_diagonal, bool):
        raise TypeError("first_diagonal must be a boolean")

    if not isinstance(direct, bool):
        raise TypeError("direct must be a boolean")

    if not isinstance(uv_layout, Integral) or uv_layout < 0 or uv_layout > 7:
        raise ValueError("uv_layout must be an integer between 0 and 7")
    uv_layout = int(uv_layout)

    # Generate the transform
    transform = FrameTransform(input_frame=frame, output_frame=Frame.canonical())

    # Extract the parameters
    theta_min = theta_bounds[0]
    theta_max = theta_bounds[1]
    height_min = height_bounds[0]
    height_max = height_bounds[1]

    # Get the indices of the vertices in the array
    index = lambda it, ih: ih * n_theta + it

    # Set the UV mapping strategy (list of 3D points -> [(0,0) ; (Nt,0) ; (0,Nh) ; (Nt,Nh)])
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

    # Create the grid of points in local coordinates
    it = numpy.arange(n_theta)  # (n_theta,)
    ih = numpy.arange(n_height)  # (n_height,)
    ft = it / (n_theta - 1)
    fh = ih / (n_height - 1)

    # Compute the local coordinates of each vertex
    height = height_min + (height_max - height_min) * fh  # (n_height, )
    theta = theta_min + (theta_max - theta_min) * ft  # (n_theta, )
    height = height[:, numpy.newaxis].repeat(n_theta, axis=1)  # (n_height, n_theta)
    theta = theta[numpy.newaxis, :].repeat(n_height, axis=0)  # (n_height, n_theta)
    radius = profile_curve(height)  # (n_height, n_theta)

    X = radius * numpy.cos(theta)  # (n_height, n_theta)
    Y = radius * numpy.sin(theta)  # (n_height, n_theta)
    Z = height  # (n_height, n_theta)

    # Convert the local points to vertices coordinates using the frame transform
    local_coordinates = numpy.stack((X, Y, Z), axis=0)  # (3, n_height, n_theta)
    local_coordinates = local_coordinates.reshape(
        (3, n_theta * n_height)
    )  # (3, n_height*n_theta)
    vertices = transform.transform(point=local_coordinates)  # (3, n_theta*n_height)
    vertices = vertices.T  # (n_theta*n_height, 3)

    # Compute the UV mapping coordinates
    if closed:
        uv_ft = (it / (n_theta))[numpy.newaxis, :].repeat(
            n_height, axis=0
        )  # (n_height, n_theta)
        uv_fh = fh[:, numpy.newaxis].repeat(n_theta, axis=1)  # (n_height, n_theta)
    else:
        uv_ft = (it / (n_theta - 1))[numpy.newaxis, :].repeat(
            n_height, axis=0
        )  # (n_height, n_theta)
        uv_fh = fh[:, numpy.newaxis].repeat(n_theta, axis=1)  # (n_height, n_theta)

    U = (
        uv_mapping[0][0]
        + uv_ft * (uv_mapping[1][0] - uv_mapping[0][0])
        + uv_fh * (uv_mapping[2][0] - uv_mapping[0][0])
    )  # (n_height, n_theta)
    V = (
        uv_mapping[0][1]
        + uv_ft * (uv_mapping[1][1] - uv_mapping[0][1])
        + uv_fh * (uv_mapping[2][1] - uv_mapping[0][1])
    )  # (n_height, n_theta)

    vertices_uvmap = numpy.stack((U, V), axis=-1).reshape(
        (n_theta * n_height, 2)
    )  # (n_theta*n_height, 2)

    # Add a fictive last vertex for UV mapping if closed case
    if closed:
        fictive_vertices_uvmap = numpy.zeros((n_height, 2))
        fictive_vertices_uvmap[ih, 0] = (
            uv_mapping[0][0]
            + 1.0 * (uv_mapping[1][0] - uv_mapping[0][0])
            + fh * (uv_mapping[2][0] - uv_mapping[0][0])
        )
        fictive_vertices_uvmap[ih, 1] = (
            uv_mapping[0][1]
            + 1.0 * (uv_mapping[1][1] - uv_mapping[0][1])
            + fh * (uv_mapping[2][1] - uv_mapping[0][1])
        )

    # Initialize the triangles and uvmap arrays
    if not closed:
        n_triangles = 2 * (n_theta - 1) * (n_height - 1)
    else:
        n_triangles = 2 * n_theta * (n_height - 1)

    triangles = numpy.zeros((n_triangles, 3), dtype=numpy.int64)
    triangles_uvmap = numpy.zeros((n_triangles, 3, 2), dtype=numpy.float32)

    # Assemble the indexes in a vectorized way
    idx = ih[:, None] * n_theta + it[None, :]  # (n_height, n_theta)

    # For ih in range(n_height - 1) and it in range(n_theta - 1): -> Build the triangles in the right order
    top_left_idx = idx[:-1, :-1]  # (n_height-1, n_theta-1)
    top_right_idx = idx[:-1, 1:]  # (n_height-1, n_theta-1)
    bottom_left_idx = idx[1:, :-1]  # (n_height-1, n_theta-1)
    bottom_right_idx = idx[1:, 1:]  # (n_height-1, n_theta-1)

    # Select the nodes index according the diagonal (n_height-1, n_theta-1)
    if first_diagonal:
        n1, n2, n3 = top_left_idx, top_right_idx, bottom_right_idx
        n4, n5, n6 = top_left_idx, bottom_right_idx, bottom_left_idx
    else:
        n1, n2, n3 = top_left_idx, top_right_idx, bottom_left_idx
        n4, n5, n6 = bottom_left_idx, top_right_idx, bottom_right_idx

    # Build the triangles
    triangle_a = numpy.stack([n1, n2, n3], axis=-1)  # (n_y-1, n_x-1, 3)
    triangle_b = numpy.stack([n4, n5, n6], axis=-1)  # (n_y-1, n_x-1, 3)
    triangle_uv_a = numpy.stack(
        [vertices_uvmap[n1, :], vertices_uvmap[n2, :], vertices_uvmap[n3, :]], axis=-2
    )  # (n_y-1, n_x-1, 3, 2)
    triangle_uv_b = numpy.stack(
        [vertices_uvmap[n4, :], vertices_uvmap[n5, :], vertices_uvmap[n6, :]], axis=-2
    )  # (n_y-1, n_x-1, 3, 2)

    # Adjust orientation if needed
    if not direct:
        triangle_a = triangle_a[..., [0, 2, 1]]
        triangle_b = triangle_b[..., [0, 2, 1]]
        triangle_uv_a = triangle_uv_a[..., [0, 2, 1], :]
        triangle_uv_b = triangle_uv_b[..., [0, 2, 1], :]

    # Organize the final triangles array
    triangle_idx_a = 2 * (
        n1 - numpy.arange(n_height - 1)[:, None]
    )  # (n_height-1, n_theta-1)
    triangle_idx_b = (
        2 * (n1 - numpy.arange(n_height - 1)[:, None]) + 1
    )  # (n_height-1, n_theta-1)

    triangles[triangle_idx_a, :] = triangle_a
    triangles[triangle_idx_b, :] = triangle_b
    triangles_uvmap[triangle_idx_a, :, :] = triangle_uv_a
    triangles_uvmap[triangle_idx_b, :, :] = triangle_uv_b

    # Add the last column of triangles connecting the last and first theta positions if closed
    if closed:
        for ih in range(n_height - 1):
            # Select the nodes of the two triangles
            if first_diagonal:
                node_1 = index(n_theta - 1, ih)
                uv_1 = vertices_uvmap[node_1, :]
                node_2 = index(0, ih)
                uv_2 = fictive_vertices_uvmap[ih, :]
                node_3 = index(0, ih + 1)
                uv_3 = fictive_vertices_uvmap[ih + 1, :]
                node_4 = index(n_theta - 1, ih)
                uv_4 = vertices_uvmap[node_4, :]
                node_5 = index(0, ih + 1)
                uv_5 = fictive_vertices_uvmap[ih + 1, :]
                node_6 = index(n_theta - 1, ih + 1)
                uv_6 = vertices_uvmap[node_6, :]
            else:
                node_1 = index(n_theta - 1, ih)
                uv_1 = vertices_uvmap[node_1, :]
                node_2 = index(0, ih)
                uv_2 = fictive_vertices_uvmap[ih, :]
                node_3 = index(n_theta - 1, ih + 1)
                uv_3 = vertices_uvmap[node_3, :]
                node_4 = index(n_theta - 1, ih + 1)
                uv_4 = vertices_uvmap[node_4, :]
                node_5 = index(0, ih)
                uv_5 = fictive_vertices_uvmap[ih, :]
                node_6 = index(0, ih + 1)
                uv_6 = fictive_vertices_uvmap[ih + 1, :]

            # Compute the triangle indices
            triangle_1 = 2 * (n_theta - 1) * (n_height - 1) + 2 * ih
            triangle_2 = 2 * (n_theta - 1) * (n_height - 1) + 2 * ih + 1

            # Set the triangles and their UV map
            if direct:
                triangles[triangle_1, :] = [node_1, node_2, node_3]
                triangles[triangle_2, :] = [node_4, node_5, node_6]
                triangles_uvmap[triangle_1, :, :] = [uv_1, uv_2, uv_3]
                triangles_uvmap[triangle_2, :, :] = [uv_4, uv_5, uv_6]
            else:
                triangles[triangle_1, :] = [node_1, node_3, node_2]
                triangles[triangle_2, :] = [node_4, node_6, node_5]
                triangles_uvmap[triangle_1, :, :] = [uv_1, uv_3, uv_2]
                triangles_uvmap[triangle_2, :, :] = [uv_4, uv_6, uv_5]

    triangles_uvmap = triangles_uvmap.reshape(
        (triangles_uvmap.shape[0], 6)
    )  # (Ntriangles, 6) - (u1,v1,u2,v2,u3,v3)

    # Prepare the mesh
    mesh = Mesh(vertices=vertices, connectivity=triangles, element_type="triangle_3")

    # Set the UV map
    mesh.uvmap = triangles_uvmap

    return mesh
