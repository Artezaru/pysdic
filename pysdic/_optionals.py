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

r"""
Lazy import of the optional dependencies of pysdic.

The optional packages are **not** imported when pysdic is loaded. A feature that
needs one of them calls :func:`import_optional` at the beginning of the method: the
package is imported on the first call, and the following calls only read the cache
of Python (``sys.modules``). If the package is missing, an explicit
:class:`ImportError` is raised, with the installation instructions.

The supported optional packages are listed in :data:`OPTIONAL_DEPENDENCIES`:

=============  ==================  ==============================================
Name           Imported module     Installation
=============  ==================  ==============================================
``py3dframe``  ``py3dframe``       ``pip install py3dframe``
``pyzernike``  ``pyzernike``       ``pip install pyzernike-polynomials``
``meshio``     ``meshio``          ``pip install meshio``
``pyvista``    ``pyvista``         ``pip install pyvista``
``vtk``        ``vtkmodules``      ``pip install vtk``
``open3d``     ``open3d``          ``pip install open3d``
``h5py``       ``h5py``            ``pip install h5py``
=============  ==================  ==============================================

All of them are installed with ``pip install pysdic[all]``, or one by one with the
extras ``frames``, ``zernike``, ``meshio``, ``vtk``, ``pyvista``, ``open3d``, ``hdf5``.

.. note::

    For ``vtk``, only the light package ``vtkmodules`` is imported (``import vtk``
    loads every VTK module, which is slow). The submodules must be imported where
    they are used, e.g. ``from vtkmodules.vtkIOLegacy import vtkDataSetReader``.

.. note::

    For type hints, import the package under ``typing.TYPE_CHECKING`` (with
    ``from __future__ import annotations``): it is then never imported at runtime.

Examples
--------
Use an optional package in a method.

.. code-block:: python
    :linenos:

    from __future__ import annotations
    from typing import TYPE_CHECKING

    # In a module of a pysdic sub-package (e.g. pysdic/geometry/point_cloud.py)
    from .._optionals import import_optional

    if TYPE_CHECKING:
        import pyvista

    class PointCloud:

        def to_pyvista(self) -> pyvista.PolyData:
            pyvista = import_optional("pyvista", "PointCloud.to_pyvista")
            return pyvista.PolyData(self._coordinates.copy())

Choose a backend depending on the installed packages.

.. code-block:: python
    :linenos:

    # In a module of a pysdic sub-package (e.g. pysdic/geometry/point_cloud.py)
    from .._optionals import has_optional

    use_package = "meshio" if has_optional("meshio") else "pyvista"

Require an optional package for a whole class (see :class:`RequiresOptionalMixin`).

.. code-block:: python
    :linenos:

    # In a module of a pysdic sub-package (e.g. pysdic/transformations/zernike_distortion.py)
    from .._optionals import import_optional

    class ZernikeDistortion(Transformation):  # Transformation inherits RequiresOptionalMixin

        _required_optionals = ("pyzernike",)  # ImportError when the class is instantiated

        def _distort(self, points):
            pyzernike = import_optional("pyzernike")  # Only reads the cache of Python
            ...

"""

import functools
import importlib
from types import ModuleType
from typing import ClassVar, Dict, NamedTuple, Optional, Tuple


__all__ = [
    "OptionalDependency",
    "OPTIONAL_DEPENDENCIES",
    "import_optional",
    "has_optional",
    "RequiresOptionalMixin",
]


class OptionalDependency(NamedTuple):
    r"""
    Description of an optional dependency of pysdic.

    Attributes
    ----------
    module: str
        The name of the module given to :func:`importlib.import_module`
        (e.g. ``"vtkmodules"`` for ``vtk``).

    pip_name: str
        The name of the package given to ``pip install`` (e.g. ``"opencv-python"``
        for ``cv2``).

    url: str
        The URL of the package (repository or website).

    """

    module: str
    pip_name: str
    url: str


OPTIONAL_DEPENDENCIES: Dict[str, OptionalDependency] = {
    "py3dframe": OptionalDependency("py3dframe", "py3dframe", "https://github.com/Artezaru/py3dframe"),
    "pyzernike": OptionalDependency("pyzernike", "pyzernike-polynomials", "https://github.com/Artezaru/pyzernike"),
    "meshio": OptionalDependency("meshio", "meshio", "https://github.com/nschloe/meshio"),
    "pyvista": OptionalDependency("pyvista", "pyvista", "https://github.com/pyvista/pyvista"),
    "vtk": OptionalDependency("vtkmodules", "vtk", "https://vtk.org"),
    "open3d": OptionalDependency("open3d", "open3d", "https://github.com/isl-org/Open3D"),
    "h5py": OptionalDependency("h5py", "h5py", "https://github.com/h5py/h5py"),
}


def _get_dependency(name: str) -> OptionalDependency:
    # Look up an optional dependency, with an explicit error for unknown names
    if not isinstance(name, str):
        raise TypeError(f"Input 'name' must be a string. Got type {type(name)}.")
    try:
        return OPTIONAL_DEPENDENCIES[name]
    except KeyError:
        raise ValueError(
            f"Unknown optional dependency '{name}'. Available: "
            f"{', '.join(sorted(OPTIONAL_DEPENDENCIES))}."
        ) from None


@functools.lru_cache(maxsize=None)
def _try_import(module: str) -> Tuple[Optional[ModuleType], Optional[BaseException]]:
    # Import a module once and cache the result, including the failures (Python
    # does not cache failed imports: without this cache, each call to
    # has_optional would search the disk again).
    try:
        return importlib.import_module(module), None
    except ImportError as error:
        return None, error


def import_optional(name: str, method: Optional[str] = None) -> ModuleType:
    r"""
    Import an optional dependency of pysdic.

    The package is imported on the first call and the result is cached: the
    following calls are almost free.

    Parameters
    ----------
    name: str
        The name of the optional dependency, a key of
        :data:`OPTIONAL_DEPENDENCIES` (e.g. ``"pyvista"``, ``"vtk"``, ``"cv2"``).

    method: Optional[str], optional (default: ``None``)
        The name of the feature that needs the package, shown in the error message
        (e.g. ``"PointCloud.to_pyvista"``). By default, a generic message is used.

    Returns
    -------
    types.ModuleType
        The imported module (``vtkmodules`` for ``"vtk"``).

    Raises
    ------
    TypeError
        If ``name`` is not a string.

    ValueError
        If ``name`` is not a known optional dependency.

    ImportError
        If the package is not installed, or if it is installed but cannot be
        imported. The original error is attached as ``__cause__``.

    Examples
    --------
    .. code-block:: python
        :linenos:

        # In a module of a pysdic sub-package (e.g. pysdic/geometry/point_cloud.py)
        from .._optionals import import_optional

        meshio = import_optional("meshio", "PointCloud.from_meshio")
        mesh = meshio.read("mesh.vtk")

    """
    dependency = _get_dependency(name)
    if method is not None and not isinstance(method, str):
        raise TypeError(f"Input 'method' must be a string or None. Got type {type(method)}.")

    module, error = _try_import(dependency.module)
    if module is None:
        feature = f"'{method}'" if method is not None else "this feature"
        raise ImportError(
            f"Missing '{name}' package to use {feature}. Install it with "
            f"'pip install {dependency.pip_name}', or from '{dependency.url}'."
        ) from error  # keeps the real cause (missing package, broken install, ...)
    return module


def has_optional(name: str) -> bool:
    r"""
    Check if an optional dependency of pysdic can be imported.

    The package is imported on the first call (if it is available) and the result
    is cached.

    Parameters
    ----------
    name: str
        The name of the optional dependency, a key of
        :data:`OPTIONAL_DEPENDENCIES` (e.g. ``"pyvista"``, ``"vtk"``, ``"cv2"``).

    Returns
    -------
    bool
        True if the package can be imported, False otherwise.

    Raises
    ------
    TypeError
        If ``name`` is not a string.

    ValueError
        If ``name`` is not a known optional dependency.

    Examples
    --------
    .. code-block:: python
        :linenos:

        # In a module of a pysdic sub-package (e.g. pysdic/geometry/point_cloud.py)
        from .._optionals import has_optional

        if has_optional("open3d"):
            print("Open3D is available.")

    """
    dependency = _get_dependency(name)
    module, _ = _try_import(dependency.module)
    return module is not None


class RequiresOptionalMixin:
    r"""
    Mixin to require optional dependencies of pysdic for a whole class.

    A class declares the optional packages it cannot work without in the class attribute
    ``_required_optionals`` (a tuple of keys of :data:`OPTIONAL_DEPENDENCIES`). The packages
    are **not** imported when the class is defined: they are imported when the class is
    instantiated, and an :class:`ImportError` with the installation instructions is raised if
    one of them is missing.

    The check is done in ``__new__``, so that:

    - it is done before any code of the ``__init__`` methods (validation of the constants, ...),
    - it cannot be bypassed by a subclass overriding ``__init__``,
    - it also covers the objects created without ``__init__`` (``copy.deepcopy``, ``pickle``).

    The requirements are inherited by the subclasses. A subclass needing more packages extends the
    tuple of its parent, e.g. ``_required_optionals = Parent._required_optionals + ("h5py",)``.
    The names are checked when the class is defined, so a typo raises an error as soon as
    pysdic is imported.

    .. important::

        Internal use only. Put the mixin **first** in the base classes, and do not define a
        ``__new__`` requiring arguments in the classes after it in the MRO (the mixin calls
        ``super().__new__(cls)`` without arguments).

    .. note::

        Only require a package for a whole class if the class cannot work without it. For an
        optional feature of a class (one method or property), call :func:`import_optional`
        in that method instead.

        The methods using the package still call :func:`import_optional` to get the module:
        it only reads the cache of Python. The module is not stored on the instance (to keep
        ``__slots__``, the serialization and ``pickle`` working).

    Examples
    --------
    .. code-block:: python
        :linenos:

        class ZernikeDistortion(Transformation):  # Transformation inherits RequiresOptionalMixin
            _required_optionals = ("pyzernike",)

        ZernikeDistortion.is_available()   # False if pyzernike is not installed
        ZernikeDistortion(n_zernike=2)     # ImportError: Missing 'pyzernike' package to use ...

    """

    __slots__ = ()

    _required_optionals: ClassVar[Tuple[str, ...]] = ()

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        # Check the declaration when the class is defined (the packages are not imported)
        required = cls._required_optionals
        if not isinstance(required, tuple):
            raise TypeError(
                f"Attribute '_required_optionals' of class '{cls.__name__}' must be a tuple of "
                f"strings. Got type {type(required)}."
            )
        for name in required:
            try:
                _get_dependency(name)
            except (TypeError, ValueError) as error:
                raise type(error)(
                    f"Invalid '_required_optionals' of class '{cls.__name__}': {error}"
                ) from None

    def __new__(cls, *args, **kwargs):
        # Import the required packages before creating the instance (cached after the first call)
        for name in cls._required_optionals:
            import_optional(name, cls.__name__)
        return super().__new__(cls)  # object.__new__ does not accept the arguments of __init__

    @classmethod
    def is_available(cls) -> bool:
        r"""
        Check if all the optional dependencies required by the class can be imported, i.e. if
        the class can be instantiated.

        Returns
        -------
        bool
            True if all the required packages can be imported (always True for a class without
            required optional dependencies), False otherwise.

        """
        return all(has_optional(name) for name in cls._required_optionals)