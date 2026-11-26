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

from typing import Optional

from .._validation import _validate_bool


# Explanation added to the error when the object was locked without a message
_DEFAULT_LOCKED_MESSAGE = (
    "It is used by another object of the package that assumes that it does not change. "
    "Work on a copy (obj.copy()) to get an object that can be modified."
)


def _validate_locked_message(message: Optional[str]) -> Optional[str]:
    r"""
    Validate the explanation given with a lock: a non-empty string or None.

    .. important::

        Internal use only.

    """
    if message is None:
        return None
    if not isinstance(message, str):
        raise TypeError(f"Input 'message' must be a string or None. Got type {type(message)}.")
    message = message.strip()
    if not message:
        raise ValueError("Input 'message' must be a non-empty string or None.")
    return message


class LockableMixin(object):
    r"""
    Mixin adding a lock to an object, to forbid the operations that modify it.

    A locked object can still be read, copied and saved, but the methods that modify it
    in place raise a :class:`RuntimeError`. The lock protects an object **owned by another
    object of the package** against accidental modifications, for example the
    transformation of a :class:`pysdic.ReparametrizedTransformation` or the mesh of a
    :class:`pysdic.SDICAnalysis`, whose consistency assumes that it does not change.

    The lock is set by the owner with a ``message`` explaining **why** the object is
    locked and what to do instead. The message is added to the error:

    .. code-block:: text

        RuntimeError: Cannot set the parameters: this PinholeIntrinsic is locked. It is used
        by a ReparametrizedTransformation, which keeps its parameters p equal to P(q) [...]

    .. important::

        The lock is not meant to be removed by the users: there is no public method to
        lock or unlock an object. Use ``is_locked`` and ``locked_message`` to know whether
        an object is locked and why, and work on a copy (``copy``) to get an object that
        can be modified.

    .. note::

        A new object is always unlocked, including the objects returned by ``copy`` and
        the loading methods (the lock is not saved).

    .. rubric:: Implementing a lockable class

    The owner locks an object with the internal method :meth:`_set_locked`. A class
    inheriting from the mixin calls :meth:`_check_unlocked` at the beginning of each
    method that modifies the object, **before any modification** (so that a refused
    operation leaves the object unchanged):

    .. code-block:: python

        class PointCloud(LockableMixin, ...):

            def filter_points(self, mask, *, inplace=False):
                if inplace:
                    self._check_unlocked("filter the points")
                ...

    An object made of lockable objects (for example a mesh made of a point cloud and a
    connectivity) overrides :meth:`_set_locked` to propagate the lock to its components:

    .. code-block:: python

        class Mesh(LockableMixin, ...):

            def _set_locked(self, locked=True, *, message=None):
                super()._set_locked(locked, message=message)
                self._vertices._set_locked(locked, message=message)
                self._connectivity._set_locked(locked, message=message)

    The lock state is stored in the attributes ``_locked`` and ``_locked_message``
    (declared in the ``__slots__`` of the mixin). They do not need to be initialized:
    an object whose lock was never set is unlocked.

    The class attribute ``_default_locked_message`` is the explanation used when the
    object was locked without a message. A class can redefine it, for example to
    mention an ``override_lock`` option.

    """

    __slots__ = ("_locked", "_locked_message")

    # Explanation used in the errors when the object was locked without a message.
    # A class can redefine it to explain its own way to bypass the lock.
    _default_locked_message: str = _DEFAULT_LOCKED_MESSAGE

    # ===================================================================
    # Properties
    # ===================================================================
    @property
    def is_locked(self) -> bool:
        r"""
        [Get] Whether the object is locked.

        A locked object is owned by another object of the package: the methods that
        modify it raise a :class:`RuntimeError` (see ``locked_message`` for the reason).

        Returns
        -------
        bool
            True if the object is locked, False otherwise.

        """
        return getattr(self, "_locked", False)

    @property
    def locked_message(self) -> Optional[str]:
        r"""
        [Get] The reason why the object is locked.

        Returns
        -------
        Optional[str]
            The explanation given by the owner when the object was locked, or None if the
            object is unlocked or was locked without a message.

        """
        # Own lock state (a subclass may redefine is_locked to include its components)
        if not getattr(self, "_locked", False):
            return None
        return getattr(self, "_locked_message", None)

    # ===================================================================
    # Internal methods
    # ===================================================================
    def _set_locked(self, locked: bool = True, *, message: Optional[str] = None) -> None:
        r"""
        Lock or unlock the object.

        When the object is locked, the methods that modify it raise a
        :class:`RuntimeError` containing ``message``.

        .. important::

            Internal use only: called by the objects of the package that own the object
            (and by ``copy`` to unlock a new copy).

        .. note::

            Locking an object that is already locked replaces its message. Unlocking an
            object removes its message.

        Parameters
        ----------
        locked : bool, optional (default: ``True``)
            True to lock the object, False to unlock it.

        message : Optional[str], optional (default: ``None``)
            The reason of the lock, added to the errors raised by the locked object. It
            should explain which object owns it, why, and what to do instead. Ignored when
            ``locked`` is False. If None, ``_default_locked_message`` is used in the errors.

        Raises
        ------
        TypeError
            If ``locked`` is not a boolean or ``message`` is not a string.

        ValueError
            If ``message`` is an empty string.

        """
        locked = _validate_bool(locked, "locked")
        message = _validate_locked_message(message)
        self._locked = locked
        self._locked_message = message if locked else None

    def _check_unlocked(self, action: str, *, override_lock: bool = False) -> None:
        r"""
        Raise an error if the object is locked.

        Call this method at the beginning of each method that modifies the object,
        before any modification.

        .. important::

            Internal use only (for the classes inheriting from the mixin).

        Parameters
        ----------
        action : str
            The operation refused, as a verb phrase completing "Cannot ..." (for example
            ``"filter the points"`` or ``"set the coordinates"``).

        override_lock : bool, optional (default: ``False``)
            If True, no error is raised even if the object is locked (for the methods
            offering an ``override_lock`` option, like ``set_parameters``).

        Raises
        ------
        RuntimeError
            If the object is locked and ``override_lock`` is False.

        """
        # Own lock state (a subclass may redefine is_locked to include its components)
        if override_lock or not getattr(self, "_locked", False):
            return
        message = getattr(self, "_locked_message", None) or self._default_locked_message
        raise RuntimeError(
            f"Cannot {action}: this {self.__class__.__name__} is locked. {message}"
        )