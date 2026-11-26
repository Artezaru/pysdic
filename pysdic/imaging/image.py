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

import os
from collections.abc import Mapping
from typing import Any, Dict, Optional, Tuple, Union
from numbers import Integral, Real
from numpy.typing import ArrayLike

import numpy
import cv2
from .._validation import (
    _validate_bool,
    _validate_integer,
    _validate_real,
    _validate_path,
    _as_array,
    _as_batch_array,
    _as_bool_mask,
    _as_integer_array,
)
from .._optionals import import_optional
from ..core.serialization import NpzSerializerMixin
from .rect_bivariate_spline import _MAX_SPLINE_DEGREE, RectBivariateSpline

# Extensions OpenCV can write without converting the data, for each non-uint8 dtype.
# uint8 is accepted by every format, so OpenCV decides (and fails if it cannot).
_WRITABLE_EXTENSIONS = {
    numpy.dtype(numpy.uint16): (".png", ".tif", ".tiff", ".jp2", ".pgm", ".ppm", ".pnm"),
    numpy.dtype(numpy.float32): (".tif", ".tiff", ".exr", ".pfm", ".hdr"),
    numpy.dtype(numpy.float64): (".tif", ".tiff"),
}


def _as_flat_pixel_mask(mask: ArrayLike, height: int, width: int) -> numpy.ndarray:
    r"""
    Convert a boolean ``mask`` of shape :math:`(H, W)` or :math:`(H \times W,)` to a
    boolean array of shape :math:`(H \times W,)` (row-major order).

    .. important::

        Internal use only (shared by :class:`pysdic.Image` and :class:`pysdic.Camera`).

    """
    mask = numpy.asarray(mask)
    if mask.ndim == 2:
        if mask.shape != (height, width):
            raise ValueError(
                f"Input 'mask' must have shape ({height}, {width}) or "
                f"({height * width},). Got shape {mask.shape}."
            )
        mask = mask.ravel()
    return _as_bool_mask(mask, height * width, "mask")


def _pixel_points_grid(
    height: int, width: int, mask: Optional[ArrayLike] = None
) -> numpy.ndarray:
    r"""
    Return the pixel points :math:`(u, v)` of a :math:`(H, W)` grid in row-major order,
    filtered by an optional boolean ``mask`` of shape :math:`(H, W)` or
    :math:`(H \times W,)`. Shape :math:`(N_p, 2)` in float64 dtype.

    .. important::

        Internal use only (shared by :class:`pysdic.Image` and :class:`pysdic.Camera`).

    """
    pixel_points = numpy.indices((height, width), dtype=numpy.float64).reshape(2, -1).T
    if mask is not None:
        pixel_points = pixel_points[_as_flat_pixel_mask(mask, height, width)]
    return pixel_points


# Sentinel of the arguments not given to Image.with_settings (None is a valid ROI)
_UNSET = object()


class Image(NpzSerializerMixin):
    r"""
    A class to represent an **immutable** 2D image to interpolate gray or colors values.

    The image is associated with an interpolation function to evaluate pixel values
    (and their derivatives) at arbitrary points in the image: one
    :class:`pysdic.imaging.rect_bivariate_spline.RectBivariateSpline` built on the ROI with
    all the channels (one scipy spline per channel). All the channels and several
    derivative orders are evaluated in a single pass (see :meth:`evaluate_at_pixel_points`).

    The coordinates system used for the image is defined as follows:

    - The ``pixel_coordinates`` are defined in the array coordinate system
      :math:`(u, v)` where :math:`u` is the row index and :math:`v` is the column index.
    - The ``image_coordinates`` are defined in the camera coordinate system
      :math:`(x, y)` where :math:`x` is the column index and :math:`y` is the row index.

    .. important::

        To avoid errors, kx and ky used by scipy are renamed ``ku`` and ``kv`` as the
        :math:`x`-axis of scipy refer to the :math:`u`-axis (ie the :math:`y`-axis of
        ``image_coordinates``).

    .. important::

        An image is **immutable**: the data, the ``roi``, ``ku``, ``kv`` and
        ``default_value`` are fixed at the creation and cannot be modified. Use
        :meth:`with_settings` to get an image with other settings (it shares the data,
        without copy) and :meth:`from_array` to create an image with other data.

        The immutability guarantees that the objects using an image (views, SDIC
        analyses, ...) always see the same values, and allows several images to share
        the same data without copy (for example the two halves of a split-screen
        image, with two ROIs).

    .. note::

        The image is stored as given by the user (integer or floating point dtype), in
        a read-only copy. It is only converted to :obj:`numpy.float64` to build the
        interpolation function.

    .. note::

        An image can be saved with all its settings in a single NPZ file with
        :meth:`to_npz_file` and loaded with :meth:`from_npz_file`: the data are stored
        exactly (any dtype, any number of channels), without conversion.

    .. note::

        The interpolation function is a cache: it is built at the creation (unless
        ``auto_construct=False``) or with :meth:`construct_interpolation_functions`, and
        can be freed with :meth:`deconstruct_interpolation_functions` to save memory.
        This does not change the values of the image.


    Parameters
    ----------
    image : ArrayLike
        The image with shape (height, width) or (height, width, channels), with an
        integer or floating point dtype (finite values only). The array is copied.

    roi : Optional[Tuple[Integral, Integral, Integral, Integral]], optional (default: ``None``)
        The region of interest (ROI) in the image, defined as a tuple
        (x, y, width, height) in pixels: the ROI contains the columns
        ``x, ..., x + width - 1`` and the rows ``y, ..., y + height - 1``. It must be
        inside the image, with ``height > ku`` and ``width > kv``. If None, the entire
        image is used. Restricting the ROI reduces the cost of building and of
        evaluating the interpolation function.

    ku : Integral, optional (default: ``3``)
        The degree of the spline in the :math:`u`-direction, between 1 and 5.
        Warning ``ku`` refers to :math:`u`-axis of the ``pixel_coordinates``
        (ie :math:`y`-axis of the ``image_coordinates`` - the ``kx`` of scipy).

    kv : Integral, optional (default: ``3``)
        The degree of the spline in the :math:`v`-direction, between 1 and 5.
        Warning ``kv`` refers to :math:`v`-axis of the ``pixel_coordinates``
        (ie :math:`x`-axis of the ``image_coordinates`` - the ``ky`` of scipy).

    default_value : Real, optional (default: ``numpy.nan``)
        The value returned by the interpolation function and its derivatives for
        evaluations out of the ROI (NaN and infinity are accepted).

    auto_construct : bool, optional (default: ``True``)
        Whether to construct the interpolation function at initialization.


    Raises
    ------
    TypeError
        If the dtype of the image is not an integer or floating point dtype, or if the
        ROI is not a tuple.

    ValueError
        If the shape of the image is invalid, if a floating point image contains
        non-finite values, if a spline degree is out of :math:`[1, 5]`, or if the ROI is
        out of the image or too small for the spline degrees.

    """

    __slots__ = [
        "_image",
        "_roi",
        "_ku",
        "_kv",
        "_default_value",
        "_interpolation_function",
    ]

    def __init__(
        self,
        image: ArrayLike,
        *,
        roi: Optional[Tuple[Integral, Integral, Integral, Integral]] = None,
        ku: Integral = 3,
        kv: Integral = 3,
        default_value: Real = numpy.nan,
        auto_construct: bool = True,
    ) -> None:
        image = self._validate_image(image)
        ku = self._validate_degree(ku, "ku")
        kv = self._validate_degree(kv, "kv")
        roi = self._validate_roi(roi, image.shape[0], image.shape[1], ku, kv)
        default_value = _validate_real(default_value, "default_value")
        auto_construct = _validate_bool(auto_construct, "auto_construct")

        self._image = image
        self._roi = roi
        self._ku = ku
        self._kv = kv
        self._default_value = default_value
        self._interpolation_function = None
        if auto_construct:
            self.construct_interpolation_functions()

    @classmethod
    def _from_shared(
        cls,
        image: numpy.ndarray,
        roi: Optional[Tuple[int, int, int, int]],
        ku: int,
        kv: int,
        default_value: float,
        function: Optional[RectBivariateSpline],
    ) -> Image:
        # New image sharing the (read-only, validated) data and the interpolation function
        new = cls.__new__(cls)
        new._image = image
        new._roi = roi
        new._ku = ku
        new._kv = kv
        new._default_value = default_value
        new._interpolation_function = function
        return new

    # ===================================================================
    # Internal validation methods
    # ===================================================================
    @staticmethod
    def _validate_image(image: ArrayLike) -> numpy.ndarray:
        # Read-only copy of the image data (integer or finite floating point values)
        if image is None:
            raise TypeError(
                "Input 'image' must be an array with shape (height, width) or "
                "(height, width, channels). Got None."
            )
        image = numpy.array(image)  # always a copy
        if not (
            numpy.issubdtype(image.dtype, numpy.integer)
            or numpy.issubdtype(image.dtype, numpy.floating)
        ):
            raise TypeError(
                "Image data type must be an integer or floating point dtype "
                f"(e.g. numpy.uint8, numpy.uint16, numpy.float32). Got {image.dtype}."
            )
        if image.ndim not in (2, 3) or 0 in image.shape:
            raise ValueError(
                "Image must have shape (height, width) with a grayscale image or "
                "(height, width, channels) with a color image, with non-zero sizes. "
                f"Got shape {image.shape}."
            )
        if numpy.issubdtype(image.dtype, numpy.floating) and not numpy.all(
            numpy.isfinite(image)
        ):
            raise ValueError("Image must contain only finite values.")
        image.flags.writeable = False
        return image

    @staticmethod
    def _validate_degree(value: Integral, name: str) -> int:
        value = _validate_integer(value, name, min_value=1)
        if value > _MAX_SPLINE_DEGREE:
            raise ValueError(
                f"Input '{name}' must be lower or equal to {_MAX_SPLINE_DEGREE}. Got {value}."
            )
        return value

    @staticmethod
    def _validate_roi(
        roi: Optional[Tuple[Integral, Integral, Integral, Integral]],
        image_height: int,
        image_width: int,
        ku: int,
        kv: int,
    ) -> Optional[Tuple[int, int, int, int]]:
        # ROI (x, y, width, height) inside the image and large enough for the spline degrees
        if roi is not None:
            try:
                roi = tuple(roi)
            except TypeError:
                raise TypeError(
                    f"ROI must be a tuple (x, y, width, height). Got type {type(roi)}."
                ) from None
            if len(roi) != 4:
                raise ValueError(
                    f"ROI must be a tuple (x, y, width, height). Got {len(roi)} values."
                )
            roi = (
                _validate_integer(roi[0], "roi x", min_value=0),
                _validate_integer(roi[1], "roi y", min_value=0),
                _validate_integer(roi[2], "roi width", min_value=1),
                _validate_integer(roi[3], "roi height", min_value=1),
            )
            x, y, width, height = roi
            if x + width > image_width or y + height > image_height:
                raise ValueError(
                    f"ROI {roi} must be within the bounds of the image "
                    f"(width={image_width}, height={image_height})."
                )
        else:
            width, height = image_width, image_height
        if height <= ku or width <= kv:
            raise ValueError(
                f"ROI (height={height}, width={width}) is too small for the spline "
                f"degrees (ku={ku}, kv={kv}): height > ku and width > kv are required."
            )
        return roi

    def _check_interpolation_functions_constructed(self) -> None:
        if self._interpolation_function is None:
            raise ValueError(
                "The interpolation function is not constructed. Please call "
                "`construct_interpolation_functions` (it is not built when the image "
                "is created with auto_construct=False or after "
                "`deconstruct_interpolation_functions`)."
            )

    def _validate_derivatives(self, derivatives: ArrayLike) -> numpy.ndarray:
        # Orders (du, dv) of shape (R, 2) with 0 <= du < ku and 0 <= dv < kv
        orders = _as_integer_array(derivatives, "derivatives", shape=(None, 2), shape_hint="(R, 2)")
        if orders.shape[0] == 0:
            raise ValueError("Input 'derivatives' must contain at least one order.")
        if numpy.any(orders < 0) or numpy.any(orders[:, 0] >= self._ku) or numpy.any(orders[:, 1] >= self._kv):
            raise ValueError(
                f"The derivative orders (du, dv) must satisfy 0 <= du < ku={self._ku} and "
                f"0 <= dv < kv={self._kv}. Got {orders.tolist()}."
            )
        return orders

    def _get_effective_roi(self) -> Tuple[int, int, int, int]:
        return self._roi if self._roi is not None else (0, 0, self.width, self.height)

    # ===================================================================
    # Properties
    # ===================================================================
    @property
    def image(self) -> numpy.ndarray:
        r"""
        [Get] The image data, with shape (height, width) or (height, width, channels)
        and its original dtype.

        .. note::

            The returned array is read-only (the data of an image cannot be modified).
            Use :meth:`to_array` to get a modifiable copy, and :meth:`from_array` to
            create an image with other data.

        Returns
        -------
        numpy.ndarray
            A read-only view of the image data.

        """
        view = self._image.view()
        view.flags.writeable = False
        return view

    @property
    def roi(self) -> Optional[Tuple[int, int, int, int]]:
        r"""
        [Get] The region of interest (ROI) in the image.

        The ROI is defined as a tuple (x, y, width, height) in pixels: it contains
        the columns ``x, ..., x + width - 1`` and the rows ``y, ..., y + height - 1``.
        Points between these pixels (bounds included) can be evaluated. If None, the
        entire image is used.

        .. seealso::

            :meth:`with_settings` to get an image with another ROI (sharing the data).

        Returns
        -------
        Optional[Tuple[int, int, int, int]]
            The region of interest (ROI) as a tuple (x, y, width, height), or None for
            the entire image.

        """
        return self._roi

    @property
    def shape(self) -> Tuple[int, ...]:
        r"""
        [Get] The shape of the image: (height, width) or (height, width, channels).

        Returns
        -------
        Tuple[int, ...]
            The shape of the image.

        """
        return self._image.shape

    @property
    def height(self) -> int:
        r"""
        [Get] The height of the image.

        Returns
        -------
        int
            The height of the image.

        """
        return self._image.shape[0]

    @property
    def width(self) -> int:
        r"""
        [Get] The width of the image.

        Returns
        -------
        int
            The width of the image.

        """
        return self._image.shape[1]

    @property
    def is_color(self) -> bool:
        r"""
        [Get] Check if the image is a color image (3D array).

        .. note::

            (height, width, 1) images are still considered color images.

        Returns
        -------
        bool
            :obj:`True` if the image is a color image, :obj:`False` if it is grayscale.

        """
        return self._image.ndim == 3

    @property
    def is_grayscale(self) -> bool:
        r"""
        [Get] Check if the image is grayscale (2D array).

        .. note::

            (height, width, 1) images are still considered color images.

        Returns
        -------
        bool
            :obj:`True` if the image is grayscale, :obj:`False` if it is color.

        """
        return self._image.ndim == 2

    @property
    def n_channels(self) -> int:
        r"""
        [Get] The number of channels in the image (1 for a grayscale image).

        Returns
        -------
        int
            The number of channels in the image.

        """
        return 1 if self._image.ndim == 2 else self._image.shape[2]

    @property
    def dtype(self) -> numpy.dtype:
        r"""
        [Get] The data type of the image.

        Returns
        -------
        :class:`numpy.dtype`
            The data type of the image.

        """
        return self._image.dtype

    @property
    def ndim(self) -> int:
        r"""
        [Get] The number of dimensions of the image.

        Returns
        -------
        int
            The number of dimensions of the image.

        """
        return self._image.ndim

    @property
    def ku(self) -> int:
        r"""
        [Get] The degree of the spline in the :math:`u`-direction.

        Warning ``ku`` refers to scipy :math:`u`-axis of the ``pixel_coordinates``
        (ie :math:`y`-axis of the ``image_coordinates``).

        Returns
        -------
        int
            The degree of the spline in the :math:`u`-direction.

        """
        return self._ku

    @property
    def kv(self) -> int:
        r"""
        [Get] The degree of the spline in the :math:`v`-direction.

        Warning ``kv`` refers to scipy :math:`v`-axis of the ``pixel_coordinates``
        (ie :math:`x`-axis of the ``image_coordinates``).

        Returns
        -------
        int
            The degree of the spline in the :math:`v`-direction.

        """
        return self._kv

    @property
    def default_value(self) -> float:
        r"""
        [Get] The value returned by the interpolation function and its derivatives
        for evaluations out of the ``roi``.

        Returns
        -------
        float
            The default value of the evaluations.

        """
        return self._default_value

    @property
    def is_constructed(self) -> bool:
        r"""
        [Get] Whether the interpolation function is constructed.

        Returns
        -------
        bool
            True if the image can be evaluated, False if
            :meth:`construct_interpolation_functions` must be called first.

        """
        return self._interpolation_function is not None

    def __repr__(self) -> str:
        return (
            f"Image(shape={self.shape}, dtype={self.dtype}, is_color={self.is_color}, "
            f"roi={self.roi})"
        )

    # ===================================================================
    # Class methods
    # ===================================================================
    @classmethod
    def from_array(cls, image_data: ArrayLike, **kwargs) -> Image:
        r"""
        Create an :class:`pysdic.Image` object from raw image data.

        Parameters
        ----------
        image_data: ArrayLike
            The raw image data with shape (height, width) for grayscale images or
            (height, width, channels) for color images, with an integer or floating
            point dtype. The dtype is kept.

        **kwargs
            Keyword arguments of :class:`pysdic.Image` (``roi``, ``ku``, ``kv``,
            ``default_value``, ``auto_construct``).


        Returns
        -------
        :class:`pysdic.Image`
            An :class:`pysdic.Image` object initialized with the provided image data.


        Examples
        --------
        Create an :class:`pysdic.Image` object from a numpy array of image data.

        .. code-block:: python
            :linenos:

            import numpy
            from pysdic import Image

            image_data = numpy.random.randint(0, 256, size=(480, 640), dtype=numpy.uint8)
            image = Image.from_array(image_data, roi=(100, 50, 200, 100))

        """
        return cls(image=image_data, **kwargs)

    @classmethod
    def from_file(cls, file_path: Union[str, os.PathLike], **kwargs) -> Image:
        r"""
        Create an :class:`pysdic.Image` object by loading image data from a file.

        The file is read with OpenCV (``cv2.IMREAD_UNCHANGED``): the dtype of the file
        is kept (e.g. :obj:`numpy.uint16` for a 16-bit PNG, :obj:`numpy.float32` for
        a float TIFF).

        .. warning::

            OpenCV reads color images in BGR order (BGRA with an alpha channel).

        .. note::

            Requires the optional dependency ``cv2`` (``pip install opencv-python``).

        Parameters
        ----------
        file_path : Union[str, os.PathLike]
            The path to the image file to be loaded.

        **kwargs
            Keyword arguments of :class:`pysdic.Image` (``roi``, ``ku``, ``kv``,
            ``default_value``, ``auto_construct``).


        Returns
        -------
        :class:`pysdic.Image`
            An :class:`pysdic.Image` object initialized with the image data
            loaded from the specified file.


        Raises
        ------
        FileNotFoundError
            If the file does not exist.

        ValueError
            If OpenCV cannot read the file as an image.


        Examples
        --------
        Create an :class:`pysdic.Image` object by loading image data from a file.

        .. code-block:: python
            :linenos:

            from pysdic import Image

            image = Image.from_file("path_to_image.png")

        """
        path = _validate_path(file_path, "file_path", must_exist=True)
        image_data = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image_data is None:
            raise ValueError(f"OpenCV could not read '{path}' as an image.")
        return cls(image=image_data, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        r"""
        Convert the image to a flat dictionary, with its data and all its settings.

        The keys are ``"image"`` (the data, with its dtype), ``"roi"`` (a list
        ``[x, y, width, height]`` or None), ``"ku"``, ``"kv"`` and ``"default_value"``. The
        interpolation function is not stored (it is a cache).

        .. note::

            The dictionary is flat (one level), as required by the NPZ format. See
            :class:`pysdic.core.serialization.NpzSerializerMixin`.

        Returns
        -------
        Dict[str, Any]
            The dictionary representation of the image (the data is a read-only view).

        """
        return {
            "image": self.image,
            "roi": None if self._roi is None else list(self._roi),
            "ku": self._ku,
            "kv": self._kv,
            "default_value": self._default_value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], *, auto_construct: bool = True) -> Image:
        r"""
        Create an image from a dictionary written by :meth:`to_dict`.

        The key ``"image"`` is required. The keys ``"roi"``, ``"ku"``, ``"kv"`` and
        ``"default_value"`` are optional (the default values of :class:`pysdic.Image` are
        used if they are missing).

        Parameters
        ----------
        data : Dict[str, Any]
            The dictionary to read.

        auto_construct : bool, optional (default: ``True``)
            Whether to construct the interpolation function of the image.

        Returns
        -------
        :class:`pysdic.Image`
            The new image.

        Raises
        ------
        TypeError
            If ``data`` is not a dictionary.

        ValueError
            If the key ``"image"`` is missing or a value is invalid (see
            :class:`pysdic.Image`).

        """
        if not isinstance(data, Mapping):
            raise TypeError(f"Input 'data' must be a dictionary. Got type {type(data)}.")
        if "image" not in data:
            raise ValueError(
                "Input dictionary must contain a key 'image' with shape (height, width) or "
                "(height, width, channels)."
            )
        roi = data.get("roi", None)
        if roi is not None:
            roi = tuple(numpy.asarray(roi).tolist())
        return cls(
            data["image"],
            roi=roi,
            ku=data.get("ku", 3),
            kv=data.get("kv", 3),
            default_value=data.get("default_value", numpy.nan),
            auto_construct=auto_construct,
        )

    def to_file(self, file_path: Union[str, os.PathLike]) -> None:
        r"""
        Save the image to a file using OpenCV, with its own dtype.

        The data are written without any conversion, so the format given by the file
        extension must support the dtype of the image:

        - :obj:`numpy.uint8`: every format supported by OpenCV.
        - :obj:`numpy.uint16`: ``.png``, ``.tif``, ``.tiff``, ``.jp2``, ``.pgm``,
          ``.ppm``, ``.pnm``.
        - :obj:`numpy.float32`: ``.tif``, ``.tiff``, ``.exr``, ``.pfm``, ``.hdr``.
        - :obj:`numpy.float64`: ``.tif``, ``.tiff``.

        Convert the image yourself (with :meth:`to_array`) to save it with another dtype.

        .. warning::

            OpenCV writes color images in BGR order (BGRA with an alpha channel).
            Use the same channel order as :meth:`from_file` to get the same colors.

        .. note::

            Requires the optional dependency ``cv2`` (``pip install opencv-python``).

        Parameters
        ----------
        file_path : Union[str, os.PathLike]
            The path where the image will be saved. The extension selects the format.


        Raises
        ------
        ValueError
            If the extension does not support the dtype of the image or if OpenCV fails
            to write the file.

        TypeError
            If the dtype of the image cannot be written by OpenCV.

        """
        path = _validate_path(file_path, "file_path")
        dtype = self._image.dtype
        if dtype != numpy.uint8:
            if dtype not in _WRITABLE_EXTENSIONS:
                raise TypeError(
                    f"Images with dtype {dtype} cannot be written by OpenCV. Supported "
                    "dtypes are uint8, uint16, float32 and float64."
                )
            if path.suffix.lower() not in _WRITABLE_EXTENSIONS[dtype]:
                raise ValueError(
                    f"Images with dtype {dtype} can only be written with the extensions "
                    f"{_WRITABLE_EXTENSIONS[dtype]}. Got '{path.suffix}'."
                )
        if not cv2.imwrite(str(path), self._image):
            raise ValueError(
                f"OpenCV failed to write the image to '{path}'. Check the extension, "
                "the number of channels and that the folder exists."
            )

    def to_array(self) -> numpy.ndarray:
        r"""
        Get a modifiable copy of the image data as a numpy array (with its original
        dtype).

        Returns
        -------
        numpy.ndarray
            A copy of the image data.

        """
        return self._image.copy()

    def copy(self) -> Image:
        r"""
        Create a copy of the :class:`pysdic.Image` object.

        Since an image is immutable, the copy **shares** the data and the interpolation
        function with the original (no data is copied): a constructed image gives a
        constructed copy. Constructing or deconstructing the interpolation function of
        one of them does not affect the other.

        Returns
        -------
        :class:`pysdic.Image`
            A new :class:`pysdic.Image` object with the same data and settings.

        """
        return self._from_shared(
            self._image, self._roi, self._ku, self._kv, self._default_value,
            self._interpolation_function,
        )

    def with_settings(
        self,
        *,
        roi: Optional[Tuple[Integral, Integral, Integral, Integral]] = _UNSET,
        ku: Integral = _UNSET,
        kv: Integral = _UNSET,
        default_value: Real = _UNSET,
        auto_construct: bool = True,
    ) -> Image:
        r"""
        Create a new image with the same data and other settings.

        The new image **shares** the data with this image (no copy): creating several
        images with different ROIs from the same data is cheap, for example for a
        split-screen setup. The settings that are not given keep their current value.

        If ``roi``, ``ku`` and ``kv`` are unchanged (for example when only
        ``default_value`` changes), the new image also shares the interpolation
        function of this image (if it is constructed). Otherwise, it is built if
        ``auto_construct`` is True.

        Parameters
        ----------
        roi : Optional[Tuple[Integral, Integral, Integral, Integral]], optional
            The region of interest (x, y, width, height) of the new image, or None for
            the entire image. By default, the ROI of this image.

        ku : Integral, optional
            The degree of the spline in the :math:`u`-direction, between 1 and 5. By
            default, the ``ku`` of this image.

        kv : Integral, optional
            The degree of the spline in the :math:`v`-direction, between 1 and 5. By
            default, the ``kv`` of this image.

        default_value : Real, optional
            The value returned for evaluations out of the ROI. By default, the
            ``default_value`` of this image.

        auto_construct : bool, optional (default: ``True``)
            Whether to construct the interpolation function of the new image, if it
            cannot be shared with this image.

        Returns
        -------
        :class:`pysdic.Image`
            The new image, sharing the data of this image.

        Raises
        ------
        ValueError
            If a spline degree is out of :math:`[1, 5]`, or if the ROI is out of the
            image or too small for the spline degrees.

        Examples
        --------
        The two halves of a split-screen image, sharing the same data:

        .. code-block:: python
            :linenos:

            import numpy
            from pysdic import Image

            image = Image.from_array(numpy.random.rand(3000, 4000), auto_construct=False)
            left = image.with_settings(roi=(0, 0, 2000, 3000))
            right = image.with_settings(roi=(2000, 0, 2000, 3000))

        """
        ku = self._ku if ku is _UNSET else self._validate_degree(ku, "ku")
        kv = self._kv if kv is _UNSET else self._validate_degree(kv, "kv")
        if roi is _UNSET:
            roi = self._roi
        roi = self._validate_roi(roi, self.height, self.width, ku, kv)
        default_value = (
            self._default_value
            if default_value is _UNSET
            else _validate_real(default_value, "default_value")
        )
        auto_construct = _validate_bool(auto_construct, "auto_construct")

        same_functions = roi == self._roi and ku == self._ku and kv == self._kv
        new = self._from_shared(
            self._image, roi, ku, kv, default_value,
            self._interpolation_function if same_functions else None,
        )
        if auto_construct and new._interpolation_function is None:
            new.construct_interpolation_functions()
        return new

    # ==================================================================
    # Methods
    # ==================================================================
    def construct_interpolation_functions(self) -> None:
        r"""
        Construct the interpolation function of the image.

        One :class:`pysdic.imaging.rect_bivariate_spline.RectBivariateSpline`
        (``kx=ku``, ``ky=kv``) is built on the pixels of the ROI with all the channels of
        the image (one scipy spline per channel, sharing the same knots). The function is
        a cache: building it again gives the same function.

        .. note::

            The function is built at the creation of the image, unless
            ``auto_construct=False``. Call this method only after
            :meth:`deconstruct_interpolation_functions` or with an image created with
            ``auto_construct=False``.

        """
        x, y, width, height = self._get_effective_roi()
        u_vector = numpy.arange(y, y + height, dtype=numpy.float64)
        v_vector = numpy.arange(x, x + width, dtype=numpy.float64)
        image_roi = self._image[y : y + height, x : x + width]  # (H, W) or (H, W, C)
        self._interpolation_function = RectBivariateSpline(
            u_vector,
            v_vector,
            image_roi.astype(numpy.float64),
            kx=self._ku,  # scipy x = u-axis = y-axis
            ky=self._kv,  # scipy y = v-axis = x-axis
        )

    def deconstruct_interpolation_functions(self) -> None:
        r"""
        Free the interpolation function of the image, for example to save memory when
        the image is not evaluated anymore.

        The values of the image do not change: call
        :meth:`construct_interpolation_functions` to build the function again before
        evaluating the image. The other images sharing this function (see :meth:`copy`
        and :meth:`with_settings`) keep it.

        """
        self._interpolation_function = None

    def evaluate_at_pixel_points(
        self, pixel_points: ArrayLike, derivatives: ArrayLike = ((0, 0),)
    ) -> Tuple[numpy.ndarray, ...]:
        r"""
        Evaluate the image and/or several of its derivatives at given ``pixel_points``
        (ie :math:`(u, v)`-axis), in a single pass.

        For each order :math:`(d_u, d_v)` of ``derivatives``, the derivative
        :math:`\frac{\partial^{d_u + d_v} I}{\partial u^{d_u} \partial v^{d_v}}` is
        returned (the image itself with :math:`(0, 0)`). Crossed derivatives are allowed.
        The search of the knots and the B-spline basis functions are computed once for all
        the orders and all the channels (see
        :class:`pysdic.imaging.rect_bivariate_spline.RectBivariateSpline`).

        Points out of the ROI (ie out of
        :math:`[y, y + height - 1] \times [x, x + width - 1]`) or with non-finite
        coordinates get the value ``default_value`` (for all the orders).

        .. warning::

            ``pixel_points`` are a swap of the camera's image points!
            Use :meth:`evaluate_at_image_points` for image points.

        Parameters
        ----------
        pixel_points: ArrayLike
            The pixel points with shape :math:`(N_p, 2)`, each point represented by
            its (row, column) coordinates.

        derivatives : ArrayLike, optional (default: ``((0, 0),)``)
            The orders :math:`(d_u, d_v)` to compute, shape :math:`(R, 2)`, with
            :math:`0 \leq d_u < ku` along the :math:`u`-axis (ie the :math:`y`-axis of
            ``image_coordinates``) and :math:`0 \leq d_v < kv` along the :math:`v`-axis
            (ie the :math:`x`-axis). For example ``((0, 0), (1, 0), (0, 1))`` for the
            image and its gradient :math:`(\partial_u I, \partial_v I)`.

        Returns
        -------
        Tuple[numpy.ndarray, ...]
            :math:`R` arrays in float64 dtype, one per order of ``derivatives`` (in the
            same order), with shape :math:`(N_p,)` for a grayscale image or
            :math:`(N_p, C)` for a color image.

        Raises
        ------
        ValueError
            If the interpolation function is not constructed or a derivative order is
            out of range.

        Examples
        --------
        .. code-block:: python
            :linenos:

            import numpy
            from pysdic import Image

            image = Image.from_array(numpy.random.rand(480, 640))
            pixel_points = numpy.array([[100.5, 200.25], [300.0, 10.75]])

            (values,) = image.evaluate_at_pixel_points(pixel_points)
            values, d_du, d_dv = image.evaluate_at_pixel_points(pixel_points, ((0, 0), (1, 0), (0, 1)))

        """
        self._check_interpolation_functions_constructed()
        pixel_points = _as_array(
            pixel_points, "pixel_points", numpy.float64, shape=(None, 2), finite=False
        )
        orders = self._validate_derivatives(derivatives)

        # Mask points out of the ROI (bounds included, NaN coordinates are excluded)
        x, y, width, height = self._get_effective_roi()
        valid_mask = (
            (pixel_points[:, 0] >= y)
            & (pixel_points[:, 0] <= y + height - 1)
            & (pixel_points[:, 1] >= x)
            & (pixel_points[:, 1] <= x + width - 1)
        )
        valid_points = pixel_points[valid_mask]
        evaluations = self._interpolation_function.evaluate(
            valid_points[:, 0], valid_points[:, 1], orders
        )  # (N_valid,) or (N_valid, C) each

        shape = (pixel_points.shape[0],) if self.is_grayscale else (pixel_points.shape[0], self.n_channels)
        outputs = []
        for values in evaluations:
            output = numpy.full(shape, self._default_value, dtype=numpy.float64)
            output[valid_mask] = values
            outputs.append(output)
        return tuple(outputs)

    def evaluate_at_image_points(
        self, image_points: ArrayLike, derivatives: ArrayLike = ((0, 0),)
    ) -> Tuple[numpy.ndarray, ...]:
        r"""
        Evaluate the image and/or several of its derivatives at given ``image_points``
        (ie :math:`(x, y)`-axis), in a single pass.

        For each order :math:`(d_x, d_y)` of ``derivatives``, the derivative
        :math:`\frac{\partial^{d_x + d_y} I}{\partial x^{d_x} \partial y^{d_y}}` is
        returned (the image itself with :math:`(0, 0)`). Crossed derivatives are allowed.

        Equivalent to :meth:`evaluate_at_pixel_points` with swapped coordinates and
        swapped orders :math:`(d_u, d_v) = (d_y, d_x)`.

        Parameters
        ----------
        image_points: ArrayLike
            The image points with shape :math:`(N_p, 2)`, each point represented by
            its (x, y) coordinates.

        derivatives : ArrayLike, optional (default: ``((0, 0),)``)
            The orders :math:`(d_x, d_y)` to compute, shape :math:`(R, 2)`, with
            :math:`0 \leq d_x < kv` and :math:`0 \leq d_y < ku`. For example
            ``((0, 0), (1, 0), (0, 1))`` for the image and its gradient
            :math:`(\partial_x I, \partial_y I)`.

        Returns
        -------
        Tuple[numpy.ndarray, ...]
            :math:`R` arrays in float64 dtype, one per order of ``derivatives`` (in the
            same order), with shape :math:`(N_p,)` for a grayscale image or
            :math:`(N_p, C)` for a color image.

        Raises
        ------
        ValueError
            If the interpolation function is not constructed or a derivative order is
            out of range.

        """
        image_points = _as_array(
            image_points, "image_points", numpy.float64, shape=(None, 2), finite=False
        )
        orders = _as_integer_array(derivatives, "derivatives", shape=(None, 2), shape_hint="(R, 2)")
        return self.evaluate_at_pixel_points(image_points[:, [1, 0]], orders[:, [1, 0]])

    @staticmethod
    def pixel_points_to_image_points(pixel_points: ArrayLike) -> numpy.ndarray:
        r"""
        Convert pixel points :math:`(u, v)` to image points :math:`(x, y)` by swapping
        the two coordinates.

        Parameters
        ----------
        pixel_points: ArrayLike
            An array of shape (..., 2) representing the pixel points (rows, columns).

        Returns
        -------
        numpy.ndarray
            An array of shape (..., 2) representing the image points :math:`(x, y)`
            in float64 dtype.

        """
        pixel_points = _as_batch_array(
            pixel_points, "pixel_points", 2, numpy.float64, finite=False
        )
        return pixel_points[..., [1, 0]]

    @staticmethod
    def image_points_to_pixel_points(image_points: ArrayLike) -> numpy.ndarray:
        r"""
        Convert image points :math:`(x, y)` to pixel points :math:`(u, v)` by swapping
        the two coordinates.

        Parameters
        ----------
        image_points: ArrayLike
            An array of shape (..., 2) representing the image points :math:`(x, y)`.

        Returns
        -------
        numpy.ndarray
            An array of shape (..., 2) representing the pixel points (rows, columns)
            in float64 dtype.

        """
        image_points = _as_batch_array(
            image_points, "image_points", 2, numpy.float64, finite=False
        )
        return image_points[..., [1, 0]]

    def get_image_pixel_points(self, mask: Optional[ArrayLike] = None) -> numpy.ndarray:
        r"""
        Create a grid of pixel points covering the entire image.

        The method is equivalent to the following code:

        .. code-block:: python

            import numpy

            pixel_points = numpy.indices((height, width)).reshape(2, -1).T
            pixel_points = pixel_points[mask.flatten()] # If mask is provided
            pixel_points = pixel_points.astype(numpy.float64)

        Parameters
        ----------
        mask : Optional[ArrayLike], optional (default: ``None``)
            A boolean mask to filter the pixel points, with shape :math:`(H \times W,)`
            or :math:`(H, W)` where :math:`H` and :math:`W` are the height and width
            of the image. If None, all pixel points are returned.

        Returns
        -------
        numpy.ndarray
            An array of shape :math:`(N_p, 2)` in float64 dtype. Each row represents
            a pixel point (row, column).

        Raises
        ------
        ValueError
            If the mask has a wrong shape.

        TypeError
            If the mask is not a boolean array.

        """
        return _pixel_points_grid(self.height, self.width, mask)

    def get_image_image_points(self, mask: Optional[ArrayLike] = None) -> numpy.ndarray:
        r"""
        Create a grid of image points covering the entire image.

        Equivalent to :meth:`get_image_pixel_points` with swapped coordinates.

        Parameters
        ----------
        mask : Optional[ArrayLike], optional (default: ``None``)
            A boolean mask to filter the image points, with shape :math:`(H \times W,)`
            or :math:`(H, W)` where :math:`H` and :math:`W` are the height and width
            of the image. If None, all image points are returned.

        Returns
        -------
        numpy.ndarray
            An array of shape :math:`(N_p, 2)` in float64 dtype. Each row represents
            an image point :math:`(x, y)`.

        """
        return self.get_image_pixel_points(mask=mask)[:, [1, 0]]

    def get_interpolation_function(self) -> Optional[RectBivariateSpline]:
        r"""
        Get the interpolation function of the image (all the channels).

        The scipy spline of each channel is available with its ``scipy_splines``
        attribute.

        .. warning::

            The interpolation function takes ``pixel_points`` :math:`(u, v)` as input,
            which are a swap of the camera's image points!

        Returns
        -------
        Optional[:class:`pysdic.imaging.rect_bivariate_spline.RectBivariateSpline`]
            The interpolation function, multi-channel for a color image (its evaluations
            have a last axis :math:`C`), or None if it is not constructed.

        """
        return self._interpolation_function

    # ==================================================================
    # Visualization methods
    # ==================================================================
    def quick_visualize(
        self,
        *,
        cmap: Optional[str] = "gray",
        clim: Optional[Tuple[Real, Real]] = None,
        figsize: Tuple[Real, Real] = (8, 6),
        title: Optional[str] = None,
    ) -> None:
        r"""
        Quickly visualize the image with :class:`pysdic.ImagePlotter` (color images are
        assumed in BGR order, as read by :meth:`from_file`).

        .. seealso::

            For more control over the visualization, use :class:`pysdic.ImagePlotter`.

        Parameters
        ----------
        cmap : Optional[str], optional (default: ``"gray"``)
            The colormap of a grayscale image.

        clim : Optional[Tuple[Real, Real]], optional
            The color limits (min, max). By default, the extrema of the image.

        figsize : Tuple[Real, Real], optional (default: (8, 6))
            The size of the figure in inches (width, height).

        title : Optional[str], optional (default: ``None``)
            The title to display on the visualization. If None, no title will be shown.

        """
        from .image_plotter import ImagePlotter

        plotter = ImagePlotter((self.height, self.width), figsize=figsize)
        plotter.add_image(self, cmap=cmap, clim=clim)
        plotter.set_title(title)
        plotter.show()