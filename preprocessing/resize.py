from collections.abc import Iterable

import cv2
import numpy as np

TARGET_SIZE = (96, 96)
DEFAULT_PADDING_VALUE = 0

def select_interpolation(
    original_size: tuple[int, int],
    resized_size: tuple[int, int],
) -> int:
    original_width, original_height = original_size
    resized_width, resized_height = resized_size

    if resized_width < original_width or resized_height < original_height:
        return cv2.INTER_AREA

    return cv2.INTER_CUBIC


def padding_value(image: np.ndarray) -> int | tuple[int, ...]:
    if image.ndim == 2:
        return DEFAULT_PADDING_VALUE

    channels = image.shape[2]
    if channels == 1:
        return DEFAULT_PADDING_VALUE

    return tuple([DEFAULT_PADDING_VALUE] * channels)


def apply_resize(image: np.ndarray) -> np.ndarray:
    target_width, target_height = TARGET_SIZE
    original_height, original_width = image.shape[:2]

    scale = min(target_width / original_width, target_height / original_height)
    resized_width = max(1, int(round(original_width * scale)))
    resized_height = max(1, int(round(original_height * scale)))

    interpolation = select_interpolation(
        (original_width, original_height),
        (resized_width, resized_height),
    )

    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=interpolation,
    )

    if resized.ndim == 2:
        canvas_shape = (target_height, target_width)
    else:
        canvas_shape = (target_height, target_width, resized.shape[2])

    canvas = np.full(
        canvas_shape,
        padding_value(image),
        dtype=image.dtype,
    )

    offset_x = (target_width - resized_width) // 2
    offset_y = (target_height - resized_height) // 2
    canvas[
        offset_y:offset_y + resized_height,
        offset_x:offset_x + resized_width,
    ] = resized

    return canvas

