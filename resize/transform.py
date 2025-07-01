import logging
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


def scale_objects(
    objects: List[Dict[str, Any]],
    scale_x: float,
    scale_y: float,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Scale and offset bounding boxes.

    Args:
        objects: List of objects with 'bounding_box' key.
        scale_x: Horizontal scale factor.
        scale_y: Vertical scale factor.
        offset_x: Horizontal offset to add after scaling.
        offset_y: Vertical offset to add after scaling.

    Returns:
        List of objects with transformed bounding boxes.
    """
    if scale_x <= 0 or scale_y <= 0:
        logging.warning(f"Invalid scale factors: x={scale_x}, y={scale_y}")
        return objects

    scaled_objects = []
    for i, obj in enumerate(objects):
        if "bounding_box" not in obj:
            logging.warning(f"Object {i} missing 'bounding_box' field, skipping")
            continue

        try:
            scaled_obj = obj.copy()
            x1, y1, x2, y2 = obj["bounding_box"]
            scaled_obj["bounding_box"] = [
                x1 * scale_x + offset_x,
                y1 * scale_y + offset_y,
                x2 * scale_x + offset_x,
                y2 * scale_y + offset_y,
            ]
            scaled_objects.append(scaled_obj)

        except (ValueError, TypeError) as e:
            logging.warning(f"Error scaling object {i}: {e}")
            continue

    return scaled_objects


def apply_stretch(
    img: np.ndarray,
    objects: List[Dict[str, Any]],
    new_w: int,
    new_h: int,
    interpolation: int = cv2.INTER_LINEAR,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Resize image and scale bounding boxes accordingly.

    Args:
        img: Input image.
        objects: List of objects with bounding boxes.
        new_w: Target width.
        new_h: Target height.
        interpolation: OpenCV interpolation method.

    Returns:
        Tuple of (resized_image, scaled_objects).
    """
    if img is None or img.size == 0:
        logging.error("Invalid input image")
        return img, objects

    h, w = img.shape[:2]
    scale_x = new_w / w
    scale_y = new_h / h

    try:
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
        scaled_objects = scale_objects(objects, scale_x, scale_y)
        return resized_img, scaled_objects

    except Exception as e:
        logging.error(f"Error during image stretching: {e}")
        return img, objects


def apply_letterbox(
    img: np.ndarray,
    objects: List[Dict[str, Any]],
    new_w: int,
    new_h: int,
    color: Tuple[int, int, int] = (114, 114, 114),
    interpolation: int = cv2.INTER_LINEAR,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Resize image to fit within a letterbox of specified size.

    Args:
        img: Input image.
        objects: List of objects with bounding boxes.
        new_w: Target width.
        new_h: Target height.
        color: Color for padding (BGR).
        interpolation: OpenCV interpolation method.

    Returns:
        Tuple of (letterboxed_image, scaled_objects).
    """
    if img is None or img.size == 0:
        logging.error("Invalid input image")
        return img, objects

    h, w = img.shape[:2]

    # Calculate scale to maintain aspect ratio
    scale = min(new_w / w, new_h / h)
    scaled_w = int(w * scale)
    scaled_h = int(h * scale)
    pad_x = (new_w - scaled_w) // 2
    pad_y = (new_h - scaled_h) // 2

    try:
        # Resize image maintaining aspect ratio
        img_resized = cv2.resize(img, (scaled_w, scaled_h), interpolation=interpolation)

        # Create canvas
        if len(img.shape) == 3:
            canvas = np.full((new_h, new_w, 3), color, dtype=np.uint8)
        else:
            canvas = np.full((new_h, new_w), color[0], dtype=np.uint8)

        # Place resized image in center
        canvas[pad_y : pad_y + scaled_h, pad_x : pad_x + scaled_w] = img_resized

        # Scale objects and apply offset
        scaled_objects = scale_objects(objects, scale, scale, pad_x, pad_y)
        return canvas, scaled_objects

    except Exception as e:
        logging.error(f"Error during letterboxing: {e}")
        return img, objects
