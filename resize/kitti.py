import logging
from pathlib import Path
from typing import Any, Dict, List, Union

# Basic logging setup
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_kitti_labels(label_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Parse a KITTI label file and return a list of object annotations.

    Args:
        label_path: Path to the KITTI label file

    Returns:
        List of dictionaries, each containing complete KITTI annotation data
    """
    label_path = Path(label_path)
    logging.debug(f"Loading labels from: {label_path}")

    if not label_path.exists():
        logging.debug(f"File not found: {label_path}")
        return []

    objects = []
    try:
        lines = label_path.read_text().splitlines()
        logging.debug(f"Found {len(lines)} lines in file")

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 15:
                logging.warning(
                    f"Line {line_num} has only {len(parts)} fields (needs 15+)"
                )
                continue

            try:
                obj = {
                    "class_name": parts[0],
                    "truncation": float(parts[1]),
                    "occlusion": int(parts[2]),
                    "alpha": float(parts[3]),
                    "bounding_box": [
                        float(parts[4]),
                        float(parts[5]),
                        float(parts[6]),
                        float(parts[7]),
                    ],
                    "3d_dimensions": [
                        float(parts[8]),
                        float(parts[9]),
                        float(parts[10]),
                    ],
                    "location": [float(parts[11]), float(parts[12]), float(parts[13])],
                    "rotation_y": float(parts[14]),
                    "score": float(parts[15]) if len(parts) > 15 else None,
                }
                objects.append(obj)

            except (ValueError, IndexError) as e:
                logging.warning(f"Error parsing line {line_num}: {e}")
                continue

    except Exception as e:
        logging.error(f"Error reading file: {e}")
        return []

    logging.debug(f"Successfully loaded {len(objects)} objects")
    return objects


def write_kitti_labels(
    label_path: Union[str, Path], objects: List[Dict[str, Any]]
) -> bool:
    """
    Write object annotations to a KITTI label file.

    Args:
        label_path: Path where to save the label file
        objects: List of object dictionaries

    Returns:
        True if successful, False otherwise
    """
    label_path = Path(label_path)
    logging.debug(f"Writing {len(objects)} objects to: {label_path}")

    # Create directory if it doesn't exist
    label_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        written_objects = 0

        with label_path.open("w") as f:
            for i, obj in enumerate(objects):
                # Check required fields
                required_fields = [
                    "class_name",
                    "truncation",
                    "occlusion",
                    "alpha",
                    "bounding_box",
                    "3d_dimensions",
                    "location",
                    "rotation_y",
                ]

                missing_fields = [
                    field for field in required_fields if field not in obj
                ]
                if missing_fields:
                    logging.warning(
                        f"Object {i} missing fields: {missing_fields}. Skipping."
                    )
                    continue

                # Check array sizes
                try:
                    if len(obj["bounding_box"]) != 4:
                        logging.warning(f"Object {i}: bounding_box must have 4 values")
                        continue
                    if len(obj["3d_dimensions"]) != 3:
                        logging.warning(f"Object {i}: 3d_dimensions must have 3 values")
                        continue
                    if len(obj["location"]) != 3:
                        logging.warning(f"Object {i}: location must have 3 values")
                        continue
                except (TypeError, KeyError):
                    logging.warning(f"Object {i} has invalid data structure")
                    continue

                # Build the line
                try:
                    line_parts = [
                        str(obj["class_name"]),
                        f"{obj['truncation']:.2f}",
                        str(obj["occlusion"]),
                        f"{obj['alpha']:.6f}",
                        f"{obj['bounding_box'][0]:.2f}",
                        f"{obj['bounding_box'][1]:.2f}",
                        f"{obj['bounding_box'][2]:.2f}",
                        f"{obj['bounding_box'][3]:.2f}",
                        f"{obj['3d_dimensions'][0]:.2f}",
                        f"{obj['3d_dimensions'][1]:.2f}",
                        f"{obj['3d_dimensions'][2]:.2f}",
                        f"{obj['location'][0]:.2f}",
                        f"{obj['location'][1]:.2f}",
                        f"{obj['location'][2]:.2f}",
                        f"{obj['rotation_y']:.6f}",
                    ]

                    # Add score if it exists
                    if obj.get("score") is not None:
                        line_parts.append(f"{obj['score']:.4f}")

                    f.write(" ".join(line_parts) + "\n")
                    written_objects += 1

                except Exception as e:
                    logging.warning(f"Error writing object {i}: {e}")
                    continue

        logging.debug(f"Successfully wrote {written_objects} objects")
        return True

    except Exception as e:
        logging.error(f"Error writing file: {e}")
        return False
