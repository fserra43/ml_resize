import argparse
import logging
from pathlib import Path
from time import perf_counter

import cv2
from tqdm import tqdm

from .kitti import load_kitti_labels, write_kitti_labels
from .transform import apply_letterbox, apply_stretch

# Available transformation modes
MODES = {"stretch": apply_stretch, "letterbox": apply_letterbox}

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def get_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(description="Resize images and KITTI annotations")

    parser.add_argument(
        "--in_img",
        type=Path,
        default=Path("data/images"),
        help="Input images directory",
    )
    parser.add_argument(
        "--in_lbl",
        type=Path,
        default=Path("data/kitti_annotations"),
        help="Input labels directory",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("results"), help="Output directory"
    )
    parser.add_argument(
        "--size",
        nargs=2,
        type=int,
        default=[284, 284],
        metavar=("W", "H"),
        help="Target size (width height)",
    )
    parser.add_argument(
        "--mode", choices=MODES.keys(), default="letterbox", help="Resize mode"
    )
    parser.add_argument("--img_ext", default="jpg", help="Image file extension")

    return parser.parse_args()


def main() -> int:
    """
    Main processing function.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    args = get_args()
    logging.info("Starting image resize process")

    # Check inputs exist
    if not args.in_img.exists():
        logging.error(f"Images directory not found: {args.in_img}")
        return 1

    if not args.in_lbl.exists():
        logging.error(f"Labels directory not found: {args.in_lbl}")
        return 1

    # Setup paths
    new_w, new_h = args.size
    transform_fn = MODES[args.mode]
    out_imgs = args.out / "images"
    out_lbls = args.out / "labels"

    logging.info(f"Using {args.mode} mode, target size: {new_w}x{new_h}")

    # Find images - support multiple extensions
    images = list(args.in_img.glob(f"*.{args.img_ext}"))
    images.extend(args.in_img.glob(f"*.{args.img_ext.upper()}"))
    images = sorted(set(images))  # Remove duplicates and sort

    if not images:
        logging.error(f"No images found with extension .{args.img_ext}")
        return 1

    logging.info(f"Found {len(images)} images to process")

    # Create output directories
    out_imgs.mkdir(parents=True, exist_ok=True)
    out_lbls.mkdir(parents=True, exist_ok=True)
    logging.info(f"Created output directories: {args.out}")

    # Process images
    start_time = perf_counter()
    processed = 0
    errors = 0

    for img_path in tqdm(images, desc="Processing"):
        try:
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                logging.warning(f"Could not load image: {img_path.name}")
                errors += 1
                continue

            # Load labels
            label_path = args.in_lbl / f"{img_path.stem}.txt"
            objects = load_kitti_labels(label_path)
            logging.debug(f"Loaded {len(objects)} objects from {label_path.name}")

            # Transform
            img_resized, objects_resized = transform_fn(img, objects, new_w, new_h)

            # Save results
            out_img_path = out_imgs / img_path.name
            success = cv2.imwrite(str(out_img_path), img_resized)
            if not success:
                logging.warning(f"Failed to save image: {img_path.name}")
                errors += 1
                continue

            out_lbl_path = out_lbls / f"{img_path.stem}.txt"
            if not write_kitti_labels(out_lbl_path, objects_resized):
                logging.warning(f"Failed to save labels: {img_path.stem}.txt")
                errors += 1
                continue

            processed += 1
            logging.debug(f"Successfully processed: {img_path.name}")

        except Exception as e:
            logging.error(f"Error processing {img_path.name}: {e}")
            errors += 1
            continue

    # Summary
    duration = perf_counter() - start_time
    logging.info(f"Processing completed in {duration:.1f}s")
    logging.info(f"Successfully processed: {processed}/{len(images)} images")

    if errors > 0:
        logging.warning(f"Encountered {errors} errors")
        return 1

    logging.info("All images processed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())
