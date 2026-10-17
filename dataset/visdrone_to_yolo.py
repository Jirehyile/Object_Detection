"""Convert VisDrone detection annotations to Ultralytics YOLO labels."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


SPLITS = (
    "VisDrone2019-DET-train",
    "VisDrone2019-DET-val",
    "VisDrone2019-DET-test-dev",
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def convert_split(data_root: Path, split_name: str) -> dict[str, int]:
    split_dir = data_root / split_name
    images_dir = split_dir / "images"
    annotations_dir = split_dir / "annotations"
    labels_dir = split_dir / "labels"

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")
    if not annotations_dir.is_dir():
        raise FileNotFoundError(f"Annotation directory not found: {annotations_dir}")

    labels_dir.mkdir(exist_ok=True)
    stats = {"images": 0, "valid": 0, "ignored": 0, "invalid": 0}

    image_paths = sorted(
        path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )

    for image_path in image_paths:
        annotation_path = annotations_dir / f"{image_path.stem}.txt"
        label_path = labels_dir / f"{image_path.stem}.txt"

        with Image.open(image_path) as image:
            image_width, image_height = image.size

        yolo_lines: list[str] = []
        if annotation_path.is_file():
            for raw_line in annotation_path.read_text(encoding="utf-8-sig").splitlines():
                fields = raw_line.strip().rstrip(",").split(",")
                if len(fields) < 8:
                    stats["invalid"] += 1
                    continue

                try:
                    x, y, width, height = map(float, fields[:4])
                    score = int(fields[4])
                    category = int(fields[5])
                except ValueError:
                    stats["invalid"] += 1
                    continue

                # VisDrone categories 0 and 11 are not part of the 10 classes.
                if score == 0 or not 1 <= category <= 10:
                    stats["ignored"] += 1
                    continue

                x1 = max(0.0, x)
                y1 = max(0.0, y)
                x2 = min(float(image_width), x + width)
                y2 = min(float(image_height), y + height)
                clipped_width = x2 - x1
                clipped_height = y2 - y1

                if clipped_width <= 0 or clipped_height <= 0:
                    stats["invalid"] += 1
                    continue

                x_center = ((x1 + x2) / 2.0) / image_width
                y_center = ((y1 + y2) / 2.0) / image_height
                normalized_width = clipped_width / image_width
                normalized_height = clipped_height / image_height
                yolo_class = category - 1

                yolo_lines.append(
                    f"{yolo_class} {x_center:.6f} {y_center:.6f} "
                    f"{normalized_width:.6f} {normalized_height:.6f}"
                )
                stats["valid"] += 1

        label_path.write_text(
            "\n".join(yolo_lines) + ("\n" if yolo_lines else ""),
            encoding="utf-8",
        )
        stats["images"] += 1

    return stats


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[2] / "Data"
    parser = argparse.ArgumentParser(
        description="Convert VisDrone train, val and test-dev labels to YOLO format."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_root,
        help=f"Dataset root (default: {default_root})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {data_root}")

    print(f"Dataset root: {data_root}")
    for split_name in SPLITS:
        stats = convert_split(data_root, split_name)
        print(
            f"{split_name}: images={stats['images']}, valid={stats['valid']}, "
            f"ignored={stats['ignored']}, invalid={stats['invalid']}"
        )


if __name__ == "__main__":
    main()
