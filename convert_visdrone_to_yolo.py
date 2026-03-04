#!/usr/bin/env python3
"""Convert VisDrone annotations to YOLO format"""

from pathlib import Path
from PIL import Image
from tqdm import tqdm


def convert_visdrone_to_yolo(img_dir, ann_dir, output_dir):
    """
    VisDrone format: <x>,<y>,<w>,<h>,<score>,<class>,<truncation>,<occlusion>
    YOLO format: <class> <x_center> <y_center> <width> <height> (normalized)

    VisDrone classes (0-indexed in file, but we map 0->ignore):
    0: ignored regions
    1: pedestrian -> 0
    2: people -> 1
    3: bicycle -> 2
    4: car -> 3
    5: van -> 4
    6: truck -> 5
    7: tricycle -> 6
    8: awning-tricycle -> 7
    9: bus -> 8
    10: motor -> 9
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    ann_files = list(Path(ann_dir).glob("*.txt"))

    for ann_file in tqdm(ann_files, desc=f"Converting {ann_dir}"):
        img_file = Path(img_dir) / (ann_file.stem + ".jpg")

        if not img_file.exists():
            continue

        # Get image dimensions
        try:
            img = Image.open(img_file)
            img_w, img_h = img.size
        except:
            continue

        # Read and convert annotations
        yolo_lines = []
        with open(ann_file, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue

                x, y, w, h = map(int, parts[:4])
                visdrone_class = int(parts[5])

                # Skip ignored regions (class 0) and invalid classes
                if visdrone_class == 0 or visdrone_class > 10:
                    continue

                # Map VisDrone class to YOLO class (1-10 -> 0-9)
                yolo_class = visdrone_class - 1

                # Convert to YOLO format (normalized)
                x_center = (x + w / 2) / img_w
                y_center = (y + h / 2) / img_h
                norm_w = w / img_w
                norm_h = h / img_h

                # Skip invalid boxes
                if norm_w <= 0 or norm_h <= 0 or x_center > 1 or y_center > 1:
                    continue

                yolo_lines.append(
                    f"{yolo_class} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n"
                )

        # Write YOLO format file
        output_file = output_dir / ann_file.name
        with open(output_file, "w") as f:
            f.writelines(yolo_lines)


if __name__ == "__main__":
    # Convert train set
    convert_visdrone_to_yolo(
        "images/train/images", "images/train/annotations", "images/train/labels"
    )

    # Convert validation set
    convert_visdrone_to_yolo(
        "images/validate/images",
        "images/validate/annotations",
        "images/validate/labels",
    )

    print("\nConversion complete!")
