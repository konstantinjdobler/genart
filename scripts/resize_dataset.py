import argparse
import sys
import cv2
from pathlib import Path
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser("Parsing resizing arguments")

parser.add_argument("--size", type=str, default=0)
parser.add_argument("--source", type=str)
parser.add_argument("--destination", type=str)

IMAGE_FILE_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png"]


def resize_image(image: np.ndarray, size: list) -> np.ndarray:
    if len(size) > 2 or type(size[0]) is not int or type(size[-1]) is not int:
        raise AttributeError("Please supply only one or two ints.")

    size_x, size_y = size[0], size[-1]
    # Set sizes to keep aspect ratio if only one size is supplied.
    if len(size) == 1:
        target_image_size = size_x
        height, width, _ = image.shape
        # Smaller edge to target_image_size, longer edge rescaled accordingly.
        if width < height:
            size_x, size_y = target_image_size, round(
                height * (target_image_size / width))
        else:
            size_x, size_y = round(
                width * (target_image_size / height)), target_image_size
    return cv2.resize(image,
                      (size_x, size_y),
                      interpolation=cv2.INTER_AREA)


if __name__ == '__main__':
    args = parser.parse_args()
    # Determine image sizes
    keep_aspect_ratio = args.size.isdigit()
    size = [0]
    try:
        size = [int(size) for size in args.size.split(',')]
        assert len(size) <= 2
    except Exception:
        print("Please specify the size as one int to keep the aspect ratio or int,int for specific resize values.",
              file=sys.stderr)
        sys.exit(-1)

    # Collect all image paths
    image_paths = []
    for extension in IMAGE_FILE_EXTENSIONS:
        image_paths.extend(list(Path(args.source).rglob(extension)))
    print(f'Number of images found: {len(image_paths)}')
    for image_path in tqdm(image_paths, desc="Resizing images..."):
        # Construct output path
        relative_path = image_path.relative_to(args.source)
        destination_path = Path(args.destination) / relative_path
        destination_path_with_correct_suffix = destination_path.with_suffix(
            ".jpg").resolve()
        destination_path_with_correct_suffix.parent.mkdir(
            parents=True, exist_ok=True)
        try:
            # Load resize and save image
            original_image = cv2.imread(str(image_path))
            smaller_image = resize_image(original_image, size)
            cv2.imwrite(str(destination_path_with_correct_suffix),
                        smaller_image)
        except Exception as e:
            print(e)
            print(f'Image skipped during exception: {image_path}')
