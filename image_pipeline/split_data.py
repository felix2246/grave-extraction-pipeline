"""
Split a labeled dataset into training and testing sets, preserving paired image and annotation files.
"""

import glob
import os
import random
import shutil

SOURCE = "grave_image_matching/output/katalog2/images"
TRAIN_DEST = "grave_image_matching/output/katalog2/images/train"
TEST_DEST = "grave_image_matching/output/katalog2/images/test"
TRAIN_RATIO = 0.8


def split_dataset(source_dir: str, train_dir: str, test_dir: str) -> None:
    """Split dataset files into train and test directories."""
    if not os.path.exists(source_dir):
        print(f"Error: Source directory '{source_dir}' does not exist.")
        return

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # group files by their filename (to keep .json and .png/.jpg together)
    json_files = glob.glob(os.path.join(source_dir, "*.json"))

    if not json_files:
        print("No JSON files found in source directory.")
        return

    # Shuffle the data
    random.seed(42)
    random.shuffle(json_files)

    # calculate split index
    split_idx = int(len(json_files) * TRAIN_RATIO)
    train_files = json_files[:split_idx]
    test_files = json_files[split_idx:]

    print(f"Found {len(json_files)} items. Splitting into:")
    print(f" - Train: {len(train_files)}")
    print(f" - Test:  {len(test_files)}")

    def move_pair(json_path: str, target_folder: str) -> None:
        """Move JSON file and its associated image to target folder."""
        filename = os.path.basename(json_path)
        shutil.move(json_path, os.path.join(target_folder, filename))

        base_name = os.path.splitext(filename)[0]
        possible_exts = [".png", ".jpg", ".jpeg", ".JPG", ".PNG"]

        found_image = False
        for ext in possible_exts:
            img_path = os.path.join(source_dir, base_name + ext)
            if os.path.exists(img_path):
                shutil.move(img_path, os.path.join(target_folder, base_name + ext))
                found_image = True
                break

        if not found_image:
            print(f"Warning: Could not find image for {filename}")

    print("Moving files...")
    for f in train_files:
        move_pair(f, train_dir)

    for f in test_files:
        move_pair(f, test_dir)

    print(f"Done! Files moved to {TRAIN_DEST} and {TEST_DEST}.")


if __name__ == "__main__":
    split_dataset(SOURCE, TRAIN_DEST, TEST_DEST)
