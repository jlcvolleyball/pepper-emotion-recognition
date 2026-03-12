"""
preprocess_fer2013.py

This code runs the preprocessing for the pipeline for the FER2013 dataset. The
dataset did not include a validation set, so this preprocessing code separates
the given train set (using some user-inputted ratio, or default to 0.1) into
the training and validation set.

For more details on the entire pipeline, visit fer2013_default_pipeline.py
"""

import argparse
import random
from pathlib import Path
from util import remove_directory, gen_image_list
from config import ALL_LABELS
import shutil

def copy_file(src, dst):
    """
    Copies the image from specified source (src) to specified destination (dst)
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists(): return
    shutil.copy2(src, dst)

def process(raw_data, out_data, val_ratio):
    """
    Creates a processed folder of the dataset, with separated data that represent the training,
    validation, and testing datasets. Note that from the Kaggle dataset, FER2013 does not include
    a validation split. val_ratio specifies how much of each class from the raw training data we
    want to put in our validation set.
    """
    train_raw = raw_data / "train"
    test_raw = raw_data / "test"
    if not train_raw.exists() or not test_raw.exists():
        raise FileNotFoundError("Train and test folders are not present in your specified directory.")

    # if the out_data already exists, remove it safely
    remove_directory(out_data)
    # create the training, validation, and testing directories in the out_data folder
    (out_data / "train").mkdir(parents=True, exist_ok=True)
    (out_data / "val").mkdir(parents=True, exist_ok=True)
    (out_data / "test").mkdir(parents=True, exist_ok=True)
    rand = random.Random()

    for cat in ALL_LABELS:
        # split the raw training into training and validation
        cat_train_loc = train_raw / cat
        if not cat_train_loc.exists():
            raise FileNotFoundError(f"Missing category {cat} in train")
        cat_train_data = gen_image_list(cat_train_loc)
        num_val = int(len(cat_train_data) * val_ratio)
        rand.shuffle(cat_train_data)
        train_data, val_data = cat_train_data[num_val:], cat_train_data[:num_val]
        for src in train_data:
            dst = out_data / "train" / cat / src.name
            copy_file(src, dst)
        for src in val_data:
            dst = out_data / "val" / cat / src.name
            copy_file(src, dst)
        # copy the raw test data as is into the new processed directory
        cat_test_loc = test_raw / cat
        if not cat_test_loc.exists():
            raise FileNotFoundError(f"Missing category {cat} in test")
        cat_test_data = gen_image_list(cat_test_loc)
        for src in cat_test_data:
            dst = out_data / "test" / cat / src.name
            copy_file(src, dst)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_data", type=Path, nargs="?", default=Path("data/processed/fer2013_older_0.6"))
    parser.add_argument("out_data", type=Path, nargs="?", default=Path("data/processed/fer2013_older_0.6_proc"))
    parser.add_argument("val_ratio", type=float, nargs="?", default=0.1)
    args = parser.parse_args()

    # process into train, validation, and test
    process(args.raw_data, args.out_data, args.val_ratio)

if __name__ == '__main__':
    main()