import argparse
import random
import shutil
from pathlib import Path
from util import remove_directory, gen_image_list
from config import ALL_LABELS

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_data", type=Path, default=Path("data/raw/fer2013/train"))
    parser.add_argument("out_data", type=Path, default=Path("data/test_subset/train"))
    args = parser.parse_args()
    remove_directory(args.out_data)
    images_per_emotion = 10

    for cat in ALL_LABELS:
        cat_dir = args.raw_data  / cat
        if not cat_dir.exists():
            raise FileNotFoundError(f"Missing category {cat}")
        images = gen_image_list(cat_dir)
        sample = random.sample(images, images_per_emotion)
        dst = args.out / cat
        dst.mkdir(parents=True, exist_ok=True)
        for path in sample:
            shutil.copy2(path, dst / path.name)

    print("Testing subset created")


if __name__ == '__main__':
    main()