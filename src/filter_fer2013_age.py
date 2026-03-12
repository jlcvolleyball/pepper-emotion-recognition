"""


fer2013_default_pipeline.py

This script runs the source code necessary for the full filtering, preprocessing, and
training pipeline. The dataset used is FER2013, filtered into a subset that only
includes faces from older adults. Training is completed using the ResNet50
architecture on this subset. Augmentation, freezing, and weight decay are
included during training (for more details, visit train_fer2013.py).

"""

import argparse
from pathlib import Path
from util import iter_images

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from collections import Counter

def main():
    device = "cpu"   # device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "nateraw/vit-age-classifier"

    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path, nargs="?", default=Path("data/raw/fer2013/test"))
    parser.add_argument("min_conf", type=float, nargs="?", default=0.50)  # only keep images if the age detector confidence >= min_conf
    parser.add_argument("out", type=Path, nargs="?", default=Path("data/processed/fer2013_older_0.6/test"))
    age_substr = ["60", "70", "80", "90", "elder"]
    ages = [s.lower() for s in age_substr]
    args = parser.parse_args()

    image_processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    model.eval()

    args.out.mkdir(parents=True, exist_ok=True)

    processed_by_emotion = Counter()
    kept_by_emotion = Counter()
    rejected_low_conf = Counter()
    rejected_out_age_range = Counter()
    kept_ct, total_ct = 0, 0
    images = args.data
    for path in iter_images(images):
        emotion_label = path.parent.name
        processed_by_emotion[emotion_label] += 1

        # preprocessing eah image to match the model's expected resolution, etc.
        im = Image.open(path).convert("RGB")
        inputs = image_processor(images=im, return_tensors="pt")
        inputs = inputs.to(device)
        # age prediction here
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        conf_score, idx = torch.max(probs, dim=-1)
        conf = float(conf_score)
        label = model.config.id2label[int(idx)].lower()

        in_age_range = any(sub in label for sub in ages)
        conf_good = conf >= args.min_conf
        # check if it is in our age ranges or if confidence is too low
        if in_age_range and conf_good:
            cur_out = args.out / emotion_label
            cur_out.mkdir(parents=True, exist_ok=True)
            out_path = cur_out / path.name
            out_path.write_bytes(path.read_bytes())
            kept_ct += 1
            kept_by_emotion[emotion_label] += 1
        else:
            if not in_age_range:
                rejected_out_age_range[emotion_label] += 1
            if not conf_good:
                rejected_low_conf[emotion_label] += 1
        total_ct += 1

    print(f"Total images processed: {total_ct}")
    print(f"Kept images (older adults, high confidence): {kept_ct}")

    # print extra stats
    all_emotions = sorted(processed_by_emotion.keys())
    for emotion in all_emotions:
        processed = processed_by_emotion[emotion]
        kept = kept_by_emotion[emotion]
        rej_out_range = rejected_out_age_range[emotion]
        rej_low_conf = rejected_low_conf[emotion]
        keep_rate = (kept / processed) if processed else 0.0
        print(f"{emotion}\t{processed}\t{kept}\t{rej_out_range}\t{rej_low_conf}\t{keep_rate:.3f}")


if __name__ == '__main__':
    main()