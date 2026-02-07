import argparse
from pathlib import Path
from util import iter_images

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

def main():
    device = "cpu"   # device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "nateraw/vit-age-classifier"

    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path, default=Path("data/raw/fer2013/train"))
    parser.add_argument("out", type=Path, default=Path("data/processed/fer2013_older/train"))
    parser.add_argument("min_conf", type=float, default=0.70) # only keep images if the age detector confidence >= min_conf
    age_substr = ["60", "70", "80", "90", "elder"]
    ages = [s.lower() for s in age_substr]
    args = parser.parse_args()

    image_processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    model.eval()

    args.out.mkdir(parents=True, exist_ok=True)
    kept_ct, total_ct = 0, 0
    images = args.data
    for path in iter_images(args.data):
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

        # check if it is in our age ranges or if confidence is too low
        if any(sub in label for sub in ages) and conf >= args.min_conf:
            emotion_label = path.parent.name
            cur_out = args.out / emotion_label
            cur_out.mkdir(parents=True, exist_ok=True)
            out_path = cur_out / path.name
            out_path.write_bytes(path.read_bytes())
            kept_ct += 1
        total_ct += 1

    print(f"Total images processed: {total_ct}")
    print(f"Kept images (older adults, high confidence): {kept_ct}")


if __name__ == '__main__':
    main()