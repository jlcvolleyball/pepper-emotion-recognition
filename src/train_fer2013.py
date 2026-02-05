import argparse
import datasets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str, default="data/processed/fer2013")
    parser.add_argument("out", type=str, default="artifacts/resnet50-fer2013")
    parser.add_argument("epochs", type=int, default=3) # NEED TO CHANGE
    parser.add_argument("lr", type=float, default=1e-5) # NEED TO CHANGE
    parser.add_argument("batch", type=int, default=60) # NEED TO CHANGE
    args = parser.parse_args()

    dataset = datasets.load_dataset("imagefolder", data_dir=args.data)
    labels = dataset["train"].features["labels"].name
    label_to_idx = {label: i for i, label in enumerate(labels)}
    idx_to_label = {i: label for i, label in enumerate(labels)}

if __name__ == '__main__':
    main()