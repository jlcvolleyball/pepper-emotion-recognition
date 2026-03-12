"""
fer2013_default_pipeline.py

This script runs the source code necessary for the full filtering, preprocessing, and
training pipeline. The dataset used is FER2013, filtered into a subset that only
includes faces from older adults. Training is completed using the ResNet50
architecture on this subset. Augmentation, freezing, and weight decay are
included during training (for more details, visit train_fer2013.py).

Purpose in research: As of 3/11/26, this is the pipeline used to create one of
the first iterations of the emotion detection model for older adults, which
unfortunately did not achieve a high accuracy rate (~30%). Alternative
methods were explored, including pivoting to another dataset (ElderReact) or
first training on the entire FER2013 dataset, and then fine-tuning on our
older adults subset.

Source code run:
- filter_fer2013_age.py
- preprocess_fer2013.py
- train_fer2013.py
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

def run_step(cmd, step_name):
    print(f"\n===== Running: {step_name} =====")
    print("Command:", " ".join(map(str, cmd)))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nStep failed: {step_name}")
        sys.exit(result.returncode)
    print(f"===== Finished: {step_name} =====\n")

def main():
    python_exe = sys.executable

    conf_score = 0.6
    valid_ratio = 0.1
    epochs = 30
    lr = 1e-3
    batch = 8

    raw_fer_dir_train = Path("data/raw/fer2013/train")
    raw_fer_dir_test = Path("data/raw/fer2013/test")
    filtered_data_train = Path(f"data/processed/fer2013_older_{conf_score}/train")
    filtered_data_test = Path(f"data/processed/fer2013_older_{conf_score}/test")
    filtered_data = Path(f"data/processed/fer2013_older_{conf_score}")
    processed_dir = Path(f"data/processed/fer2013_older_{conf_score}_proc")

    filter_script = Path("filter_fer2013_age.py")
    preprocess_script = Path("preprocess_fer2013.py")
    train_script = Path("train_fer2013.py")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"artifacts/resnet50-fer2013_{timestamp}")

    # filtering round 1: training data (creates subset of dataset of older adults)
    filter_cmds1 = [
        python_exe,
        str(filter_script),
        str(raw_fer_dir_train),
        str(filtered_data_train),
        str(conf_score)
    ]
    run_step(filter_cmds1, "Filtering for Train")

    # filtering round 2: test data (creates subset of dataset of older adults)
    filter_cmds2 = [
        python_exe,
        str(filter_script),
        str(raw_fer_dir_test),
        str(filtered_data_test),
        str(conf_score)
    ]
    run_step(filter_cmds2, "Filtering for Test")

    # run preprocessing (extracts some percentage from train dataset to validation)
    preprocess_cmds = [
        python_exe,
        str(preprocess_script),
        str(filtered_data),
        str(processed_dir),
        str(valid_ratio)
    ]
    run_step(preprocess_cmds, "Preprocessing dataset for validation")

    # run training code
    train_cmds = [
        python_exe,
        str(train_script),
        str(processed_dir),
        str(output_dir),
        str(epochs),
        str(lr),
        str(batch)
    ]
    run_step(train_cmds, "Training")

    print("Pipeline was completed successfully!")

if __name__ == '__main__':
    main()