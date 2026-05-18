"""
preproess_elderreact.py

Preprocesses raw ElderReact dataset into a CSV file with filename and
valence score columns and copies video clips into processed data folder
"""

from __future__ import annotations

import shutil
from pathlib import Path
import pandas as pd

raw_root = Path("data/raw/ElderReact_Data")
raw_train = raw_root / "ElderReact_train"
raw_test = raw_root / "ElderReact_test"
raw_val = raw_root / "ElderReact_dev"

raw_train_labels = raw_root / "train_labels.txt"
raw_test_labels = raw_root / "test_labels.txt"
raw_val_labels = raw_root / "dev_labels.txt"

out_root = Path("data/processed/elderreact")
out_clips = out_root / "clips"
out_train_csv = out_root / "train.csv"
out_test_csv = out_root / "test.csv"
out_val_csv = out_root / "val.csv"

def parse_labels(label_path):
    """
    Reads the path to the file with the labels. Only picks out the
    first column, which should be the name of the file, and the last
    column, which has the valence score.
    """
    rows = []
    with label_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start = 1):
            line = line.strip()
            if not line: continue
            parts = line.split()
            filename = parts[0]
            valence = float(parts[-1])
            rows.append({
                "filename": filename,
                "valence": valence,
            })
    df = pd.DataFrame(rows)
    print(df)
    return df

def copy_videos(df, src_dir, clips_dir):
    """
    Copies videos from the src directory into the clips directory
    """
    clips_dir.mkdir(parents=True, exist_ok=True)
    for filename in df["filename"]:
        src = src_dir / filename
        dst = clips_dir / filename
        if not src.exists():
            raise FileNotFoundError(f"Video {src} missing")
        if not dst.exists():
            shutil.copy2(src, dst)

def save_csv(df, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = df[["filename", "valence"]].copy()
    df.to_csv(out_path, index=False)

def main():
    out_root.mkdir(parents=True, exist_ok=True)
    out_clips.mkdir(parents=True, exist_ok=True)
    train_df = parse_labels(raw_train_labels)
    test_df = parse_labels(raw_test_labels)
    val_df = parse_labels(raw_val_labels)

    save_csv(train_df, out_train_csv)
    save_csv(test_df, out_test_csv)
    save_csv(val_df, out_val_csv)

    copy_videos(train_df, raw_train, out_clips)
    copy_videos(test_df, raw_test, out_clips)
    copy_videos(val_df, raw_val, out_clips)

if __name__ == '__main__':
    main()
