"""
run_elderreact.py

Runs inference and evaluation for Models 2 and 3 (video-only and multimodal) using
the saved checkpoints
"""

import random
from pathlib import Path
import json
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as tvm
import torchvision.transforms as T
import torchaudio

from transformers import Wav2Vec2Model, Wav2Vec2Processor
from train_elderreact import VideoOnlyRegressor, VideoAudioRegressor
from train_elderreact import build_multimodal_dataloaders, build_video_dataloaders

device = "cpu"

# define dataset paths, hyperparameters, etc.
# path information
train_csv = Path("data/processed/elderreact/train.csv")
val_csv = Path("data/processed/elderreact/val.csv")
test_csv = Path("data/processed/elderreact/test.csv")
video_root = Path("data/processed/elderreact/clips")

# training hyperparameters (note: in code rather than by input)
batch_size = 4
num_workers = 2
num_epochs = 10
lr = 1e-4
weight_decay = 1e-4

# video preprocessing settings
num_frames = 16
frame_size = 224
# audio preprocessing settings
audio_sr = 16000
max_audio_seconds = 4.0

# embedding and head sizes
video_emb_dim = 256
audio_emb_dim = 256
hidden_dim = 256
dropout = 0.3

# whether or not to freeze pretrained
freeze_audio_backbone = True
freeze_video_backbone = False

seed = 42

# define valid valence range
min_valence = 1.0
max_valence = 7.0


def sample_frames(num_sample, num_frames):
    """
    This function uniformly samples num_sample frame indices from a video
    Note that if the function is shorter than num_sample, the last available
    frame index is repeated in the sample
    """
    if num_sample < num_frames:
        idxs = np.linspace(0, num_sample-1, num_sample).astype(int).tolist()
        while len(idxs) < num_frames:
            idxs.append(idxs[-1])
        return idxs[:num_frames]
    else:
        idxs = np.linspace(0, num_sample - 1, num_frames).astype(int).tolist()
        return idxs

def load_video_frames(video_path, num_frames, transform):
    """
    Load uniformly sampled RGB frames from a video, and returns a tensor of shape
    [T, C, H, W]
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = sample_frames(total_frames, num_frames)

    idxs_sorted = sorted(set(idxs))
    cur_idx = 0
    cur_wanted = 0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        if cur_wanted < len(idxs_sorted) and cur_idx == idxs_sorted[cur_wanted]:
            # convert BGR format to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.uint8)
            if transform is not None: frame_tensor = transform(frame)
            else: frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            frames.append(frame_tensor)
            cur_wanted += 1
        cur_idx += 1
        if cur_wanted >= len(idxs_sorted): break
    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames loaded from video: {video_path}")
    while len(frames) < num_frames: # if needed, pad with the last frame
        frames.append(frames[-1].clone())
    frames = frames[:num_frames]
    return torch.stack(frames, dim=0)

def load_audio_from_video(video_path, target_sr, max_seconds):
    """
    Loads in the audio from the mp4 file
    Steps:
        1. Read waveform from the video
        2. Convert to mono if necessary
        3. Resample to target_sr
        4. Pad / truncate to a fixed number of seconds
    """
    waveform, sr = torchaudio.load(str(video_path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    waveform = waveform.squeeze(0)
    max_len = int(target_sr * max_seconds)
    if waveform.numel() > max_len:
        waveform = waveform[:max_len]
    elif waveform.numel() < max_len:
        pad_len = max_len - waveform.numel()
        waveform = F.pad(waveform, (0, pad_len))
    return waveform

class ValenceVideoRegressionDataset(Dataset):
    """
    This is dataset for the regression model using the valence scores,
    derived from the mp4 clips
    - The format of the CSV file should be filename,valence
    - Should support both video-only and video+audio
    """
    def __init__(self, csv_path, video_root, num_frames, frame_transform: T.Compose,
                 use_audio:bool = False, audio_sr: int = 16000, max_audio_seconds : float = 4.0,
                 wav2vec_processor: Wav2Vec2Processor | None = None, min_valence: float = 1.0,
                 max_valence: float = 7.0):
        self.csv_path = csv_path
        self.video_root = video_root
        self.num_frames = num_frames
        self.frame_transform = frame_transform
        self.use_audio = use_audio
        self.audio_sr = audio_sr
        self.max_audio_seconds = max_audio_seconds
        self.wav2vec_processor = wav2vec_processor
        self.min_valence = min_valence
        self.max_valence = max_valence

        self.df = pd.read_csv(self.csv_path)
        cols = {"filename","valence"}
        if not cols.issubset(set(self.df.columns)):
            raise ValueError(f"{self.csv_path} must contain columns filename and valence")
        if self.use_audio and self.wav2vec_processor is None:
            raise ValueError("you must provide wav2vec_processor when using use_audio=True")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        row = self.df.iloc[item]
        filename = row["filename"]
        valence = float(row["valence"])
        if not self.min_valence <= valence <= self.max_valence:
            raise ValueError("Valence scores out of range")
        video_path = self.video_root / filename
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        frames = load_video_frames(video_path=video_path, num_frames=self.num_frames, transform=self.frame_transform)
        item_dict = {
            "filename": filename,
            "frames": frames,
            "label": torch.tensor(valence, dtype=torch.float32)
        }
        if self.use_audio:
            waveform = load_audio_from_video(video_path=video_path, target_sr = self.audio_sr, max_seconds=self.max_audio_seconds)
            processed = self.wav2vec_processor(waveform.numpy(), sampling_rate=self.audio_sr, return_tensors="pt", padding=False)
            item_dict["audio_values"] = processed["input_values"].squeeze(0)
            if "attention_mask" in processed:
                item_dict["audio_mask"] = processed["attention_mask"].squeeze(0)
            else:
                item_dict["audio_mask"] = torch.ones_like(item_dict["audio_values"], dtype=torch.long)
        return item_dict

def collate_video_only(batch):
    """
    Batch the video-only examples
    """
    frames = torch.stack([x["frames"] for x in batch], dim=0)
    labels = torch.stack([x["label"] for x in batch], dim = 0)
    filenames = [x["filename"] for x in batch]
    return {
        "filename": filenames,
        "frames": frames,
        "label": labels
    }

def collate_video_audio(batch):
    """
    Batch video and audio examples
    """
    frames = torch.stack([x["frames"] for x in batch], dim = 0)
    labels = torch.stack([x["label"] for x in batch], dim = 0)
    audio_vals = [x["audio_values"] for x in batch]
    audio_masks = [x["audio_mask"] for x in batch]
    filenames = [x["filename"] for x in batch]

    audio_values = nn.utils.rnn.pad_sequence(audio_vals, batch_first=True, padding_value=0.0)
    audio_masks = nn.utils.rnn.pad_sequence(audio_masks, batch_first=True, padding_value=0)

    return {
        "filename": filenames,
        "frames": frames,
        "audio_values": audio_values,
        "audio_mask": audio_masks,
        "label": labels,
    }

def load_model_checkpoint(model, checkpoint_path):
    """
    This loads a saved checkpoint into a model
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model

def compute_metrics_from_lists(
    y_true: list[float],
    y_pred: list[float],
) -> dict[str, float]:
    """
    Compute MAE, RMSE, and MSE from python lists.
    """
    abs_errors = [abs(t - p) for t, p in zip(y_true, y_pred)]
    sq_errors = [(t - p) ** 2 for t, p in zip(y_true, y_pred)]
    mae = sum(abs_errors) / len(abs_errors)
    mse = sum(sq_errors) / len(sq_errors)
    rmse = math.sqrt(mse)
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "loss": float(mse),
    }

@torch.no_grad()
def predict_video_only_split(
    model: nn.Module,
    loader: DataLoader,
    clamp_range: bool = False,
    min_val: float = 1.0,
    max_val: float = 7.0,
) -> tuple[dict[str, float], pd.DataFrame]:
    """
    Runs inference on one split for video only model. Expected batch keys
    are filename, frames and label. Returns dict and dataframe of per-example
    predictions.
    """
    model.eval()

    all_filenames = []
    all_labels = []
    all_preds = []

    for batch in loader:
        frames = batch["frames"].to(device)   # [B, T, C, H, W]
        labels = batch["label"].to(device)    # [B]

        preds = model(frames)                 # [B]

        if clamp_range:
            preds = torch.clamp(preds, min=min_val, max=max_val)

        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())
        all_filenames.extend(batch["filename"])

    metrics = compute_metrics_from_lists(all_labels, all_preds)

    df = pd.DataFrame({
        "filename": all_filenames,
        "true_label": all_labels,
        "pred_label": all_preds,
        "pred_label_rounded": [round(x) for x in all_preds],
        "abs_error": [abs(t - p) for t, p in zip(all_labels, all_preds)],
        "sq_error": [(t - p) ** 2 for t, p in zip(all_labels, all_preds)],
    })

    return metrics, df

@torch.no_grad()
def predict_video_audio_split(
    model: nn.Module,
    loader: DataLoader,
    clamp_range: bool = False,
    min_val: float = 1.0,
    max_val: float = 7.0,
) -> tuple[dict[str, float], pd.DataFrame]:
    """
    Runs inference on one split for video+audio model. Expected batch keys
    are filename, frames, audio_values, audio_mask, and label. Returns dict
    and dataframe of per-example predictions.
    """
    model.eval()
    all_filenames = []
    all_labels = []
    all_preds = []
    for batch in loader:
        frames = batch["frames"].to(device)               # [B, T, C, H, W]
        audio_values = batch["audio_values"].to(device)   # [B, L]
        audio_mask = batch["audio_mask"].to(device)       # [B, L]
        labels = batch["label"].to(device)                # [B]

        preds = model(frames, audio_values, audio_mask)   # [B]
        if clamp_range:
            preds = torch.clamp(preds, min=min_val, max=max_val)
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())
        all_filenames.extend(batch["filename"])

    metrics = compute_metrics_from_lists(all_labels, all_preds)

    df = pd.DataFrame({
        "filename": all_filenames,
        "true_label": all_labels,
        "pred_label": all_preds,
        "pred_label_rounded": [round(x) for x in all_preds],
        "abs_error": [abs(t - p) for t, p in zip(all_labels, all_preds)],
        "sq_error": [(t - p) ** 2 for t, p in zip(all_labels, all_preds)],
    })

    return metrics, df

def save_split_results(
    output_dir: str | Path,
    split_name: str,
    metrics: dict[str, float],
    predictions_df: pd.DataFrame,
) -> None:
    """
    Save one split's metrics + predictions.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / f"{split_name}_metrics.json"
    preds_path = output_dir / f"{split_name}_predictions.csv"

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    predictions_df.to_csv(preds_path, index=False)

def build_frame_transform(train: bool) -> T.Compose:
    """
    Builds transforms for data augmentation
    """
    if train:
        return T.Compose([
            T.ToPILImage(),
            T.Resize((frame_size, frame_size)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize( # ImageNet normalization values
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    return T.Compose([
        T.ToPILImage(),
        T.Resize((frame_size, frame_size)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

def build_video_eval_dataloaders() -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build deterministic train/val/test loaders for evaluation.
    """
    eval_tf = build_frame_transform(train=False)

    train_ds = ValenceVideoRegressionDataset(
        csv_path=train_csv,
        video_root=video_root,
        num_frames=num_frames,
        frame_transform=eval_tf,
        use_audio=False,
        min_valence=min_valence,
        max_valence=max_valence,
    )
    val_ds = ValenceVideoRegressionDataset(
        csv_path=val_csv,
        video_root=video_root,
        num_frames=num_frames,
        frame_transform=eval_tf,
        use_audio=False,
        min_valence=min_valence,
        max_valence=max_valence,
    )
    test_ds = ValenceVideoRegressionDataset(
        csv_path=test_csv,
        video_root=video_root,
        num_frames=num_frames,
        frame_transform=eval_tf,
        use_audio=False,
        min_valence=min_valence,
        max_valence=max_valence,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_video_only,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_video_only,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_video_only,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def build_multimodal_eval_dataloaders() -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build deterministic train/val/test loaders for multimodal evaluation.
    """
    eval_tf = build_frame_transform(train=False)
    wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

    train_ds = ValenceVideoRegressionDataset(
        csv_path=train_csv,
        video_root=video_root,
        num_frames=num_frames,
        frame_transform=eval_tf,
        use_audio=True,
        audio_sr=audio_sr,
        max_audio_seconds=max_audio_seconds,
        wav2vec_processor=wav2vec_processor,
        min_valence=min_valence,
        max_valence=max_valence,
    )
    val_ds = ValenceVideoRegressionDataset(
        csv_path=val_csv,
        video_root=video_root,
        num_frames=num_frames,
        frame_transform=eval_tf,
        use_audio=True,
        audio_sr=audio_sr,
        max_audio_seconds=max_audio_seconds,
        wav2vec_processor=wav2vec_processor,
        min_valence=min_valence,
        max_valence=max_valence,
    )
    test_ds = ValenceVideoRegressionDataset(
        csv_path=test_csv,
        video_root=video_root,
        num_frames=num_frames,
        frame_transform=eval_tf,
        use_audio=True,
        audio_sr=audio_sr,
        max_audio_seconds=max_audio_seconds,
        wav2vec_processor=wav2vec_processor,
        min_valence=min_valence,
        max_valence=max_valence,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_video_audio,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_video_audio,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_video_audio,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader

def run_video_only_inference(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    model_kwargs: Optional[dict] = None,
    use_deterministic_train_eval: bool = True,
) -> dict[str, dict[str, float]]:
    """
    Load a saved video-only checkpoint and run inference on train/val/test.
    """
    model_kwargs = model_kwargs or {}
    if use_deterministic_train_eval:
        train_loader, val_loader, test_loader = build_video_eval_dataloaders()
    else:
        train_loader, val_loader, test_loader = build_video_dataloaders()
    model = VideoOnlyRegressor(**model_kwargs)
    model = load_model_checkpoint(model, checkpoint_path)

    all_metrics = {}

    for split_name, loader in [
        ("train", train_loader),
        ("val", val_loader),
        ("test", test_loader),
    ]:
        metrics, df = predict_video_only_split(
            model=model,
            loader=loader,
            clamp_range=True,
            min_val=1.0,
            max_val=7.0,
        )
        save_split_results(output_dir, split_name, metrics, df)
        all_metrics[split_name] = metrics
        print(f"[video-only][{split_name}] {metrics}")
    with open(Path(output_dir) / "all_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    return all_metrics

def run_video_audio_inference(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    model_kwargs: Optional[dict] = None,
    use_deterministic_train_eval: bool = True,
) -> dict[str, dict[str, float]]:
    """
    Load a saved video+audio checkpoint and run inference on train/val/test.
    """
    model_kwargs = model_kwargs or {}

    if use_deterministic_train_eval:
        train_loader, val_loader, test_loader = build_multimodal_eval_dataloaders()
    else:
        train_loader, val_loader, test_loader = build_multimodal_dataloaders()

    model = VideoAudioRegressor(**model_kwargs)
    model = load_model_checkpoint(model, checkpoint_path)

    all_metrics = {}

    for split_name, loader in [
        ("train", train_loader),
        ("val", val_loader),
        ("test", test_loader),
    ]:
        metrics, df = predict_video_audio_split(
            model=model,
            loader=loader,
            clamp_range=True,
            min_val=1.0,
            max_val=7.0,
        )

        save_split_results(output_dir, split_name, metrics, df)
        all_metrics[split_name] = metrics

        print(f"[video+audio][{split_name}] {metrics}")

    with open(Path(output_dir) / "all_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    return all_metrics

if __name__ == '__main__':
    video_metrics = run_video_only_inference(
        checkpoint_path="artifacts/elderreact_video/best_model.pt",
        output_dir="inference_outputs/video_only",
        model_kwargs={
            "video_emb_dim": video_emb_dim,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "freeze_video_backbone": False,
        },
        use_deterministic_train_eval=True,
    )

    video_audio_metrics = run_video_audio_inference(
        checkpoint_path="artifacts/elderreact_video_audio/best_model.pt",
        output_dir="inference_outputs/video_audio",
        model_kwargs={
            "video_emb_dim": video_emb_dim,
            "audio_emb_dim": audio_emb_dim,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "freeze_video_backbone": False,
            "freeze_audio_backbone": True,
        },
        use_deterministic_train_eval=True,
    )