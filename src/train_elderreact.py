import random
from pathlib import Path
import json

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as tvm
import torchvision.transforms as T
import torchaudio

from transformers import Wav2Vec2Model, Wav2Vec2Processor

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

device = "cpu"

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
    return {
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

    audio_values = nn.utils.rnn.pad_sequence(audio_vals, batch_first=True, padding_value=0.0)
    audio_masks = nn.utils.rnn.pad_sequence(audio_masks, batch_first=True, padding_value=0)

    return {
        "frames": frames,
        "audio_values": audio_values,
        "audio_mask": audio_masks,
        "label": labels,
    }

class VideoEncoder(nn.Module):
    def __init__(self, emb_dim: int = 256, freeze_backbone: bool = False) -> None:
        super().__init__()

        backbone = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
        in_dim = backbone.fc.in_features

        # This removes the ImageNet classification head (doing regression for valence scores)
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.proj = nn.Linear(in_dim, emb_dim)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = frames.shape

        # flattens batch, time so all frames processed at once
        x = frames.view(B * T, C, H, W)
        x = self.backbone(x)  # [B*T, D]
        x = self.proj(x)  # [B*T, emb_dim]

        # restoring time dimension, avg across frames.
        x = x.view(B, T, -1)
        x = x.mean(dim=1)  # [B, emb_dim]

        return x

class VideoOnlyRegressor(nn.Module):
    """
    Regression for video only (no audio), predicts valence score
    """
    def __init__(
        self,
        video_emb_dim: int = 256,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        freeze_video_backbone: bool = False,
    ):
        super().__init__()

        self.video_encoder = VideoEncoder(
            emb_dim=video_emb_dim,
            freeze_backbone=freeze_video_backbone,
        )

        self.regressor = nn.Sequential(
            nn.Linear(video_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, frames: torch.Tensor):
        """
        frames: [B, T, C, H, W]
        prediction: [B]
        """
        v = self.video_encoder(frames)
        pred = self.regressor(v).squeeze(-1)
        return pred


class AudioEncoder(nn.Module):
    """
    Encodes the audio data, based on the pretrained Wav2Vec2
    Feeds the raw waveform into Wav2Vec2, mean-pools over time, and projects to the
    embedding dimension
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        emb_dim: int = 256,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.backbone = Wav2Vec2Model.from_pretrained(model_name)
        hidden_dim = self.backbone.config.hidden_size
        self.proj = nn.Linear(hidden_dim, emb_dim)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ):
        """
        Args:
            input_values:  [B, L]
            attention_mask:[B, L] or None
        Returns:
            audio embedding: [B, emb_dim]
        """
        out = self.backbone(input_values=input_values, attention_mask=attention_mask)
        # last_hidden_state: [B, time_steps, hidden_dim]
        x = out.last_hidden_state.mean(dim=1)
        x = self.proj(x)
        return x


class VideoAudioRegressor(nn.Module):
    """
    Multimodal regressor, combines video embedding and audio embedding by
    concatenating the two embeddings, and it predicts one scalar.
    """
    def __init__(
        self,
        video_emb_dim: int = 256,
        audio_emb_dim: int = 256,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        freeze_video_backbone: bool = False,
        freeze_audio_backbone: bool = True,
    ):
        super().__init__()
        self.video_encoder = VideoEncoder(
            emb_dim=video_emb_dim,
            freeze_backbone=freeze_video_backbone,
        )
        self.audio_encoder = AudioEncoder(
            emb_dim=audio_emb_dim,
            freeze_backbone=freeze_audio_backbone,
        )
        self.regressor = nn.Sequential(
            nn.Linear(video_emb_dim + audio_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        frames: torch.Tensor,
        audio_values: torch.Tensor,
        audio_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            frames:       [B, T, C, H, W]
            audio_values: [B, L]
            audio_mask:   [B, L] or None
        Returns:
            prediction: [B]
        """
        v = self.video_encoder(frames)
        a = self.audio_encoder(audio_values, audio_mask)
        fused = torch.cat([v, a], dim=-1)
        pred = self.regressor(fused).squeeze(-1)
        return pred

def compute_regression_metrics(
    y_true: list[float],
    y_pred: list[float],
) -> dict[str, float]:
    """
    Compute regression metrics, mean abs error and root MSE
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    return {
        "mae": float(mae),
        "rmse": float(rmse),
    }

def train_one_epoch_video(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> dict[str, float]:
    """Train the video-only regressor for one epoch."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    for batch in loader:
        frames = batch["frames"].to(device)
        labels = batch["label"].to(device)  # [B]
        optimizer.zero_grad()
        preds = model(frames)               # [B]
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_regression_metrics(all_labels, all_preds)
    metrics["loss"] = float(avg_loss)
    return metrics

@torch.no_grad()
def eval_video(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    min_valence: float,
    max_valence: float,
) -> dict[str, float]:
    """
    This evalautes the video only regressor, and predictions are restricted
    here to the range [min valence, max_valence]
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    for batch in loader:
        frames = batch["frames"].to(device)
        labels = batch["label"].to(device)
        preds = model(frames)
        loss = criterion(preds, labels)
        total_loss += loss.item() * labels.size(0)
        preds = preds.clamp(min=min_valence, max=max_valence)
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_regression_metrics(all_labels, all_preds)
    metrics["loss"] = float(avg_loss)
    return metrics

def train_one_epoch_multimodal(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> dict[str, float]:
    """Train the video+audio regressor for one epoch."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    for batch in loader:
        frames = batch["frames"].to(device)
        audio_values = batch["audio_values"].to(device)
        audio_mask = batch["audio_mask"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        preds = model(frames, audio_values, audio_mask)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_regression_metrics(all_labels, all_preds)
    metrics["loss"] = float(avg_loss)
    return metrics

@torch.no_grad()
def eval_multimodal(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    min_valence: float,
    max_valence: float,
) -> dict[str, float]:
    """
    This evalautes the multimodal regressor, and predictions are restricted
    here to the range [min valence, max_valence]
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    for batch in loader:
        frames = batch["frames"].to(device)
        audio_values = batch["audio_values"].to(device)
        audio_mask = batch["audio_mask"].to(device)
        labels = batch["label"].to(device)
        preds = model(frames, audio_values, audio_mask)
        loss = criterion(preds, labels)
        total_loss += loss.item() * labels.size(0)
        preds = preds.clamp(min=min_valence, max=max_valence)
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_regression_metrics(all_labels, all_preds)
    metrics["loss"] = float(avg_loss)
    return metrics

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

def build_video_dataloaders() -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test loaders for video-only regression"""
    train_tf = build_frame_transform(train=True)
    eval_tf = build_frame_transform(train=False)
    train_ds = ValenceVideoRegressionDataset(
        csv_path=train_csv,
        video_root=video_root,
        num_frames=num_frames,
        frame_transform=train_tf,
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
        shuffle=True,
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


def build_multimodal_dataloaders() -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test loaders for video+audio regression"""
    train_tf = build_frame_transform(train=True)
    eval_tf = build_frame_transform(train=False)
    wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    train_ds = ValenceVideoRegressionDataset(
        csv_path=train_csv,
        video_root=video_root,
        num_frames=num_frames,
        frame_transform=train_tf,
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
        shuffle=True,
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

def save_run_outputs(save_dir: Path, model: nn.Module, metrics: dict) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    # save model weights
    torch.save(model.state_dict(), save_dir / "best_model.pt")
    # save all metrics together
    with (save_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

def run_video_only() -> nn.Module:
    """
    Train and evaluate the video-only regression model, select best checkpoint
    using validation MAE
    """
    print("Running VIDEO-ONLY regression pipeline")
    train_loader, val_loader, test_loader = build_video_dataloaders()
    model = VideoOnlyRegressor(
        video_emb_dim=video_emb_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        freeze_video_backbone=freeze_video_backbone,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    criterion = nn.SmoothL1Loss()
    best_val_mae = float("inf")
    best_state = None
    train_history = []
    val_history = []
    best_epoch = -1
    for epoch in range(num_epochs):
        train_metrics = train_one_epoch_video(model, train_loader, optimizer, criterion, device)
        val_metrics = eval_video(
            model,
            val_loader,
            criterion,
            device,
            min_valence=min_valence,
            max_valence=max_valence,
        )
        print(f"[Video][Epoch {epoch + 1}/{num_epochs}]")
        print("  Train:", train_metrics)
        print("  Val:  ", val_metrics)

        train_history.append({
            "epoch": epoch + 1,
            "loss": train_metrics["loss"],
            "mae": train_metrics["mae"],
            "rmse": train_metrics["rmse"],
        })
        val_history.append({
            "epoch": epoch + 1,
            "loss": val_metrics["loss"],
            "mae": val_metrics["mae"],
            "rmse": val_metrics["rmse"],
        })

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_epoch = epoch + 1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = eval_video(
        model,
        test_loader,
        criterion,
        device,
        min_valence=min_valence,
        max_valence=max_valence,
    )
    print("[Video] Test:", test_metrics)

    all_metrics = {
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
        "train": train_history,
        "val": val_history,
        "test": test_metrics,
    }

    save_run_outputs(
        save_dir=Path("artifacts/elderreact_video"),
        model=model,
        metrics=all_metrics,
    )

    return model

def run_video_audio() -> nn.Module:
    """
    rain and evaluate the video+audio regression model, select best checkpoint
    using validation MAE
    """
    print("Running VIDEO + AUDIO regression pipeline")
    train_loader, val_loader, test_loader = build_multimodal_dataloaders()
    model = VideoAudioRegressor(
        video_emb_dim=video_emb_dim,
        audio_emb_dim=audio_emb_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        freeze_video_backbone=freeze_video_backbone,
        freeze_audio_backbone=freeze_audio_backbone,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    criterion = nn.SmoothL1Loss()
    best_val_mae = float("inf")
    best_state = None
    train_history = []
    val_history = []
    best_epoch = -1
    for epoch in range(num_epochs):
        train_metrics = train_one_epoch_multimodal(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )
        val_metrics = eval_multimodal(
            model,
            val_loader,
            criterion,
            device,
            min_valence=min_valence,
            max_valence=max_valence,
        )
        print(f"[Video+Audio][Epoch {epoch + 1}/{num_epochs}]")
        print("  Train:", train_metrics)
        print("  Val:  ", val_metrics)
        train_history.append({
            "epoch": epoch + 1,
            "loss": train_metrics["loss"],
            "mae": train_metrics["mae"],
            "rmse": train_metrics["rmse"],
        })
        val_history.append({
            "epoch": epoch + 1,
            "loss": val_metrics["loss"],
            "mae": val_metrics["mae"],
            "rmse": val_metrics["rmse"],
        })
        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_epoch = epoch + 1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = eval_multimodal(
        model,
        test_loader,
        criterion,
        device,
        min_valence=min_valence,
        max_valence=max_valence,
    )
    print("[Video+Audio] Test:", test_metrics)
    all_metrics = {
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
        "train": train_history,
        "val": val_history,
        "test": test_metrics,
    }

    save_run_outputs(
        save_dir=Path("artifacts/elderreact_video_audio"),
        model=model,
        metrics=all_metrics,
    )
    return model

def set_seed(seed):
    """Helper function for reproducibility using set seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    set_seed(seed)
    video_model = run_video_only()
    multimodal_model = run_video_audio()
    _ = video_model, multimodal_model


if __name__ == '__main__':
    main()