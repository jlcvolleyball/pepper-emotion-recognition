"""
train_multimodal_sweep.py

Runs a wandb hyperparameter sweep for Models 2 and 3
"""

import random
import numpy as np
from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from transformers import Wav2Vec2Processor

from train_elderreact import (
    VideoAudioRegressor,
    ValenceVideoRegressionDataset,
    build_frame_transform,
    collate_video_audio,
    train_one_epoch_multimodal,
    eval_multimodal,
    train_csv,
    val_csv,
    test_csv,
    video_root,
    num_frames,
    frame_size,
    audio_sr,
    max_audio_seconds,
    num_workers,
    min_valence,
    max_valence,
    seed
)
device = "cpu"

# SWEEP_CONFIG = {
#     "name": "elderreact_multimodal_sweep",
#     "method": "random",
#     "metric": {
#         "name": "val/mae",
#         "goal": "minimize",
#     },
#     "parameters": {
#         "lr": {"values": [1e-4, 3e-5, 1e-5]},
#         "dropout": {"values": [0.3, 0.5]},
#         "hidden_dim": {"values": [64, 128, 256]},
#         "weight_decay": {"values": [1e-4, 5e-4, 1e-3]},
#         "freeze_video_backbone": {"values": [True, False]},
#         "freeze_audio_backbone": {"values": [True]},
#         # "video_emb_dim": {"values": [128, 256]},
#         # "audio_emb_dim": {"values": [128, 256]},
#         # "batch_size": {"values": [2, 4]},
#         # "num_epochs": {"values": [6]},
#     },
# }

SWEEP_MODEL_DIR = Path("artifacts/sweep_models")
SWEEP_MODEL_DIR.mkdir(parents=True, exist_ok=True)

SWEEP_CONFIG = {
    "name": "elderreact_multimodal_sweep_small",
    "method": "random",
    "metric": {
        "name": "val/mae",
        "goal": "minimize",
    },
    "parameters": {
        "lr": {"values": [1e-4, 3e-5]},
        "dropout": {"values": [0.3, 0.5]},
        "hidden_dim": {"values": [64, 128]},
        "weight_decay": {"values": [1e-4, 1e-3]},
    },
}
video_emb_dim = 128
audio_emb_dim = 128
batch_size = 4
num_epochs = 8
freeze_video_backbone = True
freeze_audio_backbone = True

def set_seed(seed):
    """Helper function for reproducibility using set seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_multimodal_dataloaders_for_sweep(
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
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

def train_multimodal_sweep_run() -> None:
    with wandb.init(
        project="elderreact-emotion-recognition",
        group="elderreact_sweeps_multimodal",
        tags=["sweep", "video-audio", "regression"],
        reinit=True,
    ):
        config = wandb.config

        set_seed(seed)

        train_loader, val_loader, test_loader = build_multimodal_dataloaders_for_sweep(
            batch_size=batch_size
        )

        model = VideoAudioRegressor(
            video_emb_dim=video_emb_dim,
            audio_emb_dim=audio_emb_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
            freeze_video_backbone=freeze_video_backbone,
            freeze_audio_backbone=freeze_audio_backbone,
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        criterion = nn.SmoothL1Loss()

        best_val_mae = float("inf")
        best_epoch = -1
        best_state = None

        for epoch in range(num_epochs):
            train_metrics = train_one_epoch_multimodal(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
            )

            val_metrics = eval_multimodal(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                min_valence=min_valence,
                max_valence=max_valence,
            )

            if val_metrics["mae"] < best_val_mae:
                best_val_mae = val_metrics["mae"]
                best_epoch = epoch + 1
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}

            wandb.log(
                {
                    "train/loss": train_metrics["loss"],
                    "train/mae": train_metrics["mae"],
                    "train/rmse": train_metrics["rmse"],
                    "val/loss": val_metrics["loss"],
                    "val/mae": val_metrics["mae"],
                    "val/rmse": val_metrics["rmse"],
                    "best_val_mae_so_far": best_val_mae,
                },
                step=epoch + 1,
            )

            print(
                f"[Sweep Run][Epoch {epoch + 1}/{num_epochs}] "
                f"train_mae={train_metrics['mae']:.4f} "
                f"val_mae={val_metrics['mae']:.4f}"
            )

        if best_state is not None:
            model.load_state_dict(best_state)

        test_metrics = eval_multimodal(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            min_valence=min_valence,
            max_valence=max_valence,
        )

        run_model_path = SWEEP_MODEL_DIR / f"run_{wandb.run.id}_best.pt"
        torch.save(best_state, run_model_path)
        run_metadata_path = SWEEP_MODEL_DIR / f"run_{wandb.run.id}_summary.json"
        run_summary = {
            "run_id": wandb.run.id,
            "best_epoch": best_epoch,
            "best_val_mae": best_val_mae,
            "test_loss": test_metrics["loss"],
            "test_mae": test_metrics["mae"],
            "test_rmse": test_metrics["rmse"],
            "config": dict(config),
        }
        with run_metadata_path.open("w", encoding="utf-8") as f:
            json.dump(run_summary, f, indent=2)

        model_artifact = wandb.Artifact(
            name=f"best_model_{wandb.run.id}",
            type="model",
            description="Best checkpoint for this sweep run based on validation MAE",
        )
        model_artifact.add_file(str(run_model_path))
        model_artifact.add_file(str(run_metadata_path))
        wandb.log_artifact(model_artifact)

        wandb.log(
            {
                "best_epoch": best_epoch,
                "best_val_mae": best_val_mae,
                "test/loss": test_metrics["loss"],
                "test/mae": test_metrics["mae"],
                "test/rmse": test_metrics["rmse"],
                "saved_model_path": str(run_model_path),
            }
        )

        print(f"[Sweep Run Complete] run_id={wandb.run.id}")
        print(f"  best_epoch={best_epoch}")
        print(f"  best_val_mae={best_val_mae:.4f}")
        print(f"  test_mae={test_metrics['mae']:.4f}")
        print(f"  model_saved_to={run_model_path}")


def main() -> None:
    sweep_id = wandb.sweep(
        sweep=SWEEP_CONFIG,
        project="elderreact-emotion-recognition",
    )
    print("Created sweep:", sweep_id)

    wandb.agent(
        sweep_id,
        function=train_multimodal_sweep_run,
        count=4, # 12
    )

if __name__ == "__main__":
    main()
