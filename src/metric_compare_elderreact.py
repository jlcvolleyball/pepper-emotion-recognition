import json
from pathlib import Path
import wandb

def main():
    video_metrics_path = Path("artifacts/elderreact_video/metrics.json")
    multimodal_metrics_path = Path("artifacts/elderreact_video_audio/metrics.json")

    with video_metrics_path.open("r", encoding="utf-8") as f:
        video_metrics = json.load(f)

    with multimodal_metrics_path.open("r", encoding="utf-8") as f:
        multimodal_metrics = json.load(f)

    wandb.init(
        project="elderreact-emotion-recognition",
        name="model_comparison_with_train",
        group="elderreact_comparison",
        tags=["comparison", "train-vs-val", "regression"],
        reinit=True
    )

    val_rows = []
    for v_epoch, m_epoch in zip(video_metrics["val"], multimodal_metrics["val"]):
        val_rows.append([
            v_epoch["epoch"],
            v_epoch["mae"], m_epoch["mae"],
            v_epoch["rmse"], m_epoch["rmse"],
            v_epoch["loss"], m_epoch["loss"],
        ])
    train_rows = []
    for v_epoch, m_epoch in zip(video_metrics["train"], multimodal_metrics["train"]):
        train_rows.append([
            v_epoch["epoch"],
            v_epoch["mae"], m_epoch["mae"],
            v_epoch["rmse"], m_epoch["rmse"],
            v_epoch["loss"], m_epoch["loss"],
        ])

    wandb.log({
        "compare/train_mae": wandb.plot.line_series(
            xs=[row[0] for row in train_rows],
            ys=[
                [row[1] for row in train_rows],
                [row[2] for row in train_rows],
            ],
            keys=["video_only", "video_audio"],
            title="Training MAE Comparison",
            xname="Epoch",
        ),
        "compare/train_loss": wandb.plot.line_series(
            xs=[row[0] for row in train_rows],
            ys=[
                [row[5] for row in train_rows],
                [row[6] for row in train_rows],
            ],
            keys=["video_only", "video_audio"],
            title="Training Loss Comparison",
            xname="Epoch",
        ),
        "compare/train_rmse": wandb.plot.line_series(
            xs=[row[0] for row in train_rows],
            ys=[
                [row[3] for row in train_rows],
                [row[4] for row in train_rows],
            ],
            keys=["video_only", "video_audio"],
            title="Training RMSE Comparison",
            xname="Epoch",
        ),
        "compare/val_mae": wandb.plot.line_series(
            xs=[row[0] for row in val_rows],
            ys=[
                [row[1] for row in val_rows],
                [row[2] for row in val_rows],
            ],
            keys=["video_only", "video_audio"],
            title="Validation MAE Comparison",
            xname="Epoch",
        ),
        "compare/val_loss": wandb.plot.line_series(
            xs=[row[0] for row in val_rows],
            ys=[
                [row[5] for row in val_rows],
                [row[6] for row in val_rows],
            ],
            keys=["video_only", "video_audio"],
            title="Validation Loss Comparison",
            xname="Epoch",
        ),
        "compare/val_rmse": wandb.plot.line_series(
            xs=[row[0] for row in val_rows],
            ys=[
                [row[3] for row in val_rows],
                [row[4] for row in val_rows],
            ],
            keys=["video_only", "video_audio"],
            title="Validation RMSE Comparison",
            xname="Epoch",
        ),
    })

    wandb.log({
        "compare/video_only_train_vs_val_mae": wandb.plot.line_series(
            xs=[row["epoch"] for row in video_metrics["train"]],
            ys=[
                [row["mae"] for row in video_metrics["train"]],
                [row["mae"] for row in video_metrics["val"]],
            ],
            keys=["train", "val"],
            title="Video-Only Train vs Val MAE",
            xname="Epoch",
        )
    })
    wandb.log({
        "compare/video_audio_train_vs_val_mae": wandb.plot.line_series(
            xs=[row["epoch"] for row in multimodal_metrics["train"]],
            ys=[
                [row["mae"] for row in multimodal_metrics["train"]],
                [row["mae"] for row in multimodal_metrics["val"]],
            ],
            keys=["train", "val"],
            title="Video+Audio Train vs Val MAE",
            xname="Epoch",
        )
    })

    # plot all four lines (both models train and val)
    epochs = [row["epoch"] for row in video_metrics["train"]]
    video_train_mae = [row["mae"] for row in video_metrics["train"]]
    video_val_mae = [row["mae"] for row in video_metrics["val"]]
    video_train_loss = [row["loss"] for row in video_metrics["train"]]
    video_val_loss = [row["loss"] for row in video_metrics["val"]]
    video_train_rmse = [row["rmse"] for row in video_metrics["train"]]
    video_val_rmse = [row["rmse"] for row in video_metrics["val"]]
    multimodal_train_mae = [row["mae"] for row in multimodal_metrics["train"]]
    multimodal_val_mae = [row["mae"] for row in multimodal_metrics["val"]]
    multimodal_train_loss = [row["loss"] for row in multimodal_metrics["train"]]
    multimodal_val_loss = [row["loss"] for row in multimodal_metrics["val"]]
    multimodal_train_rmse = [row["rmse"] for row in multimodal_metrics["train"]]
    multimodal_val_rmse = [row["rmse"] for row in multimodal_metrics["val"]]
    wandb.log({
        "compare/all_models_train_vs_val_mae": wandb.plot.line_series(
            xs=epochs,
            ys=[
                video_train_mae,
                video_val_mae,
                multimodal_train_mae,
                multimodal_val_mae,
            ],
            keys=[
                "video_only_train",
                "video_only_val",
                "video_audio_train",
                "video_audio_val",
            ],
            title="Train vs Val MAE: Video-Only vs Video+Audio",
            xname="Epoch",
        ),
        "compare/all_models_train_vs_val_loss": wandb.plot.line_series(
            xs=epochs,
            ys=[
                video_train_loss,
                video_val_loss,
                multimodal_train_loss,
                multimodal_val_loss,
            ],
            keys=[
                "video_only_train",
                "video_only_val",
                "video_audio_train",
                "video_audio_val",
            ],
            title="Train vs Val Loss: Video-Only vs Video+Audio",
            xname="Epoch",
        ),
        "compare/all_models_train_vs_val_rmse": wandb.plot.line_series(
            xs=epochs,
            ys=[
                video_train_rmse,
                video_val_rmse,
                multimodal_train_rmse,
                multimodal_val_rmse,
            ],
            keys=[
                "video_only_train",
                "video_only_val",
                "video_audio_train",
                "video_audio_val",
            ],
            title="Train vs Val RMSE: Video-Only vs Video+Audio",
            xname="Epoch",
        ),
    })

    wandb.finish()

if __name__ == '__main__':
    main()
