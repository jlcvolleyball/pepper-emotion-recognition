"""
train_finetune_fer2013.py

Implements a two-stage training procedure, first training ResNet-50 on the full
FER2013, then loading the stage1 checkpoint and fine-tuning it on the older adult
subset.
"""

import argparse
import numpy as np
import torch
import json
from datetime import datetime
from pathlib import Path

import datasets
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification, ResNetForImageClassification
from transformers import TrainingArguments, Trainer
import evaluate

def construct_train_tensor(processor):
    """
    Turns each image from dataset into tensor format that ResNet expects. Since this
    method is for training, we apply data augmentation
    """
    size = processor.size.get("shortest_edge", 224)
    mean = processor.image_mean
    std = processor.image_std
    return transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")), # ResNet accepts RGB
        # transforms.RandomResizedCrop(size), # picks random crop for data augmentation
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(), # converts image to flat tensors for pytorch
        transforms.Normalize(mean, std) # Resnet weights assumes normalized inputs
    ])

def construct_eval_tensor(processor):
    """
    Turns each image from dataset into tensor format that ResNet expects. Since this
    method is for evaluation (test/validation), we want determinism instead of
    randomness in the train equivalent of this method
    """
    size = processor.size.get("shortest_edge", 224)
    mean = processor.image_mean
    std = processor.image_std
    return transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),  # ResNet accepts RGB
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),  # converts image to flat tensors for pytorch
        transforms.Normalize(mean, std)  # Renet weights assumes normalized inputs
    ])

def prepare_dataset(data_dir, image_processor):
    dataset = datasets.load_dataset("imagefolder", data_dir=str(data_dir))
    labels = dataset["train"].features["label"].names
    label_to_idx = {label: i for i, label in enumerate(labels)}
    idx_to_label = {i: label for i, label in enumerate(labels)}

    train_tensors = construct_train_tensor(image_processor)
    eval_tensors = construct_eval_tensor(image_processor)
    def add_pixel_values(examples, tfm):
        """
        Converts images into tensors
        """
        examples["pixel_values"] = [tfm(img) for img in examples["image"]]
        return examples
    dataset["train"].set_transform(lambda ex: add_pixel_values(ex, train_tensors))
    dataset["validation"].set_transform(lambda ex: add_pixel_values(ex, eval_tensors))
    dataset["test"].set_transform(lambda ex: add_pixel_values(ex, eval_tensors))
    return dataset, labels, label_to_idx, idx_to_label

def make_collate_fn():
    def collate_fn(batch):
        """
        Batch collation
        """
        pixel_values = torch.stack([x["pixel_values"] for x in batch])
        labels_t = torch.tensor([x["label"] for x in batch], dtype=torch.long)
        return {
            "pixel_values": pixel_values,
            "labels": labels_t,
        }
    return collate_fn

def make_metrics_fn():
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        return {
            "accuracy": accuracy.compute(
                predictions=preds,
                references=p.label_ids
            )["accuracy"],
            "f1_macro": f1.compute(
                predictions=preds,
                references=p.label_ids,
                average="macro"
            )["f1"],
        }
    return compute_metrics

def make_training_args(output_dir, epochs, lr, batch):
    return TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        learning_rate=lr,
        num_train_epochs=epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        logging_steps=5,
        weight_decay=1e-4,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        save_total_limit=2,
    )

def train_one_stage(stage, model, dataset, output_dir, epochs, lr, batch):
    print(f"\n{'=' * 20}")
    print(f"Starting {stage}")
    print(f"Output dir: {output_dir}")
    print(f"epochs={epochs}, lr={lr}, batch={batch}")
    print(f"{'=' * 20}\n")
    trainer = Trainer(
        model=model,
        args=make_training_args(output_dir, epochs, lr, batch),
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=make_collate_fn(),
        compute_metrics=make_metrics_fn(),
    )
    trainer.train()
    val_metrics = trainer.evaluate(dataset["validation"])
    test_metrics = trainer.evaluate(dataset["test"])
    trainer.save_model(str(output_dir))

    print(f"\nFinished {stage}")
    print("Validation metrics:", val_metrics)
    print("Test metrics:", test_metrics)

    return trainer, val_metrics, test_metrics

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser()

    parser.add_argument("full_data", type=Path, nargs="?", default=Path("data/processed/fer2013"))
    parser.add_argument("subset_data", type=Path, nargs="?", default=Path("data/processed/fer2013_older_0.6_proc"))
    parser.add_argument("out_dir", type=Path, nargs="?", default=Path("artifacts/two_stage_resnet50"))

    parser.add_argument("epochs1", type=int, nargs="?", default=15)
    parser.add_argument("lr1", type=float, nargs="?", default=1e-4)
    parser.add_argument("batch1", type=int, nargs="?", default=16)

    parser.add_argument("epochs2", type=int, nargs="?", default=20)
    parser.add_argument("lr2", type=float, nargs="?", default=3e-4)
    parser.add_argument("batch2", type=int, nargs="?", default=8)

    args = parser.parse_args()
    model_name = "microsoft/resnet-50"

    out_dir = args.out_dir
    stage1_out = out_dir / "stage1_full_fer2013"
    stage2_out = out_dir / "stage2_full_fer2013"
    out_dir.mkdir(parents=True, exist_ok=True)

    # train stage 1 (on full fer2013 dataset)
    stage1_processor = AutoImageProcessor.from_pretrained(model_name)
    full_dataset, full_labels, full_label_to_idx, full_idx_to_label = prepare_dataset(args.full_data, stage1_processor)
    stage1_model = ResNetForImageClassification.from_pretrained(
        model_name,
        num_labels=len(full_labels),
        label2id=full_label_to_idx,
        id2label=full_idx_to_label,
        ignore_mismatched_sizes=True
    )
    trainer1, stage1_val_metrics, stage1_test_metrics = train_one_stage(
        "Stage 1 (Train on full FER2013)",
        stage1_model,
        full_dataset,
        stage1_out,
        args.epochs1,
        args.lr1,
        args.batch1
    )
    stage1_processor.save_pretrained(str(stage1_out))

    # train stage 2 (on fer2013 subset)
    stage2_processor = AutoImageProcessor.from_pretrained(str(stage1_out))
    subset_dataset, subset_labels, subset_label_to_idx, subset_idx_to_label = prepare_dataset(args.subset_data, stage2_processor)
    stage2_model = ResNetForImageClassification.from_pretrained(
        str(stage1_out),
        num_labels=len(subset_labels),
        label2id=subset_label_to_idx,
        id2label=subset_idx_to_label,
        ignore_mismatched_sizes=True
    )
    # freeze backbone in stage 2
    for param in stage2_model.resnet.parameters():
        param.requires_grad = False
    trainer2, stage2_val_metrics, stage2_test_metrics = train_one_stage(
        "Stage 2 (Train on subset)",
        stage2_model,
        subset_dataset,
        stage2_out,
        args.epochs2,
        args.lr2,
        args.batch2
    )
    stage2_processor.save_pretrained(stage2_out)

    # organize metrics
    run_info = {
        "timestamp": timestamp,
        "full_data": str(args.full_data),
        "subset_data": str(args.subset_data),
        "out_root": str(args.out_dir),
        "stage1": {
            "out_dir": str(stage1_out),
            "epochs": args.epochs1,
            "learning_rate": args.lr1,
            "batch_size": args.batch1,
            "validation_metrics": stage1_val_metrics,
            "test_metrics": stage1_test_metrics,
        },
        "stage2": {
            "out_dir": str(stage2_out),
            "epochs": args.epochs2,
            "learning_rate": args.lr2,
            "batch_size": args.batch2,
            "validation_metrics": stage2_val_metrics,
            "test_metrics": stage2_test_metrics,
        }
    }
    metrics_path = out_dir/f"two_stage_metrics_{timestamp}.json"
    with open(metrics_path, "w") as f:
        json.dump(run_info, f, indent=4)
    print(f"\nTwo-stage training was complete, metrics saved to {metrics_path}")

if __name__ == '__main__':
    main()
