"""
train_fer2013.py

This includes all the main training functionality necessary when running
fer2013_default_pipeline.py. Includes augmentation, freezing layers, and
weight decay. For more information on the pipeline, visit
fer2013_default_pipeline.py.
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

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str, nargs="?", default="data/processed/fer2013_older_0.6_proc")
    parser.add_argument("out", type=str, nargs="?",default=f"artifacts/resnet50-fer2013_{timestamp}")
    parser.add_argument("epochs", type=int, nargs="?", default=30)
    parser.add_argument("lr", type=float, nargs="?", default=1e-3)
    parser.add_argument("batch", type=int, nargs="?", default=8)
    args = parser.parse_args()
    model_name = "microsoft/resnet-50"

    dataset = datasets.load_dataset("imagefolder", data_dir=args.data)
    print(dataset.keys())
    labels = dataset["train"].features["label"].names
    label_to_idx = {label: i for i, label in enumerate(labels)}
    idx_to_label = {i: label for i, label in enumerate(labels)}

    image_processor = AutoImageProcessor.from_pretrained(model_name)
    model = ResNetForImageClassification.from_pretrained(
        model_name,
        num_labels = len(labels),
        label2id=label_to_idx,
        id2label=idx_to_label,
        ignore_mismatched_sizes=True
    )

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

    for param in model.resnet.parameters():
        param.requires_grad = False
    train_args = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
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

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    test_metrics = trainer.evaluate(dataset["test"])
    trainer.save_model(args.out)
    image_processor.save_pretrained(args.out)

    # save metrics
    run_info = {
        "timestamp": timestamp,
        # "model": args.model,
        "data_dir": args.data,
        "output_dir": args.out,
        "hyperparameters": {
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "batch_size": args.batch,
        },
        "metrics": test_metrics,
    }
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = artifacts_dir / f"metrics_{timestamp}.json"

    with open(metrics_path, "w") as f:
        json.dump(run_info, f, indent=4)

    print(run_info)
    print(f"Saved metrics to: {metrics_path}")

if __name__ == '__main__':
    main()
