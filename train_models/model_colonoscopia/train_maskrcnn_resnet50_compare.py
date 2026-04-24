"""
Entrena un modelo alternativo para Fase 2 basado en Mask R-CNN + ResNet50-FPN.

Objetivo:
- Aprovechar las mascaras exactas del dataset de colonoscopia.
- Obtener segmentacion del polipo, no solo bounding boxes.
- Comparar el modelo nuevo contra el YOLO actual de la app.

Salidas principales:
- train_models/model_colonoscopia/maskrcnn_resnet50_compare/
- models/colonoscopy_maskrcnn_resnet50.pth
- models/colonoscopy_maskrcnn_resnet50_meta.json

Uso recomendado:
    .\\.venv\\Scripts\\python.exe .\\train_models\\model_colonoscopia\\train_maskrcnn_resnet50_compare.py

Opciones utiles:
    --epochs 30
    --batch-size 2
    --lr 1e-4
    --score-thr 0.45
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops import box_iou
from torchvision.transforms import functional as TF


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "data" / "dataset_yolo"
OUTPUT_DIR = PROJECT_ROOT / "train_models" / "model_colonoscopia" / "maskrcnn_resnet50_compare"
OUTPUT_MODEL = PROJECT_ROOT / "models" / "colonoscopy_maskrcnn_resnet50.pth"
OUTPUT_META = PROJECT_ROOT / "models" / "colonoscopy_maskrcnn_resnet50_meta.json"
YOLO_COMPARE_MODEL = PROJECT_ROOT / "models" / "colonoscopy.pt"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CLASS_NAMES = ["background", "polyp"]
CLASS_ID = 1
RANDOM_SEED = 42


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 0
    score_thr: float = 0.45
    iou_thr: float = 0.50
    print_every: int = 10
    min_size: int = 512
    max_size: int = 1024
    early_stopping_patience: int = 8
    use_amp: bool = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train Mask R-CNN ResNet50 and compare against YOLO")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--score-thr", type=float, default=0.45)
    parser.add_argument("--iou-thr", type=float, default=0.50)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--min-size", type=int, default=512)
    parser.add_argument("--max-size", type=int, default=1024)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    args = parser.parse_args()
    return TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        score_thr=args.score_thr,
        iou_thr=args.iou_thr,
        print_every=args.print_every,
        min_size=args.min_size,
        max_size=args.max_size,
        early_stopping_patience=args.early_stopping_patience,
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def find_mask_path(mask_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = mask_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def mask_to_instances(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    binary = (mask > 127).astype(np.uint8)
    if binary.sum() == 0:
        return np.zeros((0, *binary.shape), dtype=np.uint8), np.zeros((0, 4), dtype=np.float32)

    num_labels, labels = cv2.connectedComponents(binary)
    masks: list[np.ndarray] = []
    boxes: list[list[float]] = []
    for label_idx in range(1, num_labels):
        instance = (labels == label_idx).astype(np.uint8)
        ys, xs = np.where(instance > 0)
        if ys.size == 0 or xs.size == 0:
            continue
        x1, x2 = float(xs.min()), float(xs.max() + 1)
        y1, y2 = float(ys.min()), float(ys.max() + 1)
        masks.append(instance)
        boxes.append([x1, y1, x2, y2])

    if not masks:
        return np.zeros((0, *binary.shape), dtype=np.uint8), np.zeros((0, 4), dtype=np.float32)
    return np.stack(masks, axis=0), np.array(boxes, dtype=np.float32)


class ColonMaskDataset(Dataset):
    def __init__(self, split: str, train: bool = False) -> None:
        self.split = split
        self.train = train
        self.images_dir = DATASET_DIR / "images" / split
        self.masks_dir = DATASET_DIR / "masks" / split
        if not self.images_dir.exists():
            raise FileNotFoundError(f"No existe el split de imagenes: {self.images_dir}")
        self.image_paths = sorted([p for p in self.images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
        if not self.image_paths:
            raise RuntimeError(f"No hay imagenes en {self.images_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def _augment(
        self,
        image: torch.Tensor,
        boxes: torch.Tensor,
        masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.train:
            return image, boxes, masks

        if random.random() < 0.5:
            image = torch.flip(image, dims=[2])
            if boxes.numel() > 0:
                x1 = image.shape[2] - boxes[:, 2]
                x2 = image.shape[2] - boxes[:, 0]
                boxes = boxes.clone()
                boxes[:, 0] = x1
                boxes[:, 2] = x2
            if masks.numel() > 0:
                masks = torch.flip(masks, dims=[2])

        if random.random() < 0.15:
            pil = TF.to_pil_image(image)
            pil = ImageEnhance.Contrast(pil).enhance(random.uniform(0.85, 1.15))
            pil = ImageEnhance.Color(pil).enhance(random.uniform(0.85, 1.15))
            image = TF.to_tensor(pil)

        return image, boxes, masks

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image_path = self.image_paths[index]
        mask_path = find_mask_path(self.masks_dir, image_path.stem)

        image_pil = Image.open(image_path).convert("RGB")
        image_tensor = TF.to_tensor(image_pil)

        if mask_path is not None:
            mask_np = np.array(Image.open(mask_path).convert("L"))
            masks_np, boxes_np = mask_to_instances(mask_np)
        else:
            h, w = image_tensor.shape[1:]
            masks_np = np.zeros((0, h, w), dtype=np.uint8)
            boxes_np = np.zeros((0, 4), dtype=np.float32)

        boxes = torch.as_tensor(boxes_np, dtype=torch.float32)
        masks = torch.as_tensor(masks_np, dtype=torch.uint8)
        labels = torch.full((boxes.shape[0],), CLASS_ID, dtype=torch.int64)
        area = (
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            if boxes.shape[0] > 0
            else torch.zeros((0,), dtype=torch.float32)
        )
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        image_tensor, boxes, masks = self._augment(image_tensor, boxes, masks)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([index]),
            "area": area,
            "iscrowd": iscrowd,
            "image_path": str(image_path),
        }
        return image_tensor, target


def collate_fn(batch: list[tuple[torch.Tensor, dict[str, Any]]]) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
    images, targets = zip(*batch)
    return list(images), list(targets)


def build_model(config: TrainConfig) -> nn.Module:
    model = maskrcnn_resnet50_fpn_v2(
        weights="DEFAULT",
        min_size=config.min_size,
        max_size=config.max_size,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(CLASS_NAMES))

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        len(CLASS_NAMES),
    )
    return model


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = float(np.logical_and(mask_a, mask_b).sum())
    union = float(np.logical_or(mask_a, mask_b).sum())
    if union <= 0:
        return 0.0
    return inter / union


def dice_score(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = float(np.logical_and(mask_a, mask_b).sum())
    denom = float(mask_a.sum() + mask_b.sum())
    if denom <= 0:
        return 0.0
    return (2.0 * inter) / denom


def greedy_match(iou_matrix: np.ndarray, threshold: float) -> list[tuple[int, int, float]]:
    if iou_matrix.size == 0:
        return []
    matches: list[tuple[int, int, float]] = []
    used_pred: set[int] = set()
    used_gt: set[int] = set()

    flat = []
    for pred_idx in range(iou_matrix.shape[0]):
        for gt_idx in range(iou_matrix.shape[1]):
            flat.append((pred_idx, gt_idx, float(iou_matrix[pred_idx, gt_idx])))
    flat.sort(key=lambda item: item[2], reverse=True)

    for pred_idx, gt_idx, score in flat:
        if score < threshold:
            break
        if pred_idx in used_pred or gt_idx in used_gt:
            continue
        used_pred.add(pred_idx)
        used_gt.add(gt_idx)
        matches.append((pred_idx, gt_idx, score))
    return matches


def evaluate_maskrcnn(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    score_thr: float,
    iou_thr: float,
) -> dict[str, float]:
    model.eval()
    tp = fp = fn = 0
    box_ious: list[float] = []
    mask_ious: list[float] = []
    dice_values: list[float] = []

    with torch.no_grad():
        for images, targets in loader:
            outputs = model([img.to(device) for img in images])
            for output, target in zip(outputs, targets):
                gt_boxes = target["boxes"].cpu()
                gt_masks = target["masks"].cpu().numpy().astype(bool)

                scores = output["scores"].detach().cpu().numpy()
                keep = scores >= score_thr
                pred_boxes = output["boxes"].detach().cpu()[keep]
                pred_masks_raw = output["masks"].detach().cpu()[keep]
                pred_masks = (pred_masks_raw[:, 0].numpy() >= 0.5) if pred_masks_raw.numel() > 0 else np.zeros((0, *gt_masks.shape[1:]), dtype=bool)

                if pred_masks.shape[0] == 0 and gt_masks.shape[0] == 0:
                    continue

                iou_matrix = np.zeros((pred_masks.shape[0], gt_masks.shape[0]), dtype=np.float32)
                for pred_idx in range(pred_masks.shape[0]):
                    for gt_idx in range(gt_masks.shape[0]):
                        iou_matrix[pred_idx, gt_idx] = mask_iou(pred_masks[pred_idx], gt_masks[gt_idx])

                matches = greedy_match(iou_matrix, iou_thr)
                matched_pred = {pred_idx for pred_idx, _, _ in matches}
                matched_gt = {gt_idx for _, gt_idx, _ in matches}

                tp += len(matches)
                fp += max(pred_masks.shape[0] - len(matches), 0)
                fn += max(gt_masks.shape[0] - len(matches), 0)

                if pred_boxes.shape[0] > 0 and gt_boxes.shape[0] > 0 and matches:
                    box_iou_matrix = box_iou(pred_boxes, gt_boxes).numpy()
                    for pred_idx, gt_idx, mask_iou_value in matches:
                        box_ious.append(float(box_iou_matrix[pred_idx, gt_idx]))
                        mask_ious.append(float(mask_iou_value))
                        dice_values.append(dice_score(pred_masks[pred_idx], gt_masks[gt_idx]))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-8)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_box_iou": float(np.mean(box_ious)) if box_ious else 0.0,
        "mean_mask_iou": float(np.mean(mask_ious)) if mask_ious else 0.0,
        "mean_dice": float(np.mean(dice_values)) if dice_values else 0.0,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


def evaluate_yolo_against_masks(
    dataset: ColonMaskDataset,
    score_thr: float,
    iou_thr: float,
) -> dict[str, float]:
    if not YOLO_COMPARE_MODEL.exists():
        return {
            "available": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "mean_box_iou": 0.0,
            "tp": 0.0,
            "fp": 0.0,
            "fn": 0.0,
        }

    try:
        from ultralytics import YOLO
    except ImportError:
        return {
            "available": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "mean_box_iou": 0.0,
            "tp": 0.0,
            "fp": 0.0,
            "fn": 0.0,
        }

    model = YOLO(str(YOLO_COMPARE_MODEL))
    tp = fp = fn = 0
    ious: list[float] = []

    for image_path in dataset.image_paths:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            continue
        mask_path = find_mask_path(dataset.masks_dir, image_path.stem)
        if mask_path is not None:
            mask_np = np.array(Image.open(mask_path).convert("L"))
            _, gt_boxes_np = mask_to_instances(mask_np)
            gt_boxes = torch.as_tensor(gt_boxes_np, dtype=torch.float32)
        else:
            gt_boxes = torch.zeros((0, 4), dtype=torch.float32)

        result = model.predict(source=image_bgr, verbose=False, conf=score_thr)[0]
        if getattr(result, "boxes", None) is not None and result.boxes is not None:
            pred_boxes = result.boxes.xyxy.detach().cpu()
        else:
            pred_boxes = torch.zeros((0, 4), dtype=torch.float32)

        if pred_boxes.shape[0] == 0 and gt_boxes.shape[0] == 0:
            continue

        if pred_boxes.shape[0] == 0:
            fn += int(gt_boxes.shape[0])
            continue
        if gt_boxes.shape[0] == 0:
            fp += int(pred_boxes.shape[0])
            continue

        iou_matrix = box_iou(pred_boxes, gt_boxes).numpy()
        matches = greedy_match(iou_matrix, iou_thr)
        tp += len(matches)
        fp += max(pred_boxes.shape[0] - len(matches), 0)
        fn += max(gt_boxes.shape[0] - len(matches), 0)
        for pred_idx, gt_idx, score in matches:
            ious.append(float(score))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-8)
    return {
        "available": 1.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_box_iou": float(np.mean(ious)) if ious else 0.0,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    epoch: int,
    config: TrainConfig,
) -> float:
    model.train()
    running_loss = 0.0
    step_count = 0

    for step, (images, targets) in enumerate(loader, start=1):
        images = [img.to(device) for img in images]
        clean_targets = []
        for target in targets:
            clean_targets.append(
                {
                    "boxes": target["boxes"].to(device),
                    "labels": target["labels"].to(device),
                    "masks": target["masks"].to(device),
                    "image_id": target["image_id"].to(device),
                    "area": target["area"].to(device),
                    "iscrowd": target["iscrowd"].to(device),
                }
            )

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=bool(scaler is not None)):
            loss_dict = model(images, clean_targets)
            losses = sum(loss for loss in loss_dict.values())

        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        running_loss += float(losses.item())
        step_count += 1

        if step % config.print_every == 0:
            print(f"    epoch {epoch:02d} step {step:04d} loss={float(losses.item()):.4f}")

    return running_loss / max(step_count, 1)


def save_history_plots(history_df: pd.DataFrame, output_dir: Path) -> None:
    if history_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="train_loss", color="#45b0ff", linewidth=2)
    axes[0].set_title("Loss de entrenamiento")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)

    axes[1].plot(history_df["epoch"], history_df["val_f1"], label="val_f1", color="#4bd18a", linewidth=2)
    axes[1].plot(history_df["epoch"], history_df["val_precision"], label="val_precision", color="#f4bf4f", linewidth=2)
    axes[1].plot(history_df["epoch"], history_df["val_recall"], label="val_recall", color="#ff6b7a", linewidth=2)
    axes[1].plot(history_df["epoch"], history_df["val_dice"], label="val_dice", color="#b596ff", linewidth=2)
    axes[1].set_title("Metricas de validacion")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_comparison_plot(maskrcnn_metrics: dict[str, float], yolo_metrics: dict[str, float], output_dir: Path) -> None:
    labels = ["Precision", "Recall", "F1", "Box IoU"]
    maskrcnn_values = [
        maskrcnn_metrics.get("precision", 0.0),
        maskrcnn_metrics.get("recall", 0.0),
        maskrcnn_metrics.get("f1", 0.0),
        maskrcnn_metrics.get("mean_box_iou", 0.0),
    ]
    yolo_values = [
        yolo_metrics.get("precision", 0.0),
        yolo_metrics.get("recall", 0.0),
        yolo_metrics.get("f1", 0.0),
        yolo_metrics.get("mean_box_iou", 0.0),
    ]

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, maskrcnn_values, width, label="Mask R-CNN ResNet50", color="#4bd18a")
    ax.bar(x + width / 2, yolo_values, width, label="YOLO actual", color="#45b0ff")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Comparacion sobre test")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "comparison_vs_yolo.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def draw_prediction_gallery(
    model: nn.Module,
    dataset: ColonMaskDataset,
    device: torch.device,
    output_dir: Path,
    score_thr: float,
    max_items: int = 6,
) -> None:
    model.eval()
    ensure_dir(output_dir)

    samples = dataset.image_paths[:max_items]
    with torch.no_grad():
        for idx, image_path in enumerate(samples, start=1):
            image_rgb = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
            image_tensor = TF.to_tensor(Image.open(image_path).convert("RGB")).to(device)
            output = model([image_tensor])[0]
            keep = output["scores"].detach().cpu().numpy() >= score_thr
            pred_boxes = output["boxes"].detach().cpu().numpy()[keep]
            pred_masks = output["masks"].detach().cpu().numpy()[keep, 0] >= 0.5 if output["masks"].numel() > 0 else np.zeros((0, image_rgb.shape[0], image_rgb.shape[1]), dtype=bool)

            canvas = image_rgb.copy()
            for mask in pred_masks:
                color = np.array([255, 90, 90], dtype=np.uint8)
                canvas[mask] = (0.55 * canvas[mask] + 0.45 * color).astype(np.uint8)
            for box in pred_boxes:
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (80, 220, 120), 2)

            out_path = output_dir / f"sample_{idx:02d}_{image_path.stem}.jpg"
            cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def save_artifacts_summary(
    config: TrainConfig,
    history_df: pd.DataFrame,
    best_metrics: dict[str, float],
    test_metrics: dict[str, float],
    yolo_metrics: dict[str, float],
    best_model_path: Path,
) -> None:
    summary = {
        "model_name": "maskrcnn_resnet50_fpn_v2",
        "classes": CLASS_NAMES,
        "dataset_dir": str(DATASET_DIR),
        "weights_path": str(best_model_path),
        "exported_model_path": str(OUTPUT_MODEL),
        "config": vars(config),
        "best_val_metrics": best_metrics,
        "test_metrics": test_metrics,
        "yolo_test_metrics": yolo_metrics,
        "history_rows": len(history_df),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    OUTPUT_META.write_text(
        json.dumps(
            {
                "architecture": "maskrcnn_resnet50_fpn_v2",
                "classes": CLASS_NAMES,
                "task": "instance-segmentation",
                "score_threshold_default": config.score_thr,
                "iou_threshold_eval": config.iou_thr,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def validate_dataset() -> None:
    required = [
        DATASET_DIR / "images" / "train",
        DATASET_DIR / "images" / "val",
        DATASET_DIR / "images" / "test",
        DATASET_DIR / "masks" / "train",
        DATASET_DIR / "masks" / "val",
        DATASET_DIR / "masks" / "test",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Faltan rutas del dataset:\n- " + "\n- ".join(missing) + "\nEjecuta antes prepare_dataset.py"
        )


def main() -> None:
    set_seed(RANDOM_SEED)
    config = parse_args()
    ensure_dir(OUTPUT_DIR)
    validate_dataset()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print()
    print("=" * 72)
    print("ENTRENAMIENTO ALTERNATIVO FASE 2 - MASK R-CNN + RESNET50-FPN")
    print("=" * 72)
    print(f"Dataset:       {DATASET_DIR}")
    print(f"Salida:        {OUTPUT_DIR}")
    print(f"Modelo final:  {OUTPUT_MODEL}")
    print(f"Dispositivo:   {device}")
    print(f"YOLO compare:  {YOLO_COMPARE_MODEL if YOLO_COMPARE_MODEL.exists() else 'no disponible'}")
    print()

    train_ds = ColonMaskDataset("train", train=True)
    val_ds = ColonMaskDataset("val", train=False)
    test_ds = ColonMaskDataset("test", train=False)
    print(f"Train: {len(train_ds)} imagenes")
    print(f"Val:   {len(val_ds)} imagenes")
    print(f"Test:  {len(test_ds)} imagenes")
    print()

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )
    scaler = torch.amp.GradScaler("cuda") if (config.use_amp and device.type == "cuda") else None

    best_score = -math.inf
    best_metrics: dict[str, float] = {}
    best_checkpoint = OUTPUT_DIR / "best_maskrcnn_resnet50.pth"
    last_checkpoint = OUTPUT_DIR / "last_maskrcnn_resnet50.pth"
    history_rows: list[dict[str, float]] = []
    bad_epochs = 0

    start = time.time()
    for epoch in range(1, config.epochs + 1):
        print(f"[Epoch {epoch:02d}/{config.epochs}]")
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, config)
        val_metrics = evaluate_maskrcnn(model, val_loader, device, config.score_thr, config.iou_thr)
        composite_score = (0.60 * val_metrics["f1"]) + (0.40 * val_metrics["mean_dice"])
        scheduler.step(composite_score)

        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "val_precision": float(val_metrics["precision"]),
            "val_recall": float(val_metrics["recall"]),
            "val_f1": float(val_metrics["f1"]),
            "val_box_iou": float(val_metrics["mean_box_iou"]),
            "val_mask_iou": float(val_metrics["mean_mask_iou"]),
            "val_dice": float(val_metrics["mean_dice"]),
            "score": float(composite_score),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history_rows.append(row)

        print(
            "  "
            f"loss={train_loss:.4f} "
            f"precision={val_metrics['precision']:.4f} "
            f"recall={val_metrics['recall']:.4f} "
            f"f1={val_metrics['f1']:.4f} "
            f"dice={val_metrics['mean_dice']:.4f} "
            f"mask_iou={val_metrics['mean_mask_iou']:.4f}"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": vars(config),
                "metrics": row,
            },
            last_checkpoint,
        )

        if composite_score > best_score:
            best_score = composite_score
            bad_epochs = 0
            best_metrics = {
                **val_metrics,
                "composite_score": float(composite_score),
                "epoch": float(epoch),
            }
            shutil.copy2(last_checkpoint, best_checkpoint)
            print(f"  Nuevo mejor modelo guardado: {best_checkpoint.name}")
        else:
            bad_epochs += 1

        if bad_epochs >= config.early_stopping_patience:
            print("  Early stopping activado.")
            break

        print()

    elapsed = time.time() - start
    print(f"Tiempo total de entrenamiento: {elapsed / 60.0:.1f} min")

    checkpoint = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(OUTPUT_DIR / "history.csv", index=False)
    save_history_plots(history_df, OUTPUT_DIR)

    print()
    print("Evaluando mejor modelo en test...")
    test_metrics = evaluate_maskrcnn(model, test_loader, device, config.score_thr, config.iou_thr)
    yolo_metrics = evaluate_yolo_against_masks(test_ds, config.score_thr, config.iou_thr)
    save_comparison_plot(test_metrics, yolo_metrics, OUTPUT_DIR)
    draw_prediction_gallery(model, test_ds, device, OUTPUT_DIR / "prediction_gallery", config.score_thr)

    shutil.copy2(best_checkpoint, OUTPUT_MODEL)
    save_artifacts_summary(config, history_df, best_metrics, test_metrics, yolo_metrics, best_checkpoint)

    print()
    print("=" * 72)
    print("RESUMEN FINAL")
    print("=" * 72)
    print(f"Mask R-CNN test precision: {test_metrics['precision']:.4f}")
    print(f"Mask R-CNN test recall:    {test_metrics['recall']:.4f}")
    print(f"Mask R-CNN test f1:        {test_metrics['f1']:.4f}")
    print(f"Mask R-CNN box IoU:        {test_metrics['mean_box_iou']:.4f}")
    print(f"Mask R-CNN mask IoU:       {test_metrics['mean_mask_iou']:.4f}")
    print(f"Mask R-CNN dice:           {test_metrics['mean_dice']:.4f}")
    if yolo_metrics.get("available", 0.0) >= 1.0:
        print()
        print(f"YOLO actual precision:     {yolo_metrics['precision']:.4f}")
        print(f"YOLO actual recall:        {yolo_metrics['recall']:.4f}")
        print(f"YOLO actual f1:            {yolo_metrics['f1']:.4f}")
        print(f"YOLO actual box IoU:       {yolo_metrics['mean_box_iou']:.4f}")
    else:
        print()
        print("YOLO actual no disponible para comparacion.")
    print()
    print(f"Mejor checkpoint:          {best_checkpoint}")
    print(f"Modelo exportado:          {OUTPUT_MODEL}")
    print(f"Curvas:                    {OUTPUT_DIR / 'training_curves.png'}")
    print(f"Comparacion:               {OUTPUT_DIR / 'comparison_vs_yolo.png'}")
    print(f"Resumen JSON:              {OUTPUT_DIR / 'summary.json'}")
    print()


if __name__ == "__main__":
    main()
