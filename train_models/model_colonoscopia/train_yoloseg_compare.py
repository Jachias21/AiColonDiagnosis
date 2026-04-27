"""
Entrena un YOLO de segmentacion para Fase 2 usando las mascaras exactas.

Por que este script:
- El YOLO actual detecta cajas.
- Este entrena YOLO-seg, que aprende contornos de polipo desde masks/.
- Suele ser bastante mas rapido que Mask R-CNN y mas facil de integrar en la app.
- Al final compara el nuevo YOLO-seg contra models/colonoscopy.pt.

Uso:
    .\\.venv\\Scripts\\python.exe .\\train_models\\model_colonoscopia\\train_yoloseg_compare.py

Salidas:
- data/dataset_yolo_seg/
- train_models/model_colonoscopia/yoloseg_compare/
- models/colonoscopy_yoloseg.pt
- models/colonoscopy_yoloseg_meta.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DATASET = PROJECT_ROOT / "data" / "dataset_yolo"
SEG_DATASET = PROJECT_ROOT / "data" / "dataset_yolo_seg"
TRAIN_PROJECT = PROJECT_ROOT / "train_models" / "model_colonoscopia"
TRAIN_NAME = "yoloseg_compare"
OUTPUT_MODEL = PROJECT_ROOT / "models" / "colonoscopy_yoloseg.pt"
OUTPUT_META = PROJECT_ROOT / "models" / "colonoscopy_yoloseg_meta.json"
CURRENT_YOLO_MODEL = PROJECT_ROOT / "models" / "colonoscopy.pt"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_BASE_MODEL = "yolov8m-seg.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO segmentation and compare against current YOLO detector")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="Modelo base Ultralytics. Ej: yolov8s-seg.pt, yolov8m-seg.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--patience", type=int, default=22)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="", help="'' auto, 0 GPU, cpu CPU")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.60)
    parser.add_argument("--rebuild-dataset", action="store_true", help="Regenera labels de segmentacion aunque ya existan")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def image_files(path: Path) -> list[Path]:
    return sorted([p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS])


def find_mask(mask_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = mask_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def mask_to_yolo_segments(mask_path: Path, image_shape: tuple[int, int]) -> list[str]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []

    img_h, img_w = image_shape
    if mask.shape[:2] != (img_h, img_w):
        mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

    binary = (mask > 127).astype(np.uint8) * 255
    contours, _hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines: list[str] = []

    min_area = max(12.0, img_h * img_w * 0.00003)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        epsilon = max(1.0, 0.0025 * cv2.arcLength(contour, True))
        polygon = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if polygon.shape[0] < 3:
            continue

        coords: list[str] = []
        for x, y in polygon:
            x_norm = min(max(float(x) / max(img_w, 1), 0.0), 1.0)
            y_norm = min(max(float(y) / max(img_h, 1), 0.0), 1.0)
            coords.extend([f"{x_norm:.6f}", f"{y_norm:.6f}"])

        if len(coords) >= 6:
            lines.append("0 " + " ".join(coords))

    return lines


def build_segmentation_dataset(rebuild: bool = False) -> Path:
    yaml_path = SEG_DATASET / "data.yaml"
    sentinel = SEG_DATASET / ".built_from_masks"
    if yaml_path.exists() and sentinel.exists() and not rebuild:
        print(f"Dataset de segmentacion ya preparado: {SEG_DATASET}")
        return yaml_path

    if SEG_DATASET.exists() and rebuild:
        shutil.rmtree(SEG_DATASET)

    print("Preparando dataset YOLO-seg desde masks/...")
    for split in ("train", "val", "test"):
        src_images = SOURCE_DATASET / "images" / split
        src_masks = SOURCE_DATASET / "masks" / split
        dst_images = SEG_DATASET / "images" / split
        dst_labels = SEG_DATASET / "labels" / split
        ensure_dir(dst_images)
        ensure_dir(dst_labels)

        if not src_images.exists():
            raise FileNotFoundError(f"No existe {src_images}")

        positives = 0
        negatives = 0
        for image_path in image_files(src_images):
            dst_image = dst_images / image_path.name
            if not dst_image.exists():
                shutil.copy2(image_path, dst_image)

            image = cv2.imread(str(image_path))
            if image is None:
                continue
            h, w = image.shape[:2]
            mask_path = find_mask(src_masks, image_path.stem) if src_masks.exists() else None
            lines = mask_to_yolo_segments(mask_path, (h, w)) if mask_path else []

            label_path = dst_labels / f"{image_path.stem}.txt"
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            if lines:
                positives += 1
            else:
                negatives += 1

        print(f"  {split}: {positives} positivos con mascara, {negatives} negativos")

    yaml_data = {
        "path": str(SEG_DATASET.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "polyp"},
    }
    yaml_path.write_text(yaml.safe_dump(yaml_data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    sentinel.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    print(f"Dataset listo: {yaml_path}")
    return yaml_path


def read_results_csv(run_dir: Path) -> pd.DataFrame:
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    return df


def save_extra_plots(run_dir: Path, comparison: dict[str, Any]) -> None:
    df = read_results_csv(run_dir)
    if not df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        x = df["epoch"] if "epoch" in df else np.arange(len(df))

        loss_cols = [c for c in df.columns if "loss" in c.lower()]
        for col in loss_cols:
            axes[0].plot(x, df[col], label=col, linewidth=1.8)
        axes[0].set_title("Losses YOLO-seg")
        axes[0].grid(alpha=0.25)
        axes[0].legend(fontsize=8)

        metric_cols = [c for c in df.columns if "metrics/" in c.lower()]
        for col in metric_cols:
            axes[1].plot(x, df[col], label=col, linewidth=1.8)
        axes[1].set_title("Metricas YOLO-seg")
        axes[1].set_ylim(0.0, 1.05)
        axes[1].grid(alpha=0.25)
        axes[1].legend(fontsize=8)

        plt.tight_layout()
        fig.savefig(run_dir / "custom_training_curves.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    labels = ["Precision", "Recall", "mAP50", "mAP50-95"]
    seg = comparison.get("yoloseg", {})
    old = comparison.get("current_yolo", {})
    seg_values = [seg.get(k, 0.0) for k in ("precision", "recall", "map50", "map5095")]
    old_values = [old.get(k, 0.0) for k in ("precision", "recall", "map50", "map5095")]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    width = 0.36
    ax.bar(x - width / 2, seg_values, width, label="YOLO-seg nuevo", color="#4bd18a")
    ax.bar(x + width / 2, old_values, width, label="YOLO actual", color="#45b0ff")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Comparacion de validacion/test")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    plt.tight_layout()
    fig.savefig(run_dir / "custom_comparison_vs_current_yolo.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def metrics_from_ultralytics(result: Any, task: str) -> dict[str, float]:
    box = getattr(result, "box", None)
    seg = getattr(result, "seg", None)
    source = seg if task == "segment" and seg is not None else box
    if source is None:
        return {"precision": 0.0, "recall": 0.0, "map50": 0.0, "map5095": 0.0}
    return {
        "precision": float(getattr(source, "mp", 0.0)),
        "recall": float(getattr(source, "mr", 0.0)),
        "map50": float(getattr(source, "map50", 0.0)),
        "map5095": float(getattr(source, "map", 0.0)),
    }


def main() -> None:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Falta ultralytics. Ejecuta uv sync o instala ultralytics.") from exc

    yaml_path = build_segmentation_dataset(rebuild=args.rebuild_dataset)
    ensure_dir(TRAIN_PROJECT)
    ensure_dir(OUTPUT_MODEL.parent)

    print()
    print("=" * 78)
    print("ENTRENAMIENTO FASE 2 - YOLO SEGMENTACION")
    print("=" * 78)
    print(f"Base model: {args.base_model}")
    print(f"Dataset:    {yaml_path}")
    print(f"Epochs:     {args.epochs}")
    print(f"Batch:      {args.batch}")
    print(f"Image size: {args.imgsz}")
    print(f"Salida:     {TRAIN_PROJECT / TRAIN_NAME}")
    print()

    model = YOLO(args.base_model)
    model.train(
        data=str(yaml_path),
        task="segment",
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        patience=args.patience,
        optimizer="AdamW",
        lr0=0.002,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        cos_lr=True,
        close_mosaic=12,
        mosaic=0.8,
        mixup=0.05,
        copy_paste=0.20,
        hsv_h=0.015,
        hsv_s=0.55,
        hsv_v=0.35,
        translate=0.08,
        scale=0.45,
        degrees=3.0,
        fliplr=0.5,
        flipud=0.0,
        cache=True,
        project=str(TRAIN_PROJECT),
        name=TRAIN_NAME,
        exist_ok=True,
        plots=True,
        save=True,
        seed=42,
        workers=args.workers,
        device=args.device,
        verbose=True,
    )

    run_dir = TRAIN_PROJECT / TRAIN_NAME
    best_pt = run_dir / "weights" / "best.pt"
    last_pt = run_dir / "weights" / "last.pt"
    chosen = best_pt if best_pt.exists() else last_pt
    if not chosen.exists():
        raise SystemExit("No se encontro best.pt ni last.pt tras el entrenamiento.")

    print()
    print("Validando YOLO-seg nuevo...")
    seg_model = YOLO(str(chosen))
    seg_val = seg_model.val(data=str(yaml_path), split="test", task="segment", imgsz=args.imgsz, batch=args.batch, conf=args.conf, iou=args.iou, plots=True)
    seg_metrics = metrics_from_ultralytics(seg_val, task="segment")

    yolo_metrics = {"precision": 0.0, "recall": 0.0, "map50": 0.0, "map5095": 0.0, "available": 0.0}
    if CURRENT_YOLO_MODEL.exists():
        print("Validando YOLO actual para comparar...")
        current = YOLO(str(CURRENT_YOLO_MODEL))
        current_val = current.val(data=str(PROJECT_ROOT / "data" / "dataset_yolo" / "data.yaml"), split="test", task="detect", imgsz=args.imgsz, batch=args.batch, conf=args.conf, iou=args.iou, plots=False)
        yolo_metrics = metrics_from_ultralytics(current_val, task="detect")
        yolo_metrics["available"] = 1.0

    comparison = {
        "base_model": args.base_model,
        "dataset": str(yaml_path),
        "trained_weights": str(chosen),
        "exported_model": str(OUTPUT_MODEL),
        "yoloseg": seg_metrics,
        "current_yolo": yolo_metrics,
        "thresholds": {"conf": args.conf, "iou": args.iou},
    }
    (run_dir / "comparison_summary.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    save_extra_plots(run_dir, comparison)

    shutil.copy2(chosen, OUTPUT_MODEL)
    OUTPUT_META.write_text(
        json.dumps(
            {
                "architecture": "ultralytics-yolo-seg",
                "base_model": args.base_model,
                "task": "segment",
                "classes": ["polyp"],
                "source_weights": str(chosen),
                "score_threshold_default": args.conf,
                "iou_threshold_default": args.iou,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("RESULTADO")
    print("=" * 78)
    print(f"YOLO-seg precision: {seg_metrics['precision']:.4f}")
    print(f"YOLO-seg recall:    {seg_metrics['recall']:.4f}")
    print(f"YOLO-seg mAP50:     {seg_metrics['map50']:.4f}")
    print(f"YOLO-seg mAP50-95:  {seg_metrics['map5095']:.4f}")
    if yolo_metrics.get("available"):
        print()
        print(f"YOLO actual precision: {yolo_metrics['precision']:.4f}")
        print(f"YOLO actual recall:    {yolo_metrics['recall']:.4f}")
        print(f"YOLO actual mAP50:     {yolo_metrics['map50']:.4f}")
        print(f"YOLO actual mAP50-95:  {yolo_metrics['map5095']:.4f}")
    print()
    print(f"Modelo exportado: {OUTPUT_MODEL}")
    print(f"Graficas:         {run_dir}")
    print()


if __name__ == "__main__":
    main()
