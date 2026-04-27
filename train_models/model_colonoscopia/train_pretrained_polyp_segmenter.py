"""
Fine-tune a pretrained medical polyp segmenter for Phase 2.

This script is intentionally separate from the production app path. It prepares a
deduplicated segmentation dataset, fine-tunes the Hugging Face UNet3+
EfficientNet model trained on Kvasir-SEG, and compares it against the current
YOLO baseline before exporting anything for later app integration.

Typical use:
    .\\.venv\\Scripts\\python.exe .\\train_models\\model_colonoscopia\\train_pretrained_polyp_segmenter.py

Fast smoke test:
    .\\.venv\\Scripts\\python.exe .\\train_models\\model_colonoscopia\\train_pretrained_polyp_segmenter.py --prepare-only

Outputs:
    data/dataset_polyp_segmenter/
    data/external_polyp_sources/
    train_models/model_colonoscopia/pretrained_polyp_segmenter/
    models/colonoscopy_unet3plus_effnet.pt
    models/colonoscopy_unet3plus_effnet_meta.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import random
import shutil
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageEnhance
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_YOLO_DATASET = PROJECT_ROOT / "data" / "dataset_yolo"
EXTERNAL_ROOT = PROJECT_ROOT / "data" / "external_polyp_sources"
PREPARED_DATASET = PROJECT_ROOT / "data" / "dataset_polyp_segmenter"
OUTPUT_DIR = PROJECT_ROOT / "train_models" / "model_colonoscopia" / "pretrained_polyp_segmenter"
OUTPUT_MODEL = PROJECT_ROOT / "models" / "colonoscopy_unet3plus_effnet.pt"
OUTPUT_META = PROJECT_ROOT / "models" / "colonoscopy_unet3plus_effnet_meta.json"

HF_MODEL_ID = "andreribeiro87/unet3plus-efficientnet-kvasir-seg"
KVAZIR_SEG_URLS = [
    "https://datasets.simula.no/downloads/kvasir-seg.zip",
    "https://datasets.simula.no/kvasir-seg/kvasir-seg.zip",
]
CURRENT_YOLO_MODEL = PROJECT_ROOT / "models" / "colonoscopy.pt"
YOLOSEG_MODEL = PROJECT_ROOT / "models" / "colonoscopy_yoloseg.pt"
MASKRCNN_SUMMARY = PROJECT_ROOT / "train_models" / "model_colonoscopia" / "maskrcnn_resnet50_compare" / "summary.json"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
RANDOM_SEED = 42
CVC_HINTS = [
    EXTERNAL_ROOT / "cvc_clinicdb",
    EXTERNAL_ROOT / "cvc-clinicdb",
    PROJECT_ROOT / "data" / "cvc_clinicdb",
    PROJECT_ROOT / "data" / "cvc-clinicdb",
]


@dataclass(frozen=True)
class RawSample:
    image_path: Path
    mask_path: Path | None
    source: str
    source_split: str | None = None
    is_positive: bool = True


@dataclass
class TrainConfig:
    epochs: int = 55
    frozen_epochs: int = 5
    batch_size: int = 0
    image_size: int = 384
    lr: float = 8e-4
    min_lr_factor: float = 0.02
    weight_decay: float = 2.5e-6
    patience: int = 12
    workers: int = 4
    threshold: float = 0.50
    min_mask_area_ratio: float = 0.00035
    seed: int = RANDOM_SEED
    device: str = ""
    download_kvasir: bool = True
    download_hf_kvasir: bool = False
    download_hyperkvasir_normals: bool = False
    max_external_positive: int = 2000
    max_external_negative: int = 2000
    rebuild_dataset: bool = False
    prepare_only: bool = False
    skip_yolo_compare: bool = False


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Fine-tune pretrained UNet3+ EfficientNet polyp segmenter")
    parser.add_argument("--epochs", type=int, default=55)
    parser.add_argument("--frozen-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=0, help="0 = auto by GPU/CPU")
    parser.add_argument("--image-size", type=int, default=384, choices=[256, 320, 352, 384, 448, 512])
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--min-lr-factor", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=2.5e-6)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--min-mask-area-ratio", type=float, default=0.00035)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--device", default="", help="'' auto, '0' CUDA device, or 'cpu'")
    parser.add_argument("--rebuild-dataset", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-yolo-compare", action="store_true")
    parser.add_argument("--no-download-kvasir", dest="download_kvasir", action="store_false")
    parser.add_argument("--download-hf-kvasir", action="store_true", help="Optional HF augmented Kvasir source")
    parser.add_argument("--download-hyperkvasir-normals", action="store_true", help="Optional, streaming and capped")
    parser.add_argument("--max-external-positive", type=int, default=2000)
    parser.add_argument("--max-external-negative", type=int, default=2000)
    args = parser.parse_args()
    return TrainConfig(**vars(args))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_images(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def find_mask(mask_dir: Path, stem: str) -> Path | None:
    if not mask_dir.exists():
        return None
    for ext in IMAGE_EXTS:
        candidate = mask_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def image_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def average_hash(path: Path, size: int = 12) -> str:
    with Image.open(path) as img:
        img = img.convert("L").resize((size, size), Image.Resampling.BILINEAR)
        arr = np.asarray(img, dtype=np.float32)
    bits = arr > float(arr.mean())
    packed = np.packbits(bits.astype(np.uint8).reshape(-1))
    return packed.tobytes().hex()


def mask_has_pixels(mask_path: Path | None) -> bool:
    if mask_path is None or not mask_path.exists():
        return False
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    return bool(mask is not None and int((mask > 10).sum()) > 0)


def load_local_samples() -> list[RawSample]:
    samples: list[RawSample] = []
    for split in ("train", "val", "test"):
        image_dir = LOCAL_YOLO_DATASET / "images" / split
        mask_dir = LOCAL_YOLO_DATASET / "masks" / split
        if not image_dir.exists():
            continue
        for image_path in list_images(image_dir):
            mask_path = find_mask(mask_dir, image_path.stem)
            samples.append(
                RawSample(
                    image_path=image_path,
                    mask_path=mask_path,
                    source="local_dataset_yolo",
                    source_split=split,
                    is_positive=mask_has_pixels(mask_path),
                )
            )
    if not samples:
        raise FileNotFoundError(f"No local samples found in {LOCAL_YOLO_DATASET}")
    return samples


def download_file(urls: list[str], output_path: Path, timeout: int = 30) -> bool:
    if output_path.exists() and output_path.stat().st_size > 0:
        return True
    ensure_dir(output_path.parent)
    for url in urls:
        for verify in (True, False):
            try:
                with requests.get(url, stream=True, timeout=timeout, verify=verify) as response:
                    if response.status_code != 200:
                        continue
                    total = int(response.headers.get("content-length", 0))
                    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
                    with tmp_path.open("wb") as fh, tqdm(
                        total=total,
                        unit="B",
                        unit_scale=True,
                        desc=f"Downloading {output_path.name}",
                    ) as bar:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                fh.write(chunk)
                                bar.update(len(chunk))
                    tmp_path.replace(output_path)
                    return True
            except Exception:
                continue
    return False


def locate_kvasir_pairs(root: Path) -> list[RawSample]:
    image_dirs: list[Path] = []
    mask_dirs: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        name = path.name.lower()
        if "image" in name:
            image_dirs.append(path)
        if "mask" in name:
            mask_dirs.append(path)

    pairs: list[RawSample] = []
    for image_dir in image_dirs:
        best_mask_dir = None
        for mask_dir in mask_dirs:
            if image_dir.parent == mask_dir.parent or image_dir.parent.parent == mask_dir.parent.parent:
                best_mask_dir = mask_dir
                break
        if best_mask_dir is None and mask_dirs:
            best_mask_dir = mask_dirs[0]
        if best_mask_dir is None:
            continue
        for image_path in list_images(image_dir):
            mask_path = find_mask(best_mask_dir, image_path.stem)
            if mask_path is not None:
                pairs.append(RawSample(image_path, mask_path, "kvasir_seg_official", None, True))
    return pairs


def locate_generic_pairs(root: Path, source: str) -> list[RawSample]:
    image_dirs: list[Path] = []
    mask_dirs: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        name = path.name.lower()
        if any(token in name for token in ("image", "images", "original", "frames")):
            image_dirs.append(path)
        if any(token in name for token in ("mask", "masks", "ground", "ground_truth", "gt")):
            mask_dirs.append(path)

    pairs: list[RawSample] = []
    for image_dir in image_dirs:
        for image_path in list_images(image_dir):
            mask_path = None
            for mask_dir in mask_dirs:
                mask_path = find_mask(mask_dir, image_path.stem)
                if mask_path is not None:
                    break
            if mask_path is not None:
                pairs.append(RawSample(image_path, mask_path, source, None, True))
    return pairs


def maybe_download_kvasir(config: TrainConfig) -> list[RawSample]:
    if not config.download_kvasir:
        return []
    zip_path = EXTERNAL_ROOT / "kvasir_seg" / "kvasir-seg.zip"
    extract_dir = EXTERNAL_ROOT / "kvasir_seg" / "extracted"
    if not extract_dir.exists():
        ok = download_file(KVAZIR_SEG_URLS, zip_path)
        if not ok:
            print("Kvasir-SEG download skipped: official zip could not be downloaded.")
            return []
        ensure_dir(extract_dir)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(extract_dir)
    samples = locate_kvasir_pairs(extract_dir)
    print(f"Kvasir-SEG official samples found: {len(samples)}")
    return samples


def maybe_import_cvc_clinicdb() -> list[RawSample]:
    for root in CVC_HINTS:
        if not root.exists():
            continue
        if root.suffix.lower() == ".zip":
            extract_dir = root.with_suffix("")
            if not extract_dir.exists():
                ensure_dir(extract_dir)
                with zipfile.ZipFile(root) as archive:
                    archive.extractall(extract_dir)
            root = extract_dir
        samples = locate_generic_pairs(root, "cvc_clinicdb")
        if samples:
            print(f"CVC-ClinicDB local samples found: {len(samples)} in {root}")
            return samples
    print("CVC-ClinicDB skipped: no local folder found and Kaggle usually requires credentials.")
    print(f"  Put it under one of: {', '.join(str(p) for p in CVC_HINTS[:2])}")
    return []


def save_pil_sample(image: Any, mask: Any | None, target_dir: Path, stem: str, source: str, positive: bool) -> RawSample:
    ensure_dir(target_dir / "images")
    ensure_dir(target_dir / "masks")
    image_path = target_dir / "images" / f"{stem}.jpg"
    mask_path = target_dir / "masks" / f"{stem}.png"

    if isinstance(image, Image.Image):
        img = image.convert("RGB")
    else:
        img = Image.fromarray(np.asarray(image)).convert("RGB")
    img.save(image_path, quality=95)

    if mask is not None:
        if isinstance(mask, Image.Image):
            m = mask.convert("L")
        else:
            m = Image.fromarray(np.asarray(mask)).convert("L")
        m.save(mask_path)
        positive = mask_has_pixels(mask_path)
    else:
        Image.new("L", img.size, 0).save(mask_path)
        positive = False

    return RawSample(image_path, mask_path if positive else None, source, None, positive)


def maybe_download_hf_kvasir(config: TrainConfig) -> list[RawSample]:
    if not config.download_hf_kvasir:
        return []
    try:
        from datasets import load_dataset
    except Exception as exc:
        print(f"HF Kvasir skipped: datasets import failed: {exc}")
        return []

    out_dir = EXTERNAL_ROOT / "hf_kvasir_augmented"
    manifest = out_dir / "manifest.csv"
    if manifest.exists() and not config.rebuild_dataset:
        return read_manifest_samples(manifest)

    samples: list[RawSample] = []
    try:
        ds = load_dataset("andreribeiro87/kvasir-seg-augmented", split="train", streaming=True)
        for idx, row in enumerate(ds):
            if len(samples) >= config.max_external_positive:
                break
            image = row.get("image") or row.get("pixel_values")
            mask = row.get("mask") or row.get("label") or row.get("segmentation_mask")
            if image is None or mask is None:
                continue
            samples.append(save_pil_sample(image, mask, out_dir, f"hf_kvasir_{idx:06d}", "hf_kvasir_augmented", True))
    except Exception as exc:
        print(f"HF Kvasir skipped: {exc}")
        return []

    write_manifest_samples(manifest, samples)
    print(f"HF Kvasir augmented samples saved: {len(samples)}")
    return samples


def maybe_download_hyperkvasir_normals(config: TrainConfig) -> list[RawSample]:
    if not config.download_hyperkvasir_normals:
        return []
    try:
        from datasets import load_dataset
    except Exception as exc:
        print(f"HyperKvasir normals skipped: datasets import failed: {exc}")
        return []

    out_dir = EXTERNAL_ROOT / "hyperkvasir_normals"
    manifest = out_dir / "manifest.csv"
    if manifest.exists() and not config.rebuild_dataset:
        return read_manifest_samples(manifest)

    samples: list[RawSample] = []
    try:
        ds = load_dataset("SimulaMet-HOST/HyperKvasir", split="train", streaming=True)
        for idx, row in enumerate(ds):
            if len(samples) >= config.max_external_negative:
                break
            image = row.get("image")
            if image is None:
                continue
            label_text = " ".join(str(row.get(k, "")).lower() for k in ("label", "class", "category", "finding"))
            if "polyp" in label_text:
                continue
            samples.append(save_pil_sample(image, None, out_dir, f"hyper_norm_{idx:06d}", "hyperkvasir_normal", False))
    except Exception as exc:
        print(f"HyperKvasir normals skipped: {exc}")
        return []

    write_manifest_samples(manifest, samples)
    print(f"HyperKvasir normal samples saved: {len(samples)}")
    return samples


def read_manifest_samples(path: Path) -> list[RawSample]:
    samples: list[RawSample] = []
    with path.open("r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            image_path = Path(row["image_path"])
            mask_path = Path(row["mask_path"]) if row.get("mask_path") else None
            samples.append(
                RawSample(
                    image_path=image_path,
                    mask_path=mask_path,
                    source=row.get("source", "unknown"),
                    source_split=row.get("split") or None,
                    is_positive=row.get("is_positive", "1") == "1",
                )
            )
    return samples


def write_manifest_samples(path: Path, samples: Iterable[RawSample]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["image_path", "mask_path", "source", "split", "is_positive"])
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "image_path": str(sample.image_path),
                    "mask_path": str(sample.mask_path or ""),
                    "source": sample.source,
                    "split": sample.source_split or "",
                    "is_positive": "1" if sample.is_positive else "0",
                }
            )


def copy_sample(sample: RawSample, split: str, index: int) -> dict[str, str]:
    image = Image.open(sample.image_path).convert("RGB")
    mask = Image.open(sample.mask_path).convert("L") if sample.mask_path and sample.mask_path.exists() else Image.new("L", image.size, 0)
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.Resampling.NEAREST)

    safe_source = "".join(c if c.isalnum() else "_" for c in sample.source.lower())
    stem = f"{split}_{index:06d}_{safe_source}"
    image_dst = PREPARED_DATASET / "images" / split / f"{stem}.jpg"
    mask_dst = PREPARED_DATASET / "masks" / split / f"{stem}.png"
    ensure_dir(image_dst.parent)
    ensure_dir(mask_dst.parent)
    image.save(image_dst, quality=95)
    mask.save(mask_dst)
    return {
        "split": split,
        "image_path": str(image_dst),
        "mask_path": str(mask_dst),
        "source": sample.source,
        "source_split": sample.source_split or "",
        "is_positive": "1" if mask_has_pixels(mask_dst) else "0",
        "sha256": image_sha256(image_dst),
        "ahash": average_hash(image_dst),
    }


def build_prepared_dataset(config: TrainConfig) -> Path:
    manifest_path = PREPARED_DATASET / "manifest.csv"
    if manifest_path.exists() and not config.rebuild_dataset:
        print(f"Prepared dataset already exists: {manifest_path}")
        return manifest_path

    if PREPARED_DATASET.exists():
        shutil.rmtree(PREPARED_DATASET)

    local_samples = load_local_samples()
    external_samples: list[RawSample] = []
    external_samples.extend(maybe_download_kvasir(config))
    external_samples.extend(maybe_import_cvc_clinicdb())
    external_samples.extend(maybe_download_hf_kvasir(config))
    external_samples.extend(maybe_download_hyperkvasir_normals(config))

    protected_test_hashes = {
        average_hash(s.image_path)
        for s in local_samples
        if s.source_split == "test"
    }
    seen_hashes: set[str] = set()
    rows: list[dict[str, str]] = []
    counters = {"train": 0, "val": 0, "test": 0}
    build_report: dict[str, Any] = {
        "local_samples": len(local_samples),
        "external_samples_seen": len(external_samples),
        "external_added": 0,
        "external_duplicate_skipped": 0,
        "external_test_duplicate_skipped": 0,
    }

    ordered_local = sorted(local_samples, key=lambda s: 0 if s.source_split == "test" else 1)
    for sample in ordered_local:
        split = sample.source_split or "train"
        key = average_hash(sample.image_path)
        if key in seen_hashes:
            continue
        seen_hashes.add(key)
        rows.append(copy_sample(sample, split, counters[split]))
        counters[split] += 1

    rng = random.Random(config.seed)
    positives = [s for s in external_samples if s.is_positive]
    negatives = [s for s in external_samples if not s.is_positive]
    rng.shuffle(positives)
    rng.shuffle(negatives)
    positives = positives[: config.max_external_positive]
    negatives = negatives[: config.max_external_negative]

    for sample in positives + negatives:
        key = average_hash(sample.image_path)
        if key in protected_test_hashes:
            build_report["external_test_duplicate_skipped"] += 1
            continue
        if key in seen_hashes:
            build_report["external_duplicate_skipped"] += 1
            continue
        seen_hashes.add(key)
        split = "val" if rng.random() < 0.15 else "train"
        rows.append(copy_sample(sample, split, counters[split]))
        counters[split] += 1
        build_report["external_added"] += 1

    ensure_dir(PREPARED_DATASET)
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["split", "image_path", "mask_path", "source", "source_split", "is_positive", "sha256", "ahash"],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = pd.DataFrame(rows).groupby(["split", "is_positive", "source"]).size().reset_index(name="count")
    summary.to_csv(PREPARED_DATASET / "summary.csv", index=False)
    build_report["final_rows"] = len(rows)
    build_report["split_counts"] = counters
    (PREPARED_DATASET / "build_report.json").write_text(json.dumps(build_report, indent=2), encoding="utf-8")
    print(f"Prepared dataset: {manifest_path}")
    print(summary.to_string(index=False))
    print(json.dumps(build_report, indent=2))
    return manifest_path


class PolypSegmentationDataset(Dataset):
    def __init__(self, manifest: Path, split: str, image_size: int, train: bool) -> None:
        self.rows = [row for row in csv.DictReader(manifest.open("r", encoding="utf-8")) if row["split"] == split]
        self.image_size = image_size
        self.train = train
        if not self.rows:
            raise RuntimeError(f"No rows for split '{split}' in {manifest}")

    def __len__(self) -> int:
        return len(self.rows)

    def _augment(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        if not self.train:
            return image, mask
        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        if random.random() < 0.15:
            angle = random.uniform(-6.0, 6.0)
            image = TF.affine(image, angle=angle, translate=[0, 0], scale=1.0, shear=[0.0, 0.0], interpolation=InterpolationMode.BILINEAR)
            mask = TF.affine(mask, angle=angle, translate=[0, 0], scale=1.0, shear=[0.0, 0.0], interpolation=InterpolationMode.NEAREST)
        if random.random() < 0.25:
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.20))
            image = ImageEnhance.Color(image).enhance(random.uniform(0.85, 1.15))
        return image, mask

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image = Image.open(row["image_path"]).convert("RGB")
        mask = Image.open(row["mask_path"]).convert("L")
        image = TF.resize(image, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.image_size, self.image_size], interpolation=InterpolationMode.NEAREST)
        image, mask = self._augment(image, mask)
        image_tensor = TF.to_tensor(image)
        mask_tensor = (TF.to_tensor(mask) > 0.5).float()
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_path": row["image_path"],
            "mask_path": row["mask_path"],
        }


class DiceFocalBCELoss(nn.Module):
    def __init__(self, dice_weight: float = 0.55, focal_weight: float = 0.35, bce_weight: float = 0.10, gamma: float = 1.4) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.bce_weight = bce_weight
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        smooth = 1.0
        intersection = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice_loss = 1.0 - ((2.0 * intersection + smooth) / (union + smooth)).mean()

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        focal = ((1.0 - pt) ** self.gamma * bce).mean()
        bce_loss = bce.mean()
        return self.dice_weight * dice_loss + self.focal_weight * focal + self.bce_weight * bce_loss


def choose_device(device_arg: str) -> torch.device:
    if device_arg.lower() == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        index = int(device_arg) if device_arg.strip().isdigit() else 0
        return torch.device(f"cuda:{index}")
    return torch.device("cpu")


def auto_batch_size(device: torch.device, image_size: int) -> int:
    if device.type != "cuda":
        return 4 if image_size <= 384 else 2
    total_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    if total_gb >= 16:
        return 16 if image_size <= 384 else 10
    if total_gb >= 10:
        return 10 if image_size <= 384 else 6
    return 6 if image_size <= 384 else 4


def load_pretrained_model(device: torch.device) -> nn.Module:
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    snapshot = Path(snapshot_download(HF_MODEL_ID))
    sys.path.insert(0, str(snapshot))
    importlib.invalidate_caches()
    from modeling_unet3plus import UNet3PlusConfig, UNet3PlusForSegmentation

    config = UNet3PlusConfig.from_pretrained(str(snapshot))
    model = UNet3PlusForSegmentation(config)
    state_dict = load_file(str(snapshot / "model.safetensors"))
    try:
        missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    except TypeError:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Pretrained load warning: missing keys={len(missing)}")
    if unexpected:
        print(f"Pretrained load warning: unexpected keys={len(unexpected)}")
    model.to(device)
    if device.type == "cuda":
        model.to(memory_format=torch.channels_last)
    return model


def set_encoder_trainable(model: nn.Module, trainable: bool) -> None:
    changed = 0
    for name, param in model.named_parameters():
        if ".backbone." in name or name.endswith("backbone") or "model.backbone" in name:
            param.requires_grad = trainable
            changed += 1
    if changed == 0:
        print("Warning: no explicit backbone parameters found to freeze/unfreeze.")
    else:
        print(f"Backbone trainable={trainable} parameters affected={changed}")


def unpack_logits(output: Any) -> torch.Tensor:
    if isinstance(output, dict):
        return output["logits"]
    if isinstance(output, (list, tuple)):
        return output[0]
    return output


def binary_metrics(pred: np.ndarray, target: np.ndarray, min_area_ratio: float) -> dict[str, float]:
    pred = pred.astype(bool)
    target = target.astype(bool)
    min_area = max(4, int(pred.size * min_area_ratio))
    if int(pred.sum()) < min_area:
        pred[:] = False
    tp_pixels = np.logical_and(pred, target).sum()
    pred_pixels = pred.sum()
    target_pixels = target.sum()
    union_pixels = np.logical_or(pred, target).sum()
    dice = (2.0 * tp_pixels + 1.0) / (pred_pixels + target_pixels + 1.0)
    mask_iou = (tp_pixels + 1.0) / (union_pixels + 1.0)
    return {
        "has_pred": float(pred_pixels > 0),
        "has_target": float(target_pixels > 0),
        "dice": float(dice),
        "mask_iou": float(mask_iou),
        "box_iou": float(box_iou_from_masks(pred, target)),
    }


def box_iou_from_masks(pred: np.ndarray, target: np.ndarray) -> float:
    def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
        ys, xs = np.where(mask > 0)
        if xs.size == 0 or ys.size == 0:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

    a = bbox(pred)
    b = bbox(target)
    if a is None or b is None:
        return 0.0
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return float(inter / max(area_a + area_b - inter, 1))


@torch.no_grad()
def evaluate_unet(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    min_area_ratio: float,
) -> dict[str, float]:
    model.eval()
    rows: list[dict[str, float]] = []
    total_time = 0.0
    n_images = 0
    for batch in tqdm(loader, desc="Evaluating UNet3+", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        if device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        output = model(pixel_values=images)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        total_time += time.perf_counter() - started
        probs = torch.sigmoid(unpack_logits(output)).detach().cpu().numpy()
        targets = masks.detach().cpu().numpy()
        for pred, target in zip(probs, targets, strict=False):
            rows.append(binary_metrics(pred[0] >= threshold, target[0] > 0.5, min_area_ratio))
            n_images += 1
    return aggregate_rows(rows, total_time, n_images)


def aggregate_rows(rows: list[dict[str, float]], elapsed: float, n_images: int) -> dict[str, float]:
    tp = fp = fn = tn = 0
    dice_values: list[float] = []
    miou_values: list[float] = []
    biou_values: list[float] = []
    for row in rows:
        has_pred = row["has_pred"] > 0
        has_target = row["has_target"] > 0
        if has_pred and has_target:
            tp += 1
            dice_values.append(row["dice"])
            miou_values.append(row["mask_iou"])
            biou_values.append(row["box_iou"])
        elif has_pred and not has_target:
            fp += 1
        elif not has_pred and has_target:
            fn += 1
        else:
            tn += 1
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_dice": float(np.mean(dice_values)) if dice_values else 0.0,
        "mean_mask_iou": float(np.mean(miou_values)) if miou_values else 0.0,
        "mean_box_iou": float(np.mean(biou_values)) if biou_values else 0.0,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "ms_per_image": float((elapsed / max(n_images, 1)) * 1000.0),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
) -> float:
    model.train()
    losses: list[float] = []
    for batch in tqdm(loader, desc="Training", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        if device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            output = model(pixel_values=images)
            loss = loss_fn(unpack_logits(output), masks)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


def make_loaders(config: TrainConfig, manifest: Path, device: torch.device) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    batch_size = config.batch_size or auto_batch_size(device, config.image_size)
    workers = config.workers if sys.platform != "win32" else min(config.workers, 2)
    train_ds = PolypSegmentationDataset(manifest, "train", config.image_size, train=True)
    val_ds = PolypSegmentationDataset(manifest, "val", config.image_size, train=False)
    test_ds = PolypSegmentationDataset(manifest, "test", config.image_size, train=False)
    kwargs = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
    }
    return (
        DataLoader(train_ds, shuffle=True, drop_last=False, **kwargs),
        DataLoader(val_ds, shuffle=False, drop_last=False, **kwargs),
        DataLoader(test_ds, shuffle=False, drop_last=False, **kwargs),
        batch_size,
    )


def train_model(config: TrainConfig, manifest: Path) -> tuple[dict[str, float], Path, list[dict[str, float]], nn.Module]:
    ensure_dir(OUTPUT_DIR)
    ensure_dir(OUTPUT_MODEL.parent)
    device = choose_device(config.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    train_loader, val_loader, test_loader, batch_size = make_loaders(config, manifest, device)
    print(f"Batch size: {batch_size}")
    model = load_pretrained_model(device)
    set_encoder_trainable(model, trainable=False)

    loss_fn = DiceFocalBCELoss()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(config.epochs, 1), eta_min=config.lr * config.min_lr_factor)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    use_amp = device.type == "cuda"
    best_score = -math.inf
    best_path = OUTPUT_DIR / "best_unet3plus_effnet.pt"
    bad_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        if epoch == config.frozen_epochs + 1:
            set_encoder_trainable(model, trainable=True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr * 0.35, weight_decay=config.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(config.epochs - config.frozen_epochs, 1),
                eta_min=config.lr * config.min_lr_factor,
            )

        loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler, use_amp)
        val_metrics = evaluate_unet(model, val_loader, device, config.threshold, config.min_mask_area_ratio)
        scheduler.step()
        score = 0.45 * val_metrics["f1"] + 0.35 * val_metrics["mean_dice"] + 0.20 * val_metrics["mean_mask_iou"]
        row = {"epoch": float(epoch), "loss": loss, "lr": optimizer.param_groups[0]["lr"], "score": score, **val_metrics}
        history.append(row)
        print(
            f"epoch={epoch:03d} loss={loss:.4f} f1={val_metrics['f1']:.4f} "
            f"dice={val_metrics['mean_dice']:.4f} iou={val_metrics['mean_mask_iou']:.4f} "
            f"precision={val_metrics['precision']:.4f} recall={val_metrics['recall']:.4f}"
        )
        if score > best_score:
            best_score = score
            bad_epochs = 0
            torch.save(
                {
                    "architecture": "hf_unet3plus_efficientnet",
                    "hf_model_id": HF_MODEL_ID,
                    "state_dict": model.state_dict(),
                    "config": asdict(config),
                    "threshold": config.threshold,
                    "image_size": config.image_size,
                    "best_val_metrics": val_metrics,
                    "epoch": epoch,
                },
                best_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= config.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    pd.DataFrame(history).to_csv(OUTPUT_DIR / "history.csv", index=False)
    save_training_curves(history)

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    test_metrics = evaluate_unet(model, test_loader, device, config.threshold, config.min_mask_area_ratio)
    torch.save(checkpoint, OUTPUT_MODEL)
    return test_metrics, best_path, history, model


def load_test_rows(manifest: Path) -> list[dict[str, str]]:
    return [row for row in csv.DictReader(manifest.open("r", encoding="utf-8")) if row["split"] == "test"]


def yolo_result_to_mask(result: Any, image_shape: tuple[int, int]) -> np.ndarray:
    h, w = image_shape
    pred = np.zeros((h, w), dtype=np.uint8)
    masks = getattr(result, "masks", None)
    if masks is not None and getattr(masks, "xy", None) is not None:
        for poly in masks.xy:
            pts = np.asarray(poly, dtype=np.int32)
            if pts.size >= 6:
                cv2.fillPoly(pred, [pts], 1)
        return pred.astype(bool)
    boxes = getattr(result, "boxes", None)
    if boxes is not None and getattr(boxes, "xyxy", None) is not None:
        for box in boxes.xyxy.detach().cpu().numpy():
            x1, y1, x2, y2 = [int(round(v)) for v in box[:4]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                pred[y1:y2, x1:x2] = 1
    return pred.astype(bool)


def evaluate_ultralytics_model(model_path: Path, manifest: Path, config: TrainConfig, task_name: str) -> dict[str, float]:
    if not model_path.exists():
        return {"available": 0.0}
    try:
        from ultralytics import YOLO
    except Exception as exc:
        return {"available": 0.0, "error": str(exc)}

    model = YOLO(str(model_path))
    rows: list[dict[str, float]] = []
    elapsed = 0.0
    test_rows = load_test_rows(manifest)
    for row in tqdm(test_rows, desc=f"Evaluating {task_name}", leave=False):
        image = cv2.imread(row["image_path"])
        target = cv2.imread(row["mask_path"], cv2.IMREAD_GRAYSCALE)
        if image is None or target is None:
            continue
        h, w = image.shape[:2]
        started = time.perf_counter()
        result = model.predict(row["image_path"], imgsz=config.image_size, conf=config.threshold, iou=0.60, verbose=False)[0]
        elapsed += time.perf_counter() - started
        pred = yolo_result_to_mask(result, (h, w))
        rows.append(binary_metrics(pred, target > 127, config.min_mask_area_ratio))
    metrics = aggregate_rows(rows, elapsed, len(rows))
    metrics["available"] = 1.0
    return metrics


def read_maskrcnn_metrics() -> dict[str, float]:
    if not MASKRCNN_SUMMARY.exists():
        return {"available": 0.0}
    try:
        summary = json.loads(MASKRCNN_SUMMARY.read_text(encoding="utf-8"))
        metrics = dict(summary.get("test_metrics", {}))
        metrics["available"] = 1.0
        metrics["source"] = "existing_summary"
        return metrics
    except Exception as exc:
        return {"available": 0.0, "error": str(exc)}


def save_training_curves(history: list[dict[str, float]]) -> None:
    if not history:
        return
    df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(df["epoch"], df["loss"], label="loss", color="#ff6f91")
    axes[0].set_title("Training loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    for key in ("f1", "mean_dice", "mean_mask_iou", "precision", "recall"):
        if key in df:
            axes[1].plot(df["epoch"], df[key], label=key)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_title("Validation metrics")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "training_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_comparison_plot(comparison: dict[str, dict[str, float]]) -> None:
    labels = ["precision", "recall", "f1", "mean_dice", "mean_mask_iou", "mean_box_iou"]
    available = [(name, metrics) for name, metrics in comparison.items() if metrics.get("available", 1.0)]
    if not available:
        return
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(labels))
    width = 0.80 / max(len(available), 1)
    for idx, (name, metrics) in enumerate(available):
        values = [float(metrics.get(label, 0.0)) for label in labels]
        ax.bar(x - 0.4 + width / 2 + idx * width, values, width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Fixed test comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "comparison_vs_baselines.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def overlay_mask(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    base = image.convert("RGBA")
    alpha = Image.new("L", base.size, 0)
    alpha_arr = np.asarray(alpha).copy()
    mask_resized = cv2.resize(mask.astype(np.uint8), base.size, interpolation=cv2.INTER_NEAREST) > 0
    alpha_arr[mask_resized] = 110
    overlay = Image.new("RGBA", base.size, color + (0,))
    overlay.putalpha(Image.fromarray(alpha_arr))
    return Image.alpha_composite(base, overlay).convert("RGB")


@torch.no_grad()
def save_prediction_gallery(model: nn.Module, manifest: Path, config: TrainConfig) -> None:
    device = next(model.parameters()).device
    out_dir = OUTPUT_DIR / "prediction_gallery"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    rows = load_test_rows(manifest)
    rng = random.Random(config.seed)
    rng.shuffle(rows)
    selected = rows[:16]
    model.eval()
    for idx, row in enumerate(selected, start=1):
        original = Image.open(row["image_path"]).convert("RGB")
        gt = np.asarray(Image.open(row["mask_path"]).convert("L")) > 127
        resized = TF.resize(original, [config.image_size, config.image_size], interpolation=InterpolationMode.BILINEAR)
        tensor = TF.to_tensor(resized).unsqueeze(0).to(device)
        if device.type == "cuda":
            tensor = tensor.contiguous(memory_format=torch.channels_last)
        pred_logits = unpack_logits(model(pixel_values=tensor))
        pred = (torch.sigmoid(pred_logits)[0, 0].detach().cpu().numpy() >= config.threshold)
        pred = cv2.resize(pred.astype(np.uint8), original.size, interpolation=cv2.INTER_NEAREST) > 0

        panels = [
            original,
            overlay_mask(original, gt, (80, 180, 255)),
            overlay_mask(original, pred, (255, 90, 110)),
        ]
        canvas = Image.new("RGB", (original.width * 3, original.height), (20, 24, 34))
        for col, panel in enumerate(panels):
            canvas.paste(panel, (col * original.width, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 10), "original", fill=(255, 255, 255))
        draw.text((original.width + 10, 10), "gt", fill=(255, 255, 255))
        draw.text((original.width * 2 + 10, 10), "pred", fill=(255, 255, 255))
        canvas.save(out_dir / f"sample_{idx:02d}.jpg", quality=92)


def write_meta(config: TrainConfig, test_metrics: dict[str, float], comparison: dict[str, dict[str, float]], history_rows: int) -> None:
    meta = {
        "architecture": "hf_unet3plus_efficientnet",
        "hf_model_id": HF_MODEL_ID,
        "task": "semantic-segmentation",
        "classes": ["background", "polyp"],
        "model_path": str(OUTPUT_MODEL),
        "dataset": str(PREPARED_DATASET),
        "config": asdict(config),
        "test_metrics": test_metrics,
        "comparison": comparison,
        "history_rows": history_rows,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    OUTPUT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> None:
    config = parse_args()
    set_seed(config.seed)
    ensure_dir(EXTERNAL_ROOT)
    ensure_dir(OUTPUT_DIR)
    manifest = build_prepared_dataset(config)
    if config.prepare_only:
        print("Prepare-only finished.")
        return

    test_metrics, best_path, history, model = train_model(config, manifest)
    comparison: dict[str, dict[str, float]] = {"unet3plus_effnet": {**test_metrics, "available": 1.0}}
    if not config.skip_yolo_compare:
        comparison["current_yolo"] = evaluate_ultralytics_model(CURRENT_YOLO_MODEL, manifest, config, "YOLO current")
        comparison["yoloseg"] = evaluate_ultralytics_model(YOLOSEG_MODEL, manifest, config, "YOLO-seg")
    comparison["maskrcnn_resnet50"] = read_maskrcnn_metrics()

    (OUTPUT_DIR / "comparison_summary.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    save_comparison_plot(comparison)
    save_prediction_gallery(model, manifest, config)
    write_meta(config, test_metrics, comparison, len(history))

    print()
    print("=" * 80)
    print("RESULT")
    print("=" * 80)
    print(f"Best checkpoint: {best_path}")
    print(f"Exported model:  {OUTPUT_MODEL}")
    print(f"Metadata:        {OUTPUT_META}")
    print(f"Curves/gallery:  {OUTPUT_DIR}")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
