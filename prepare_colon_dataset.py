"""
prepare_colon_dataset.py
========================
Prepara el dataset colon_image_sets (clasificación binaria) con split
train/val/test para entrenamiento de un modelo de clasificación.

Estructura de entrada:
  data/colon_image_sets/
    colon_aca/   (5000 imágenes cancerígenas)
    colon_n/     (5000 imágenes normales)

Estructura de salida:
  dataset_colon/
    train/
      colon_aca/
      colon_n/
    val/
      colon_aca/
      colon_n/
    test/
      colon_aca/
      colon_n/

Ejecutar desde la raíz del proyecto:
    python prepare_colon_dataset.py
"""

from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path

# ──────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────
RANDOM_SEED: int = 42
DRY_RUN: bool = False

TRAIN_RATIO: float = 0.70
VAL_RATIO:   float = 0.15
TEST_RATIO:  float = 0.15

OUTPUT_DIR_NAME: str = "dataset_colon"
CLASSES: list[str] = ["colon_aca", "colon_n"]
IMAGE_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


# ──────────────────────────────────────────────
# FUNCIONES
# ──────────────────────────────────────────────

def find_colon_dataset(root: Path) -> Path | None:
    """Busca la carpeta que contenga subcarpetas colon_aca y colon_n."""
    target = {"colon_aca", "colon_n"}
    for dirpath in root.rglob("*"):
        if not dirpath.is_dir():
            continue
        children = {c.name for c in dirpath.iterdir() if c.is_dir()}
        if target.issubset(children):
            return dirpath
    return None


def list_images(folder: Path) -> list[Path]:
    """Lista archivos de imagen válidos en una carpeta."""
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def split_list(items: list[Path], seed: int) -> dict[str, list[Path]]:
    """Divide una lista en train/val/test según los ratios."""
    rng = random.Random(seed)
    shuffled = items.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    return {
        "train": shuffled[:n_train],
        "val":   shuffled[n_train:n_train + n_val],
        "test":  shuffled[n_train + n_val:],
    }


def main() -> None:
    print("=" * 55)
    print("  PREPARACIÓN DE DATASET COLON (CLASIFICACIÓN)")
    print("=" * 55)
    print()

    project_root = Path(__file__).resolve().parent
    print(f"Raíz del proyecto: {project_root}")

    # Buscar carpeta
    print("▸ Buscando carpeta colon_image_sets...")
    dataset_dir = find_colon_dataset(project_root)
    if dataset_dir is None:
        print("✗ ERROR: No se encontró carpeta con colon_aca y colon_n.")
        sys.exit(1)
    print(f"  ✓ Encontrada en: {dataset_dir}")
    print()

    # Leer imágenes por clase
    output_dir = project_root / OUTPUT_DIR_NAME
    counters: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    total_copied = 0

    for cls in CLASSES:
        cls_dir = dataset_dir / cls
        images = list_images(cls_dir)
        print(f"▸ {cls}: {len(images)} imágenes")

        splits = split_list(images, RANDOM_SEED)

        for split_name, split_images in splits.items():
            dst_dir = output_dir / split_name / cls
            if not DRY_RUN:
                dst_dir.mkdir(parents=True, exist_ok=True)

            for img in split_images:
                dst = dst_dir / img.name
                if not DRY_RUN:
                    shutil.copy2(img, dst)
                total_copied += 1

            counters[split_name] += len(split_images)
            print(f"    {split_name:>5s}: {len(split_images)}")

    print()
    print("=" * 55)
    print("  RESUMEN")
    print("=" * 55)
    print(f"  Dataset origen:  {dataset_dir}")
    print(f"  Dataset destino: {output_dir}")
    print(f"  Clases:          {', '.join(CLASSES)}")
    print(f"  ─────────────────────────────────")
    for split_name, count in counters.items():
        print(f"  {split_name:>5s}: {count}")
    print(f"  Total copiadas:  {total_copied}")
    print(f"  DRY_RUN:         {DRY_RUN}")
    print("=" * 55)


if __name__ == "__main__":
    main()
