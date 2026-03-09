"""
prepare_dataset.py
==================
Script para preparar un dataset final balanceado 50/50 para entrenar YOLOv8
con imágenes de colonoscopia.

- 1000 imágenes positivas (con pólipo) → de Kvasir-SEG
- 1000 imágenes negativas (sin pólipo) → de Kvasir general (normal-*)

Genera:
  dataset_yolo/
    images/  train/ val/ test/
    labels/  train/ val/ test/
    masks/   train/ val/ test/
    data.yaml

Ejecutar desde la raíz del proyecto:
    python prepare_dataset.py
"""

from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# ──────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────
DRY_RUN: bool = False          # True → solo simula, no copia archivos
RANDOM_SEED: int = 42          # Semilla fija para reproducibilidad

NUM_POSITIVE: int = 1000       # Imágenes positivas esperadas
NUM_NEGATIVE: int = 1000       # Imágenes negativas deseadas

# Reparto de negativas por carpeta
NEG_DISTRIBUTION: dict[str, int] = {
    "normal-cecum":   333,
    "normal-pylorus":  333,
    "normal-z-line":   334,
}

# Splits
TRAIN_RATIO: float = 0.70
VAL_RATIO:   float = 0.15
TEST_RATIO:  float = 0.15

OUTPUT_DIR_NAME: str = "dataset_yolo"

# Extensiones de imagen válidas
IMAGE_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


# ──────────────────────────────────────────────
# FUNCIONES DE DETECCIÓN DE CARPETAS
# ──────────────────────────────────────────────

def find_kvasir_general(root: Path) -> Optional[Path]:
    """
    Busca la carpeta de Kvasir general dentro del proyecto.
    La identifica porque contiene las subcarpetas:
    'normal-cecum', 'normal-pylorus' y 'normal-z-line'.
    """
    target_names = {"normal-cecum", "normal-pylorus", "normal-z-line"}
    for dirpath in root.rglob("*"):
        if not dirpath.is_dir():
            continue
        children = {c.name for c in dirpath.iterdir() if c.is_dir()}
        if target_names.issubset(children):
            return dirpath
    return None


def find_kvasir_seg(root: Path) -> Optional[Path]:
    """
    Busca la carpeta de Kvasir-SEG dentro del proyecto.
    La identifica porque contiene 'images', 'masks' y un archivo
    que coincida con *bboxes.json (ej: kavsir_bboxes.json o kvasir_bboxes.json).
    """
    for dirpath in root.rglob("*"):
        if not dirpath.is_dir():
            continue
        children_dirs = {c.name for c in dirpath.iterdir() if c.is_dir()}
        children_files = {c.name for c in dirpath.iterdir() if c.is_file()}
        has_images = "images" in children_dirs
        has_masks = "masks" in children_dirs
        has_bboxes = any("bboxes" in f and f.endswith(".json") for f in children_files)
        if has_images and has_masks and has_bboxes:
            return dirpath
    return None


def find_bboxes_json(seg_dir: Path) -> Path:
    """Encuentra el archivo *bboxes.json dentro de Kvasir-SEG."""
    candidates = list(seg_dir.glob("*bboxes*.json"))
    if not candidates:
        print(f"✗ ERROR: No se encontró ningún archivo *bboxes*.json en {seg_dir}")
        sys.exit(1)
    return candidates[0]


# ──────────────────────────────────────────────
# FUNCIONES DE BOUNDING BOX / ETIQUETAS
# ──────────────────────────────────────────────

def bbox_from_json(
    entry: dict, img_width: int, img_height: int
) -> list[str]:
    """
    Convierte las bboxes del JSON a formato YOLO normalizado.
    Devuelve una lista de líneas 'class_id x_center y_center w h'.
    """
    lines: list[str] = []
    for box in entry.get("bbox", []):
        xmin = box["xmin"]
        ymin = box["ymin"]
        xmax = box["xmax"]
        ymax = box["ymax"]
        # Clamp a rango válido
        xmin = max(0, min(xmin, img_width))
        xmax = max(0, min(xmax, img_width))
        ymin = max(0, min(ymin, img_height))
        ymax = max(0, min(ymax, img_height))
        if xmax <= xmin or ymax <= ymin:
            continue
        x_center = ((xmin + xmax) / 2.0) / img_width
        y_center = ((ymin + ymax) / 2.0) / img_height
        w = (xmax - xmin) / img_width
        h = (ymax - ymin) / img_height
        lines.append(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
    return lines


def bbox_from_mask(mask_path: Path) -> list[str]:
    """
    Calcula el bounding box a partir de la máscara binaria.
    Devuelve una línea YOLO normalizada.
    Fallback si el JSON no tiene entrada para una imagen.
    """
    mask = np.array(Image.open(mask_path).convert("L"))
    # Binarizar (umbral > 127)
    binary = (mask > 127).astype(np.uint8)
    if binary.sum() == 0:
        return []
    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)
    ymin, ymax = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
    xmin, xmax = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])
    h_img, w_img = mask.shape[:2]
    if xmax <= xmin or ymax <= ymin:
        return []
    x_center = ((xmin + xmax) / 2.0) / w_img
    y_center = ((ymin + ymax) / 2.0) / h_img
    w = (xmax - xmin) / w_img
    h = (ymax - ymin) / h_img
    return [f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"]


# ──────────────────────────────────────────────
# FUNCIONES DE UTILIDAD
# ──────────────────────────────────────────────

def list_images(folder: Path) -> list[Path]:
    """Lista archivos de imagen válidos en una carpeta."""
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def create_dirs(base: Path) -> None:
    """Crea la estructura de carpetas del dataset YOLO."""
    for sub in ("images", "labels", "masks"):
        for split in ("train", "val", "test"):
            d = base / sub / split
            if not DRY_RUN:
                d.mkdir(parents=True, exist_ok=True)
            else:
                print(f"  [DRY_RUN] mkdir {d}")


def split_indices(
    n: int, seed: int
) -> tuple[list[int], list[int], list[int]]:
    """
    Divide n índices en train/val/test según los ratios definidos.
    """
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    train = indices[:n_train]
    val = indices[n_train:n_train + n_val]
    test = indices[n_train + n_val:]
    return sorted(train), sorted(val), sorted(test)


def write_data_yaml(output_dir: Path) -> None:
    """Genera el archivo data.yaml compatible con YOLOv8."""
    content = (
        f"path: {output_dir.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"\n"
        f"names:\n"
        f"  0: polyp\n"
    )
    yaml_path = output_dir / "data.yaml"
    if not DRY_RUN:
        yaml_path.write_text(content, encoding="utf-8")
    print(f"  → data.yaml generado en {yaml_path}")


# ──────────────────────────────────────────────
# FUNCIÓN PRINCIPAL
# ──────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  PREPARACIÓN DE DATASET YOLO PARA COLONOSCOPIA")
    print("=" * 60)
    print()

    # ── 1. Raíz del proyecto ──
    project_root = Path(__file__).resolve().parent
    print(f"Raíz del proyecto: {project_root}")
    print()

    # ── 2. Detección automática de carpetas ──
    print("▸ Buscando carpeta Kvasir general...")
    kvasir_general = find_kvasir_general(project_root)
    if kvasir_general is None:
        print("✗ ERROR: No se encontró la carpeta de Kvasir general.")
        print("  Asegúrate de que existan subcarpetas 'normal-cecum', "
              "'normal-pylorus', 'normal-z-line' dentro del proyecto.")
        sys.exit(1)
    print(f"  ✓ Kvasir general encontrado en: {kvasir_general}")

    print("▸ Buscando carpeta Kvasir-SEG...")
    kvasir_seg = find_kvasir_seg(project_root)
    if kvasir_seg is None:
        print("✗ ERROR: No se encontró la carpeta de Kvasir-SEG.")
        print("  Asegúrate de que exista una carpeta con 'images', 'masks' "
              "y un archivo *bboxes.json dentro del proyecto.")
        sys.exit(1)
    print(f"  ✓ Kvasir-SEG encontrado en: {kvasir_seg}")
    print()

    # ── 3. Cargar archivos ──
    seg_images_dir = kvasir_seg / "images"
    seg_masks_dir = kvasir_seg / "masks"
    bboxes_json_path = find_bboxes_json(kvasir_seg)
    print(f"▸ Archivo de bboxes: {bboxes_json_path.name}")

    # Cargar JSON de bounding boxes
    with open(bboxes_json_path, "r", encoding="utf-8") as f:
        bboxes_data: dict = json.load(f)
    print(f"  ✓ Entradas en JSON de bboxes: {len(bboxes_data)}")

    # Listar imágenes positivas
    pos_images = list_images(seg_images_dir)
    print(f"  ✓ Imágenes positivas (Kvasir-SEG): {len(pos_images)}")
    if len(pos_images) != NUM_POSITIVE:
        print(f"  ⚠ AVISO: Se esperaban {NUM_POSITIVE} positivas, "
              f"se encontraron {len(pos_images)}.")

    # Validar que cada positiva tiene máscara
    missing_masks = 0
    for img_path in pos_images:
        mask_candidates = [
            seg_masks_dir / img_path.name,
            seg_masks_dir / (img_path.stem + ".png"),
            seg_masks_dir / (img_path.stem + ".jpg"),
        ]
        if not any(m.exists() for m in mask_candidates):
            missing_masks += 1
            print(f"  ⚠ Máscara faltante para: {img_path.name}")
    if missing_masks > 0:
        print(f"  ⚠ Total de máscaras faltantes: {missing_masks}")
    else:
        print("  ✓ Todas las positivas tienen máscara.")

    # Listar imágenes negativas
    print()
    print("▸ Seleccionando imágenes negativas...")
    rng = random.Random(RANDOM_SEED)
    neg_images: list[Path] = []
    for folder_name, count in NEG_DISTRIBUTION.items():
        folder = kvasir_general / folder_name
        if not folder.exists():
            print(f"✗ ERROR: La carpeta '{folder_name}' no existe en {kvasir_general}")
            sys.exit(1)
        available = list_images(folder)
        if len(available) < count:
            print(f"✗ ERROR: '{folder_name}' tiene {len(available)} imágenes, "
                  f"se necesitan {count}.")
            sys.exit(1)
        selected = rng.sample(available, count)
        neg_images.extend(selected)
        print(f"  ✓ {folder_name}: {count} seleccionadas de {len(available)} disponibles")

    print(f"  ✓ Total negativas seleccionadas: {len(neg_images)}")
    if len(neg_images) != NUM_NEGATIVE:
        print(f"  ⚠ AVISO: Se esperaban {NUM_NEGATIVE} negativas, "
              f"se seleccionaron {len(neg_images)}.")
    print()

    # ── 4. Preparar listas con metadatos ──
    # Estructura: (img_path, mask_path_or_None, label_lines, new_stem)
    positives: list[tuple[Path, Optional[Path], list[str], str]] = []
    labels_from_json = 0
    labels_from_mask = 0
    labels_failed = 0

    for idx, img_path in enumerate(pos_images, start=1):
        stem = img_path.stem
        new_stem = f"pos_{idx:04d}"

        # Buscar máscara
        mask_path: Optional[Path] = None
        for ext in (".jpg", ".png", ".jpeg"):
            candidate = seg_masks_dir / (stem + ext)
            if candidate.exists():
                mask_path = candidate
                break

        # Intentar generar label desde JSON
        label_lines: list[str] = []
        if stem in bboxes_data:
            entry = bboxes_data[stem]
            img_w = entry.get("width", 0)
            img_h = entry.get("height", 0)
            if img_w > 0 and img_h > 0:
                label_lines = bbox_from_json(entry, img_w, img_h)
            if label_lines:
                labels_from_json += 1

        # Fallback: calcular desde máscara
        if not label_lines and mask_path is not None and mask_path.exists():
            label_lines = bbox_from_mask(mask_path)
            if label_lines:
                labels_from_mask += 1

        if not label_lines:
            labels_failed += 1
            print(f"  ⚠ Sin bbox válida para: {img_path.name}")

        positives.append((img_path, mask_path, label_lines, new_stem))

    print("▸ Etiquetas positivas generadas:")
    print(f"  ✓ Desde JSON: {labels_from_json}")
    print(f"  ✓ Desde máscara (fallback): {labels_from_mask}")
    if labels_failed > 0:
        print(f"  ⚠ Sin etiqueta válida: {labels_failed}")
    print()

    negatives: list[tuple[Path, Optional[Path], list[str], str]] = []
    for idx, img_path in enumerate(neg_images, start=1):
        new_stem = f"neg_{idx:04d}"
        # Las negativas tienen label vacío
        negatives.append((img_path, None, [], new_stem))

    # ── 5. Split estratificado ──
    print("▸ Dividiendo en train/val/test...")
    pos_train_idx, pos_val_idx, pos_test_idx = split_indices(len(positives), RANDOM_SEED)
    neg_train_idx, neg_val_idx, neg_test_idx = split_indices(len(negatives), RANDOM_SEED + 1)

    splits: dict[str, list[tuple[Path, Optional[Path], list[str], str]]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    for idx in pos_train_idx:
        splits["train"].append(positives[idx])
    for idx in pos_val_idx:
        splits["val"].append(positives[idx])
    for idx in pos_test_idx:
        splits["test"].append(positives[idx])
    for idx in neg_train_idx:
        splits["train"].append(negatives[idx])
    for idx in neg_val_idx:
        splits["val"].append(negatives[idx])
    for idx in neg_test_idx:
        splits["test"].append(negatives[idx])

    # Conteos
    for split_name, items in splits.items():
        n_pos = sum(1 for _, _, lbl, s in items if s.startswith("pos_"))
        n_neg = sum(1 for _, _, lbl, s in items if s.startswith("neg_"))
        print(f"  {split_name:>5s}: {len(items)} total  "
              f"({n_pos} positivas + {n_neg} negativas)")
    print()

    # ── 6. Crear estructura de directorios ──
    output_dir = project_root / OUTPUT_DIR_NAME
    print(f"▸ Creando estructura en: {output_dir}")
    create_dirs(output_dir)
    print()

    # ── 7. Copiar archivos ──
    print("▸ Copiando archivos...")
    counters = {
        "images": 0,
        "labels": 0,
        "masks": 0,
    }

    for split_name, items in splits.items():
        for img_path, mask_path, label_lines, new_stem in items:
            img_ext = img_path.suffix.lower()

            # Copiar imagen
            dst_img = output_dir / "images" / split_name / f"{new_stem}{img_ext}"
            if not DRY_RUN:
                shutil.copy2(img_path, dst_img)
            counters["images"] += 1

            # Escribir label (.txt)
            dst_label = output_dir / "labels" / split_name / f"{new_stem}.txt"
            if not DRY_RUN:
                dst_label.write_text("\n".join(label_lines), encoding="utf-8")
            counters["labels"] += 1

            # Copiar máscara (solo positivas)
            if mask_path is not None and mask_path.exists():
                mask_ext = mask_path.suffix.lower()
                dst_mask = output_dir / "masks" / split_name / f"{new_stem}{mask_ext}"
                if not DRY_RUN:
                    shutil.copy2(mask_path, dst_mask)
                counters["masks"] += 1

    if DRY_RUN:
        print("  [DRY_RUN] No se copiaron archivos.")
    else:
        print(f"  ✓ Imágenes copiadas: {counters['images']}")
        print(f"  ✓ Labels creados:    {counters['labels']}")
        print(f"  ✓ Máscaras copiadas: {counters['masks']}")
    print()

    # ── 8. Generar data.yaml ──
    print("▸ Generando data.yaml...")
    write_data_yaml(output_dir)
    print()

    # ── 9. Validaciones finales ──
    print("▸ Validaciones finales...")
    errors = 0
    for split_name in ("train", "val", "test"):
        imgs = list((output_dir / "images" / split_name).iterdir()) if not DRY_RUN else []
        lbls = list((output_dir / "labels" / split_name).iterdir()) if not DRY_RUN else []
        if not DRY_RUN:
            img_stems = {p.stem for p in imgs}
            lbl_stems = {p.stem for p in lbls}
            missing_labels = img_stems - lbl_stems
            missing_images = lbl_stems - img_stems
            if missing_labels:
                print(f"  ⚠ {split_name}: {len(missing_labels)} imágenes sin label")
                errors += len(missing_labels)
            if missing_images:
                print(f"  ⚠ {split_name}: {len(missing_images)} labels sin imagen")
                errors += len(missing_images)
            print(f"  {split_name:>5s}: {len(imgs)} imágenes, {len(lbls)} labels ✓")
    if errors == 0 and not DRY_RUN:
        print("  ✓ Todas las validaciones pasaron correctamente.")
    print()

    # ── 10. Resumen final ──
    total = len(positives) + len(negatives)
    print("=" * 60)
    print("  RESUMEN FINAL")
    print("=" * 60)
    print(f"  Kvasir general:       {kvasir_general}")
    print(f"  Kvasir-SEG:           {kvasir_seg}")
    print(f"  Positivas:            {len(positives)}")
    print(f"  Negativas:            {len(negatives)}")
    print(f"  Total:                {total}")
    print("  ─────────────────────────────────")
    for split_name, items in splits.items():
        n_pos = sum(1 for _, _, _, s in items if s.startswith("pos_"))
        n_neg = sum(1 for _, _, _, s in items if s.startswith("neg_"))
        print(f"  {split_name:>5s}: {len(items):>4d}  "
              f"({n_pos} pos + {n_neg} neg)")
    print("  ─────────────────────────────────")
    print(f"  Imágenes copiadas:    {counters['images']}")
    print(f"  Labels creados:       {counters['labels']}")
    print(f"  Máscaras copiadas:    {counters['masks']}")
    print(f"  Labels desde JSON:    {labels_from_json}")
    print(f"  Labels desde máscara: {labels_from_mask}")
    print("  ─────────────────────────────────")
    print(f"  Dataset generado en:  {output_dir.resolve()}")
    print(f"  DRY_RUN:              {DRY_RUN}")
    print("=" * 60)


if __name__ == "__main__":
    main()
