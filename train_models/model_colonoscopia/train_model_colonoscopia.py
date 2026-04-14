"""Entrena el modelo YOLOv8 de colonoscopia y copia el mejor peso a models/."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

# Configuración principal del entrenamiento.
YOLO_BASE_MODEL: str = "yolov8s.pt"
EPOCHS: int = 100
BATCH_SIZE: int = 8
IMAGE_SIZE: int = 640
LR0: float = 0.01
LRF: float = 0.01
OPTIMIZER: str = "SGD"
MOMENTUM: float = 0.937
WEIGHT_DECAY: float = 0.0005
WARMUP_EPOCHS: float = 3.0
PATIENCE: int = 50

# Augmentations usadas por YOLO.
HSV_H: float = 0.015
HSV_S: float = 0.7
HSV_V: float = 0.4
TRANSLATE: float = 0.1
SCALE: float = 0.5
FLIPLR: float = 0.5
FLIPUD: float = 0.5
MOSAIC: float = 1.0
CLOSE_MOSAIC: int = 10
MIXUP: float = 0.1

# Rutas relativas a la raíz del proyecto.
DATASET_YAML: str = "data/dataset_yolo/data.yaml"
TRAIN_PROJECT: str = "train_models/model_colonoscopia"
TRAIN_NAME: str = "entrenamiento"
OUTPUT_MODEL: str = "models/colonoscopy.pt"


def validate_dataset(project_root: Path, yaml_path: Path) -> bool:
    """Comprueba que el dataset mínimo para entrenar existe."""
    print("▸ Validando dataset...")

    if not yaml_path.exists():
        print(f"  ✗ No se encontró {yaml_path}")
        print("    Ejecuta primero: uv run python prepare_dataset.py")
        return False
    print(f"  ✓ data.yaml encontrado: {yaml_path}")

    dataset_dir = yaml_path.parent
    required_dirs = ["images/train", "images/val", "labels/train", "labels/val"]

    for rel_dir in required_dirs:
        directory = dataset_dir / rel_dir
        if not directory.exists() or not directory.is_dir():
            print(f"  ✗ Carpeta faltante: {directory}")
            return False
        print(f"  ✓ {rel_dir}: {len(list(directory.iterdir()))} archivos")

    train_imgs = list((dataset_dir / "images" / "train").iterdir())
    if len(train_imgs) < 10:
        print(f"  ⚠ Solo {len(train_imgs)} imágenes en train")
        return False

    for split in ("train", "val"):
        img_stems = {p.stem for p in (dataset_dir / "images" / split).iterdir()}
        lbl_stems = {p.stem for p in (dataset_dir / "labels" / split).iterdir()}
        missing = img_stems - lbl_stems
        if missing:
            print(f"  ⚠ {split}: {len(missing)} imágenes sin label")

    print("  ✓ Dataset válido")
    print()
    return True


def train(project_root: Path) -> Path | None:
    """Ejecuta el entrenamiento y devuelve el mejor peso disponible."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("✗ ERROR: ultralytics no está instalado.")
        print("  Instala con: uv add ultralytics")
        sys.exit(1)

    yaml_path = project_root / DATASET_YAML

    print(f"▸ Cargando modelo base: {YOLO_BASE_MODEL}")
    print("  (si es la primera vez, se descargará automáticamente)")
    model = YOLO(YOLO_BASE_MODEL)
    print()

    print("▸ Iniciando entrenamiento...")
    print("=" * 60)
    print(f"  Base:          {YOLO_BASE_MODEL}")
    print(f"  Epochs:        {EPOCHS}")
    print(f"  Batch size:    {BATCH_SIZE}")
    print(f"  Image size:    {IMAGE_SIZE}")
    print(f"  Optimizer:     {OPTIMIZER}")
    print(f"  LR inicial:    {LR0}")
    print(f"  LR final:      lr0 × {LRF} = {LR0 * LRF}")
    print(f"  Patience:      {PATIENCE} epochs")
    print(f"  Dataset:       {yaml_path}")
    print("=" * 60)
    print()

    model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMAGE_SIZE,
        patience=PATIENCE,
        optimizer=OPTIMIZER,
        lr0=LR0,
        lrf=LRF,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        warmup_epochs=WARMUP_EPOCHS,
        hsv_h=HSV_H,
        hsv_s=HSV_S,
        hsv_v=HSV_V,
        translate=TRANSLATE,
        scale=SCALE,
        fliplr=FLIPLR,
        flipud=FLIPUD,
        mosaic=MOSAIC,
        close_mosaic=CLOSE_MOSAIC,
        mixup=MIXUP,
        project=str(project_root / TRAIN_PROJECT),
        name=TRAIN_NAME,
        exist_ok=True,
        verbose=True,
        device="",
        workers=8,
        seed=42,
        save=True,
        plots=True,
    )

    train_dir = project_root / TRAIN_PROJECT / TRAIN_NAME
    best_pt = train_dir / "weights" / "best.pt"
    if best_pt.exists():
        print()
        print(f"  ✓ Mejor modelo guardado en: {best_pt}")
        return best_pt

    print()
    print("  ⚠ No se encontró best.pt, revisando last.pt...")
    last_pt = train_dir / "weights" / "last.pt"
    if last_pt.exists():
        print(f"  ✓ Usando last.pt: {last_pt}")
        return last_pt

    print("  ✗ No se encontraron pesos entrenados.")
    return None


def copy_model_to_output(best_pt: Path, project_root: Path) -> None:
    """Copia el peso elegido a la ruta final usada por la app."""
    output = project_root / OUTPUT_MODEL
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_pt, output)
    print(f"  ✓ Modelo copiado a: {output}")
    print("    detect_realtime.py lo usará directamente desde ahí.")


def print_results_summary(project_root: Path) -> None:
    """Lista los archivos importantes del entrenamiento."""
    train_dir = project_root / TRAIN_PROJECT / TRAIN_NAME

    print()
    print("═" * 60)
    print("  ARCHIVOS GENERADOS")
    print("═" * 60)

    important_files = {
        "weights/best.pt": "Mejor modelo",
        "weights/last.pt": "Último checkpoint",
        "results.csv": "Métricas por epoch",
        "results.png": "Curvas principales",
        "confusion_matrix.png": "Matriz de confusión",
        "confusion_matrix_normalized.png": "Matriz normalizada",
        "F1_curve.png": "Curva F1",
        "P_curve.png": "Curva Precision",
        "R_curve.png": "Curva Recall",
        "PR_curve.png": "Curva Precision-Recall",
        "val_batch0_pred.jpg": "Predicciones de validación",
        "val_batch0_labels.jpg": "Labels de validación",
        "args.yaml": "Parámetros usados",
    }

    for rel_path, description in important_files.items():
        full_path = train_dir / rel_path
        exists = "✓" if full_path.exists() else "✗"
        print(f"  {exists} {rel_path}")
        print(f"      → {description}")

    print()
    print(f"  Carpeta completa: {train_dir}")
    print("═" * 60)


def main() -> None:
    """Valida dataset, entrena, copia el modelo y muestra el resumen."""
    print()
    print("═" * 60)
    print("  ENTRENAMIENTO YOLO — DETECCIÓN DE PÓLIPOS")
    print("  Modelo: Colonoscopia (Fase 2)")
    print("═" * 60)
    print()

    project_root = Path(__file__).resolve().parent.parent.parent
    yaml_path = project_root / DATASET_YAML

    print(f"Raíz del proyecto: {project_root}")
    print(f"Dataset YAML:      {yaml_path}")
    print()

    if not validate_dataset(project_root, yaml_path):
        print()
        print("✗ Dataset no válido. Ejecuta primero:")
        print("    uv run python prepare_dataset.py")
        sys.exit(1)

    best_pt = train(project_root)
    if best_pt is None:
        print()
        print("✗ El entrenamiento falló. Revisa los logs de arriba.")
        sys.exit(1)

    print()
    print("▸ Copiando modelo entrenado a carpeta de producción...")
    copy_model_to_output(best_pt, project_root)
    print_results_summary(project_root)

    print()
    print("✓ Entrenamiento completado.")
    print()
    print("Próximos pasos:")
    print("  1. Revisa las gráficas generadas")
    print("  2. Ajusta epochs, modelo base o augmentations si hace falta")
    print("  3. Ejecuta la app: uv run python detect_realtime.py")
    print()


if __name__ == "__main__":
    main()
