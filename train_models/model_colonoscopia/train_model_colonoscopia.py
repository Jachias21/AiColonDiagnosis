"""
train_model_colonoscopia.py
===========================
Script para entrenar un modelo YOLOv8 de detección de pólipos en colonoscopia.

Usa el dataset generado por `prepare_dataset.py`, que produce:
    dataset_yolo/
        images/  train/ val/ test/
        labels/  train/ val/ test/
        data.yaml          ← archivo de configuración del dataset

El modelo entrenado se copia automáticamente a `models/colonoscopy.pt`
para que `detect_realtime.py` lo encuentre sin cambiar nada.

Ejecutar desde la raíz del proyecto:
    uv run python train_models/model_colonoscopia/train_model_colonoscopia.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

# ══════════════════════════════════════════════════════════════
# CONFIGURACIÓN  —  Ajusta estos valores antes de entrenar
# ══════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────
# MODELO BASE
# ──────────────────────────────────────────────
# YOLOv8 viene en varios tamaños. Cuanto más grande, más preciso pero más lento:
#
#   yolov8n.pt  →  Nano    (~3.2M params)   Muy rápido, menor precisión
#   yolov8s.pt  →  Small   (~11.2M params)  Buen balance velocidad/precisión
#   yolov8m.pt  →  Medium  (~25.9M params)  Más preciso, más lento
#   yolov8l.pt  →  Large   (~43.7M params)  Alta precisión
#   yolov8x.pt  →  XLarge  (~68.2M params)  Máxima precisión, muy lento
#
# Para empezar usamos "s" (Small): es el mejor balance para un dataset de ~2000
# imágenes. Con datasets pequeños, modelos muy grandes tienden a sobreajustar
# (memorizan en vez de aprender). "s" aprende patrones generales sin saturarse.
YOLO_BASE_MODEL: str = "yolov8s.pt"

# ──────────────────────────────────────────────
# HIPERPARÁMETROS DE ENTRENAMIENTO
# ──────────────────────────────────────────────

# EPOCHS (número de pasadas completas sobre todo el dataset)
# Cada epoch el modelo ve TODAS las imágenes una vez. Con pocas epochs (10-20)
# el modelo no aprende lo suficiente (subajuste). Con muchas (300+) puede memorizar
# el dataset (sobreajuste). 100 epochs es un punto de partida razonable para ~2000
# imágenes; YOLOv8 tiene "early stopping" que para automáticamente si deja de mejorar.
EPOCHS: int = 100

# BATCH SIZE (imágenes procesadas a la vez)
# Valores comunes: 8, 16, 32.
# - Mayor batch → entrenamiento más estable y rápido, pero usa más VRAM.
# - Menor batch → menos memoria, pero gradientes más ruidosos.
# Con 8 GB VRAM: batch=16 funciona bien. Con 6 GB (RTX 4050): batch=8.
# Usa -1 para que YOLO lo calcule automáticamente según tu GPU.
BATCH_SIZE: int = 8

# IMAGE SIZE (resolución de entrada al modelo)
# Las imágenes se redimensionan a este tamaño antes de pasar al modelo.
# 640 es el estándar de YOLOv8. Más grande (1280) → más detalle pero más lento
# y usa mucha más VRAM. 640 es óptimo para la mayoría de casos con pólipos,
# ya que los pólipos suelen ser objetos medianos-grandes en la imagen.
IMAGE_SIZE: int = 640

# LEARNING RATE INICIAL (lr0)
# Controla cuánto cambian los pesos en cada paso. Muy alto (0.1) → el modelo
# oscila y no converge. Muy bajo (0.0001) → aprende extremadamente lento.
# 0.01 es el default de YOLO y funciona bien con el scheduler cosine.
LR0: float = 0.01

# LEARNING RATE FINAL (lrf)
# Factor multiplicador del lr al final del entrenamiento. lrf=0.01 significa
# que al terminar, el lr será lr0 × 0.01 = 0.0001. Esto permite afinar los
# últimos detalles con pasos muy pequeños sin destruir lo aprendido.
LRF: float = 0.01

# OPTIMIZER
# - "SGD": Stochastic Gradient Descent con momentum. Más estable, convergencia
#   predecible. Es el default de YOLO y el más probado en detección de objetos.
# - "Adam": Adaptativo, converge más rápido al inicio pero puede sobreajustar
#   en datasets pequeños. Usar si SGD no mejora tras muchas epochs.
# - "AdamW": Como Adam pero con weight decay desacoplado, mejor regularización.
OPTIMIZER: str = "SGD"

# MOMENTUM (solo aplica con SGD)
# Simula inercia: el optimizador "recuerda" la dirección de los últimos pasos
# y no cambia bruscamente. 0.937 es el default de YOLO — más alto (0.98) da
# más estabilidad pero menos agilidad para cambiar de dirección.
MOMENTUM: float = 0.937

# WEIGHT DECAY (regularización L2)
# Penaliza pesos muy grandes para evitar sobreajuste. 0.0005 es el estándar.
# Valores más altos (0.001) fuerzan modelos más simples = menos sobreajuste
# pero posiblemente menor precisión. Con datasets pequeños (~2000 imgs) un
# valor de 0.0005 a 0.001 es ideal.
WEIGHT_DECAY: float = 0.0005

# WARMUP (calentamiento)
# Al iniciar, el lr sube gradualmente durante WARMUP_EPOCHS epochs.
# ¿Por qué? Al principio los pesos son aleatorios y los gradientes caóticos.
# Si aplicas el lr completo desde el primer paso, el modelo puede "explotar"
# (los pesos divergen). El warmup evita esto subiendo el lr poco a poco.
WARMUP_EPOCHS: float = 3.0

# PATIENCE (paciencia para early stopping)
# Si durante PATIENCE epochs consecutivas la métrica de validación no mejora,
# el entrenamiento se detiene automáticamente. Evita gastar tiempo y sobreajuste.
# 50 epochs de paciencia es generoso — da tiempo al modelo para superar "mesetas"
# donde parece estancado pero luego mejora.
PATIENCE: int = 50

# ──────────────────────────────────────────────
# DATA AUGMENTATION (aumento de datos)
# ──────────────────────────────────────────────
# YOLOv8 aplica augmentations automáticamente. Estas son las que controlamos:

# HSV (tono, saturación, valor)
# Varía aleatoriamente los colores de cada imagen. Esto obliga al modelo a no
# depender de colores exactos — importante porque la iluminación del endoscopio
# varía mucho de un paciente a otro y el tono del tejido cambia.
HSV_H: float = 0.015    # Variación de tono (hue). 0.015 = ±1.5% del espectro
HSV_S: float = 0.7      # Variación de saturación. 0.7 = hasta ±70%
HSV_V: float = 0.4      # Variación de brillo (value). 0.4 = hasta ±40%

# TRANSLATE (desplazamiento)
# Mueve la imagen aleatoriamente. 0.1 = hasta un 10% del ancho/alto.
# Simula que el pólipo no siempre está centrado en el encuadre.
TRANSLATE: float = 0.1

# SCALE (escala)
# Zoom in/out aleatorio. 0.5 = escala entre 0.5x y 1.5x.
# Los pólipos varían mucho de tamaño según la distancia del endoscopio.
SCALE: float = 0.5

# FLIPLR (volteo horizontal)
# Probabilidad de voltear la imagen horizontalmente. 0.5 = 50% de las veces.
# Tiene sentido porque un pólipo puede aparecer en cualquier lado del colon.
FLIPLR: float = 0.5

# FLIPUD (volteo vertical)
# Probabilidad de voltear verticalmente. En colonoscopia no hay "arriba/abajo"
# definido (el endoscopio rota), así que es válido usarlo.
FLIPUD: float = 0.5

# MOSAIC (mosaico)
# Combina 4 imágenes en una sola. Obliga al modelo a detectar objetos en
# diferentes contextos, tamaños y posiciones. Muy efectivo pero puede ser
# problemático si los objetos son muy grandes. 1.0 = activo siempre.
# Se desactiva automáticamente en las últimas 10 epochs (close_mosaic).
MOSAIC: float = 1.0

# CLOSE_MOSAIC (desactivar mosaico al final)
# Desactiva mosaic en las últimas N epochs. En las fases finales del
# entrenamiento, queremos que el modelo vea imágenes "normales" (sin mezclar)
# para afinar las detecciones en condiciones reales.
CLOSE_MOSAIC: int = 10

# MIXUP
# Mezcla dos imágenes con transparencia. Poco útil en detección médica porque
# crea escenas irreales. Lo dejamos bajo (0.1) para algo de variación sin
# distorsionar demasiado.
MIXUP: float = 0.1

# ──────────────────────────────────────────────
# RUTAS
# ──────────────────────────────────────────────

# Ruta al data.yaml generado por prepare_dataset.py
# Se resuelve relativa a la raíz del proyecto (2 niveles arriba de este script)
DATASET_YAML: str = "data/dataset_yolo/data.yaml"

# Carpeta donde YOLO guarda los resultados del entrenamiento
# (pesos, métricas, gráficas, matriz de confusión, etc.)
TRAIN_PROJECT: str = "train_models/model_colonoscopia"
TRAIN_NAME: str = "entrenamiento"

# Ruta final donde copiar el mejor modelo para que detect_realtime.py lo use
OUTPUT_MODEL: str = "models/colonoscopy.pt"


# ══════════════════════════════════════════════════════════════
# FUNCIONES
# ══════════════════════════════════════════════════════════════

def validate_dataset(project_root: Path, yaml_path: Path) -> bool:
    """
    Verifica que el dataset existe y tiene la estructura correcta.
    Es importante validar ANTES de entrenar para no perder tiempo
    si falta algo.
    """
    print("▸ Validando dataset...")

    if not yaml_path.exists():
        print(f"  ✗ No se encontró {yaml_path}")
        print("    Ejecuta primero: uv run python prepare_dataset.py")
        return False
    print(f"  ✓ data.yaml encontrado: {yaml_path}")

    dataset_dir = yaml_path.parent

    # Verificar que existen las carpetas de imágenes y labels
    required_dirs = [
        "images/train", "images/val",
        "labels/train", "labels/val",
    ]
    for rel_dir in required_dirs:
        d = dataset_dir / rel_dir
        if not d.exists() or not d.is_dir():
            print(f"  ✗ Carpeta faltante: {d}")
            return False
        count = len(list(d.iterdir()))
        print(f"  ✓ {rel_dir}: {count} archivos")

    # Verificar que train tiene imágenes suficientes
    train_imgs = list((dataset_dir / "images" / "train").iterdir())
    if len(train_imgs) < 10:
        print(f"  ⚠ Solo {len(train_imgs)} imágenes en train — muy pocas para entrenar")
        return False

    # Verificar que labels y images están alineados
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
    """
    Ejecuta el entrenamiento de YOLOv8.
    Devuelve la ruta al mejor modelo (best.pt) o None si falla.
    """
    # ── Importar ultralytics aquí (no arriba) ──
    # ¿Por qué? Porque ultralytics tarda unos segundos en cargar y queremos
    # que las validaciones del dataset fallen rápido sin esperar ese import.
    # También porque así el script puede mostrar un error claro si ultralytics
    # no está instalado.
    try:
        from ultralytics import YOLO
    except ImportError:
        print("✗ ERROR: ultralytics no está instalado.")
        print("  Instala con: uv add ultralytics")
        sys.exit(1)

    yaml_path = project_root / DATASET_YAML

    # ── Cargar modelo base ──
    # YOLO descarga automáticamente el modelo preentrenado en COCO (80 clases).
    # Este modelo ya sabe detectar objetos genéricos (personas, coches, etc.).
    # Al entrenar con nuestro dataset, "transfiere" ese conocimiento al dominio
    # médico (pólipos). Esto se llama TRANSFER LEARNING y es mucho más efectivo
    # que entrenar desde cero, especialmente con datasets pequeños (~2000 imgs).
    print(f"▸ Cargando modelo base: {YOLO_BASE_MODEL}")
    print("  (si es la primera vez, se descargará automáticamente)")
    model = YOLO(YOLO_BASE_MODEL)
    print()

    # ── Entrenar ──
    # model.train() ejecuta todo el pipeline:
    #   1. Carga el dataset según data.yaml
    #   2. Aplica augmentations en cada batch
    #   3. Forward pass → calcula predicciones
    #   4. Calcula loss (box_loss + cls_loss + dfl_loss)
    #   5. Backward pass → gradientes
    #   6. Optimizer step → actualiza pesos
    #   7. Cada epoch evalúa en val y guarda métricas
    #   8. Guarda best.pt (mejor mAP) y last.pt (última epoch)
    print("▸ Iniciando entrenamiento...")
    print("=" * 60)
    print(f"  Base:          {YOLO_BASE_MODEL}")
    print(f"  Epochs:        {EPOCHS}")
    print(f"  Batch size:    {BATCH_SIZE}")
    print(f"  Image size:    {IMAGE_SIZE}")
    print(f"  Optimizer:     {OPTIMIZER}")
    print(f"  LR inicial:    {LR0}")
    print(f"  LR final:      lr0 × {LRF} = {LR0 * LRF}")
    print(f"  Patience:      {PATIENCE} epochs (early stopping)")
    print(f"  Dataset:       {yaml_path}")
    print("=" * 60)
    print()

    model.train(
        # ── Dataset ──
        data=str(yaml_path),

        # ── Entrenamiento ──
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMAGE_SIZE,
        patience=PATIENCE,

        # ── Optimizador ──
        optimizer=OPTIMIZER,
        lr0=LR0,
        lrf=LRF,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        warmup_epochs=WARMUP_EPOCHS,

        # ── Data Augmentation ──
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

        # ── Salida ──
        # project: carpeta padre donde YOLO crea la subcarpeta del run
        # name: nombre de la subcarpeta (entrenamiento/)
        project=str(project_root / TRAIN_PROJECT),
        name=TRAIN_NAME,

        # exist_ok=True → si ya existe la carpeta, la sobreescribe
        # en vez de crear "entrenamiento2", "entrenamiento3", etc.
        exist_ok=True,

        # ── Otros ──
        # verbose: muestra progreso detallado en consola
        verbose=True,

        # device: qué hardware usar
        # "" → YOLO autodetecta (usa GPU si hay CUDA, sino CPU)
        # Puedes forzar: device="0" (GPU 0), device="cpu"
        device="",

        # workers: hilos para cargar datos. Más = más rápido pero más RAM.
        # 8 es buen balance. En Windows a veces causa problemas; si falla,
        # reduce a 0 o 2.
        workers=8,

        # seed: semilla para reproducibilidad
        seed=42,

        # save: guardar checkpoints (best.pt y last.pt)
        save=True,

        # plots: genera gráficas de métricas, matriz de confusión, etc.
        # Muy útil para analizar el rendimiento y saber si hay sobreajuste.
        plots=True,
    )

    # ── Localizar best.pt ──
    # YOLO guarda los pesos en: {project}/{name}/weights/best.pt
    train_dir = project_root / TRAIN_PROJECT / TRAIN_NAME
    best_pt = train_dir / "weights" / "best.pt"

    if best_pt.exists():
        print()
        print(f"  ✓ Mejor modelo guardado en: {best_pt}")
        return best_pt
    else:
        print()
        print("  ⚠ No se encontró best.pt — revisando last.pt...")
        last_pt = train_dir / "weights" / "last.pt"
        if last_pt.exists():
            print(f"  ✓ Usando last.pt: {last_pt}")
            return last_pt
        print("  ✗ No se encontraron pesos entrenados.")
        return None


def copy_model_to_output(best_pt: Path, project_root: Path) -> None:
    """
    Copia el mejor modelo a la carpeta de models/ del proyecto.
    Así detect_realtime.py lo encuentra automáticamente sin tener
    que cambiar rutas ni buscar en subcarpetas de entrenamiento.
    """
    output = project_root / OUTPUT_MODEL
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_pt, output)
    print(f"  ✓ Modelo copiado a: {output}")
    print("    detect_realtime.py lo usará directamente desde ahí.")


def print_results_summary(project_root: Path) -> None:
    """
    Muestra un resumen de los archivos generados tras el entrenamiento.
    YOLO genera muchos archivos útiles para entender el rendimiento.
    """
    train_dir = project_root / TRAIN_PROJECT / TRAIN_NAME

    print()
    print("═" * 60)
    print("  ARCHIVOS GENERADOS")
    print("═" * 60)

    important_files = {
        "weights/best.pt": "Mejor modelo (mayor mAP en validación)",
        "weights/last.pt": "Modelo de la última epoch",
        "results.csv": "Métricas por epoch (loss, mAP, precision, recall)",
        "results.png": "Gráfica de evolución de las métricas",
        "confusion_matrix.png": "Matriz de confusión (predicho vs real)",
        "confusion_matrix_normalized.png": "Matriz de confusión normalizada",
        "F1_curve.png": "Curva F1 vs confianza",
        "P_curve.png": "Curva Precision vs confianza",
        "R_curve.png": "Curva Recall vs confianza",
        "PR_curve.png": "Curva Precision-Recall",
        "val_batch0_pred.jpg": "Predicciones en ejemplos de validación",
        "val_batch0_labels.jpg": "Labels reales de validación (referencia)",
        "args.yaml": "Todos los hiperparámetros usados (para reproducir)",
    }

    for rel_path, description in important_files.items():
        full_path = train_dir / rel_path
        exists = "✓" if full_path.exists() else "✗"
        print(f"  {exists} {rel_path}")
        print(f"      → {description}")

    print()
    print(f"  Carpeta completa: {train_dir}")
    print("═" * 60)


# ══════════════════════════════════════════════════════════════
# EJECUCIÓN
# ══════════════════════════════════════════════════════════════

def main() -> None:
    print()
    print("═" * 60)
    print("  ENTRENAMIENTO YOLO — DETECCIÓN DE PÓLIPOS")
    print("  Modelo: Colonoscopia (Fase 2)")
    print("═" * 60)
    print()

    # ── Resolver la raíz del proyecto ──
    # Este script está en train_models/model_colonoscopia/
    # así que la raíz del proyecto está 2 niveles arriba.
    project_root = Path(__file__).resolve().parent.parent.parent
    print(f"Raíz del proyecto: {project_root}")

    yaml_path = project_root / DATASET_YAML
    print(f"Dataset YAML:      {yaml_path}")
    print()

    # ── Paso 1: Validar que el dataset existe ──
    if not validate_dataset(project_root, yaml_path):
        print()
        print("✗ Dataset no válido. Ejecuta primero:")
        print("    uv run python prepare_dataset.py")
        sys.exit(1)

    # ── Paso 2: Entrenar ──
    best_pt = train(project_root)

    if best_pt is None:
        print()
        print("✗ El entrenamiento falló. Revisa los logs de arriba.")
        sys.exit(1)

    # ── Paso 3: Copiar modelo a models/ ──
    print()
    print("▸ Copiando modelo entrenado a carpeta de producción...")
    copy_model_to_output(best_pt, project_root)

    # ── Paso 4: Resumen ──
    print_results_summary(project_root)

    print()
    print("✓ Entrenamiento completado.")
    print()
    print("Próximos pasos:")
    print("  1. Revisa las gráficas generadas para evaluar el rendimiento")
    print("  2. Si el mAP es bajo, prueba:")
    print("     - Aumentar EPOCHS (150-200)")
    print("     - Cambiar modelo base a yolov8m.pt")
    print("     - Ajustar augmentations")
    print("  3. Ejecuta la app: uv run python detect_realtime.py")
    print()


if __name__ == "__main__":
    main()
