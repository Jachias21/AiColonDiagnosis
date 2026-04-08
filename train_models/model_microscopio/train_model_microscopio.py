"""
train_model_microscopio.py
==========================
Competición de modelos de clasificación para detectar cáncer de colon
en imágenes de microscopio (histopatología).

Entrena secuencialmente varios modelos con transfer learning (ImageNet)
y genera un ranking comparativo al final.

Modelos en competición:
  1. EfficientNet-B0   — Ligero y eficiente, muy usado en médico
  2. EfficientNet-B1   — Un poco más grande que B0, más precisión
  3. ResNet-50         — Clásico robusto, referencia en papers
  4. ConvNeXt-Tiny     — CNN moderna, competitivo con transformers
  5. DenseNet-121      — Reutiliza features, popular en patología
  6. ViT-S (Small)     — Vision Transformer, patrones globales

Dataset:
  data/dataset_colon/
    train/ (3500 colon_aca + 3500 colon_n)
    val/   (750 + 750)
    test/  (750 + 750)

Uso:
    uv run python train_models/model_microscopio/train_model_microscopio.py

Tras ejecutar se genera:
  - train_models/model_microscopio/resultados/  (pesos, métricas, gráficas)
  - models/microscopy.pt  (copia del mejor modelo del ranking)
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
import time

# Forzar UTF-8 en stdout/stderr para Windows (evita UnicodeEncodeError con cp1252)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from dataclasses import dataclass, field
from pathlib import Path

import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

# ══════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════

# ── Rutas ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "dataset_colon"
OUTPUT_DIR = Path(__file__).resolve().parent / "resultados"
MODELS_DIR = PROJECT_ROOT / "models"

# ── Hiperparámetros ──
# Número de épocas para entrenamiento final.
# Para entrenar "bien" un único modelo con transfer learning, 20 épocas
# suele dar mejor convergencia que 8-10.
EPOCHS: int = 20

# --- Selección de modelo ---
# Dejamos solo DenseNet-121 porque ya validaste que rinde muy bien.
# Si quieres competición completa, cambia ONLY_MODEL = None.
ONLY_MODEL: str | None = "DenseNet-121"
OVERRIDE_EPOCHS: int | None = None

# Batch size: 32 cabe bien en 6GB VRAM para todos estos modelos.
# EfficientNet y ResNet usan ~2-3GB, ViT un poco más (~3-4GB).
BATCH_SIZE: int = 32

# Learning rate: 1e-3 para la cabeza nueva (capa clasificadora).
# Las capas congeladas (backbone) no se actualizan, así que un lr
# relativamente alto para la cabeza está bien.
LEARNING_RATE: float = 1e-3

# Máximo de imágenes por clase en train.
# None = usar TODAS las imágenes disponibles (entrenamiento final).
MAX_TRAIN_PER_CLASS: int | None = None

# Tamaño de imagen: 224x224 es el estándar para todos estos modelos.
# Fueron pre-entrenados con este tamaño en ImageNet.
IMAGE_SIZE: int = 224

# Workers para cargar datos en paralelo.
# En Windows, 2 workers + persistent_workers=True es más rápido que 4,
# porque evita reiniciar los procesos de carga entre épocas.
NUM_WORKERS: int = 2

# Nombres de clases (orden alfabético, como lo lee ImageFolder)
CLASS_NAMES: list[str] = ["colon_aca", "colon_n"]
# colon_aca = adenocarcinoma (cáncer)
# colon_n   = normal (sano)


# ══════════════════════════════════════════════
# DEFINICIÓN DE MODELOS
# ══════════════════════════════════════════════

@dataclass
class ModelConfig:
    """Configuración de un modelo en la competición."""
    name: str               # Nombre corto para mostrar
    build_fn: str           # Nombre de la función en torchvision.models
    weights: str            # Nombre de los pesos pre-entrenados
    classifier_attr: str    # Atributo que contiene la capa clasificadora
    in_features_path: str   # Cómo acceder a in_features de la última capa
    description: str        # Descripción breve para el ranking


# Cada modelo necesita saber:
# 1. Cómo construirlo (build_fn)
# 2. Qué pesos usar (weights)
# 3. Dónde está la cabeza clasificadora para reemplazarla (classifier_attr)
COMPETITION_MODELS: list[ModelConfig] = [
    ModelConfig(
        name="EfficientNet-B0",
        build_fn="efficientnet_b0",
        weights="EfficientNet_B0_Weights.IMAGENET1K_V1",
        classifier_attr="classifier",
        in_features_path="classifier.1.in_features",
        description="Ligero y eficiente (5.3M params)",
    ),
    ModelConfig(
        name="EfficientNet-B1",
        build_fn="efficientnet_b1",
        weights="EfficientNet_B1_Weights.IMAGENET1K_V1",
        classifier_attr="classifier",
        in_features_path="classifier.1.in_features",
        description="Versión más grande de B0 (7.8M params)",
    ),
    ModelConfig(
        name="ResNet-50",
        build_fn="resnet50",
        weights="ResNet50_Weights.IMAGENET1K_V1",
        classifier_attr="fc",
        in_features_path="fc.in_features",
        description="Referencia clásica en papers (25.6M params)",
    ),
    ModelConfig(
        name="ConvNeXt-Tiny",
        build_fn="convnext_tiny",
        weights="ConvNeXt_Tiny_Weights.IMAGENET1K_V1",
        classifier_attr="classifier",
        in_features_path="classifier.2.in_features",
        description="CNN moderna estilo transformer (28.6M params)",
    ),
    ModelConfig(
        name="DenseNet-121",
        build_fn="densenet121",
        weights="DenseNet121_Weights.IMAGENET1K_V1",
        classifier_attr="classifier",
        in_features_path="classifier.in_features",
        description="Reutiliza features entre capas (8.0M params)",
    ),
    ModelConfig(
        name="ViT-B/16",
        build_fn="vit_b_16",
        weights="ViT_B_16_Weights.IMAGENET1K_V1",
        classifier_attr="heads",
        in_features_path="heads.head.in_features",
        description="Vision Transformer, patrones globales (86.6M params)",
    ),
]


def build_model(config: ModelConfig, num_classes: int, device: torch.device) -> nn.Module:
    """
    Construye un modelo con transfer learning:
    1. Carga pesos pre-entrenados de ImageNet
    2. Congela todas las capas del backbone (no se entrenan)
    3. Reemplaza la cabeza clasificadora por una nueva para nuestras clases
    """
    # 1. Cargar modelo con pesos pre-entrenados
    weights = eval(f"models.{config.weights}")
    model_fn = getattr(models, config.build_fn)
    model = model_fn(weights=weights)

    # 2. Congelar backbone — solo entrenaremos la cabeza nueva
    # Esto es transfer learning: el backbone ya sabe extraer features
    # de imágenes (bordes, texturas, formas) gracias a ImageNet.
    # Solo necesitamos enseñarle a clasificar nuestras 2 clases.
    for param in model.parameters():
        param.requires_grad = False

    # 3. Reemplazar cabeza clasificadora
    # Cada arquitectura guarda su clasificador en un sitio diferente
    in_features = _get_in_features(model, config.in_features_path)

    new_head = nn.Sequential(
        nn.Dropout(0.3),          # Regularización para evitar overfitting
        nn.Linear(in_features, num_classes),
    )

    _set_classifier(model, config.classifier_attr, new_head, config.name)

    # Los parámetros de la cabeza nueva SÍ se entrenan
    return model.to(device)


def _get_in_features(model: nn.Module, path: str) -> int:
    """Navega el modelo para encontrar in_features de la última capa."""
    obj = model
    for attr in path.split("."):
        if attr.isdigit():
            obj = obj[int(attr)]
        else:
            obj = getattr(obj, attr)
    return obj


def _set_classifier(model: nn.Module, attr: str, new_head: nn.Module, name: str) -> None:
    """Reemplaza la cabeza clasificadora según la arquitectura."""
    if name.startswith("EfficientNet"):
        # EfficientNet: classifier = Sequential([Dropout, Linear])
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(new_head[1].in_features, new_head[1].out_features),
        )
    elif name == "ConvNeXt-Tiny":
        # ConvNeXt: classifier = Sequential([LayerNorm, Flatten, Linear])
        old_cls = model.classifier
        model.classifier = nn.Sequential(
            old_cls[0],  # LayerNorm2d
            old_cls[1],  # Flatten
            nn.Dropout(0.3),
            nn.Linear(new_head[1].in_features, new_head[1].out_features),
        )
    elif name == "ViT-B/16":
        # ViT: heads = Sequential(head=Linear)
        model.heads = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(new_head[1].in_features, new_head[1].out_features),
        )
    elif name == "DenseNet-121":
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(new_head[1].in_features, new_head[1].out_features),
        )
    elif name == "ResNet-50":
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(new_head[1].in_features, new_head[1].out_features),
        )
    else:
        setattr(model, attr, new_head)

    # Descongelar la nueva cabeza para que se entrene
    for param in _get_head_params(model, name):
        param.requires_grad = True


def _get_head_params(model: nn.Module, name: str):
    """Devuelve los parámetros entrenables de la cabeza."""
    if name.startswith("EfficientNet") or name == "DenseNet-121":
        return model.classifier.parameters()
    elif name == "ConvNeXt-Tiny":
        return model.classifier.parameters()
    elif name == "ViT-B/16":
        return model.heads.parameters()
    elif name == "ResNet-50":
        return model.fc.parameters()
    else:
        return model.parameters()


# ══════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════

def get_data_loaders() -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Crea los DataLoaders para train, val y test.

    Transformaciones:
    - Train: augmentation (flip, rotación, color jitter) + normalización ImageNet
    - Val/Test: solo resize + normalización ImageNet

    La normalización con media y std de ImageNet es OBLIGATORIA cuando
    usamos pesos pre-entrenados de ImageNet — el modelo espera esos valores.
    """
    # Media y desviación estándar de ImageNet (estándar para transfer learning)
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),           # Flip horizontal aleatorio
        transforms.RandomVerticalFlip(),              # Flip vertical (útil en histología)
        transforms.RandomRotation(15),                # Rotación ±15 grados
        transforms.ColorJitter(                       # Variación de color/brillo
            brightness=0.2, contrast=0.2,
            saturation=0.1, hue=0.05,
        ),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    full_train_dataset = datasets.ImageFolder(DATASET_DIR / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(DATASET_DIR / "val", transform=eval_transform)
    test_dataset = datasets.ImageFolder(DATASET_DIR / "test", transform=eval_transform)

    # Submuestrear train a MAX_TRAIN_PER_CLASS por clase para acelerar
    if MAX_TRAIN_PER_CLASS and MAX_TRAIN_PER_CLASS < len(full_train_dataset) // len(full_train_dataset.classes):
        indices_by_class: dict[int, list[int]] = {}
        for idx, (_, label) in enumerate(full_train_dataset.samples):
            indices_by_class.setdefault(label, []).append(idx)
        selected: list[int] = []
        for cls_indices in indices_by_class.values():
            if len(cls_indices) > MAX_TRAIN_PER_CLASS:
                selected.extend(random.sample(cls_indices, MAX_TRAIN_PER_CLASS))
            else:
                selected.extend(cls_indices)
        random.shuffle(selected)
        train_dataset = Subset(full_train_dataset, selected)
        print(f"  Train: {len(train_dataset)} imágenes (submuestreado de {len(full_train_dataset)}, {MAX_TRAIN_PER_CLASS}/clase)")
    else:
        train_dataset = full_train_dataset
        print(f"  Train: {len(train_dataset)} imágenes")

    print(f"  Val:   {len(val_dataset)} imágenes")
    print(f"  Test:  {len(test_dataset)} imágenes")
    print(f"  Clases: {full_train_dataset.classes}")

    # persistent_workers=True: mantiene los workers vivos entre épocas
    # (clave en Windows para evitar el coste de reiniciar procesos cada época)
    _use_persistent = NUM_WORKERS > 0
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=_use_persistent,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=_use_persistent,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=_use_persistent,
    )
    return train_loader, val_loader, test_loader


# ══════════════════════════════════════════════
# ENTRENAMIENTO Y EVALUACIÓN
# ══════════════════════════════════════════════

@dataclass
class TrainResult:
    """Resultado del entrenamiento de un modelo."""
    name: str
    description: str
    best_val_acc: float = 0.0
    best_val_loss: float = float("inf")
    test_acc: float = 0.0
    test_loss: float = 0.0
    test_precision: float = 0.0
    test_recall: float = 0.0
    test_f1: float = 0.0
    test_tp: int = 0
    test_fp: int = 0
    test_fn: int = 0   # Falsos Negativos: dijo "normal" pero era cáncer
    test_tn: int = 0
    train_time_sec: float = 0.0
    epoch_history: list[dict] = field(default_factory=list)
    best_epoch: int = 0


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Entrena una época. Devuelve (loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """
    Evalúa el modelo. Devuelve dict con: loss, accuracy, precision, recall,
    f1, tp, fp, fn, tn.

    Clase positiva = 0 (colon_aca = cáncer).
    Un Falso Negativo (FN) significa que el modelo dijo "normal" cuando
    en realidad era cáncer — el error más peligroso en diagnóstico médico.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    total = len(all_labels)
    avg_loss = running_loss / total

    preds = np.array(all_preds)
    labels = np.array(all_labels)
    accuracy = (preds == labels).mean()

    # Métricas por clase (binario: clase 0 = colon_aca, clase 1 = colon_n)
    # Para cáncer (clase 0): TP = predice cáncer cuando es cáncer
    tp = int(((preds == 0) & (labels == 0)).sum())  # Verdadero Positivo
    fp = int(((preds == 0) & (labels == 1)).sum())  # Falso Positivo (dice cáncer, es normal)
    fn = int(((preds == 1) & (labels == 0)).sum())  # Falso Negativo (dice normal, es cáncer) ← PELIGROSO
    tn = int(((preds == 1) & (labels == 1)).sum())  # Verdadero Negativo

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Sensibilidad: detectar todos los cánceres
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "loss": avg_loss,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def train_model(
    config: ModelConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    model_output_dir: Path,
) -> TrainResult:
    """
    Entrena un modelo completo:
    1. Construye modelo con transfer learning
    2. Entrena N épocas, guardando el mejor por val accuracy
    3. Evalúa en test con los mejores pesos
    4. Guarda pesos + métricas
    """
    result = TrainResult(name=config.name, description=config.description)

    print("\n  Construyendo modelo...")
    model = build_model(config, num_classes=len(CLASS_NAMES), device=device)

    # Contar parámetros entrenables vs totales
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parámetros totales:     {total_params:>12,}")
    print(f"  Parámetros entrenables: {trainable_params:>12,}")
    print(f"  Congelados:             {total_params - trainable_params:>12,}")

    # CrossEntropy es la loss estándar para clasificación
    criterion = nn.CrossEntropyLoss()

    # Adam: optimizador adaptativo, funciona bien con transfer learning
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
    )

    # ReduceLROnPlateau: reduce el learning rate cuando val_loss se estanca
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=False,
    )

    best_weights_path = model_output_dir / "best.pt"
    model_output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
        )
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)
        val_loss = val_metrics["loss"]
        val_acc = val_metrics["accuracy"]

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        # Guardar mejor modelo
        is_best = val_acc > result.best_val_acc
        if is_best:
            result.best_val_acc = val_acc
            result.best_val_loss = val_loss
            result.best_epoch = epoch
            torch.save(model.state_dict(), best_weights_path)

        marker = " ★" if is_best else ""
        print(
            f"    Época {epoch:>2}/{EPOCHS} │ "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} │ "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} │ "
            f"lr={current_lr:.1e}{marker}"
        )

        result.epoch_history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "train_acc": round(train_acc, 5),
            "val_loss": round(val_loss, 5),
            "val_acc": round(val_acc, 5),
            "lr": current_lr,
        })

    result.train_time_sec = time.time() - start_time

    # Evaluar en test con los mejores pesos
    print(f"\n  Evaluando en test (mejores pesos, época {result.best_epoch})...")
    model.load_state_dict(torch.load(best_weights_path, weights_only=True))
    t = evaluate(model, test_loader, criterion, device)
    result.test_acc = t["accuracy"]
    result.test_loss = t["loss"]
    result.test_precision = t["precision"]
    result.test_recall = t["recall"]
    result.test_f1 = t["f1"]
    result.test_tp = t["tp"]
    result.test_fp = t["fp"]
    result.test_fn = t["fn"]
    result.test_tn = t["tn"]

    print(f"  Test accuracy:  {t['accuracy']:.4f}")
    print(f"  Test precision: {t['precision']:.4f}")
    print(f"  Test recall:    {t['recall']:.4f}  (sensibilidad — detectar cánceres)")
    print(f"  Test F1:        {t['f1']:.4f}")
    print("  Matriz de confusión (clase positiva = cáncer):")
    print(f"    TP={t['tp']:>4}  FP={t['fp']:>4}")
    print(f"    FN={t['fn']:>4}  TN={t['tn']:>4}")
    if t['fn'] > 0:
        print(f"    ⚠ {t['fn']} falsos negativos: el modelo NO detectó cáncer en {t['fn']} casos")
    print(f"  Tiempo total:   {result.train_time_sec:.1f}s")

    # Guardar métricas a CSV (para el dashboard)
    csv_path = model_output_dir / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=result.epoch_history[0].keys())
        writer.writeheader()
        writer.writerows(result.epoch_history)

    return result


# ══════════════════════════════════════════════
# RANKING Y RESULTADOS
# ══════════════════════════════════════════════

def print_ranking(results: list[TrainResult]) -> TrainResult:
    """Imprime ranking ordenado por test accuracy y devuelve el ganador."""
    # Ordenar por test accuracy (descendente)
    ranked = sorted(results, key=lambda r: r.test_acc, reverse=True)

    print()
    print("═" * 80)
    print("  🏆  RANKING FINAL — COMPETICIÓN DE MODELOS")
    print("═" * 80)
    print()
    print(f"  {'#':>2}  {'Modelo':<20} {'Test Acc':>9} {'Recall':>7} "
          f"{'F1':>7} {'Precisión':>10} {'FN':>4} {'FP':>4} {'Tiempo':>8}")
    print("  " + "─" * 78)

    medals = ["🥇", "🥈", "🥉", "  ", "  ", "  "]
    for i, r in enumerate(ranked):
        medal = medals[i] if i < len(medals) else "  "
        minutes = r.train_time_sec / 60
        fn_mark = " ⚠" if r.test_fn > 0 else ""
        print(
            f"  {medal} {r.name:<20} {r.test_acc:>8.2%} {r.test_recall:>6.2%} "
            f"{r.test_f1:>7.4f} {r.test_precision:>9.4f} {r.test_fn:>4}{fn_mark:>3} "
            f"{r.test_fp:>3} {minutes:>6.1f}min"
        )

    print()
    w = ranked[0]
    print(f"  🏆 GANADOR: {w.name} — Acc: {w.test_acc:.2%} | Recall: {w.test_recall:.2%} | FN: {w.test_fn}")
    if w.test_fn > 0:
        print(f"     ⚠ El mejor modelo tiene {w.test_fn} falso(s) negativo(s) en test")
        print("       (cánceres que el modelo clasificó como normales)")
    print("═" * 80)

    return ranked[0]


def save_competition_summary(results: list[TrainResult], output_dir: Path) -> None:
    """Guarda un resumen JSON con todos los resultados para el dashboard."""
    ranked = sorted(results, key=lambda r: r.test_acc, reverse=True)

    summary = {
        "competition": {
            "task": "Clasificación binaria: cáncer de colon (histopatología)",
            "dataset": str(DATASET_DIR),
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "image_size": IMAGE_SIZE,
        },
        "ranking": [
            {
                "rank": i + 1,
                "name": r.name,
                "description": r.description,
                "test_accuracy": round(r.test_acc, 5),
                "test_f1": round(r.test_f1, 5),
                "test_precision": round(r.test_precision, 5),
                "test_recall": round(r.test_recall, 5),
                "test_loss": round(r.test_loss, 5),
                "confusion_matrix": {
                    "tp": r.test_tp, "fp": r.test_fp,
                    "fn": r.test_fn, "tn": r.test_tn,
                },
                "best_val_accuracy": round(r.best_val_acc, 5),
                "best_epoch": r.best_epoch,
                "train_time_seconds": round(r.train_time_sec, 1),
            }
            for i, r in enumerate(ranked)
        ],
        "winner": ranked[0].name,
    }

    summary_path = output_dir / "competition_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Resumen guardado en: {summary_path}")


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════

def main() -> None:
    print("=" * 80)
    print("  🏆  COMPETICIÓN DE MODELOS — CLASIFICACIÓN CÁNCER DE COLON")
    print("=" * 80)
    print()

    # ── Verificar GPU ──
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU: {gpu_name} ({vram:.1f} GB VRAM)")
    else:
        device = torch.device("cpu")
        print("  ⚠ GPU no detectada, usando CPU (será más lento)")
    print(f"  Device: {device}")
    print()

    # ── Verificar dataset ──
    if not DATASET_DIR.exists():
        print(f"  ✗ Dataset no encontrado: {DATASET_DIR}")
        print("    Ejecuta: uv run python prepare_colon_dataset.py")
        sys.exit(1)

    # ── Cargar datos ──
    print("▸ Cargando dataset...")
    train_loader, val_loader, test_loader = get_data_loaders()
    print()

    # ── Competición ──
    models_to_run = COMPETITION_MODELS
    if ONLY_MODEL:
        models_to_run = [m for m in COMPETITION_MODELS if m.name == ONLY_MODEL]
        if not models_to_run:
            print(f"  ✗ Modelo no encontrado: {ONLY_MODEL}")
            print("    Revisa ONLY_MODEL o déjalo en None para competir todos")
            sys.exit(1)
        if OVERRIDE_EPOCHS is not None:
            # Ajustar global para que el resto del script use el valor rapido
            global EPOCHS
            EPOCHS = OVERRIDE_EPOCHS

    print(f"▸ Modelos en competición: {len(models_to_run)}")
    print(f"▸ Épocas por modelo: {EPOCHS}")
    print(f"▸ Batch size: {BATCH_SIZE}")
    print(f"▸ Learning rate: {LEARNING_RATE}")
    print()

    results: list[TrainResult] = []

    for i, config in enumerate(models_to_run, 1):
        print("─" * 80)
        print(f"  [{i}/{len(models_to_run)}] {config.name}")
        print(f"  {config.description}")
        print("─" * 80)

        model_dir = OUTPUT_DIR / config.name.replace("/", "-").replace(" ", "_")

        try:
            result = train_model(
                config, train_loader, val_loader, test_loader,
                device, model_dir,
            )
            results.append(result)
        except Exception as e:
            print(f"\n  ✗ ERROR entrenando {config.name}: {e}")
            print("  Saltando al siguiente modelo...\n")
            continue

        # Liberar VRAM entre modelos
        torch.cuda.empty_cache()

    if not results:
        print("\n  ✗ Ningún modelo se entrenó correctamente.")
        sys.exit(1)

    # ── Ranking ──
    winner = print_ranking(results)

    # ── Guardar resumen ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_competition_summary(results, OUTPUT_DIR)

    # ── Copiar mejor modelo a models/ ──
    winner_dir = OUTPUT_DIR / winner.name.replace("/", "-").replace(" ", "_")
    best_pt = winner_dir / "best.pt"
    if best_pt.exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        dst = MODELS_DIR / "microscopy.pt"
        shutil.copy2(best_pt, dst)
        print(f"  Mejor modelo copiado a: {dst}")

        # Guardar también metadata del ganador para que detect_realtime
        # sepa qué arquitectura cargar
        meta = {
            "architecture": winner.name,
            "build_fn": next(
                c.build_fn for c in COMPETITION_MODELS if c.name == winner.name
            ),
            "class_names": CLASS_NAMES,
            "image_size": IMAGE_SIZE,
            "test_accuracy": round(winner.test_acc, 5),
        }
        meta_path = MODELS_DIR / "microscopy_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"  Metadata guardada en: {meta_path}")

    print()
    print("✓ Competición finalizada.")


if __name__ == "__main__":
    main()
