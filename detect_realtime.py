"""
detect_realtime.py
==================
Sistema de diagnóstico de cáncer de colon en 3 fases:

  Fase 1 → Historial médico: carga CSV/JSON, modelo predice riesgo de cáncer.
  Fase 2 → Colonoscopia:     vídeo/webcam, modelo detecta pólipos.
  Fase 3 → Foto histológica: imagen de tejido, modelo clasifica malignidad.

Flujo:
  Inicio → Fase 1 → (positivo?) → Fase 2 → (pólipos?) → Fase 3 → Resultado
                ↓ (negativo)           ↓ (sin pólipos)
              Inicio                 Inicio

Ejecutar:
    python detect_realtime.py

Controles en ventanas de vídeo:
    q  → Finalizar fase
    s  → Capturar screenshot
    p  → Pausar / reanudar
"""

from __future__ import annotations

import csv
import json
import sys
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any, Optional

# Forzar UTF-8 en stdout/stderr para Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np

# ══════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════

# Rutas a los modelos (cambiar cuando estén entrenados)
MODEL_HISTORY: str = "models/history_model.pkl"     # Modelo de historial clínico
MODEL_COLONOSCOPY: str = "models/colonoscopy.pt"    # YOLO para pólipos
MODEL_MICROSCOPY: str = "models/microscopy.pt"      # Clasificación tejido
MODEL_MICROSCOPY_META: str = "models/microscopy_meta.json"  # Metadata del modelo ganador

# Umbrales
CONFIDENCE_THRESHOLD: float = 0.25
HISTORY_RISK_THRESHOLD: float = 0.5    # Umbral para considerar riesgo positivo
COLONOSCOPY_CONFIDENCE_THRESHOLD: float = 0.50  # Reduce falsos positivos de YOLO
POLYP_CONFIRM_SECONDS: float = 1.0      # Tiempo mínimo persistente para contar pólipo
POLYP_TRACK_IOU_THRESHOLD: float = 0.25
POLYP_TRACK_MAX_MISSING_SECONDS: float = 0.35

# Rendimiento
FRAME_SKIP: int = 1
INFERENCE_SIZE: int = 640

# Vídeo de salida
SAVE_OUTPUT: bool = False

# Colores BGR para detecciones por fase
COLORS = {
    "polyp":   ((0, 255, 0),   (0, 180, 0)),     # Verde
    "cancer":  ((0, 0, 255),   (0, 0, 180)),      # Rojo
    "default": ((255, 200, 0), (200, 150, 0)),     # Azul claro
}
TEXT_COLOR: tuple[int, int, int] = (255, 255, 255)

WEBCAM_INDEX: int = 0
SCREENSHOTS_DIR: str = "screenshots"

# Estilo de la GUI
BG_DARK = "#1e1e2e"
BG_CARD = "#313244"
FG_TEXT = "#cdd6f4"
FG_SUB = "#a6adc8"
ACCENT_BLUE = "#89b4fa"
ACCENT_GREEN = "#a6e3a1"
ACCENT_RED = "#f38ba8"
ACCENT_YELLOW = "#f9e2af"
ACCENT_MAUVE = "#cba6f7"
BTN_INACTIVE = "#585b70"


# ══════════════════════════════════════════════
# CARGA DE MODELOS
# ══════════════════════════════════════════════

def load_yolo_model(model_path: str, label: str) -> Any:
    """
    Carga un modelo YOLOv8.
    Devuelve None si no existe (modo demo).
    """
    model_file = Path(model_path)
    if not model_file.exists():
        print(f"  ⚠ Modelo {label} no encontrado: '{model_path}' → modo DEMO")
        return None
    try:
        from ultralytics import YOLO
        model = YOLO(str(model_file))
        print(f"  ✓ Modelo {label} cargado: {model_path}")
        return model
    except ImportError:
        print("  ✗ ultralytics no instalado. Instala con: uv add ultralytics")
        return None
    except Exception as e:
        print(f"  ✗ Error cargando {label}: {e}")
        return None


def load_history_model(model_path: str) -> Any:
    """
    Carga el modelo de historial clínico (sklearn o similar).
    Devuelve None si no existe (modo demo).
    """
    model_file = Path(model_path)
    if not model_file.exists():
        print(f"  ⚠ Modelo historial no encontrado: '{model_path}' → modo DEMO")
        return None
    try:
        import pickle
        with open(model_file, "rb") as f:
            model = pickle.load(f)
        print(f"  ✓ Modelo historial cargado: {model_path}")
        return model
    except Exception as e:
        print(f"  ✗ Error cargando historial: {e}")
        return None


def load_classification_model(model_path: str, meta_path: str, label: str) -> dict | None:
    """
    Carga un modelo de clasificación (EfficientNet, ResNet, etc.).
    Necesita el archivo de metadata JSON para saber qué arquitectura usar.
    Devuelve un dict con 'model', 'transform', 'class_names' o None (modo demo).
    """
    import json as _json

    model_file = Path(model_path)
    meta_file = Path(meta_path)

    if not model_file.exists():
        print(f"  ⚠ Modelo {label} no encontrado: '{model_path}' → modo DEMO")
        return None
    if not meta_file.exists():
        print(f"  ⚠ Metadata no encontrada: '{meta_path}' → modo DEMO")
        return None

    try:
        import torch
        from torchvision import models, transforms

        with open(meta_file, "r", encoding="utf-8") as f:
            meta = _json.load(f)

        build_fn = meta["build_fn"]
        class_names = meta.get("class_names", ["colon_aca", "colon_n"])
        image_size = meta.get("image_size", 224)
        arch_name = meta.get("architecture", "")

        # Construir modelo con la misma arquitectura usada en entrenamiento
        model_fn = getattr(models, build_fn)
        model = model_fn(weights=None)  # Sin pesos pre-entrenados

        # Reemplazar cabeza igual que en el entrenamiento
        import torch.nn as nn
        num_classes = len(class_names)

        if "efficientnet" in build_fn:
            in_f = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(0.3), nn.Linear(in_f, num_classes),
            )
        elif "resnet" in build_fn:
            in_f = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(0.3), nn.Linear(in_f, num_classes),
            )
        elif "convnext" in build_fn:
            in_f = model.classifier[2].in_features
            old_cls = model.classifier
            model.classifier = nn.Sequential(
                old_cls[0], old_cls[1],
                nn.Dropout(0.3), nn.Linear(in_f, num_classes),
            )
        elif "densenet" in build_fn:
            in_f = model.classifier.in_features
            model.classifier = nn.Sequential(
                nn.Dropout(0.3), nn.Linear(in_f, num_classes),
            )
        elif "vit" in build_fn:
            in_f = model.heads.head.in_features
            model.heads = nn.Sequential(
                nn.Dropout(0.3), nn.Linear(in_f, num_classes),
            )

        # Cargar pesos entrenados
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(str(model_file), map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        # Transform para inferencia (mismo que en entrenamiento)
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])

        print(f"  ✓ Modelo {label} cargado: {arch_name} ({model_path})")
        return {
            "model": model,
            "transform": transform,
            "class_names": class_names,
            "device": device,
            "architecture": arch_name,
        }
    except Exception as e:
        print(f"  ✗ Error cargando {label}: {e}")
        return None


# ══════════════════════════════════════════════
# FASE 1: HISTORIAL MÉDICO
# ══════════════════════════════════════════════

def load_patient_data(filepath: str) -> Optional[dict]:
    """
    Carga datos del paciente desde CSV o JSON.
    Devuelve un diccionario con los datos.
    """
    path = Path(filepath)
    if not path.exists():
        print(f"  ✗ Archivo no encontrado: {filepath}")
        return None

    try:
        if path.suffix.lower() == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Si es una lista, tomar el primer registro
            if isinstance(data, list):
                data = data[0] if data else {}
            return data

        elif path.suffix.lower() == ".csv":
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if not rows:
                print("  ✗ CSV vacío.")
                return None
            return rows[0]  # Primer paciente

        else:
            print(f"  ✗ Formato no soportado: {path.suffix}")
            return None

    except Exception as e:
        print(f"  ✗ Error leyendo archivo: {e}")
        return None


def predict_cancer_risk(
    model: Any, patient_data: dict
) -> tuple[bool, float]:
    """
    Predice si el paciente tiene riesgo de cáncer.
    Sin modelo → siempre positivo (modo demo para poder avanzar).
    Devuelve (tiene_riesgo, probabilidad).
    """
    if model is None:
        # Modo demo: siempre pasa a la siguiente fase
        return True, 0.99

    try:
        # Convertir datos numéricos a array para sklearn
        features = np.array([
            [float(v) for v in patient_data.values() if _is_numeric(v)]
        ])
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)[0]
            risk = float(proba[1]) if len(proba) > 1 else float(proba[0])
        else:
            pred = model.predict(features)[0]
            risk = float(pred)
        is_positive = risk >= HISTORY_RISK_THRESHOLD
        return is_positive, risk
    except Exception as e:
        print(f"  ⚠ Error en predicción: {e}")
        print("  → Continuando en modo demo (resultado positivo)")
        return True, 0.99


def _is_numeric(value) -> bool:
    """Comprueba si un valor es convertible a float."""
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


# ══════════════════════════════════════════════
# FASES 2 y 3: DETECCIÓN EN VÍDEO
# ══════════════════════════════════════════════

def run_inference(
    model: Any,
    frame: np.ndarray,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> list[dict]:
    """Ejecuta YOLO sobre un frame y devuelve las detecciones."""
    results = model.predict(
        source=frame,
        imgsz=INFERENCE_SIZE,
        conf=confidence_threshold,
        verbose=False,
    )
    detections: list[dict] = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
            conf = float(boxes.conf[i].cpu().numpy())
            cls_id = int(boxes.cls[i].cpu().numpy())
            cls_name = result.names.get(cls_id, f"class_{cls_id}")
            detections.append({
                "x1": int(x1), "y1": int(y1),
                "x2": int(x2), "y2": int(y2),
                "confidence": conf,
                "class_name": cls_name,
            })
    return detections


def draw_detections(
    frame: np.ndarray, detections: list[dict], phase: str = "polyp"
) -> np.ndarray:
    """Dibuja bounding boxes coloreadas según la fase."""
    box_color, bg_color = COLORS.get(phase, COLORS["default"])
    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        conf = det["confidence"]
        cls_name = det["class_name"]
        track_id = det.get("track_id")
        label = f"{cls_name} #{track_id} {conf:.2f}" if track_id is not None else f"{cls_name} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), bg_color, -1)
        cv2.putText(
            frame, label, (x1 + 4, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1, cv2.LINE_AA,
        )
    return frame


def draw_hud(
    frame: np.ndarray,
    fps: float,
    frame_num: int,
    total_frames: int,
    phase_label: str,
    detections_count: int,
    model_loaded: bool,
    total_positives: int = 0,
    paused: bool = False,
) -> np.ndarray:
    """Dibuja el HUD con información de estado."""
    h, _w = frame.shape[:2]
    lines = [
        f"Fase: {phase_label}",
        f"FPS: {fps:.1f}",
        f"Detecciones (frame): {detections_count}",
        f"Total positivos: {total_positives}",
    ]
    if total_frames > 0:
        progress = frame_num / total_frames * 100
        lines.append(f"Progreso: {progress:.0f}%")
    if not model_loaded:
        lines.append("MODO DEMO (sin modelo)")
    if paused:
        lines.append("Estado: PAUSADO")

    y_offset = 25
    for line in lines:
        cv2.putText(
            frame, line, (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA,
        )
        y_offset += 22

    controls = "Controles: clic en botones | q salir | p pausa | s captura"
    cv2.putText(
        frame, controls, (10, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA,
    )
    return frame


def _get_screen_size() -> tuple[int, int]:
    """Obtiene el tamaño de pantalla para centrar ventanas OpenCV."""
    if sys.platform == "win32":
        try:
            import ctypes

            user32 = ctypes.windll.user32
            return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        except Exception:
            pass

    probe = tk.Tk()
    probe.withdraw()
    probe.update_idletasks()
    size = (probe.winfo_screenwidth(), probe.winfo_screenheight())
    probe.destroy()
    return size


def _center_cv_window(
    window_name: str,
    content_width: int,
    content_height: int,
    max_width: int = 1280,
    max_height: int = 720,
) -> tuple[int, int]:
    """Redimensiona y centra una ventana OpenCV en pantalla."""
    scale = min(
        max_width / max(content_width, 1),
        max_height / max(content_height, 1),
        1.0,
    )
    view_w = max(int(content_width * scale), 640)
    view_h = max(int(content_height * scale), 360)

    screen_w, screen_h = _get_screen_size()
    x = max((screen_w - view_w) // 2, 0)
    y = max((screen_h - view_h) // 2, 0)

    cv2.resizeWindow(window_name, view_w, view_h)
    cv2.moveWindow(window_name, x, y)
    return view_w, view_h


def _draw_action_buttons(
    frame: np.ndarray,
    actions: list[tuple[str, str, tuple[int, int, int]]],
    title: str = "Controles",
) -> tuple[np.ndarray, dict[str, tuple[int, int, int, int]]]:
    """Dibuja una botonera simple dentro de la ventana OpenCV."""
    h, w = frame.shape[:2]
    panel_w = 230
    button_h = 40
    gap = 10
    pad = 14
    title_h = 36
    panel_h = title_h + (len(actions) * button_h) + ((len(actions) - 1) * gap) + pad * 2

    x1 = max(w - panel_w - 16, 16)
    y1 = 16
    x2 = x1 + panel_w
    y2 = min(y1 + panel_h, h - 16)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 28), -1)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (120, 120, 140), 1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    cv2.putText(
        frame, title, (x1 + 12, y1 + 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 245), 2, cv2.LINE_AA,
    )

    buttons: dict[str, tuple[int, int, int, int]] = {}
    btn_y = y1 + title_h
    for key, label, color in actions:
        btn_x1 = x1 + pad
        btn_y1 = btn_y
        btn_x2 = x2 - pad
        btn_y2 = btn_y1 + button_h
        buttons[key] = (btn_x1, btn_y1, btn_x2, btn_y2)

        cv2.rectangle(frame, (btn_x1, btn_y1), (btn_x2, btn_y2), color, -1)
        cv2.rectangle(frame, (btn_x1, btn_y1), (btn_x2, btn_y2), (245, 245, 245), 1)
        cv2.putText(
            frame, label, (btn_x1 + 12, btn_y1 + 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.62, (18, 18, 25), 2, cv2.LINE_AA,
        )
        btn_y += button_h + gap

    return frame, buttons


def _set_mouse_buttons_handler(
    window_name: str,
    state: dict[str, Any],
) -> None:
    """Asigna un handler para capturar clicks sobre la botonera OpenCV."""

    def _handler(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONUP:
            return
        for action, (x1, y1, x2, y2) in state.get("buttons", {}).items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                state["clicked"] = action
                break

    cv2.setMouseCallback(window_name, _handler)


def _format_ratio(numerator: int, denominator: int) -> str:
    """Formatea una razón como porcentaje legible."""
    if denominator <= 0:
        return "N/A"
    return f"{(numerator / denominator) * 100:.1f}%"


def _bbox_iou(box_a: dict[str, Any], box_b: dict[str, Any]) -> float:
    """Calcula IoU entre dos cajas."""
    x_left = max(int(box_a["x1"]), int(box_b["x1"]))
    y_top = max(int(box_a["y1"]), int(box_b["y1"]))
    x_right = min(int(box_a["x2"]), int(box_b["x2"]))
    y_bottom = min(int(box_a["y2"]), int(box_b["y2"]))

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection = float((x_right - x_left) * (y_bottom - y_top))
    area_a = float(
        max(int(box_a["x2"]) - int(box_a["x1"]), 0)
        * max(int(box_a["y2"]) - int(box_a["y1"]), 0)
    )
    area_b = float(
        max(int(box_b["x2"]) - int(box_b["x1"]), 0)
        * max(int(box_b["y2"]) - int(box_b["y1"]), 0)
    )
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _update_polyp_tracks(
    detections: list[dict],
    active_tracks: list[dict[str, Any]],
    next_track_id: int,
    iou_threshold: float = 0.3,
    max_missing_frames: int = 10,
    min_confirm_frames: int = 30,
) -> tuple[list[dict[str, Any]], int, int]:
    """Asocia detecciones a tracks y devuelve cuántos pólipos únicos nuevos confirmar."""
    for track in active_tracks:
        track["matched"] = False

    unique_new_polyps = 0

    for det in detections:
        best_track = None
        best_iou = 0.0
        for track in active_tracks:
            if track["matched"]:
                continue
            iou = _bbox_iou(det, track["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_track = track

        if best_track is not None and best_iou >= iou_threshold:
            best_track["bbox"] = {
                "x1": det["x1"],
                "y1": det["y1"],
                "x2": det["x2"],
                "y2": det["y2"],
            }
            best_track["confidence"] = det["confidence"]
            best_track["missing"] = 0
            best_track["hits"] += 1
            best_track["matched"] = True
            det["track_id"] = best_track["track_id"]
            if not best_track["counted"] and best_track["hits"] >= min_confirm_frames:
                best_track["counted"] = True
                unique_new_polyps += 1
        else:
            new_track = {
                "track_id": next_track_id,
                "bbox": {
                    "x1": det["x1"],
                    "y1": det["y1"],
                    "x2": det["x2"],
                    "y2": det["y2"],
                },
                "confidence": det["confidence"],
                "missing": 0,
                "hits": 1,
                "counted": False,
                "matched": True,
            }
            active_tracks.append(new_track)
            det["track_id"] = next_track_id
            next_track_id += 1

    surviving_tracks: list[dict[str, Any]] = []
    for track in active_tracks:
        if not track["matched"]:
            track["missing"] += 1
        if track["missing"] <= max_missing_frames:
            surviving_tracks.append(track)

    return surviving_tracks, next_track_id, unique_new_polyps


def _normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """Normaliza un mapa a rango 0..255."""
    if heatmap.size == 0:
        return heatmap
    heatmap = np.maximum(heatmap, 0)
    max_value = float(heatmap.max())
    if max_value <= 1e-8:
        return np.zeros_like(heatmap, dtype=np.uint8)
    return np.uint8((heatmap / max_value) * 255.0)


def create_detection_explanation(
    frame: np.ndarray,
    detections: list[dict],
) -> dict[str, Any] | None:
    """Crea un mapa de atención simple a partir de las detecciones YOLO."""
    if frame is None or not detections:
        return None

    h, w = frame.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)

    for det in detections:
        x1 = max(int(det["x1"]), 0)
        y1 = max(int(det["y1"]), 0)
        x2 = min(int(det["x2"]), w)
        y2 = min(int(det["y2"]), h)
        if x2 <= x1 or y2 <= y1:
            continue
        heatmap[y1:y2, x1:x2] += float(det["confidence"])

    if not np.any(heatmap):
        return None

    blur_size = max(31, (min(h, w) // 12) | 1)
    heatmap = cv2.GaussianBlur(heatmap, (blur_size, blur_size), 0)
    heatmap_norm = _normalize_heatmap(heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 0.55, heatmap_color, 0.45, 0)

    cv2.putText(
        overlay,
        "Mapa de atencion: zonas con mas peso en la deteccion",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return {
        "original": frame.copy(),
        "overlay": overlay,
        "title": "Explicacion visual - Colonoscopia",
        "method": "Mapa de detecciones ponderado por confianza",
    }


def _find_last_conv_layer(model: Any) -> Any | None:
    """Encuentra la última capa convolucional 2D del modelo."""
    import torch.nn as nn

    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    return last_conv


def create_gradcam_explanation(
    model_bundle: dict | None,
    frame: np.ndarray,
    target_class_name: str | None = None,
) -> dict[str, Any] | None:
    """Genera un Grad-CAM básico para el modelo de clasificación."""
    if model_bundle is None or frame is None:
        return None

    import torch
    import torch.nn.functional as F

    model = model_bundle["model"]
    transform = model_bundle["transform"]
    device = model_bundle["device"]
    class_names = model_bundle["class_names"]
    target_layer = model_bundle.get("gradcam_layer") or _find_last_conv_layer(model)
    model_bundle["gradcam_layer"] = target_layer

    if target_layer is None:
        return None

    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def forward_hook(_module, _inputs, output):
        activations.append(output.detach())

    def backward_hook(_module, _grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    handle_fw = target_layer.register_forward_hook(forward_hook)
    handle_bw = target_layer.register_full_backward_hook(backward_hook)

    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = transform(rgb).unsqueeze(0).to(device)

        model.zero_grad(set_to_none=True)
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]

        if target_class_name and target_class_name in class_names:
            class_idx = class_names.index(target_class_name)
        else:
            class_idx = int(probs.argmax().item())

        score = outputs[:, class_idx].sum()
        score.backward()

        if not activations or not gradients:
            return None

        activation = activations[-1]
        gradient = gradients[-1]
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = F.interpolate(
            cam,
            size=(frame.shape[0], frame.shape[1]),
            mode="bilinear",
            align_corners=False,
        )
        cam_np = cam[0, 0].detach().cpu().numpy()
        heatmap_norm = _normalize_heatmap(cam_np)
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(frame, 0.55, heatmap_color, 0.45, 0)

        label = class_names[class_idx]
        cv2.putText(
            overlay,
            f"Grad-CAM: zonas que mas activan {label}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return {
            "original": frame.copy(),
            "overlay": overlay,
            "title": "Explicacion visual - Histologia",
            "method": f"Grad-CAM sobre {label}",
        }
    except Exception as exc:
        print(f"  ⚠ No se pudo generar Grad-CAM: {exc}")
        return None
    finally:
        handle_fw.remove()
        handle_bw.remove()


def show_explanation_window(explanation: dict[str, Any] | None) -> None:
    """Muestra la explicación visual en una ventana centrada."""
    if explanation is None:
        messagebox.showinfo(
            "Explicacion visual",
            "No hay un mapa explicativo disponible para este resultado.",
        )
        return

    original = explanation["original"]
    overlay = explanation["overlay"]
    if original.shape[:2] != overlay.shape[:2]:
        overlay = cv2.resize(overlay, (original.shape[1], original.shape[0]))

    title_bar = np.full((60, original.shape[1] * 2 + 20, 3), 26, dtype=np.uint8)
    cv2.putText(
        title_bar,
        "Original",
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        title_bar,
        explanation.get("method", "Mapa explicativo"),
        (original.shape[1] + 40, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    separator = np.full((original.shape[0], 20, 3), 20, dtype=np.uint8)
    content = np.hstack([original, separator, overlay])
    canvas = np.vstack([title_bar, content])

    window_name = explanation.get("title", "Explicacion visual")
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    _center_cv_window(window_name, canvas.shape[1], canvas.shape[0], max_width=1500, max_height=900)

    ui_state: dict[str, Any] = {"buttons": {}, "clicked": None}
    _set_mouse_buttons_handler(window_name, ui_state)

    while True:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        frame_to_show = canvas.copy()
        frame_to_show, buttons = _draw_action_buttons(
            frame_to_show,
            [("close", "Cerrar", (70, 95, 220))],
            title="Explicacion",
        )
        ui_state["buttons"] = buttons
        cv2.imshow(window_name, frame_to_show)

        key = cv2.waitKey(30) & 0xFF
        clicked = ui_state.pop("clicked", None)
        if key in (27, ord("q"), 13, ord(" ")):
            break
        if clicked == "close":
            break

    try:
        cv2.destroyWindow(window_name)
    except cv2.error:
        pass


def save_screenshot(frame: np.ndarray, prefix: str = "screenshot") -> str:
    """Guarda screenshot del frame actual."""
    screenshots = Path(SCREENSHOTS_DIR)
    screenshots.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = screenshots / f"{prefix}_{timestamp}.jpg"
    cv2.imwrite(str(path), frame)
    return str(path)


def process_video_phase(
    source,
    mode: str,
    model: Any,
    phase_label: str,
    phase_color: str = "polyp",
) -> tuple[bool, dict[str, Any]]:
    """
    Procesa vídeo/webcam con un modelo YOLO.
    Devuelve (hubo_detecciones, métricas_de_fase).
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"  ✗ No se pudo abrir la fuente: {source}")
        return False, {
            "source_mode": mode,
            "frames_processed": 0,
            "positive_frames": 0,
            "total_detections": 0,
            "peak_detections": 0,
            "unique_polyps": 0,
            "avg_confidence": 0.0,
            "max_confidence": 0.0,
            "confidence_threshold": COLONOSCOPY_CONFIDENCE_THRESHOLD,
            "min_confirm_seconds": POLYP_CONFIRM_SECONDS,
            "min_confirm_frames": 0,
            "completion_ratio": None,
            "end_reason": "error",
            "explanation": None,
        }

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if mode == "Video" else 0
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    effective_fps = max(src_fps / max(FRAME_SKIP, 1), 1.0)
    min_confirm_frames = max(8, int(round(POLYP_CONFIRM_SECONDS * effective_fps)))
    max_missing_frames = max(
        3,
        int(round(POLYP_TRACK_MAX_MISSING_SECONDS * effective_fps)),
    )

    print(f"  Resolución: {width}x{height}")
    if total_frames > 0:
        print(f"  Frames: {total_frames} | FPS fuente: {src_fps:.1f}")

    print(
        "  Filtro de pólipos: "
        f"confianza >= {COLONOSCOPY_CONFIDENCE_THRESHOLD:.0%}, "
        f"persistencia >= {POLYP_CONFIRM_SECONDS:.1f}s "
        f"({min_confirm_frames} frames efectivos)"
    )

    # Writer opcional
    writer = None
    if SAVE_OUTPUT and mode == "Video":
        out_name = f"output_{phase_color}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_name, fourcc, src_fps, (width, height))

    window_name = f"{phase_label} - {mode}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    _center_cv_window(window_name, width, height)

    frame_count = 0
    total_positives = 0
    total_detections = 0
    peak_detections = 0
    unique_polyps = 0
    confidence_sum = 0.0
    max_confidence = 0.0
    paused = False
    prev_time = time.time()
    fps_display = 0.0
    last_frame = None
    last_visual_frame = None
    best_detection_score = -1.0
    best_focus_frame = None
    best_focus_detections: list[dict] = []
    end_reason = "completed"
    active_tracks: list[dict[str, Any]] = []
    next_track_id = 1
    ui_state: dict[str, Any] = {"buttons": {}, "clicked": None}
    _set_mouse_buttons_handler(window_name, ui_state)

    print("  ▸ Procesando... (usa los botones o pulsa 'q' para finalizar fase)")

    while True:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            end_reason = "window"
            break

        if not paused:
            ret, frame = cap.read()
            if not ret:
                if mode == "Video":
                    end_reason = "completed"
                    break
                else:
                    continue

            frame_count += 1
            if FRAME_SKIP > 1 and frame_count % FRAME_SKIP != 0:
                continue

            # Inferencia
            detections: list[dict] = []
            if model is not None:
                detections = run_inference(
                    model,
                    frame,
                    confidence_threshold=COLONOSCOPY_CONFIDENCE_THRESHOLD,
                )
            detections_in_frame = len(detections)
            active_tracks, next_track_id, new_unique_polyps = _update_polyp_tracks(
                detections,
                active_tracks,
                next_track_id,
                iou_threshold=POLYP_TRACK_IOU_THRESHOLD,
                max_missing_frames=max_missing_frames,
                min_confirm_frames=min_confirm_frames,
            )
            unique_polyps += new_unique_polyps
            if detections_in_frame:
                total_positives += 1
                total_detections += detections_in_frame
                peak_detections = max(peak_detections, detections_in_frame)
                confidence_sum += sum(det["confidence"] for det in detections)
                max_confidence = max(
                    max_confidence,
                    max(det["confidence"] for det in detections),
                )
                detection_score = sum(det["confidence"] for det in detections)
                if detection_score > best_detection_score:
                    best_detection_score = detection_score
                    best_focus_frame = frame.copy()
                    best_focus_detections = [det.copy() for det in detections]

            frame = draw_detections(frame, detections, phase_color)

            # FPS
            now = time.time()
            dt = now - prev_time
            if dt > 0:
                fps_display = 0.7 * fps_display + 0.3 * (1.0 / dt)
            prev_time = now

            frame = draw_hud(
                frame,
                fps=fps_display,
                frame_num=frame_count,
                total_frames=total_frames,
                phase_label=phase_label,
                detections_count=len(detections),
                model_loaded=model is not None,
                total_positives=total_positives,
                paused=paused,
            )

            if writer is not None:
                writer.write(frame)

            last_visual_frame = frame.copy()

        if last_visual_frame is None:
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                end_reason = "keyboard"
                break
            continue

        display_frame = last_visual_frame.copy()
        actions = [
            ("exit", "Finalizar", (70, 95, 220)),
            ("pause", "Reanudar" if paused else "Pausar", (90, 170, 245)),
            ("screenshot", "Captura", (170, 210, 120)),
        ]
        display_frame, buttons = _draw_action_buttons(
            display_frame, actions, title="Acciones"
        )
        ui_state["buttons"] = buttons
        last_frame = display_frame.copy()
        cv2.imshow(window_name, display_frame)

        key = cv2.waitKey(1) & 0xFF
        clicked = ui_state.pop("clicked", None)

        if key == ord("q") or clicked == "exit":
            end_reason = "button" if clicked == "exit" else "keyboard"
            break
        elif key == ord("s") or clicked == "screenshot":
            if last_frame is None:
                continue
            path = save_screenshot(last_frame, prefix=phase_color)
            print(f"    📸 Screenshot: {path}")
        elif key == ord("p") or clicked == "pause":
            paused = not paused

    cap.release()
    if writer is not None:
        writer.release()
    try:
        cv2.destroyWindow(window_name)
    except cv2.error:
        pass

    print(f"  Frames procesados: {frame_count}")
    print(f"  Frames con detección: {total_positives}")

    has_detections = unique_polyps > 0
    completion_ratio = (
        min(frame_count / total_frames, 1.0) if total_frames > 0 else None
    )
    avg_confidence = (
        confidence_sum / total_detections if total_detections > 0 else 0.0
    )
    stats = {
        "source_mode": mode,
        "frames_processed": frame_count,
        "positive_frames": total_positives,
        "total_detections": total_detections,
        "peak_detections": peak_detections,
        "unique_polyps": unique_polyps,
        "avg_confidence": avg_confidence,
        "max_confidence": max_confidence,
        "confidence_threshold": COLONOSCOPY_CONFIDENCE_THRESHOLD,
        "min_confirm_seconds": POLYP_CONFIRM_SECONDS,
        "min_confirm_frames": min_confirm_frames,
        "completion_ratio": completion_ratio,
        "end_reason": end_reason,
        "explanation": create_detection_explanation(
            best_focus_frame, best_focus_detections
        ),
    }
    return has_detections, stats


def run_classification_inference(
    model_bundle: dict, frame: np.ndarray
) -> tuple[str, float]:
    """
    Ejecuta clasificación sobre un frame.
    Devuelve (clase_predicha, confianza).
    """
    import torch

    model = model_bundle["model"]
    transform = model_bundle["transform"]
    device = model_bundle["device"]
    class_names = model_bundle["class_names"]

    # frame es BGR (OpenCV) → RGB para el transform
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = transform(rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        conf, cls_idx = probs.max(0)

    return class_names[cls_idx.item()], float(conf.item())


def process_video_classification(
    source,
    mode: str,
    model_bundle: dict | None,
    phase_label: str,
) -> tuple[bool, int]:
    """
    Procesa vídeo/webcam con un modelo de CLASIFICACIÓN.
    En vez de bounding boxes, muestra la predicción como overlay de texto.
    Devuelve (hubo_detecciones_cancer, total_frames_con_cancer).
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"  ✗ No se pudo abrir la fuente: {source}")
        return False, 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if mode == "Video" else 0
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  Resolución: {width}x{height}")
    if total_frames > 0:
        print(f"  Frames: {total_frames} | FPS fuente: {src_fps:.1f}")

    window_name = f"{phase_label} - {mode}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(width, 1280), min(height, 720))

    frame_count = 0
    total_cancer = 0
    paused = False
    prev_time = time.time()
    fps_display = 0.0
    last_frame = None

    print("  ▸ Procesando... (pulsa 'q' para finalizar fase)")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                if mode == "Video":
                    break
                else:
                    continue

            frame_count += 1
            if FRAME_SKIP > 1 and frame_count % FRAME_SKIP != 0:
                continue

            # Clasificación
            if model_bundle is not None:
                cls_name, confidence = run_classification_inference(model_bundle, frame)
                is_cancer = (cls_name == "colon_aca")
            else:
                # Modo demo
                cls_name = "DEMO"
                confidence = 0.0
                is_cancer = False

            if is_cancer and confidence >= CONFIDENCE_THRESHOLD:
                total_cancer += 1

            # Dibujar resultado de clasificación en el frame
            if is_cancer:
                label = f"CANCER: {confidence:.0%}"
                color = (0, 0, 255)  # Rojo
                # Borde rojo alrededor del frame
                cv2.rectangle(frame, (5, 5), (width - 5, height - 5), (0, 0, 255), 4)
            else:
                label = f"NORMAL: {confidence:.0%}"
                color = (0, 255, 0)  # Verde

            # Fondo semitransparente para el texto
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (400, 50), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            cv2.putText(
                frame, label, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA,
            )

            # FPS
            now = time.time()
            dt = now - prev_time
            if dt > 0:
                fps_display = 0.7 * fps_display + 0.3 * (1.0 / dt)
            prev_time = now

            # HUD
            frame = draw_hud(
                frame,
                fps=fps_display,
                frame_num=frame_count,
                total_frames=total_frames,
                phase_label=phase_label,
                detections_count=1 if (is_cancer and confidence >= CONFIDENCE_THRESHOLD) else 0,
                model_loaded=model_bundle is not None,
                total_positives=total_cancer,
            )

            last_frame = frame.copy()
            cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s") and last_frame is not None:
            path = save_screenshot(last_frame, prefix="cancer")
            print(f"    📸 Screenshot: {path}")
        elif key == ord("p"):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()

    print(f"  Frames procesados: {frame_count}")
    print(f"  Frames con cáncer: {total_cancer}")

    return total_cancer > 0, total_cancer


def process_image_classification(
    image_path: str,
    model_bundle: dict | None,
    phase_label: str,
) -> dict[str, Any]:
    """
    Procesa una sola imagen con un modelo de clasificación.
    Devuelve métricas resumidas de la clasificación.
    """
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"  ✗ No se pudo abrir la imagen: {image_path}")
        return {
            "image_path": image_path,
            "image_name": Path(image_path).name,
            "is_malignant": False,
            "confidence": 0.0,
            "class_name": "ERROR",
            "demo_mode": model_bundle is None,
            "explanation": None,
        }

    print(f"  Imagen cargada: {Path(image_path).name}")
    print(f"  Resolución: {frame.shape[1]}x{frame.shape[0]}")

    if model_bundle is not None:
        cls_name, confidence = run_classification_inference(model_bundle, frame)
        is_cancer = cls_name == "colon_aca"
    else:
        cls_name = "DEMO"
        confidence = 0.0
        is_cancer = False

    display = frame.copy()
    border_color = (0, 0, 255) if is_cancer else (0, 255, 0)
    status_text = "MALIGNO" if is_cancer else "NO MALIGNO"
    detail_text = f"{status_text}: {confidence:.1%}"

    cv2.rectangle(
        display,
        (5, 5),
        (display.shape[1] - 5, display.shape[0] - 5),
        border_color,
        4,
    )
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (520, 70), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)
    cv2.putText(
        display,
        phase_label,
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        display,
        detail_text,
        (12, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        border_color,
        2,
        cv2.LINE_AA,
    )

    window_name = f"{phase_label} - Imagen"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    _center_cv_window(window_name, display.shape[1], display.shape[0], max_height=760)

    ui_state: dict[str, Any] = {"buttons": {}, "clicked": None}
    _set_mouse_buttons_handler(window_name, ui_state)

    while True:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        frame_to_show = display.copy()
        actions = [
            ("continue", "Continuar", (90, 170, 245)),
            ("exit", "Cerrar vista", (70, 95, 220)),
        ]
        frame_to_show, buttons = _draw_action_buttons(
            frame_to_show, actions, title="Siguiente paso"
        )
        ui_state["buttons"] = buttons
        cv2.imshow(window_name, frame_to_show)

        key = cv2.waitKey(30) & 0xFF
        clicked = ui_state.pop("clicked", None)
        if key in (13, 27, ord("q"), ord(" ")):
            break
        if clicked in {"continue", "exit"}:
            break

    try:
        cv2.destroyWindow(window_name)
    except cv2.error:
        pass

    print(f"  Resultado: {status_text} ({confidence:.2%})")
    return {
        "image_path": image_path,
        "image_name": Path(image_path).name,
        "is_malignant": is_cancer and confidence >= CONFIDENCE_THRESHOLD,
        "confidence": confidence,
        "class_name": cls_name,
        "demo_mode": model_bundle is None,
        "explanation": create_gradcam_explanation(model_bundle, frame, cls_name),
    }


# ══════════════════════════════════════════════
# GUI – VENTANAS POR FASE
# ══════════════════════════════════════════════

def _center_window(win: tk.Tk | tk.Toplevel, w: int, h: int) -> None:
    """Centra una ventana en la pantalla."""
    win.update_idletasks()
    x = (win.winfo_screenwidth() // 2) - (w // 2)
    y = (win.winfo_screenheight() // 2) - (h // 2)
    win.geometry(f"{w}x{h}+{x}+{y}")
    win.lift()
    try:
        win.attributes("-topmost", True)
        win.after(150, lambda: win.attributes("-topmost", False))
    except tk.TclError:
        pass
    try:
        win.focus_force()
    except tk.TclError:
        pass


def _make_button(parent, text: str, color: str, command, **kwargs: Any) -> tk.Button:
    """Crea un botón estilizado."""
    defaults: dict[str, Any] = {
        "font": ("Segoe UI", 13, "bold"),
        "width": 32,
        "height": 2,
        "relief": "flat",
        "cursor": "hand2",
        "fg": BG_DARK,
        "activebackground": color,
    }
    defaults.update(kwargs)
    return tk.Button(parent, text=text, bg=color, command=command, **defaults)


def _back_button(parent, command) -> tk.Button:
    """Crea un botón de volver."""
    return tk.Button(
        parent, text="← Volver al inicio", bg=BTN_INACTIVE, fg=FG_TEXT,
        activebackground="#6c7086", command=command,
        font=("Segoe UI", 10), width=32, height=1, relief="flat", cursor="hand2",
    )


def _metric_row(parent, label: str, value: str, value_color: str = FG_TEXT) -> None:
    """Añade una fila sencilla de métrica a una tarjeta tkinter."""
    row = tk.Frame(parent, bg=BG_CARD)
    row.pack(fill="x", pady=3)
    tk.Label(
        row, text=label, font=("Segoe UI", 10, "bold"),
        fg=FG_SUB, bg=BG_CARD, anchor="w",
    ).pack(side="left")
    tk.Label(
        row, text=value, font=("Segoe UI", 10, "bold"),
        fg=value_color, bg=BG_CARD, anchor="e",
    ).pack(side="right")


# ── Pantalla principal ──────────────────────

def show_main_menu() -> str:
    """
    Pantalla principal.
    Devuelve 'start', 'phase1', 'phase2', 'phase3' o 'exit'.
    """
    result = {"action": "exit"}

    root = tk.Tk()
    root.title("AiColonDiagnosis")
    root.configure(bg=BG_DARK)
    root.resizable(False, False)

    tk.Label(
        root, text="🏥  AiColonDiagnosis",
        font=("Segoe UI", 22, "bold"), fg=FG_TEXT, bg=BG_DARK,
    ).pack(pady=(30, 5))

    tk.Label(
        root, text="Sistema de diagnóstico de cáncer de colon",
        font=("Segoe UI", 11), fg=FG_SUB, bg=BG_DARK,
    ).pack(pady=(0, 10))

    # Info de fases
    info_frame = tk.Frame(root, bg=BG_CARD, padx=20, pady=15)
    info_frame.pack(padx=30, pady=10, fill="x")

    phases = [
        ("Fase 1", "Historial médico del paciente", ACCENT_YELLOW),
        ("Fase 2", "Colonoscopia — detección de pólipos", ACCENT_GREEN),
        ("Fase 3", "Foto histológica — clasificación de malignidad", ACCENT_RED),
    ]
    for name, desc, color in phases:
        row = tk.Frame(info_frame, bg=BG_CARD)
        row.pack(fill="x", pady=3)
        tk.Label(
            row, text=f"  ● {name}:", font=("Segoe UI", 10, "bold"),
            fg=color, bg=BG_CARD, anchor="w",
        ).pack(side="left")
        tk.Label(
            row, text=f"  {desc}", font=("Segoe UI", 10),
            fg=FG_SUB, bg=BG_CARD, anchor="w",
        ).pack(side="left")

    tk.Label(root, text="", bg=BG_DARK).pack(pady=5)

    def on_start():
        result["action"] = "start"
        root.destroy()

    def on_exit():
        root.destroy()

    # Botón principal: flujo completo
    _make_button(
        root, "▶  Diagnóstico completo (3 fases)", ACCENT_BLUE, on_start,
    ).pack(pady=(0, 12))

    # Separador
    tk.Label(
        root, text="── o ir directamente a una fase ──",
        font=("Segoe UI", 9), fg=FG_SUB, bg=BG_DARK,
    ).pack(pady=(0, 8))

    # Botones de fases individuales
    phases_frame = tk.Frame(root, bg=BG_DARK)
    phases_frame.pack(pady=(0, 8))

    def make_phase_action(p):
        def action():
            result["action"] = p
            root.destroy()
        return action

    phase_btns = [
        ("📋 Historial", ACCENT_YELLOW, "phase1"),
        ("📹 Colonoscopia", ACCENT_GREEN, "phase2"),
        ("🔬 Foto histológica", ACCENT_MAUVE, "phase3"),
    ]
    for text, color, phase_key in phase_btns:
        tk.Button(
            phases_frame, text=text, bg=color, fg=BG_DARK,
            activebackground=color, command=make_phase_action(phase_key),
            font=("Segoe UI", 10, "bold"), width=17, height=1,
            relief="flat", cursor="hand2",
        ).pack(side="left", padx=5)

    tk.Label(root, text="", bg=BG_DARK).pack(pady=2)

    tk.Button(
        root, text="Salir", bg=BTN_INACTIVE, fg=FG_TEXT,
        activebackground="#6c7086", command=on_exit,
        font=("Segoe UI", 10), width=32, height=1, relief="flat", cursor="hand2",
    ).pack(pady=(0, 20))

    _center_window(root, 680, 520)
    root.mainloop()
    return result["action"]


# ── Fase 1: Selección de archivo ────────────

def show_phase1_menu() -> tuple[str, Optional[str]]:
    """
    Pantalla Fase 1. Devuelve ('load', filepath) o ('exit', None).
    """
    result: dict = {"action": "exit", "path": None}

    root = tk.Tk()
    root.title("Fase 1 — Historial Médico")
    root.configure(bg=BG_DARK)
    root.resizable(False, False)

    tk.Label(
        root, text="📋  Fase 1: Historial Médico",
        font=("Segoe UI", 18, "bold"), fg=ACCENT_YELLOW, bg=BG_DARK,
    ).pack(pady=(25, 5))

    tk.Label(
        root,
        text="Carga el historial clínico del paciente\n(formato CSV o JSON)",
        font=("Segoe UI", 11), fg=FG_SUB, bg=BG_DARK,
    ).pack(pady=(0, 20))

    def on_load():
        filepath = filedialog.askopenfilename(
            parent=root,
            title="Seleccionar historial del paciente",
            filetypes=[
                ("Datos", "*.csv *.json"),
                ("CSV", "*.csv"),
                ("JSON", "*.json"),
                ("Todos", "*.*"),
            ],
        )
        if filepath:
            result["action"] = "load"
            result["path"] = filepath
            root.destroy()

    def on_back():
        result["action"] = "exit"
        root.destroy()

    _make_button(root, "📂  Cargar historial", ACCENT_YELLOW, on_load).pack(pady=(0, 10))
    _back_button(root, on_back).pack(pady=(0, 20))

    _center_window(root, 480, 300)
    root.mainloop()
    return result["action"], result.get("path")


# ── Resultado Fase 1 ────────────────────────

def show_phase1_result(
    patient_data: dict,
    is_positive: bool,
    probability: float,
    demo_mode: bool = False,
    numeric_fields: int = 0,
) -> str:
    """
    Muestra el resultado de la Fase 1.
    Devuelve 'next' o 'restart'.
    """
    result = {"action": "restart"}

    root = tk.Tk()
    root.title("Fase 1 — Resultado")
    root.configure(bg=BG_DARK)
    root.resizable(False, False)

    tk.Label(
        root, text="📋  Resultado del análisis",
        font=("Segoe UI", 16, "bold"), fg=ACCENT_YELLOW, bg=BG_DARK,
    ).pack(pady=(20, 10))

    # Datos del paciente (resumen, máximo 8 campos)
    data_frame = tk.Frame(root, bg=BG_CARD, padx=15, pady=10)
    data_frame.pack(padx=25, pady=5, fill="x")

    for key, value in list(patient_data.items())[:8]:
        row = tk.Frame(data_frame, bg=BG_CARD)
        row.pack(fill="x", pady=1)
        tk.Label(
            row, text=f"{key}:", font=("Segoe UI", 9, "bold"),
            fg=FG_SUB, bg=BG_CARD, width=20, anchor="e",
        ).pack(side="left")
        tk.Label(
            row, text=f" {value}", font=("Segoe UI", 9),
            fg=FG_TEXT, bg=BG_CARD, anchor="w",
        ).pack(side="left")

    analysis_frame = tk.Frame(root, bg=BG_CARD, padx=15, pady=12)
    analysis_frame.pack(padx=25, pady=(8, 5), fill="x")
    _metric_row(analysis_frame, "Campos cargados", str(len(patient_data)))
    _metric_row(analysis_frame, "Campos numéricos usados", str(numeric_fields))
    _metric_row(
        analysis_frame,
        "Probabilidad estimada",
        f"{probability:.1%}",
        ACCENT_RED if is_positive else ACCENT_GREEN,
    )
    _metric_row(
        analysis_frame,
        "Modo de ejecución",
        "Demo (sin modelo clínico)" if demo_mode else "Modelo cargado",
        ACCENT_YELLOW if demo_mode else ACCENT_GREEN,
    )

    # Resultado positivo/negativo
    if is_positive:
        color = ACCENT_RED
        text = f"⚠️  RIESGO DETECTADO  ({probability:.0%})"
        subtext = "Se recomienda proceder a colonoscopia"
    else:
        color = ACCENT_GREEN
        text = f"✅  SIN RIESGO  ({probability:.0%})"
        subtext = "No se detectaron indicios de riesgo"

    tk.Label(root, text="", bg=BG_DARK).pack(pady=3)
    tk.Label(
        root, text=text,
        font=("Segoe UI", 14, "bold"), fg=color, bg=BG_DARK,
    ).pack()
    tk.Label(
        root, text=subtext,
        font=("Segoe UI", 10), fg=FG_SUB, bg=BG_DARK,
    ).pack(pady=(0, 15))

    if demo_mode:
        tk.Label(
            root,
            text="Nota: esta fase avanzó en modo demo porque falta el modelo clínico.",
            font=("Segoe UI", 9),
            fg=ACCENT_YELLOW,
            bg=BG_DARK,
        ).pack(pady=(0, 12))

    def on_next():
        result["action"] = "next"
        root.destroy()

    def on_restart():
        result["action"] = "restart"
        root.destroy()

    if is_positive:
        _make_button(
            root, "▶  Siguiente fase: Colonoscopia", ACCENT_GREEN, on_next,
        ).pack(pady=(0, 8))

    _back_button(root, on_restart).pack(pady=(0, 15))

    _center_window(root, 540, 620)
    root.mainloop()
    return result["action"]


# ── Fases 2/3: Menú de fuente de vídeo ─────

def show_video_menu(
    phase_num: int, phase_title: str, color: str
) -> tuple[str, Optional[str]]:
    """
    Menú para elegir vídeo o webcam.
    Devuelve ('video', path), ('webcam', None) o ('exit', None).
    """
    result: dict = {"action": "exit", "path": None}

    root = tk.Tk()
    root.title(f"Fase {phase_num} — {phase_title}")
    root.configure(bg=BG_DARK)
    root.resizable(False, False)

    emoji = "🔬" if phase_num == 3 else "📹"
    tk.Label(
        root, text=f"{emoji}  Fase {phase_num}: {phase_title}",
        font=("Segoe UI", 18, "bold"), fg=color, bg=BG_DARK,
    ).pack(pady=(25, 5))

    tk.Label(
        root, text="Selecciona la fuente de entrada",
        font=("Segoe UI", 11), fg=FG_SUB, bg=BG_DARK,
    ).pack(pady=(0, 20))
    tk.Label(
        root,
        text="Durante la fase verás botones visibles para finalizar, pausar y capturar.",
        font=("Segoe UI", 9),
        fg=FG_SUB,
        bg=BG_DARK,
        wraplength=360,
        justify="center",
    ).pack(pady=(0, 16))

    def on_video():
        filepath = filedialog.askopenfilename(
            parent=root,
            title=f"Seleccionar vídeo — {phase_title}",
            filetypes=[
                ("Vídeos", "*.mp4 *.avi *.mkv *.mov *.wmv"),
                ("Todos", "*.*"),
            ],
        )
        if filepath:
            result["action"] = "video"
            result["path"] = filepath
            root.destroy()

    def on_webcam():
        result["action"] = "webcam"
        root.destroy()

    def on_back():
        result["action"] = "exit"
        root.destroy()

    _make_button(root, "📹  Usar Vídeo", color, on_video).pack(pady=(0, 10))
    _make_button(root, "📷  Usar Webcam", ACCENT_BLUE, on_webcam).pack(pady=(0, 10))
    _back_button(root, on_back).pack(pady=(5, 20))

    _center_window(root, 500, 390)
    root.mainloop()
    return result["action"], result.get("path")


def show_image_menu(
    phase_num: int, phase_title: str, color: str
) -> tuple[str, Optional[str]]:
    """
    Menú para seleccionar una imagen estática.
    Devuelve ('image', path) o ('exit', None).
    """
    result: dict = {"action": "exit", "path": None}

    root = tk.Tk()
    root.title(f"Fase {phase_num} — {phase_title}")
    root.configure(bg=BG_DARK)
    root.resizable(False, False)

    tk.Label(
        root, text=f"🔬  Fase {phase_num}: {phase_title}",
        font=("Segoe UI", 18, "bold"), fg=color, bg=BG_DARK,
    ).pack(pady=(25, 5))

    tk.Label(
        root, text="Selecciona la imagen que quieres analizar",
        font=("Segoe UI", 11), fg=FG_SUB, bg=BG_DARK,
    ).pack(pady=(0, 20))
    tk.Label(
        root,
        text="La vista de imagen también mostrará controles visibles para continuar o cerrarla.",
        font=("Segoe UI", 9),
        fg=FG_SUB,
        bg=BG_DARK,
        wraplength=380,
        justify="center",
    ).pack(pady=(0, 16))

    def on_image():
        filepath = filedialog.askopenfilename(
            parent=root,
            title=f"Seleccionar imagen — {phase_title}",
            filetypes=[
                ("Imágenes", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("Todos", "*.*"),
            ],
        )
        if filepath:
            result["action"] = "image"
            result["path"] = filepath
            root.destroy()

    def on_back():
        result["action"] = "exit"
        root.destroy()

    _make_button(root, "🖼  Subir Imagen", color, on_image).pack(pady=(0, 10))
    _back_button(root, on_back).pack(pady=(5, 20))

    _center_window(root, 520, 340)
    root.mainloop()
    return result["action"], result.get("path")


# ── Resultado de fase de vídeo ──────────────

def show_video_result(
    phase_num: int,
    phase_title: str,
    stats: dict[str, Any],
    has_next_phase: bool,
) -> str:
    """
    Resultado tras fase de vídeo.
    Devuelve 'next', 'restart'.
    """
    result = {"action": "restart"}

    root = tk.Tk()
    root.title(f"Fase {phase_num} — Resultado")
    root.configure(bg=BG_DARK)
    root.resizable(False, False)

    tk.Label(
        root, text=f"Fase {phase_num}: {phase_title}",
        font=("Segoe UI", 16, "bold"), fg=FG_TEXT, bg=BG_DARK,
    ).pack(pady=(25, 10))

    has_detections = stats["unique_polyps"] > 0
    if has_detections:
        color = ACCENT_RED
        text = f"⚠️  {stats['unique_polyps']} POLIPOS CONFIRMADOS"
        subtext = "Se confirmaron regiones sospechosas persistentes durante el analisis."
    else:
        color = ACCENT_GREEN
        text = "✅  SIN POLIPOS CONFIRMADOS"
        subtext = "Las detecciones breves o inestables se descartaron como candidatas."

    tk.Label(
        root, text=text,
        font=("Segoe UI", 13, "bold"), fg=color, bg=BG_DARK,
    ).pack(pady=(10, 3))
    tk.Label(
        root, text=subtext,
        font=("Segoe UI", 10), fg=FG_SUB, bg=BG_DARK,
    ).pack(pady=(0, 20))

    metrics = tk.Frame(root, bg=BG_CARD, padx=18, pady=14)
    metrics.pack(padx=25, pady=(0, 14), fill="x")
    _metric_row(metrics, "Fuente", stats["source_mode"])
    _metric_row(metrics, "Frames procesados", str(stats["frames_processed"]))
    _metric_row(
        metrics,
        "Frames con candidatos",
        f"{stats['positive_frames']} ({_format_ratio(stats['positive_frames'], stats['frames_processed'])})",
        ACCENT_YELLOW if stats["positive_frames"] > 0 and not has_detections else color,
    )
    _metric_row(
        metrics,
        "Polipos confirmados",
        str(stats["unique_polyps"]),
        ACCENT_RED if stats["unique_polyps"] > 0 else ACCENT_GREEN,
    )
    _metric_row(metrics, "Detecciones totales", str(stats["total_detections"]))
    _metric_row(metrics, "Confianza minima", f"{stats['confidence_threshold']:.0%}")
    _metric_row(
        metrics,
        "Persistencia minima",
        f"{stats['min_confirm_seconds']:.1f}s ({stats['min_confirm_frames']} frames)",
    )
    _metric_row(metrics, "Pico en un frame", str(stats["peak_detections"]))
    _metric_row(metrics, "Confianza media", f"{stats['avg_confidence']:.1%}")
    _metric_row(metrics, "Confianza máxima", f"{stats['max_confidence']:.1%}")

    if stats["completion_ratio"] is not None:
        _metric_row(
            metrics,
            "Vídeo revisado",
            f"{stats['completion_ratio'] * 100:.1f}%",
        )

    ending_text = {
        "button": "La fase se cerró desde el botón Finalizar.",
        "keyboard": "La fase se cerró desde el teclado.",
        "window": "La fase se cerró desde la propia ventana.",
        "completed": "La fuente se analizó hasta el final.",
        "error": "No se pudo completar el análisis.",
    }.get(stats["end_reason"], "La fase terminó.")
    tk.Label(
        root,
        text=ending_text,
        font=("Segoe UI", 9),
        fg=FG_SUB,
        bg=BG_DARK,
    ).pack(pady=(0, 14))

    def on_next():
        result["action"] = "next"
        root.destroy()

    def on_restart():
        result["action"] = "restart"
        root.destroy()

    def on_explanation():
        show_explanation_window(stats.get("explanation"))

    if stats.get("explanation") is not None:
        _make_button(
            root,
            "🧠  Ver en que se ha fijado el modelo",
            ACCENT_BLUE,
            on_explanation,
        ).pack(pady=(0, 8))

    if has_detections and has_next_phase:
        if phase_num == 2:
            next_label = "▶  Siguiente: Análisis microscópico"
        else:
            next_label = "▶  Ver resultado final"
        _make_button(root, next_label, ACCENT_MAUVE, on_next).pack(pady=(0, 8))

    _back_button(root, on_restart).pack(pady=(0, 20))

    _center_window(root, 560, 510)
    root.mainloop()
    return result["action"]


def show_classification_result(
    phase_num: int,
    phase_title: str,
    result_data: dict[str, Any],
    has_next_phase: bool,
) -> str:
    """
    Resultado tras clasificar una sola imagen.
    Devuelve 'next' o 'restart'.
    """
    result = {"action": "restart"}

    root = tk.Tk()
    root.title(f"Fase {phase_num} — Resultado")
    root.configure(bg=BG_DARK)
    root.resizable(False, False)

    tk.Label(
        root, text=f"Fase {phase_num}: {phase_title}",
        font=("Segoe UI", 16, "bold"), fg=FG_TEXT, bg=BG_DARK,
    ).pack(pady=(25, 10))

    is_malignant = result_data["is_malignant"]
    confidence = result_data["confidence"]
    class_name = result_data["class_name"]
    if is_malignant:
        color = ACCENT_RED
        text = "⚠️  MUESTRA MALIGNA"
        subtext = f"El modelo detecta malignidad con confianza {confidence:.1%}"
    else:
        color = ACCENT_GREEN
        text = "✅  MUESTRA NO MALIGNA"
        subtext = (
            f"El modelo la clasifica como no maligna con confianza {confidence:.1%}"
        )

    tk.Label(
        root, text=text,
        font=("Segoe UI", 13, "bold"), fg=color, bg=BG_DARK,
    ).pack(pady=(10, 3))
    tk.Label(
        root, text=subtext,
        font=("Segoe UI", 10), fg=FG_SUB, bg=BG_DARK,
    ).pack(pady=(0, 10))

    metrics = tk.Frame(root, bg=BG_CARD, padx=18, pady=14)
    metrics.pack(padx=25, pady=(0, 16), fill="x")
    _metric_row(metrics, "Imagen analizada", result_data["image_name"])
    _metric_row(metrics, "Clase predicha", class_name)
    _metric_row(
        metrics,
        "Confianza del modelo",
        f"{confidence:.1%}",
        color,
    )
    _metric_row(
        metrics,
        "Interpretación",
        "Maligna" if is_malignant else "No maligna",
        color,
    )
    _metric_row(
        metrics,
        "Modo de ejecución",
        "Demo (sin modelo)" if result_data["demo_mode"] else "Modelo cargado",
        ACCENT_YELLOW if result_data["demo_mode"] else ACCENT_GREEN,
    )

    note = (
        "La clasificación histológica apunta a malignidad y debe revisarse clínicamente."
        if is_malignant
        else "La imagen no fue clasificada como maligna en esta fase."
    )
    tk.Label(
        root, text=note,
        font=("Segoe UI", 9), fg=FG_SUB, bg=BG_DARK, wraplength=450,
        justify="center",
    ).pack(pady=(0, 14))

    def on_next():
        result["action"] = "next"
        root.destroy()

    def on_restart():
        result["action"] = "restart"
        root.destroy()

    def on_explanation():
        show_explanation_window(result_data.get("explanation"))

    if result_data.get("explanation") is not None:
        _make_button(
            root,
            "🧠  Ver en que se ha fijado el modelo",
            ACCENT_BLUE,
            on_explanation,
        ).pack(pady=(0, 8))

    if has_next_phase:
        _make_button(root, "▶  Ver resultado final", ACCENT_MAUVE, on_next).pack(
            pady=(0, 8)
        )

    _back_button(root, on_restart).pack(pady=(0, 20))

    _center_window(root, 560, 500)
    root.mainloop()
    return result["action"]


# ── Resultado final ─────────────────────────

def show_final_result(
    phase1_result: dict[str, Any],
    phase2_result: dict[str, Any],
    phase3_result: dict[str, Any],
) -> None:
    """Pantalla de resultado final del diagnóstico completo."""
    root = tk.Tk()
    root.title("Resultado Final — AiColonDiagnosis")
    root.configure(bg=BG_DARK)
    root.resizable(False, False)

    tk.Label(
        root, text="🏥  Resultado Final del Diagnóstico",
        font=("Segoe UI", 18, "bold"), fg=FG_TEXT, bg=BG_DARK,
    ).pack(pady=(25, 15))

    # Resumen
    summary = tk.Frame(root, bg=BG_CARD, padx=20, pady=15)
    summary.pack(padx=30, pady=5, fill="x")

    phase1_positive = phase1_result["is_positive"]
    phase2_positive_frames = phase2_result["positive_frames"]
    phase3_malignant = phase3_result["is_malignant"]
    rows = [
        ("Fase 1 — Historial:",
         (
             "No ejecutada"
             if not phase1_result.get("executed") else
             f"Riesgo detectado ({phase1_result['probability']:.1%})"
             if phase1_positive else
             f"Sin riesgo ({phase1_result['probability']:.1%})"
         ),
         ACCENT_YELLOW if not phase1_result.get("executed")
         else ACCENT_RED if phase1_positive else ACCENT_GREEN),
        ("Fase 2 — Colonoscopia:",
         (
             "No ejecutada"
             if not phase2_result.get("executed") else
              f"{phase2_result['unique_polyps']} polipos confirmados, "
              f"{phase2_positive_frames} frames candidatos"
         ),
         ACCENT_YELLOW if not phase2_result.get("executed")
          else ACCENT_RED if phase2_result["unique_polyps"] > 0 else ACCENT_GREEN),
        ("Fase 3 — Foto histológica:",
         (
             "No ejecutada"
             if not phase3_result.get("executed") else
             f"Muestra maligna ({phase3_result['confidence']:.1%})"
             if phase3_malignant else
             f"Muestra no maligna ({phase3_result['confidence']:.1%})"
         ),
         ACCENT_YELLOW if not phase3_result.get("executed")
         else ACCENT_RED if phase3_malignant else ACCENT_GREEN),
    ]
    for label, value, color in rows:
        row = tk.Frame(summary, bg=BG_CARD)
        row.pack(fill="x", pady=3)
        tk.Label(
            row, text=label, font=("Segoe UI", 10, "bold"),
            fg=FG_SUB, bg=BG_CARD, width=24, anchor="e",
        ).pack(side="left")
        tk.Label(
            row, text=f"  {value}", font=("Segoe UI", 10, "bold"),
            fg=color, bg=BG_CARD, anchor="w",
        ).pack(side="left")

    # Conclusión
    if phase3_malignant:
        conclusion = "⚠️  Posible resultado final: muestra maligna"
        detail = (
            "La fase histológica fue positiva. Es la señal más fuerte del flujo "
            "actual y conviene una revisión médica prioritaria."
        )
        conclusion_color = ACCENT_RED
    elif phase1_positive and phase2_positive_frames > 0:
        conclusion = "⚠️  Posible resultado final: caso sospechoso"
        detail = (
            "Hubo riesgo inicial y hallazgos en colonoscopia, aunque la imagen "
            "histológica no se clasificó como maligna."
        )
        conclusion_color = ACCENT_YELLOW
    elif phase2_positive_frames > 0:
        conclusion = "⚠️  Posible resultado final: hallazgos a revisar"
        detail = (
            "Se detectaron pólipos o regiones sospechosas en colonoscopia. "
            "No equivale por sí solo a malignidad."
        )
        conclusion_color = ACCENT_YELLOW
    else:
        conclusion = "✅  Posible resultado final: sin hallazgos malignos claros"
        detail = (
            "En el flujo ejecutado no apareció evidencia fuerte de malignidad."
        )
        conclusion_color = ACCENT_GREEN

    tk.Label(root, text="", bg=BG_DARK).pack(pady=3)
    tk.Label(
        root, text=conclusion,
        font=("Segoe UI", 12, "bold"), fg=conclusion_color, bg=BG_DARK,
    ).pack(pady=(5, 15))
    tk.Label(
        root, text=detail,
        font=("Segoe UI", 10), fg=FG_SUB, bg=BG_DARK,
        wraplength=560, justify="center",
    ).pack(pady=(0, 12))

    analysis = tk.Frame(root, bg=BG_CARD, padx=20, pady=14)
    analysis.pack(padx=30, pady=(0, 16), fill="x")
    _metric_row(
        analysis,
        "Fase 1 en modo demo",
        "Sí" if phase1_result["demo_mode"] else "No",
        ACCENT_YELLOW if phase1_result["demo_mode"] else ACCENT_GREEN,
    )
    _metric_row(
        analysis,
        "Cobertura colonoscopia",
        (
            f"{phase2_result['completion_ratio'] * 100:.1f}%"
            if phase2_result["completion_ratio"] is not None else
            "Webcam / sin duración fija"
            if phase2_result.get("executed") else
            "No ejecutada"
        ),
    )
    _metric_row(
        analysis,
        "Polipos confirmados",
        str(phase2_result["unique_polyps"]) if phase2_result.get("executed") else "No ejecutada",
        ACCENT_YELLOW if not phase2_result.get("executed")
        else ACCENT_RED if phase2_result["unique_polyps"] > 0 else ACCENT_GREEN,
    )
    _metric_row(
        analysis,
        "Clase histológica",
        phase3_result["class_name"] if phase3_result.get("executed") else "No ejecutada",
        ACCENT_YELLOW if not phase3_result.get("executed")
        else ACCENT_RED if phase3_malignant else ACCENT_GREEN,
    )

    tk.Button(
        root, text="Cerrar", bg=BTN_INACTIVE, fg=FG_TEXT,
        activebackground="#6c7086", command=root.destroy,
        font=("Segoe UI", 11, "bold"), width=32, height=2,
        relief="flat", cursor="hand2",
    ).pack(pady=(0, 20))

    _center_window(root, 640, 520)
    root.mainloop()


# ══════════════════════════════════════════════
# FLUJO PRINCIPAL
# ══════════════════════════════════════════════

def main() -> None:
    print("=" * 55)
    print("  AiColonDiagnosis — Sistema de Diagnóstico")
    print("=" * 55)
    print()

    # Cargar modelos al inicio
    print("▸ Cargando modelos...")
    history_model = load_history_model(MODEL_HISTORY)
    colonoscopy_model = load_yolo_model(MODEL_COLONOSCOPY, "Colonoscopia")
    microscopy_model = load_classification_model(
        MODEL_MICROSCOPY, MODEL_MICROSCOPY_META, "Microscopio",
    )
    print()

    # ── Bucle principal (permite reiniciar) ──
    while True:

        # ── MENÚ PRINCIPAL ──
        action = show_main_menu()
        if action == "exit":
            break

        # ───────────────────────────────────────────
        # Funciones auxiliares para ejecutar cada fase
        # ───────────────────────────────────────────

        def run_phase1() -> tuple[bool, dict[str, Any]]:
            """Ejecuta Fase 1. Devuelve (quiere_avanzar, resumen_de_fase)."""
            print("─" * 55)
            print("  FASE 1: HISTORIAL MÉDICO")
            print("─" * 55)

            act, filepath = show_phase1_menu()
            if act != "load" or filepath is None:
                return False, {
                    "executed": False,
                    "status": "OMITIDO",
                    "is_positive": False,
                    "probability": 0.0,
                    "demo_mode": history_model is None,
                    "numeric_fields": 0,
                    "total_fields": 0,
                }

            patient_data = load_patient_data(filepath)
            if patient_data is None:
                messagebox.showerror(
                    "Error", "No se pudieron cargar los datos del paciente."
                )
                return False, {
                    "executed": False,
                    "status": "ERROR",
                    "is_positive": False,
                    "probability": 0.0,
                    "demo_mode": history_model is None,
                    "numeric_fields": 0,
                    "total_fields": 0,
                }

            print(f"  ✓ Datos cargados: {len(patient_data)} campos")
            is_pos, prob = predict_cancer_risk(history_model, patient_data)
            numeric_fields = sum(1 for v in patient_data.values() if _is_numeric(v))
            st = "RIESGO" if is_pos else "SIN RIESGO"
            print(f"  Resultado: {st} ({prob:.2%})")
            phase1_result = {
                "executed": True,
                "status": st,
                "is_positive": is_pos,
                "probability": prob,
                "demo_mode": history_model is None,
                "numeric_fields": numeric_fields,
                "total_fields": len(patient_data),
            }

            act = show_phase1_result(
                patient_data,
                is_pos,
                prob,
                demo_mode=phase1_result["demo_mode"],
                numeric_fields=numeric_fields,
            )
            if act == "next":
                return True, phase1_result
            return False, phase1_result

        def run_phase2(has_next_phase: bool = False) -> tuple[bool, dict[str, Any]]:
            """Ejecuta Fase 2. Devuelve (quiere_avanzar, resumen_de_fase)."""
            print()
            print("─" * 55)
            print("  FASE 2: COLONOSCOPIA")
            print("─" * 55)

            act, video_path = show_video_menu(2, "Colonoscopia", ACCENT_GREEN)
            if act == "exit":
                return False, {
                    "executed": False,
                    "source_mode": "No iniciado",
                    "frames_processed": 0,
                    "positive_frames": 0,
                    "total_detections": 0,
                    "peak_detections": 0,
                    "unique_polyps": 0,
                    "avg_confidence": 0.0,
                    "max_confidence": 0.0,
                    "confidence_threshold": COLONOSCOPY_CONFIDENCE_THRESHOLD,
                    "min_confirm_seconds": POLYP_CONFIRM_SECONDS,
                    "min_confirm_frames": 0,
                    "completion_ratio": None,
                    "end_reason": "cancelled",
                    "explanation": None,
                }

            source = video_path if act == "video" else WEBCAM_INDEX
            mode = "Video" if act == "video" else "Webcam"
            print(f"  Modo: {mode}")

            has_det, phase2_stats = process_video_phase(
                source, mode, colonoscopy_model,
                phase_label="Fase 2: Colonoscopia",
                phase_color="polyp",
            )

            act = show_video_result(
                phase_num=2,
                phase_title="Colonoscopia",
                stats=phase2_stats,
                has_next_phase=has_next_phase,
            )
            if act == "next":
                phase2_stats["executed"] = True
                return True, phase2_stats
            phase2_stats["executed"] = True
            return False, phase2_stats

        def run_phase3(has_next_phase: bool = False) -> tuple[bool, dict[str, Any]]:
            """Ejecuta Fase 3. Devuelve (quiere_ver_resultado, resumen_de_fase)."""
            print()
            print("─" * 55)
            print("  FASE 3: ANÁLISIS MICROSCÓPICO")
            print("─" * 55)

            act, image_path = show_image_menu(
                3, "Análisis Microscópico", ACCENT_MAUVE
            )
            if act == "exit":
                return False, {
                    "executed": False,
                    "image_path": "",
                    "image_name": "",
                    "is_malignant": False,
                    "confidence": 0.0,
                    "class_name": "NO_INICIADO",
                    "demo_mode": microscopy_model is None,
                    "explanation": None,
                }

            phase3_result = process_image_classification(
                image_path, microscopy_model,
                phase_label="Fase 3: Foto histológica",
            )
            phase3_result["executed"] = True

            act = show_classification_result(
                phase_num=3,
                phase_title="Análisis Microscópico",
                result_data=phase3_result,
                has_next_phase=has_next_phase,
            )
            if act == "next":
                return True, phase3_result
            return False, phase3_result

        # ───────────────────────────────────────────
        # Ejecutar según la acción del menú principal
        # ───────────────────────────────────────────

        if action == "start":
            # ══ FLUJO COMPLETO: Fase 1 → 2 → 3 → Resultado ══
            default_phase2_result = {
                "executed": False,
                "source_mode": "No iniciado",
                "frames_processed": 0,
                "positive_frames": 0,
                "total_detections": 0,
                "peak_detections": 0,
                "unique_polyps": 0,
                "avg_confidence": 0.0,
                "max_confidence": 0.0,
                "confidence_threshold": COLONOSCOPY_CONFIDENCE_THRESHOLD,
                "min_confirm_seconds": POLYP_CONFIRM_SECONDS,
                "min_confirm_frames": 0,
                "completion_ratio": None,
                "end_reason": "cancelled",
                "explanation": None,
            }
            default_phase3_result = {
                "executed": False,
                "image_path": "",
                "image_name": "",
                "is_malignant": False,
                "confidence": 0.0,
                "class_name": "NO_EJECUTADA",
                "demo_mode": microscopy_model is None,
                "explanation": None,
            }

            advance, phase1_result = run_phase1()
            if not advance:
                if phase1_result.get("executed") and not phase1_result["is_positive"]:
                    show_final_result(
                        phase1_result=phase1_result,
                        phase2_result=default_phase2_result,
                        phase3_result=default_phase3_result,
                    )
                continue

            advance, phase2_result = run_phase2(has_next_phase=True)
            if not advance:
                if phase2_result.get("executed") and phase2_result["unique_polyps"] == 0:
                    show_final_result(
                        phase1_result=phase1_result,
                        phase2_result=phase2_result,
                        phase3_result=default_phase3_result,
                    )
                continue

            advance, phase3_result = run_phase3(has_next_phase=True)
            if advance:
                print()
                print("═" * 55)
                print("  RESULTADO FINAL")
                print("═" * 55)
                print(f"  Fase 1 — Historial:    {phase1_result['status']}")
                print(
                    "  Fase 2 — Colonoscopia: "
                    f"{phase2_result['unique_polyps']} polipos confirmados, "
                    f"{phase2_result['positive_frames']} frames candidatos"
                )
                print(
                    "  Fase 3 — Foto histológica:  "
                    + (
                        "muestra maligna"
                        if phase3_result["is_malignant"]
                        else "muestra no maligna"
                    )
                )
                print("═" * 55)

                show_final_result(
                    phase1_result=phase1_result,
                    phase2_result=phase2_result,
                    phase3_result=phase3_result,
                )

        elif action == "phase1":
            # ══ SOLO FASE 1 ══
            run_phase1()

        elif action == "phase2":
            # ══ SOLO FASE 2: Colonoscopia ══
            run_phase2()

        elif action == "phase3":
            # ══ SOLO FASE 3: Foto histológica ══
            run_phase3()

        # Vuelve al menú principal automáticamente

    print()
    print("✓ Aplicación cerrada.")


if __name__ == "__main__":
    main()
