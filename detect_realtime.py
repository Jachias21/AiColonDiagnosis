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

def run_inference(model: Any, frame: np.ndarray) -> list[dict]:
    """Ejecuta YOLO sobre un frame y devuelve las detecciones."""
    results = model.predict(
        source=frame,
        imgsz=INFERENCE_SIZE,
        conf=CONFIDENCE_THRESHOLD,
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
        label = f"{cls_name} {conf:.2f}"

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

    y_offset = 25
    for line in lines:
        cv2.putText(
            frame, line, (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA,
        )
        y_offset += 22

    controls = "q:Finalizar fase | s:Screenshot | p:Pausa"
    cv2.putText(
        frame, controls, (10, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA,
    )
    return frame


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
) -> tuple[bool, int]:
    """
    Procesa vídeo/webcam con un modelo YOLO.
    Devuelve (hubo_detecciones, total_frames_con_detección).
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

    # Writer opcional
    writer = None
    if SAVE_OUTPUT and mode == "Video":
        out_name = f"output_{phase_color}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_name, fourcc, src_fps, (width, height))

    window_name = f"{phase_label} - {mode}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(width, 1280), min(height, 720))

    frame_count = 0
    total_positives = 0
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

            # Inferencia
            detections: list[dict] = []
            if model is not None:
                detections = run_inference(model, frame)
            if detections:
                total_positives += 1

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
            )

            if writer is not None:
                writer.write(frame)

            last_frame = frame.copy()
            cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s") and last_frame is not None:
            path = save_screenshot(last_frame, prefix=phase_color)
            print(f"    📸 Screenshot: {path}")
        elif key == ord("p"):
            paused = not paused

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    print(f"  Frames procesados: {frame_count}")
    print(f"  Frames con detección: {total_positives}")

    has_detections = total_positives > 0
    return has_detections, total_positives


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
) -> tuple[bool, float, str]:
    """
    Procesa una sola imagen con un modelo de clasificación.
    Devuelve (es_maligno, confianza, nombre_clase).
    """
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"  ✗ No se pudo abrir la imagen: {image_path}")
        return False, 0.0, "ERROR"

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
    cv2.imshow(window_name, display)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)

    print(f"  Resultado: {status_text} ({confidence:.2%})")
    return is_cancer and confidence >= CONFIDENCE_THRESHOLD, confidence, cls_name


# ══════════════════════════════════════════════
# GUI – VENTANAS POR FASE
# ══════════════════════════════════════════════

def _center_window(win: tk.Tk | tk.Toplevel, w: int, h: int) -> None:
    """Centra una ventana en la pantalla."""
    win.update_idletasks()
    x = (win.winfo_screenwidth() // 2) - (w // 2)
    y = (win.winfo_screenheight() // 2) - (h // 2)
    win.geometry(f"{w}x{h}+{x}+{y}")


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
    patient_data: dict, is_positive: bool, probability: float
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

    _center_window(root, 500, 480)
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

    def on_video():
        filepath = filedialog.askopenfilename(
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

    _center_window(root, 480, 340)
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

    def on_image():
        filepath = filedialog.askopenfilename(
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

    _center_window(root, 500, 290)
    root.mainloop()
    return result["action"], result.get("path")


# ── Resultado de fase de vídeo ──────────────

def show_video_result(
    phase_num: int,
    phase_title: str,
    has_detections: bool,
    total_positives: int,
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

    if has_detections:
        color = ACCENT_RED
        text = f"⚠️  DETECCIONES: {total_positives} frames"
        subtext = "Se encontraron regiones sospechosas"
    else:
        color = ACCENT_GREEN
        text = "✅  SIN DETECCIONES"
        subtext = "No se encontraron regiones sospechosas"

    tk.Label(
        root, text=text,
        font=("Segoe UI", 13, "bold"), fg=color, bg=BG_DARK,
    ).pack(pady=(10, 3))
    tk.Label(
        root, text=subtext,
        font=("Segoe UI", 10), fg=FG_SUB, bg=BG_DARK,
    ).pack(pady=(0, 20))

    def on_next():
        result["action"] = "next"
        root.destroy()

    def on_restart():
        result["action"] = "restart"
        root.destroy()

    if has_detections and has_next_phase:
        if phase_num == 2:
            next_label = "▶  Siguiente: Análisis microscópico"
        else:
            next_label = "▶  Ver resultado final"
        _make_button(root, next_label, ACCENT_MAUVE, on_next).pack(pady=(0, 8))

    _back_button(root, on_restart).pack(pady=(0, 20))

    _center_window(root, 500, 320)
    root.mainloop()
    return result["action"]


def show_classification_result(
    phase_num: int,
    phase_title: str,
    is_malignant: bool,
    confidence: float,
    class_name: str,
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
    tk.Label(
        root, text=f"Clase predicha: {class_name}",
        font=("Segoe UI", 10, "bold"), fg=FG_TEXT, bg=BG_DARK,
    ).pack(pady=(0, 20))

    def on_next():
        result["action"] = "next"
        root.destroy()

    def on_restart():
        result["action"] = "restart"
        root.destroy()

    if has_next_phase:
        _make_button(root, "▶  Ver resultado final", ACCENT_MAUVE, on_next).pack(
            pady=(0, 8)
        )

    _back_button(root, on_restart).pack(pady=(0, 20))

    _center_window(root, 520, 340)
    root.mainloop()
    return result["action"]


# ── Resultado final ─────────────────────────

def show_final_result(
    phase1_positive: bool,
    phase2_positives: int,
    phase3_positives: int,
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

    rows = [
        ("Fase 1 — Historial:",
         "Riesgo detectado" if phase1_positive else "Sin riesgo",
         ACCENT_RED if phase1_positive else ACCENT_GREEN),
        ("Fase 2 — Colonoscopia:",
         f"{phase2_positives} frames con pólipos",
         ACCENT_RED if phase2_positives > 0 else ACCENT_GREEN),
        ("Fase 3 — Foto histológica:",
         "Muestra maligna" if phase3_positives > 0 else "Muestra no maligna",
         ACCENT_RED if phase3_positives > 0 else ACCENT_GREEN),
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
    all_positive = (
        phase1_positive and phase2_positives > 0 and phase3_positives > 0
    )
    if all_positive:
        conclusion = "⚠️  Se recomienda consulta especializada urgente"
        conclusion_color = ACCENT_RED
    else:
        conclusion = "ℹ️  Resultados parcialmente negativos"
        conclusion_color = ACCENT_YELLOW

    tk.Label(root, text="", bg=BG_DARK).pack(pady=3)
    tk.Label(
        root, text=conclusion,
        font=("Segoe UI", 12, "bold"), fg=conclusion_color, bg=BG_DARK,
    ).pack(pady=(5, 15))

    tk.Button(
        root, text="Cerrar", bg=BTN_INACTIVE, fg=FG_TEXT,
        activebackground="#6c7086", command=root.destroy,
        font=("Segoe UI", 11, "bold"), width=32, height=2,
        relief="flat", cursor="hand2",
    ).pack(pady=(0, 20))

    _center_window(root, 520, 380)
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

        def run_phase1() -> tuple[bool, str]:
            """Ejecuta Fase 1. Devuelve (quiere_avanzar, status_text)."""
            print("─" * 55)
            print("  FASE 1: HISTORIAL MÉDICO")
            print("─" * 55)

            act, filepath = show_phase1_menu()
            if act != "load" or filepath is None:
                return False, "OMITIDO"

            patient_data = load_patient_data(filepath)
            if patient_data is None:
                messagebox.showerror(
                    "Error", "No se pudieron cargar los datos del paciente."
                )
                return False, "ERROR"

            print(f"  ✓ Datos cargados: {len(patient_data)} campos")
            is_pos, prob = predict_cancer_risk(history_model, patient_data)
            st = "RIESGO" if is_pos else "SIN RIESGO"
            print(f"  Resultado: {st} ({prob:.2%})")

            act = show_phase1_result(patient_data, is_pos, prob)
            if act == "next":
                return True, st
            return False, st

        def run_phase2() -> tuple[bool, int]:
            """Ejecuta Fase 2. Devuelve (tiene_detecciones, conteo)."""
            print()
            print("─" * 55)
            print("  FASE 2: COLONOSCOPIA")
            print("─" * 55)

            act, video_path = show_video_menu(2, "Colonoscopia", ACCENT_GREEN)
            if act == "exit":
                return False, 0

            source = video_path if act == "video" else WEBCAM_INDEX
            mode = "Video" if act == "video" else "Webcam"
            print(f"  Modo: {mode}")

            has_det, count = process_video_phase(
                source, mode, colonoscopy_model,
                phase_label="Fase 2: Colonoscopia",
                phase_color="polyp",
            )

            act = show_video_result(
                phase_num=2,
                phase_title="Colonoscopia",
                has_detections=has_det,
                total_positives=count,
                has_next_phase=True,
            )
            if act == "next":
                return True, count
            return False, count

        def run_phase3() -> tuple[bool, int]:
            """Ejecuta Fase 3. Devuelve (quiere_ver_resultado, conteo)."""
            print()
            print("─" * 55)
            print("  FASE 3: ANÁLISIS MICROSCÓPICO")
            print("─" * 55)

            act, image_path = show_image_menu(
                3, "Análisis Microscópico", ACCENT_MAUVE
            )
            if act == "exit":
                return False, 0

            has_det, confidence, class_name = process_image_classification(
                image_path, microscopy_model,
                phase_label="Fase 3: Foto histológica",
            )

            act = show_classification_result(
                phase_num=3,
                phase_title="Análisis Microscópico",
                is_malignant=has_det,
                confidence=confidence,
                class_name=class_name,
                has_next_phase=True,
            )
            if act == "next":
                return True, 1 if has_det else 0
            return False, 1 if has_det else 0

        # ───────────────────────────────────────────
        # Ejecutar según la acción del menú principal
        # ───────────────────────────────────────────

        if action == "start":
            # ══ FLUJO COMPLETO: Fase 1 → 2 → 3 → Resultado ══
            advance, status = run_phase1()
            if not advance:
                continue

            advance, polyp_count = run_phase2()
            if not advance:
                continue

            advance, cancer_count = run_phase3()
            if advance:
                print()
                print("═" * 55)
                print("  RESULTADO FINAL")
                print("═" * 55)
                print(f"  Fase 1 — Historial:    {status}")
                print(f"  Fase 2 — Colonoscopia: {polyp_count} detecciones")
                print(
                    "  Fase 3 — Foto histológica:  "
                    + ("muestra maligna" if cancer_count > 0 else "muestra no maligna")
                )
                print("═" * 55)

                show_final_result(
                    phase1_positive=(status == "RIESGO"),
                    phase2_positives=polyp_count,
                    phase3_positives=cancer_count,
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
