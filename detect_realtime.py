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
import shutil
import sys
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any, Optional
import pandas as pd

# Forzar UTF-8 en stdout/stderr para Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np
from PIL import Image, ImageTk

# ══════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════

# Rutas a los modelos (cambiar cuando estén entrenados)
MODEL_HISTORY: str = "models/catboost_crc_risk_model.cbm"     # Modelo de historial clínico
MODEL_COLONOSCOPY: str = "models/colonoscopy.pt"    # YOLO para pólipos
MODEL_COLONOSCOPY_SEGMENTER: str = "models/colonoscopy_unet3plus_effnet.pt"  # UNet3+ comparativo
MODEL_MICROSCOPY: str = "models/microscopy.pt"      # Clasificación tejido
MODEL_MICROSCOPY_META: str = "models/microscopy_meta.json"  # Metadata del modelo ganador

# Umbrales
CONFIDENCE_THRESHOLD: float = 0.25
HISTORY_RISK_THRESHOLD: float = 0.5    # Umbral para considerar riesgo positivo
COLONOSCOPY_CONFIDENCE_THRESHOLD: float = 0.50  # Reduce falsos positivos de YOLO
COLONOSCOPY_SEGMENTER_THRESHOLD: float = 0.50
COLONOSCOPY_SEGMENTER_MIN_AREA_RATIO: float = 0.00035
POLYP_CONFIRM_SECONDS: float = 0.5      # Tiempo minimo persistente para contar polipo con UNet3+
POLYP_TRACK_IOU_THRESHOLD: float = 0.25
POLYP_TRACK_MAX_MISSING_SECONDS: float = 4.0

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
ANALYSIS_HISTORY_DIR: str = "patients_history"

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


def load_polyp_segmenter(model_path: str = MODEL_COLONOSCOPY_SEGMENTER, label: str = "Segmentador colonoscopia") -> dict[str, Any] | None:
    """Carga el UNet3+ EfficientNet entrenado como comparador visual de Fase 2."""
    model_file = Path(model_path)
    if not model_file.exists():
        print(f"  ⚠ Modelo {label} no encontrado: '{model_path}' → comparador desactivado")
        return None
    try:
        import torch
        from train_models.model_colonoscopia import train_pretrained_polyp_segmenter as segmenter

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(str(model_file), map_location=device, weights_only=False)
        model = segmenter.load_pretrained_model(device)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        model.eval()
        if device.type == "cuda":
            model.to(memory_format=torch.channels_last)
        image_size = int(checkpoint.get("image_size", checkpoint.get("config", {}).get("image_size", 352)))
        threshold = float(checkpoint.get("threshold", checkpoint.get("config", {}).get("threshold", COLONOSCOPY_SEGMENTER_THRESHOLD)))
        print(f"  ✓ Modelo {label} cargado: {model_path} ({image_size}px)")
        return {
            "model": model,
            "device": device,
            "image_size": image_size,
            "threshold": threshold,
            "architecture": checkpoint.get("architecture", "hf_unet3plus_efficientnet"),
        }
    except Exception as e:
        print(f"  ✗ Error cargando {label}: {e}")
        return None


def load_history_model(model_path: str) -> Any:
    """Carga el modelo CatBoost nativo."""
    model_file = Path(model_path)
    if not model_file.exists():
        print(f"  ⚠ Modelo historial no encontrado: '{model_path}' → modo DEMO")
        return None
    try:
        from catboost import CatBoostClassifier
        model = CatBoostClassifier()
        model.load_model(str(model_file))
        print(f"  ✓ Modelo CatBoost cargado: {model_path}")
        return model
    except ImportError:
        print("  ✗ CatBoost no instalado. Ejecuta: uv pip install catboost")
        return None
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


HISTORY_MODEL_COLUMNS = [
    "Age", "Gender", "Country", "Urban_or_Rural", "Family_History",
    "Inflammatory_Bowel_Disease", "Obesity_BMI", "Diabetes",
    "Smoking_History", "Alcohol_Consumption", "Diet_Risk",
    "Physical_Activity", "Screening_History",
]
HISTORY_CAT_FEATURES = ["Gender", "Country", "Urban_or_Rural"]
HISTORY_FEATURE_LABELS = {
    "Age": "Edad",
    "Gender": "Genero",
    "Country": "Pais",
    "Urban_or_Rural": "Entorno",
    "Family_History": "Historial familiar",
    "Inflammatory_Bowel_Disease": "Enfermedad inflamatoria",
    "Obesity_BMI": "Obesidad / BMI",
    "Diabetes": "Diabetes",
    "Smoking_History": "Tabaquismo",
    "Alcohol_Consumption": "Alcohol",
    "Diet_Risk": "Riesgo dieta",
    "Physical_Activity": "Actividad fisica",
    "Screening_History": "Cribado previo",
}


def prepare_history_dataframe(patient_data: dict) -> pd.DataFrame:
    """Prepara los datos del paciente exactamente como espera el modelo CatBoost."""
    datos_limpios = {col: patient_data.get(col, "Unknown") for col in HISTORY_MODEL_COLUMNS}
    df_paciente = pd.DataFrame([datos_limpios])

    mapeos = {
        "Obesity_BMI": {"Normal": 0, "Overweight": 1, "Obese": 2},
        "Diet_Risk": {"Low": 0, "Moderate": 1, "High": 2},
        "Physical_Activity": {"Low": 0, "Moderate": 1, "High": 2},
        "Family_History": {"No": 0, "Yes": 1},
        "Inflammatory_Bowel_Disease": {"No": 0, "Yes": 1},
        "Smoking_History": {"No": 0, "Yes": 1},
        "Alcohol_Consumption": {"No": 0, "Yes": 1},
        "Diabetes": {"No": 0, "Yes": 1},
        "Screening_History": {"Never": 0, "Irregular": 1, "Regular": 2},
    }

    for col, mapping in mapeos.items():
        if col in df_paciente.columns:
            df_paciente[col] = df_paciente[col].map(mapping).fillna(0).astype(int)

    for col in HISTORY_CAT_FEATURES:
        df_paciente[col] = df_paciente[col].astype(str)

    return df_paciente


def explain_history_prediction(model: Any, patient_data: dict) -> dict[str, Any]:
    """Calcula explicabilidad SHAP de CatBoost para la Fase 1."""
    if model is None:
        return {
            "available": False,
            "method": "SHAP CatBoost",
            "message": "No hay modelo clinico cargado. La fase 1 esta en modo demo.",
            "features": [],
        }

    try:
        from catboost import Pool

        df_paciente = prepare_history_dataframe(patient_data)
        pool = Pool(df_paciente, cat_features=HISTORY_CAT_FEATURES)
        shap_values = np.asarray(model.get_feature_importance(pool, type="ShapValues"))
        if shap_values.ndim != 2 or shap_values.shape[1] < 2:
            raise ValueError("Formato SHAP inesperado")

        impacts = shap_values[0, :-1]
        expected_value = float(shap_values[0, -1])
        probability = float(model.predict_proba(df_paciente)[0][1])
        features: list[dict[str, Any]] = []

        for feature_name, impact in zip(HISTORY_MODEL_COLUMNS, impacts):
            impact_value = float(impact)
            features.append({
                "name": feature_name,
                "label": HISTORY_FEATURE_LABELS.get(feature_name, feature_name),
                "value": str(patient_data.get(feature_name, "Unknown")),
                "encoded_value": str(df_paciente.iloc[0][feature_name]),
                "impact": impact_value,
                "abs_impact": abs(impact_value),
                "direction": "sube el riesgo" if impact_value >= 0 else "baja el riesgo",
            })

        features.sort(key=lambda item: item["abs_impact"], reverse=True)
        return {
            "available": True,
            "method": "SHAP CatBoost",
            "message": "Impacto de cada variable sobre la prediccion de riesgo.",
            "expected_value": expected_value,
            "probability": probability,
            "features": features,
        }
    except Exception as e:
        print(f"  ⚠ No se pudo calcular SHAP de Fase 1: {e}")
        return {
            "available": False,
            "method": "SHAP CatBoost",
            "message": f"No se pudo calcular la explicacion SHAP: {e}",
            "features": [],
        }


def predict_cancer_risk(model: Any, patient_data: dict) -> tuple[bool, float]:
    """Prepara los datos del paciente y predice el riesgo con CatBoost."""
    if model is None:
        return True, 0.99

    try:
        # 1. Definir columnas exactas en el orden que se entrenó
        columnas = [
            'Age', 'Gender', 'Country', 'Urban_or_Rural', 'Family_History', 
            'Inflammatory_Bowel_Disease', 'Obesity_BMI', 'Diabetes', 
            'Smoking_History', 'Alcohol_Consumption', 'Diet_Risk', 
            'Physical_Activity', 'Screening_History'
        ]
        
        # 2. Filtrar solo las columnas clínicas
        datos_limpios = {col: patient_data.get(col, "Unknown") for col in columnas}
        df_paciente = pd.DataFrame([datos_limpios])
        
        # 3. Mapeo Ordinal ACTUALIZADO (Con Screening_History)
        mapeos = {
            'Obesity_BMI': {'Normal': 0, 'Overweight': 1, 'Obese': 2},
            'Diet_Risk': {'Low': 0, 'Moderate': 1, 'High': 2},
            'Physical_Activity': {'Low': 0, 'Moderate': 1, 'High': 2}, 
            'Family_History': {'No': 0, 'Yes': 1},
            'Inflammatory_Bowel_Disease': {'No': 0, 'Yes': 1},
            'Smoking_History': {'No': 0, 'Yes': 1},
            'Alcohol_Consumption': {'No': 0, 'Yes': 1},
            'Diabetes': {'No': 0, 'Yes': 1},
            'Screening_History': {'Never': 0, 'Irregular': 1, 'Regular': 2} # ¡CLAVE!
        }
        
        for col, mapping in mapeos.items():
            if col in df_paciente.columns:
                df_paciente[col] = df_paciente[col].map(mapping).fillna(0).astype(int)

        # 4. Asegurar que las categóricas son tipo texto para CatBoost
        cat_features = ['Gender', 'Country', 'Urban_or_Rural']
        for col in cat_features:
            df_paciente[col] = df_paciente[col].astype(str)

        # 5. Inferencia con CatBoost
        proba = model.predict_proba(df_paciente)[0]
        risk = float(proba[1]) # Probabilidad de clase 1 (Riesgo)
        
        is_positive = risk >= HISTORY_RISK_THRESHOLD
        return is_positive, risk
        
    except Exception as e:
        # Hacemos un print muy visual para que, si vuelve a fallar, lo veas en la consola
        print(f"\n❌ ERROR FATAL DE INFERENCIA EN CATBOOST: {e}\n")
        return True, 0.99

def save_phase1_result(patient_data: dict, is_positive: bool, probability: float) -> None:
    """
    Guarda los datos del paciente y el resultado de la predicción en la carpeta /resultados.
    """
    import os
    
    # 1. Asegurar que el directorio base existe
    results_dir = Path("resultados")
    results_dir.mkdir(exist_ok=True)
    
    # 2. Generar un nombre de archivo único
    # Intentamos usar el DNI o el Nombre si existen. Si no, usamos un timestamp.
    identifier = patient_data.get("DNI", patient_data.get("Nombre", ""))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if identifier:
        # Limpiamos el identificador para que sea válido como nombre de archivo
        safe_id = "".join(c for c in str(identifier) if c.isalnum() or c in (' ', '_')).replace(' ', '_')
        filename = f"paciente_{safe_id}_{timestamp}.json"
    else:
        filename = f"paciente_anon_{timestamp}.json"
        
    filepath = results_dir / filename
    
    # 3. Preparar el diccionario de salida
    output_data = {
        "informacion_paciente": patient_data,
        "analisis_ia": {
            "fecha_analisis": datetime.now().isoformat(),
            "riesgo_detectado": bool(is_positive),
            "probabilidad_exacta": float(probability),
            "modelo_usado": "CatBoost_Phase1"
        }
    }
    
    # 4. Guardar en JSON
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        print(f"  💾 Resultado guardado en: {filepath}")
    except Exception as e:
        print(f"  ⚠ Error al guardar el resultado: {e}")

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


def run_segmenter_inference(
    segmenter: dict[str, Any] | None,
    frame: np.ndarray,
    threshold: float | None = None,
) -> list[dict]:
    """Ejecuta el segmentador UNet3+ y devuelve regiones con mascara exacta."""
    if segmenter is None:
        return []
    try:
        import torch
        from torchvision.transforms import functional as TF

        model = segmenter["model"]
        device = segmenter["device"]
        image_size = int(segmenter.get("image_size", 352))
        threshold = float(threshold if threshold is not None else segmenter.get("threshold", COLONOSCOPY_SEGMENTER_THRESHOLD))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        tensor = TF.to_tensor(resized).unsqueeze(0).to(device)
        if device.type == "cuda":
            tensor = tensor.contiguous(memory_format=torch.channels_last)

        with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            output = model(pixel_values=tensor)
            logits = output["logits"] if isinstance(output, dict) else output
            prob = torch.sigmoid(logits)[0, 0].detach().float().cpu().numpy()

        h, w = frame.shape[:2]
        prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = (prob >= threshold).astype(np.uint8)
        min_area = max(12, int(h * w * COLONOSCOPY_SEGMENTER_MIN_AREA_RATIO))
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: list[dict] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            region_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(region_mask, [contour], -1, 1, thickness=-1)
            region_probs = prob[region_mask > 0]
            confidence = float(region_probs.mean()) if region_probs.size else float(prob.max())
            detections.append(
                {
                    "x1": int(x),
                    "y1": int(y),
                    "x2": int(x + bw),
                    "y2": int(y + bh),
                    "confidence": confidence,
                    "class_name": "unet-polyp",
                    "mask": region_mask.astype(bool),
                    "contour": contour,
                    "model_name": "UNet3+",
                }
            )
        detections.sort(key=lambda det: float(det["confidence"]), reverse=True)
        return detections
    except Exception as e:
        print(f"  ⚠ Error en segmentador UNet3+: {e}")
        return []


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

        mask = det.get("mask")
        contour = det.get("contour")
        if mask is not None:
            overlay = frame.copy()
            overlay[np.asarray(mask, dtype=bool)] = box_color
            cv2.addWeighted(overlay, 0.32, frame, 0.68, 0, dst=frame)
        if contour is not None:
            cv2.drawContours(frame, [contour], -1, box_color, 2)
        else:
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
    _focus_cv_window(window_name)
    return view_w, view_h


def _focus_cv_window(window_name: str) -> None:
    """Intenta traer al frente una ventana OpenCV sin romper compatibilidad."""
    try:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        cv2.waitKey(1)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 0)
    except cv2.error:
        pass


def _destroy_cv_window(window_name: str) -> None:
    """Cierra una ventana OpenCV y procesa eventos pendientes."""
    try:
        cv2.destroyWindow(window_name)
        cv2.waitKey(1)
    except cv2.error:
        pass


def _destroy_all_cv_windows() -> None:
    """Cierra ventanas OpenCV pendientes antes de volver a Tkinter."""
    try:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    except cv2.error:
        pass


def _resize_for_display(image: np.ndarray, max_width: int = 1000, max_height: int = 650) -> np.ndarray:
    """Redimensiona una imagen para verla centrada sin ocupar toda la pantalla."""
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return image
    scale = min(max_width / w, max_height / h, 1.0)
    new_w = max(int(w * scale), 1)
    new_h = max(int(h * scale), 1)
    if new_w == w and new_h == h:
        return image
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _bgr_to_photo(image: np.ndarray) -> ImageTk.PhotoImage:
    """Convierte imagen OpenCV BGR a PhotoImage de Tkinter."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(Image.fromarray(rgb))


def _show_image_tk_window(
    title: str,
    image: np.ndarray,
    actions: list[tuple[str, str, str]],
    max_width: int = 1000,
    max_height: int = 650,
    parent: tk.Tk | tk.Toplevel | None = None,
) -> str:
    """Muestra una imagen en Tkinter con botones reales y devuelve la acción."""
    result = {"action": actions[0][0] if actions else "close"}
    display = _resize_for_display(image, max_width=max_width, max_height=max_height)

    root = tk.Tk() if parent is None else tk.Toplevel(parent)
    root.title(title)
    root.configure(bg=BG_DARK)
    root.resizable(False, False)
    if parent is not None:
        root.transient(parent)

    photo = _bgr_to_photo(display)
    label = tk.Label(root, image=photo, bg=BG_DARK)
    label.image = photo
    label.pack(padx=12, pady=(12, 8))

    buttons = tk.Frame(root, bg=BG_DARK)
    buttons.pack(pady=(0, 12))

    def make_action(action: str):
        def _action():
            result["action"] = action
            root.destroy()
        return _action

    for action, text, color in actions:
        tk.Button(
            buttons,
            text=text,
            bg=color,
            fg=BG_DARK,
            activebackground=color,
            command=make_action(action),
            font=("Segoe UI", 13, "bold"),
            width=20,
            height=2,
            relief="flat",
            cursor="hand2",
        ).pack(side="left", padx=6)

    window_w = min(display.shape[1] + 24, max_width + 24)
    window_h = min(display.shape[0] + 112, max_height + 130)
    _center_window(root, window_w, window_h)
    if parent is not None:
        root.grab_set()
        parent.wait_window(root)
    else:
        root.mainloop()
    return result["action"]


def _run_cv_modal(parent: tk.Tk | tk.Toplevel, callback) -> None:
    """Oculta una ventana Tk mientras se muestra una ventana OpenCV modal."""
    try:
        parent.update_idletasks()
        parent_w = max(parent.winfo_width(), parent.winfo_reqwidth(), 400)
        parent_h = max(parent.winfo_height(), parent.winfo_reqheight(), 300)
    except tk.TclError:
        parent_w, parent_h = 560, 500
    try:
        parent.withdraw()
    except tk.TclError:
        pass
    try:
        callback()
    finally:
        try:
            parent.deiconify()
            _center_window(parent, parent_w, parent_h)
        except tk.TclError:
            pass


def _draw_action_buttons(
    frame: np.ndarray,
    actions: list[tuple[str, str, tuple[int, int, int]]],
    title: str = "Controles",
) -> tuple[np.ndarray, dict[str, tuple[int, int, int, int]]]:
    """Dibuja una botonera simple dentro de la ventana OpenCV."""
    h, w = frame.shape[:2]
    panel_w = 170
    button_h = 34
    gap = 8
    pad = 10
    title_h = 30
    panel_h = title_h + (len(actions) * button_h) + ((len(actions) - 1) * gap) + pad * 2

    x1 = max(w - panel_w - 8, 8)
    y1 = 8
    x2 = x1 + panel_w
    y2 = min(y1 + panel_h, h - 8)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 28), -1)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (120, 120, 140), 1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    cv2.putText(
        frame, title, (x1 + 10, y1 + 21),
        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 240, 245), 1, cv2.LINE_AA,
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
            frame, label, (btn_x1 + 10, btn_y1 + 23),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (18, 18, 25), 2, cv2.LINE_AA,
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


def _bbox_center_distance(box_a: dict[str, Any], box_b: dict[str, Any]) -> float:
    ax = (float(box_a["x1"]) + float(box_a["x2"])) / 2.0
    ay = (float(box_a["y1"]) + float(box_a["y2"])) / 2.0
    bx = (float(box_b["x1"]) + float(box_b["x2"])) / 2.0
    by = (float(box_b["y1"]) + float(box_b["y2"])) / 2.0
    return float(np.hypot(ax - bx, ay - by))


def _bbox_match_radius(box_a: dict[str, Any], box_b: dict[str, Any]) -> float:
    aw = max(float(box_a["x2"]) - float(box_a["x1"]), 1.0)
    ah = max(float(box_a["y2"]) - float(box_a["y1"]), 1.0)
    bw = max(float(box_b["x2"]) - float(box_b["x1"]), 1.0)
    bh = max(float(box_b["y2"]) - float(box_b["y1"]), 1.0)
    return max(140.0, 2.8 * max(aw, ah, bw, bh))


def _update_polyp_tracks(
    detections: list[dict],
    active_tracks: list[dict[str, Any]],
    next_track_id: int,
    iou_threshold: float = 0.3,
    max_missing_frames: int = 10,
    min_confirm_frames: int = 30,
) -> tuple[list[dict[str, Any]], int, int]:
    """Asocia detecciones a tracks y devuelve cuántos pólipos únicos nuevos confirmar.

    La asociacion usa IoU primero y cercania del centro como fallback. Esto evita
    contar como nuevo polipo una region que desaparece unos segundos por movimiento
    de camara y vuelve cerca de la misma zona.
    """
    for track in active_tracks:
        track["matched"] = False

    unique_new_polyps = 0

    for det in detections:
        best_track = None
        best_score = -1.0
        for track in active_tracks:
            if track["matched"]:
                continue
            iou = _bbox_iou(det, track["bbox"])
            distance = _bbox_center_distance(det, track["bbox"])
            radius = _bbox_match_radius(det, track["bbox"])
            center_match = distance <= radius
            center_score = max(0.0, 1.0 - (distance / max(radius, 1.0)))
            score = max(iou, center_score * 0.95)
            if center_match and track["missing"] > 0:
                score += 0.20
            if score > best_score:
                best_score = score
                best_track = track

        if best_track is not None and best_score >= 0.12:
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
            fallback_track = None
            fallback_distance = float("inf")
            for track in active_tracks:
                if track["matched"] or not track.get("counted", False):
                    continue
                if int(track.get("missing", 0)) > max_missing_frames:
                    continue
                distance = _bbox_center_distance(det, track["bbox"])
                radius = _bbox_match_radius(det, track["bbox"]) * 2.2
                if distance <= radius and distance < fallback_distance:
                    fallback_track = track
                    fallback_distance = distance

            if fallback_track is not None:
                fallback_track["bbox"] = {
                    "x1": det["x1"],
                    "y1": det["y1"],
                    "x2": det["x2"],
                    "y2": det["y2"],
                }
                fallback_track["confidence"] = det["confidence"]
                fallback_track["missing"] = 0
                fallback_track["hits"] += 1
                fallback_track["matched"] = True
                det["track_id"] = fallback_track["track_id"]
                continue

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


def show_explanation_window(
    explanation: dict[str, Any] | None,
    parent: tk.Tk | tk.Toplevel | None = None,
) -> None:
    """Muestra la explicación visual en una ventana centrada."""
    if explanation is None:
        messagebox.showinfo(
            "Explicacion visual",
            "No hay un mapa explicativo disponible para este resultado.",
            parent=parent,
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

    _show_image_tk_window(
        explanation.get("title", "Explicacion visual"),
        canvas,
        [("close", "Cerrar", BTN_INACTIVE)],
        max_width=1100,
        max_height=720,
        parent=parent,
    )


def save_screenshot(frame: np.ndarray, prefix: str = "screenshot") -> str:
    """Guarda screenshot del frame actual."""
    screenshots = Path(SCREENSHOTS_DIR)
    screenshots.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = screenshots / f"{prefix}_{timestamp}.jpg"
    cv2.imwrite(str(path), frame)
    return str(path)


def _json_safe(value: Any) -> Any:
    """Convierte estructuras con arrays/paths a JSON simple."""
    if isinstance(value, np.ndarray):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        safe: dict[str, Any] = {}
        for k, v in value.items():
            converted = _json_safe(v)
            if converted is not None:
                safe[k] = converted
        return safe
    if isinstance(value, list):
        return [_json_safe(v) for v in value if _json_safe(v) is not None]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_history_image(record_dir: Path, name: str, image: np.ndarray | None) -> str | None:
    if image is None:
        return None
    path = record_dir / name
    cv2.imwrite(str(path), image)
    return str(path)


def save_analysis_history(kind: str, title: str, result_data: dict[str, Any]) -> str:
    """Guarda metadatos e imágenes de un análisis para consulta posterior."""
    history_dir = Path(ANALYSIS_HISTORY_DIR)
    history_dir.mkdir(exist_ok=True)
    record_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    record_dir = history_dir / f"{record_id}_{kind}"
    record_dir.mkdir(exist_ok=True)

    saved_images: dict[str, str] = {}
    if kind == "phase3":
        image_results = result_data.get("images") or [result_data]
        for idx, image_result in enumerate(image_results, start=1):
            prefix = f"foto_{idx:02d}"
            image_path = image_result.get("image_path")
            if image_path and Path(image_path).exists():
                dst = record_dir / f"{prefix}_original{Path(image_path).suffix.lower() or '.jpg'}"
                shutil.copy2(image_path, dst)
                saved_images[f"{prefix}_original"] = str(dst)
                if idx == 1:
                    saved_images["original"] = str(dst)
            preview = _write_history_image(
                record_dir, f"{prefix}_resultado.jpg", image_result.get("preview")
            )
            if preview:
                saved_images[f"{prefix}_resultado"] = preview
                if idx == 1:
                    saved_images["resultado"] = preview
            explanation = image_result.get("explanation") or {}
            overlay = _write_history_image(
                record_dir, f"{prefix}_enfoque.jpg", explanation.get("overlay")
            )
            if overlay:
                saved_images[f"{prefix}_enfoque"] = overlay
                if idx == 1:
                    saved_images["enfoque"] = overlay
    elif kind == "phase2":
        explanation = result_data.get("explanation") or {}
        original = _write_history_image(record_dir, "frame_original.jpg", explanation.get("original"))
        overlay = _write_history_image(record_dir, "enfoque.jpg", explanation.get("overlay"))
        if original:
            saved_images["original"] = original
        if overlay:
            saved_images["enfoque"] = overlay

    metadata = {
        "id": record_id,
        "kind": kind,
        "title": title,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "result": _json_safe(result_data),
        "images": saved_images,
    }
    metadata_path = record_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"  Historial guardado: {metadata_path}")
    return str(metadata_path)


def _safe_slug(text: str) -> str:
    allowed = [c if c.isalnum() or c in ("-", "_") else "_" for c in text.strip()]
    slug = "".join(allowed).strip("_")
    return slug or "paciente"


def save_patient_consultation_history(
    patient_name: str,
    phase1_result: dict[str, Any],
    phase2_result: dict[str, Any],
    phase3_result: dict[str, Any],
) -> str:
    """Guarda una consulta completa agrupada por paciente."""
    history_dir = Path(ANALYSIS_HISTORY_DIR)
    history_dir.mkdir(exist_ok=True)
    record_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    record_dir = history_dir / f"{record_id}_{_safe_slug(patient_name)}"
    record_dir.mkdir(exist_ok=True)

    saved_images: dict[str, str] = {}
    phase2_explanation = phase2_result.get("explanation") or {}
    p2_original = _write_history_image(
        record_dir, "colonoscopia_frame_original.jpg", phase2_explanation.get("original")
    )
    p2_focus = _write_history_image(
        record_dir, "colonoscopia_enfoque.jpg", phase2_explanation.get("overlay")
    )
    if p2_original:
        saved_images["colonoscopia_original"] = p2_original
        saved_images["colonoscopia_resultado"] = p2_original
    if p2_focus:
        saved_images["colonoscopia_enfoque"] = p2_focus

    image_results = phase3_result.get("images") or []
    for idx, image_result in enumerate(image_results, start=1):
        prefix = f"histologia_{idx:02d}"
        image_path = image_result.get("image_path")
        if image_path and Path(image_path).exists():
            dst = record_dir / f"{prefix}_original{Path(image_path).suffix.lower() or '.jpg'}"
            shutil.copy2(image_path, dst)
            saved_images[f"{prefix}_original"] = str(dst)
            if idx == 1:
                saved_images["histologia_original"] = str(dst)
        preview = _write_history_image(
            record_dir, f"{prefix}_resultado.jpg", image_result.get("preview")
        )
        if preview:
            saved_images[f"{prefix}_resultado"] = preview
            if idx == 1:
                saved_images["histologia_resultado"] = preview
        explanation = image_result.get("explanation") or {}
        focus = _write_history_image(
            record_dir, f"{prefix}_enfoque.jpg", explanation.get("overlay")
        )
        if focus:
            saved_images[f"{prefix}_enfoque"] = focus
            if idx == 1:
                saved_images["histologia_enfoque"] = focus

    metadata = {
        "id": record_id,
        "kind": "consultation",
        "title": f"Consulta completa - {patient_name}",
        "patient_name": patient_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "result": {
            "phase1": _json_safe(phase1_result),
            "phase2": _json_safe(phase2_result),
            "phase3": _json_safe(phase3_result),
        },
        "images": saved_images,
    }
    metadata_path = record_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"  Historial de paciente guardado: {metadata_path}")
    return str(metadata_path)


def _load_history_records() -> list[dict[str, Any]]:
    records = []
    for path in sorted(Path(ANALYSIS_HISTORY_DIR).glob("*/metadata.json"), reverse=True):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    return records


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
    min_confirm_frames = max(4, min(6, int(round(POLYP_CONFIRM_SECONDS * effective_fps))))
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
    _destroy_cv_window(window_name)

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
    _destroy_all_cv_windows()

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

    _show_image_tk_window(
        f"{phase_label} - Imagen",
        display,
        [
            ("continue", "Continuar", ACCENT_BLUE),
            ("close", "Cerrar vista", BTN_INACTIVE),
        ],
        max_width=1000,
        max_height=650,
    )

    print(f"  Resultado: {status_text} ({confidence:.2%})")
    explanation = create_gradcam_explanation(model_bundle, frame, cls_name)
    return {
        "image_path": image_path,
        "image_name": Path(image_path).name,
        "is_malignant": is_cancer and confidence >= CONFIDENCE_THRESHOLD,
        "confidence": confidence,
        "class_name": cls_name,
        "demo_mode": model_bundle is None,
        "preview": display,
        "explanation": explanation,
    }


def process_image_batch_classification(
    image_paths: list[str],
    model_bundle: dict | None,
    phase_label: str,
) -> dict[str, Any]:
    """Procesa varias imágenes histológicas y devuelve un resumen agregado."""
    image_results: list[dict[str, Any]] = []
    for idx, image_path in enumerate(image_paths, start=1):
        print(f"  Imagen {idx}/{len(image_paths)}")
        image_results.append(
            process_image_classification(
                image_path,
                model_bundle,
                phase_label=f"{phase_label} ({idx}/{len(image_paths)})",
            )
        )

    valid_results = [r for r in image_results if r.get("class_name") != "ERROR"]
    malignant_results = [r for r in valid_results if r.get("is_malignant")]
    best_result = max(
        valid_results,
        key=lambda r: float(r.get("confidence", 0.0)),
        default=image_results[0] if image_results else {},
    )
    best_malignant = max(
        malignant_results,
        key=lambda r: float(r.get("confidence", 0.0)),
        default=None,
    )
    representative = best_malignant or best_result

    return {
        "image_path": representative.get("image_path", ""),
        "image_name": (
            representative.get("image_name", "")
            if len(image_results) == 1
            else f"{len(image_results)} imágenes analizadas"
        ),
        "is_malignant": len(malignant_results) > 0,
        "confidence": float(representative.get("confidence", 0.0)),
        "class_name": representative.get("class_name", "N/A"),
        "demo_mode": model_bundle is None,
        "preview": representative.get("preview"),
        "explanation": representative.get("explanation"),
        "images": image_results,
        "total_images": len(image_results),
        "valid_images": len(valid_results),
        "malignant_count": len(malignant_results),
        "non_malignant_count": len(valid_results) - len(malignant_results),
    }


# ══════════════════════════════════════════════
# GUI – VENTANAS POR FASE
# ══════════════════════════════════════════════

def _center_window(win: tk.Tk | tk.Toplevel, w: int, h: int) -> None:
    """Centra una ventana en la pantalla."""
    win.update_idletasks()
    screen_w = win.winfo_screenwidth()
    screen_h = win.winfo_screenheight()
    req_w = max(w, win.winfo_reqwidth())
    req_h = max(h, win.winfo_reqheight())
    final_w = min(req_w, max(screen_w - 80, 320))
    final_h = min(req_h, max(screen_h - 100, 240))
    x = max((screen_w - final_w) // 2, 0)
    y = max((screen_h - final_h) // 2, 0)
    win.geometry(f"{final_w}x{final_h}+{x}+{y}")
    win.update_idletasks()
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
        parent, text="← Volver", bg=BTN_INACTIVE, fg=FG_TEXT,
        activebackground="#6c7086", command=command,
        font=("Segoe UI", 10), width=32, height=1, relief="flat", cursor="hand2",
    )


def ask_patient_name() -> str | None:
    """Pide el nombre del paciente al iniciar una consulta completa."""
    result = {"name": None}
    root = tk.Tk()
    root.title("Nueva consulta")
    root.configure(bg=BG_DARK)
    root.resizable(False, False)

    tk.Label(
        root,
        text="Nombre del paciente",
        font=("Segoe UI", 18, "bold"),
        fg=FG_TEXT,
        bg=BG_DARK,
    ).pack(pady=(24, 8))
    tk.Label(
        root,
        text="Se usará para guardar el historial completo de esta consulta.",
        font=("Segoe UI", 10),
        fg=FG_SUB,
        bg=BG_DARK,
    ).pack(pady=(0, 14))

    name_var = tk.StringVar()
    entry = tk.Entry(root, textvariable=name_var, font=("Segoe UI", 12), width=34)
    entry.pack(pady=(0, 16))

    def on_start():
        name = name_var.get().strip()
        if not name:
            messagebox.showwarning(
                "Paciente requerido",
                "Introduce un nombre de paciente para iniciar la consulta.",
                parent=root,
            )
            return
        result["name"] = name
        root.destroy()

    def on_cancel():
        root.destroy()

    buttons = tk.Frame(root, bg=BG_DARK)
    buttons.pack(pady=(0, 20))
    tk.Button(
        buttons,
        text="Iniciar consulta",
        bg=ACCENT_BLUE,
        fg=BG_DARK,
        activebackground=ACCENT_BLUE,
        command=on_start,
        font=("Segoe UI", 11, "bold"),
        width=18,
        height=2,
        relief="flat",
        cursor="hand2",
    ).pack(side="left", padx=6)
    tk.Button(
        buttons,
        text="Cancelar",
        bg=BTN_INACTIVE,
        fg=FG_TEXT,
        activebackground="#6c7086",
        command=on_cancel,
        font=("Segoe UI", 11, "bold"),
        width=14,
        height=2,
        relief="flat",
        cursor="hand2",
    ).pack(side="left", padx=6)

    root.bind("<Return>", lambda _event: on_start())
    _center_window(root, 460, 250)
    entry.focus_set()
    root.mainloop()
    return result["name"]


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
    Devuelve 'start', 'phase1', 'phase2', 'phase3', 'history' o 'exit'.
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

    def on_history():
        result["action"] = "history"
        root.destroy()

    _make_button(
        root,
        "📚  Historial de pacientes",
        BTN_INACTIVE,
        on_history,
        height=1,
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

    _center_window(root, 680, 570)
    root.mainloop()
    return result["action"]


def show_history_window() -> None:
    """Muestra registros guardados y permite ver imágenes/enfoques."""
    records = _load_history_records()

    root = tk.Tk()
    root.title("Historial de pacientes")
    root.configure(bg=BG_DARK)
    root.resizable(False, False)

    tk.Label(
        root,
        text="📚  Historial de pacientes",
        font=("Segoe UI", 18, "bold"),
        fg=FG_TEXT,
        bg=BG_DARK,
    ).pack(pady=(22, 8))

    if not records:
        tk.Label(
            root,
            text="Todavía no hay consultas de pacientes guardadas.",
            font=("Segoe UI", 11),
            fg=FG_SUB,
            bg=BG_DARK,
        ).pack(pady=(20, 30))
        _back_button(root, root.destroy).pack(pady=(0, 20))
        _center_window(root, 520, 240)
        root.mainloop()
        return

    list_frame = tk.Frame(root, bg=BG_CARD, padx=12, pady=12)
    list_frame.pack(padx=24, pady=(0, 12), fill="both")

    listbox = tk.Listbox(
        list_frame,
        width=72,
        height=10,
        font=("Segoe UI", 10),
        bg="#11111b",
        fg=FG_TEXT,
        selectbackground=ACCENT_BLUE,
        selectforeground=BG_DARK,
        relief="flat",
    )
    listbox.pack(side="left", fill="both")
    scrollbar = tk.Scrollbar(list_frame, command=listbox.yview)
    scrollbar.pack(side="right", fill="y")
    listbox.config(yscrollcommand=scrollbar.set)

    for record in records:
        result = record.get("result", {})
        if record.get("kind") == "consultation":
            phase2 = result.get("phase2", {})
            phase3 = result.get("phase3", {})
            summary = (
                f"{record['created_at']} | {record.get('patient_name', 'Paciente')} | "
                f"{phase2.get('unique_polyps', 0)} pólipos | "
                f"{phase3.get('malignant_count', 0)}/{phase3.get('total_images', 0)} histología maligna"
            )
        elif record.get("kind") == "phase3":
            summary = (
                f"{record['created_at']} | Foto histológica | "
                f"{result.get('malignant_count', 1 if result.get('is_malignant') else 0)}/"
                f"{result.get('total_images', 1)} malignas | "
                f"{result.get('class_name', 'N/A')} | "
                f"{float(result.get('confidence', 0.0)):.1%}"
            )
        else:
            summary = (
                f"{record['created_at']} | Colonoscopia | "
                f"{result.get('unique_polyps', 0)} pólipos confirmados"
            )
        listbox.insert("end", summary)

    detail = tk.Text(
        root,
        width=72,
        height=7,
        bg=BG_CARD,
        fg=FG_TEXT,
        font=("Consolas", 9),
        relief="flat",
        wrap="word",
    )
    detail.pack(padx=24, pady=(0, 12))

    def selected_record() -> dict[str, Any] | None:
        selection = listbox.curselection()
        if not selection:
            return None
        return records[selection[0]]

    def refresh_detail(_event=None):
        record = selected_record()
        detail.delete("1.0", "end")
        if record is None:
            return
        result = record.get("result", {})
        detail.insert("end", f"Tipo: {record.get('title')}\n")
        detail.insert("end", f"Fecha: {record.get('created_at')}\n")
        if record.get("kind") == "consultation":
            phase1 = result.get("phase1", {})
            phase2 = result.get("phase2", {})
            phase3 = result.get("phase3", {})
            detail.insert("end", f"Paciente: {record.get('patient_name', '')}\n")
            detail.insert("end", f"Fase 1: {phase1.get('status', 'N/A')} ({float(phase1.get('probability', 0.0)):.1%})\n")
            detail.insert("end", f"Fase 2: {phase2.get('unique_polyps', 0)} pólipos confirmados\n")
            detail.insert("end", f"Fase 3: {phase3.get('malignant_count', 0)}/{phase3.get('total_images', 0)} imágenes malignas\n")
            detail.insert("end", f"Clase destacada: {phase3.get('class_name', 'N/A')}\n")
            detail.insert("end", f"Confianza destacada: {float(phase3.get('confidence', 0.0)):.1%}\n")
        elif record.get("kind") == "phase3":
            detail.insert("end", f"Imágenes: {result.get('total_images', 1)}\n")
            detail.insert("end", f"Imagen destacada: {result.get('image_name', '')}\n")
            if int(result.get("total_images", 1)) > 1:
                detail.insert("end", f"Malignas: {result.get('malignant_count', 0)}\n")
                detail.insert("end", f"No malignas: {result.get('non_malignant_count', 0)}\n")
            detail.insert("end", f"Clase: {result.get('class_name', '')}\n")
            detail.insert("end", f"Confianza: {float(result.get('confidence', 0.0)):.1%}\n")
            detail.insert("end", f"Resultado: {'Maligna' if result.get('is_malignant') else 'No maligna'}\n")
        else:
            detail.insert("end", f"Pólipos confirmados: {result.get('unique_polyps', 0)}\n")
            detail.insert("end", f"Frames candidatos: {result.get('positive_frames', 0)}\n")
            detail.insert("end", f"Confianza máxima: {float(result.get('max_confidence', 0.0)):.1%}\n")

    def show_history_image(image_key: str):
        record = selected_record()
        if record is None:
            return
        image_path = record.get("images", {}).get(image_key)
        if not image_path or not Path(image_path).exists():
            messagebox.showinfo("Historial", "No hay imagen disponible para este registro.")
            return
        image = cv2.imread(image_path)
        if image is None:
            messagebox.showerror("Historial", "No se pudo abrir la imagen guardada.")
            return
        _show_image_tk_window(
            f"Historial - {image_key}",
            image,
            [("close", "Cerrar", BTN_INACTIVE)],
            max_width=1000,
            max_height=650,
            parent=root,
        )

    def show_main_history_image():
        record = selected_record()
        if record is None:
            return
        images = record.get("images", {})
        for key in ("histologia_resultado", "colonoscopia_original", "resultado", "original"):
            if images.get(key):
                show_history_image(key)
                return
        show_history_image("original")

    def show_focus_history_image():
        record = selected_record()
        if record is None:
            return
        images = record.get("images", {})
        for key in ("histologia_enfoque", "colonoscopia_enfoque", "enfoque"):
            if images.get(key):
                show_history_image(key)
                return
        show_history_image("enfoque")

    def show_phase1_history():
        record = selected_record()
        if record is None:
            return
        if record.get("kind") != "consultation":
            messagebox.showinfo("Historial", "Este registro no contiene una consulta completa.", parent=root)
            return
        phase1 = record.get("result", {}).get("phase1", {})
        win = tk.Toplevel(root)
        win.title("Historial - Fase 1")
        win.configure(bg=BG_DARK)
        win.resizable(False, False)
        win.transient(root)
        tk.Label(
            win,
            text=f"Fase 1 - {record.get('patient_name', 'Paciente')}",
            font=("Segoe UI", 16, "bold"),
            fg=ACCENT_YELLOW,
            bg=BG_DARK,
        ).pack(pady=(20, 12))
        card = tk.Frame(win, bg=BG_CARD, padx=18, pady=14)
        card.pack(padx=24, pady=(0, 16), fill="x")
        _metric_row(card, "Estado", phase1.get("status", "N/A"))
        _metric_row(card, "Probabilidad", f"{float(phase1.get('probability', 0.0)):.1%}")
        _metric_row(card, "Modo demo", "Sí" if phase1.get("demo_mode") else "No")
        _metric_row(card, "Campos totales", str(phase1.get("total_fields", 0)))
        _make_button(
            win,
            "Ver en que se ha fijado el modelo",
            ACCENT_BLUE,
            lambda: show_phase1_explanation_window(phase1.get("explanation"), parent=win),
            width=30,
        ).pack(pady=(0, 10))
        _back_button(win, win.destroy).pack(pady=(0, 18))
        _center_window(win, 520, 380)
        win.grab_set()
        root.wait_window(win)

    def show_phase2_history():
        record = selected_record()
        if record is None:
            return
        images = record.get("images", {})
        if images.get("colonoscopia_enfoque"):
            show_history_image("colonoscopia_enfoque")
        elif images.get("colonoscopia_original"):
            show_history_image("colonoscopia_original")
        else:
            messagebox.showinfo("Historial", "No hay imagen guardada para Fase 2.", parent=root)

    def show_phase3_history():
        record = selected_record()
        if record is None:
            return
        images = record.get("images", {})
        phase3 = record.get("result", {}).get("phase3", {})
        image_results = phase3.get("images", [])
        if not image_results:
            messagebox.showinfo("Historial", "No hay imágenes guardadas para Fase 3.", parent=root)
            return

        win = tk.Toplevel(root)
        win.title("Historial - Fase 3")
        win.configure(bg=BG_DARK)
        win.resizable(False, False)
        win.transient(root)
        tk.Label(
            win,
            text="Fase 3 - Fotos histológicas",
            font=("Segoe UI", 16, "bold"),
            fg=ACCENT_MAUVE,
            bg=BG_DARK,
        ).pack(pady=(20, 10))

        listbox_photos = tk.Listbox(
            win,
            width=64,
            height=min(max(len(image_results), 4), 10),
            font=("Segoe UI", 10),
            bg="#11111b",
            fg=FG_TEXT,
            selectbackground=ACCENT_BLUE,
            selectforeground=BG_DARK,
            relief="flat",
        )
        listbox_photos.pack(padx=20, pady=(0, 12))
        for idx, image_result in enumerate(image_results, start=1):
            label = "Maligna" if image_result.get("is_malignant") else "No maligna"
            listbox_photos.insert(
                "end",
                f"{idx:02d}. {image_result.get('image_name', '')} | {label} | {float(image_result.get('confidence', 0.0)):.1%}",
            )
        listbox_photos.selection_set(0)

        def selected_photo_index() -> int | None:
            selection = listbox_photos.curselection()
            if not selection:
                return None
            return int(selection[0]) + 1

        def show_photo(kind: str):
            idx = selected_photo_index()
            if idx is None:
                return
            key = f"histologia_{idx:02d}_{kind}"
            show_history_image(key)

        buttons_photos = tk.Frame(win, bg=BG_DARK)
        buttons_photos.pack(pady=(0, 18))
        tk.Button(
            buttons_photos,
            text="Ver resultado",
            bg=ACCENT_BLUE,
            fg=BG_DARK,
            command=lambda: show_photo("resultado"),
            font=("Segoe UI", 10, "bold"),
            width=16,
            height=2,
            relief="flat",
            cursor="hand2",
        ).pack(side="left", padx=6)
        tk.Button(
            buttons_photos,
            text="Ver enfoque",
            bg=ACCENT_MAUVE,
            fg=BG_DARK,
            command=lambda: show_photo("enfoque"),
            font=("Segoe UI", 10, "bold"),
            width=16,
            height=2,
            relief="flat",
            cursor="hand2",
        ).pack(side="left", padx=6)
        _back_button(buttons_photos, win.destroy).pack(side="left", padx=6)
        _center_window(win, 620, 390)
        win.grab_set()
        root.wait_window(win)

    listbox.bind("<<ListboxSelect>>", refresh_detail)
    listbox.selection_set(0)
    refresh_detail()

    buttons = tk.Frame(root, bg=BG_DARK)
    buttons.pack(pady=(0, 18))
    tk.Button(
        buttons,
        text="Ver Fase 1",
        bg=ACCENT_YELLOW,
        fg=BG_DARK,
        command=show_phase1_history,
        font=("Segoe UI", 10, "bold"),
        width=13,
        relief="flat",
        cursor="hand2",
    ).pack(side="left", padx=6)
    tk.Button(
        buttons,
        text="Ver Fase 2",
        bg=ACCENT_GREEN,
        fg=BG_DARK,
        command=show_phase2_history,
        font=("Segoe UI", 10, "bold"),
        width=13,
        relief="flat",
        cursor="hand2",
    ).pack(side="left", padx=6)
    tk.Button(
        buttons,
        text="Ver Fase 3",
        bg=ACCENT_MAUVE,
        fg=BG_DARK,
        command=show_phase3_history,
        font=("Segoe UI", 10, "bold"),
        width=13,
        relief="flat",
        cursor="hand2",
    ).pack(side="left", padx=6)
    _back_button(buttons, root.destroy).pack(side="left", padx=6)

    _center_window(root, 680, 560)
    root.mainloop()


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

def show_phase1_explanation_window(
    explanation: dict[str, Any] | None,
    parent: tk.Tk | tk.Toplevel | None = None,
) -> None:
    """Muestra las variables SHAP que mas han influido en la Fase 1."""
    win = tk.Toplevel(parent) if parent is not None else tk.Tk()
    win.title("Explicacion visual - Fase 1")
    win.configure(bg=BG_DARK)
    win.resizable(False, False)
    if parent is not None:
        win.transient(parent)

    tk.Label(
        win,
        text="Fase 1 - En que se ha fijado el modelo",
        font=("Segoe UI", 16, "bold"),
        fg=ACCENT_YELLOW,
        bg=BG_DARK,
    ).pack(pady=(20, 8))

    available = bool(explanation and explanation.get("available"))
    message = str((explanation or {}).get("message", "No hay explicacion disponible."))
    probability = (explanation or {}).get("probability")

    summary = tk.Frame(win, bg=BG_CARD, padx=16, pady=12)
    summary.pack(padx=24, pady=(0, 12), fill="x")
    _metric_row(summary, "Metodo", str((explanation or {}).get("method", "SHAP CatBoost")))
    if probability is not None:
        _metric_row(summary, "Probabilidad", f"{float(probability):.1%}")
    tk.Label(
        summary,
        text=message,
        font=("Segoe UI", 9),
        fg=FG_SUB if available else ACCENT_YELLOW,
        bg=BG_CARD,
        wraplength=640,
        justify="left",
    ).pack(anchor="w", pady=(8, 0))

    if available:
        header = tk.Frame(win, bg=BG_DARK)
        header.pack(padx=24, fill="x")
        tk.Label(header, text="Variable", width=24, anchor="w", font=("Segoe UI", 9, "bold"), fg=FG_SUB, bg=BG_DARK).pack(side="left")
        tk.Label(header, text="Valor", width=18, anchor="w", font=("Segoe UI", 9, "bold"), fg=FG_SUB, bg=BG_DARK).pack(side="left")
        tk.Label(header, text="Impacto", width=12, anchor="e", font=("Segoe UI", 9, "bold"), fg=FG_SUB, bg=BG_DARK).pack(side="left")
        tk.Label(header, text="Direccion", width=16, anchor="w", font=("Segoe UI", 9, "bold"), fg=FG_SUB, bg=BG_DARK).pack(side="left", padx=(12, 0))

        rows = tk.Frame(win, bg=BG_CARD, padx=12, pady=10)
        rows.pack(padx=24, pady=(4, 12), fill="x")
        for feature in (explanation or {}).get("features", [])[:10]:
            impact = float(feature.get("impact", 0.0))
            color = ACCENT_RED if impact >= 0 else ACCENT_GREEN
            row = tk.Frame(rows, bg=BG_CARD)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=str(feature.get("label", "")), width=24, anchor="w", font=("Segoe UI", 9, "bold"), fg=FG_TEXT, bg=BG_CARD).pack(side="left")
            tk.Label(row, text=str(feature.get("value", ""))[:22], width=18, anchor="w", font=("Segoe UI", 9), fg=FG_SUB, bg=BG_CARD).pack(side="left")
            tk.Label(row, text=f"{impact:+.3f}", width=12, anchor="e", font=("Segoe UI", 9, "bold"), fg=color, bg=BG_CARD).pack(side="left")
            tk.Label(row, text=str(feature.get("direction", "")), width=16, anchor="w", font=("Segoe UI", 9), fg=color, bg=BG_CARD).pack(side="left", padx=(12, 0))

    _back_button(win, win.destroy).pack(pady=(0, 18))
    _center_window(win, 720, 560 if available else 260)
    if parent is not None:
        win.grab_set()
        parent.wait_window(win)
    else:
        win.mainloop()


def show_phase1_result(
    patient_data: dict,
    is_positive: bool,
    probability: float,
    demo_mode: bool = False,
    numeric_fields: int = 0, # Mantenemos el parámetro para no romper llamadas anteriores
    explanation: dict[str, Any] | None = None,
) -> str:
    """
    Muestra el resultado de la Fase 1 priorizando variables clínicas de alto impacto.
    Devuelve 'next' o 'restart'.
    """
    result = {"action": "restart"}

    root = tk.Tk()
    root.title("Fase 1 — Resultado")
    root.configure(bg=BG_DARK)
    root.resizable(False, False)

    tk.Label(
        root, text="📋  Resultado del análisis clínico",
        font=("Segoe UI", 16, "bold"), fg=ACCENT_YELLOW, bg=BG_DARK,
    ).pack(pady=(20, 10))

    # --- NUEVA LÓGICA: Priorización de variables para el médico ---
    data_frame = tk.Frame(root, bg=BG_CARD, padx=15, pady=10)
    data_frame.pack(padx=25, pady=5, fill="x")

    # Lista de campos clave que el médico NECESITA ver de un vistazo
    campos_clave = [
        'Nombre', 'DNI', 'Age', 'Gender', 'Family_History', 
        'Inflammatory_Bowel_Disease', 'Obesity_BMI', 'Diabetes'
    ]
    
    # Nombres amigables para la interfaz (opcional pero recomendado)
    nombres_amigables = {
        'Age': 'Edad',
        'Gender': 'Género',
        'Family_History': 'Historial Familiar',
        'Inflammatory_Bowel_Disease': 'Enf. Inflamatoria (IBD)',
        'Obesity_BMI': 'Nivel de Obesidad'
    }

    campos_a_mostrar = {}
    
    # 1. Extraer primero los campos prioritarios (máximo 8)
    for campo in campos_clave:
        if campo in patient_data:
            etiqueta = nombres_amigables.get(campo, campo)
            campos_a_mostrar[etiqueta] = patient_data[campo]
            if len(campos_a_mostrar) >= 8:
                break

    # 2. Si hay menos de 8, rellenamos con el resto de datos disponibles
    if len(campos_a_mostrar) < 8:
        for k, v in patient_data.items():
            etiqueta = nombres_amigables.get(k, k)
            if etiqueta not in campos_a_mostrar:
                campos_a_mostrar[etiqueta] = v
            if len(campos_a_mostrar) >= 8:
                break

    # Dibujar la tabla de datos del paciente
    for key, value in campos_a_mostrar.items():
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

    # --- MÉTRICAS ACTUALIZADAS ---
    analysis_frame = tk.Frame(root, bg=BG_CARD, padx=15, pady=12)
    analysis_frame.pack(padx=25, pady=(8, 5), fill="x")
    
    _metric_row(analysis_frame, "Datos procesados", str(len(patient_data)))
    # Quitamos lo de 'Campos numéricos' porque CatBoost usa texto y números nativamente
    _metric_row(analysis_frame, "Variables clínicas evaluadas", "13 (CatBoost)")
    _metric_row(
        analysis_frame,
        "Probabilidad estimada",
        f"{probability:.1%}",
        ACCENT_RED if is_positive else ACCENT_GREEN,
    )
    _metric_row(
        analysis_frame,
        "Modo de ejecución",
        "Demo (sin modelo clínico)" if demo_mode else "Motor de Riesgo Activo",
        ACCENT_YELLOW if demo_mode else ACCENT_GREEN,
    )

    # Resultado positivo/negativo
    if is_positive:
        color = ACCENT_RED
        text = f"⚠️  RIESGO DETECTADO  ({probability:.0%})"
        subtext = "Se requiere confirmación diagnóstica (Fase 2)"
    else:
        color = ACCENT_GREEN
        text = f"✅  SIN RIESGO  ({probability:.0%})"
        subtext = "Probabilidad basal normal. No requiere intervención."

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

    _make_button(
        root,
        "Ver en que se ha fijado el modelo",
        ACCENT_BLUE,
        lambda: show_phase1_explanation_window(explanation, parent=root),
    ).pack(pady=(0, 8))

    if is_positive:
        _make_button(
            root, "▶  Siguiente fase: Colonoscopia", ACCENT_GREEN, on_next,
        ).pack(pady=(0, 8))

    _back_button(root, on_restart).pack(pady=(0, 15))

    _center_window(root, 540, 640) # Aumentado ligeramente para evitar recortes
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
) -> tuple[str, list[str]]:
    """
    Menú para seleccionar una imagen estática.
    Devuelve ('image', path) o ('exit', None).
    """
    result: dict = {"action": "exit", "paths": []}

    root = tk.Tk()
    root.title(f"Fase {phase_num} — {phase_title}")
    root.configure(bg=BG_DARK)
    root.resizable(False, False)

    tk.Label(
        root, text=f"🔬  Fase {phase_num}: {phase_title}",
        font=("Segoe UI", 18, "bold"), fg=color, bg=BG_DARK,
    ).pack(pady=(25, 5))

    tk.Label(
        root, text="Selecciona una o varias imágenes para analizar",
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
        filepaths = filedialog.askopenfilenames(
            parent=root,
            title=f"Seleccionar imágenes — {phase_title}",
            filetypes=[
                ("Imágenes", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("Todos", "*.*"),
            ],
        )
        if filepaths:
            result["action"] = "image"
            result["paths"] = list(filepaths)
            root.destroy()

    def on_back():
        result["action"] = "exit"
        root.destroy()

    _make_button(root, "🖼  Subir imágenes", color, on_image).pack(pady=(0, 10))
    _back_button(root, on_back).pack(pady=(5, 20))

    _center_window(root, 520, 340)
    root.mainloop()
    return result["action"], result.get("paths", [])


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
        show_explanation_window(stats.get("explanation"), parent=root)

    if stats.get("explanation") is not None:
        _make_button(
            root,
            "🧠  Ver en que se ha fijado el modelo",
            ACCENT_BLUE,
            on_explanation,
        ).pack(pady=(0, 8))

    if has_detections and has_next_phase:
        if phase_num == 2:
            next_label = "▶  Siguiente"
        else:
            next_label = "▶  Siguiente"
        _make_button(root, next_label, ACCENT_MAUVE, on_next).pack(pady=(0, 8))

    _back_button(root, on_restart).pack(pady=(0, 20))

    _center_window(root, 560, 650)
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
    total_images = int(result_data.get("total_images", 1))
    malignant_count = int(result_data.get("malignant_count", 1 if is_malignant else 0))
    if is_malignant:
        color = ACCENT_RED
        text = "⚠️  MUESTRA MALIGNA" if total_images == 1 else f"⚠️  {malignant_count}/{total_images} IMÁGENES MALIGNAS"
        subtext = f"Mayor señal maligna con confianza {confidence:.1%}"
    else:
        color = ACCENT_GREEN
        text = "✅  MUESTRA NO MALIGNA" if total_images == 1 else "✅  SIN IMÁGENES MALIGNAS"
        subtext = (
            f"Máxima confianza revisada {confidence:.1%}"
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
    _metric_row(metrics, "Imágenes analizadas", str(total_images))
    _metric_row(metrics, "Imagen destacada", result_data["image_name"])
    if total_images > 1:
        _metric_row(
            metrics,
            "Malignas / no malignas",
            f"{malignant_count} / {result_data.get('non_malignant_count', 0)}",
            ACCENT_RED if malignant_count > 0 else ACCENT_GREEN,
        )
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
        "Al menos una imagen apunta a malignidad y debe revisarse clínicamente."
        if is_malignant
        else "Ninguna imagen fue clasificada como maligna en esta fase."
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
        show_explanation_window(result_data.get("explanation"), parent=root)

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

    _center_window(root, 600, 540)
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
             f"{phase3_result.get('malignant_count', 1)}/{phase3_result.get('total_images', 1)} imágenes malignas ({phase3_result['confidence']:.1%})"
             if phase3_malignant else
             f"0/{phase3_result.get('total_images', 1)} imágenes malignas ({phase3_result['confidence']:.1%})"
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
            phase1_explanation = explain_history_prediction(history_model, patient_data)
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
                "explanation": phase1_explanation,
            }

            act = show_phase1_result(
                patient_data,
                is_pos,
                prob,
                demo_mode=phase1_result["demo_mode"],
                numeric_fields=numeric_fields,
                explanation=phase1_explanation,
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
            phase2_stats["executed"] = True
            if act == "next":
                return True, phase2_stats
            return False, phase2_stats

        def run_phase3(has_next_phase: bool = False) -> tuple[bool, dict[str, Any]]:
            """Ejecuta Fase 3. Devuelve (quiere_ver_resultado, resumen_de_fase)."""
            print()
            print("─" * 55)
            print("  FASE 3: ANÁLISIS MICROSCÓPICO")
            print("─" * 55)

            act, image_paths = show_image_menu(
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
                    "images": [],
                    "total_images": 0,
                    "malignant_count": 0,
                    "non_malignant_count": 0,
                }

            phase3_result = process_image_batch_classification(
                image_paths, microscopy_model,
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
            patient_name = ask_patient_name()
            if patient_name is None:
                continue

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
                "images": [],
                "total_images": 0,
                "malignant_count": 0,
                "non_malignant_count": 0,
            }

            advance, phase1_result = run_phase1()
            if not advance:
                if phase1_result.get("executed") and not phase1_result["is_positive"]:
                    save_patient_consultation_history(
                        patient_name,
                        phase1_result,
                        default_phase2_result,
                        default_phase3_result,
                    )
                    show_final_result(
                        phase1_result=phase1_result,
                        phase2_result=default_phase2_result,
                        phase3_result=default_phase3_result,
                    )
                continue

            advance, phase2_result = run_phase2(has_next_phase=True)
            if not advance:
                if phase2_result.get("executed") and phase2_result["unique_polyps"] == 0:
                    save_patient_consultation_history(
                        patient_name,
                        phase1_result,
                        phase2_result,
                        default_phase3_result,
                    )
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
                    + f"{phase3_result.get('malignant_count', 0)}/"
                    + f"{phase3_result.get('total_images', 1)} imágenes malignas"
                )
                print("═" * 55)

                save_patient_consultation_history(
                    patient_name,
                    phase1_result,
                    phase2_result,
                    phase3_result,
                )
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

        elif action == "history":
            show_history_window()

        # Vuelve al menú principal automáticamente

    print()
    print("✓ Aplicación cerrada.")


if __name__ == "__main__":
    main()
