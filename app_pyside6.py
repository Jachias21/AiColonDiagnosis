"""
App principal PySide6 para AiColonDiagnosis.

Redisenada como dashboard clinico con flujo guiado, paneles de metricas,
tablas verificables y graficas embebidas para cada fase.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import QObject, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import detect_realtime as core


APP_BG = "#0b1220"
SURFACE_BG = "#111827"
CARD_BG = "#162235"
CARD_ALT_BG = "#1d2d44"
BORDER = "#28405e"
TEXT_MAIN = "#e8eef8"
TEXT_SUB = "#98abc4"
TEXT_MUTED = "#6e83a0"
BLUE = "#45b0ff"
CYAN = "#27d3c3"
GREEN = "#4bd18a"
RED = "#ff6b7a"
YELLOW = "#f4bf4f"
ORANGE = "#f59e67"
MAUVE = "#b596ff"
PENDING_DIR = Path("pending_consultations")


DEFAULT_PHASE1 = {
    "executed": False,
    "status": "NO_EJECUTADA",
    "is_positive": False,
    "probability": 0.0,
    "demo_mode": True,
    "numeric_fields": 0,
    "total_fields": 0,
    "explanation": None,
    "patient_data": {},
    "source_name": "",
}

DEFAULT_PHASE2 = {
    "executed": False,
    "source_mode": "No iniciado",
    "frames_processed": 0,
    "positive_frames": 0,
    "total_detections": 0,
    "peak_detections": 0,
    "unique_polyps": 0,
    "current_detection_count": 0,
    "currently_detecting": False,
    "avg_confidence": 0.0,
    "max_confidence": 0.0,
    "confidence_threshold": core.COLONOSCOPY_SEGMENTER_THRESHOLD,
    "segmenter_model_loaded": False,
    "segmenter_positive_frames": 0,
    "segmenter_total_detections": 0,
    "segmenter_peak_detections": 0,
    "segmenter_max_confidence": 0.0,
    "segmenter_threshold": core.COLONOSCOPY_SEGMENTER_THRESHOLD,
    "min_confirm_seconds": core.POLYP_CONFIRM_SECONDS,
    "min_confirm_frames": 0,
    "completion_ratio": None,
    "end_reason": "cancelled",
    "explanation": None,
}

DEFAULT_PHASE3 = {
    "executed": False,
    "not_applicable": False,
    "status": "NO_EJECUTADA",
    "skip_reason": "",
    "image_path": "",
    "image_name": "",
    "is_malignant": False,
    "confidence": 0.0,
    "class_name": "NO_EJECUTADA",
    "demo_mode": True,
    "explanation": None,
    "images": [],
    "total_images": 0,
    "malignant_count": 0,
    "non_malignant_count": 0,
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _percent(value: Any) -> str:
    return f"{_safe_float(value):.1%}"


def _resolve_history_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if path.exists():
        return path
    candidate = Path.cwd() / path_value
    if candidate.exists():
        return candidate
    return None


def _status_text(phase_data: dict[str, Any]) -> str:
    if phase_data.get("not_applicable"):
        return "No aplicable"
    if not phase_data.get("executed"):
        return "Pendiente"
    return "Completada"


def _phase2_has_confirmed_polyps(phase2: dict[str, Any]) -> bool:
    return bool(phase2.get("executed") and _safe_int(phase2.get("unique_polyps", 0)) > 0)


def _phase2_findings_value(phase2: dict[str, Any], pending: str = "Pendiente") -> str:
    if not phase2.get("executed"):
        return pending
    return "Si" if _phase2_has_confirmed_polyps(phase2) else "No"


def _phase2_findings_status(phase2: dict[str, Any], pending: str = "Fase 2 pendiente") -> str:
    if not phase2.get("executed"):
        return pending
    return "Hallazgos detectados" if _phase2_has_confirmed_polyps(phase2) else "Sin hallazgos persistentes"


def _phase3_is_not_applicable(phase3: dict[str, Any]) -> bool:
    return bool(phase3.get("not_applicable"))


def _phase3_is_complete(phase3: dict[str, Any]) -> bool:
    return bool(phase3.get("executed") or phase3.get("not_applicable"))


def _conclusion_from_results(
    phase1: dict[str, Any],
    phase2: dict[str, Any],
    phase3: dict[str, Any],
) -> tuple[str, str, str]:
    phase2_positive = _phase2_has_confirmed_polyps(phase2)
    phase3_na = _phase3_is_not_applicable(phase3)
    if phase3.get("is_malignant"):
        return (
            "Posible resultado final: muestra maligna",
            "La fase histologica es la senal mas fuerte del flujo actual y requiere revision medica prioritaria.",
            RED,
        )
    if phase1.get("is_positive") and phase2_positive:
        return (
            "Posible resultado final: caso sospechoso",
            "Hay riesgo clinico y hallazgos endoscopicos. Conviene correlacion clinica y revision experta.",
            ORANGE,
        )
    if phase2.get("executed") and not phase2_positive and phase3_na:
        return (
            "Posible resultado final: sin hallazgos persistentes",
            "La colonoscopia no confirmo hallazgos persistentes y por eso la fase histologica no aplica en esta consulta.",
            GREEN,
        )
    if phase2_positive:
        return (
            "Posible resultado final: hallazgos a revisar",
            "Existen hallazgos en colonoscopia, aunque la histologia no marca malignidad destacada.",
            YELLOW,
        )
    return (
        "Posible resultado final: sin hallazgos malignos claros",
        "No aparece evidencia fuerte de malignidad en las fases ejecutadas.",
        GREEN,
    )


def _save_pending_consultation(
    patient_name: str,
    phase1_result: dict[str, Any],
    phase2_result: dict[str, Any],
    phase3_result: dict[str, Any],
    pending_id: str | None = None,
) -> str:
    """Guarda o actualiza una consulta pendiente para poder retomarla."""
    PENDING_DIR.mkdir(exist_ok=True)
    created_at = datetime.now().isoformat(timespec="seconds")
    if pending_id:
        record_id = pending_id
        record_dir = PENDING_DIR / record_id
        record_dir.mkdir(exist_ok=True)
        metadata_path = record_dir / "metadata.json"
        if metadata_path.exists():
            try:
                existing = json.loads(metadata_path.read_text(encoding="utf-8"))
                created_at = existing.get("created_at", created_at)
            except Exception:
                pass
    else:
        record_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{core._safe_slug(patient_name)}_{int(time.time() * 1000) % 1000:03d}"
        record_dir = PENDING_DIR / record_id
        record_dir.mkdir(exist_ok=True)

    saved_images: dict[str, str] = {}
    phase2_explanation = phase2_result.get("explanation") or {}
    p2_original = core._write_history_image(record_dir, "colonoscopia_frame_original.jpg", phase2_explanation.get("original"))
    p2_focus = core._write_history_image(record_dir, "colonoscopia_enfoque.jpg", phase2_explanation.get("overlay"))
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
        preview = core._write_history_image(record_dir, f"{prefix}_resultado.jpg", image_result.get("preview"))
        if preview:
            saved_images[f"{prefix}_resultado"] = preview
            if idx == 1:
                saved_images["histologia_resultado"] = preview
        explanation = image_result.get("explanation") or {}
        focus = core._write_history_image(record_dir, f"{prefix}_enfoque.jpg", explanation.get("overlay"))
        if focus:
            saved_images[f"{prefix}_enfoque"] = focus
            if idx == 1:
                saved_images["histologia_enfoque"] = focus

    metadata = {
        "id": record_id,
        "kind": "pending",
        "title": f"Consulta pendiente - {patient_name}",
        "patient_name": patient_name,
        "created_at": created_at,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "result": {
            "phase1": core._json_safe(phase1_result),
            "phase2": core._json_safe(phase2_result),
            "phase3": core._json_safe(phase3_result),
        },
        "images": saved_images,
    }
    metadata_path = record_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return record_id


def _load_pending_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not PENDING_DIR.exists():
        return records
    for path in sorted(PENDING_DIR.glob("*/metadata.json"), reverse=True):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    records.sort(key=lambda item: item.get("updated_at", item.get("created_at", "")), reverse=True)
    return records


def _delete_pending_record(record_id: str) -> None:
    record_dir = PENDING_DIR / record_id
    if record_dir.exists():
        shutil.rmtree(record_dir, ignore_errors=True)


@dataclass
class AppModels:
    history_model: Any
    colonoscopy_model: Any
    colonoscopy_segmenter: dict[str, Any] | None
    microscopy_model: dict[str, Any] | None


class MetricCard(QFrame):
    def __init__(self, title: str, accent: str = BLUE):
        super().__init__()
        self._accent = accent
        self.setObjectName("metricCard")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(5)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("metricTitle")
        layout.addWidget(self.title_label)

        self.value_label = QLabel("--")
        self.value_label.setObjectName("metricValue")
        layout.addWidget(self.value_label)

        self.subtitle_label = QLabel("Sin datos")
        self.subtitle_label.setObjectName("metricSubtitle")
        self.subtitle_label.setWordWrap(True)
        layout.addWidget(self.subtitle_label)

        self._set_accent(accent)

    def _set_accent(self, color: str) -> None:
        self._accent = color
        self.setStyleSheet(
            f"""
            QFrame#metricCard {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #15263b, stop:0.55 #18304a, stop:1 #122336);
                border: 1px solid {BORDER};
                border-left: 4px solid {color};
                border-radius: 18px;
            }}
            QFrame#metricCard QLabel {{
                background: transparent;
                border: none;
            }}
            """
        )

    def set_data(self, value: str, subtitle: str, accent: str | None = None) -> None:
        self.value_label.setText(value)
        self.subtitle_label.setText(subtitle)
        if accent is not None and accent != self._accent:
            self._set_accent(accent)


class InsightCard(QFrame):
    def __init__(self, title: str):
        super().__init__()
        self.setObjectName("insightCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("sectionTitle")
        layout.addWidget(self.title_label)

        self.body_label = QLabel("")
        self.body_label.setObjectName("bodyText")
        self.body_label.setWordWrap(True)
        layout.addWidget(self.body_label)

    def set_text(self, text: str) -> None:
        self.body_label.setText(text)


class BarChartWidget(QWidget):
    def __init__(self, title: str = ""):
        super().__init__()
        self.title = title
        self.items: list[dict[str, Any]] = []
        self.setMinimumHeight(220)

    def set_items(self, items: list[dict[str, Any]]) -> None:
        self.items = items[:8]
        self.update()

    def paintEvent(self, _event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(CARD_BG))

        rect = self.rect().adjusted(18, 16, -18, -16)
        painter.setPen(QColor(TEXT_MAIN))
        title_font = QFont("Segoe UI", 11, QFont.Weight.DemiBold)
        painter.setFont(title_font)
        painter.drawText(rect.adjusted(0, 0, 0, 0), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft, self.title)

        chart_rect = rect.adjusted(0, 32, 0, 0)
        if not self.items:
            painter.setPen(QColor(TEXT_SUB))
            painter.setFont(QFont("Segoe UI", 10))
            painter.drawText(chart_rect, Qt.AlignmentFlag.AlignCenter, "Sin datos para representar")
            return

        max_value = max(max(float(item.get("value", 0.0)), 0.0) for item in self.items) or 1.0
        row_h = max(28, int(chart_rect.height() / max(len(self.items), 1)))
        label_w = min(180, int(chart_rect.width() * 0.38))
        value_w = 70
        track_x = chart_rect.x() + label_w + 10
        track_w = max(chart_rect.width() - label_w - value_w - 22, 40)

        for idx, item in enumerate(self.items):
            y = chart_rect.y() + idx * row_h
            label_rect = QRectF(chart_rect.x(), y + 2, label_w, row_h - 4)
            painter.setPen(QColor(TEXT_SUB))
            painter.setFont(QFont("Segoe UI", 9))
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, str(item.get("label", "")))

            track_rect = QRectF(track_x, y + row_h * 0.35, track_w, 12)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#22354c"))
            painter.drawRoundedRect(track_rect, 6, 6)

            value = max(float(item.get("value", 0.0)), 0.0)
            ratio = min(value / max_value, 1.0)
            fill_rect = QRectF(track_rect.x(), track_rect.y(), track_rect.width() * ratio, track_rect.height())
            painter.setBrush(QColor(item.get("color", BLUE)))
            painter.drawRoundedRect(fill_rect, 6, 6)

            value_rect = QRectF(track_rect.right() + 8, y + 2, value_w, row_h - 4)
            painter.setPen(QColor(TEXT_MAIN))
            painter.drawText(value_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, str(item.get("text", "")))


class ImageComparisonDialog(QMainWindow):
    def __init__(
        self,
        title: str,
        original: np.ndarray | None = None,
        overlay: np.ndarray | None = None,
        original_path: Path | None = None,
        overlay_path: Path | None = None,
        subtitle: str = "",
    ) -> None:
        super().__init__()
        self.setWindowTitle(title)
        self.resize(1280, 760)

        original = self._load_frame(original, original_path)
        overlay = self._load_frame(overlay, overlay_path)

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        header = QLabel(title)
        header.setStyleSheet("font-size: 24px; font-weight: 700; color: #f3f8ff;")
        layout.addWidget(header)

        if subtitle:
            subtitle_label = QLabel(subtitle)
            subtitle_label.setWordWrap(True)
            subtitle_label.setStyleSheet(f"color: {TEXT_SUB}; font-size: 13px;")
            layout.addWidget(subtitle_label)

        self.consultation_split = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(self.consultation_split, 1)

        self.original_label = self._image_panel("Original", original)
        self.overlay_label = self._image_panel("Enfoque del modelo", overlay)
        split.addWidget(self.original_label)
        split.addWidget(self.overlay_label)
        split.setSizes([620, 620])

        close_btn = QPushButton("Cerrar vista")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn, 0, Qt.AlignmentFlag.AlignRight)

        self.setStyleSheet(
            f"""
            QMainWindow, QWidget {{
                background: {APP_BG};
                color: {TEXT_MAIN};
                font-family: 'Segoe UI';
            }}
            QLabel[panelTitle="true"] {{
                font-size: 15px;
                font-weight: 700;
                color: {TEXT_MAIN};
            }}
            QLabel[imagePanel="true"] {{
                background: {CARD_BG};
                border: 1px solid {BORDER};
                border-radius: 16px;
                color: {TEXT_SUB};
            }}
            QPushButton {{
                background: #24415d;
                border: 1px solid #3f6388;
                border-radius: 12px;
                padding: 10px 16px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background: #2d4f72;
            }}
            """
        )

    @staticmethod
    def _load_frame(frame: np.ndarray | None, path: Path | None) -> np.ndarray | None:
        if frame is not None:
            return frame
        if path is not None and path.exists():
            return cv2.imread(str(path))
        return None

    def _image_panel(self, title: str, frame: np.ndarray | None) -> QWidget:
        wrapper = QFrame()
        wrapper.setStyleSheet(
            f"QFrame {{ background: {CARD_BG}; border: 1px solid {BORDER}; border-radius: 18px; }}"
        )
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title_label = QLabel(title)
        title_label.setProperty("panelTitle", True)
        layout.addWidget(title_label)

        image_label = QLabel("No hay imagen disponible")
        image_label.setProperty("imagePanel", True)
        image_label.setMinimumSize(540, 580)
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(image_label, 1)

        if frame is not None:
            qimg = VideoPhaseController.to_qimage(frame)
            pix = QPixmap.fromImage(qimg).scaled(
                560,
                600,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            image_label.setPixmap(pix)

        return wrapper


class VideoPhaseController(QObject):
    frame_ready = Signal(QImage)
    stats_changed = Signal(dict)
    finished = Signal(dict)
    info = Signal(str)

    def __init__(self, model: Any):
        super().__init__()
        self.model = model
        self.segmenter_model: dict[str, Any] | None = None
        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self.cap: cv2.VideoCapture | None = None
        self.mode = "Video"
        self.total_frames = 0
        self.frame_count = 0
        self.total_positives = 0
        self.total_detections = 0
        self.peak_detections = 0
        self.unique_polyps = 0
        self.current_detection_count = 0
        self.confidence_sum = 0.0
        self.max_confidence = 0.0
        self.paused = False
        self.prev_time = time.time()
        self.fps_display = 0.0
        self.best_detection_score = -1.0
        self.best_focus_frame: np.ndarray | None = None
        self.best_focus_detections: list[dict[str, Any]] = []
        self.segmenter_positive_frames = 0
        self.segmenter_total_detections = 0
        self.segmenter_peak_detections = 0
        self.segmenter_max_confidence = 0.0
        self.active_tracks: list[dict[str, Any]] = []
        self.next_track_id = 1
        self.min_confirm_frames = 0
        self.max_missing_frames = 0
        self.end_reason = "completed"
        self.last_visual_frame: np.ndarray | None = None
        self.source_fps = 30.0

    def set_segmenter_model(self, segmenter_model: dict[str, Any] | None) -> None:
        self.segmenter_model = segmenter_model

    def start(self, source: int | str, mode: str) -> bool:
        if self.cap is not None:
            self.stop("restart")

        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            self.info.emit("No se pudo abrir la fuente de video.")
            self.cap = None
            return False

        self.mode = mode
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if mode == "Video" else 0
        src_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.source_fps = float(src_fps)
        effective_fps = max(src_fps / max(core.FRAME_SKIP, 1), 1.0)
        self.min_confirm_frames = max(4, min(6, int(round(core.POLYP_CONFIRM_SECONDS * effective_fps))))
        self.max_missing_frames = max(6, int(round(core.POLYP_TRACK_MAX_MISSING_SECONDS * effective_fps)))
        self.frame_count = 0
        self.total_positives = 0
        self.total_detections = 0
        self.peak_detections = 0
        self.unique_polyps = 0
        self.current_detection_count = 0
        self.confidence_sum = 0.0
        self.max_confidence = 0.0
        self.paused = False
        self.prev_time = time.time()
        self.fps_display = 0.0
        self.best_detection_score = -1.0
        self.best_focus_frame = None
        self.best_focus_detections = []
        self.segmenter_positive_frames = 0
        self.segmenter_total_detections = 0
        self.segmenter_peak_detections = 0
        self.segmenter_max_confidence = 0.0
        self.active_tracks = []
        self.next_track_id = 1
        self.end_reason = "completed"
        self.last_visual_frame = None
        self.timer.start(15)
        return True

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def is_seekable(self) -> bool:
        return self.cap is not None and self.mode == "Video" and self.total_frames > 0

    def seek_relative_seconds(self, seconds: float) -> bool:
        if not self.is_seekable() or self.cap is None:
            self.info.emit("El avance y retroceso solo estan disponibles para archivos de video.")
            return False

        fps = self.source_fps or self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        target_frame = current_frame + int(round(seconds * fps))
        target_frame = max(0, min(target_frame, max(self.total_frames - 1, 0)))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        self.frame_count = target_frame
        self.info.emit(
            f"Posicion del video: {target_frame / max(fps, 1.0):.1f}s / {self.total_frames / max(fps, 1.0):.1f}s"
        )
        return True

    def stop(self, reason: str = "button") -> None:
        self.end_reason = reason
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        avg_confidence = self.confidence_sum / self.total_detections if self.total_detections > 0 else 0.0
        completion_ratio = min(self.frame_count / self.total_frames, 1.0) if self.total_frames > 0 else None
        stats = {
            "executed": True,
            "source_mode": self.mode,
            "frames_processed": self.frame_count,
            "positive_frames": self.total_positives,
            "total_detections": self.total_detections,
            "peak_detections": self.peak_detections,
            "unique_polyps": self.unique_polyps,
            "current_detection_count": self.current_detection_count,
            "currently_detecting": self.current_detection_count > 0,
            "avg_confidence": avg_confidence,
            "max_confidence": self.max_confidence,
            "confidence_threshold": core.COLONOSCOPY_SEGMENTER_THRESHOLD,
            "segmenter_model_loaded": self.segmenter_model is not None,
            "segmenter_positive_frames": self.segmenter_positive_frames,
            "segmenter_total_detections": self.segmenter_total_detections,
            "segmenter_peak_detections": self.segmenter_peak_detections,
            "segmenter_max_confidence": self.segmenter_max_confidence,
            "segmenter_threshold": core.COLONOSCOPY_SEGMENTER_THRESHOLD,
            "min_confirm_seconds": core.POLYP_CONFIRM_SECONDS,
            "min_confirm_frames": self.min_confirm_frames,
            "completion_ratio": completion_ratio,
            "end_reason": self.end_reason,
            "explanation": (
                core.create_detection_explanation(self.best_focus_frame, self.best_focus_detections)
                if self.best_focus_frame is not None
                else None
            ),
        }
        self.stats_changed.emit(stats)
        self.finished.emit(stats)

    def current_frame(self) -> np.ndarray | None:
        return self.last_visual_frame.copy() if self.last_visual_frame is not None else None

    @staticmethod
    def _draw_panel_title(frame: np.ndarray, title: str, detections_count: int, color: tuple[int, int, int]) -> np.ndarray:
        label = f"{title} · detecciones: {detections_count}"
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 42), (10, 18, 32), -1)
        cv2.putText(frame, label, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)
        return frame

    @staticmethod
    def _compose_comparison_frame(yolo_frame: np.ndarray, segmenter_frame: np.ndarray) -> np.ndarray:
        target_h = min(yolo_frame.shape[0], segmenter_frame.shape[0])
        if yolo_frame.shape[0] != target_h:
            yolo_frame = cv2.resize(
                yolo_frame,
                (int(yolo_frame.shape[1] * target_h / yolo_frame.shape[0]), target_h),
                interpolation=cv2.INTER_AREA,
            )
        if segmenter_frame.shape[0] != target_h:
            segmenter_frame = cv2.resize(
                segmenter_frame,
                (int(segmenter_frame.shape[1] * target_h / segmenter_frame.shape[0]), target_h),
                interpolation=cv2.INTER_AREA,
            )
        separator = np.full((target_h, 6, 3), (22, 34, 53), dtype=np.uint8)
        return np.hstack([yolo_frame, separator, segmenter_frame])

    def _tick(self) -> None:
        if self.cap is None:
            return

        if self.paused:
            if self.last_visual_frame is not None:
                self.frame_ready.emit(self.to_qimage(self.last_visual_frame))
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop("completed")
            return

        self.frame_count += 1
        if core.FRAME_SKIP > 1 and self.frame_count % core.FRAME_SKIP != 0:
            return

        detections = core.run_segmenter_inference(
            self.segmenter_model,
            frame,
            threshold=core.COLONOSCOPY_SEGMENTER_THRESHOLD,
        )

        detections_in_frame = len(detections)
        self.current_detection_count = detections_in_frame
        self.active_tracks, self.next_track_id, new_unique_polyps = core._update_polyp_tracks(
            detections,
            self.active_tracks,
            self.next_track_id,
            iou_threshold=core.POLYP_TRACK_IOU_THRESHOLD,
            max_missing_frames=self.max_missing_frames,
            min_confirm_frames=self.min_confirm_frames,
        )
        self.unique_polyps += new_unique_polyps

        if detections_in_frame:
            self.total_positives += 1
            self.total_detections += detections_in_frame
            self.peak_detections = max(self.peak_detections, detections_in_frame)
            self.confidence_sum += sum(float(det["confidence"]) for det in detections)
            self.max_confidence = max(self.max_confidence, max(float(det["confidence"]) for det in detections))
            score = sum(float(det["confidence"]) for det in detections)
            if score > self.best_detection_score:
                self.best_detection_score = score
                self.best_focus_frame = frame.copy()
                self.best_focus_detections = [det.copy() for det in detections]

        if detections_in_frame:
            self.segmenter_positive_frames += 1
            self.segmenter_total_detections += detections_in_frame
            self.segmenter_peak_detections = max(self.segmenter_peak_detections, detections_in_frame)
            self.segmenter_max_confidence = max(
                self.segmenter_max_confidence,
                max(float(det["confidence"]) for det in detections),
            )

        if self.segmenter_model is None:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 42), (10, 18, 32), -1)
            cv2.putText(
                frame,
                "UNet3+ segmentador no cargado",
                (14, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (80, 180, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            frame = core.draw_detections(frame, detections, "default")
            frame = self._draw_panel_title(
                frame,
                "UNet3+ segmentacion",
                detections_in_frame,
                (255, 210, 70),
            )

        now = time.time()
        dt = now - self.prev_time
        if dt > 0:
            self.fps_display = 0.7 * self.fps_display + 0.3 * (1.0 / dt)
        self.prev_time = now

        frame = core.draw_hud(
            frame,
            fps=self.fps_display,
            frame_num=self.frame_count,
            total_frames=self.total_frames,
            phase_label="Fase 2: Colonoscopia",
            detections_count=detections_in_frame,
            model_loaded=self.segmenter_model is not None,
            total_positives=self.total_positives,
            paused=self.paused,
        )

        self.last_visual_frame = frame
        self.frame_ready.emit(self.to_qimage(frame))
        self.stats_changed.emit(
            {
                "frames_processed": self.frame_count,
                "positive_frames": self.total_positives,
                "unique_polyps": self.unique_polyps,
                "current_detection_count": self.current_detection_count,
                "currently_detecting": self.current_detection_count > 0,
                "min_confirm_frames": self.min_confirm_frames,
                "peak_detections": self.peak_detections,
                "max_confidence": self.max_confidence,
                "segmenter_positive_frames": self.segmenter_positive_frames,
                "segmenter_total_detections": self.segmenter_total_detections,
                "segmenter_peak_detections": self.segmenter_peak_detections,
                "segmenter_max_confidence": self.segmenter_max_confidence,
            }
        )

    @staticmethod
    def to_qimage(frame_bgr: np.ndarray) -> QImage:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        return QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AiColonDiagnosis - Panel Clinico")
        self.resize(1280, 800)
        self.setMinimumSize(1120, 720)

        self.models = self._load_models()

        self.current_consultation_record: dict[str, Any] | None = None
        self.current_pending_id: str | None = None
        self.active_patient_name = ""
        self.session_ready = False
        self.report_unlocked = False
        self.read_only_history_mode = False
        self.phase2_override_acknowledged = False

        self.phase1_result = dict(DEFAULT_PHASE1)
        self.phase2_result = dict(DEFAULT_PHASE2)
        self.phase3_result = dict(DEFAULT_PHASE3)
        self.phase3_selected_index = 0

        self.video_controller = VideoPhaseController(None)
        self.video_controller.set_segmenter_model(self.models.colonoscopy_segmenter)
        self.video_controller.frame_ready.connect(self._on_video_frame)
        self.video_controller.stats_changed.connect(self._on_video_stats)
        self.video_controller.finished.connect(self._on_video_finished)
        self.video_controller.info.connect(self._show_status)

        self._build_ui()
        self._apply_style()
        self._refresh_all_views()
        self._center_on_screen()

    def _center_on_screen(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        geometry = screen.availableGeometry()
        frame = self.frameGeometry()
        frame.moveCenter(geometry.center())
        self.move(frame.topLeft())

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        QTimer.singleShot(0, self._update_responsive_layout)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._update_responsive_layout()

    def _load_models(self) -> AppModels:
        self.statusBar().showMessage("Cargando modelos...")
        history_model = core.load_history_model(core.MODEL_HISTORY)
        colonoscopy_model = None  # YOLO queda disponible en models/, pero Fase 2 usa UNet3+ como principal.
        colonoscopy_segmenter = core.load_polyp_segmenter(core.MODEL_COLONOSCOPY_SEGMENTER, "Colonoscopia UNet3+")
        microscopy_model = core.load_classification_model(
            core.MODEL_MICROSCOPY,
            core.MODEL_MICROSCOPY_META,
            "Microscopio",
        )
        return AppModels(history_model, colonoscopy_model, colonoscopy_segmenter, microscopy_model)

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("appRoot")
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        nav = QFrame()
        nav.setObjectName("nav")
        nav_layout = QVBoxLayout(nav)
        nav_layout.setContentsMargins(18, 18, 18, 18)
        nav_layout.setSpacing(12)

        title = QLabel("AiColonDiagnosis")
        title.setObjectName("navTitle")
        nav_layout.addWidget(title)

        subtitle = QLabel("Panel de consulta asistida por IA")
        subtitle.setObjectName("subtle")
        subtitle.setWordWrap(True)
        nav_layout.addWidget(subtitle)

        self.patient_banner = QLabel("Sin consulta activa")
        self.patient_banner.setObjectName("patientBanner")
        self.patient_banner.setWordWrap(True)
        nav_layout.addWidget(self.patient_banner)

        self.btn_load_history = self._nav_button("Historial y pendientes")
        self.btn_new_consultation = self._nav_button("Crear nueva consulta")
        self.btn_home = self._nav_button("Resumen de consulta")
        self.btn_phase1 = self._nav_button("Fase 1 · Historial")
        self.btn_phase2 = self._nav_button("Fase 2 · Colonoscopia")
        self.btn_phase3 = self._nav_button("Fase 3 · Histologia")
        self.btn_report = self._nav_button("Informe final")
        self.btn_back = self._nav_button("Volver al inicio")
        self.btn_close = self._nav_button("Cerrar")

        for btn in (
            self.btn_load_history,
            self.btn_new_consultation,
            self.btn_home,
            self.btn_phase1,
            self.btn_phase2,
            self.btn_phase3,
            self.btn_report,
        ):
            nav_layout.addWidget(btn)

        nav_layout.addStretch(1)
        nav_layout.addWidget(self.btn_back)
        nav_layout.addWidget(self.btn_close)

        self.stack = QStackedWidget()
        self.page_home = self._build_home_page()
        self.page_consultation = self._build_consultation_page()
        self.page_phase1 = self._build_phase1_page()
        self.page_phase2 = self._build_phase2_page()
        self.page_phase3 = self._build_phase3_page()
        self.page_report = self._build_report_page()

        for page in (
            self.page_home,
            self.page_consultation,
            self.page_phase1,
            self.page_phase2,
            self.page_phase3,
            self.page_report,
        ):
            self.stack.addWidget(page)

        outer.addWidget(nav, 0)
        outer.addWidget(self.stack, 1)

        self.btn_load_history.clicked.connect(self._show_consultation_page)
        self.btn_new_consultation.clicked.connect(self._create_new_consultation)
        self.btn_home.clicked.connect(lambda: self.stack.setCurrentWidget(self.page_home))
        self.btn_phase1.clicked.connect(lambda: self.stack.setCurrentWidget(self.page_phase1))
        self.btn_phase2.clicked.connect(self._show_phase2_page)
        self.btn_phase3.clicked.connect(self._show_phase3_page)
        self.btn_report.clicked.connect(lambda: self.stack.setCurrentWidget(self.page_report))
        self.btn_back.clicked.connect(self._back_to_main_options)
        self.btn_close.clicked.connect(self.close)
        self.stack.currentChanged.connect(self._on_stack_changed)

        self._update_navigation()

    def _build_scroll_page(self, content: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(content)
        return scroll

    def _build_home_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(18)

        hero = QFrame()
        hero.setObjectName("heroCard")
        hero_layout = QVBoxLayout(hero)
        hero_layout.setContentsMargins(24, 22, 24, 22)
        hero_layout.setSpacing(10)

        title = QLabel("Panel de verificacion clinica")
        title.setObjectName("pageTitle")
        hero_layout.addWidget(title)

        self.home_patient_label = QLabel("No hay consulta activa.")
        self.home_patient_label.setObjectName("bodyText")
        self.home_patient_label.setWordWrap(True)
        hero_layout.addWidget(self.home_patient_label)

        self.home_conclusion = QLabel("Inicia una consulta nueva o carga un historial para ver el resumen.")
        self.home_conclusion.setObjectName("alertNeutral")
        self.home_conclusion.setWordWrap(True)
        hero_layout.addWidget(self.home_conclusion)
        layout.addWidget(hero)

        cards = QGridLayout()
        cards.setHorizontalSpacing(14)
        cards.setVerticalSpacing(14)
        self.home_card_phase1 = MetricCard("Riesgo clinico", YELLOW)
        self.home_card_phase2 = MetricCard("Colonoscopia", GREEN)
        self.home_card_phase3 = MetricCard("Histologia", MAUVE)
        self.home_card_flow = MetricCard("Siguiente paso", BLUE)
        cards.addWidget(self.home_card_phase1, 0, 0)
        cards.addWidget(self.home_card_phase2, 0, 1)
        cards.addWidget(self.home_card_phase3, 1, 0)
        cards.addWidget(self.home_card_flow, 1, 1)
        layout.addLayout(cards)

        middle = QHBoxLayout()
        middle.setSpacing(14)

        phase_panel = QFrame()
        phase_panel.setObjectName("sectionCard")
        phase_layout = QVBoxLayout(phase_panel)
        phase_layout.setContentsMargins(18, 16, 18, 16)
        phase_layout.setSpacing(12)
        phase_title = QLabel("Estado del flujo")
        phase_title.setObjectName("sectionTitle")
        phase_layout.addWidget(phase_title)

        self.flow_phase1 = self._make_phase_status_row("Fase 1", "Historial clinico")
        self.flow_phase2 = self._make_phase_status_row("Fase 2", "Colonoscopia")
        self.flow_phase3 = self._make_phase_status_row("Fase 3", "Foto histologica")
        for row in (self.flow_phase1, self.flow_phase2, self.flow_phase3):
            phase_layout.addWidget(row)
        middle.addWidget(phase_panel, 1)

        self.home_chart = BarChartWidget("Indicadores clave de la consulta")
        self.home_chart.setObjectName("chartCard")
        middle.addWidget(self.home_chart, 1)
        layout.addLayout(middle)

        lower = QHBoxLayout()
        lower.setSpacing(14)
        self.home_recommendation = InsightCard("Lectura orientativa")
        self.home_checklist = InsightCard("Checklist de validacion")
        lower.addWidget(self.home_recommendation, 1)
        lower.addWidget(self.home_checklist, 1)
        layout.addLayout(lower)

        return self._build_scroll_page(page)

    def _build_consultation_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(16)

        header = QLabel("Historiales y consultas pendientes")
        header.setObjectName("pageTitle")
        layout.addWidget(header)

        subtitle = QLabel(
            "Puedes retomar consultas pendientes o abrir historiales ya finalizados para revisar las tres fases."
        )
        subtitle.setObjectName("subtle")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        self.consultation_split = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(self.consultation_split, 1)

        left = QFrame()
        left.setObjectName("sectionCard")
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(18, 16, 18, 16)
        left_layout.setSpacing(12)
        left_layout.addWidget(QLabel("Consultas pendientes"))
        self.pending_list = QListWidget()
        self.pending_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        left_layout.addWidget(self.pending_list, 1)
        pending_actions = QHBoxLayout()
        self.btn_continue_pending = QPushButton("Continuar")
        self.btn_delete_pending = QPushButton("Eliminar")
        self.btn_continue_pending.clicked.connect(self._load_selected_pending)
        self.btn_delete_pending.clicked.connect(self._delete_selected_pending)
        pending_actions.addWidget(self.btn_continue_pending)
        pending_actions.addWidget(self.btn_delete_pending)
        left_layout.addLayout(pending_actions)

        left_layout.addWidget(QLabel("Historial completado"))
        self.consultation_list = QListWidget()
        self.consultation_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        left_layout.addWidget(self.consultation_list, 1)
        actions = QHBoxLayout()
        self.btn_load_consultation = QPushButton("Cargar perfil")
        self.btn_refresh_consultations = QPushButton("Actualizar")
        self.btn_load_consultation.clicked.connect(self._load_selected_consultation)
        self.btn_refresh_consultations.clicked.connect(self._refresh_consultation_lists)
        actions.addWidget(self.btn_load_consultation)
        actions.addWidget(self.btn_refresh_consultations)
        left_layout.addLayout(actions)
        self.consultation_split.addWidget(left)

        right = QFrame()
        right.setObjectName("sectionCard")
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(18, 16, 18, 16)
        right_layout.setSpacing(14)

        self.consultation_title = QLabel("Selecciona una consulta")
        self.consultation_title.setObjectName("sectionTitle")
        right_layout.addWidget(self.consultation_title)

        consultation_cards = QGridLayout()
        consultation_cards.setHorizontalSpacing(12)
        consultation_cards.setVerticalSpacing(12)
        self.consultation_card_phase1 = MetricCard("Fase 1", YELLOW)
        self.consultation_card_phase2 = MetricCard("Fase 2", GREEN)
        self.consultation_card_phase3 = MetricCard("Fase 3", MAUVE)
        self.consultation_card_created = MetricCard("Consulta", BLUE)
        consultation_cards.addWidget(self.consultation_card_phase1, 0, 0)
        consultation_cards.addWidget(self.consultation_card_phase2, 0, 1)
        consultation_cards.addWidget(self.consultation_card_phase3, 1, 0)
        consultation_cards.addWidget(self.consultation_card_created, 1, 1)
        right_layout.addLayout(consultation_cards)

        self.consultation_chart = BarChartWidget("Balance clinico guardado")
        right_layout.addWidget(self.consultation_chart)

        consultation_focus_actions = QHBoxLayout()
        self.btn_consultation_phase1_focus = QPushButton("Ver F1 · SHAP")
        self.btn_consultation_phase2_focus = QPushButton("Ver F2 · Mapa de calor")
        self.btn_consultation_phase3_focus = QPushButton("Ver F3 · Mapa de calor")
        self.btn_consultation_phase1_focus.clicked.connect(self._open_selected_record_phase1_explanation)
        self.btn_consultation_phase2_focus.clicked.connect(self._open_selected_record_phase2_explanation)
        self.btn_consultation_phase3_focus.clicked.connect(self._open_selected_record_phase3_explanation)
        consultation_focus_actions.addWidget(self.btn_consultation_phase1_focus)
        consultation_focus_actions.addWidget(self.btn_consultation_phase2_focus)
        consultation_focus_actions.addWidget(self.btn_consultation_phase3_focus)
        consultation_focus_actions.addStretch(1)
        right_layout.addLayout(consultation_focus_actions)

        self.consultation_detail = QTextEdit()
        self.consultation_detail.setReadOnly(True)
        self.consultation_detail.setMinimumHeight(150)
        right_layout.addWidget(self.consultation_detail)
        self.consultation_split.addWidget(right)
        self.consultation_split.setSizes([420, 840])

        self.pending_list.currentItemChanged.connect(self._on_pending_selection)
        self.consultation_list.currentItemChanged.connect(self._on_consultation_selection)
        self._update_consultation_focus_buttons(None)
        self._refresh_consultation_lists()
        return self._build_scroll_page(page)

    def _build_phase1_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(16)

        header_row = QHBoxLayout()
        title = QLabel("Fase 1 · Historial medico")
        title.setObjectName("pageTitle")
        header_row.addWidget(title)
        header_row.addStretch(1)
        self.btn_run_phase1 = QPushButton("Cargar CSV/JSON y analizar")
        self.btn_run_phase1.clicked.connect(self._run_phase1)
        self.btn_phase1_explanation = QPushButton("Ver explicacion SHAP")
        self.btn_phase1_explanation.clicked.connect(self._open_phase1_explanation)
        header_row.addWidget(self.btn_phase1_explanation)
        header_row.addWidget(self.btn_run_phase1)
        layout.addLayout(header_row)

        cards = QGridLayout()
        cards.setHorizontalSpacing(12)
        cards.setVerticalSpacing(12)
        self.phase1_card_status = MetricCard("Estado", YELLOW)
        self.phase1_card_prob = MetricCard("Probabilidad", BLUE)
        self.phase1_card_fields = MetricCard("Variables procesadas", GREEN)
        self.phase1_card_top = MetricCard("Factor dominante", ORANGE)
        cards.addWidget(self.phase1_card_status, 0, 0)
        cards.addWidget(self.phase1_card_prob, 0, 1)
        cards.addWidget(self.phase1_card_fields, 1, 0)
        cards.addWidget(self.phase1_card_top, 1, 1)
        layout.addLayout(cards)

        self.phase1_split = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(self.phase1_split, 1)

        left = QFrame()
        left.setObjectName("sectionCard")
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(18, 16, 18, 16)
        left_layout.setSpacing(12)
        left_layout.addWidget(self._section_label("Datos clinicos cargados"))
        self.phase1_patient_table = self._make_table(["Campo", "Valor"])
        left_layout.addWidget(self.phase1_patient_table, 1)
        self.phase1_summary = QTextEdit()
        self.phase1_summary.setReadOnly(True)
        self.phase1_summary.setMinimumHeight(120)
        left_layout.addWidget(self.phase1_summary)
        self.phase1_split.addWidget(left)

        right = QFrame()
        right.setObjectName("sectionCard")
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(18, 16, 18, 16)
        right_layout.setSpacing(12)
        self.phase1_chart = BarChartWidget("Variables con mayor impacto")
        right_layout.addWidget(self.phase1_chart)
        right_layout.addWidget(self._section_label("Tabla de explicabilidad"))
        self.phase1_feature_table = self._make_table(["Variable", "Valor", "Impacto", "Direccion"])
        right_layout.addWidget(self.phase1_feature_table, 1)
        self.phase1_split.addWidget(right)
        self.phase1_split.setSizes([520, 760])

        return self._build_scroll_page(page)

    def _build_phase2_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(16)

        title = QLabel("Fase 2 · Colonoscopia")
        title.setObjectName("pageTitle")
        layout.addWidget(title)

        controls_card = QFrame()
        controls_card.setObjectName("sectionCard")
        controls_layout = QHBoxLayout(controls_card)
        controls_layout.setContentsMargins(16, 12, 16, 12)
        controls_layout.setSpacing(8)
        self.btn_phase2_explanation = QPushButton("Ver enfoque del modelo")
        self.btn_phase2_explanation.clicked.connect(self._open_phase2_explanation)
        self.start_video_btn = QPushButton("Abrir video")
        self.start_webcam_btn = QPushButton("Usar webcam")
        self.rewind_btn = QPushButton("<< 10 s")
        self.forward_btn = QPushButton("10 s >>")
        self.pause_btn = QPushButton("Pausar / Reanudar")
        self.stop_btn = QPushButton("Finalizar")
        self.shot_btn = QPushButton("Captura")
        self.start_video_btn.clicked.connect(self._start_phase2_video)
        self.start_webcam_btn.clicked.connect(self._start_phase2_webcam)
        self.rewind_btn.clicked.connect(lambda: self._seek_phase2(-10.0))
        self.forward_btn.clicked.connect(lambda: self._seek_phase2(10.0))
        self.pause_btn.clicked.connect(self.video_controller.toggle_pause)
        self.stop_btn.clicked.connect(lambda: self.video_controller.stop("button"))
        self.shot_btn.clicked.connect(self._save_phase2_screenshot)
        for btn in (
            self.btn_phase2_explanation,
            self.start_video_btn,
            self.start_webcam_btn,
            self.rewind_btn,
            self.forward_btn,
            self.pause_btn,
            self.stop_btn,
            self.shot_btn,
        ):
            controls_layout.addWidget(btn)
        controls_layout.addStretch(1)
        layout.addWidget(controls_card)

        self.phase2_split = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(self.phase2_split, 1)

        left = QFrame()
        left.setObjectName("sectionCard")
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(18, 16, 18, 16)
        left_layout.setSpacing(12)
        self.phase2_left_stack = QStackedWidget()

        live_panel = QWidget()
        live_layout = QVBoxLayout(live_panel)
        live_layout.setContentsMargins(0, 0, 0, 0)
        live_layout.setSpacing(12)
        self.video_status = QLabel("Video no iniciado. Selecciona una fuente para comenzar el analisis.")
        self.video_status.setObjectName("subtle")
        self.video_status.setWordWrap(True)
        live_layout.addWidget(self.video_status)
        self.video_label = QLabel("Esperando fuente de video")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(380, 260)
        self.video_label.setObjectName("videoPanel")
        live_layout.addWidget(self.video_label, 1)
        self.phase2_left_stack.addWidget(live_panel)

        review_panel = QWidget()
        review_layout = QVBoxLayout(review_panel)
        review_layout.setContentsMargins(0, 0, 0, 0)
        review_layout.setSpacing(12)
        self.phase2_review_status = QLabel("Se mostrara aqui la mejor evidencia visual guardada.")
        self.phase2_review_status.setObjectName("subtle")
        self.phase2_review_status.setWordWrap(True)
        review_layout.addWidget(self.phase2_review_status)

        self.phase2_review_split = QSplitter(Qt.Orientation.Horizontal)
        self.phase2_original_preview = QLabel("Aun no hay frame representativo guardado")
        self.phase2_original_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.phase2_original_preview.setMinimumSize(240, 220)
        self.phase2_original_preview.setObjectName("videoPanel")
        self.phase2_review_split.addWidget(self.phase2_original_preview)

        self.phase2_focus_preview = QLabel("El mapa de enfoque aparecera aqui al finalizar el analisis")
        self.phase2_focus_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.phase2_focus_preview.setMinimumSize(240, 220)
        self.phase2_focus_preview.setObjectName("videoPanel")
        self.phase2_review_split.addWidget(self.phase2_focus_preview)
        self.phase2_review_split.setSizes([1, 1])
        review_layout.addWidget(self.phase2_review_split, 1)
        self.phase2_left_stack.addWidget(review_panel)

        left_layout.addWidget(self.phase2_left_stack, 1)
        self.phase2_split.addWidget(left)

        right = QFrame()
        right.setObjectName("sectionCard")
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(18, 16, 18, 16)
        right_layout.setSpacing(12)

        cards = QGridLayout()
        cards.setHorizontalSpacing(10)
        cards.setVerticalSpacing(10)
        self.phase2_card_frames = MetricCard("Frames procesados", BLUE)
        self.phase2_card_candidates = MetricCard("Estado actual", ORANGE)
        self.phase2_card_polyps = MetricCard("Hallazgos detectados", GREEN)
        self.phase2_card_conf = MetricCard("Confianza maxima", RED)
        cards.addWidget(self.phase2_card_frames, 0, 0)
        cards.addWidget(self.phase2_card_candidates, 0, 1)
        cards.addWidget(self.phase2_card_polyps, 1, 0)
        cards.addWidget(self.phase2_card_conf, 1, 1)
        right_layout.addLayout(cards)

        self.phase2_chart = BarChartWidget("Resumen cuantitativo de la exploracion")
        right_layout.addWidget(self.phase2_chart)

        self.phase2_summary = QTextEdit()
        self.phase2_summary.setReadOnly(True)
        self.phase2_summary.setMinimumHeight(130)
        right_layout.addWidget(self.phase2_summary)
        self.phase2_split.addWidget(right)
        self.phase2_split.setSizes([820, 560])

        return self._build_scroll_page(page)

    def _build_phase3_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(16)

        header_row = QHBoxLayout()
        title = QLabel("Fase 3 · Foto histologica")
        title.setObjectName("pageTitle")
        header_row.addWidget(title)
        header_row.addStretch(1)
        self.btn_phase3_explanation = QPushButton("Ver enfoque del modelo")
        self.btn_phase3_explanation.clicked.connect(self._open_phase3_explanation)
        self.btn_run_phase3 = QPushButton("Seleccionar una o varias imagenes")
        self.btn_run_phase3.clicked.connect(self._run_phase3)
        header_row.addWidget(self.btn_phase3_explanation)
        header_row.addWidget(self.btn_run_phase3)
        layout.addLayout(header_row)

        cards = QGridLayout()
        cards.setHorizontalSpacing(12)
        cards.setVerticalSpacing(12)
        self.phase3_card_result = MetricCard("Resultado principal", MAUVE)
        self.phase3_card_conf = MetricCard("Confianza destacada", RED)
        self.phase3_card_volume = MetricCard("Muestras analizadas", BLUE)
        self.phase3_card_balance = MetricCard("Balance malignidad", GREEN)
        cards.addWidget(self.phase3_card_result, 0, 0)
        cards.addWidget(self.phase3_card_conf, 0, 1)
        cards.addWidget(self.phase3_card_volume, 1, 0)
        cards.addWidget(self.phase3_card_balance, 1, 1)
        layout.addLayout(cards)

        self.phase3_split = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(self.phase3_split, 1)

        left = QFrame()
        left.setObjectName("sectionCard")
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(18, 16, 18, 16)
        left_layout.setSpacing(12)
        self.phase3_preview_title = QLabel("Muestra destacada")
        self.phase3_preview_title.setObjectName("sectionTitle")
        left_layout.addWidget(self.phase3_preview_title)
        phase3_visuals = QHBoxLayout()
        phase3_visuals.setSpacing(12)
        self.phase3_visual_split = QSplitter(Qt.Orientation.Horizontal)
        self.phase3_preview = QLabel("Sin muestra seleccionada")
        self.phase3_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.phase3_preview.setMinimumSize(240, 260)
        self.phase3_preview.setObjectName("videoPanel")
        self.phase3_visual_split.addWidget(self.phase3_preview)
        self.phase3_result_preview = QLabel("La vista analizada aparecera aqui al procesar una muestra")
        self.phase3_result_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.phase3_result_preview.setMinimumSize(240, 260)
        self.phase3_result_preview.setObjectName("videoPanel")
        self.phase3_visual_split.addWidget(self.phase3_result_preview)
        self.phase3_focus_preview = QLabel("El Grad-CAM aparecera aqui al procesar una muestra")
        self.phase3_focus_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.phase3_focus_preview.setMinimumSize(240, 260)
        self.phase3_focus_preview.setObjectName("videoPanel")
        self.phase3_visual_split.addWidget(self.phase3_focus_preview)
        self.phase3_visual_split.setSizes([1, 1, 1])
        phase3_visuals.addWidget(self.phase3_visual_split, 1)
        left_layout.addLayout(phase3_visuals, 1)
        self.phase3_summary = QTextEdit()
        self.phase3_summary.setReadOnly(True)
        self.phase3_summary.setMinimumHeight(120)
        left_layout.addWidget(self.phase3_summary)
        self.phase3_split.addWidget(left)

        right = QFrame()
        right.setObjectName("sectionCard")
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(18, 16, 18, 16)
        right_layout.setSpacing(12)
        self.phase3_chart = BarChartWidget("Confianza por imagen")
        right_layout.addWidget(self.phase3_chart)
        right_layout.addWidget(self._section_label("Tabla de muestras"))
        self.phase3_table = self._make_table(["Imagen", "Clase", "Confianza", "Resultado"])
        self.phase3_table.itemSelectionChanged.connect(self._on_phase3_table_selection)
        right_layout.addWidget(self.phase3_table, 1)
        self.phase3_split.addWidget(right)
        self.phase3_split.setSizes([820, 560])

        return self._build_scroll_page(page)

    def _build_report_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(16)

        header_row = QHBoxLayout()
        title = QLabel("Informe final de la consulta")
        title.setObjectName("pageTitle")
        header_row.addWidget(title)
        header_row.addStretch(1)
        btn = QPushButton("Regenerar informe")
        btn.clicked.connect(self._update_final_report)
        header_row.addWidget(btn)
        layout.addLayout(header_row)

        self.report_banner = QLabel("Aun no hay datos suficientes para elaborar el informe.")
        self.report_banner.setObjectName("alertNeutral")
        self.report_banner.setWordWrap(True)
        layout.addWidget(self.report_banner)

        report_focus_actions = QHBoxLayout()
        self.btn_report_phase1_focus = QPushButton("Ver F1 · SHAP")
        self.btn_report_phase2_focus = QPushButton("Ver F2 · Mapa de calor")
        self.btn_report_phase3_focus = QPushButton("Ver F3 · Mapa de calor")
        self.btn_report_phase1_focus.clicked.connect(self._open_phase1_explanation)
        self.btn_report_phase2_focus.clicked.connect(self._open_phase2_explanation)
        self.btn_report_phase3_focus.clicked.connect(self._open_phase3_explanation)
        report_focus_actions.addWidget(self.btn_report_phase1_focus)
        report_focus_actions.addWidget(self.btn_report_phase2_focus)
        report_focus_actions.addWidget(self.btn_report_phase3_focus)
        report_focus_actions.addStretch(1)
        layout.addLayout(report_focus_actions)

        cards = QGridLayout()
        cards.setHorizontalSpacing(12)
        cards.setVerticalSpacing(12)
        self.report_card_patient = MetricCard("Paciente", BLUE)
        self.report_card_phase1 = MetricCard("Fase 1", YELLOW)
        self.report_card_phase2 = MetricCard("Fase 2", GREEN)
        self.report_card_phase3 = MetricCard("Fase 3", MAUVE)
        cards.addWidget(self.report_card_patient, 0, 0)
        cards.addWidget(self.report_card_phase1, 0, 1)
        cards.addWidget(self.report_card_phase2, 1, 0)
        cards.addWidget(self.report_card_phase3, 1, 1)
        layout.addLayout(cards)

        middle = QHBoxLayout()
        middle.setSpacing(14)
        self.report_chart = BarChartWidget("Severidad relativa por fase")
        middle.addWidget(self.report_chart, 1)

        right = QFrame()
        right.setObjectName("sectionCard")
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(18, 16, 18, 16)
        right_layout.setSpacing(12)
        right_layout.addWidget(self._section_label("Matriz de decision"))
        self.report_matrix = self._make_table(["Elemento", "Valor", "Lectura"])
        right_layout.addWidget(self.report_matrix, 1)
        middle.addWidget(right, 1)
        layout.addLayout(middle)

        self.final_report = QTextEdit()
        self.final_report.setReadOnly(True)
        layout.addWidget(self.final_report, 1)
        return self._build_scroll_page(page)

    def _make_phase_status_row(self, phase: str, label: str) -> QWidget:
        row = QFrame()
        row.setObjectName("phaseRow")
        layout = QHBoxLayout(row)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(10)

        left = QVBoxLayout()
        title = QLabel(phase)
        title.setObjectName("phaseRowTitle")
        subtitle = QLabel(label)
        subtitle.setObjectName("subtle")
        left.addWidget(title)
        left.addWidget(subtitle)
        layout.addLayout(left, 1)

        progress = QProgressBar()
        progress.setRange(0, 100)
        progress.setValue(0)
        progress.setTextVisible(False)
        progress.setMinimumWidth(180)
        layout.addWidget(progress)

        badge = QLabel("Pendiente")
        badge.setObjectName("badgeIdle")
        layout.addWidget(badge)

        row.progress = progress  # type: ignore[attr-defined]
        row.badge = badge  # type: ignore[attr-defined]
        return row

    @staticmethod
    def _section_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("sectionTitle")
        return label

    def _make_table(self, headers: list[str]) -> QTableWidget:
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setWordWrap(True)
        table.setMinimumHeight(120)
        return table

    def _set_table_rows(
        self,
        table: QTableWidget,
        rows: list[list[str]],
        highlight_column: int | None = None,
    ) -> None:
        table.setRowCount(len(rows))
        for row_idx, row in enumerate(rows):
            for col_idx, value in enumerate(row):
                item = QTableWidgetItem(value)
                if highlight_column is not None and col_idx == highlight_column:
                    if "maligna" in value.lower() or "riesgo" in value.lower():
                        item.setForeground(QColor(RED))
                    elif "no maligna" in value.lower() or "sin riesgo" in value.lower():
                        item.setForeground(QColor(GREEN))
                table.setItem(row_idx, col_idx, item)
        table.resizeRowsToContents()

    @staticmethod
    def _set_preview_label_image(label: QLabel, frame: np.ndarray | None, fallback: str) -> None:
        if frame is None:
            label.setPixmap(QPixmap())
            label.setText(fallback)
            return
        pixmap = QPixmap.fromImage(VideoPhaseController.to_qimage(frame)).scaled(
            label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        label.setText("")
        label.setPixmap(pixmap)

    def _set_phase2_left_mode(self, review_mode: bool, review_text: str | None = None) -> None:
        self.phase2_left_stack.setCurrentIndex(1 if review_mode else 0)
        if review_text:
            self.phase2_review_status.setText(review_text)

    @staticmethod
    def _build_phase3_not_applicable_result(reason: str) -> dict[str, Any]:
        result = dict(DEFAULT_PHASE3)
        result.update(
            {
                "not_applicable": True,
                "status": "NO_APLICA",
                "skip_reason": reason,
            }
        )
        return result

    def _confirm_phase2_continuation(self) -> bool:
        if self.read_only_history_mode:
            return True
        if not self.phase1_result.get("executed"):
            QMessageBox.information(self, "Fase 2", "Completa primero la Fase 1.")
            return False
        if self.phase1_result.get("is_positive") or self.phase2_override_acknowledged:
            return True

        probability = _percent(self.phase1_result.get("probability", 0.0))
        answer = QMessageBox.warning(
            self,
            "Paso a Fase 2 con riesgo bajo",
            (
                "La Fase 1 indica un perfil sin riesgo elevado "
                f"({probability}). Solo deberias continuar a colonoscopia si hay criterio medico para verificarlo.\n\n"
                "Quieres continuar igualmente?"
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.phase2_override_acknowledged = True
            return True
        return False

    def _show_phase2_page(self) -> None:
        if not self.read_only_history_mode and not self._confirm_phase2_continuation():
            return
        self.stack.setCurrentWidget(self.page_phase2)

    def _show_phase3_page(self) -> None:
        if self.read_only_history_mode:
            self.stack.setCurrentWidget(self.page_phase3)
            return
        if not self.phase2_result.get("executed"):
            QMessageBox.information(self, "Fase 3", "Completa primero la Fase 2.")
            return
        if not _phase2_has_confirmed_polyps(self.phase2_result):
            QMessageBox.information(
                self,
                "Fase 3",
                "La Fase 2 no confirmo hallazgos persistentes. La consulta termina en Fase 2 y la histologia no aplica.",
            )
            self.stack.setCurrentWidget(self.page_report)
            return
        self.stack.setCurrentWidget(self.page_phase3)

    def _refresh_all_views(self) -> None:
        self._render_phase1_view()
        self._render_phase2_view()
        self._render_phase3_view()
        self._refresh_home_summary()
        self._update_final_report()
        self._update_navigation()
        self._update_responsive_layout()

    def _screen_available_geometry(self):
        screen = self.screen() or QApplication.primaryScreen()
        return screen.availableGeometry() if screen is not None else None

    def _on_stack_changed(self, _index: int) -> None:
        current = self.stack.currentWidget()
        if current is self.page_phase2:
            QTimer.singleShot(0, self._render_phase2_view)
        elif current is self.page_phase3:
            QTimer.singleShot(0, self._render_phase3_view)
        elif current is self.page_report:
            QTimer.singleShot(0, self._update_final_report)

    def _update_responsive_layout(self) -> None:
        geometry = self._screen_available_geometry()
        if geometry is None:
            return

        available_h = geometry.height()
        available_w = geometry.width()
        current_h = max(self.height(), 720)
        current_w = max(self.width(), 1120)
        basis_h = min(current_h, available_h)
        basis_w = min(current_w, available_w)
        content_w = max(basis_w - 360, 720)

        video_h = max(int(basis_h * 0.42), 300)
        video_h = min(video_h, 520)
        histology_h = max(int(basis_h * 0.34), 260)
        histology_h = min(histology_h, 430)
        summary_h = max(int(basis_h * 0.16), 120)
        summary_h = min(summary_h, 180)
        detail_h = max(int(basis_h * 0.18), 130)
        detail_h = min(detail_h, 220)
        table_h = max(int(basis_h * 0.22), 130)
        table_h = min(table_h, 260)
        review_h = max(int(basis_h * 0.28), 210)
        review_h = min(review_h, 360)

        self.consultation_split.setOrientation(
            Qt.Orientation.Vertical if content_w < 1120 else Qt.Orientation.Horizontal
        )
        self.phase1_split.setOrientation(
            Qt.Orientation.Vertical if content_w < 1180 else Qt.Orientation.Horizontal
        )
        self.phase2_split.setOrientation(
            Qt.Orientation.Vertical if content_w < 1280 else Qt.Orientation.Horizontal
        )
        self.phase2_review_split.setOrientation(
            Qt.Orientation.Vertical if content_w < 980 else Qt.Orientation.Horizontal
        )
        self.phase3_split.setOrientation(
            Qt.Orientation.Vertical if content_w < 1280 else Qt.Orientation.Horizontal
        )
        self.phase3_visual_split.setOrientation(
            Qt.Orientation.Vertical if content_w < 980 else Qt.Orientation.Horizontal
        )

        self.video_label.setMinimumHeight(video_h)
        self.video_label.setMaximumHeight(video_h + 30)
        self.phase3_preview.setMinimumHeight(histology_h)
        self.phase3_preview.setMaximumHeight(histology_h + 30)
        self.phase3_result_preview.setMinimumHeight(histology_h)
        self.phase3_result_preview.setMaximumHeight(histology_h + 30)
        self.phase3_focus_preview.setMinimumHeight(histology_h)
        self.phase3_focus_preview.setMaximumHeight(histology_h + 30)
        self.phase2_original_preview.setMinimumHeight(review_h)
        self.phase2_focus_preview.setMinimumHeight(review_h)
        self.phase2_summary.setMaximumHeight(summary_h)
        self.phase3_summary.setMaximumHeight(summary_h)
        self.phase1_summary.setMaximumHeight(summary_h)
        self.consultation_detail.setMaximumHeight(detail_h)
        self.phase1_patient_table.setMaximumHeight(table_h)
        self.phase1_feature_table.setMaximumHeight(table_h)
        self.phase3_table.setMaximumHeight(max(int(basis_h * 0.30), 180))
        self.phase3_preview.setMinimumWidth(220)
        self.phase3_result_preview.setMinimumWidth(220)
        self.phase3_focus_preview.setMinimumWidth(220)
        self.phase2_original_preview.setMinimumWidth(220)
        self.phase2_focus_preview.setMinimumWidth(220)

        if self.consultation_split.orientation() == Qt.Orientation.Horizontal:
            self.consultation_split.setSizes([380, max(content_w - 380, 540)])
        else:
            self.consultation_split.setSizes([420, 620])
        if self.phase1_split.orientation() == Qt.Orientation.Horizontal:
            self.phase1_split.setSizes([max(int(content_w * 0.42), 360), max(int(content_w * 0.58), 420)])
        else:
            self.phase1_split.setSizes([360, 520])
        if self.phase2_split.orientation() == Qt.Orientation.Horizontal:
            self.phase2_split.setSizes([max(int(content_w * 0.52), 420), max(int(content_w * 0.48), 420)])
        else:
            self.phase2_split.setSizes([420, 620])
        if self.phase3_split.orientation() == Qt.Orientation.Horizontal:
            self.phase3_split.setSizes([max(int(content_w * 0.52), 420), max(int(content_w * 0.48), 420)])
        else:
            self.phase3_split.setSizes([420, 620])

        compact_buttons = basis_w < 1500
        if compact_buttons:
            for btn in (
                self.start_video_btn,
                self.start_webcam_btn,
                self.rewind_btn,
                self.forward_btn,
                self.pause_btn,
                self.stop_btn,
                self.shot_btn,
            ):
                btn.setStyleSheet("padding: 8px 10px;")
            self.pause_btn.setText("Pausar")
            self.start_webcam_btn.setText("Webcam")
        else:
            for btn in (
                self.start_video_btn,
                self.start_webcam_btn,
                self.rewind_btn,
                self.forward_btn,
                self.pause_btn,
                self.stop_btn,
                self.shot_btn,
            ):
                btn.setStyleSheet("")
            self.pause_btn.setText("Pausar / Reanudar")
            self.start_webcam_btn.setText("Usar webcam")

        self._render_phase2_view()
        self._render_phase3_view()

    def _run_phase1(self) -> None:
        if self.read_only_history_mode:
            QMessageBox.information(self, "Fase 1", "Modo historial: esta fase esta en solo lectura.")
            return

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar historial del paciente",
            str(Path.cwd()),
            "Datos (*.csv *.json);;CSV (*.csv);;JSON (*.json)",
        )
        if not path:
            return

        patient_data = core.load_patient_data(path)
        if patient_data is None:
            QMessageBox.critical(self, "Fase 1", "No se pudieron cargar los datos del paciente.")
            return

        is_positive, probability = core.predict_cancer_risk(self.models.history_model, patient_data)
        explanation = core.explain_history_prediction(self.models.history_model, patient_data)
        numeric_fields = sum(1 for v in patient_data.values() if core._is_numeric(v))

        self.phase1_result = {
            "executed": True,
            "status": "RIESGO" if is_positive else "SIN RIESGO",
            "is_positive": is_positive,
            "probability": probability,
            "demo_mode": self.models.history_model is None,
            "numeric_fields": numeric_fields,
            "total_fields": len(patient_data),
            "explanation": explanation,
            "patient_data": patient_data,
            "source_name": Path(path).name,
        }
        self.phase2_override_acknowledged = bool(is_positive)
        self.report_unlocked = True
        self._refresh_all_views()
        self._autosave_snapshot("fase1")
        self._show_status("Fase 1 completada.")

    def _start_phase2_video(self) -> None:
        if self.read_only_history_mode:
            QMessageBox.information(self, "Fase 2", "Modo historial: esta fase esta en solo lectura.")
            return
        if not self._confirm_phase2_continuation():
            return

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar video de colonoscopia",
            str(Path.cwd()),
            "Videos (*.mp4 *.avi *.mkv *.mov *.wmv)",
        )
        if not path:
            return
        if self.video_controller.start(path, "Video"):
            self._set_phase2_left_mode(False)
            self.video_status.setText("Analizando video de colonoscopia en tiempo real.")
            self.rewind_btn.setEnabled(True)
            self.forward_btn.setEnabled(True)
            self._show_status("Fase 2 iniciada con video.")

    def _start_phase2_webcam(self) -> None:
        if self.read_only_history_mode:
            QMessageBox.information(self, "Fase 2", "Modo historial: esta fase esta en solo lectura.")
            return
        if not self._confirm_phase2_continuation():
            return
        if self.video_controller.start(core.WEBCAM_INDEX, "Webcam"):
            self._set_phase2_left_mode(False)
            self.video_status.setText("Analizando colonoscopia desde webcam.")
            self.rewind_btn.setEnabled(False)
            self.forward_btn.setEnabled(False)
            self._show_status("Fase 2 iniciada con webcam.")

    def _save_phase2_screenshot(self) -> None:
        frame = self.video_controller.current_frame()
        if frame is None:
            return
        path = core.save_screenshot(frame, prefix="polyp")
        self._show_status(f"Captura guardada: {path}")

    def _seek_phase2(self, seconds: float) -> None:
        if self.read_only_history_mode:
            return
        moved = self.video_controller.seek_relative_seconds(seconds)
        if moved:
            direction = "adelante" if seconds > 0 else "atras"
            self._show_status(f"Video desplazado {abs(int(seconds))} s hacia {direction}.")

    def _on_video_frame(self, image: QImage) -> None:
        self._set_phase2_left_mode(False)
        pixmap = QPixmap.fromImage(image)
        scaled = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled)

    def _on_video_stats(self, stats: dict[str, Any]) -> None:
        frames = _safe_int(stats.get("frames_processed", 0))
        positive = _safe_int(stats.get("positive_frames", 0))
        unique = _safe_int(stats.get("unique_polyps", 0))
        has_findings = unique > 0
        current_detection_count = _safe_int(stats.get("current_detection_count", 0))
        currently_detecting = bool(stats.get("currently_detecting", current_detection_count > 0))
        peak = _safe_int(stats.get("peak_detections", 0))
        max_conf = _safe_float(stats.get("max_confidence", 0.0))
        seg_positive = _safe_int(stats.get("segmenter_positive_frames", 0))
        seg_total = _safe_int(stats.get("segmenter_total_detections", 0))

        self.phase2_card_frames.set_data(str(frames), "Frames revisados hasta ahora", BLUE)
        self.phase2_card_candidates.set_data("Detectando" if currently_detecting else "No detectando", f"Regiones visibles ahora: {current_detection_count}", ORANGE if currently_detecting else BLUE)
        self.phase2_card_polyps.set_data("Si" if has_findings else "No", f"Se confirma tras {max(4, _safe_int(stats.get('min_confirm_frames', 4), 4))} frames seguidos", RED if has_findings else GREEN)
        self.phase2_card_conf.set_data(_percent(max_conf), f"Pico en un frame: {peak}", RED if max_conf >= 0.7 else YELLOW)

        chart_items = [
            {"label": "Frames revisados", "value": max(frames, 0), "text": str(frames), "color": BLUE},
            {"label": "Frames candidatos", "value": max(positive, 0), "text": str(positive), "color": ORANGE},
            {"label": "Detectando ahora", "value": 1 if currently_detecting else 0, "text": "Si" if currently_detecting else "No", "color": ORANGE if currently_detecting else BLUE},
            {"label": "Hallazgos detectados", "value": 1 if has_findings else 0, "text": "Si" if has_findings else "No", "color": RED if has_findings else GREEN},
            {"label": "Confianza maxima", "value": max_conf, "text": _percent(max_conf), "color": RED},
        ]
        self.phase2_chart.set_items(chart_items)

        candidate_rate = positive / max(frames, 1)
        self.phase2_summary.setPlainText(
            "\n".join(
                [
                    f"Fuente actual: {stats.get('source_mode', self.phase2_result.get('source_mode', 'N/A'))}",
                    f"Frames revisados: {frames}",
                    f"Tasa de frames candidatos: {candidate_rate:.1%}",
                    f"Detectando ahora: {'Si' if currently_detecting else 'No'}",
                    f"Hallazgos persistentes: {'Si' if has_findings else 'No'}",
                    f"Confianza maxima observada: {_percent(max_conf)}",
                    "",
                    "Modelo activo:",
                    f"UNet3+ segmentacion: {positive} frames candidatos, {seg_total} regiones segmentadas, pico {peak}, confianza max {_percent(max_conf)}",
                    "",
                    "Interpretacion rapida:",
                    (
                        "La exploracion presenta hallazgos persistentes y conviene revisar las regiones destacadas."
                        if has_findings
                        else "De momento no hay persistencia suficiente para confirmar polipos estables."
                    ),
                ]
            )
        )

    def _on_video_finished(self, stats: dict[str, Any]) -> None:
        self.phase2_result = stats
        if _phase2_has_confirmed_polyps(stats):
            self.phase3_result = dict(DEFAULT_PHASE3)
        else:
            self.phase3_result = self._build_phase3_not_applicable_result(
                "La colonoscopia no confirmo hallazgos persistentes. La fase histologica se omite y la consulta termina en Fase 2."
            )
        self.video_status.setText("Analisis de colonoscopia finalizado.")
        self.rewind_btn.setEnabled(False)
        self.forward_btn.setEnabled(False)
        self.report_unlocked = True
        self._refresh_all_views()
        self._autosave_snapshot("fase2")
        self._show_status("Fase 2 finalizada.")

    def _run_phase3(self) -> None:
        if self.read_only_history_mode:
            QMessageBox.information(self, "Fase 3", "Modo historial: esta fase esta en solo lectura.")
            return
        if not self.phase2_result.get("executed"):
            QMessageBox.information(self, "Fase 3", "Completa primero la Fase 2.")
            return
        if not _phase2_has_confirmed_polyps(self.phase2_result):
            QMessageBox.information(
                self,
                "Fase 3",
                "La Fase 2 no confirmo hallazgos persistentes. No hay muestra que analizar en histologia.",
            )
            self.phase3_result = self._build_phase3_not_applicable_result(
                "La colonoscopia no confirmo hallazgos persistentes. La fase histologica se omite y la consulta termina en Fase 2."
            )
            self._refresh_all_views()
            self._autosave_snapshot("fase3-no-aplica")
            return

        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Seleccionar imagenes histologicas",
            str(Path.cwd()),
            "Imagenes (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if not paths:
            return

        image_results: list[dict[str, Any]] = []
        for image_path in paths:
            frame = cv2.imread(image_path)
            if frame is None:
                continue

            if self.models.microscopy_model is not None:
                cls_name, confidence = core.run_classification_inference(self.models.microscopy_model, frame)
                is_cancer = cls_name == "colon_aca" and confidence >= core.CONFIDENCE_THRESHOLD
                explanation = core.create_gradcam_explanation(self.models.microscopy_model, frame, cls_name)
            else:
                cls_name, confidence = "DEMO", 0.0
                is_cancer = False
                explanation = None

            display = frame.copy()
            border_color = (0, 0, 255) if is_cancer else (0, 255, 0)
            banner_color = (24, 32, 48)
            cv2.rectangle(display, (6, 6), (display.shape[1] - 6, display.shape[0] - 6), border_color, 4)
            cv2.rectangle(display, (0, 0), (display.shape[1], 62), banner_color, -1)
            result_label = "MALIGNA" if is_cancer else "NO MALIGNA"
            cv2.putText(
                display,
                f"{result_label} · {cls_name}",
                (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                display,
                f"Confianza: {_percent(confidence)}",
                (12, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                border_color,
                2,
                cv2.LINE_AA,
            )

            image_results.append(
                {
                    "image_path": image_path,
                    "image_name": Path(image_path).name,
                    "is_malignant": is_cancer,
                    "confidence": float(confidence),
                    "class_name": cls_name,
                    "preview": display,
                    "explanation": explanation,
                }
            )

        if not image_results:
            QMessageBox.warning(self, "Fase 3", "No se pudo procesar ninguna imagen.")
            return

        malignant_results = [r for r in image_results if r.get("is_malignant")]
        best_result = max(image_results, key=lambda r: float(r.get("confidence", 0.0)))
        best_malignant = max(malignant_results, key=lambda r: float(r.get("confidence", 0.0)), default=None)
        representative = best_malignant or best_result

        self.phase3_result = {
            "executed": True,
            "image_path": representative.get("image_path", ""),
            "image_name": (
                representative.get("image_name", "")
                if len(image_results) == 1
                else f"{len(image_results)} imagenes analizadas"
            ),
            "is_malignant": len(malignant_results) > 0,
            "confidence": float(representative.get("confidence", 0.0)),
            "class_name": representative.get("class_name", "N/A"),
            "demo_mode": self.models.microscopy_model is None,
            "explanation": representative.get("explanation"),
            "images": image_results,
            "total_images": len(image_results),
            "valid_images": len(image_results),
            "malignant_count": len(malignant_results),
            "non_malignant_count": len(image_results) - len(malignant_results),
        }

        self.phase3_selected_index = next(
            (idx for idx, item in enumerate(image_results) if item.get("image_name") == representative.get("image_name")),
            0,
        )

        self.report_unlocked = True
        self._render_phase3_view()
        self._refresh_home_summary()
        self._update_final_report()
        self._update_navigation()
        self._autosave_snapshot("fase3")
        self._show_status("Fase 3 completada.")

    def _refresh_consultation_lists(self) -> None:
        self.pending_list.clear()
        pending_records = _load_pending_records()
        for record in pending_records:
            result = record.get("result", {})
            phase1 = result.get("phase1", {})
            phase2 = result.get("phase2", {})
            phase3 = result.get("phase3", {})
            completed_steps = sum(
                1
                for phase in (phase1, phase2)
                if isinstance(phase, dict) and phase.get("executed")
            )
            if isinstance(phase3, dict) and _phase3_is_complete(phase3):
                completed_steps += 1
            label = (
                f"{record.get('updated_at', record.get('created_at', ''))} | "
                f"{record.get('patient_name', 'Paciente')} | "
                f"Pendiente {completed_steps}/3"
            )
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, record)
            self.pending_list.addItem(item)

        self.consultation_list.clear()
        records = [r for r in core._load_history_records() if r.get("kind") == "consultation"]
        for record in records:
            patient = record.get("patient_name", "Paciente")
            created = record.get("created_at", "")
            phase1 = (record.get("result") or {}).get("phase1", {})
            phase2 = (record.get("result") or {}).get("phase2", {})
            phase3 = (record.get("result") or {}).get("phase3", {})
            label_parts = [
                f"{created} | {patient}",
                f"P1 {_percent(phase1.get('probability', 0.0))}",
                f"P2 {'si' if _phase2_has_confirmed_polyps(phase2) else 'no'} hallazgos",
            ]
            if not _phase3_is_not_applicable(phase3):
                label_parts.append(
                    f"P3 {phase3.get('malignant_count', 0)}/{phase3.get('total_images', 0)} malignas"
                )
            label = " | ".join(label_parts)
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, record)
            self.consultation_list.addItem(item)

    def _show_consultation_page(self) -> None:
        self._refresh_consultation_lists()
        self.stack.setCurrentWidget(self.page_consultation)

    def _on_consultation_selection(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        if current is not None and self.pending_list.currentItem() is not None:
            self.pending_list.blockSignals(True)
            self.pending_list.clearSelection()
            self.pending_list.blockSignals(False)
        if current is None:
            self.consultation_detail.clear()
            self.consultation_title.setText("Selecciona una consulta")
            self.consultation_chart.set_items([])
            self.consultation_card_phase3.show()
            self._update_consultation_focus_buttons(None)
            return

        record = current.data(Qt.ItemDataRole.UserRole) or {}
        result = record.get("result", {})
        phase1 = result.get("phase1", {})
        phase2 = result.get("phase2", {})
        phase3 = result.get("phase3", {})
        self.consultation_title.setText(f"Consulta guardada · {record.get('patient_name', 'Paciente')}")

        self.consultation_card_phase1.set_data(
            phase1.get("status", "N/A"),
            f"Probabilidad {_percent(phase1.get('probability', 0.0))}",
            RED if phase1.get("is_positive") else GREEN,
        )
        self.consultation_card_phase2.set_data(
            _phase2_findings_value(phase2, "No"),
            f"{_safe_int(phase2.get('positive_frames', 0))} frames candidatos",
            RED if _phase2_has_confirmed_polyps(phase2) else GREEN,
        )
        if _phase3_is_not_applicable(phase3):
            self.consultation_card_phase3.hide()
        else:
            self.consultation_card_phase3.show()
            self.consultation_card_phase3.set_data(
                f"{_safe_int(phase3.get('malignant_count', 0))}/{_safe_int(phase3.get('total_images', 0))}",
                f"Clase {phase3.get('class_name', 'N/A')}",
                RED if phase3.get("is_malignant") else GREEN,
            )
        self.consultation_card_created.set_data(
            record.get("created_at", "").replace("T", " "),
            "Consulta completa guardada",
            BLUE,
        )

        chart_items = [
            {"label": "Riesgo clinico", "value": _safe_float(phase1.get("probability", 0.0)), "text": _percent(phase1.get("probability", 0.0)), "color": RED if phase1.get("is_positive") else GREEN},
            {"label": "Hallazgos endoscopicos", "value": 1 if _phase2_has_confirmed_polyps(phase2) else 0, "text": "Si" if _phase2_has_confirmed_polyps(phase2) else "No", "color": ORANGE if _phase2_has_confirmed_polyps(phase2) else GREEN},
            {"label": "Frames candidatos", "value": _safe_int(phase2.get("positive_frames", 0)), "text": str(_safe_int(phase2.get("positive_frames", 0))), "color": BLUE},
        ]
        if not _phase3_is_not_applicable(phase3):
            chart_items.append(
                {"label": "Histologia maligna", "value": _safe_int(phase3.get("malignant_count", 0)), "text": str(_safe_int(phase3.get("malignant_count", 0))), "color": MAUVE}
            )
        self.consultation_chart.set_items(chart_items)

        conclusion, detail, _color = _conclusion_from_results(phase1, phase2, phase3)
        phase_lines = [
            f"- Fase 1: {phase1.get('status', 'N/A')} ({_percent(phase1.get('probability', 0.0))})",
            f"- Fase 2: {_phase2_findings_status(phase2)}",
        ]
        if not _phase3_is_not_applicable(phase3):
            phase_lines.append(
                f"- Fase 3: {_safe_int(phase3.get('malignant_count', 0))}/{_safe_int(phase3.get('total_images', 0))} malignas"
            )
        self.consultation_detail.setPlainText(
            "\n".join(
                [
                    f"Paciente: {record.get('patient_name', '')}",
                    f"Fecha: {record.get('created_at', '')}",
                    "",
                    "Resumen medico disponible:",
                    *phase_lines,
                    "",
                    f"Lectura orientativa: {conclusion}",
                    detail,
                    "",
                    "Pulsa 'Cargar perfil' para desplegar las tres fases con sus paneles completos.",
                ]
            )
        )
        self._update_consultation_focus_buttons(record)

    def _on_pending_selection(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        if current is not None and self.consultation_list.currentItem() is not None:
            self.consultation_list.blockSignals(True)
            self.consultation_list.clearSelection()
            self.consultation_list.blockSignals(False)
        if current is None:
            if self.consultation_list.currentItem() is None:
                self.consultation_detail.clear()
                self.consultation_title.setText("Selecciona una consulta")
                self.consultation_chart.set_items([])
                self.consultation_card_phase3.show()
                self._update_consultation_focus_buttons(None)
            return

        record = current.data(Qt.ItemDataRole.UserRole) or {}
        result = record.get("result", {})
        phase1 = result.get("phase1", {})
        phase2 = result.get("phase2", {})
        phase3 = result.get("phase3", {})
        completed_steps = sum(
            1 for phase in (phase1, phase2) if isinstance(phase, dict) and phase.get("executed")
        )
        if isinstance(phase3, dict) and _phase3_is_complete(phase3):
            completed_steps += 1
        self.consultation_title.setText(f"Consulta pendiente · {record.get('patient_name', 'Paciente')}")
        self.consultation_card_phase1.set_data(
            phase1.get("status", "Pendiente") if phase1.get("executed") else "Pendiente",
            f"Probabilidad {_percent(phase1.get('probability', 0.0))}" if phase1.get("executed") else "Fase 1 sin ejecutar",
            RED if phase1.get("is_positive") else GREEN if phase1.get("executed") else YELLOW,
        )
        self.consultation_card_phase2.set_data(
            _phase2_findings_value(phase2),
            _phase2_findings_status(phase2, "Fase 2 sin ejecutar"),
            RED if _phase2_has_confirmed_polyps(phase2) else GREEN if phase2.get("executed") else YELLOW,
        )
        if _phase3_is_not_applicable(phase3):
            self.consultation_card_phase3.hide()
        else:
            self.consultation_card_phase3.show()
            self.consultation_card_phase3.set_data(
                f"{_safe_int(phase3.get('malignant_count', 0))}/{_safe_int(phase3.get('total_images', 0))}" if phase3.get("executed") else "Pendiente",
                f"Clase {phase3.get('class_name', 'N/A')}" if phase3.get("executed") else "Fase 3 sin ejecutar",
                RED if phase3.get("is_malignant") else GREEN if phase3.get("executed") else YELLOW,
            )
        self.consultation_card_created.set_data(
            record.get("updated_at", "").replace("T", " "),
            f"Consulta pendiente · {completed_steps}/3 fases",
            ORANGE,
        )
        pending_chart_items = [
            {"label": "Fases completadas", "value": completed_steps, "text": f"{completed_steps}/3", "color": ORANGE},
            {"label": "Riesgo clinico", "value": _safe_float(phase1.get("probability", 0.0)), "text": _percent(phase1.get("probability", 0.0)), "color": YELLOW},
            {"label": "Hallazgos endoscopicos", "value": 1 if _phase2_has_confirmed_polyps(phase2) else 0, "text": "Si" if _phase2_has_confirmed_polyps(phase2) else "No", "color": RED if _phase2_has_confirmed_polyps(phase2) else GREEN},
        ]
        if not _phase3_is_not_applicable(phase3):
            pending_chart_items.append(
                {"label": "Histologia maligna", "value": _safe_int(phase3.get("malignant_count", 0)), "text": str(_safe_int(phase3.get("malignant_count", 0))), "color": MAUVE}
            )
        self.consultation_chart.set_items(pending_chart_items)
        pending_phase_lines = [
            f"- Fase 1: {phase1.get('status', 'Pendiente') if phase1.get('executed') else 'Pendiente'}",
            f"- Fase 2: {_phase2_findings_status(phase2)}" if phase2.get("executed") else "- Fase 2: pendiente",
        ]
        if not _phase3_is_not_applicable(phase3):
            pending_phase_lines.append(
                f"- Fase 3: {_safe_int(phase3.get('malignant_count', 0))}/{_safe_int(phase3.get('total_images', 0))} malignas" if phase3.get("executed") else "- Fase 3: pendiente"
            )
        self.consultation_detail.setPlainText(
            "\n".join(
                [
                    f"Paciente: {record.get('patient_name', '')}",
                    f"Creada: {record.get('created_at', '')}",
                    f"Ultima actualizacion: {record.get('updated_at', '')}",
                    "",
                    f"Estado: consulta pendiente ({completed_steps}/3 fases completas)",
                    *pending_phase_lines,
                    "",
                    "Pulsa 'Continuar' para retomar la consulta o 'Eliminar' para borrarla.",
                ]
            )
        )
        self._update_consultation_focus_buttons(record)

    def _create_new_consultation(self) -> None:
        patient, ok = QInputDialog.getText(self, "Nueva consulta", "Nombre del paciente:")
        patient = patient.strip()
        if not ok or not patient:
            return

        self.current_consultation_record = None
        self.current_pending_id = None
        self.active_patient_name = patient
        self.session_ready = True
        self.report_unlocked = False
        self.read_only_history_mode = False
        self.phase1_result = dict(DEFAULT_PHASE1)
        self.phase2_result = dict(DEFAULT_PHASE2)
        self.phase3_result = dict(DEFAULT_PHASE3)
        self.phase2_override_acknowledged = False
        self.phase3_selected_index = 0
        self.video_label.setText("Esperando fuente de video")
        self.video_label.setPixmap(QPixmap())
        self.current_pending_id = _save_pending_consultation(
            self.active_patient_name,
            self.phase1_result,
            self.phase2_result,
            self.phase3_result,
            pending_id=self.current_pending_id,
        )
        self._refresh_all_views()
        self._refresh_consultation_lists()
        self.stack.setCurrentWidget(self.page_phase1)
        self._show_status(f"Consulta nueva creada para {patient} y guardada como pendiente.")

    def _load_selected_pending(self) -> None:
        item = self.pending_list.currentItem()
        if item is None:
            QMessageBox.information(self, "Pendientes", "Selecciona una consulta pendiente.")
            return

        record = item.data(Qt.ItemDataRole.UserRole) or {}
        result = record.get("result", {})
        self.current_consultation_record = record
        self.current_pending_id = record.get("id")
        self.phase1_result = result.get("phase1", dict(DEFAULT_PHASE1))
        self.phase2_result = result.get("phase2", dict(DEFAULT_PHASE2))
        self.phase3_result = result.get("phase3", dict(DEFAULT_PHASE3))
        self.active_patient_name = record.get("patient_name", "")
        self.session_ready = True
        self.report_unlocked = any(
            phase.get("executed")
            for phase in (self.phase1_result, self.phase2_result, self.phase3_result)
            if isinstance(phase, dict)
        )
        self.read_only_history_mode = False
        self.phase2_override_acknowledged = bool(self.phase1_result.get("is_positive"))
        self.phase3_selected_index = 0
        self._refresh_all_views()
        if not self.phase1_result.get("executed"):
            self.stack.setCurrentWidget(self.page_phase1)
        elif not self.phase2_result.get("executed"):
            self.stack.setCurrentWidget(self.page_phase2)
        elif _phase3_is_not_applicable(self.phase3_result):
            self.stack.setCurrentWidget(self.page_report)
        elif not self.phase3_result.get("executed"):
            self.stack.setCurrentWidget(self.page_phase3)
        else:
            self.stack.setCurrentWidget(self.page_home)
        self._show_status(f"Consulta pendiente cargada para {self.active_patient_name}.")

    def _delete_selected_pending(self) -> None:
        item = self.pending_list.currentItem()
        if item is None:
            QMessageBox.information(self, "Pendientes", "Selecciona una consulta pendiente para eliminarla.")
            return
        record = item.data(Qt.ItemDataRole.UserRole) or {}
        patient = record.get("patient_name", "Paciente")
        answer = QMessageBox.question(
            self,
            "Eliminar consulta pendiente",
            f"Se eliminara la consulta pendiente de {patient}. Esta accion no se puede deshacer.",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        record_id = record.get("id")
        if record_id:
            _delete_pending_record(str(record_id))
        if self.current_pending_id == record_id:
            self.current_pending_id = None
        self._refresh_consultation_lists()
        self.consultation_detail.clear()
        self.consultation_title.setText("Selecciona una consulta")
        self.consultation_chart.set_items([])
        self._show_status(f"Consulta pendiente eliminada: {patient}.")

    def _load_selected_consultation(self) -> None:
        item = self.consultation_list.currentItem()
        if item is None:
            QMessageBox.information(self, "Consulta", "Selecciona un perfil del historial.")
            return

        record = item.data(Qt.ItemDataRole.UserRole) or {}
        result = record.get("result", {})
        self.current_consultation_record = record
        self.current_pending_id = None
        self.phase1_result = result.get("phase1", dict(DEFAULT_PHASE1))
        self.phase2_result = result.get("phase2", dict(DEFAULT_PHASE2))
        self.phase3_result = result.get("phase3", dict(DEFAULT_PHASE3))
        self.active_patient_name = record.get("patient_name", "")
        self.session_ready = True
        self.report_unlocked = True
        self.read_only_history_mode = True
        self.phase2_override_acknowledged = True
        self.phase3_selected_index = 0

        self._refresh_all_views()
        self.stack.setCurrentWidget(self.page_home)
        self._show_status("Perfil historico cargado en todas las fases.")

    def _render_phase1_view(self) -> None:
        phase1 = self.phase1_result
        explanation = phase1.get("explanation") or {}
        patient_data = phase1.get("patient_data") or {}

        if not phase1.get("executed"):
            self.phase1_card_status.set_data("Pendiente", "Aun no se ha ejecutado la fase 1", YELLOW)
            self.phase1_card_prob.set_data("--", "Sin probabilidad disponible", BLUE)
            self.phase1_card_fields.set_data("0", "Variables clinicas", GREEN)
            self.phase1_card_top.set_data("--", "Todavia no hay variable dominante", ORANGE)
            self.phase1_chart.set_items([])
            self._set_table_rows(self.phase1_patient_table, [])
            self._set_table_rows(self.phase1_feature_table, [])
            self.phase1_summary.setPlainText(
                "Carga un CSV o JSON clinico para obtener riesgo, variables evaluadas y explicabilidad SHAP."
            )
            self.btn_phase1_explanation.setEnabled(False)
            return

        status = phase1.get("status", "N/A")
        probability = _safe_float(phase1.get("probability", 0.0))
        top_feature = (explanation.get("features") or [{}])[0]
        top_label = top_feature.get("label", "Sin dato")
        top_text = f"{top_feature.get('value', '')} | impacto {float(top_feature.get('impact', 0.0)):+.3f}"

        self.phase1_card_status.set_data(status, "Motor de riesgo activo" if not phase1.get("demo_mode") else "Modo demo", RED if phase1.get("is_positive") else GREEN)
        self.phase1_card_prob.set_data(_percent(probability), "Probabilidad de riesgo estimada", RED if probability >= 0.5 else GREEN)
        self.phase1_card_fields.set_data(str(_safe_int(phase1.get("total_fields", 0))), f"{_safe_int(phase1.get('numeric_fields', 0))} numericos", BLUE)
        self.phase1_card_top.set_data(top_label, top_text, ORANGE)

        patient_rows = [[str(key), str(value)] for key, value in patient_data.items()]
        self._set_table_rows(self.phase1_patient_table, patient_rows)

        feature_rows: list[list[str]] = []
        chart_items: list[dict[str, Any]] = []
        for feature in (explanation.get("features") or [])[:8]:
            impact = float(feature.get("impact", 0.0))
            feature_rows.append([
                str(feature.get("label", "")),
                str(feature.get("value", "")),
                f"{impact:+.3f}",
                str(feature.get("direction", "")),
            ])
            chart_items.append(
                {
                    "label": str(feature.get("label", "")),
                    "value": abs(impact),
                    "text": f"{impact:+.3f}",
                    "color": RED if impact >= 0 else GREEN,
                }
            )
        self._set_table_rows(self.phase1_feature_table, feature_rows)
        self.phase1_chart.set_items(chart_items)

        self.phase1_summary.setPlainText(
            "\n".join(
                [
                    f"Archivo origen: {phase1.get('source_name', 'N/A')}",
                    f"Estado del modelo: {'demo' if phase1.get('demo_mode') else 'real'}",
                    f"Resultado del cribado: {status}",
                    "",
                    "Lectura orientativa:",
                    (
                        "El perfil clinico se clasifica como de riesgo y justifica revisar las fases posteriores."
                        if phase1.get("is_positive")
                        else "El perfil clinico no muestra riesgo elevado en esta inferencia aislada."
                    ),
                ]
            )
        )
        self.btn_phase1_explanation.setEnabled(bool(explanation.get("features")))

    def _render_phase2_view(self) -> None:
        phase2 = self.phase2_result
        if not phase2.get("executed"):
            self._set_phase2_left_mode(False)
            self.phase2_card_frames.set_data("0", "Sin exploracion", BLUE)
            self.phase2_card_candidates.set_data("No detectando", "Sin video activo", BLUE)
            self.phase2_card_polyps.set_data("No", "Sin hallazgos persistentes", GREEN)
            self.phase2_card_conf.set_data("--", "Sin confianza registrada", RED)
            self.phase2_chart.set_items([])
            self.phase2_summary.setPlainText(
                "Abre un video o webcam para revisar frames, confianza, persistencia y hallazgos confirmados."
            )
            self._set_preview_label_image(
                self.phase2_original_preview,
                None,
                "Aun no hay frame representativo guardado",
            )
            self._set_preview_label_image(
                self.phase2_focus_preview,
                None,
                "El mapa de enfoque aparecera aqui al finalizar el analisis",
            )
            return

        frames = _safe_int(phase2.get("frames_processed", 0))
        positive = _safe_int(phase2.get("positive_frames", 0))
        has_findings = _phase2_has_confirmed_polyps(phase2)
        current_detection_count = _safe_int(phase2.get("current_detection_count", 0))
        currently_detecting = bool(phase2.get("currently_detecting", current_detection_count > 0))
        peak = _safe_int(phase2.get("peak_detections", 0))
        avg_conf = _safe_float(phase2.get("avg_confidence", 0.0))
        max_conf = _safe_float(phase2.get("max_confidence", 0.0))
        seg_positive = _safe_int(phase2.get("segmenter_positive_frames", 0))
        seg_total = _safe_int(phase2.get("segmenter_total_detections", 0))
        seg_peak = _safe_int(phase2.get("segmenter_peak_detections", 0))
        seg_max_conf = _safe_float(phase2.get("segmenter_max_confidence", 0.0))
        completion = phase2.get("completion_ratio")
        candidate_rate = positive / max(frames, 1)

        self.phase2_card_frames.set_data(str(frames), f"Fuente: {phase2.get('source_mode', 'N/A')}", BLUE)
        self.phase2_card_candidates.set_data("Detectando" if currently_detecting else "No detectando", f"Estado al cerrar · regiones visibles: {current_detection_count}", ORANGE if currently_detecting else BLUE)
        self.phase2_card_polyps.set_data("Si" if has_findings else "No", f"Persistencia minima {phase2.get('min_confirm_seconds', 0)} s", RED if has_findings else GREEN)
        self.phase2_card_conf.set_data(_percent(max_conf), f"Media {_percent(avg_conf)} | pico {peak}", RED if max_conf >= 0.75 else YELLOW)

        chart_items = [
            {"label": "Cobertura", "value": _safe_float(completion, 0.0), "text": _percent(completion) if completion is not None else "Webcam", "color": BLUE},
            {"label": "Frames candidatos", "value": candidate_rate, "text": f"{candidate_rate:.1%}", "color": ORANGE},
            {"label": "Detectando al cierre", "value": 1 if currently_detecting else 0, "text": "Si" if currently_detecting else "No", "color": ORANGE if currently_detecting else BLUE},
            {"label": "Hallazgos detectados", "value": 1 if has_findings else 0, "text": "Si" if has_findings else "No", "color": RED if has_findings else GREEN},
            {"label": "Confianza media", "value": avg_conf, "text": _percent(avg_conf), "color": CYAN},
            {"label": "Confianza maxima", "value": max_conf, "text": _percent(max_conf), "color": RED},
        ]
        self.phase2_chart.set_items(chart_items)

        summary_lines = [
            f"Fuente de analisis: {phase2.get('source_mode', 'N/A')}",
            f"Frames procesados: {frames}",
            f"Frames con candidatos: {positive} ({candidate_rate:.1%})",
            f"Detectando al cerrar: {'Si' if currently_detecting else 'No'}",
            f"Hallazgos persistentes: {'Si' if has_findings else 'No'}",
            f"Detecciones totales: {_safe_int(phase2.get('total_detections', 0))}",
            f"Confianza media / maxima: {_percent(avg_conf)} / {_percent(max_conf)}",
        ]
        if phase2.get("segmenter_model_loaded") or seg_total > 0:
            summary_lines.extend(
                [
                    "",
                    "Modelo activo UNet3+ segmentacion:",
                    f"Frames candidatos: {seg_positive}",
                    f"Regiones segmentadas: {seg_total}",
                    f"Pico en un frame: {seg_peak}",
                    f"Confianza maxima: {_percent(seg_max_conf)}",
                ]
            )
        if completion is not None:
            summary_lines.append(f"Video revisado: {_percent(completion)}")
        summary_lines.extend(
            [
                "",
                "Lectura orientativa:",
                (
                    "La deteccion mantiene persistencia suficiente para marcar hallazgos que merecen verificacion."
                    if has_findings
                    else "No hay persistencia suficiente para confirmar polipos estables con el umbral actual. La consulta puede cerrarse aqui sin pasar a histologia."
                ),
            ]
        )
        self.phase2_summary.setPlainText("\n".join(summary_lines))
        visuals = self._resolve_phase2_visuals_for(self.phase2_result, self.current_consultation_record)
        original = visuals[0] if visuals else None
        overlay = visuals[1] if visuals else None
        self._set_preview_label_image(
            self.phase2_original_preview,
            original,
            "No hay frame original disponible para esta consulta.",
        )
        self._set_preview_label_image(
            self.phase2_focus_preview,
            overlay,
            "No hay mapa de enfoque disponible para esta consulta.",
        )
        show_review = self.video_controller.cap is None and (self.read_only_history_mode or visuals is not None)
        review_text = (
            "Consulta en modo lectura. Se muestran el frame representativo y el mapa de enfoque guardados."
            if self.read_only_history_mode
            else "Analisis finalizado. Se muestran el mejor frame detectado y el mapa de enfoque asociado."
        )
        self._set_phase2_left_mode(show_review, review_text)

    def _render_phase3_view(self) -> None:
        phase3 = self.phase3_result
        if _phase3_is_not_applicable(phase3):
            reason = str(phase3.get("skip_reason", "La fase histologica no aplica en esta consulta."))
            self.phase3_preview_title.setText("Fase 3 omitida")
            self.phase3_card_result.set_data("No aplicable", "Sin hallazgos persistentes en Fase 2", BLUE)
            self.phase3_card_conf.set_data("--", "No hay inferencia histologica", BLUE)
            self.phase3_card_volume.set_data("0", "Muestras requeridas", BLUE)
            self.phase3_card_balance.set_data("--", "Fase omitida con criterio", BLUE)
            self.phase3_chart.set_items([])
            self._set_table_rows(self.phase3_table, [])
            self._set_preview_label_image(self.phase3_preview, None, "No hay imagen histologica porque la fase no aplica")
            self._set_preview_label_image(self.phase3_result_preview, None, "No hay vista analizada porque la fase no aplica")
            self._set_preview_label_image(self.phase3_focus_preview, None, "No hay Grad-CAM porque la fase no aplica")
            self.phase3_summary.setPlainText(reason)
            return

        if not phase3.get("executed"):
            self.phase3_preview_title.setText("Comparacion de la muestra destacada")
            self.phase3_card_result.set_data("Pendiente", "Sin clasificacion", MAUVE)
            self.phase3_card_conf.set_data("--", "Sin confianza", RED)
            self.phase3_card_volume.set_data("0", "Muestras", BLUE)
            self.phase3_card_balance.set_data("0 / 0", "Balance malignidad", GREEN)
            self.phase3_chart.set_items([])
            self._set_table_rows(self.phase3_table, [])
            self._set_preview_label_image(self.phase3_preview, None, "Sin muestra seleccionada")
            self._set_preview_label_image(
                self.phase3_result_preview,
                None,
                "La imagen analizada aparecera aqui al procesar una muestra",
            )
            self._set_preview_label_image(
                self.phase3_focus_preview,
                None,
                "El Grad-CAM aparecera aqui al procesar una muestra",
            )
            self.phase3_summary.setPlainText(
                "Selecciona una o varias imagenes histologicas para obtener clasificacion, confianza y Grad-CAM."
            )
            return

        total = _safe_int(phase3.get("total_images", 0))
        malignant = _safe_int(phase3.get("malignant_count", 0))
        non_malignant = _safe_int(phase3.get("non_malignant_count", 0))
        confidence = _safe_float(phase3.get("confidence", 0.0))
        class_name = phase3.get("class_name", "N/A")

        self.phase3_card_result.set_data(
            "MALIGNA" if phase3.get("is_malignant") else "NO MALIGNA",
            f"Clase destacada: {class_name}",
            RED if phase3.get("is_malignant") else GREEN,
        )
        self.phase3_card_conf.set_data(_percent(confidence), "Confianza destacada", RED if confidence >= 0.8 else YELLOW)
        self.phase3_card_volume.set_data(str(total), "Imagenes histologicas analizadas", BLUE)
        self.phase3_card_balance.set_data(f"{malignant} / {non_malignant}", "Malignas / no malignas", RED if malignant > 0 else GREEN)

        rows: list[list[str]] = []
        chart_items: list[dict[str, Any]] = []
        for idx, image_result in enumerate(phase3.get("images", []), start=1):
            result_text = "Maligna" if image_result.get("is_malignant") else "No maligna"
            rows.append(
                [
                    str(image_result.get("image_name", f"Imagen {idx}")),
                    str(image_result.get("class_name", "")),
                    _percent(image_result.get("confidence", 0.0)),
                    result_text,
                ]
            )
            chart_items.append(
                {
                    "label": f"{idx:02d}. {str(image_result.get('image_name', ''))[:18]}",
                    "value": _safe_float(image_result.get("confidence", 0.0)),
                    "text": _percent(image_result.get("confidence", 0.0)),
                    "color": RED if image_result.get("is_malignant") else GREEN,
                }
            )

        self._set_table_rows(self.phase3_table, rows, highlight_column=3)
        if rows:
            index = min(max(self.phase3_selected_index, 0), len(rows) - 1)
            self.phase3_table.selectRow(index)
            self.phase3_selected_index = index
        self.phase3_chart.set_items(chart_items)
        self._render_phase3_preview()

        self.phase3_summary.setPlainText(
            "\n".join(
                [
                    f"Imagenes analizadas: {total}",
                    f"Muestras malignas: {malignant}",
                    f"Muestras no malignas: {non_malignant}",
                    f"Muestra principal: {phase3.get('image_name', 'N/A')}",
                    f"Clase principal: {class_name}",
                    "Vistas mostradas: original, imagen analizada y Grad-CAM",
                    "",
                    "Lectura orientativa:",
                    (
                        "Hay al menos una imagen con patron maligno destacado. Conviene revision histopatologica prioritaria."
                        if phase3.get("is_malignant")
                        else "No se observa patron maligno predominante en las imagenes analizadas."
                    ),
                ]
            )
        )

    def _render_phase3_preview(self) -> None:
        images = self.phase3_result.get("images", [])
        if not images:
            return
        idx = min(max(self.phase3_selected_index, 0), len(images) - 1)
        selected = images[idx]
        self.phase3_preview_title.setText(f"Muestra destacada · {selected.get('image_name', '')}")
        analyzed = selected.get("preview")
        if analyzed is None:
            path = _resolve_history_path(selected.get("image_path"))
            if path is not None:
                analyzed = cv2.imread(str(path))
        if analyzed is None and self.current_consultation_record is not None:
            key = f"histologia_{idx + 1:02d}_resultado"
            path = _resolve_history_path((self.current_consultation_record.get("images") or {}).get(key))
            if path is not None:
                analyzed = cv2.imread(str(path))
        visuals = self._resolve_phase3_visuals_for(self.phase3_result, self.current_consultation_record, idx)
        original = visuals[0] if visuals else analyzed
        overlay = visuals[1] if visuals else None
        self._set_preview_label_image(
            self.phase3_preview,
            original if original is not None else analyzed,
            "No hay imagen original disponible",
        )
        self._set_preview_label_image(
            self.phase3_result_preview,
            analyzed,
            "No hay imagen analizada disponible",
        )
        self._set_preview_label_image(self.phase3_focus_preview, overlay, "No hay Grad-CAM disponible")

    def _on_phase3_table_selection(self) -> None:
        row = self.phase3_table.currentRow()
        if row < 0:
            return
        self.phase3_selected_index = row
        self._render_phase3_preview()

    def _selected_browser_record(self) -> dict[str, Any] | None:
        if self.pending_list.currentItem() is not None:
            return self.pending_list.currentItem().data(Qt.ItemDataRole.UserRole) or {}
        if self.consultation_list.currentItem() is not None:
            return self.consultation_list.currentItem().data(Qt.ItemDataRole.UserRole) or {}
        return None

    def _update_consultation_focus_buttons(self, record: dict[str, Any] | None) -> None:
        result = (record or {}).get("result", {})
        phase1 = result.get("phase1", {}) if isinstance(result, dict) else {}
        phase2 = result.get("phase2", {}) if isinstance(result, dict) else {}
        phase3 = result.get("phase3", {}) if isinstance(result, dict) else {}
        self.btn_consultation_phase1_focus.setEnabled(bool((phase1.get("explanation") or {}).get("features")))
        self.btn_consultation_phase2_focus.setEnabled(self._resolve_phase2_visuals_for(phase2, record) is not None)
        self.btn_consultation_phase3_focus.setEnabled(self._resolve_phase3_visuals_for(phase3, record, 0) is not None)

    def _open_selected_record_phase1_explanation(self) -> None:
        record = self._selected_browser_record()
        if not record:
            QMessageBox.information(self, "Explicacion", "Selecciona una consulta o pendiente primero.")
            return
        self._show_phase1_explanation_for((record.get("result") or {}).get("phase1", {}))

    def _open_selected_record_phase2_explanation(self) -> None:
        record = self._selected_browser_record()
        if not record:
            QMessageBox.information(self, "Explicacion", "Selecciona una consulta o pendiente primero.")
            return
        self._show_phase2_explanation_for((record.get("result") or {}).get("phase2", {}), record)

    def _open_selected_record_phase3_explanation(self) -> None:
        record = self._selected_browser_record()
        if not record:
            QMessageBox.information(self, "Explicacion", "Selecciona una consulta o pendiente primero.")
            return
        self._show_phase3_explanation_for((record.get("result") or {}).get("phase3", {}), record, 0)

    def _open_phase1_explanation(self) -> None:
        self._show_phase1_explanation_for(self.phase1_result)

    def _show_phase1_explanation_for(self, phase1_result: dict[str, Any]) -> None:
        explanation = phase1_result.get("explanation") or {}
        if not explanation.get("features"):
            QMessageBox.information(self, "Fase 1", "No hay explicacion SHAP disponible.")
            return
        dialog = QMainWindow(self)
        dialog.setWindowTitle("Fase 1 - Variables con mayor impacto")
        dialog.resize(980, 700)

        root = QWidget()
        dialog.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        title = QLabel("En que se ha fijado el modelo")
        title.setStyleSheet("font-size: 22px; font-weight: 700; color: #f3f8ff;")
        layout.addWidget(title)

        subtitle = QLabel(str(explanation.get("message", "")))
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet(f"color: {TEXT_SUB};")
        layout.addWidget(subtitle)

        chart = BarChartWidget("Ranking de impacto SHAP")
        chart.set_items(
            [
                {
                    "label": str(feature.get("label", "")),
                    "value": abs(float(feature.get("impact", 0.0))),
                    "text": f"{float(feature.get('impact', 0.0)):+.3f}",
                    "color": RED if float(feature.get("impact", 0.0)) >= 0 else GREEN,
                }
                for feature in (explanation.get("features") or [])[:10]
            ]
        )
        layout.addWidget(chart, 1)

        table = self._make_table(["Variable", "Valor", "Impacto", "Direccion"])
        self._set_table_rows(
            table,
            [
                [
                    str(feature.get("label", "")),
                    str(feature.get("value", "")),
                    f"{float(feature.get('impact', 0.0)):+.3f}",
                    str(feature.get("direction", "")),
                ]
                for feature in (explanation.get("features") or [])[:12]
            ],
        )
        layout.addWidget(table, 1)

        dialog.setStyleSheet(self.styleSheet())
        dialog.show()

    def _resolve_phase2_visuals_for(
        self,
        phase2_result: dict[str, Any] | None,
        record: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None] | None:
        phase2_result = phase2_result or {}
        explanation = phase2_result.get("explanation") or {}
        if explanation.get("original") is not None or explanation.get("overlay") is not None:
            return explanation.get("original"), explanation.get("overlay")

        source_record = record or self.current_consultation_record
        if source_record is not None:
            images = source_record.get("images") or {}
            original_path = _resolve_history_path(images.get("colonoscopia_original"))
            overlay_path = _resolve_history_path(images.get("colonoscopia_enfoque"))
            if original_path or overlay_path:
                original = cv2.imread(str(original_path)) if original_path else None
                overlay = cv2.imread(str(overlay_path)) if overlay_path else None
                return original, overlay
        return None

    def _resolve_phase3_visuals_for(
        self,
        phase3_result: dict[str, Any] | None,
        record: dict[str, Any] | None = None,
        selected_index: int | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None] | None:
        phase3_result = phase3_result or {}
        index = self.phase3_selected_index if selected_index is None else selected_index
        images = phase3_result.get("images", [])
        if images:
            idx = min(max(index, 0), len(images) - 1)
            item = images[idx]
            explanation = item.get("explanation") or {}
            original = None
            overlay = None
            image_path = _resolve_history_path(item.get("image_path"))
            if image_path is not None:
                original = cv2.imread(str(image_path))
            if explanation.get("original") is not None:
                original = explanation.get("original")
            if explanation.get("overlay") is not None:
                overlay = explanation.get("overlay")
            if overlay is None and original is not None and self.models.microscopy_model is not None:
                fallback = core.create_gradcam_explanation(
                    self.models.microscopy_model,
                    original,
                    str(item.get("class_name") or ""),
                )
                if fallback is not None:
                    original = fallback.get("original", original)
                    overlay = fallback.get("overlay")
            if original is not None or overlay is not None:
                return original, overlay

        source_record = record or self.current_consultation_record
        if source_record is not None:
            idx = index + 1
            images_map = source_record.get("images") or {}
            original_path = _resolve_history_path(images_map.get(f"histologia_{idx:02d}_original") or images_map.get("histologia_original"))
            overlay_path = _resolve_history_path(images_map.get(f"histologia_{idx:02d}_enfoque") or images_map.get("histologia_enfoque"))
            if original_path or overlay_path:
                original = cv2.imread(str(original_path)) if original_path else None
                overlay = cv2.imread(str(overlay_path)) if overlay_path else None
                if overlay is None and original is not None and self.models.microscopy_model is not None:
                    image_items = phase3_result.get("images") or []
                    if image_items:
                        safe_idx = min(max(index, 0), len(image_items) - 1)
                        class_name = str((image_items[safe_idx] or {}).get("class_name", ""))
                    else:
                        class_name = str(phase3_result.get("class_name", ""))
                    fallback = core.create_gradcam_explanation(
                        self.models.microscopy_model,
                        original,
                        class_name,
                    )
                    if fallback is not None:
                        original = fallback.get("original", original)
                        overlay = fallback.get("overlay")
                return original, overlay
        return None

    def _open_phase2_explanation(self) -> None:
        self._show_phase2_explanation_for(self.phase2_result, self.current_consultation_record)

    def _show_phase2_explanation_for(
        self,
        phase2_result: dict[str, Any],
        record: dict[str, Any] | None = None,
    ) -> None:
        visuals = self._resolve_phase2_visuals_for(phase2_result, record)
        if visuals is None:
            QMessageBox.information(self, "Fase 2", "No hay mapa de enfoque disponible para esta fase.")
            return
        original, overlay = visuals
        viewer = ImageComparisonDialog(
            "Fase 2 - Enfoque del modelo",
            original=original,
            overlay=overlay,
            subtitle="Comparacion entre el frame original y el mapa de atencion ponderado por detecciones.",
        )
        viewer.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        viewer.show()

    def _open_phase3_explanation(self) -> None:
        self._show_phase3_explanation_for(self.phase3_result, self.current_consultation_record, self.phase3_selected_index)

    def _show_phase3_explanation_for(
        self,
        phase3_result: dict[str, Any],
        record: dict[str, Any] | None = None,
        selected_index: int = 0,
    ) -> None:
        visuals = self._resolve_phase3_visuals_for(phase3_result, record, selected_index)
        if visuals is None:
            QMessageBox.information(self, "Fase 3", "No hay Grad-CAM disponible para la muestra seleccionada.")
            return
        original, overlay = visuals
        viewer = ImageComparisonDialog(
            "Fase 3 - Enfoque del modelo",
            original=original,
            overlay=overlay,
            subtitle="Comparacion entre la imagen histologica original y el Grad-CAM de la muestra seleccionada.",
        )
        viewer.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        viewer.show()

    def _refresh_home_summary(self) -> None:
        patient_text = (
            f"Consulta activa de {self.active_patient_name}."
            if self.active_patient_name
            else "No hay consulta activa. Crea una nueva o carga un historial."
        )
        if self.active_patient_name and not self.read_only_history_mode and not self._consultation_is_complete():
            patient_text += " Estado actual: consulta pendiente."
        self.home_patient_label.setText(patient_text)

        phase1 = self.phase1_result
        phase2 = self.phase2_result
        phase3 = self.phase3_result
        conclusion, detail, color = _conclusion_from_results(phase1, phase2, phase3)
        self.home_conclusion.setText(f"{conclusion}\n{detail}")
        self.home_conclusion.setStyleSheet(
            f"background: rgba(0,0,0,0.12); border: 1px solid {color}; border-radius: 16px; padding: 14px; color: {TEXT_MAIN}; font-weight: 600;"
        )

        if not self.session_ready:
            self.home_card_phase1.set_data("--", "Sin consulta iniciada", YELLOW)
            self.home_card_phase2.set_data("--", "Sin consulta iniciada", GREEN)
            self.home_card_phase3.set_data("--", "Sin consulta iniciada", MAUVE)
            self.home_card_flow.set_data("Iniciar", "Crea nueva consulta o carga un historial", BLUE)
        else:
            self.home_card_phase1.set_data(
                _percent(phase1.get("probability", 0.0)) if phase1.get("executed") else "Pendiente",
                phase1.get("status", "Fase 1 pendiente"),
                RED if phase1.get("is_positive") else GREEN if phase1.get("executed") else YELLOW,
            )
            self.home_card_phase2.set_data(
                _phase2_findings_value(phase2),
                _phase2_findings_status(phase2),
                RED if _phase2_has_confirmed_polyps(phase2) else GREEN if phase2.get("executed") else YELLOW,
            )
            if _phase3_is_not_applicable(phase3):
                self.home_card_phase3.set_data("No aplica", "Sin hallazgos persistentes en Fase 2", BLUE)
            else:
                self.home_card_phase3.set_data(
                    f"{_safe_int(phase3.get('malignant_count', 0))}/{_safe_int(phase3.get('total_images', 0))}" if phase3.get("executed") else "Pendiente",
                    "Muestras malignas / total" if phase3.get("executed") else "Fase 3 pendiente",
                    RED if phase3.get("is_malignant") else GREEN if phase3.get("executed") else YELLOW,
                )
            self.home_card_flow.set_data(*self._next_step_summary())

        self._set_phase_row(self.flow_phase1, phase1.get("executed"), phase1.get("status", "Pendiente"), RED if phase1.get("is_positive") else GREEN)
        self._set_phase_row(self.flow_phase2, phase2.get("executed"), _phase2_findings_status(phase2), RED if _phase2_has_confirmed_polyps(phase2) else GREEN)
        if _phase3_is_not_applicable(phase3):
            self._set_phase_row(self.flow_phase3, True, "No aplicable", BLUE, "Omitida")
        else:
            self._set_phase_row(self.flow_phase3, phase3.get("executed"), f"{_safe_int(phase3.get('malignant_count', 0))}/{_safe_int(phase3.get('total_images', 0))} malignas", RED if phase3.get("is_malignant") else GREEN)

        chart_items = [
            {"label": "Riesgo clinico", "value": _safe_float(phase1.get("probability", 0.0)), "text": _percent(phase1.get("probability", 0.0)), "color": YELLOW},
            {"label": "Hallazgos endoscopicos", "value": 1 if _phase2_has_confirmed_polyps(phase2) else 0, "text": "Si" if _phase2_has_confirmed_polyps(phase2) else "No", "color": RED if _phase2_has_confirmed_polyps(phase2) else GREEN},
            {"label": "Histologia maligna", "value": _safe_int(phase3.get("malignant_count", 0)), "text": str(_safe_int(phase3.get("malignant_count", 0))), "color": MAUVE},
        ]
        self.home_chart.set_items(chart_items)

        self.home_recommendation.set_text(detail)
        self.home_checklist.set_text(
            "\n".join(
                [
                    "1. Verifica que el nombre del paciente corresponde a la consulta.",
                    "2. Contrasta la probabilidad clinica con las variables SHAP principales.",
                    "3. Revisa la persistencia de hallazgos en colonoscopia.",
                    "4. Confirma la muestra histologica mas representativa y su Grad-CAM.",
                ]
            )
        )

        banner_lines = ["Sin consulta activa"] if not self.session_ready else [f"Paciente activo: {self.active_patient_name}"]
        if self.session_ready and not self.read_only_history_mode:
            banner_lines.append("Consulta pendiente" if not self._consultation_is_complete() else "Consulta completa")
        else:
            banner_lines.append("Modo lectura" if self.read_only_history_mode else "Modo consulta")
        self.patient_banner.setText("\n".join(banner_lines))

    def _next_step_summary(self) -> tuple[str, str, str]:
        if not self.phase1_result.get("executed"):
            return "Fase 1", "Empieza cargando el historial clinico", BLUE
        if not self.phase2_result.get("executed"):
            if self.phase1_result.get("is_positive"):
                return "Fase 2", "Continua con colonoscopia para verificar el riesgo detectado", GREEN
            return "Fase 2", "Puedes continuar con aviso clinico si aun necesitas verificar por colonoscopia", ORANGE
        if _phase3_is_not_applicable(self.phase3_result):
            return "Informe", "La consulta termina en Fase 2 porque no se confirmaron hallazgos persistentes", BLUE
        if not self.phase3_result.get("executed"):
            return "Fase 3", "Completa la histologia para cerrar el caso", MAUVE
        return "Informe", "La consulta tiene datos para un informe completo", BLUE

    def _consultation_is_complete(self) -> bool:
        return bool(
            self.phase1_result.get("executed")
            and self.phase2_result.get("executed")
            and _phase3_is_complete(self.phase3_result)
        )

    def _set_phase_row(
        self,
        row: QWidget,
        executed: bool,
        detail: str,
        color: str,
        badge_text: str = "Completada",
    ) -> None:
        progress = row.progress  # type: ignore[attr-defined]
        badge = row.badge  # type: ignore[attr-defined]
        progress.setValue(100 if executed else 18)
        badge.setText(badge_text if executed else "Pendiente")
        badge.setStyleSheet(
            f"background: rgba(0,0,0,0.10); border: 1px solid {color if executed else BORDER}; color: {color if executed else TEXT_SUB}; border-radius: 10px; padding: 6px 10px; font-weight: 700;"
        )
        row.setToolTip(detail)

    def _update_final_report(self) -> None:
        p1 = self.phase1_result
        p2 = self.phase2_result
        p3 = self.phase3_result
        patient = self.active_patient_name or "Paciente no definido"
        conclusion, detail, color = _conclusion_from_results(p1, p2, p3)

        self.report_banner.setText(f"{conclusion}\n{detail}")
        self.report_banner.setStyleSheet(
            f"background: rgba(0,0,0,0.12); border: 1px solid {color}; border-radius: 16px; padding: 14px; color: {TEXT_MAIN}; font-size: 15px; font-weight: 700;"
        )

        self.report_card_patient.set_data(patient, "Consulta activa" if self.session_ready else "Sin consulta", BLUE)
        self.report_card_phase1.set_data(
            p1.get("status", "Pendiente") if p1.get("executed") else "Pendiente",
            f"Probabilidad {_percent(p1.get('probability', 0.0))}" if p1.get("executed") else "Fase 1 no ejecutada",
            RED if p1.get("is_positive") else GREEN if p1.get("executed") else YELLOW,
        )
        self.report_card_phase2.set_data(
            _phase2_findings_value(p2),
            _phase2_findings_status(p2, "Fase 2 no ejecutada"),
            RED if _phase2_has_confirmed_polyps(p2) else GREEN if p2.get("executed") else YELLOW,
        )
        if _phase3_is_not_applicable(p3):
            self.report_card_phase3.set_data("No aplica", "Sin hallazgos persistentes en Fase 2", BLUE)
        else:
            self.report_card_phase3.set_data(
                "MALIGNA" if p3.get("is_malignant") else "NO MALIGNA" if p3.get("executed") else "Pendiente",
                f"{_safe_int(p3.get('malignant_count', 0))}/{_safe_int(p3.get('total_images', 0))} muestras malignas" if p3.get("executed") else "Fase 3 no ejecutada",
                RED if p3.get("is_malignant") else GREEN if p3.get("executed") else YELLOW,
            )

        self.report_chart.set_items(
            [
                {"label": "Riesgo clinico", "value": _safe_float(p1.get("probability", 0.0)), "text": _percent(p1.get("probability", 0.0)), "color": YELLOW},
                {"label": "Confianza endoscopica", "value": _safe_float(p2.get("max_confidence", 0.0)), "text": _percent(p2.get("max_confidence", 0.0)), "color": GREEN},
                {"label": "Confianza histologica", "value": _safe_float(p3.get("confidence", 0.0)), "text": _percent(p3.get("confidence", 0.0)), "color": MAUVE},
            ]
        )
        matrix_rows = [
            ["Fase 1", p1.get("status", "Pendiente"), f"Probabilidad {_percent(p1.get('probability', 0.0))}"],
            ["Fase 2", _phase2_findings_status(p2), f"{_safe_int(p2.get('positive_frames', 0))} frames candidatos"],
            ["Conclusion", conclusion, "Lectura orientativa del sistema"],
        ]
        if _phase3_is_not_applicable(p3):
            matrix_rows.insert(2, ["Fase 3", "No aplicable", str(p3.get("skip_reason", ""))])
        else:
            matrix_rows.insert(2, ["Fase 3", p3.get("class_name", "Pendiente"), f"{_safe_int(p3.get('malignant_count', 0))}/{_safe_int(p3.get('total_images', 0))} malignas"])
        self._set_table_rows(self.report_matrix, matrix_rows)

        explanation = p1.get("explanation") or {}
        feature_lines = []
        for feature in (explanation.get("features") or [])[:5]:
            feature_lines.append(
                f"- {feature.get('label', 'N/A')}: {feature.get('value', '')} | impacto {float(feature.get('impact', 0.0)):+.3f}"
            )
        if not feature_lines:
            feature_lines.append("- Sin datos SHAP disponibles.")

        report_lines = [
            "INFORME DE CONSULTA - AICOLONDIAGNOSIS",
            "=" * 48,
            f"Paciente: {patient}",
            "",
            "Resumen de fases:",
            f"- Fase 1: {_status_text(p1)} | {p1.get('status', 'N/A')} | {_percent(p1.get('probability', 0.0))}",
            f"- Fase 2: {_status_text(p2)} | {_phase2_findings_status(p2)} | {_percent(p2.get('max_confidence', 0.0))} max.",
            "",
            "Variables clinicas con mayor peso:",
            *feature_lines,
            "",
            f"Conclusion orientativa: {conclusion}",
            detail,
        ]
        if _phase3_is_not_applicable(p3):
            report_lines.insert(8, f"- Fase 3: No aplicable | {p3.get('skip_reason', '')}")
        else:
            report_lines.insert(
                8,
                f"- Fase 3: {_status_text(p3)} | {_safe_int(p3.get('malignant_count', 0))}/{_safe_int(p3.get('total_images', 0))} malignas | {p3.get('class_name', 'N/A')}",
            )
        self.final_report.setPlainText("\n".join(report_lines))

    def _update_navigation(self) -> None:
        initial_mode = not self.session_ready
        self.btn_load_history.setVisible(initial_mode)
        self.btn_new_consultation.setVisible(initial_mode)
        self.btn_home.setVisible(not initial_mode)
        self.btn_phase1.setVisible(not initial_mode)
        self.btn_phase2.setVisible(not initial_mode)
        self.btn_phase3.setVisible(not initial_mode)
        self.btn_report.setVisible(not initial_mode)
        self.btn_back.setVisible(not initial_mode)
        self.btn_phase2_explanation.setVisible(False)
        self.btn_phase3_explanation.setVisible(False)
        self.btn_consultation_phase1_focus.setVisible(False)
        self.btn_consultation_phase2_focus.setVisible(False)
        self.btn_consultation_phase3_focus.setVisible(False)
        self.btn_report_phase1_focus.setVisible(False)
        self.btn_report_phase2_focus.setVisible(False)
        self.btn_report_phase3_focus.setVisible(False)

        if initial_mode:
            return

        phase1_ready = bool(self.phase1_result.get("executed"))
        phase2_ready = bool(self.phase2_result.get("executed"))
        phase3_allowed = _phase2_has_confirmed_polyps(self.phase2_result)
        phase3_page_available = (self.read_only_history_mode and not _phase3_is_not_applicable(self.phase3_result)) or phase3_allowed
        report_ready = phase1_ready or self.read_only_history_mode

        self.btn_phase1.setEnabled(True)
        self.btn_phase2.setEnabled(self.read_only_history_mode or phase1_ready)
        self.btn_phase3.setEnabled(phase3_page_available)
        self.btn_report.setEnabled(report_ready)

        read_only = self.read_only_history_mode
        self.btn_run_phase1.setVisible(not read_only)
        self.start_video_btn.setVisible(not read_only)
        self.start_webcam_btn.setVisible(not read_only)
        self.rewind_btn.setVisible(not read_only)
        self.forward_btn.setVisible(not read_only)
        self.pause_btn.setVisible(not read_only)
        self.stop_btn.setVisible(not read_only)
        self.shot_btn.setVisible(not read_only)
        self.btn_run_phase3.setVisible(not read_only)
        self.rewind_btn.setEnabled(self.video_controller.is_seekable())
        self.forward_btn.setEnabled(self.video_controller.is_seekable())

    def _back_to_main_options(self) -> None:
        self.session_ready = False
        self.report_unlocked = False
        self.current_consultation_record = None
        self.current_pending_id = None
        self.active_patient_name = ""
        self.read_only_history_mode = False
        self.phase1_result = dict(DEFAULT_PHASE1)
        self.phase2_result = dict(DEFAULT_PHASE2)
        self.phase3_result = dict(DEFAULT_PHASE3)
        self.phase2_override_acknowledged = False
        self.phase3_selected_index = 0
        self.video_label.setPixmap(QPixmap())
        self.video_label.setText("Esperando fuente de video")
        self._refresh_all_views()
        self.stack.setCurrentWidget(self.page_home)
        self._show_status("Has vuelto al inicio.")

    def _autosave_snapshot(self, phase_name: str) -> None:
        if self.read_only_history_mode:
            return
        if not self.session_ready or not self.active_patient_name:
            return
        if not self.phase1_result.get("executed"):
            return

        if self._consultation_is_complete():
            core.save_patient_consultation_history(
                self.active_patient_name,
                self.phase1_result,
                self.phase2_result,
                self.phase3_result,
            )
            if self.current_pending_id:
                _delete_pending_record(self.current_pending_id)
                self.current_pending_id = None
            self._refresh_consultation_lists()
            self._show_status(f"Consulta completada y movida al historial tras {phase_name}.")
            return

        self.current_pending_id = _save_pending_consultation(
            self.active_patient_name,
            self.phase1_result,
            self.phase2_result,
            self.phase3_result,
            pending_id=self.current_pending_id,
        )
        self._refresh_consultation_lists()
        self._show_status(f"Consulta pendiente actualizada tras {phase_name}.")

    def _show_status(self, text: str) -> None:
        self.statusBar().showMessage(text, 5000)

    @staticmethod
    def _nav_button(text: str) -> QPushButton:
        btn = QPushButton(text)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        return btn

    def _apply_style(self) -> None:
        self.setStyleSheet(
            f"""
            QMainWindow, QWidget#appRoot {{
                background: {APP_BG};
                color: {TEXT_MAIN};
                font-family: 'Segoe UI';
                font-size: 14px;
            }}
            QLabel {{
                background: transparent;
            }}
            QFrame#nav {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {SURFACE_BG}, stop:0.55 #0f1b2c, stop:1 #09121d);
                border-right: 1px solid #1d3149;
                min-width: 300px;
                max-width: 300px;
            }}
            QLabel#navTitle {{
                font-size: 26px;
                font-weight: 800;
                color: #f4f8ff;
            }}
            QLabel#pageTitle {{
                font-size: 30px;
                font-weight: 800;
                color: #f6fbff;
            }}
            QLabel#sectionTitle {{
                font-size: 16px;
                font-weight: 700;
                color: #f2f7ff;
            }}
            QLabel#subtle {{
                color: {TEXT_SUB};
                font-size: 13px;
            }}
            QLabel#bodyText {{
                color: {TEXT_MAIN};
                font-size: 15px;
                line-height: 1.5;
            }}
            QLabel#patientBanner {{
                background: rgba(255,255,255,0.03);
                border: 1px solid #23374f;
                border-radius: 14px;
                padding: 12px;
                color: {TEXT_SUB};
                font-size: 13px;
                font-weight: 600;
            }}
            QLabel#alertNeutral {{
                background: rgba(255,255,255,0.03);
                border: 1px solid {BORDER};
                border-radius: 14px;
                padding: 12px 14px;
                color: {TEXT_MAIN};
            }}
            QFrame#heroCard {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #13263b, stop:1 #0f1d2e);
                border: 1px solid #254360;
                border-radius: 22px;
            }}
            QFrame#sectionCard {{
                background: {CARD_BG};
                border: 1px solid {BORDER};
                border-radius: 18px;
            }}
            QFrame#phaseRow {{
                background: rgba(255,255,255,0.02);
                border: 1px solid #24384f;
                border-radius: 14px;
            }}
            QLabel#phaseRowTitle {{
                font-size: 14px;
                font-weight: 700;
                color: {TEXT_MAIN};
            }}
            QLabel#metricTitle {{
                color: {TEXT_SUB};
                font-size: 12px;
                font-weight: 700;
                text-transform: uppercase;
            }}
            QLabel#metricValue {{
                color: #f8fbff;
                font-size: 26px;
                font-weight: 800;
            }}
            QLabel#metricSubtitle {{
                color: {TEXT_SUB};
                font-size: 12px;
                line-height: 1.4;
            }}
            QTextEdit, QListWidget, QTableWidget, QLineEdit {{
                background: rgba(255,255,255,0.02);
                border: 1px solid #28405e;
                border-radius: 14px;
                color: {TEXT_MAIN};
                padding: 10px;
                selection-background-color: #2a567a;
            }}
            QHeaderView::section {{
                background: #102030;
                color: {TEXT_SUB};
                border: 0;
                padding: 8px;
                font-weight: 700;
            }}
            QListWidget::item {{
                padding: 10px 8px;
                border-radius: 8px;
            }}
            QListWidget::item:selected {{
                background: rgba(69,176,255,0.24);
                color: #ffffff;
            }}
            QTableWidget {{
                gridline-color: #22364f;
                alternate-background-color: rgba(255,255,255,0.02);
            }}
            QProgressBar {{
                background: #102031;
                border: 1px solid #27415c;
                border-radius: 8px;
                min-height: 10px;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {BLUE}, stop:1 {CYAN});
                border-radius: 7px;
            }}
            QPushButton {{
                background: #1f344d;
                border: 1px solid #375778;
                border-radius: 12px;
                color: #eef5ff;
                padding: 10px 14px;
                font-weight: 700;
            }}
            QPushButton:hover {{
                background: #2a4563;
            }}
            QPushButton:pressed {{
                background: #16314a;
            }}
            QPushButton:disabled {{
                background: #122133;
                color: {TEXT_MUTED};
                border: 1px solid #22354d;
            }}
            QLabel#videoPanel {{
                background: #0b1623;
                border: 1px solid #2b4360;
                border-radius: 18px;
                color: {TEXT_SUB};
            }}
            QScrollArea {{
                border: 0;
            }}
            """
        )


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
