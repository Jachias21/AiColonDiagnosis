"""
App principal PySide6 para AiColonDiagnosis.

Mantiene Streamlit para analiticas y usa una UI de escritorio fluida
sin pantallas Tkinter ni ventanas OpenCV sueltas.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import detect_realtime as core


APP_BG = "#0f1720"
CARD_BG = "#1d2733"
PANEL_BG = "#111922"
TEXT_MAIN = "#e7ecf3"
TEXT_SUB = "#9eb0c7"
BLUE = "#23a6d5"
GREEN = "#4bd18a"
RED = "#f56c6c"
YELLOW = "#e8c547"


DEFAULT_PHASE1 = {
    "executed": False,
    "status": "NO_EJECUTADA",
    "is_positive": False,
    "probability": 0.0,
    "demo_mode": True,
    "numeric_fields": 0,
    "total_fields": 0,
    "explanation": None,
}

DEFAULT_PHASE2 = {
    "executed": False,
    "source_mode": "No iniciado",
    "frames_processed": 0,
    "positive_frames": 0,
    "total_detections": 0,
    "peak_detections": 0,
    "unique_polyps": 0,
    "avg_confidence": 0.0,
    "max_confidence": 0.0,
    "confidence_threshold": core.COLONOSCOPY_CONFIDENCE_THRESHOLD,
    "min_confirm_seconds": core.POLYP_CONFIRM_SECONDS,
    "min_confirm_frames": 0,
    "completion_ratio": None,
    "end_reason": "cancelled",
    "explanation": None,
}

DEFAULT_PHASE3 = {
    "executed": False,
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


@dataclass
class AppModels:
    history_model: Any
    colonoscopy_model: Any
    microscopy_model: dict[str, Any] | None


class VideoPhaseController(QObject):
    frame_ready = Signal(QImage)
    stats_changed = Signal(dict)
    finished = Signal(dict)
    info = Signal(str)

    def __init__(self, model: Any):
        super().__init__()
        self.model = model
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
        self.confidence_sum = 0.0
        self.max_confidence = 0.0
        self.paused = False
        self.prev_time = time.time()
        self.fps_display = 0.0
        self.best_detection_score = -1.0
        self.best_focus_frame: np.ndarray | None = None
        self.best_focus_detections: list[dict] = []
        self.active_tracks: list[dict[str, Any]] = []
        self.next_track_id = 1
        self.min_confirm_frames = 0
        self.max_missing_frames = 0
        self.end_reason = "completed"
        self.last_visual_frame: np.ndarray | None = None

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
        effective_fps = max(src_fps / max(core.FRAME_SKIP, 1), 1.0)
        self.min_confirm_frames = max(8, int(round(core.POLYP_CONFIRM_SECONDS * effective_fps)))
        self.max_missing_frames = max(3, int(round(core.POLYP_TRACK_MAX_MISSING_SECONDS * effective_fps)))
        self.paused = False
        self.prev_time = time.time()
        self.fps_display = 0.0
        self.best_detection_score = -1.0
        self.best_focus_frame = None
        self.best_focus_detections = []
        self.active_tracks = []
        self.next_track_id = 1
        self.end_reason = "completed"
        self.last_visual_frame = None
        self.timer.start(15)
        return True

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def stop(self, reason: str = "button") -> None:
        self.end_reason = reason
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        avg_confidence = self.confidence_sum / self.total_detections if self.total_detections > 0 else 0.0
        completion_ratio = (
            min(self.frame_count / self.total_frames, 1.0)
            if self.total_frames > 0
            else None
        )
        stats = {
            "executed": True,
            "source_mode": self.mode,
            "frames_processed": self.frame_count,
            "positive_frames": self.total_positives,
            "total_detections": self.total_detections,
            "peak_detections": self.peak_detections,
            "unique_polyps": self.unique_polyps,
            "avg_confidence": avg_confidence,
            "max_confidence": self.max_confidence,
            "confidence_threshold": core.COLONOSCOPY_CONFIDENCE_THRESHOLD,
            "min_confirm_seconds": core.POLYP_CONFIRM_SECONDS,
            "min_confirm_frames": self.min_confirm_frames,
            "completion_ratio": completion_ratio,
            "end_reason": self.end_reason,
            "explanation": (
                core.create_detection_explanation(
                    self.best_focus_frame,
                    self.best_focus_detections,
                )
                if self.best_focus_frame is not None
                else None
            ),
        }
        self.stats_changed.emit(stats)
        self.finished.emit(stats)

    def current_frame(self) -> np.ndarray | None:
        return self.last_visual_frame.copy() if self.last_visual_frame is not None else None

    def _tick(self) -> None:
        if self.cap is None:
            return

        if self.paused:
            if self.last_visual_frame is not None:
                image = self._to_qimage(self.last_visual_frame)
                self.frame_ready.emit(image)
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop("completed")
            return

        self.frame_count += 1
        if core.FRAME_SKIP > 1 and self.frame_count % core.FRAME_SKIP != 0:
            return

        detections: list[dict] = []
        if self.model is not None:
            detections = core.run_inference(
                self.model,
                frame,
                confidence_threshold=core.COLONOSCOPY_CONFIDENCE_THRESHOLD,
            )

        detections_in_frame = len(detections)
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
            self.confidence_sum += sum(det["confidence"] for det in detections)
            self.max_confidence = max(
                self.max_confidence,
                max(det["confidence"] for det in detections),
            )
            score = sum(det["confidence"] for det in detections)
            if score > self.best_detection_score:
                self.best_detection_score = score
                self.best_focus_frame = frame.copy()
                self.best_focus_detections = [det.copy() for det in detections]

        frame = core.draw_detections(frame, detections, "polyp")

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
            model_loaded=self.model is not None,
            total_positives=self.total_positives,
            paused=self.paused,
        )

        self.last_visual_frame = frame
        image = self._to_qimage(frame)
        self.frame_ready.emit(image)

        quick_stats = {
            "frames_processed": self.frame_count,
            "positive_frames": self.total_positives,
            "unique_polyps": self.unique_polyps,
            "peak_detections": self.peak_detections,
            "max_confidence": self.max_confidence,
        }
        self.stats_changed.emit(quick_stats)

    @staticmethod
    def _to_qimage(frame_bgr: np.ndarray) -> QImage:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        return QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AiColonDiagnosis - App Principal")
        self.resize(1450, 900)

        self.models = self._load_models()

        self.current_consultation_record: dict[str, Any] | None = None
        self.active_patient_name = ""
        self.session_ready = False
        self.report_unlocked = False
        self.read_only_history_mode = False

        self.phase1_result = dict(DEFAULT_PHASE1)
        self.phase2_result = dict(DEFAULT_PHASE2)
        self.phase3_result = dict(DEFAULT_PHASE3)

        self.video_controller = VideoPhaseController(self.models.colonoscopy_model)
        self.video_controller.frame_ready.connect(self._on_video_frame)
        self.video_controller.stats_changed.connect(self._on_video_stats)
        self.video_controller.finished.connect(self._on_video_finished)
        self.video_controller.info.connect(self._show_status)

        self._build_ui()
        self._apply_style()
        self._refresh_home_summary()

    def _load_models(self) -> AppModels:
        self.statusBar().showMessage("Cargando modelos...")
        history_model = core.load_history_model(core.MODEL_HISTORY)
        colonoscopy_model = core.load_yolo_model(core.MODEL_COLONOSCOPY, "Colonoscopia")
        microscopy_model = core.load_classification_model(
            core.MODEL_MICROSCOPY,
            core.MODEL_MICROSCOPY_META,
            "Microscopio",
        )
        return AppModels(history_model, colonoscopy_model, microscopy_model)

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        nav = QFrame()
        nav.setObjectName("nav")
        nav_layout = QVBoxLayout(nav)
        nav_layout.setContentsMargins(18, 20, 18, 20)
        nav_layout.setSpacing(10)

        title = QLabel("AiColonDiagnosis")
        title.setObjectName("navTitle")
        nav_layout.addWidget(title)

        subtitle = QLabel("App principal PySide6")
        subtitle.setObjectName("subtle")
        nav_layout.addWidget(subtitle)

        self.btn_load_history = self._nav_button("Cargar historial")
        self.btn_new_consultation = self._nav_button("Crear nueva consulta")
        self.btn_phase1 = self._nav_button("Fase 1 - Historial")
        self.btn_phase2 = self._nav_button("Fase 2 - Colonoscopia")
        self.btn_phase3 = self._nav_button("Fase 3 - Histologia")
        self.btn_report = self._nav_button("Informe final")
        self.btn_back = self._nav_button("Volver")
        self.btn_close = self._nav_button("Cerrar")

        nav_layout.addWidget(self.btn_load_history)
        nav_layout.addWidget(self.btn_new_consultation)
        nav_layout.addWidget(self.btn_phase1)
        nav_layout.addWidget(self.btn_phase2)
        nav_layout.addWidget(self.btn_phase3)
        nav_layout.addWidget(self.btn_report)
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

        self.stack.addWidget(self.page_home)
        self.stack.addWidget(self.page_consultation)
        self.stack.addWidget(self.page_phase1)
        self.stack.addWidget(self.page_phase2)
        self.stack.addWidget(self.page_phase3)
        self.stack.addWidget(self.page_report)

        layout.addWidget(nav, 0)
        layout.addWidget(self.stack, 1)

        self.btn_load_history.clicked.connect(self._show_consultation_page)
        self.btn_new_consultation.clicked.connect(self._create_new_consultation)
        self.btn_phase1.clicked.connect(lambda: self.stack.setCurrentWidget(self.page_phase1))
        self.btn_phase2.clicked.connect(lambda: self.stack.setCurrentWidget(self.page_phase2))
        self.btn_phase3.clicked.connect(lambda: self.stack.setCurrentWidget(self.page_phase3))
        self.btn_report.clicked.connect(lambda: self.stack.setCurrentWidget(self.page_report))
        self.btn_back.clicked.connect(self._back_to_main_options)
        self.btn_close.clicked.connect(self.close)

        self._update_sidebar_state()

    def _build_home_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(14)

        title = QLabel("Inicio de consulta")
        title.setObjectName("pageTitle")
        layout.addWidget(title)

        hint = QLabel(
            "En la barra lateral solo tienes dos opciones iniciales: "
            "cargar historial o crear nueva consulta."
        )
        hint.setWordWrap(True)
        hint.setObjectName("subtle")
        layout.addWidget(hint)

        self.home_summary = QTextEdit()
        self.home_summary.setReadOnly(True)
        layout.addWidget(self.home_summary, 1)
        return page

    def _build_phase1_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        title = QLabel("Fase 1 - Historial medico")
        title.setObjectName("pageTitle")
        layout.addWidget(title)

        self.btn_run_phase1 = QPushButton("Cargar CSV/JSON y analizar")
        self.btn_run_phase1.clicked.connect(self._run_phase1)
        layout.addWidget(self.btn_run_phase1)

        self.phase1_output = QTextEdit()
        self.phase1_output.setReadOnly(True)
        layout.addWidget(self.phase1_output, 1)
        return page

    def _build_consultation_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        title = QLabel("Seleccionar historial")
        title.setObjectName("pageTitle")
        layout.addWidget(title)

        subtitle = QLabel(
            "Selecciona un perfil de historial y cárgalo para ver todas las fases con sus resultados."
        )
        subtitle.setWordWrap(True)
        subtitle.setObjectName("subtle")
        layout.addWidget(subtitle)

        layout.addWidget(QLabel("Consultas guardadas"))
        self.consultation_list = QListWidget()
        layout.addWidget(self.consultation_list, 1)

        actions = QHBoxLayout()
        self.btn_load_consultation = QPushButton("Cargar perfil seleccionado")
        self.btn_load_consultation.clicked.connect(self._load_selected_consultation)
        self.btn_refresh_consultations = QPushButton("Actualizar lista")
        self.btn_refresh_consultations.clicked.connect(self._refresh_consultation_list)
        actions.addWidget(self.btn_load_consultation)
        actions.addWidget(self.btn_refresh_consultations)
        actions.addStretch(1)
        layout.addLayout(actions)

        self.consultation_detail = QTextEdit()
        self.consultation_detail.setReadOnly(True)
        self.consultation_detail.setMaximumHeight(220)
        layout.addWidget(self.consultation_detail)

        self.consultation_list.currentItemChanged.connect(self._on_consultation_selection)
        self._refresh_consultation_list()
        return page

    def _build_phase2_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(10)

        title = QLabel("Fase 2 - Colonoscopia en vivo (embebida)")
        title.setObjectName("pageTitle")
        layout.addWidget(title)

        controls = QHBoxLayout()
        self.start_video_btn = QPushButton("Abrir video")
        self.start_webcam_btn = QPushButton("Usar webcam")
        self.pause_btn = QPushButton("Pausar/Reanudar")
        self.stop_btn = QPushButton("Finalizar")
        self.shot_btn = QPushButton("Captura")

        self.start_video_btn.clicked.connect(self._start_phase2_video)
        self.start_webcam_btn.clicked.connect(self._start_phase2_webcam)
        self.pause_btn.clicked.connect(self.video_controller.toggle_pause)
        self.stop_btn.clicked.connect(lambda: self.video_controller.stop("button"))
        self.shot_btn.clicked.connect(self._save_phase2_screenshot)

        controls.addWidget(self.start_video_btn)
        controls.addWidget(self.start_webcam_btn)
        controls.addWidget(self.pause_btn)
        controls.addWidget(self.stop_btn)
        controls.addWidget(self.shot_btn)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.video_label = QLabel("Esperando fuente de video...")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(960, 540)
        self.video_label.setStyleSheet("border: 2px solid #334a63; border-radius: 10px;")
        layout.addWidget(self.video_label, 1)

        self.phase2_output = QTextEdit()
        self.phase2_output.setReadOnly(True)
        self.phase2_output.setMaximumHeight(210)
        layout.addWidget(self.phase2_output)
        return page

    def _build_phase3_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(10)

        title = QLabel("Fase 3 - Clasificacion histologica")
        title.setObjectName("pageTitle")
        layout.addWidget(title)

        self.btn_run_phase3 = QPushButton("Seleccionar una o varias imagenes")
        self.btn_run_phase3.clicked.connect(self._run_phase3)
        layout.addWidget(self.btn_run_phase3)

        self.phase3_preview = QLabel("Previsualizacion")
        self.phase3_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.phase3_preview.setMinimumSize(840, 420)
        self.phase3_preview.setStyleSheet("border: 2px solid #334a63; border-radius: 10px;")
        layout.addWidget(self.phase3_preview, 1)

        self.phase3_output = QTextEdit()
        self.phase3_output.setReadOnly(True)
        self.phase3_output.setMaximumHeight(220)
        layout.addWidget(self.phase3_output)
        return page

    def _build_history_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 24, 24, 24)

        title = QLabel("Historial guardado")
        title.setObjectName("pageTitle")
        layout.addWidget(title)

        self.history_list = QListWidget()
        layout.addWidget(self.history_list, 1)

        self.history_detail = QTextEdit()
        self.history_detail.setReadOnly(True)
        self.history_detail.setMaximumHeight(220)
        layout.addWidget(self.history_detail)

        self.history_list.currentItemChanged.connect(self._on_history_selection)
        return page

    def _build_report_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        title = QLabel("Informe final")
        title.setObjectName("pageTitle")
        layout.addWidget(title)

        btn = QPushButton("Regenerar informe")
        btn.clicked.connect(self._update_final_report)
        layout.addWidget(btn)

        self.final_report = QTextEdit()
        self.final_report.setReadOnly(True)
        layout.addWidget(self.final_report, 1)
        return page

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
        }

        lines: list[str] = []
        lines.append(f"Archivo: {Path(path).name}")
        lines.append(f"Campos: {len(patient_data)}")
        lines.append(f"Resultado: {'RIESGO' if is_positive else 'SIN RIESGO'}")
        lines.append(f"Probabilidad: {probability:.1%}")
        lines.append(" ")
        lines.append("Top variables que mas influyeron:")
        for feature in (explanation or {}).get("features", [])[:8]:
            impact = float(feature.get("impact", 0.0))
            lines.append(
                f"- {feature.get('label', 'N/A')}: {feature.get('value', '')} | "
                f"impacto {impact:+.3f} ({feature.get('direction', '')})"
            )

        self.phase1_output.setPlainText("\n".join(lines))
        self._refresh_home_summary()
        self._update_final_report()
        self._autosave_snapshot("fase1")
        self._show_status("Fase 1 completada.")

    def _start_phase2_video(self) -> None:
        if self.read_only_history_mode:
            QMessageBox.information(self, "Fase 2", "Modo historial: esta fase esta en solo lectura.")
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
            self._show_status("Fase 2 iniciada con video.")

    def _start_phase2_webcam(self) -> None:
        if self.read_only_history_mode:
            QMessageBox.information(self, "Fase 2", "Modo historial: esta fase esta en solo lectura.")
            return
        if self.video_controller.start(core.WEBCAM_INDEX, "Webcam"):
            self._show_status("Fase 2 iniciada con webcam.")

    def _save_phase2_screenshot(self) -> None:
        frame = self.video_controller.current_frame()
        if frame is None:
            return
        path = core.save_screenshot(frame, prefix="polyp")
        self._show_status(f"Captura guardada: {path}")

    def _on_video_frame(self, image: QImage) -> None:
        pixmap = QPixmap.fromImage(image)
        scaled = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled)

    def _on_video_stats(self, stats: dict[str, Any]) -> None:
        lines = [
            f"Frames procesados: {stats.get('frames_processed', 0)}",
            f"Frames con candidatos: {stats.get('positive_frames', 0)}",
            f"Polipos confirmados: {stats.get('unique_polyps', 0)}",
            f"Pico detecciones/frame: {stats.get('peak_detections', 0)}",
            f"Confianza maxima: {float(stats.get('max_confidence', 0.0)):.1%}",
        ]
        self.phase2_output.setPlainText("\n".join(lines))

    def _on_video_finished(self, stats: dict[str, Any]) -> None:
        self.phase2_result = stats
        self._render_phase2_output(stats)
        self._refresh_home_summary()
        self._update_final_report()
        self._autosave_snapshot("fase2")
        self._show_status("Fase 2 finalizada.")

    def _run_phase3(self) -> None:
        if self.read_only_history_mode:
            QMessageBox.information(self, "Fase 3", "Modo historial: esta fase esta en solo lectura.")
            return

        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Seleccionar imagenes histologicas",
            str(Path.cwd()),
            "Imagenes (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if not paths:
            return

        if self.models.microscopy_model is None:
            QMessageBox.warning(
                self,
                "Fase 3",
                "El modelo de microscopio no esta cargado. Se mostraran resultados demo.",
            )

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
            cv2.rectangle(
                display,
                (6, 6),
                (display.shape[1] - 6, display.shape[0] - 6),
                border_color,
                4,
            )

            image_results.append(
                {
                    "image_path": image_path,
                    "image_name": Path(image_path).name,
                    "is_malignant": is_cancer,
                    "confidence": confidence,
                    "class_name": cls_name,
                    "preview": display,
                    "explanation": explanation,
                }
            )

        valid_results = image_results
        malignant_results = [r for r in valid_results if r.get("is_malignant")]
        best_result = max(valid_results, key=lambda r: float(r.get("confidence", 0.0)), default={})
        best_malignant = max(malignant_results, key=lambda r: float(r.get("confidence", 0.0)), default=None)
        representative = best_malignant or best_result

        if representative.get("preview") is not None:
            qimg = VideoPhaseController._to_qimage(representative["preview"])
            pixmap = QPixmap.fromImage(qimg).scaled(
                self.phase3_preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.phase3_preview.setPixmap(pixmap)

        self.phase3_result = {
            "executed": True,
            "image_path": representative.get("image_path", ""),
            "image_name": (
                representative.get("image_name", "")
                if len(valid_results) == 1
                else f"{len(valid_results)} imagenes analizadas"
            ),
            "is_malignant": len(malignant_results) > 0,
            "confidence": float(representative.get("confidence", 0.0)),
            "class_name": representative.get("class_name", "N/A"),
            "demo_mode": self.models.microscopy_model is None,
            "explanation": representative.get("explanation"),
            "images": valid_results,
            "total_images": len(valid_results),
            "valid_images": len(valid_results),
            "malignant_count": len(malignant_results),
            "non_malignant_count": len(valid_results) - len(malignant_results),
        }

        lines = [
            f"Imagenes analizadas: {self.phase3_result['total_images']}",
            f"Malignas / no malignas: {self.phase3_result['malignant_count']} / {self.phase3_result['non_malignant_count']}",
            f"Clase destacada: {self.phase3_result['class_name']}",
            f"Confianza destacada: {self.phase3_result['confidence']:.1%}",
            f"Resultado final Fase 3: {'MALIGNA' if self.phase3_result['is_malignant'] else 'NO MALIGNA'}",
        ]
        self.phase3_output.setPlainText("\n".join(lines))
        self.report_unlocked = True
        self._refresh_home_summary()
        self._update_final_report()
        self._update_sidebar_state()
        self._autosave_snapshot("fase3")
        self._show_status("Fase 3 completada.")

    def _show_consultation_page(self) -> None:
        self.stack.setCurrentWidget(self.page_consultation)
        self._refresh_consultation_list()

    def _refresh_consultation_list(self) -> None:
        self.consultation_list.clear()
        records = [r for r in core._load_history_records() if r.get("kind") == "consultation"]
        for record in records:
            patient = record.get("patient_name", "Paciente")
            created = record.get("created_at", "")
            phase2 = (record.get("result") or {}).get("phase2", {})
            phase3 = (record.get("result") or {}).get("phase3", {})
            label = (
                f"{created} | {patient} | "
                f"P2 {phase2.get('unique_polyps', 0)} polipos | "
                f"P3 {phase3.get('malignant_count', 0)}/{phase3.get('total_images', 0)} malignas"
            )
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, record)
            self.consultation_list.addItem(item)

    def _on_consultation_selection(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        if current is None:
            self.consultation_detail.clear()
            return

        record = current.data(Qt.ItemDataRole.UserRole) or {}
        result = record.get("result", {})
        phase1 = result.get("phase1", {})
        phase2 = result.get("phase2", {})
        phase3 = result.get("phase3", {})
        lines = [
            f"Paciente: {record.get('patient_name', '')}",
            f"Fecha: {record.get('created_at', '')}",
            "",
            "Resumen por fases:",
            f"Fase 1: {phase1.get('status', 'N/A')} ({float(phase1.get('probability', 0.0)):.1%})",
            f"Fase 2: {phase2.get('unique_polyps', 0)} polipos confirmados",
            f"Fase 3: {phase3.get('malignant_count', 0)}/{phase3.get('total_images', 0)} malignas",
            "",
            "Pulsa 'Cargar perfil seleccionado' para cargar este historial en todas las pestanas.",
        ]
        self.consultation_detail.setPlainText("\n".join(lines))

    def _create_new_consultation(self) -> None:
        patient, ok = QInputDialog.getText(
            self,
            "Nueva consulta",
            "Nombre del paciente:",
        )
        patient = patient.strip()
        if not ok or not patient:
            return

        self.current_consultation_record = None
        self.active_patient_name = patient
        self.session_ready = True
        self.report_unlocked = False
        self.read_only_history_mode = False
        self.phase1_result = dict(DEFAULT_PHASE1)
        self.phase2_result = dict(DEFAULT_PHASE2)
        self.phase3_result = dict(DEFAULT_PHASE3)
        self._render_phase1_output(self.phase1_result)
        self._render_phase2_output(self.phase2_result)
        self._render_phase3_output(self.phase3_result)
        self._refresh_home_summary()
        self._update_final_report()
        self._update_sidebar_state()
        self._apply_phase_controls_state()
        self.stack.setCurrentWidget(self.page_phase1)
        self._show_status(f"Consulta nueva creada para {patient}.")

    def _load_selected_consultation(self) -> None:
        item = self.consultation_list.currentItem()
        if item is None:
            QMessageBox.information(self, "Consulta", "Selecciona un perfil del historial.")
            return

        record = item.data(Qt.ItemDataRole.UserRole) or {}
        result = record.get("result", {})
        self.current_consultation_record = record

        self.phase1_result = result.get("phase1", dict(DEFAULT_PHASE1))
        self.phase2_result = result.get("phase2", dict(DEFAULT_PHASE2))
        self.phase3_result = result.get("phase3", dict(DEFAULT_PHASE3))
        self.active_patient_name = record.get("patient_name", "")
        self.session_ready = True
        self.report_unlocked = True
        self.read_only_history_mode = True

        self._render_phase1_output(self.phase1_result)
        self._render_phase2_output(self.phase2_result)
        self._render_phase3_output(self.phase3_result, images=record.get("images", {}))
        self._refresh_home_summary()
        self._update_final_report()
        self._update_sidebar_state()
        self._apply_phase_controls_state()
        self.stack.setCurrentWidget(self.page_phase1)
        self._show_status("Perfil historico cargado en todas las fases.")

    def _render_phase1_output(self, phase1: dict[str, Any]) -> None:
        if not phase1.get("executed"):
            self.phase1_output.setPlainText("Fase 1 no ejecutada en esta consulta.")
            return

        lines = [
            f"Estado: {phase1.get('status', 'N/A')}",
            f"Probabilidad: {float(phase1.get('probability', 0.0)):.1%}",
            f"Modo demo: {'Si' if phase1.get('demo_mode') else 'No'}",
            "",
            "Top variables en las que se fijo el modelo:",
        ]
        explanation = phase1.get("explanation") or {}
        for feature in explanation.get("features", [])[:8]:
            impact = float(feature.get("impact", 0.0))
            lines.append(
                f"- {feature.get('label', 'N/A')}: {feature.get('value', '')} | impacto {impact:+.3f}"
            )
        if len(lines) == 5:
            lines.append("- No hay datos de explicabilidad disponibles.")
        self.phase1_output.setPlainText("\n".join(lines))

    def _render_phase2_output(self, phase2: dict[str, Any]) -> None:
        if not phase2.get("executed"):
            self.phase2_output.setPlainText("Fase 2: no hay datos.")
            return

        lines = [
            f"Fuente: {phase2.get('source_mode', 'N/A')}",
            f"Frames procesados: {phase2.get('frames_processed', 0)}",
            f"Frames con candidatos: {phase2.get('positive_frames', 0)}",
            f"Polipos confirmados: {phase2.get('unique_polyps', 0)}",
            f"Confianza media: {float(phase2.get('avg_confidence', 0.0)):.1%}",
            f"Confianza maxima: {float(phase2.get('max_confidence', 0.0)):.1%}",
        ]
        self.phase2_output.setPlainText("\n".join(lines))

    def _render_phase3_output(
        self,
        phase3: dict[str, Any],
        images: dict[str, Any] | None = None,
    ) -> None:
        if not phase3.get("executed"):
            self.phase3_output.setPlainText("Fase 3: no hay datos.")
            self.phase3_preview.setText("Previsualizacion")
            return

        lines = [
            f"Imagenes analizadas: {phase3.get('total_images', 0)}",
            f"Malignas / no malignas: {phase3.get('malignant_count', 0)} / {phase3.get('non_malignant_count', 0)}",
            f"Clase destacada: {phase3.get('class_name', 'N/A')}",
            f"Confianza destacada: {float(phase3.get('confidence', 0.0)):.1%}",
            f"Resultado final Fase 3: {'MALIGNA' if phase3.get('is_malignant') else 'NO MALIGNA'}",
        ]
        self.phase3_output.setPlainText("\n".join(lines))

        img_path = None
        if images:
            img_path = images.get("histologia_resultado") or images.get("histologia_original")
        if img_path and Path(img_path).exists():
            frame = cv2.imread(img_path)
            if frame is not None:
                qimg = VideoPhaseController._to_qimage(frame)
                pixmap = QPixmap.fromImage(qimg).scaled(
                    self.phase3_preview.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self.phase3_preview.setPixmap(pixmap)
                return
            self.phase3_preview.setText("No hay datos de imagen para esta fase.")

    def _update_final_report(self) -> None:
        p1 = self.phase1_result
        p2 = self.phase2_result
        p3 = self.phase3_result

        def _as_float(value: Any, default: float = 0.0) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        p2_unique = int(_as_float(p2.get("unique_polyps", 0), 0.0))
        p1_probability = _as_float(p1.get("probability", 0.0), 0.0)
        p2_max_conf = _as_float(p2.get("max_confidence", 0.0), 0.0)
        p3_conf = _as_float(p3.get("confidence", 0.0), 0.0)
        p3_malignant = int(_as_float(p3.get("malignant_count", 0), 0.0))
        p3_total = int(_as_float(p3.get("total_images", 0), 0.0))

        patient = self.active_patient_name or "Paciente no definido"
        if p3.get("is_malignant"):
            conclusion = "Posible resultado final: muestra maligna"
        elif p1.get("is_positive") and p2_unique > 0:
            conclusion = "Posible resultado final: caso sospechoso"
        elif p2_unique > 0:
            conclusion = "Posible resultado final: hallazgos a revisar"
        else:
            conclusion = "Posible resultado final: sin hallazgos malignos claros"

        lines = [
            "INFORME DE CONSULTA - AiColonDiagnosis",
            "=" * 42,
            f"Paciente: {patient}",
            "",
            "Fase 1 - Historial medico",
            f"- Ejecutada: {'Si' if p1.get('executed') else 'No'}",
            f"- Estado: {p1.get('status', 'N/A')}",
            f"- Probabilidad: {p1_probability:.1%}",
            "",
            "Fase 2 - Colonoscopia",
            f"- Ejecutada: {'Si' if p2.get('executed') else 'No'}",
            f"- Polipos confirmados: {p2_unique}",
            f"- Frames con candidatos: {p2.get('positive_frames', 0)}",
            f"- Confianza maxima: {p2_max_conf:.1%}",
            "",
            "Fase 3 - Histologia",
            f"- Ejecutada: {'Si' if p3.get('executed') else 'No'}",
            f"- Malignas / total: {p3_malignant} / {p3_total}",
            f"- Clase destacada: {p3.get('class_name', 'N/A')}",
            f"- Confianza destacada: {p3_conf:.1%}",
            "",
            "En lo que se ha fijado el modelo (Fase 1)",
        ]
        explanation = p1.get("explanation")
        if not isinstance(explanation, dict):
            explanation = {}
        for feature in explanation.get("features", [])[:5]:
            lines.append(
                f"- {feature.get('label', 'N/A')}: {feature.get('value', '')} | impacto {float(feature.get('impact', 0.0)):+.3f}"
            )
        if not explanation.get("features"):
            lines.append("- Sin datos de explicabilidad disponibles.")

        lines.extend([
            "",
            f"Conclusion: {conclusion}",
        ])
        self.final_report.setPlainText("\n".join(lines))

    def _update_sidebar_state(self) -> None:
        initial_mode = not self.session_ready
        self.btn_load_history.setVisible(initial_mode)
        self.btn_new_consultation.setVisible(initial_mode)

        self.btn_phase1.setVisible(not initial_mode)
        self.btn_phase2.setVisible(not initial_mode)
        self.btn_phase3.setVisible(not initial_mode)
        self.btn_report.setVisible((not initial_mode) and self.report_unlocked)
        self.btn_back.setVisible(not initial_mode)
        self._apply_phase_controls_state()

    def _apply_phase_controls_state(self) -> None:
        read_only = self.read_only_history_mode
        self.btn_run_phase1.setVisible(not read_only)
        self.start_video_btn.setVisible(not read_only)
        self.start_webcam_btn.setVisible(not read_only)
        self.pause_btn.setVisible(not read_only)
        self.stop_btn.setVisible(not read_only)
        self.shot_btn.setVisible(not read_only)
        self.btn_run_phase3.setVisible(not read_only)

    def _back_to_main_options(self) -> None:
        self.session_ready = False
        self.report_unlocked = False
        self.current_consultation_record = None
        self.active_patient_name = ""
        self.read_only_history_mode = False
        self._update_sidebar_state()
        self.stack.setCurrentWidget(self.page_home)
        self._show_status("Has vuelto a las opciones principales.")

    def _autosave_snapshot(self, phase_name: str) -> None:
        if self.read_only_history_mode:
            return
        if not self.session_ready:
            return
        if not self.active_patient_name:
            return
        if not self.phase1_result.get("executed"):
            return

        core.save_patient_consultation_history(
            self.active_patient_name,
            self.phase1_result,
            self.phase2_result,
            self.phase3_result,
        )
        self._refresh_consultation_list()
        self._show_status(f"Datos guardados automaticamente tras {phase_name}.")

    def _try_autosave(self) -> None:
        if not self.session_ready:
            return
        if not self.active_patient_name:
            return
        if not self.phase1_result.get("executed"):
            return
        if not self.phase2_result.get("executed"):
            return
        if not self.phase3_result.get("executed"):
            return

        core.save_patient_consultation_history(
            self.active_patient_name,
            self.phase1_result,
            self.phase2_result,
            self.phase3_result,
        )
        self._refresh_consultation_list()
        self._show_status("Consulta completada y guardada automaticamente.")

    def _show_history_page(self) -> None:
        self._show_consultation_page()

    def _on_history_selection(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        if current is None:
            self.history_detail.clear()
            return
        record = current.data(Qt.ItemDataRole.UserRole) or {}
        result = record.get("result", {})
        lines = [
            f"Tipo: {record.get('kind', '')}",
            f"Fecha: {record.get('created_at', '')}",
            f"Paciente: {record.get('patient_name', '')}",
            " ",
            "Resumen:",
        ]
        lines.append(str(result))
        self.history_detail.setPlainText("\n".join(lines))

    def _save_consultation(self) -> None:
        patient = self.active_patient_name.strip()
        if not patient:
            QMessageBox.warning(self, "Guardar consulta", "Introduce un nombre de paciente.")
            return
        if not self.phase1_result.get("executed"):
            QMessageBox.warning(self, "Guardar consulta", "Ejecuta al menos la Fase 1 antes de guardar.")
            return

        core.save_patient_consultation_history(
            patient,
            self.phase1_result,
            self.phase2_result,
            self.phase3_result,
        )
        QMessageBox.information(self, "Guardar consulta", "Consulta guardada en patients_history.")
        self._refresh_consultation_list()
        self._update_final_report()
        self._show_status("Consulta guardada.")

    def _refresh_home_summary(self) -> None:
        p1 = self.phase1_result
        p2 = self.phase2_result
        p3 = self.phase3_result

        def _as_float(value: Any, default: float = 0.0) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        p2_unique = int(_as_float(p2.get("unique_polyps", 0), 0.0))
        p1_prob = _as_float(p1.get("probability", 0.0), 0.0)
        p3_malignant_count = int(_as_float(p3.get("malignant_count", 0), 0.0))
        p3_total_images = int(_as_float(p3.get("total_images", 0), 0.0))

        if p3.get("is_malignant"):
            conclusion = "Posible resultado final: muestra maligna"
        elif p1.get("is_positive") and p2_unique > 0:
            conclusion = "Posible resultado final: caso sospechoso"
        elif p2_unique > 0:
            conclusion = "Posible resultado final: hallazgos a revisar"
        else:
            conclusion = "Posible resultado final: sin hallazgos malignos claros"

        lines = [
            "Estado de fases",
            "",
            f"Fase 1: {'EJECUTADA' if p1.get('executed') else 'NO EJECUTADA'} | "
            f"{p1.get('status', 'N/A')} | {p1_prob:.1%}",
            f"Fase 2: {'EJECUTADA' if p2.get('executed') else 'NO EJECUTADA'} | "
            f"Polipos confirmados: {p2_unique}",
            f"Fase 3: {'EJECUTADA' if p3.get('executed') else 'NO EJECUTADA'} | "
            f"{p3_malignant_count}/{p3_total_images} malignas",
            "",
            f"Conclusion actual: {conclusion}",
        ]
        self.home_summary.setPlainText("\n".join(lines))

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
            QMainWindow, QWidget {{
                background: {APP_BG};
                color: {TEXT_MAIN};
                font-family: 'Segoe UI';
                font-size: 14px;
            }}
            QFrame#nav {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {PANEL_BG}, stop:1 #0b121a);
                border-right: 1px solid #223244;
                min-width: 300px;
                max-width: 300px;
            }}
            QLabel#navTitle {{
                font-size: 24px;
                font-weight: 700;
                color: #f4f8ff;
            }}
            QLabel#pageTitle {{
                font-size: 28px;
                font-weight: 700;
                color: #f4f8ff;
            }}
            QLabel#subtle {{
                color: {TEXT_SUB};
                font-size: 13px;
            }}
            QTextEdit, QListWidget, QLineEdit {{
                background: {CARD_BG};
                border: 1px solid #2d4058;
                border-radius: 10px;
                color: {TEXT_MAIN};
                padding: 10px;
            }}
            QPushButton {{
                background: #203247;
                border: 1px solid #3a5574;
                border-radius: 10px;
                color: #eaf2ff;
                padding: 10px 14px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background: #2a4361;
            }}
            QPushButton:pressed {{
                background: #183048;
            }}
            """
        )


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
