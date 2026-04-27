# AiColonDiagnosis 🔬

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PySide6](https://img.shields.io/badge/GUI-PySide6-green.svg)](https://wiki.qt.io/Qt_for_Python)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)

Proyecto final integrativo de Inteligencia Artificial para el diagnóstico asistido de cáncer de colon. Está estructurado como una herramienta de apoyo clínico integral, abarcando desde la evaluación preventiva del historial médico hasta el análisis por imagenología avanzada.

## 🌟 Características Principales

El sistema acompaña el flujo de diagnóstico a través de **3 Fases** o modelos:

1. **Fase 1: Análisis de Historial Médico**
   - Evalúa el riesgo del paciente usando **CatBoost/XGBoost** sobre datos tabulares y clínicos.
   - Proporciona interpretabilidad mediante valores **SHAP**, detallando variables predictoras de riesgo.
2. **Fase 2: Colonoscopia en Tiempo Real (Vídeo/Webcam)**
   - Segmentación y detección simultánea de pólipos utilizando modelos preentrenados y finetuneados (arquitecturas de YOLO y UNet3+).
   - Incorpora validación de fotogramas para reducir falsos positivos (el pólipo debe persistir en el tiempo de manera continua por ~1 seg).
3. **Fase 3: Análisis de Imagen Histológica**
   - Clasificación final sobre biopsias / imágenes de microscopio confirmando el tipo de tejido tumoral o benigno.

> 📖 **Para usuarios no técnicos:** Hemos incluido un [Manual de Usuario (Manual_de_Usuario.md)](Manual_de_Usuario.md) que te guiará paso a paso en cómo abrir y usar la aplicación sin conocimientos previos.

---

## 🛠️ Instalación y Configuración Inicial

El proyecto se gestiona idealmente con el gestor de dependencias **`uv`**.

1. Clona el repositorio a tu máquina local.
2. Inicia e instala el entorno virtual con:
```bash
uv sync
```

De forma alternativa, el proyecto cuenta con la carpeta local `.venv` la cual puedes usar directamente activando el entorno en Windows (`.\.venv\Scripts\activate`) o Linux/Mac (`source .venv/bin/activate`).

---

## 🚀 Ejecución de la Plataforma

### Aplicación Médica Principal (Interfaz PySide6)

Esta es la herramienta gráfica base donde se concentra todo el flujo clínico:

**Con `uv`:**
```bash
uv run python app_pyside6.py
```
**O usando tu entorno virtual activado:**
```bash
python app_pyside6.py
```

### Dashboard Analítico y Panel de Control

Para evaluar el entrenamiento, curvas de aprendizaje y comparativas de modelos, despliega el dashboard web:

**Con `uv`:**
```bash
uv run streamlit run dashboard.py
```

---

## 📁 Archivos y Modelos Clave

El repositorio se compone actualmente de los productos finales ejecutables:

- `app_pyside6.py`: Motor gráfico principal y aplicación integral para las 3 fases (PySide6).
- `detect_realtime.py`: Lógica fundacional para la inferencia, captura de video y cálculos predictivos.
- `dashboard.py`: Panel de analítica en `Streamlit`.
- **Modelos Finales (`models/`)**:
   - `catboost_crc_risk_model.cbm`: Modelo tabular de la Fase 1 (si no está disponible, un fallback demo se activa).
   - `colonoscopy.pt` / `colonoscopy_unet3plus_effnet.pt`: Modelos principales asignados de segmentación y detección para imagen laparoscópica.
   - `microscopy.pt`: Modelo de clasificación celular de patología (Fase 3) (+ `microscopy_meta.json`).

---

## 🧪 Sección para Desarrolladores

### 1. Regeneración de Datasets Experimentales
Si deseas recrear desde cero los datos de segmentación:
```bash
python prepare_dataset.py
python prepare_colon_dataset.py
```
*Nota: Los datasets finales ya se encuentran persistidos en `data/dataset_yolo/` y `data/dataset_colon/` por lo que este paso puede ser directamente evitado.*

### 2. Entrenamientos y Comparativas
Los flujos de investigación han derivado en múltiples pruebas comparando distintos enfoques de segmentación (YOLO frente a arquitecturas tipo *Mask R-CNN* o basados en *UNet*).

- **Entrenar Segmentación con MaskRCNN (ResNet50):**
  ```bash
  python train_models/model_colonoscopia/train_maskrcnn_resnet50_compare.py
  ```
- **Entrenar Segmentador UNet3+ (basado en EfficientNet):**
  Añadido fine-tuning sobre `andreribeiro87/unet3plus-efficientnet-kvasir-seg` deduplicando filtraciones usando Kvasir-SEG o datasets externos (como *CVC-ClinicDB*).
  ```bash
  # Preparar y verificar fugas de datos sin entrenar:
  python train_models/model_colonoscopia/train_pretrained_polyp_segmenter.py --prepare-only

  # Lanzar entrenamiento rápido (~8 epochs):
  python train_models/model_colonoscopia/train_pretrained_polyp_segmenter.py --epochs 8 --image-size 352
  ```

Las curvas, métricas resultantes `.json` y evaluaciones se volcarán respectivamente en los sub-directorios del pipeline en `train_models/`.
