# 🏥 AiColonDiagnosis

Sistema de diagnóstico de cáncer de colon basado en inteligencia artificial, con un flujo de 3 fases:

| Fase | Descripción | Modelo |
|------|-------------|--------|
| **1 — Historial médico** | Carga datos clínicos del paciente (CSV/JSON) y predice riesgo de cáncer | `history_model.pkl` (sklearn) |
| **2 — Colonoscopia** | Analiza vídeo/webcam en tiempo real para detectar pólipos | `colonoscopy.pt` (YOLOv8) |
| **3 — Microscopio** | Analiza vídeo/webcam de tejido para detectar células cancerígenas | `microscopy.pt` (YOLOv8) |

Si el resultado de una fase es negativo, el flujo se detiene. Si las 3 fases son positivas, se muestra un resumen final con recomendación.

---

## Requisitos previos

- **Python 3.12** o superior
- **uv** — gestor de paquetes ([instalar uv](https://docs.astral.sh/uv/getting-started/installation/))

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone <url-del-repo>
cd AiColonDiagnosis

# 2. Instalar dependencias (uv crea el entorno virtual automáticamente)
uv sync
```

Esto instalará todas las dependencias definidas en `pyproject.toml`:

- `numpy` — operaciones numéricas
- `opencv-python` — procesamiento de vídeo y GUI de detección
- `pillow` — manipulación de imágenes
- `ultralytics` — YOLOv8 para detección de objetos

---

## Modelos

La aplicación espera los modelos entrenados dentro de una carpeta `models/` en la raíz del proyecto.

### Estructura esperada

```
AiColonDiagnosis/
├── models/
│   ├── history_model.pkl      ← Modelo de historial clínico (sklearn)
│   ├── colonoscopy.pt         ← Modelo YOLO para pólipos
│   └── microscopy.pt          ← Modelo YOLO para tejido cancerígeno
├── detect_realtime.py
├── ...
```

### Cómo obtener / colocar cada modelo

| Archivo | Formato | Cómo generarlo |
|---------|---------|----------------|
| `models/history_model.pkl` | Pickle (sklearn) | Entrenar un clasificador (RandomForest, XGBoost, etc.) con datos clínicos y guardarlo con `pickle.dump()` o `joblib.dump()` |
| `models/colonoscopy.pt` | YOLOv8 `.pt` | Entrenar con `yolo train data=dataset_yolo/data.yaml` usando el dataset preparado por `prepare_dataset.py` |
| `models/microscopy.pt` | YOLOv8 `.pt` | Entrenar con `yolo train data=dataset_colon/data.yaml` usando el dataset preparado por `prepare_colon_dataset.py` |

> **Nota:** Si los modelos no están presentes, la app funciona en **modo demo** (siempre pasa a la siguiente fase) para poder probar la interfaz completa sin modelos entrenados.

### Crear la carpeta de modelos

```bash
mkdir models
```

Luego copia los archivos `.pt` y `.pkl` dentro de esa carpeta.

---

## Uso

### Aplicación de diagnóstico (3 fases)

```bash
uv run python detect_realtime.py
```

Se abrirá una interfaz gráfica (tkinter) que guía al usuario por las 3 fases del diagnóstico.

**Controles durante las fases de vídeo:**

| Tecla | Acción |
|-------|--------|
| `q` | Finalizar la fase actual |
| `s` | Capturar screenshot |
| `p` | Pausar / reanudar |

Los screenshots se guardan en la carpeta `screenshots/`.

### Preparar datasets

```bash
# Preparar dataset YOLO (Kvasir-SEG + Kvasir normal → 1000 positivos + 1000 negativos)
uv run python prepare_dataset.py

# Preparar dataset de clasificación (colon_aca + colon_n → train/val/test)
uv run python prepare_colon_dataset.py
```

---

## Estructura del proyecto

```
AiColonDiagnosis/
├── detect_realtime.py         # App principal — diagnóstico en 3 fases
├── prepare_dataset.py         # Preparar dataset YOLO desde Kvasir
├── prepare_colon_dataset.py   # Preparar dataset clasificación colon
├── main.py                    # (entrada alternativa)
├── pyproject.toml             # Dependencias del proyecto
├── models/                    # ← Colocar modelos aquí
│   ├── history_model.pkl
│   ├── colonoscopy.pt
│   └── microscopy.pt
├── data/                      # Datos originales (no incluidos en git)
│   ├── colon_image_sets/
│   └── colonoscopia/
├── dataset_yolo/              # Generado por prepare_dataset.py
├── dataset_colon/             # Generado por prepare_colon_dataset.py
└── screenshots/               # Screenshots capturados durante detección
```

---

## Licencia

Proyecto académico.