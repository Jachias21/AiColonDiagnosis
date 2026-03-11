# 🏥 AiColonDiagnosis

Sistema de diagnóstico de cáncer de colon basado en inteligencia artificial, con un flujo de 3 fases:

| Fase | Descripción | Modelo |
|------|-------------|--------|
| **1 — Historial médico** | Carga datos clínicos del paciente (CSV/JSON) y predice riesgo de cáncer | `history_model.pkl` (sklearn) |
| **2 — Colonoscopia** | Analiza vídeo/webcam en tiempo real para detectar pólipos | `colonoscopy.pt` (YOLOv8) |
| **3 — Microscopio** | Analiza vídeo/webcam de tejido para detectar células cancerígenas | `microscopy.pt` (YOLOv8) |

El flujo completo va Fase 1 → 2 → 3 → Resultado final. Si una fase es negativa, el flujo se detiene.
Desde el menú principal también se puede **acceder directamente a cualquier fase individual** para probar un modelo sin pasar por las anteriores.

---

## Requisitos previos

- **Python 3.12** o superior
- **uv** — gestor de paquetes ([instalar uv](https://docs.astral.sh/uv/getting-started/installation/))
- **GPU NVIDIA** con CUDA (recomendado para entrenamiento e inferencia en tiempo real)
  - El proyecto está configurado para PyTorch con CUDA 12.4 en Windows

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

| Paquete | Uso |
|---------|-----|
| `numpy` | Operaciones numéricas |
| `opencv-python` | Procesamiento de vídeo y GUI de detección |
| `pillow` | Manipulación de imágenes |
| `ultralytics` | YOLOv8 para detección de objetos |
| `torch` + `torchvision` | Backend de deep learning (CUDA 12.4 en Windows) |
| `streamlit` | Dashboard web para probar modelos, ver métricas y entrenar |
| `plotly` | Gráficas interactivas en el dashboard |
| `pandas` | Manejo de datos tabulares |

---

## Modelos

La aplicación espera los modelos entrenados dentro de `models/` en la raíz del proyecto.

### Estructura esperada

```
models/
├── history_model.pkl      ← Modelo de historial clínico (sklearn)
├── colonoscopy.pt         ← Modelo YOLO para pólipos
└── microscopy.pt          ← Modelo YOLO para tejido cancerígeno
```

### Cómo obtener cada modelo

| Archivo | Formato | Cómo generarlo |
|---------|---------|----------------|
| `history_model.pkl` | Pickle (sklearn) | Entrenar un clasificador con datos clínicos y guardarlo con `pickle.dump()` |
| `colonoscopy.pt` | YOLOv8 `.pt` | Entrenar con el script `train_models/model_colonoscopia/` o desde el dashboard |
| `microscopy.pt` | YOLOv8 `.pt` | Entrenar con dataset de microscopía |

> **Nota:** Si los modelos no están presentes, la app funciona en **modo demo** (siempre pasa a la siguiente fase) para probar la interfaz sin modelos entrenados.

---

## Uso

### 1. Aplicación de diagnóstico (3 fases)

```bash
uv run python detect_realtime.py
```

Se abrirá una interfaz gráfica (tkinter) con el menú principal:

- **▶ Diagnóstico completo** — Flujo secuencial: Fase 1 → 2 → 3 → Resultado
- **📋 Historial** — Ir directamente a Fase 1
- **📹 Colonoscopia** — Ir directamente a Fase 2 (detección de pólipos en vídeo)
- **🔬 Microscopio** — Ir directamente a Fase 3

**Controles durante las fases de vídeo:**

| Tecla | Acción |
|-------|--------|
| `q` | Finalizar la fase actual |
| `s` | Capturar screenshot |
| `p` | Pausar / reanudar |

Los screenshots se guardan en `screenshots/`.

### 2. Dashboard web (Streamlit)

```bash
uv run streamlit run dashboard.py
```

El dashboard tiene 3 páginas:

| Página | Función |
|--------|---------|
| **🔬 Probar modelo** | Subir una imagen o elegir del dataset, ejecutar inferencia y ver detecciones con bounding boxes |
| **📊 Métricas** | Visualizar curvas de pérdida, mAP, precisión/recall y matriz de confusión del entrenamiento |
| **🏋️ Entrenar modelo** | Configurar hiperparámetros (época, batch, lr, augmentation...) y lanzar entrenamiento |

### 3. Preparar datasets

```bash
# Dataset YOLO para colonoscopia (Kvasir-SEG + normales → 1000 pos + 1000 neg)
uv run python prepare_dataset.py

# Dataset clasificación colon (colon_aca + colon_n → train/val/test)
uv run python prepare_colon_dataset.py
```

Los datasets se generan en `data/dataset_yolo/` y `data/dataset_colon/` respectivamente.

### 4. Entrenar modelo de colonoscopia (script directo)

```bash
uv run python train_models/model_colonoscopia/train_model_colonoscopia.py
```

También se puede entrenar desde el dashboard (página "🏋️ Entrenar modelo").

---

## Estructura del proyecto

```
AiColonDiagnosis/
├── detect_realtime.py                          # App principal — diagnóstico en 3 fases con GUI
├── dashboard.py                                # Dashboard Streamlit (probar/métricas/entrenar)
├── prepare_dataset.py                          # Preparar dataset YOLO desde Kvasir
├── prepare_colon_dataset.py                    # Preparar dataset clasificación colon
├── main.py                                     # Entrada alternativa
├── pyproject.toml                              # Dependencias y configuración del proyecto
├── models/                                     # Modelos entrenados
│   ├── history_model.pkl
│   ├── colonoscopy.pt
│   └── microscopy.pt
├── train_models/                               # Scripts y resultados de entrenamiento
│   ├── model_colonoscopia/
│   │   ├── train_model_colonoscopia.py         # Script de entrenamiento con comentarios
│   │   └── entrenamiento/                      # Resultados (results.csv, weights/, gráficas)
│   ├── model_historial/
│   └── model_microscopio/
├── data/                                       # Datos originales (no incluidos en git)
│   ├── colon_image_sets/                       # Imágenes de clasificación
│   ├── colonoscopia/                           # Datasets Kvasir
│   ├── dataset_yolo/                           # Generado por prepare_dataset.py
│   └── dataset_colon/                          # Generado por prepare_colon_dataset.py
└── screenshots/                                # Capturas durante detección en vídeo
```

---

## Datos

Los datasets originales no se incluyen en el repositorio (ver `.gitignore`). Se necesitan:

| Dataset | Carpeta | Uso |
|---------|---------|-----|
| **Kvasir-SEG** | `data/colonoscopia/kvasir-seg/` | Imágenes + máscaras + bboxes de pólipos |
| **Kvasir Dataset v3** | `data/colonoscopia/kvasir-dataset-v3/` | Imágenes normales (cécum, píloro, z-line) |
| **Colon Image Sets** | `data/colon_image_sets/` | `colon_aca` (adenocarcinoma) + `colon_n` (normal) |

---

## Licencia

Proyecto académico.