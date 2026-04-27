# Manual de Despliegue y Arquitectura (Técnico) - AiColonDiagnosis 

Este documento está dirigido a desarrolladores, ingenieros de datos y administradores del sistema ("el mecánico del coche"). Aquí documentamos cómo instalar la aplicación, los engranajes internos, entrenamientos y configuración avanzada.

---

## 1. Arquitectura del Proyecto

El sistema está desarrollado nativamente en `Python 3.12`. Se apoya en 3 pilares principales:
- **PySide6**: Para el frontend (motor de escritorio de la aplicación principal).
- **Streamlit**: Para el panel de telemetría (Dashboard web analítico).
- **PyTorch y Ultralytics / Scikit-Learn**: Stack para los flujos de inferencia de IA.

---

## 2. Requisitos previos e Instalación

Para asegurar dependencias estables y resolver problemas de colisión cruzada (típicos de paquetes de visión integrados con librerías nativas), empleamos **`uv`**.

### Clonar y preparar:

```bash
git clone https://github.com/TuUsuario/AiColonDiagnosis.git
cd AiColonDiagnosis
```

### Gestión de Entorno con UV (Recomendado):

Sincroniza y crea tu entorno virtual usando el archivo `uv.lock` blindado en el repositorio:

```bash
uv sync
```

*(Nota: En sistemas puramente heredados, puedes instalar convencionalmente usando `.venv/bin/pip install -r requirements.txt` si se provee).*

---

## 3. Despliegue de los Servicios Principales

### Arrancar Inferencia Front-End (App médica central)

Este script levanta todas las piezas y enruta los tres flujos AI en una ventana nativa del sistema operativo:

**Mac y Linux:** `./.venv/bin/python app_pyside6.py`
**Windows:** `.\.venv\Scripts\python.exe app_pyside6.py`

*(O si `uv` está mapeado globalmente: `uv run python app_pyside6.py`)*

### Arrancar la Telemetría (Dashboard Analítico)

Es esencial para visualizar la distribución de dataset y métricas post-entrenamiento:

**Mac/Linux:** `./.venv/bin/python -m streamlit run dashboard.py`
**Windows:** `.\.venv\Scripts\python.exe -m streamlit run dashboard.py`

El servicio web abrirá por defecto en el puerto **8501** (`http://localhost:8501`).

---

## 4. Estructura de Core y Directorios

- `app_pyside6.py`: Capa de la interfaz contemporánea (UI Views & Controllers).
- `detect_realtime.py`: Clases base que controlan los recursos GPU / CPU y los buffers de vídeo usando OpenCV.
- `models/`: Archivos persistidos clave. Si falta un modelo o pesos estáticos:
   - *Fase 1*: `catboost_crc_risk_model.cbm`. (La app posee un Fallback Mock si estuviera ausente).
   - *Fase 2*: `colonoscopy.pt` o `colonoscopy_unet3plus_effnet.pt`.
   - *Fase 3*: `microscopy.pt` y su `microscopy_meta.json`.

---

## 5. Mantenimiento y Modelos de Entrenamiento Secundario

El proyecto permite retener iteraciones de desarrollo dentro del código base para realizar refactoring.

### Reproducir los Datasets
No es requerido para correr la app ya que se exponen en `data/dataset_colon/`, pero para inyectar nueva RAW data:
```bash
python prepare_dataset.py       # Para segmentacion YOLO/Microscopia
python prepare_colon_dataset.py # Estructuracion en folders de entrenamiento
```

### Flujos de Pipeline alternativos de segmentado
Tenemos implementados pipelines de comparación:

1. **Mask R-CNN con back ResNet50**:
   Útil como *Baseline*:
   ```bash
   python train_models/model_colonoscopia/train_maskrcnn_resnet50_compare.py
   ```

2. **Detección Eficiente usando U-Net3+ sobre EfficientNet**:
   Si quieres deducir sin filtraciones (Data Leak) usando Kvasir:
   ```bash
   # Comprobacion / preparacion seca:
   python train_models/model_colonoscopia/train_pretrained_polyp_segmenter.py --prepare-only
   
   # Ejecucion forzada
   python train_models/model_colonoscopia/train_pretrained_polyp_segmenter.py --epochs 8
   ```

Toda la trazabilidad experimental se volcará sobre las subcarpetas dentro de `train_models/`.
