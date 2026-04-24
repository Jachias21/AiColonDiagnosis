# AiColonDiagnosis

Proyecto de diagnóstico asistido por IA para cáncer de colon con 3 fases:

1. Historial médico
2. Colonoscopia
3. Imagen histológica

## Estado actual del proyecto

Esto es lo que ya está montado en este repo y sí merece la pena ejecutar ahora mismo:

- La app principal `detect_realtime.py` está lista para abrirse.
- La Fase 2 tiene modelo real cargable: `models/colonoscopy.pt`.
- La Fase 3 tiene modelo real cargable: `models/microscopy.pt` + `models/microscopy_meta.json`.
- La Fase 1 usa `models/catboost_crc_risk_model.cbm` si existe; si no, funciona en modo demo.
- Ya existen los datasets generados en `data/dataset_yolo/` y `data/dataset_colon/`.
- Ya existen resultados de entrenamiento de colonoscopia en `train_models/model_colonoscopia/entrenamiento/`.

## Cómo ejecutarlo ahora mismo

Todos los comandos de abajo están pensados para ejecutarse desde la carpeta del proyecto:

```powershell
cd C:\Programas\Proyecto2\AiColonDiagnosis
```

### Opción recomendada: usar el entorno ya creado

Este repo ya tiene `.venv`, así que puedes arrancarlo directamente sin reinstalar nada:

### 1. App principal de diagnóstico (PySide6)

```powershell
.\.venv\Scripts\python.exe .\main.py
```

Si prefieres abrir directamente el script de la nueva interfaz:

```powershell
.\.venv\Scripts\python.exe .\app_pyside6.py
```

Qué te vas a encontrar:

- Menú principal con acceso al flujo completo o a cada fase por separado.
- Fase 1: carga de datos clínicos con CatBoost si el modelo está disponible, o modo demo si falta.
- Fase 2: detección de pólipos con el modelo real `colonoscopy.pt`.
- Fase 3: clasificación de imagen histológica con el modelo real `microscopy.pt`.

Notas rápidas:

- En la Fase 2 puedes usar webcam o vídeo, según el flujo de la app.
- En la Fase 1 puedes abrir una explicación SHAP para ver qué variables clínicas han pesado más en la predicción.
- En la Fase 2 la app separa candidatos de pólipos confirmados: un candidato solo cuenta como pólipo si supera la confianza mínima y persiste aproximadamente 1 segundo.
- En la Fase 3 puedes seleccionar una o varias imágenes histológicas en la misma ejecución.
- La app guarda el historial por paciente en `patients_history/` solo cuando ejecutas la consulta completa.
- Los screenshots se guardan en `screenshots/`.

Controles de vídeo:

- `q`: salir de la fase actual
- `s`: guardar screenshot
- `p`: pausar o reanudar

### 2. Dashboard web (analíticas / entrenamiento)

```powershell
.\.venv\Scripts\python.exe -m streamlit run .\dashboard.py
```

Después abre en el navegador la URL que te muestre Streamlit, normalmente:

```text
http://localhost:8501
```

Lo más útil del dashboard ahora mismo:

- Ver métricas del entrenamiento de colonoscopia ya guardado.
- Lanzar entrenamiento YOLO de colonoscopia usando el dataset ya preparado.
- Probar inferencia sobre modelos YOLO de colonoscopia.
- Entrenar un modelo alternativo de segmentación con ResNet50 para comparar contra YOLO:

```powershell
.\.venv\Scripts\python.exe .\train_models\model_colonoscopia\train_maskrcnn_resnet50_compare.py
```

Ese script:

- usa las máscaras exactas de `data/dataset_yolo/masks/`
- entrena `Mask R-CNN + ResNet50-FPN`
- guarda curvas y gráficas de entrenamiento
- compara el modelo nuevo contra `models/colonoscopy.pt`
- exporta el nuevo modelo a `models/colonoscopy_maskrcnn_resnet50.pth`

Importante:

- El dashboard está orientado sobre todo al flujo YOLO de colonoscopia.
- La inferencia de microscopía que ya funciona de verdad está integrada en `detect_realtime.py`.

## Si prefieres usar `uv`

Si en tu máquina quieres reconstruir el entorno desde cero:

```powershell
uv sync
uv run python .\detect_realtime.py
uv run streamlit run .\dashboard.py
```

## Qué no hace falta ejecutar ahora

No necesitas preparar datasets otra vez para probar el proyecto actual, porque ya existen:

- `data/dataset_yolo/`
- `data/dataset_colon/`

Por eso estos scripts no son necesarios para arrancar lo que ya funciona:

```powershell
python .\prepare_dataset.py
python .\prepare_colon_dataset.py
```

Úsalos solo si quieres regenerar los datasets.

## Archivos importantes

- `app_pyside6.py`: app principal moderna (PySide6)
- `detect_realtime.py`: motor original y funciones de inferencia reutilizadas por la app principal
- `dashboard.py`: panel Streamlit
- `main.py`: archivo de ejemplo, no es la entrada real del proyecto
- `models/catboost_crc_risk_model.cbm`: modelo CatBoost de historial médico
- `models/colonoscopy.pt`: modelo de colonoscopia
- `models/colonoscopy_unet3plus_effnet.pt`: segmentador UNet3+ usado como modelo principal en Fase 2
- `models/microscopy.pt`: modelo de microscopía
- `models/microscopy_meta.json`: metadata necesaria para cargar el modelo de microscopía
- `train_models/model_colonoscopia/entrenamiento/`: métricas y pesos ya generados

## Nuevo entrenamiento de segmentacion de polipos

Para probar un segmentador medico preentrenado antes de tocar el YOLO de produccion:

```powershell
.\.venv\Scripts\python.exe .\train_models\model_colonoscopia\train_pretrained_polyp_segmenter.py
```

El script nuevo:

- parte de `andreribeiro87/unet3plus-efficientnet-kvasir-seg`
- descarga Kvasir-SEG oficial si no existe
- deduplica contra el test local para evitar fugas de datos
- mantiene `models/colonoscopy.pt` como baseline
- exporta el candidato a `models/colonoscopy_unet3plus_effnet.pt`
- guarda curvas, galeria visual y comparativa en `train_models/model_colonoscopia/pretrained_polyp_segmenter/`

Comandos utiles:

```powershell
# Solo preparar datos y comprobar duplicados, sin entrenar
.\.venv\Scripts\python.exe .\train_models\model_colonoscopia\train_pretrained_polyp_segmenter.py --prepare-only

# Entrenamiento rapido de prueba
.\.venv\Scripts\python.exe .\train_models\model_colonoscopia\train_pretrained_polyp_segmenter.py --epochs 8 --image-size 352

# Anadir normales desde HyperKvasir en modo streaming limitado
.\.venv\Scripts\python.exe .\train_models\model_colonoscopia\train_pretrained_polyp_segmenter.py --download-hyperkvasir-normals
```

Si tienes CVC-ClinicDB descargado de Kaggle, ponlo en:

```text
data/external_polyp_sources/cvc_clinicdb/
```

El script lo importara automaticamente si encuentra carpetas de imagenes y mascaras.

## Limitación actual

Si falta el archivo:

```text
models/catboost_crc_risk_model.cbm
```

Mientras no exista, la Fase 1 seguirá en modo demo y el flujo continuará hacia las fases siguientes.
