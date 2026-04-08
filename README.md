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
- La Fase 1 no tiene todavía `models/history_model.pkl`, así que funciona en modo demo.
- Ya existen los datasets generados en `data/dataset_yolo/` y `data/dataset_colon/`.
- Ya existen resultados de entrenamiento de colonoscopia en `train_models/model_colonoscopia/entrenamiento/`.

## Cómo ejecutarlo ahora mismo

Todos los comandos de abajo están pensados para ejecutarse desde la carpeta del proyecto:

```powershell
cd C:\Programas\Proyecto2\AiColonDiagnosis
```

### Opción recomendada: usar el entorno ya creado

Este repo ya tiene `.venv`, así que puedes arrancarlo directamente sin reinstalar nada:

### 1. App principal de diagnóstico

```powershell
.\.venv\Scripts\python.exe .\detect_realtime.py
```

Qué te vas a encontrar:

- Menú principal con acceso al flujo completo o a cada fase por separado.
- Fase 1: carga de datos clínicos en modo demo.
- Fase 2: detección de pólipos con el modelo real `colonoscopy.pt`.
- Fase 3: clasificación de imagen histológica con el modelo real `microscopy.pt`.

Notas rápidas:

- En la Fase 2 puedes usar webcam o vídeo, según el flujo de la app.
- En la Fase 3 se analiza una imagen única.
- Los screenshots se guardan en `screenshots/`.

Controles de vídeo:

- `q`: salir de la fase actual
- `s`: guardar screenshot
- `p`: pausar o reanudar

### 2. Dashboard web

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

- `detect_realtime.py`: app principal
- `dashboard.py`: panel Streamlit
- `main.py`: archivo de ejemplo, no es la entrada real del proyecto
- `models/colonoscopy.pt`: modelo de colonoscopia
- `models/microscopy.pt`: modelo de microscopía
- `models/microscopy_meta.json`: metadata necesaria para cargar el modelo de microscopía
- `train_models/model_colonoscopia/entrenamiento/`: métricas y pesos ya generados

## Limitación actual

Falta el archivo:

```text
models/history_model.pkl
```

Mientras no exista, la Fase 1 seguirá en modo demo y el flujo continuará hacia las fases siguientes.
