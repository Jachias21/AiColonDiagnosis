# Manual de Usuario Integral - AiColonDiagnosis

¡Bienvenido a AiColonDiagnosis! Este es un manual detallado diseñado **para todo el mundo**. No importa si eres un desarrollador experto, un profesional de la salud o un usuario que nunca antes ha abierto una terminal. Aquí te explicaremos paso a paso qué es este sistema y cómo utilizarlo de forma sencilla.

---

## 1. ¿Qué es AiColonDiagnosis?

**AiColonDiagnosis** es un software interactivo asistido por Inteligencia Artificial diseñado para ayudar en el diagnóstico temprano del cáncer de colon. El sistema te acompaña mediante 3 filtros o "fases":

1. **Fase 1: Historial Médico.** Se analizan los datos clínicos y características previas del paciente (edad, antecedentes, etc.) utilizando modelos predictivos para evaluar el riesgo inicial.
2. **Fase 2: Colonoscopia.** Se utiliza la cámara en vivo o un vídeo grabado para detectar pólipos en tiempo real durante una inspección visual.
3. **Fase 3: Imagen Histológica.** Se examinan imágenes de microscopio (biopsias) enviadas desde un laboratorio para confirmar si el tejido analizado es maligno o benigno.

---

## 2. Requisitos iniciales (Abriendo la puerta al sistema)

Para usar este programa necesitas utilizar lo que se conoce como **Terminal** o **Símbolo del sistema**. Es una aplicación que permite comunicarnos con el ordenador a través de texto.

### ¿Cómo abrir la Terminal?
*   **En Windows:** Pulsa la tecla `Windows` en tu teclado, escribe `cmd` o `Terminal` y pulsa *Enter*.
*   **En Mac:** Pulsa `Command` (Cmd ⌘) + `Espacio`, escribe `Terminal` y pulsa *Enter*.
*   **En Linux:** Pulsa `Ctrl + Alt + T` al mismo tiempo.

Se abrirá una ventana de fondo negro o blanco donde puedes escribir comandos.

### ¿Cómo llegar al programa?
Una vez tengas la Terminal abierta, debes decirle a tu ordenador dónde están guardados los archivos de la aplicación. Para ello, utiliza el comando `cd` (cambiar directorio) seguido de un espacio y de la ruta donde descargaste este proyecto. Por ejemplo:

**Si usas Windows:**
```bash
cd "C:\Ruta\Hacia\La\Carpeta\AiColonDiagnosis"
```

**Si usas Mac o Linux:**
```bash
cd "/Ruta/Hacia/La/Carpeta/AiColonDiagnosis"
```

---

## 3. ¿Cómo ejecutar y usar la Aplicación Principal?

### Arrancando el programa

Asegúrate de haber hecho el paso anterior (estar en la carpeta del programa). Ahora, ejecuta (pulsando *Enter*) tu comando según tu sistema:

**En Windows:**
```bash
.\.venv\Scripts\python.exe main.py
```

**En Mac o Linux:**
```bash
./.venv/bin/python main.py
```

*(Si prefieres arrancar el motor más moderno de forma directa, puedes teclear: `app_pyside6.py` en lugar de `main.py` al final del comando anterior).*

Al hacerlo, se te abrirá una **ventana de menú principal**. 

### Opciones de la Interfaz

* **Flujo Completo:** Te permite analizar a un paciente pasando secuencialmente por la Fase 1, luego la Fase 2 y finalmente la Fase 3.
* **Módulos Individuales:** Si sólo quieres hacer un análisis rápido de un historial (Fase 1) o analizar una foto de microscopio (Fase 3), puedes seleccionar cada opción por separado.

### Controles Ocultos del Teclado (Muy útiles)
Durante la revisión médica visual (sobre todo en imágenes o vídeos en la Fase 2), recuerda estas teclas:
* `p` --> **Pausar o reanudar** el vídeo.
* `s` --> Tomar una **captura de pantalla** o *Screenshot*. Éstas se guardarán automáticamente en la carpeta `screenshots/`.
* `q` --> **Salir / Cerrar** la fase actual para volver atrás.

---

## 4. Detalles de las Fases del Análisis

### Fase 1: Análisis de Historial Médico
Escribirás los datos o abrirás un archivo de las constantes vitales y características del paciente. El programa usará un modelo de IA (CatBoost) o una versión "Demo" (si el modelo no está cargado) para calcular el riesgo general.

### Fase 2: Análisis por Vídeo (Colonoscopia)
El programa se conecta a la cámara web o abre un vídeo médico. En tiempo real, la IA detectará y señalará posibles pólipos en pantalla.

### Fase 3: Análisis Microscópico
Podrás cargar una o varias imágenes de biopsia al mismo tiempo. El programa indicará los resultados bajo el microscopio.

---

## 5. El "Dashboard" o Panel Estadístico

Aparte de la aplicación médica, existe otra aplicación web muy completa enfocada en gráficas, métricas e información técnica.

Para abrirlo, vuelve a tu **Terminal**, asegúrate de estar en la carpeta del proyecto y ejecuta el siguiente comando aplicable a tu ecosistema:

**En Windows:**
```bash
.\.venv\Scripts\python.exe -m streamlit run dashboard.py
```

**En Mac o Linux:**
```bash
./.venv/bin/python -m streamlit run dashboard.py
```

---

## 6. Apartado exclusivo para Desarrolladores

Si eres analista de datos, programador o ingeniero, aquí tienes datos extra sobre cómo está armado esto por dentro:

* **Gestión de Paquetes:** El proyecto utiliza `uv` o entornos virtuales.
* **Archivos Clave:**
   * `app_pyside6.py`: Capa de la interfaz moderna (PySide6).
   * `detect_realtime.py`: Algoritmos base para el procesamiento en tiempo real.
   * `dashboard.py`: Panel web con `Streamlit`.
* **Modelos Finales (`models/`):**
   * `colonoscopy.pt` / `colonoscopy_unet3plus_effnet.pt`: Segmentación de vídeo.
   * `microscopy.pt`: Clasificación celular.
* **Entrenamientos Secundarios:**
   * Si deseas correr rutinas de entrenamiento exploratorio (Mask R-CNN, ResNet50, arquitecturas YOLO) usa los scripts bajo la carpeta `train_models/model_colonoscopia/`. 
     Ejemplo en Windows: `.\.venv\Scripts\python.exe train_models\model_colonoscopia\train_maskrcnn_resnet50_compare.py`.

---

**¡Disfruta usando AiColonDiagnosis!**
