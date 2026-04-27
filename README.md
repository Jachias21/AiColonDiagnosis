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

---

## 📚 Orientación y Manuales

Hemos dividido la documentación en dos perfiles distintos para facilitar su lectura según tus necesidades:

### 1. Manual para Profesionales de la Salud y Usuarios Base 🩺 (El "Conductor")
Si deseas utilizar la aplicación gráficamente, saber cómo reportar el riesgo médico, entender la interfaz y las teclas disponibles, consulta nuestro manual interactivo y visual en el que no requiere tener conocimientos previos de programación.
👉 **[Lee el Manual de Usuario aquí](Manual_de_Usuario.md)**

### 2. Manual de Despliegue y Arquitectura ⚙️ (El "Mecánico")
Si te dedicas al desarrollo de software, ciencia de datos o eres el técnico encargado de arrancar y compilar el entorno desde la terminal, sincronizando entornos nativos (`uv`), o deseas re-entrenar modelos o lanzar las métricas de *Streamlit*, este manual detalla los esquemas en profundidad.
👉 **[Lee el Manual de Despliegue aquí](Manual_de_Despliegue.md)**
