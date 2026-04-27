# Manual de Usuario Integral - AiColonDiagnosis 🩺

¡Bienvenido a AiColonDiagnosis! Este manual explica cómo **utilizar** la aplicación médica en tu día a día, como profesional de la salud o usuario final. No necesitas tener conocimientos técnicos.

---

## 1. ¿Qué es AiColonDiagnosis?

**AiColonDiagnosis** es un asistente interactivo diseñado para ayudar en la detección del cáncer de colon. La aplicación te guía a lo largo de un flujo de 3 etapas complementarias:

1. **Fases 1 (Historial Médico):** Analiza el riesgo inicial a partir de constantes vitales y características previas del paciente.
2. **Fase 2 (Colonoscopia):** Permite detectar pólipos en tiempo real durante una inspección en vídeo.
3. **Fase 3 (Microscopía):** Verifica si el tejido de la biopsia analizado en el laboratorio es benigno o maligno.

---

## 2. Abriendo la Aplicación

Consulta con tu equipo técnico cómo te han facilitado el acceso. Generalmente, basta con abrir el acceso directo de **AiColonDiagnosis**. 
Al abrirse, verás el **Menú Principal**, desde el que podrás elegir:
- **Flujo Completo:** Para ver de manera seguida el estado del paciente.
- **Acceso rápido a una fase específica:** Si solo necesitas comprobar unas constantes o mirar una imagen de microscopio suelta.

---

## 3. Uso de la Aplicación (El flujo Médico)

### Fase 1: Riesgo por Historial Médico
1. Ingresa los datos solicitados en los formularios sobre tu paciente (edad, índice de masa corporal, antecedentes, etc.).
2. Pulsa en el botón para Evaluar.
3. Observarás un indicador de riesgo y una gráfica ("Explicación SHAP") que resalta en verde o rojo qué factores han bajado o incrementado el nivel de alerta (por ejemplo: si es fumador, aumentará la alerta en rojo).

### Fase 2: Análisis de Colonoscopia (Vídeo)
En esta fase, el sistema revisará un vídeo médico o la cámara. Mientras el vídeo avanza, la Inteligencia Artificial dibujará "cajas" rojas cuando detecte un posible pólipo.
*   **Nota de confianza:** El sistema solo alertará fijamente cuando esté muy seguro. Si el cuadro parpadea una fracción de segundo, es que el sistema ha descartado la zona.

**Teclas Especiales en esta Fase:**
- `p` 👉 **Pausar o reanudar** el vídeo. Ideal para examinar mejor una zona concreta.
- `s` 👉 Tomar una **captura de pantalla** o *Screenshot*. Útil para anexarlo al informe del paciente (se guardan automáticamente en tu ordenador).
- `q` 👉 **Cerrar** el vídeo y volver a la app.

### Fase 3: Análisis Microscópico
1. Haz clic en "Añadir Archivos" o cargar imágenes. 
2. Puedes subir múltiples fotografías a la vez procedentes del laboratorio.
3. El sistema aplicará un filtro de color automático (segmentación visual) y te dirá con exactitud de qué naturaleza es el tejido.

---

**Con esto, estás preparado para utilizar el sistema y tomar mejores decisiones clínicas.** 

*(Si tienes problemas con la instalación, la pantalla se queda negra o deseas personalizar el sistema internamente, consulta el `Manual_de_Despliegue.md` diseñado para equipos técnicos).*
