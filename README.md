# -- Sistema de Seguridad Facial con Clasificación de Zonas -- 

Creado con ayuda de nuestro amigo Claudio (Cluade).

Sistema de seguridad en tiempo real que combina **reconocimiento facial** con **clasificación de zonas del hogar**. Detecta intrusos, identifica en qué zona están (cocina, dormitorio, pasillo...) y manda una notificación push al móvil.

Proyecto Final de Computer Vision — Master en Deep Learning · MIOTI · 2026

---

## ¿Qué hace?

```
Frame de cámara
      ↓
InsightFace (ArcFace) → ¿está en la whitelist?
      ↓
ConvNeXt-Small (fastai) → ¿en qué zona está?
      ↓
"Intruso detectado en COCINA"
      ↓
Alerta push (ntfy) + captura guardada en alerts_log/
```

---

## Stack

| Componente | Tecnología |
|---|---|
| Detección facial | YOLOv12n-face |
| Reconocimiento | InsightFace (ArcFace, embeddings 512d) |
| Clasificación de zonas | ConvNeXt-Small fine-tuned (fastai) |
| Dataset zonas | MIT Indoor Scenes CVPR-67 (8 clases filtradas) |
| Alertas | ntfy.sh (push móvil) |
| Panel de control | Gradio |
| Demo clasificador | [HuggingFace Hub](https://huggingface.co/mynorhm/security-room-classifier) |

**Zonas detectadas:** `bathroom · bedroom · corridor · dining_room · garage · kitchen · livingroom · stairscase`

---

## Estructura del proyecto

```
CV_ProyectoFinal/
├── 01_setup_and_test.ipynb        # test de cámara y detección YOLO
├── 02_embeddings_whitelist.ipynb  # registro de personas autorizadas
├── 03_pipeline_live.ipynb         # pipeline en tiempo real
├── 04_gradio_panel.ipynb          # panel de control Gradio
├── src/
│   └── sistema.py                 # script principal del sistema
├── whitelist/                     # embeddings de personas autorizadas (se crea al registrar)
├── alerts_log/                    # capturas de intrusos (se crea automáticamente)
├── models/
│   ├── yolov12n-face.pt           # modelo YOLO (descargar manualmente, ver paso 4)
│   └── clases.json                # clases del clasificador de zonas
└── pyproject.toml                 # dependencias del proyecto (uv)
```

---

## Instalación desde 0

### Requisitos
- Python 3.11
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- PC con cámara (probado en M1 Pro)

### Paso 1 — Clona el repo

```bash
git clone https://github.com/Mynozy/CV_ProyectoFinal.git
cd CV_ProyectoFinal
```

### Paso 2 — Crea el entorno y instala dependencias

```bash
uv venv --python 3.11
uv sync
```

### Paso 3 — Descarga el modelo de zonas desde HuggingFace

```python
# ejecuta esto en un notebook o script Python
from huggingface_hub import hf_hub_download
import shutil

ruta = hf_hub_download(
    repo_id="mynorhm/security-room-classifier",
    filename="full_model.pth",
    repo_type="model"
)
shutil.copy(ruta, "models/full_model.pth")
print("✅ Modelo descargado")
```

### Paso 4 — Descarga el modelo YOLO

Ve a [github.com/akanametov/yolo-face/releases](https://github.com/akanametov/yolo-face/releases) y descarga `yolov12n-face.pt`. Colócalo en `models/`.

### Paso 5 — Configura ntfy en el móvil

1. Descarga la app **ntfy** en iOS o Android
2. Suscríbete al canal que definas en `src/sistema.py` → `NTFY_CANAL`
3. Cada alerta de intruso llegará como notificación push

### Paso 6 — Arranca JupyterLab

```bash
uv run jupyter lab
```

---

## Uso

### Registrar personas en la whitelist

Abre `02_embeddings_whitelist.ipynb` y ejecuta la celda de registro. El sistema captura 5 fotos de la persona y guarda el embedding promedio.

### Arrancar el sistema

**Opción A — Desde terminal:**
```bash
uv run python src/sistema.py
```

**Opción B — Desde el panel Gradio:**

Abre `04_gradio_panel.ipynb`, ejecuta todas las celdas y pulsa **▶️ Arrancar sistema** en la interfaz.

### Panel de control (Gradio)

| Pestaña | Función |
|---|---|
| 🎥 Control del Sistema | Arrancar / parar el sistema en vivo |
| 👤 Gestionar Whitelist | Añadir / eliminar personas autorizadas |
| 🚨 Log de Alertas | Ver las últimas capturas de intrusos |

---

## Configuración

Edita las variables al principio de `src/sistema.py`:

```python
CAMARA_INDEX      = 0      # índice de la cámara (0 o 1)
UMBRAL_COSENO     = 0.5    # umbral de reconocimiento (0-1, más alto = más estricto)
NTFY_CANAL        = "SistemaSeguridad2026"  # tu canal de ntfy
COOLDOWN_SEGUNDOS = 10     # segundos mínimos entre alertas del mismo intruso
```

---

## Resultados del clasificador de zonas

| Métrica | Valor |
|---|---|
| Accuracy | 95.1% |
| Error rate | 0.048 |
| Épocas entrenadas | 12 (early stopping) |
| Arquitectura | ConvNeXt-Small (ImageNet-22k) |
| Regularización | MixUp + Label Smoothing |

---

## Demo

El clasificador de zonas está disponible públicamente en HuggingFace:

👉 [huggingface.co/mynorhm/security-room-classifier](https://huggingface.co/mynorhm/security-room-classifier)
