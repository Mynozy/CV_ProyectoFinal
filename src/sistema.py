import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import cv2
import numpy as np
import pickle
import time
from datetime import datetime
import urllib.request
import json
import torch
from fastai.vision.all import *
from torchvision import transforms
from PIL import Image as PILImage
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis

# --- configuración ---
# cambia estos valores según tu setup
CAMARA_INDEX      = 0
UMBRAL_COSENO     = 0.5
WHITELIST_PATH    = Path("whitelist/whitelist.pkl")
ALERTS_PATH       = Path("alerts_log")
NTFY_CANAL        = "SistemaSeguridad2026"
COOLDOWN_SEGUNDOS = 10

# --- cargo InsightFace para reconocimiento facial ---
insight = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
insight.prepare(ctx_id=-1, det_size=(640, 640))
print("✅ InsightFace cargado")

# --- cargo la whitelist de personas autorizadas ---
with open(WHITELIST_PATH, "rb") as f:
    whitelist = pickle.load(f)
print(f"✅ Whitelist cargada: {list(whitelist.keys())}")

# --- cargo el clasificador de zonas entrenado con fastai ---
# modelo ConvNeXt-Small fine-tuned en MIT Indoor Scenes (8 clases domésticas)
learn_zonas = torch.load("models/full_model.pth", map_location="cpu", weights_only=False)
learn_zonas.eval()

with open("models/clases.json") as f:
    clases_zonas = json.load(f)

# mismas transformaciones que usé en entrenamiento
tfms_zonas = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])
print(f"✅ Clasificador de zonas cargado: {clases_zonas}")


# --- funciones del pipeline ---

def reconocer(frame):
    """detecto caras en el frame y comparo contra la whitelist"""
    faces      = insight.get(frame)
    resultados = []

    for face in faces:
        emb          = face.embedding.reshape(1, -1)
        mejor_nombre = "Intruso"
        mejor_score  = 0.0

        for nombre, emb_ref in whitelist.items():
            score = cosine_similarity(emb, emb_ref.reshape(1, -1))[0][0]
            if score > mejor_score:
                mejor_score  = score
                mejor_nombre = nombre if score >= UMBRAL_COSENO else "Intruso"

        resultados.append({
            "nombre"    : mejor_nombre,
            "score"     : mejor_score,
            "bbox"      : face.bbox.astype(int),
            "autorizado": mejor_nombre != "Intruso"
        })

    return resultados


def clasificar_zona(frame):
    """clasifico en qué zona de la casa está el intruso usando el modelo entrenado"""
    img    = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor = tfms_zonas(img).unsqueeze(0)

    with torch.no_grad():
        probs = torch.softmax(learn_zonas(tensor), dim=1)[0]

    idx  = probs.argmax().item()
    return clases_zonas[idx], probs[idx].item()


def dibujar(frame, resultados, zona=None):
    """dibujo bboxes, etiquetas y zona detectada sobre el frame"""
    out = frame.copy()

    for r in resultados:
        x1, y1, x2, y2 = r["bbox"]
        color = (0, 255, 0) if r["autorizado"] else (0, 0, 255)
        label = f"{r['nombre']} ({r['score']:.2f})"

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
        cv2.putText(out, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # muestro la zona en la esquina superior izquierda en amarillo
    if zona:
        cv2.putText(out, f"Zona: {zona.upper()}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    return out


def enviar_alerta_ntfy(nombre, score, zona):
    """mando notificación push al móvil incluyendo la zona detectada"""
    # configuración de ntfy:
    # 1. descarga la app ntfy en tu móvil
    # 2. suscríbete al canal con el nombre que definiste arriba
    # 3. recibirás una notificación cada vez que se detecte un intruso
    try:
        ts  = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        req = urllib.request.Request(
            f"https://ntfy.sh/{NTFY_CANAL}",
            data=f"Cara no reconocida en {zona.upper()} a las {ts} — Score: {score:.3f}".encode("utf-8"),
            headers={
                "Title"   : f"Intruso en {zona.upper()}",
                "Priority": "high",
                "Tags"    : "rotating_light,camera"
            },
            method="POST"
        )
        urllib.request.urlopen(req)
        print(f"📱 Notificación enviada — {zona.upper()}")
    except Exception as e:
        print(f"❌ Error ntfy: {e}")


def guardar_alerta(frame, nombre, score, resultados):
    """guardo el frame del intruso con bbox, label y zona en alerts_log"""
    zona, zona_score = clasificar_zona(frame)
    ts               = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname            = ALERTS_PATH / f"alerta_{ts}_{nombre}_{zona}.jpg"

    frame_viz = dibujar(frame, resultados, zona)
    cv2.imwrite(str(fname), frame_viz)

    print(f"🚨 ALERTA — {nombre} detectado en {zona.upper()} (zona score: {zona_score:.2f})")
    enviar_alerta_ntfy(nombre, score, zona)
    return fname


# --- loop principal ---
print("🎥 Sistema de seguridad arrancado — Q para salir")

cap           = cv2.VideoCapture(CAMARA_INDEX)
ultimo_alerta = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    resultados = reconocer(frame)
    ahora      = time.time()

    for r in resultados:
        if not r["autorizado"]:
            ultimo = ultimo_alerta.get(r["nombre"], 0)
            if ahora - ultimo > COOLDOWN_SEGUNDOS:
                guardar_alerta(frame, r["nombre"], r["score"], resultados)
                ultimo_alerta[r["nombre"]] = ahora

    # muestro la zona en tiempo real aunque no haya intruso
    zona, _ = clasificar_zona(frame)
    frame_viz = dibujar(frame, resultados, zona)
    cv2.imshow("Sistema de Seguridad", frame_viz)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("👋 Cerrando sistema...")
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Sistema cerrado")