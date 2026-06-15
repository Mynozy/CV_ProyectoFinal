# app pública de Gradio — clasificador de zonas
import os
import gradio as gr
import torch
import json
from PIL import Image
import torchvision.transforms as transforms
from fastai.vision.all import *
from huggingface_hub import hf_hub_download
import shutil

# descargo el modelo desde HuggingFace si no existe localmente
os.makedirs("models", exist_ok=True)

if not os.path.exists("models/full_model.pth"):
    print("Descargando modelo desde HuggingFace...")
    ruta = hf_hub_download(
        repo_id="mynorhm/security-room-classifier",
        filename="full_model.pth",
        repo_type="model"
    )
    shutil.copy(ruta, "models/full_model.pth")
    print("Modelo descargado")

if not os.path.exists("models/clases.json"):
    ruta = hf_hub_download(
        repo_id="mynorhm/security-room-classifier",
        filename="clases.json",
        repo_type="model"
    )
    shutil.copy(ruta, "models/clases.json")

# cargo el modelo y las clases
learn_zonas = torch.load("models/full_model.pth", map_location="cpu", weights_only=False)
learn_zonas.eval()

with open("models/clases.json") as f:
    clases = json.load(f)

tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def clasificar_zona(imagen):
    img    = Image.fromarray(imagen).convert("RGB")
    tensor = tfms(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(learn_zonas(tensor), dim=1)[0]
    return dict(zip(clases, map(float, probs)))

demo = gr.Interface(
    fn=clasificar_zona,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=4),
    title="Security Room Classifier",
    description="Sube una imagen e identifica en que zona de la casa fue tomada.",
)

demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))