import io
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

from model import get_model

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "/app/checkpoints/classifier_v1.pt")
ARCHITECTURE = os.environ.get("MODEL_ARCHITECTURE", "resnet18")
NUM_CLASSES = int(os.environ.get("NUM_CLASSES", "10"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

inference_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616],
    ),
])

app = FastAPI(title="CIFAR-10 Classifier")

model = None
model_load_error = None


@app.on_event("startup")
def load_model():
    global model, model_load_error
    try:
        checkpoint_path = Path(CHECKPOINT_PATH)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        m = get_model(architecture=ARCHITECTURE, num_classes=NUM_CLASSES)
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        m.load_state_dict(checkpoint["model_state_dict"])
        m.to(DEVICE)
        m.eval()

        model = m
        print(f"Model loaded from {checkpoint_path} (val_accuracy={checkpoint.get('val_accuracy')})")
    except Exception as e:
        model_load_error = str(e)
        print(f"Model failed to load: {model_load_error}")


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {model_load_error}")
    return {"status": "ok"}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    tensor = inference_transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = F.softmax(logits, dim=1).squeeze(0)

    predicted_idx = int(probabilities.argmax())
    return {
        "predicted_class": CIFAR10_CLASSES[predicted_idx],
        "confidence": round(float(probabilities[predicted_idx]), 4),
        "probabilities": {
            CIFAR10_CLASSES[i]: round(float(p), 4)
            for i, p in enumerate(probabilities)
        },
    }
