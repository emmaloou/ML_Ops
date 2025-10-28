import os, io
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import torch
from torchvision import models, transforms

classes = ["dandelion", "grass"]
img_size = 224
MODEL_PATH = Path(os.getenv("MODEL_LOCAL_PATH", "./models/model.pth"))

app = FastAPI(title="Plants Classifier API")

_model = None
_tfm = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def _load_model_from_disk():
    global _model
    if not MODEL_PATH.exists():
        return False
    m = models.mobilenet_v2(weights=None)
    m.classifier[1] = torch.nn.Linear(m.last_channel, 2)
    state = torch.load(MODEL_PATH, map_location="cpu")
    m.load_state_dict(state)
    m.eval()
    _model = m
    return True

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_found": MODEL_PATH.exists(),
        "model_loaded": _model is not None,
        "model_path": str(MODEL_PATH),
    }

@app.post("/admin/load_model")
def admin_load_model():
    if _load_model_from_disk():
        return {"loaded": True, "path": str(MODEL_PATH)}
    raise HTTPException(status_code=404, detail=f"Model file not found at {MODEL_PATH}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if _model is None:
        # try auto-load once
        if not _load_model_from_disk():
            raise HTTPException(status_code=503, detail="Model not loaded yet. Train first, then call /admin/load_model.")
    content = await file.read()
    img = Image.open(io.BytesIO(content)).convert("RGB")
    x = _tfm(img).unsqueeze(0)
    with torch.no_grad():
        logits = _model(x)
        probs = torch.softmax(logits, dim=1).flatten().tolist()
    idx = int(torch.argmax(logits, dim=1))
    return {"label": classes[idx], "proba": float(probs[idx])}