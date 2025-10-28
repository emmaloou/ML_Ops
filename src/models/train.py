# src/models/train.py
import os
from pathlib import Path
import time

import mlflow
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

import boto3
from botocore.client import Config

# --- env / paths ---
DATA_DIR = Path("data/raw")                 # filled by make_dataset.py
MODEL_DIR = Path("models"); MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "model.pth"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_EXPERIMENT   = os.getenv("MLFLOW_EXPERIMENT", "plants_cls")

S3_ENDPOINT = os.getenv("S3_ENDPOINT")
S3_BUCKET   = os.getenv("S3_BUCKET", "plants")
AWS_KEY     = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET  = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION  = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=AWS_KEY,
        aws_secret_access_key=AWS_SECRET,
        config=Config(signature_version="s3v4"),
        region_name=AWS_REGION,
    )

def upload_model_to_s3(local_path: Path, key: str):
    s3 = s3_client()
    # ensure bucket exists
    buckets = [b["Name"] for b in s3.list_buckets().get("Buckets", [])]
    if S3_BUCKET not in buckets:
        s3.create_bucket(Bucket=S3_BUCKET)
    with open(local_path, "rb") as f:
        s3.upload_fileobj(f, S3_BUCKET, key)
    return f"s3://{S3_BUCKET}/{key}"

def get_loaders(img_size=224, batch_size=32, val_split=0.2):
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    ds = datasets.ImageFolder(str(DATA_DIR), transform=tfm)
    n_val = int(len(ds) * val_split)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0),
        ds.classes
    )

def train_one_epoch(model, loader, device, optimizer, criterion):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward(); optimizer.step()
        loss_sum += loss.item() * y.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return loss_sum/total, correct/total

@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * y.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return loss_sum/total, correct/total

def main():
    # MLflow setup
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    device = torch.device("cpu")
    train_loader, val_loader, classes = get_loaders()
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    with mlflow.start_run(run_name=f"mobilenet_v2_{int(time.time())}") as run:
        mlflow.log_param("arch", "mobilenet_v2")
        mlflow.log_param("epochs", 2)
        mlflow.log_param("batch_size", 32)
        mlflow.log_param("img_size", 224)
        mlflow.log_param("classes", classes)

        for epoch in range(1, 2+1):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, device, optimizer, criterion)
            va_loss, va_acc = evaluate(model, val_loader, device, criterion)
            mlflow.log_metric("train_loss", tr_loss, step=epoch)
            mlflow.log_metric("train_acc",  tr_acc,  step=epoch)
            mlflow.log_metric("val_loss",   va_loss, step=epoch)
            mlflow.log_metric("val_acc",    va_acc,  step=epoch)
            print(f"Epoch {epoch}: train_acc={tr_acc:.3f} val_acc={va_acc:.3f}")

        # save & log
        torch.save(model.state_dict(), MODEL_PATH)
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="artifacts")

        # upload to MinIO
        s3_key = "models/latest/model.pth"
        s3_uri = upload_model_to_s3(MODEL_PATH, s3_key)
        mlflow.log_param("model_s3_uri", s3_uri)
        print("Saved to:", MODEL_PATH, "and", s3_uri)

if __name__ == "__main__":
    main()