# src/data/make_dataset.py
import os
from pathlib import Path
import requests
from tqdm import tqdm
import boto3
from botocore.client import Config

BASE = "https://raw.githubusercontent.com/btphan95/greenr-airflow/refs/heads/master/data"
LABELS = ["dandelion", "grass"]
N = 200  # download 00000000.jpg .. 00000199.jpg

def iter_urls():
    for label in LABELS:
        for i in range(N):
            yield f"{BASE}/{label}/{i:08d}.jpg", label

def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("S3_ENDPOINT"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        config=Config(signature_version="s3v4"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )

def ensure_bucket(s3, bucket: str):
    # create bucket if missing
    buckets = [b["Name"] for b in s3.list_buckets().get("Buckets", [])]
    if bucket not in buckets:
        s3.create_bucket(Bucket=bucket)

def main():
    # local raw folder (mounted on host)
    root = Path("data/raw")
    root.mkdir(parents=True, exist_ok=True)

    bucket = os.getenv("S3_BUCKET", "plants")
    s3 = s3_client()
    ensure_bucket(s3, bucket)

    # download + upload
    for url, label in tqdm(list(iter_urls()), desc="Downloading"):
        dest = root / label
        dest.mkdir(parents=True, exist_ok=True)
        fname = dest / url.split("/")[-1]

        # download if not present
        if not fname.exists():
            try:
                r = requests.get(url, timeout=20)
                r.raise_for_status()
                fname.write_bytes(r.content)
            except Exception as e:
                print("Skip:", url, e)
                continue

        # upload to S3
        key = f"images/{label}/{fname.name}"
        with open(fname, "rb") as f:
            s3.upload_fileobj(f, bucket, key)

    print("✅ Done: local data in data/raw/* and S3 objects in", f"s3://{bucket}/images/*")

if __name__ == "__main__":
    main()