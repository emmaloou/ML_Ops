# src/data/sync_from_minio.py
import os
from pathlib import Path
from tqdm import tqdm
import boto3
from botocore.config import Config

BUCKET = os.getenv("S3_BUCKET", "plants")
PREFIX = os.getenv("S3_PREFIX", "images/")   # images/dandelion/* and images/grass/*
ENDPOINT = os.getenv("S3_ENDPOINT_URL", "http://minio:9000")
AK = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
SK = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
LOCAL_ROOT = Path("data/raw")  # what your train.py expects

def main():
    LOCAL_ROOT.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client(
        "s3",
        aws_access_key_id=AK,
        aws_secret_access_key=SK,
        endpoint_url=ENDPOINT,
        config=Config(signature_version="s3v4"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )

    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=BUCKET, Prefix=PREFIX)

    count = 0
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]  # e.g. images/dandelion/00000000.jpg
            if key.endswith("/"):
                continue
            rel = key.split("/", 1)[1]  # drop leading "images/"
            dest = LOCAL_ROOT / rel     # data/raw/dandelion/00000000.jpg
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                s3.download_file(BUCKET, key, str(dest))
                count += 1
    print(f"Synced {count} files to {LOCAL_ROOT.resolve()}")

if __name__ == "__main__":
    main()