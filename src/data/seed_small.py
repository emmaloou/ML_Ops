# src/data/seed_small.py
import requests
from pathlib import Path

BASE = "https://raw.githubusercontent.com/btphan95/greenr-airflow/refs/heads/master/data"
TARGET = Path("data/raw")
TARGET.mkdir(parents=True, exist_ok=True)

samples = {"dandelion": [0,1,2,3,4], "grass": [0,1,2,3,4]}

for label, ids in samples.items():
    (TARGET / label).mkdir(parents=True, exist_ok=True)
    for i in ids:
        url = f"{BASE}/{label}/{i:08d}.jpg"
        dst = TARGET / label / f"{i:08d}.jpg"
        if dst.exists():
            continue
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        dst.write_bytes(r.content)

print("✅ Seeded:", TARGET.resolve())