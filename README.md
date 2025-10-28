# Flower Classifier (Dandelion vs Grass)

## Setup

```bash
pip install -r requirements.txt

streamlit run streamlit_app.py
```

## Optional: MinIO Object Storage

```bash
docker run -d \
  -p 9000:9000 \
  -p 9001:9001 \
  --name minio \
  -e "MINIO_ROOT_USER=minioadmin" \
  -e "MINIO_ROOT_PASSWORD=minioadmin" \
  minio/minio server /data --console-address ":9001"
```
