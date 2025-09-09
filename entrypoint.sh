#!/bin/bash
set -e

echo "[entrypoint] starting container..."

# Agar embeddings S3 zip url diya gaya hai to download + unzip
if [ -n "$EMBEDDINGS_ZIP_URL" ]; then
  echo "[entrypoint] Downloading embeddings from $EMBEDDINGS_ZIP_URL"
  curl -L "$EMBEDDINGS_ZIP_URL" -o /tmp/embeddings_bundle.zip
  echo "[entrypoint] Unzipping embeddings..."
  unzip -o /tmp/embeddings_bundle.zip -d /app/app/
  echo "[entrypoint] Embeddings ready."
fi

# Server run karo
echo "[entrypoint] launching server.py ..."
exec python server.py
