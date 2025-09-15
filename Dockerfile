# ---------- Base ----------
FROM python:3.11-slim

# system deps (uvicorn speedups, build tools, and utilities for unzipping)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl unzip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- New Part: Add an argument for your S3 URL ---
ARG EMBEDDINGS_ZIP_URL

# --- New Part: Download and unzip embeddings during the build ---
# This runs only when the image is built, not when the container starts.
RUN if [ -n "$EMBEDDINGS_ZIP_URL" ]; then \
      echo "Downloading embeddings from $EMBEDDINGS_ZIP_URL..."; \
      curl -L "$EMBEDDINGS_ZIP_URL" -o /tmp/embeddings_bundle.zip && \
      echo "Unzipping embeddings to /app/app/"; \
      unzip -o /tmp/embeddings_bundle.zip -d /app/app/ && \
      rm /tmp/embeddings_bundle.zip && \
      echo "Embeddings are ready."; \
    else \
      echo "Warning: EMBEDDINGS_ZIP_URL not provided during build."; \
    fi

# copy requirements first (better caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# now copy the whole project over the unzipped embeddings
COPY . /app

# security: python unbuffered + no pyc
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Container को env से port मिलेगा
# Container को env से port मिलेगा
ENV PORT=8080

EXPOSE 8080
CMD ["python3", "server.py"]
