# ---------- Base ----------
FROM python:3.11-slim

# system deps (uvicorn speedups, build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy requirements first (better caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# now copy the whole project
COPY . /app

# security: python unbuffered + no pyc
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# FastAPI listens on 0.0.0.0:8000
EXPOSE 8000

# Start server
CMD ["python", "server.py"]
