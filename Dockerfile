FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.cache/huggingface
ENV STREAMLIT_SERVER_HEADLESS=true

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt .

RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements-docker.txt

COPY app ./app
COPY scripts ./scripts
COPY data/rag_corpus ./data/rag_corpus

RUN mkdir -p reports models/rag && \
    python scripts/rag/build_vector_db.py

EXPOSE 8501

CMD ["streamlit", "run", "app/rag_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
