FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy config and Python requirements
COPY requirements.txt config.py /app/
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download and cache models to make the image run completely offline
RUN python -c "import config; from sentence_transformers import SentenceTransformer; SentenceTransformer(config.EMBEDDING_MODEL_NAME)"
RUN python -c "import config; from transformers import T5ForConditionalGeneration, T5Tokenizer; T5Tokenizer.from_pretrained(config.PARSING_MODEL_NAME); T5ForConditionalGeneration.from_pretrained(config.PARSING_MODEL_NAME)"

# Copy application source code and static assets
COPY rank.py /app/
COPY app.py /app/
COPY src/ /app/src/
COPY data/ /app/data/
COPY static/ /app/static/

# Expose FastAPI port
EXPOSE 8501

# Healthcheck to verify the web app status
HEALTHCHECK CMD curl --fail http://localhost:8501/ || exit 1

# Default command to run Uvicorn
ENTRYPOINT ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8501"]

