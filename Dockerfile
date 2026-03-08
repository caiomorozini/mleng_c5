FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./

RUN pip install --no-cache-dir uv && \
    uv pip install --system -r pyproject.toml

COPY app/ ./app/
COPY src/ ./src/

COPY models/ ./models/

EXPOSE 8000

ENV MODEL_PATH=/app/models/predictor.keras
ENV SCALER_PATH=/app/models/scaler.pkl
ENV CONFIG_PATH=/app/models/model_config.json
ENV PYTHONPATH=/app

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
