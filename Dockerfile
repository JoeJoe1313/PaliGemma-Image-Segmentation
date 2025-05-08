FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# COPY ./app /app/app

# RUN mkdir -p /app/config
RUN mkdir -p /models

ENV HF_TOKEN=${HF_TOKEN}
ENV MODEL_PATH="google/paligemma2-3b-mix-448"
ENV MODEL_CACHE_DIR="/models"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
