FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV HF_TOKEN=${HF_TOKEN}
ENV MODEL_ID="google/paligemma2-3b-mix-448"
ENV MODELS_DIR="/app/models"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
