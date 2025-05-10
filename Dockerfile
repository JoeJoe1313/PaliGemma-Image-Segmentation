FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV MODEL_ID=google/paligemma2-3b-mix-448
ENV MODELS_DIR=/app/models
ENV TARGET_WIDTH=448
ENV TARGET_HEIGHT=448

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
