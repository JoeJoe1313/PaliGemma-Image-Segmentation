
FROM python:3.10-slim


WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

EXPOSE 8000

ENV HF_TOKEN=${HF_TOKEN}

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
