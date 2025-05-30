# trace-processor/Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY consumer.py .
COPY pdf_to_text.py .

CMD ["python", "consumer.py"]