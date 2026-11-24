FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY *.py ./
COPY static ./static
# Knowledge base de exemplo; o compose.yaml monta ./knowledge por cima.
COPY knowledge ./knowledge
ENV DB_PATH=/app/data/chat.sqlite \
    KB_PATH=/app/data/kb.json \
    KB_DIR=/app/knowledge
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
