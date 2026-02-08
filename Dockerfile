FROM python:3.10-slim

WORKDIR /app

# Install only what is needed
RUN pip install --no-cache-dir \
    runpod \
    torch \
    transformers \
    accelerate

COPY handler.py .

CMD ["python", "-u", "handler.py"]
