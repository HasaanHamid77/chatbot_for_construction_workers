FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all code files
COPY handler.py .
COPY rag_service.py .
COPY main.py .

# Copy documents (add your PDFs/DOCX here)
COPY documents/ ./documents/

# Expose port for FastAPI
EXPOSE 8000

# Run FastAPI server
CMD ["python", "main.py"]