FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY handler.py .
COPY rag_service.py .

# Copy documents (add your PDFs/DOCX here)
COPY documents/ ./documents/

# RunPod serverless will call handler.py
CMD ["python", "-u", "handler.py"]
```

---

## **FILE 5: .dockerignore**
```
__pycache__
*.pyc
.git
.env
*.md