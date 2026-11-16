FROM python:3.11-slim

WORKDIR /app

ARG RERANKER_MODEL=BAAI/bge-reranker-base
ARG RERANKER_REVISION=2cfc18c9415c912f9d8155881c133215df768a70
ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
ENV HF_HOME=/opt/huggingface
ENV RERANKER_MODEL=${RERANKER_MODEL}
ENV RERANKER_REVISION=${RERANKER_REVISION}
ENV RERANKER_LOCAL_FILES_ONLY=true
ENV EMBEDDING_MODEL=${EMBEDDING_MODEL}
ENV FASTEMBED_CACHE_DIR=/opt/fastembed
ENV FASTEMBED_LOCAL_FILES_ONLY=true

# Prevent Python from writing pyc files to disc and stdout buffering
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "import os; from huggingface_hub import snapshot_download; snapshot_download(repo_id=os.environ['RERANKER_MODEL'], revision=os.environ['RERANKER_REVISION'])"
RUN python -c "import os; from fastembed import TextEmbedding; TextEmbedding(model_name=os.environ['EMBEDDING_MODEL'], cache_dir=os.environ['FASTEMBED_CACHE_DIR'])"

COPY ./app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]