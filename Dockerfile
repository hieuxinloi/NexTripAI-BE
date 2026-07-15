FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app
RUN addgroup --system nextrip && adduser --system --ingroup nextrip nextrip
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
USER nextrip
EXPOSE 8000
CMD ["sh", "-c", "uvicorn src.app:app --host 0.0.0.0 --port ${PORT}"]
