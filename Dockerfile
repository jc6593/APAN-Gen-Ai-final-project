FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    REVIEWLENS_BUNDLE_DIR=/app/demo_bundle_v2

WORKDIR /app

RUN addgroup --system reviewlens \
    && adduser --system --ingroup reviewlens --no-create-home reviewlens

COPY requirements-api.txt ./requirements-api.txt
RUN python -m pip install --no-cache-dir -r requirements-api.txt

COPY --chown=reviewlens:reviewlens app ./app
COPY --chown=reviewlens:reviewlens demo_bundle_v2 ./demo_bundle_v2

USER reviewlens

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).read()"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
