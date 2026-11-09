# Label review website - CPU only, no model
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PORT=5080

RUN pip install flask==3.0.2 waitress pillow pillow_heif tifffile imagecodecs numpy

WORKDIR /app
COPY app.py ./
COPY static ./static
RUN mkdir -p uploads labels

EXPOSE 5080
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5080/api/config', timeout=8)"
CMD ["python", "app.py"]
