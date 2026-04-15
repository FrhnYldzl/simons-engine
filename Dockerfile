FROM python:3.12-slim

WORKDIR /app

# Sistem bagimliliklari (hmmlearn C extension icin gcc gerekli)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Python bagimliliklari
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama dosyalari
COPY server/ ./server/

# PYTHONPATH ayarla
ENV PYTHONPATH=/app/server

# Railway PORT env var inject eder
ENV PORT=8080
EXPOSE 8080

# Calisma dizini
WORKDIR /app/server

# Baslat — /bin/sh ile PORT env var okur
CMD ["/bin/sh", "-c", "python -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
