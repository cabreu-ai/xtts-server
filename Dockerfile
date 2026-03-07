FROM python:3.10-slim

WORKDIR /app

# Dependencias del sistema
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    libgl1 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Evitar prompt de licencia de Coqui
ENV COQUI_TOS_AGREED=1

# Instalar deps Python compatibles con XTTS
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copiar aplicación
COPY server.py /app/server.py
COPY templates /app/templates

# Carpetas persistentes
RUN mkdir -p /app/voices /app/outputs

EXPOSE 8000

CMD ["uvicorn","server:app","--host","0.0.0.0","--port","8000"]