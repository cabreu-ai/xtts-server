FROM python:3.10-slim

WORKDIR /app

# Dependencias del sistema necesarias para TTS
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    libgl1 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Evitar prompts interactivos de Coqui
ENV COQUI_TOS_AGREED=1

# Instalar dependencias Python con versiones compatibles
RUN pip install --no-cache-dir \
    torch==2.1.2 \
    torchaudio==2.1.2 \
    transformers==4.36.2 \
    TTS==0.22.0 \
    fastapi \
    uvicorn \
    python-multipart \
    jinja2

# Copiar aplicación
COPY . /app

# Crear carpetas persistentes
RUN mkdir -p /app/voices /app/outputs

# Puerto del servidor
EXPOSE 8000

# Arrancar API
CMD ["uvicorn","server:app","--host","0.0.0.0","--port","8000"]