# Usar imagen base de Python
FROM python:3.11-slim

# Establecer directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar dependencias de Python
COPY api/requirements.txt .
COPY web/requirements.txt ./web_requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r web_requirements.txt

# Copiar el código de la API
COPY api/ .

# Copiar el código de la interfaz web
COPY web/ ./web/

# Copiar los modelos entrenados
COPY models/ ./models/

# Crear directorio para logs (opcional)
RUN mkdir -p /app/logs

# Exponer puertos
EXPOSE 8000
EXPOSE 8501

# Comando para ejecutar la aplicación
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]