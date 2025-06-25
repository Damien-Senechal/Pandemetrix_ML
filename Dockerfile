FROM python:3.9-slim

# Variables d'environnement
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=main.py

# Répertoire de travail
WORKDIR /app

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copie et installation des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code source
COPY . .

# Création des dossiers nécessaires
RUN mkdir -p data/raw data/processed models logs

# Port exposé
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:5000/api/v1/covid/health || exit 1

# Commande de démarrage
CMD ["python", "main.py"]
