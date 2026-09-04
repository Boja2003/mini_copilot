# Image de prod : la meme sur ta machine et sur le serveur.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Les dependances d'abord, le code ensuite : tant que requirements.txt ne
# change pas, Docker reutilise le cache et le rebuild prend 2 secondes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Ne jamais tourner en root dans un conteneur.
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

# $PORT : Fly.io et Railway imposent le port a l'execution.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
