"""Lancement du serveur en local.

Sous Windows, utilise ce script plutot que la commande `uvicorn` directe :
il installe la bonne boucle d'evenements avant de demarrer le serveur (voir
app/boucle.py). En production, c'est le Dockerfile qui lance uvicorn, sur
Linux, ou le probleme n'existe pas.

    python run.py
"""

import uvicorn

from app.boucle import configurer

if __name__ == "__main__":
    configurer()
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
