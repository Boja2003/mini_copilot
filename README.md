# mini_copilot — étape 1

Un endpoint `/chat` qui interroge directement l'API Mistral, conteneurisé et
prêt à déployer. Pas encore de RAG : c'est délibéré. On met d'abord en ligne
un service trivial, pour que « déployer » soit un problème déjà résolu quand
on attaquera le retrieval.

## Le trajet d'une requête

```
POST /chat  ──►  main.py         validation Pydantic, gestion d'erreur HTTP
                    │
                    ▼
                 llm.py          construit le prompt, POST vers l'API Mistral
                    │                  ▲
                    ▼                  └── étape 2 : le retrieval s'insère ICI
                 config.py       lit la clé API depuis l'environnement
```

| Fichier | Rôle |
|---|---|
| `app/main.py` | Couche HTTP. Valide, délègue, traduit les pannes en codes HTTP. |
| `app/llm.py` | La couture du projet. Tout ce qui parle au LLM. |
| `app/config.py` | Config et secrets, lus depuis l'environnement. |
| `Dockerfile` | Empaquetage : la fin du « ça marche chez moi ». |
| `docker-compose.yml` | Un seul service pour l'instant ; Postgres+pgvector s'ajoutera à l'étape 2. |
| `tests/test_api.py` | Tests sans réseau : le LLM est remplacé par un double. |

## Démarrer en local

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements-dev.txt

cp .env.example .env            # puis colle ta vraie clé Mistral dedans
uvicorn app.main:app --reload
```

### Vérifier que l'étape est réussie

1. Le service est vivant :
   ```bash
   curl http://localhost:8000/health
   ```
   → `{"status":"ok"}`

2. Le modèle répond :
   ```bash
   curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "{\"question\":\"Explique-moi ce qu'est un embedding en deux phrases.\"}"
   ```

3. Plus simple que curl : ouvre **http://localhost:8000/docs**. FastAPI
   génère une interface de test interactive à partir de tes modèles Pydantic.

## Tests et qualité

```bash
pytest -q
ruff check .
```

Les tests ne touchent jamais l'API Mistral (un test qui dépend du réseau est
un pari, pas un test). La CI GitHub Actions rejoue `ruff` + `pytest` + le
build Docker à chaque push.

## Docker

```bash
docker compose up --build
```

## Secrets

`.env` est dans `.gitignore` : ta clé ne partira jamais sur GitHub.
`.env.example` est le modèle versionné, sans secret. En production, la clé
est fournie comme variable d'environnement (`fly secrets set`, ou l'onglet
Variables de Railway) — jamais dans l'image.

## Dépannage

### `502` avec `LLM HTTP 429 ... quota epuise pour CE modele`

**Chez Mistral, le quota est par modèle, pas par compte.** Un compte
parfaitement actif peut avoir 0 req/min sur un modèle et 750 sur un autre.
Ne cherche pas du côté de l'activation du compte : regarde l'en-tête
`x-ratelimit-limit-req-minute`. S'il vaut `0`, change `MISTRAL_MODEL`.

Pour savoir quels modèles te sont ouverts, `GET /v1/models` liste ceux que
ta clé peut voir — mais il ne dit pas lesquels ont du quota. Seul un vrai
POST le révèle. Relevé sur ce compte :

| Modèle | Statut | req/min |
|---|---|---|
| `mistral-large-latest` | 403, hors abonnement | — |
| `mistral-medium-latest` | 429 | 0 |
| `mistral-small-latest` | 429 | 0 |
| `magistral-small-latest` | 429 | 0 |
| `ministral-8b-latest` | OK (défaut du projet) | 188 |
| `open-mistral-nemo` | OK | 188 |
| `open-mistral-7b` | OK | 188 |
| `ministral-3b-latest` | OK | 750 |
| `mistral-embed` | OK (utile à l'étape 2) | 60 |

Un `403 « not available in your subscription »` est différent : le modèle
est hors plan, changer de modèle est la seule option.

### Une réponse « vide », ou un comportement qui ne correspond pas au code

Vérifie d'abord que tu tapes bien sur *ton* serveur, et pas sur un ancien
resté ouvert sur le même port :

```bash
netstat -ano | findstr :8000
```

Plus d'une ligne `LISTENING` = plusieurs serveurs se disputent le port, et
`uvicorn` a pu échouer à démarrer en silence. Ferme les anciens
(`taskkill /PID <pid> /F`) ou lance sur un autre port (`--port 8017`).

Depuis la correction, un contenu vide renvoyé par le LLM ne passe plus en
`200` : il devient un `502 « Le LLM a renvoye une reponse vide »`.

## Suite

- [x] **Étape 1** — squelette FastAPI + `/chat` + Docker
- [ ] **Étape 1b** — déployer sur Fly.io ou Railway pendant que c'est trivial
- [ ] **Étape 2** — ingestion de documents + pgvector + retrieval
- [ ] **Étape 3** — Langfuse pour tracer chaque appel LLM
- [ ] **Étape 4** — Ragas + golden set pour mesurer le retrieval
- [ ] **Étape 5** — couche agent (LangGraph) pour le multi-étapes
- [ ] **Étape 6** — CD, cache Redis, modèle auto-hébergé (vLLM / Ollama)
