# mini_copilot — étape 1

**En ligne : https://mini-copilot.fly.dev** — `/health` est public,
`/chat` exige le header `X-API-Key`.

Un endpoint `/chat` qui interroge directement l'API Mistral, conteneurisé et
prêt à déployer. Pas encore de RAG : c'est délibéré. On met d'abord en ligne
un service trivial, pour que « déployer » soit un problème déjà résolu quand
on attaquera le retrieval.

## Le trajet d'une requête

```
POST /chat ──► main.py        clé X-API-Key, validation Pydantic, codes HTTP
                  │
                  ▼
              llm.py          orchestre le RAG
                  │
                  ├──► retrieval.py ──► embeddings.py   question → vecteur
                  │         │                            (mistral-embed)
                  │         └──► db.py   pgvector : ORDER BY embedding <=> ?
                  │                      → les 5 passages les plus proches
                  │
                  └──► mistral.py       prompt = passages + question → réponse
```

Ingestion (hors ligne, `python -m app.ingest`) :

```
fichier ──► parsers.py ──► chunking.py ──► embeddings.py ──► db.py
            pdf / pptx     ~1200 car.      vecteurs 1024D    documents
            md / txt       + recouvrement  par lots de 32    + chunks
```

| Fichier | Rôle |
|---|---|
| `app/main.py` | Couche HTTP. Valide, délègue, traduit les pannes en codes HTTP. |
| `app/llm.py` | Orchestration du RAG : chercher les passages, puis répondre. |
| `app/retrieval.py` | Recherche vectorielle (`<=>` de pgvector). |
| `app/embeddings.py` | Texte → vecteurs, par lots, avec back-off sur quota. |
| `app/parsers.py` | Un parser par format + normalisation du texte. |
| `app/chunking.py` | Découpage en passages indexables. |
| `app/ingest.py` | Pipeline d'ingestion, en ligne de commande. |
| `app/db.py` | Postgres + pgvector, SQL écrit à la main. |
| `app/mistral.py` | Client HTTP partagé vers Mistral. |
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

2. Le modèle répond. `/chat` exige le header `X-API-Key` (valeur =
   `API_KEY` de ton `.env`) ; `/health` reste public car Fly.io l'interroge :
   ```bash
   curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -H "X-API-Key: TA_CLE_DE_SERVICE" -d "{\"question\":\"Explique-moi ce qu'est un embedding en deux phrases.\"}"
   ```
   Sans le header, tu dois obtenir un `401` — et l'appel ne part jamais
   chez Mistral, donc ton quota est protégé.

3. Plus simple que curl : ouvre **http://localhost:8000/docs**. FastAPI
   génère une interface de test interactive à partir de tes modèles Pydantic.

## Le corpus

```bash
docker compose up -d db
```

```bash
python -m app.ingest "C:/chemin/vers/tes/cours"
```

Formats lus : `.pdf`, `.pptx`, `.md`, `.txt`, `.rst`. Un format inconnu est
signalé et ignoré, pas une erreur fatale. Ajouter le `.docx` demain = écrire
une fonction et l'inscrire dans `PARSERS`, rien d'autre.

L'ingestion est **rejouable** : une empreinte SHA-256 par fichier permet de
ne retraiter que ce qui a changé. `--force` réindexe tout.

### Ce que ce corpus a appris

Relevé sur 15 documents (6 PDF, 9 PPTX) → **1076 chunks** :

- **Les ligatures des PDF LaTeX cassent la recherche.** `final` y est encodé
  par le caractère unique U+FB01 : sans normalisation, il ne matche jamais
  une recherche sur `final`.
- **Postgres refuse les octets NUL** que produit l'extraction PDF. Nettoyés
  à la normalisation.
- **La recherche vectorielle seule rate les noms propres.** « Quelles sont
  les étapes de l'algorithme de Welsh & Powell ? » ne remontait PAS le cours
  sur Welsh & Powell : la tournure « étapes de l'algorithme » domine le
  vecteur et attire le pseudo-code de Floyd-Warshall. Reformulée en
  « algorithme Welsh Powell coloration », la même recherche trouvait le bon
  passage à 0.097. D'où la **recherche hybride** : vectoriel + mots-clés,
  fusionnés par RRF. Le vectoriel comprend le sens, les mots-clés ne ratent
  jamais un terme rare.
- **Le seuil de distance ne détecte pas le hors-sujet.** Mesuré : questions
  du cours 0.145–0.286, questions hors-sujet 0.272–0.396 — **les deux
  distributions se chevauchent**, aucun seuil ne sépare proprement. Le
  filtre de distance n'est donc qu'un garde-fou grossier ; c'est le prompt
  système qui interdit au modèle de répondre hors des passages fournis.
- **Les tableaux et les notes de présentateur des `.pptx`** portent souvent
  l'essentiel du cours : ils sont extraits, pas seulement les titres.

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

## Mise en ligne (Fly.io)

Fly construit l'image **sur ses serveurs** : Docker Desktop n'a pas besoin
de tourner. La configuration est dans [`fly.toml`](fly.toml).

Prérequis, une seule fois :

```bash
powershell -Command "iwr https://fly.io/install.ps1 -useb | iex"
```

Puis, dans un nouveau terminal :

```bash
fly auth signup
```

(ou `fly auth login` si tu as déjà un compte). Ensuite :

```bash
fly launch --no-deploy
```

`fly launch` détecte le `Dockerfile` et le `fly.toml` existant. Si le nom
`mini-copilot` est déjà pris sur Fly, il te proposera d'en choisir un autre.

**Les secrets, avant le premier déploiement.** Ils ne sont jamais dans
`fly.toml`, qui est versionné — ils vivent dans le coffre de Fly, chiffrés,
injectés comme variables d'environnement au démarrage :

```bash
fly secrets set MISTRAL_API_KEY=... API_KEY=...
```

Reprends les deux valeurs de ton `.env` local. Puis :

```bash
fly deploy
```

### Vérifier le déploiement

```bash
curl https://mini-copilot.fly.dev/health
```

```bash
curl -X POST https://mini-copilot.fly.dev/chat -H "Content-Type: application/json" -H "X-API-Key: TA_CLE_DE_SERVICE" -d "{\"question\":\"Dis bonjour.\"}"
```

Sans le header, `401` : c'est le comportement voulu, l'endpoint est public
mais pas ouvert.

Commandes utiles : `fly logs` (flux en direct), `fly status` (état des
machines), `fly secrets list` (noms des secrets, jamais leurs valeurs).

### Coût et réveil à froid

`auto_stop_machines = "stop"` et `min_machines_running = 0` : la machine
s'éteint faute de trafic et redémarre à la requête suivante. Zéro coût au
repos, contre quelques secondes de réveil à froid sur le premier appel.

## Suite

- [x] **Étape 1** — squelette FastAPI + `/chat` + Docker
- [x] **Étape 1b** — déployé sur Fly.io (région cdg, scale-to-zero)
- [x] **Étape 2** — ingestion multi-format + pgvector + retrieval
- [ ] **Étape 3** — Langfuse pour tracer chaque appel LLM
- [ ] **Étape 4** — Ragas + golden set pour mesurer le retrieval
- [ ] **Étape 5** — couche agent (LangGraph) pour le multi-étapes
- [ ] **Étape 6** — CD, cache Redis, modèle auto-hébergé (vLLM / Ollama)
