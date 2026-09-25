# mini_copilot

**En ligne : https://mini-copilot.fly.dev** — `/health` est public,
`/chat` exige le header `X-API-Key`.
**Code : https://github.com/Boja2003/mini_copilot** — lint, tests et build
Docker rejoués à chaque push.

Un assistant qui répond à partir de mes supports de cours, et uniquement à
partir d'eux : recherche hybride (vecteurs + mots-clés) dans Postgres et
pgvector, réponse générée par Mistral avec ses sources (document et page),
refus explicite quand les cours ne contiennent pas la réponse. Chaque
évolution du retrieval est décidée sur un golden set mesuré, pas à l'œil, et
chaque requête en production laisse une trace détaillée.

## Le trajet d'une requête

```
POST /chat ──► main.py        clé X-API-Key, validation Pydantic, codes HTTP
                  │           ouvre la trace « chat » (observabilite.py)
                  ▼
              llm.py          orchestre le RAG
                  │
                  ├──► retrieval.py      recherche hybride, fusion RRF
                  │       │
                  │       ├──► reecriture.py   (option) question → requêtes
                  │       │                    en anglais technique
                  │       ├──► embeddings.py   requêtes → vecteurs (mistral-embed)
                  │       └──► db.py           pgvector <=> + tsvector
                  │                            → les 5 passages retenus
                  │
                  └──► mistral.py       prompt = passages + question → réponse
```

Chaque étape de ce trajet devient une étape de la trace : durée, entrées,
sorties et tokens consommés (voir « Observabilité »).

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
| `app/retrieval.py` | Recherche hybride (vecteurs + mots-clés), fusion RRF. |
| `app/reecriture.py` | Réécriture de requête : français → anglais technique. |
| `app/embeddings.py` | Texte → vecteurs, par lots, avec back-off sur quota. |
| `app/parsers.py` | Un parser par format + normalisation du texte. |
| `app/chunking.py` | Découpage en passages indexables. |
| `app/ingest.py` | Pipeline d'ingestion, en ligne de commande. |
| `app/db.py` | Postgres + pgvector, SQL écrit à la main. |
| `app/mistral.py` | Client HTTP partagé et appel chat vers Mistral. |
| `app/observabilite.py` | Traces Langfuse : le seul module qui parle au SDK. |
| `app/config.py` | Config et secrets, lus depuis l'environnement. |
| `eval/` | Golden set, métriques, mesures et synthèse. |
| `Dockerfile` | Empaquetage : la fin du « ça marche chez moi ». |
| `docker-compose.yml` | L'API et sa base Postgres + pgvector. |
| `tests/` | 74 tests sans réseau : LLM, embeddings, base et traces doublés. |

## Démarrer en local

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements-dev.txt

cp .env.example .env            # puis colle tes vraies clés dedans
docker compose up -d db
python run.py
```

Sous Windows, `python run.py` plutôt que `uvicorn` directement : psycopg en
mode asynchrone exige une boucle d'événements que Windows n'utilise pas par
défaut (voir `app/boucle.py`). Attention, c'est plus subtil qu'il n'y paraît :
uvicorn choisit lui-même sa boucle, et ne retient la bonne que parce que
`run.py` active le rechargement automatique, qui le fait tourner dans un
sous-processus. En production, sur Linux, la question ne se pose pas.

### Vérifier que ça marche

1. Le service est vivant :
   ```bash
   curl http://localhost:8000/health
   ```
   → `{"status":"ok"}`

2. Le modèle répond. `/chat` exige le header `X-API-Key` (valeur =
   `API_KEY` de ton `.env`) ; `/health` reste public car Fly.io l'interroge :
   ```bash
   curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -H "X-API-Key: TA_CLE_DE_SERVICE" -d "{\"question\":\"Comment fonctionne l'algorithme de Dijkstra ?\"}"
   ```
   Sans le header, tu dois obtenir un `401` — et l'appel ne part jamais
   chez Mistral, donc ton quota est protégé.

3. Plus simple que curl : ouvre **http://localhost:8000/docs**.

## Le corpus

```bash
python -m app.ingest "C:/chemin/vers/tes/cours"
```

Formats lus : `.pdf`, `.pptx`, `.md`, `.txt`, `.rst`. Un format inconnu est
signalé et ignoré, pas une erreur fatale. Ajouter le `.docx` demain = écrire
une fonction et l'inscrire dans `PARSERS`, rien d'autre.

L'ingestion est **rejouable** : une empreinte SHA-256 par fichier permet de
ne retraiter que ce qui a changé. `--force` réindexe tout. Limite connue :
elle ajoute et remplace, mais **ne supprime jamais** un document dont le
fichier a disparu du dossier.

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
  vecteur et attire le pseudo-code de Floyd-Warshall. D'où la **recherche
  hybride** : vectoriel + mots-clés, fusionnés par RRF.
- **Le seuil de distance ne détecte pas le hors-sujet.** Questions du cours
  0.145–0.286, questions hors-sujet 0.272–0.396 : **les deux distributions
  se chevauchent**. C'est le prompt système qui interdit au modèle de
  répondre hors des passages fournis.
- **Les tableaux et les notes de présentateur des `.pptx`** portent souvent
  l'essentiel du cours : ils sont extraits, pas seulement les titres.

## Évaluation

### Pourquoi des métriques sans LLM (et pas Ragas)

Si le golden set dit « la réponse est dans `III. Breadth first search` »,
vérifier que ce document remonte est une comparaison : Hit@5, MRR,
Précision@5, calculés à la main dans `eval/metriques.py`. Gratuit, et le
calcul est déterministe.

Ragas juge surtout la *génération* avec un LLM qui découpe la réponse en
affirmations puis vérifie chacune. Sur ce compte, les seuls modèles ouverts
font 3 à 8 milliards de paramètres : un juge de cette taille donne des scores
bruités et des sorties JSON illisibles. L'évaluation de la génération reste
à faire, avec un juge adapté.

### Le golden set

`eval/golden_set.yaml` — 63 questions. Tout le corpus est en anglais, donc
les catégories isolent l'effet de la langue :

| Catégorie | Principe | Exemple |
|---|---|---|
| `fr_proche` | français, terme clé proche de l'anglais | « algorithme de Dijkstra » |
| `fr_eloigne` | français, terme clé sans ressemblance | « parcours en largeur » / *breadth-first search* |
| `en` | groupe témoin en anglais | « How does breadth-first search work? » |
| `hors_sujet` | rien à trouver | « recette de la tarte aux pommes » |

Des **paires** relient une question française à la même question en anglais :
l'écart entre les deux mesure la langue, toutes choses égales par ailleurs.

Règles d'étiquetage :
- **jamais avec notre propre retrieval**, sinon la mesure est circulaire ;
- une page n'est étiquetée que si elle a été lue ;
- une étiquette corrigée *parce que le système la contestait* est signalée
  comme telle dans le fichier : on ne relit que là où le système conteste,
  jamais là où il approuve à tort. Correction asymétrique, à connaître.

Biais principal : ce golden set a été rédigé par celui qui construit le
système. **Ce sont tes vraies questions de révision qui comptent** : ajoute-les.

### Mesurer

```bash
python -m eval.retrieval --nom essai --strategie hybride
python -m eval.retrieval --nom essai-reecriture --strategie reecriture --comparer essai
```

```bash
python -m eval.synthese --groupe hybride=h-r1,h-r2 --groupe reecriture=r-r1,r-r2,r-r3
```

Chaque résultat (`eval/resultats/<nom>.json`) enregistre le commit (suffixé
`+modifications` si le code n'était pas commité), l'empreinte du golden set,
les paramètres, la latence, et pour chaque question les requêtes lancées et
les passages renvoyés. `eval.synthese` refuse de comparer des mesures notées
avec des golden sets différents.

L'historique du dépôt a été réécrit avant publication (adresse e-mail de
l'auteur) : les identifiants de commit enregistrés dans les résultats
renvoient à l'ancien historique. La table
[`eval/resultats/correspondance_commits.md`](eval/resultats/correspondance_commits.md)
fait le lien.

### Ce que les mesures ont appris

- **Le pipeline n'est pas parfaitement reproductible.** Lors d'une première
  série, deux exécutions identiques de la stratégie hybride ont renvoyé, pour
  les mêmes chunks, des distances différant jusqu'à 1e-3 : une quasi-égalité
  s'est inversée. La série suivante, elle, a donné deux exécutions identiques.
  Source non établie (trois appels d'embeddings rapprochés donnent des
  vecteurs identiques). Un écart d'un passage entre deux variantes est du
  bruit : on répète les mesures avant de conclure.
- **La réécriture n'est déterministe ni à température 0, ni avec
  `random_seed`.** D'où les requêtes enregistrées au moment de la mesure :
  rappeler le LLM après coup ne redonne pas ce que la mesure a vu.
- **Un document retiré du dossier reste indexé**, et il pèse sur les
  résultats : sur une première mesure, 2 des 4 régressions de la réécriture
  venaient d'un livre alors absent du dossier corpus.

### Résultats

Mesures répétées sur le golden set `7be5592d2ef6`, stratégies alternées dans
le temps. 58 questions avec réponse attendue, dont 20 en français à termes
éloignés. Moyenne sur les répétitions, [min–max] entre crochets.

| Stratégie | Répét. | Hit@5 | MRR | Précision@5 | FR éloigné Hit@5 | FR éloigné MRR | FR éloigné Précision@5 | Latence médiane |
|---|---|---|---|---|---|---|---|---|
| Hybride | 2 | 0.966 | 0.908 | 0.803 | 0.900 | 0.783 | 0.640 | 333 ms [280–386] |
| Réécriture `8b` | 3 | 1.000 | 0.988 [0.98–0.99] | 0.912 [0.91–0.92] | 1.000 | 0.967 [0.95–0.97] | 0.877 [0.87–0.89] | 1278 ms [1218–1368] |
| Réécriture `3b` | 3 | 1.000 | 0.988 [0.98–0.99] | 0.916 [0.91–0.92] | 1.000 | 0.975 | 0.877 [0.87–0.89] | 1044 ms [973–1113] |

Ce qui est établi (étendues disjointes) :
- **la réécriture améliore le retrieval**, avec les deux modèles et sur toutes
  les métriques de qualité ; le gain est le plus fort là où il était attendu,
  en français à termes éloignés (MRR 0.78 → 0.97, précision 0.64 → 0.88) ;
- les deux échecs de l'hybride (le parcours en largeur, à chaque répétition)
  disparaissent, et aucune question n'est instable d'une répétition à l'autre ;
- elle coûte ~0.7 à ~0.9 s de retrieval en médiane ;
- **entre `8b` et `3b`, seule la latence diffère** (`3b` plus rapide d'environ
  230 ms) : aucune différence de qualité n'est établie.

### Réécriture de requête

`REECRITURE_REQUETES=true` active la recherche multi-requêtes : la question
d'origine plus deux reformulations en anglais technique, dont les classements
sont fusionnés par RRF. **Activée en production avec `ministral-3b-latest`**,
sur la foi des mesures ci-dessus. `MISTRAL_REECRITURE_MODEL` choisit le modèle.

Trois garde-fous :
- la question d'origine est toujours conservée : une mauvaise reformulation
  ajoute un classement, elle ne remplace jamais celui qui marchait ;
- une panne de la réécriture retombe sur la recherche simple, jamais sur une
  erreur de `/chat` ;
- **aucun exemple du golden set dans le prompt** (un test le vérifie) : y
  écrire « parcours en largeur → breadth-first search » soufflerait la
  réponse à la question d'évaluation, et le score ne mesurerait plus rien.

## Observabilité

Chaque requête `/chat` devient une **trace Langfuse** : réécriture, embeddings,
chaque recherche SQL et génération, avec durées, entrées, sorties et tokens.
`/chat` renvoie l'identifiant de la trace (`trace_id`), qui relie une réponse
signalée à sa trace complète.

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com   # dépend de ta région
LANGFUSE_ENVIRONMENT=local                     # « production » sur Fly.io
```

Quatre règles :
- **sans les deux clés, le traçage est désactivé** et l'application tourne
  exactement comme avant ;
- **l'ingestion et l'évaluation ne tracent pas** : 1076 chunks ou 63 questions
  produiraient des milliers d'observations sans requête à laquelle les
  rattacher ;
- **aucun test n'envoie de trace** (`tests/conftest.py`), car le `.env` local
  contient les vraies clés et pydantic-settings le lit ;
- **la trace ne commence qu'après l'authentification** : une requête refusée
  ne produit rien.

### Ce que la première trace a répondu

L'étape existait pour une question précise : après l'activation de la
réécriture, une réponse en production était passée de 6,5 s à 16,9 s, sans
qu'on sache pourquoi. Mesure, sur la même question, en local puis en
production :

| Étape | Local | Production |
|---|---|---|
| Réécriture (`ministral-3b`) | 0,65 s | 0,55 s |
| Embeddings | 0,40 s | 0,27 s |
| Recherches SQL (3) | 0,72 + 0,06 + 0,06 s | 0,06 + 0,04 + 0,07 s |
| **Génération de la réponse** | **10,62 s (85 %)** | **10,30 s (91 %)** |
| Total | 12,52 s | 11,29 s |

- **La génération domine**, et de loin. Le retrieval complet, réécriture
  comprise, tient en environ 1 s en production.
- **Mais sa durée n'est pas une simple fonction de la longueur de la
  réponse** : 1310 tokens en 10,62 s (≈ 123 tokens/s) contre 549 tokens en
  10,30 s (≈ 53 tokens/s). Le débit de Mistral a varié d'un facteur 2,3 entre
  deux appels. La longueur compte, la charge du fournisseur aussi, et deux
  mesures ne suffisent pas à les séparer.
- **La première requête SQL coûte 12 fois les suivantes en local** (0,72 s
  contre 0,06 s) : c'est l'ouverture de la connexion vers Neon. En production,
  où l'application et la base sont toutes deux en Europe, l'écart disparaît.

### Un piège du SDK

`fermer()` appelle `flush()` et jamais `shutdown()`. Les clients Langfuse
partagent un gestionnaire de ressources par clé publique : `shutdown()` arrête
ses threads d'envoi, un nouveau client avec la même clé réutilise ce
gestionnaire arrêté, et la fermeture suivante attend pour toujours. C'est ce
qui figeait la suite de tests, où chaque test démarre puis arrête
l'application. L'arrêt définitif reste assuré : le SDK enregistre lui-même son
`shutdown()` via `atexit`.

## Tests et qualité

```bash
pytest -q
ruff check .
```

Les tests ne touchent jamais l'API Mistral, ni la base, ni Langfuse (un test
qui dépend du réseau est un pari, pas un test). La CI GitHub Actions rejoue
`ruff` + `pytest` + le build Docker à chaque push.

## Docker

```bash
docker compose up --build
```

## Secrets

`.env` et `.env.prod` sont dans `.gitignore` : tes clés ne partiront jamais
sur GitHub. `.env.example` est le modèle versionné, sans secret. En
production, les clés sont fournies comme variables d'environnement
(`fly secrets set`) — jamais dans l'image ni dans `fly.toml`.

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
| `ministral-8b-latest` | OK (génération) | 188 |
| `open-mistral-nemo` | OK | 188 |
| `open-mistral-7b` | OK | 188 |
| `ministral-3b-latest` | OK (réécriture) | 750 |
| `mistral-embed` | OK | 60 |

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

Un contenu vide renvoyé par le LLM ne passe jamais en `200` : il devient un
`502 « Le LLM a renvoye une reponse vide »`.

### `psycopg ... ProactorEventLoop` au démarrage

Boucle d'événements incompatible sous Windows : lance `python run.py`. Si tu
démarres uvicorn toi-même **sans rechargement automatique**, impose la boucle :
`asyncio.run(uvicorn.Server(config).serve(), loop_factory=asyncio.SelectorEventLoop)`.

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

**Les secrets, avant le premier déploiement.** Ils ne sont jamais dans
`fly.toml`, qui est versionné — ils vivent dans le coffre de Fly, chiffrés,
injectés comme variables d'environnement au démarrage :

```bash
fly secrets set MISTRAL_API_KEY=... API_KEY=... DATABASE_URL=... LANGFUSE_PUBLIC_KEY=... LANGFUSE_SECRET_KEY=...
```

Puis :

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

## Limites connues

- L'ingestion **ne supprime pas** un document dont le fichier a disparu.
- Le golden set est rédigé par le constructeur du système : à enrichir avec
  de vraies questions de révision.
- Le générateur du guide PDF (`docs/`) n'est pas versionné : le guide décrit
  l'état antérieur à l'étape 3 et ne peut pas être régénéré tel quel.
- Deux machines Fly alors qu'une suffirait (`fly scale count 1`).
- Les actions GitHub `checkout@v4` et `setup-python@v5` sont signalées comme
  dépréciées (Node.js 20).
- Les formules mathématiques sont mutilées par l'extraction PDF.

## Suite

- [x] **Étape 1** — squelette FastAPI + `/chat` + Docker
- [x] **Étape 1b** — déployé sur Fly.io (région cdg, scale-to-zero)
- [x] **Étape 2** — ingestion multi-format + pgvector + recherche hybride
- [x] **Étape 3** — observabilité Langfuse, chaque requête tracée
- [x] **Étape 4** — golden set, métriques de retrieval, réécriture mesurée
- [x] **Dépôt publié** sur GitHub, CI active
- [ ] **Étape 4b** — évaluation de la génération : citations valides, refus du
      hors-sujet, fidélité aux passages
- [ ] **Étape 5** — couche agent (LangGraph) pour le multi-étapes
- [ ] **Étape 6** — CD, cache Redis, modèle auto-hébergé (vLLM / Ollama)
