# Correspondance des identifiants de commit

L'historique a ete reecrit avant la publication du depot, pour remplacer
l'adresse e-mail de l'auteur par l'adresse anonyme fournie par GitHub.
Tous les identifiants de commit ont donc change.

Les resultats d'evaluation (`eval/resultats/*.json`, champ `commit`) et
plusieurs messages de commit citent les ANCIENS identifiants. Cette table
permet de les retrouver : sans elle, la provenance des mesures, construite
a l'etape 4, deviendrait invérifiable.

| Ancien | Nouveau | Commit |
|---|---|---|
| `06c43e4` | `c9b1e32` | Ã‰tape 1 : squelette FastAPI avec endpoint /chat vers l'API Mistral |
| `4c76430` | `20d9fab` | Ã‰tape 1b : authentification par clÃ© partagÃ©e et configuration Fly.io |
| `f0576fd` | `a9b8e0d` | DÃ©ploiement en ligne sur Fly.io |
| `071b9e2` | `e807530` | Ã‰tape 2 : RAG multi-format sur pgvector |
| `d1e56f7` | `a689dce` | Recherche hybride : vectoriel + mots-clÃ©s fusionnÃ©s par RRF |
| `e22883e` | `7070c61` | Ã‰chec rapide et lisible quand la base est injoignable |
| `7df0e69` | `eb7258d` | PrÃ©pare le raccordement Ã  un Postgres distant |
| `25d09f4` | `8d26aee` | Ã‰largit la rÃ¨gle .gitignore aux variantes sans point initial |
| `2ec8ef4` | `99426bd` | DÃ©lai LLM portÃ© Ã  90 s et exceptions muettes rendues lisibles |
| `b82a66a` | `e80c48e` | Ã‰tape 4 : Ã©valuation du retrieval et baseline mesurÃ©e |
| `4f5051f` | `c637b90` | RÃ©Ã©criture de requÃªte (dÃ©sactivÃ©e par dÃ©faut) et Ã©valuation traÃ§able |
| `0649b60` | `8cbb77f` | PremiÃ¨res mesures rÃ©pÃ©tÃ©es sur le golden set v2 |
| `e080f3f` | `7bbbedb` | Mesures rÃ©pÃ©tÃ©es terminÃ©es : la rÃ©Ã©criture de requÃªte est Ã©tablie |
| `5056403` | `8d2e90f` | Active la rÃ©Ã©criture de requÃªte en production (ministral-3b) |
| `cbdf8ae` | `658eee3` | Ã‰tape 3 : observabilitÃ© avec Langfuse |
| `61540b5` | `0529662` | Ajoute le guide complet du projet (PDF) |
