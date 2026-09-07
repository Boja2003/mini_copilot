"""Recherche des passages pertinents dans la base vectorielle.

Le coeur du RAG tient dans l'operateur `<=>` de pgvector : la distance
cosinus entre deux vecteurs. Trier par cette distance revient a classer les
passages du plus proche sens au plus eloigne — c'est ca, la « recherche
semantique » : pas de mots-cles, une comparaison de directions dans un
espace a 1024 dimensions.
"""

import logging
from dataclasses import dataclass

from pgvector import Vector

from .db import get_pool
from .embeddings import embarquer_un

logger = logging.getLogger(__name__)

NB_PASSAGES = 5
# On tire large dans chaque moteur avant de fusionner : un passage 12e en
# vectoriel et 3e en mots-cles doit pouvoir remonter.
NB_CANDIDATS = 25

# Filtre GROSSIER sur le bras vectoriel, et rien de plus. Mesure sur ce
# corpus : les questions du cours obtiennent 0.145 a 0.286, les questions
# hors-sujet 0.272 a 0.396 — les deux distributions se chevauchent, donc
# aucun seuil ne separe proprement. On le regle pour ne JAMAIS perdre un
# bon passage. C'est le prompt, dans llm.py, qui porte le vrai garde-fou
# anti-hallucination.
DISTANCE_MAXIMALE = 0.35

# Constante de la fusion RRF (Reciprocal Rank Fusion). Elle amortit l'ecart
# entre les premieres places : sans elle, le 1er ecraserait tout le reste.
# 60 est la valeur de la publication d'origine, et un bon defaut.
K_RRF = 60


@dataclass(frozen=True)
class Passage:
    contenu: str
    source: str
    titre: str
    page: int | None
    distance: float

    @property
    def citation(self) -> str:
        return f"{self.titre} (p. {self.page})" if self.page else self.titre


async def chercher(question: str, nb: int = NB_PASSAGES) -> list[Passage]:
    """Recherche hybride : vectorielle + mots-cles, fusionnees par RRF.

    Pourquoi deux moteurs plutot qu'un :
      - le vectoriel comprend le SENS ("colorier un graphe" trouve un cours
        sur la coloration meme sans le mot), mais il dilue les termes rares.
        Mesure sur ce corpus : « Quelles sont les etapes de l'algorithme de
        Welsh & Powell ? » ne remontait PAS le cours sur Welsh & Powell — la
        tournure « etapes de l'algorithme » dominait le vecteur et attirait
        le pseudo-code de Floyd-Warshall ;
      - les mots-cles ne comprennent rien, mais ils ne ratent jamais un nom
        propre exact.
    Chacun rattrape la faiblesse de l'autre.

    La fusion est un RRF : chaque moteur vote avec 1/(K + rang). On combine
    des RANGS et non des scores, ce qui evite d'avoir a comparer une
    distance cosinus a un ts_rank — deux grandeurs sans commune mesure.
    """
    vecteur = Vector(await embarquer_un(question))

    async with get_pool().connection() as conn:
        lignes = await (
            await conn.execute(
                """
                WITH vectoriel AS (
                    SELECT c.id,
                           row_number() OVER (ORDER BY c.embedding <=> %(v)s)
                               AS rang
                    FROM chunks c
                    WHERE c.embedding <=> %(v)s <= %(dmax)s
                    ORDER BY c.embedding <=> %(v)s
                    LIMIT %(k)s
                ),
                mots_cles AS (
                    SELECT c.id,
                           row_number() OVER (ORDER BY ts_rank(c.tsv, q) DESC)
                               AS rang
                    FROM chunks c,
                         websearch_to_tsquery('simple', %(q)s) q
                    WHERE c.tsv @@ q
                    ORDER BY ts_rank(c.tsv, q) DESC
                    LIMIT %(k)s
                ),
                fusion AS (
                    SELECT COALESCE(v.id, m.id) AS id,
                           COALESCE(1.0 / (%(krrf)s + v.rang), 0)
                         + COALESCE(1.0 / (%(krrf)s + m.rang), 0) AS score
                    FROM vectoriel v
                    FULL OUTER JOIN mots_cles m ON m.id = v.id
                )
                SELECT c.contenu, d.source, d.titre, c.page,
                       c.embedding <=> %(v)s AS distance
                FROM fusion f
                JOIN chunks c ON c.id = f.id
                JOIN documents d ON d.id = c.document_id
                ORDER BY f.score DESC, distance ASC
                LIMIT %(n)s
                """,
                {
                    "v": vecteur,
                    "q": question,
                    "k": NB_CANDIDATS,
                    "dmax": DISTANCE_MAXIMALE,
                    "krrf": K_RRF,
                    "n": nb,
                },
            )
        ).fetchall()

    passages = [
        Passage(contenu=r[0], source=r[1], titre=r[2], page=r[3], distance=float(r[4]))
        for r in lignes
    ]
    logger.info(
        "Retrieval hybride : %s passages (meilleure distance %.3f)",
        len(passages),
        passages[0].distance if passages else -1,
    )
    return passages


def construire_contexte(passages: list[Passage]) -> str:
    """Met les passages en forme pour le prompt, chacun avec sa source.

    Numeroter les passages permet au modele de citer « [2] » plutot que de
    paraphraser sans reference : c'est ce qui rend la reponse verifiable.
    """
    return "\n\n".join(
        f"[{i}] ({p.citation})\n{p.contenu}" for i, p in enumerate(passages, start=1)
    )
