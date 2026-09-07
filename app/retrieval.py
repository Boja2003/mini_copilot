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
# Filtre GROSSIER, et rien de plus. Mesure sur ce corpus : les questions du
# cours obtiennent 0.145 a 0.286, les questions hors-sujet 0.272 a 0.396 —
# les deux distributions se chevauchent, donc aucun seuil ne separe
# proprement. On le regle donc pour ne JAMAIS perdre un bon passage (au-dela
# du pire cas utile) et se contenter d'ecarter le bruit franc. C'est le
# prompt, dans llm.py, qui porte le vrai garde-fou anti-hallucination.
DISTANCE_MAXIMALE = 0.35


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
    # Vector() et pas la liste brute : sans ce type explicite, psycopg
    # envoie un double precision[], et l'operateur <=> n'existe pas pour
    # ce type (contrairement a l'INSERT, ou Postgres convertit tout seul).
    vecteur = Vector(await embarquer_un(question))

    async with get_pool().connection() as conn:
        lignes = await (
            await conn.execute(
                """
                SELECT c.contenu, d.source, d.titre, c.page,
                       c.embedding <=> %s AS distance
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                ORDER BY c.embedding <=> %s
                LIMIT %s
                """,
                (vecteur, vecteur, nb),
            )
        ).fetchall()

    passages = [
        Passage(contenu=r[0], source=r[1], titre=r[2], page=r[3], distance=float(r[4]))
        for r in lignes
        if float(r[4]) <= DISTANCE_MAXIMALE
    ]
    logger.info(
        "Retrieval : %s/%s passages retenus (meilleure distance %.3f)",
        len(passages),
        len(lignes),
        float(lignes[0][4]) if lignes else -1,
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
