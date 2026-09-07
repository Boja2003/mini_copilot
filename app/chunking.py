"""Decoupage du texte en morceaux indexables.

C'est LA decision qui fait la qualite d'un RAG, bien plus que le choix du
modele :
  - morceaux trop gros -> le passage pertinent est noye dans du hors-sujet,
    et l'embedding, qui resume tout le morceau en un seul vecteur, devient
    flou ;
  - morceaux trop petits -> on perd le contexte, et un theoreme se retrouve
    separe de sa demonstration.

On decoupe sur les frontieres naturelles (paragraphes, puis phrases) plutot
qu'a l'aveugle tous les N caracteres : couper au milieu d'une phrase abime
le sens que l'embedding est justement cense capturer.
"""

import re
from dataclasses import dataclass

# ~1200 caracteres = environ 300 tokens. Assez pour un raisonnement complet
# dans un cours de maths, assez court pour rester precis.
TAILLE_CIBLE = 1200
# Le recouvrement evite qu'une idee coupee en deux soit perdue par les deux
# morceaux : la fin du precedent est rappelee au debut du suivant.
RECOUVREMENT = 180
# En dessous, c'est un titre isole ou un numero de page : du bruit.
TAILLE_MINIMALE = 80


@dataclass(frozen=True)
class Chunk:
    position: int
    page: int
    contenu: str


def _couper_en_dur(texte: str) -> list[str]:
    """Coupe aux espaces un bloc qui depasse encore la taille cible."""
    if len(texte) <= TAILLE_CIBLE:
        return [texte]
    morceaux, courant = [], ""
    for mot in texte.split(" "):
        if len(courant) + len(mot) + 1 > TAILLE_CIBLE and courant:
            morceaux.append(courant.strip())
            courant = ""
        courant += mot + " "
    if courant.strip():
        morceaux.append(courant.strip())
    return morceaux


def _decouper_texte(texte: str) -> list[str]:
    """Un bloc de texte -> morceaux de taille cible, sur frontieres naturelles."""
    paragraphes = [p.strip() for p in re.split(r"\n\s*\n", texte) if p.strip()]
    if not paragraphes:
        return []

    # Un paragraphe plus long que la cible est re-decoupe en phrases.
    unites: list[str] = []
    for para in paragraphes:
        if len(para) <= TAILLE_CIBLE:
            unites.append(para)
            continue
        phrases = re.split(r"(?<=[.!?])\s+", para)
        tampon = ""
        for phrase in phrases:
            if len(tampon) + len(phrase) + 1 > TAILLE_CIBLE and tampon:
                unites.append(tampon.strip())
                tampon = ""
            tampon += phrase + " "
        if tampon.strip():
            unites.append(tampon.strip())

    # Filet de securite : un bloc sans ligne vide NI ponctuation de fin de
    # phrase (typique des legendes de figures mathematiques) traverse les
    # deux decoupages precedents intact. On le coupe alors aux espaces, la
    # seule frontiere qui reste. Sans ca, un seul chunk peut faire 3x la
    # cible et noyer le passage utile.
    unites = [p for u in unites for p in _couper_en_dur(u)]

    # Regroupement : on remplit jusqu'a la cible, avec recouvrement.
    morceaux: list[str] = []
    courant = ""
    for unite in unites:
        if len(courant) + len(unite) + 2 > TAILLE_CIBLE and courant:
            morceaux.append(courant.strip())
            courant = courant[-RECOUVREMENT:] if RECOUVREMENT else ""
        courant += unite + "\n\n"
    if courant.strip():
        morceaux.append(courant.strip())

    return [m for m in morceaux if len(m) >= TAILLE_MINIMALE]


def decouper(unites: list[tuple[int, str]]) -> list[Chunk]:
    """Transforme les (page, texte) d'un document en chunks numerotes.

    Le numero de page est conserve : c'est lui qui permettra de repondre
    « d'apres la page 12 de ton cours » au lieu d'une affirmation en l'air.
    """
    chunks: list[Chunk] = []
    position = 0
    for page, texte in unites:
        for morceau in _decouper_texte(texte):
            chunks.append(Chunk(position=position, page=page, contenu=morceau))
            position += 1
    return chunks
