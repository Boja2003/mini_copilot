"""Lecture des documents, un parser par format.

Le decoupage se fait en deux temps, et c'est delibere :
  1. ICI : lire un fichier -> une liste de (numero de page, texte brut)
  2. dans chunking.py : decouper ce texte en morceaux indexables

Separer les deux permet d'ajouter un format sans toucher au decoupage, et
inversement. Ajouter le .docx demain = ecrire une fonction et l'inscrire
dans PARSERS, rien d'autre.
"""

import logging
import pathlib
import unicodedata
from collections.abc import Callable

from pptx import Presentation
from pypdf import PdfReader

logger = logging.getLogger(__name__)

# Une "unite" = une page de PDF ou une diapo de PPTX. On garde ce numero
# jusqu'au bout : c'est ce qui permettra de citer « page 12 » dans la
# reponse, et c'est ce qui rend un RAG credible.
Unite = tuple[int, str]


def normaliser(texte: str) -> str:
    """Repare le texte des PDF avant tout traitement.

    Les PDF LaTeX utilisent des ligatures typographiques : "final" y est
    encode par le caractere unique U+FB01 (fi). Sans normalisation, ce mot
    ne matcherait jamais une recherche sur "final" — ni pour l'embedding,
    ni pour un humain qui lit le passage cite. NFKD les decompose.
    """
    texte = unicodedata.normalize("NFKC", texte)
    texte = texte.replace("\ufb00", "ff").replace("\ufb01", "fi")
    texte = texte.replace("\ufb02", "fl").replace("\ufb03", "ffi")
    texte = texte.replace("\ufb04", "ffl").replace("\ufb05", "st")
    # Traits d'union de fin de ligne : "opti-\nmisation" -> "optimisation".
    texte = texte.replace("-\n", "")
    # Espaces insecables et variantes -> espace simple.
    texte = texte.replace("\u00a0", " ").replace("\u202f", " ")
    # Caracteres de controle : l'extraction PDF en produit (octets NUL
    # notamment), et Postgres refuse categoriquement 0x00 dans un champ
    # texte. On garde \n et \t, qui portent la structure.
    texte = "".join(c for c in texte if c in "\n\t" or unicodedata.category(c) != "Cc")
    return texte


def lire_pdf(chemin: pathlib.Path) -> list[Unite]:
    lecteur = PdfReader(str(chemin))
    unites = []
    for numero, page in enumerate(lecteur.pages, start=1):
        try:
            texte = page.extract_text() or ""
        except Exception:
            # Une page illisible ne doit pas faire echouer tout le document.
            logger.warning("Page %s illisible dans %s", numero, chemin.name)
            continue
        unites.append((numero, normaliser(texte)))
    return unites


def lire_pptx(chemin: pathlib.Path) -> list[Unite]:
    presentation = Presentation(str(chemin))
    unites = []
    for numero, diapo in enumerate(presentation.slides, start=1):
        morceaux = []
        for forme in diapo.shapes:
            if forme.has_text_frame and forme.text_frame.text.strip():
                morceaux.append(forme.text_frame.text)
            # Les tableaux portent souvent l'essentiel d'une diapo de cours.
            if getattr(forme, "has_table", False) and forme.has_table:
                for ligne in forme.table.rows:
                    cellules = [c.text.strip() for c in ligne.cells]
                    if any(cellules):
                        morceaux.append(" | ".join(cellules))
        # Les notes du presentateur expliquent souvent ce que la diapo montre
        # sans le dire : c'est du contenu de premier choix pour un RAG.
        if diapo.has_notes_slide:
            notes = diapo.notes_slide.notes_text_frame.text.strip()
            if notes:
                morceaux.append(f"[Notes] {notes}")
        unites.append((numero, normaliser("\n".join(morceaux))))
    return unites


def lire_texte(chemin: pathlib.Path) -> list[Unite]:
    """Markdown, texte brut : un seul bloc, il n'y a pas de pagination."""
    return [(1, normaliser(chemin.read_text(encoding="utf-8", errors="replace")))]


PARSERS: dict[str, Callable[[pathlib.Path], list[Unite]]] = {
    ".pdf": lire_pdf,
    ".pptx": lire_pptx,
    ".md": lire_texte,
    ".txt": lire_texte,
    ".rst": lire_texte,
}

FORMATS_SUPPORTES = sorted(PARSERS)


def lire(chemin: pathlib.Path) -> list[Unite]:
    parser = PARSERS.get(chemin.suffix.lower())
    if parser is None:
        raise ValueError(
            f"Format non supporte : {chemin.suffix} "
            f"(supportes : {', '.join(FORMATS_SUPPORTES)})"
        )
    return parser(chemin)
