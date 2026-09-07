"""Tests du decoupage.

Le decoupage est ce qui fait la qualite d'un RAG : ces tests verrouillent
les proprietes dont depend le retrieval.
"""

from app.chunking import RECOUVREMENT, TAILLE_CIBLE, TAILLE_MINIMALE, decouper


def test_numero_de_page_conserve() -> None:
    # Sans le numero de page, impossible de citer « d'apres la page 12 » :
    # la reponse cesse d'etre verifiable.
    chunks = decouper([(7, "Un paragraphe assez long. " * 20)])
    assert all(c.page == 7 for c in chunks)


def test_positions_croissantes_et_uniques() -> None:
    chunks = decouper([(1, "Phrase. " * 300), (2, "Autre phrase. " * 300)])
    positions = [c.position for c in chunks]
    assert positions == sorted(positions)
    assert len(set(positions)) == len(positions)


def test_aucun_chunk_ne_depasse_la_borne() -> None:
    # La borne est la cible + le recouvrement : au-dela, le passage utile
    # se noie dans du hors-sujet et l'embedding devient flou.
    texte = "Une phrase de taille moyenne pour remplir. " * 200
    chunks = decouper([(1, texte)])
    assert chunks
    assert max(len(c.contenu) for c in chunks) <= TAILLE_CIBLE + RECOUVREMENT


def test_bloc_sans_frontiere_naturelle_est_coupe_quand_meme() -> None:
    # Cas reel du corpus : une legende de figure mathematique, sans ligne
    # vide ni ponctuation de fin de phrase. Sans filet de securite, elle
    # passait entiere et produisait un chunk de 3x la cible.
    texte = "mot " * 2000
    chunks = decouper([(1, texte)])
    assert max(len(c.contenu) for c in chunks) <= TAILLE_CIBLE + RECOUVREMENT


def test_fragments_trop_courts_ecartes() -> None:
    # Un numero de page isole n'est pas un passage : l'indexer pollue.
    assert decouper([(1, "42")]) == []
    assert all(len(c.contenu) >= TAILLE_MINIMALE for c in decouper([(1, "ok. " * 100)]))


def test_texte_vide_ne_produit_rien() -> None:
    assert decouper([(1, "")]) == []
    assert decouper([(1, "   \n\n  ")]) == []


def test_document_multipage_couvre_toutes_les_pages() -> None:
    pages = [(n, f"Contenu de la page {n}. " * 30) for n in range(1, 6)]
    chunks = decouper(pages)
    assert {c.page for c in chunks} == {1, 2, 3, 4, 5}
