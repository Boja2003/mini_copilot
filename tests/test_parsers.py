"""Tests de la normalisation du texte.

Chaque cas ici correspond a un vrai defaut rencontre sur le corpus, pas a
un scenario imagine.
"""

from app.parsers import FORMATS_SUPPORTES, lire, normaliser


def test_ligatures_typographiques_reparees() -> None:
    # Les PDF LaTeX encodent "final" avec le caractere unique U+FB01.
    # Sans reparation, ce mot ne matche jamais une recherche sur "final".
    assert normaliser("the \ufb01nal result") == "the final result"
    assert normaliser("di\ufb00erent") == "different"
    assert normaliser("\ufb02ow") == "flow"


def test_octets_nul_supprimes() -> None:
    # Postgres refuse categoriquement 0x00 dans un champ texte : sans ce
    # nettoyage, l'ingestion echoue en DataError.
    assert "\x00" not in normaliser("texte\x00avec\x00nul")
    assert normaliser("a\x00b") == "ab"


def test_sauts_de_ligne_et_tabulations_conserves() -> None:
    # Ils portent la structure : les supprimer collerait les paragraphes.
    assert normaliser("a\nb\tc") == "a\nb\tc"


def test_cesure_de_fin_de_ligne_recollee() -> None:
    assert normaliser("opti-\nmisation") == "optimisation"


def test_espaces_insecables_normalises() -> None:
    assert normaliser("a\u00a0b") == "a b"


def test_format_inconnu_rejete(tmp_path) -> None:
    fichier = tmp_path / "cours.xyz"
    fichier.write_text("contenu")
    try:
        lire(fichier)
    except ValueError as exc:
        assert ".xyz" in str(exc)
    else:
        raise AssertionError("un format inconnu doit lever ValueError")


def test_texte_brut_lu(tmp_path) -> None:
    fichier = tmp_path / "notes.md"
    fichier.write_text("# Titre\n\nUn paragraphe.", encoding="utf-8")
    unites = lire(fichier)
    assert unites[0][0] == 1
    assert "Un paragraphe." in unites[0][1]


def test_formats_supportes_annonces() -> None:
    assert ".pdf" in FORMATS_SUPPORTES
    assert ".pptx" in FORMATS_SUPPORTES
