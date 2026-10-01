"""Evaluation de la generation : la reponse respecte-t-elle ses passages ?

L'etape 4 a mesure le retrieval — les bons passages arrivent-ils sous les
yeux du modele ? Celle-ci mesure ce que le modele en fait. Les traces de
l'etape 3 ont montre que la generation pese 85 a 91 % du temps : c'est
desormais la partie la moins mesuree du pipeline.

Aucun juge LLM ici, et c'est delibere. Les seuls modeles ouverts sur ce
compte font 3 a 8 milliards de parametres, trop justes pour juger une
reponse. On commence donc par ce qui se verifie sans juge, exactement comme
a l'etape 4 :

  - citations valides : chaque [n] de la reponse designe-t-il un passage
    reellement fourni ? Un [7] alors que 5 passages ont ete donnes est une
    reference inventee, et c'est verifiable sans interpretation ;
  - reponse sourcee : quand des passages existent, la reponse en cite-t-elle
    au moins un ? Sans citation, elle n'est pas verifiable ;
  - refus correct : une question hors-sujet obtient-elle le refus prevu ?
  - refus injustifie : le modele refuse-t-il alors qu'un passage du document
    attendu etait dans son contexte ? L'etape 4 ayant mesure le retrieval,
    un tel refus est imputable a la generation seule.

Ce qui reste hors de portee sans juge : la fidelite fine (la phrase citee
dit-elle vraiment ce que la reponse lui fait dire ?) et l'exactitude.

Usage :
    python -m eval.generation --nom base
    python -m eval.generation --nom essai --limite 10
    python -m eval.generation --nom base-v2 --rejouer base
"""

import argparse
import asyncio
import datetime as dt
import json
import logging
import pathlib
import re
import sys
import time
import unicodedata
from collections import defaultdict

from app.boucle import configurer as configurer_boucle
from app.config import get_settings
from app.db import close_pool, open_pool
from app.llm import generate_answer
from app.mistral import LLMError, close_client
from app.observabilite import desactiver as desactiver_traces

from .metriques import moyenne
from .retrieval import (
    RESULTATS,
    charger_golden_set,
    commit_courant,
    empreinte_golden_set,
)

logging.basicConfig(level=logging.WARNING)

# La formule exacte imposee par le prompt systeme de app/llm.py. On la
# cherche sous forme normalisee (sans accents, sans casse) : le modele
# reformule parfois la ponctuation autour, jamais ces quelques mots.
MARQUEUR_REFUS = "je ne trouve pas"

# Le modele cite « [1] », mais aussi « [1, p. 7] » ou « [2, 5] ». On lit donc
# le CONTENU de chaque crochet, et on n'y garde que les elements entierement
# numeriques : dans « [2, p. 10] », le 10 est un numero de page, le compter
# inventerait une citation hors bornes. Premiere version de ce controle :
# elle n'acceptait que des crochets purement numeriques, et classait une
# reponse correctement sourcee comme non sourcee.
CROCHETS = re.compile(r"\[([^\[\]]{1,40})\]")


def normaliser(texte: str) -> str:
    sans_accents = unicodedata.normalize("NFKD", texte)
    return "".join(c for c in sans_accents if not unicodedata.combining(c)).casefold()


# Delimiteurs de mathematiques en ligne. Le modele ecrit des intervalles
# \([0, 1]\) ou $[a, b]$ : ce ne sont pas des citations.
AVANT_MATH = ("\(", "$")
APRES_MATH = ("\)", "$")


def extraire_citations(texte: str) -> set[int]:
    r"""Les numeros de passage cites dans la reponse, par exemple [2].

    Deux pieges, rencontres l'un et l'autre sur de vraies reponses :
      - « [1, p. 7] » cite le passage 1, pas les passages 1 et 7 ;
      - « \([0, 1]\) » est un intervalle mathematique. Le corpus porte sur
        l'optimisation convexe et la recherche par section doree : les
        intervalles y sont partout, et les compter inventerait des
        citations hors bornes sur des reponses correctes.
    """
    numeros = set()
    for crochet in CROCHETS.finditer(texte):
        avant = texte[max(0, crochet.start() - 2) : crochet.start()]
        apres = texte[crochet.end() : crochet.end() + 2]
        if avant.endswith(AVANT_MATH) or apres.startswith(APRES_MATH):
            continue
        for element in crochet.group(1).split(","):
            element = element.strip()
            # Les passages sont numerotes a partir de 1 : un 0 vient d'une
            # borne d'intervalle, jamais d'une citation.
            if element.isdigit() and int(element) >= 1:
                numeros.add(int(element))
    return numeros


def est_un_refus(texte: str) -> bool:
    """La reponse est-elle le refus prevu par le prompt systeme ?"""
    return MARQUEUR_REFUS in normaliser(texte)


def evaluer_reponse(
    texte: str, documents_fournis: list[str], documents_attendus: list[str]
) -> dict:
    """Les quatre controles, pour une reponse.

    `documents_fournis` est la liste des documents des passages donnes au
    modele, dans l'ordre : sa longueur borne les citations valides, et son
    intersection avec `documents_attendus` dit si la reponse etait possible.
    """
    citations = extraire_citations(texte)
    nb_passages = len(documents_fournis)
    hors_bornes = sorted(n for n in citations if not 1 <= n <= nb_passages)
    refus = est_un_refus(texte)
    bon_passage_fourni = bool(set(documents_fournis) & set(documents_attendus))

    resultat = {
        "refus": refus,
        "citations": sorted(citations),
        "citations_hors_bornes": hors_bornes,
    }
    # Une reponse qui ne cite rien ne peut pas citer faux : on ne compte le
    # controle que sur les reponses qui citent.
    if citations:
        resultat["citations_valides"] = not hors_bornes
    if documents_attendus:
        # Un refus n'a pas a citer : on ne lui reproche pas son absence de
        # source, seulement le fait de refuser (controle suivant).
        if not refus:
            resultat["cite_une_source"] = bool(citations)
        if bon_passage_fourni:
            resultat["refus_injustifie"] = refus
    else:
        resultat["refus_correct"] = refus
    return resultat


def _marques(ligne: dict) -> str:
    marques = ""
    if ligne.get("citations_hors_bornes"):
        marques += " CITATION-HORS-BORNES"
    if ligne.get("refus_injustifie"):
        marques += " REFUS-INJUSTIFIE"
    if ligne.get("refus_correct") is False:
        marques += " PAS-DE-REFUS"
    if ligne.get("cite_une_source") is False:
        marques += " SANS-SOURCE"
    return marques


async def interroger(questions: list[dict], limite: int | None) -> list[dict]:
    lignes = []
    for q in questions[: limite or len(questions)]:
        debut = time.perf_counter()
        try:
            reponse = await generate_answer(q["question"])
        except LLMError as exc:
            # Une panne sur une question ne doit pas perdre les 62 autres.
            print(f"  ! {q['id']} : {exc}", flush=True)
            lignes.append(
                {
                    "id": q["id"],
                    "categorie": q["categorie"],
                    "question": q["question"],
                    "erreur": str(exc),
                }
            )
            continue

        documents_fournis = [p.titre for p in reponse.passages]
        ligne = {
            "id": q["id"],
            "categorie": q["categorie"],
            "question": q["question"],
            "duree_ms": round((time.perf_counter() - debut) * 1000),
            "reponse": reponse.texte,
            "passages": [
                {"document": p.titre, "page": p.page} for p in reponse.passages
            ],
            "erreur": None,
        }
        ligne |= evaluer_reponse(
            reponse.texte, documents_fournis, q.get("documents") or []
        )
        lignes.append(ligne)
        marques = _marques(ligne)
        print(f"  {'x' if marques else '.'} {q['id']}{marques}", flush=True)
    return lignes


CONTROLES = [
    ("citations_valides", "citations valides"),
    ("cite_une_source", "reponse sourcee"),
    ("refus_correct", "refus correct"),
    ("refus_injustifie", "refus injustifie"),
]


def agreger(lignes: list[dict]) -> dict:
    par_cat: dict[str, list[dict]] = defaultdict(list)
    for ligne in lignes:
        if ligne.get("erreur"):
            continue
        par_cat[ligne["categorie"]].append(ligne)
        par_cat["TOUTES"].append(ligne)

    def resume(groupe: list[dict]) -> dict:
        res = {"n": len(groupe)}
        for cle, _ in CONTROLES:
            valeurs = [float(g[cle]) for g in groupe if cle in g]
            res[cle] = (
                {"taux": round(moyenne(valeurs), 3), "n": len(valeurs)}
                if valeurs
                else None
            )
        return res

    return {cat: resume(groupe) for cat, groupe in sorted(par_cat.items())}


def afficher(agregats: dict) -> None:
    entete = f"{'categorie':<12} {'n':>3} " + " ".join(
        f"{libelle:>20}" for _, libelle in CONTROLES
    )
    print("\n" + entete)
    print("-" * len(entete))
    for cat, m in agregats.items():
        cellules = []
        for cle, _ in CONTROLES:
            valeur = m[cle]
            cellules.append(
                f"{'-':>20}"
                if valeur is None
                else f"{valeur['taux']:.3f} (n={valeur['n']})".rjust(20)
            )
        print(f"{cat:<12} {m['n']:>3} " + " ".join(cellules))


def afficher_defauts(lignes: list[dict]) -> None:
    """Le detail compte plus que la moyenne : un seul refus injustifie est un
    vrai probleme, meme a 98 % de reussite."""
    familles = [
        ("citations hors bornes", lambda x: x.get("citations_hors_bornes")),
        ("refus injustifie", lambda x: x.get("refus_injustifie")),
        ("pas de refus sur hors-sujet", lambda x: x.get("refus_correct") is False),
        ("reponse sans source", lambda x: x.get("cite_une_source") is False),
        ("erreur", lambda x: x.get("erreur")),
    ]
    for libelle, test in familles:
        concernees = [x["id"] for x in lignes if test(x)]
        print(f"  {libelle:<30} {len(concernees):>2} : {', '.join(concernees) or '-'}")


def enregistrer(
    nom: str, lignes: list[dict], parametres: dict, supplement: dict | None = None
) -> pathlib.Path:
    agregats = agreger(lignes)
    afficher(agregats)
    print("\nDefauts, question par question :")
    afficher_defauts(lignes)

    durees = sorted(x["duree_ms"] for x in lignes if "duree_ms" in x)
    latence = (
        {
            "mediane": durees[len(durees) // 2],
            "p90": durees[int(0.9 * (len(durees) - 1))],
        }
        if durees
        else {}
    )

    RESULTATS.mkdir(exist_ok=True)
    sortie = {
        "nom": nom,
        "date": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "commit": commit_courant(),
        "golden_set": empreinte_golden_set(),
        "parametres": parametres,
        "latence_ms": latence,
        **(supplement or {}),
        "agregats": agregats,
        "questions": lignes,
    }
    chemin = RESULTATS / f"generation-{nom}.json"
    chemin.write_text(
        json.dumps(sortie, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"\nLatence : mediane {latence.get('mediane')} ms, p90 {latence.get('p90')} ms"
    )
    print(f"Resultats : {chemin}")
    return chemin


def rejouer(source_nom: str, nom: str) -> None:
    """Recalcule les controles sur des reponses deja enregistrees.

    Les reponses du modele sont conservees dans le fichier de resultats.
    Quand un controle evolue — la lecture des citations, d'abord trop stricte
    pour « [1, p. 7] » —, on remesure sans redemander une seule reponse :
    aucun quota consomme, et aucune variabilite du modele introduite entre
    les deux mesures, donc l'ecart observe vient bien du controle corrige.
    """
    chemin = RESULTATS / f"generation-{source_nom}.json"
    source = json.loads(chemin.read_text(encoding="utf-8"))
    attendus_par_id = {
        q["id"]: (q.get("documents") or []) for q in charger_golden_set()
    }

    lignes = []
    for ancienne in source["questions"]:
        if ancienne.get("erreur"):
            lignes.append(ancienne)
            continue
        ligne = {
            cle: ancienne[cle]
            for cle in (
                "id",
                "categorie",
                "question",
                "duree_ms",
                "reponse",
                "passages",
                "erreur",
            )
            if cle in ancienne
        }
        ligne |= evaluer_reponse(
            ancienne["reponse"],
            [p["document"] for p in ancienne.get("passages", [])],
            attendus_par_id.get(ancienne["id"], []),
        )
        lignes.append(ligne)
        marques = _marques(ligne)
        if marques:
            print(f"  x {ligne['id']}{marques}", flush=True)

    print(f"{len(lignes)} reponses rejouees depuis « {source_nom} »")
    enregistrer(
        nom,
        lignes,
        source.get("parametres", {}),
        {"rejoue_depuis": source_nom, "date_des_reponses": source.get("date")},
    )


async def principal(nom: str, limite: int | None) -> None:
    questions = charger_golden_set()
    settings = get_settings()
    await open_pool()
    try:
        print(
            f"{min(limite or len(questions), len(questions))} questions, "
            f"modele {settings.mistral_model}, "
            f"reecriture {'active' if settings.reecriture_requetes else 'inactive'}"
        )
        lignes = await interroger(questions, limite)
    finally:
        await close_pool()
        await close_client()

    enregistrer(
        nom,
        lignes,
        {
            "modele": settings.mistral_model,
            "reecriture_requetes": settings.reecriture_requetes,
            "modele_reecriture": settings.mistral_reecriture_model,
        },
    )


if __name__ == "__main__":
    parseur = argparse.ArgumentParser(description="Evaluation de la generation")
    parseur.add_argument("--nom", required=True, help="nom du fichier de resultats")
    parseur.add_argument("--limite", type=int, help="n'interroger que les N premieres")
    parseur.add_argument(
        "--rejouer",
        metavar="SOURCE",
        help="recalculer les controles sur les reponses deja enregistrees",
    )
    args = parseur.parse_args()
    try:
        if args.rejouer:
            # Aucun appel au modele ni a la base : pas de boucle a configurer.
            rejouer(args.rejouer, args.nom)
        else:
            # Comme l'evaluation du retrieval : pas de trace, le temps d'envoi
            # fausserait la latence et remplirait le projet Langfuse.
            desactiver_traces()
            configurer_boucle()
            asyncio.run(principal(args.nom, args.limite))
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERREUR : {exc}", file=sys.stderr)
        sys.exit(1)
