"""Evaluation multi-sauts : le retrieval couvre-t-il toutes les faces ?

Le golden set principal est sature — 1.000 en Hit@5, 1.000 sur les quatre
controles de generation — parce que ses questions sont a un saut : une
question, un passage, une reponse. Il ne peut donc rien mesurer d'un agent.

Ici, chaque question a plusieurs FACES (voir golden_set_multisauts.yaml) :
des elements distincts qu'une bonne reponse doit couvrir. Comparer Dijkstra
et Bellman demande les pages de l'un ET celles de l'autre, a trente pages
d'ecart dans le meme document.

La mesure est volontairement simple et deterministe : une face est couverte
des qu'un passage renvoye tombe sur l'une de ses pages.

  - couverture        : part des faces couvertes, moyennee sur les questions
  - couverture totale : part des questions dont TOUTES les faces sont couvertes

C'est la seconde qui compte : repondre a moitie a une question de
comparaison, c'est ne pas y repondre.

Trois bras, et le temoin n'est pas optionnel :

  actuel  reecriture + fusion RRF, la configuration de production
  simple  une seule recherche hybride sur la question brute
  agent   une recherche par aspect, puis entrelacement

Le temoin « simple » separe ce que l'agent doit a sa decomposition de ce
qu'il doit au simple fait de ne pas fusionner par RRF. La premiere version
de cette mesure ne l'avait pas, et a credite la decomposition d'un gain
obtenu alors qu'elle etait en panne (voir _simple).

Usage :
    python -m eval.multisauts --nom actuel --strategie actuel
    python -m eval.multisauts --nom simple --strategie simple --comparer actuel
    python -m eval.multisauts --nom agent --strategie agent --comparer simple
"""

import argparse
import asyncio
import datetime as dt
import hashlib
import json
import logging
import pathlib
import sys
import time
from collections.abc import Awaitable, Callable

import yaml

from app.agent import chercher_par_faces
from app.boucle import configurer as configurer_boucle
from app.config import get_settings
from app.db import close_pool, open_pool
from app.mistral import close_client
from app.observabilite import desactiver as desactiver_traces
from app.retrieval import Passage, chercher, chercher_avec_reecriture_detail

from .metriques import moyenne
from .retrieval import RESULTATS, commit_courant

logging.basicConfig(level=logging.WARNING)

RACINE = pathlib.Path(__file__).parent
GOLDEN_SET = RACINE / "golden_set_multisauts.yaml"
K = 5

# Une strategie renvoie les passages ET les requetes qu'elle a lancees : sans
# elles, un echec ne s'explique pas apres coup, et la decomposition n'est pas
# reproductible (temperature 0 n'y suffit pas, cf. eval/metriques.py).
Strategie = Callable[[str, int], Awaitable[tuple[list[Passage], list[str]]]]


async def _actuel(question: str, nb: int) -> tuple[list[Passage], list[str]]:
    """Le pipeline tel qu'il tourne en production, pour servir de reference."""
    if get_settings().reecriture_requetes:
        detail = await chercher_avec_reecriture_detail(question, nb)
        return detail.passages, detail.requetes
    return await chercher(question, nb), [question]


async def _simple(question: str, nb: int) -> tuple[list[Passage], list[str]]:
    """Le TEMOIN : une seule recherche hybride, sur la question brute.

    Ce bras existe a cause d'une mesure faussee. La premiere comparaison
    opposait « actuel » (reecriture + RRF) a « agent », en ignorant que
    l'agent, quand sa decomposition echoue, lance exactement cette
    recherche-ci. La decomposition est tombee en repli 14 fois sur 14 : le
    gain mesure ne venait pas d'elle, mais du fait que la fusion RRF de
    requetes reecrites concentre les passages sur un seul aspect. Sans ce
    temoin, l'agent peut etre credite d'un gain qui n'est pas le sien.
    """
    return await chercher(question, nb), [question]


async def _agent(question: str, nb: int) -> tuple[list[Passage], list[str]]:
    """Une recherche par aspect de la question, puis entrelacement."""
    detail = await chercher_par_faces(question, nb)
    return detail.passages, detail.requetes


# Les variantes a comparer.
STRATEGIES: dict[str, Strategie] = {
    "actuel": _actuel,
    "simple": _simple,
    "agent": _agent,
}


def pages_en_entiers(valeur) -> list[int]:
    """Accepte [1, 2, 3] ou "9-37" ou "16,22" ou "23".

    Les faces valent des sections entieres : les ecrire page par page
    rendrait le fichier illisible et les erreurs invisibles.
    """
    if isinstance(valeur, int):
        return [valeur]
    if isinstance(valeur, list):
        return [int(p) for p in valeur]
    pages: list[int] = []
    for morceau in str(valeur).split(","):
        morceau = morceau.strip()
        if "-" in morceau:
            debut, _, fin = morceau.partition("-")
            pages.extend(range(int(debut), int(fin) + 1))
        elif morceau:
            pages.append(int(morceau))
    return pages


def charger(chemin: pathlib.Path = GOLDEN_SET) -> list[dict]:
    """Charge les questions et refuse toute incoherence de structure.

    Une erreur ici (identifiant en double, face sans page) fausserait les
    chiffres sans rien signaler.
    """
    questions = yaml.safe_load(chemin.read_text(encoding="utf-8"))
    ids = [q["id"] for q in questions]
    doublons = sorted({i for i in ids if ids.count(i) > 1})
    if doublons:
        raise ValueError(f"Identifiants en double : {doublons}")
    for q in questions:
        faces = q.get("faces") or []
        if len(faces) < 2:
            raise ValueError(f"{q['id']} : une question multi-sauts a au moins 2 faces")
        for face in faces:
            if not face.get("nom"):
                raise ValueError(f"{q['id']} : une face sans nom")
            pages = face.get("pages") or {}
            if not pages or not all(pages.values()):
                raise ValueError(f"{q['id']} / {face.get('nom')} : face sans page")
            face["pages"] = {
                document: pages_en_entiers(valeur) for document, valeur in pages.items()
            }
    return questions


def empreinte(chemin: pathlib.Path = GOLDEN_SET) -> str:
    """Empreinte des etiquettes : deux resultats ne se comparent que s'ils ont
    ete notes avec les memes."""
    contenu = chemin.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(contenu.encode("utf-8")).hexdigest()[:12]


def faces_couvertes(faces: list[dict], passages: list[Passage]) -> list[bool]:
    """Pour chaque face, un passage renvoye tombe-t-il sur l'une de ses pages ?"""
    trouves = {(p.titre, p.page) for p in passages}
    couvertes = []
    for face in faces:
        attendues = {
            (document, page)
            for document, pages in face["pages"].items()
            for page in pages
        }
        couvertes.append(bool(trouves & attendues))
    return couvertes


def evaluer_question(faces: list[dict], passages: list[Passage]) -> dict:
    couvertes = faces_couvertes(faces, passages)
    return {
        "couverture": round(sum(couvertes) / len(couvertes), 3),
        "couverture_totale": float(all(couvertes)),
        "faces_manquantes": [
            face["nom"] for face, ok in zip(faces, couvertes, strict=True) if not ok
        ],
    }


async def mesurer(strategie: Strategie, questions: list[dict]) -> list[dict]:
    lignes = []
    for q in questions:
        debut = time.perf_counter()
        passages, requetes = await strategie(q["question"], K)
        ligne = {
            "id": q["id"],
            "question": q["question"],
            "duree_ms": round((time.perf_counter() - debut) * 1000),
            "requetes": requetes,
            "renvoyes": [
                {"document": p.titre, "page": p.page, "distance": round(p.distance, 4)}
                for p in passages
            ],
        }
        ligne |= evaluer_question(q["faces"], passages)
        lignes.append(ligne)
        manquantes = ligne["faces_manquantes"]
        marque = "x" if manquantes else "."
        detail = f"  manque : {', '.join(manquantes)}" if manquantes else ""
        print(f"  {marque} {q['id']}{detail}", flush=True)
    return lignes


def agreger(lignes: list[dict]) -> dict:
    durees = sorted(x["duree_ms"] for x in lignes)
    return {
        "n": len(lignes),
        # Questions pour lesquelles la recherche a tourne sur la question
        # brute. Pour « simple » c'est 14 par construction ; pour « agent »
        # c'est le nombre de replis de la decomposition, et il doit valoir 0
        # pour que les chiffres parlent bien de la decomposition.
        "question_brute": sum(
            1 for x in lignes if x.get("requetes") == [x.get("question")]
        ),
        "couverture": round(moyenne([x["couverture"] for x in lignes]), 3),
        "couverture_totale": round(
            moyenne([x["couverture_totale"] for x in lignes]), 3
        ),
        "latence_mediane_ms": durees[len(durees) // 2] if durees else None,
    }


def afficher(agregats: dict, reference: dict | None = None) -> None:
    print()
    for cle in (
        "couverture",
        "couverture_totale",
        "question_brute",
        "latence_mediane_ms",
    ):
        valeur = agregats[cle]
        ligne = f"  {cle:<22} {valeur}"
        if reference and reference.get(cle) is not None and valeur is not None:
            ecart = valeur - reference[cle]
            ligne += f"   (reference {reference[cle]}, ecart {ecart:+.3f})"
        print(ligne)


def afficher_changements(lignes: list[dict], reference: dict) -> None:
    avant = {x["id"]: x for x in reference["questions"]}
    gagnees = [
        x["id"]
        for x in lignes
        if x["couverture_totale"] == 1
        and avant.get(x["id"], {}).get("couverture_totale") == 0
    ]
    perdues = [
        x["id"]
        for x in lignes
        if x["couverture_totale"] == 0
        and avant.get(x["id"], {}).get("couverture_totale") == 1
    ]
    print(f"\n  gagnees ({len(gagnees)}) : {', '.join(gagnees) or '-'}")
    print(f"  perdues ({len(perdues)}) : {', '.join(perdues) or '-'}")


def rejouer(source_nom: str, nom: str) -> None:
    """Renote des passages deja enregistres avec les etiquettes actuelles.

    Les passages renvoyes sont conserves dans le fichier de resultats : quand
    les etiquettes changent — ici, des faces elargies a leur section —, on
    renote sans relancer une seule recherche. Aucun quota consomme, et aucune
    variabilite du retrieval introduite entre les deux notations : l'ecart
    observe vient bien des etiquettes corrigees.
    """
    chemin = RESULTATS / f"multisauts-{source_nom}.json"
    source = json.loads(chemin.read_text(encoding="utf-8"))
    questions = {q["id"]: q for q in charger()}

    lignes = []
    for ancienne in source["questions"]:
        faces = questions[ancienne["id"]]["faces"]
        passages = [
            Passage(
                contenu="",
                source="",
                titre=r["document"],
                page=r["page"],
                distance=r.get("distance", 0.0),
            )
            for r in ancienne["renvoyes"]
        ]
        ligne = {
            cle: ancienne[cle]
            for cle in ("id", "question", "duree_ms", "requetes", "renvoyes")
            if cle in ancienne
        }
        ligne |= evaluer_question(faces, passages)
        lignes.append(ligne)
        manquantes = ligne["faces_manquantes"]
        marque = "x" if manquantes else "."
        detail = f"  manque : {', '.join(manquantes)}" if manquantes else ""
        print(f"  {marque} {ligne['id']}{detail}", flush=True)

    print(f"\n{len(lignes)} questions renotees depuis « {source_nom} »")
    agregats = agreger(lignes)
    afficher(agregats)
    RESULTATS.mkdir(exist_ok=True)
    sortie = RESULTATS / f"multisauts-{nom}.json"
    sortie.write_text(
        json.dumps(
            {
                "nom": nom,
                "strategie": source.get("strategie"),
                "date": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                "commit": commit_courant(),
                "golden_set": empreinte(),
                "renote_depuis": source_nom,
                "date_des_recherches": source.get("date"),
                "parametres": source.get("parametres", {}),
                "agregats": agregats,
                "questions": lignes,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  resultats : {sortie}")


async def principal(nom: str, strategie_nom: str, comparer: str | None) -> None:
    questions = charger()
    reference = None
    if comparer:
        chemin = RESULTATS / f"multisauts-{comparer}.json"
        reference = json.loads(chemin.read_text(encoding="utf-8"))
        if reference.get("golden_set") != empreinte():
            raise ValueError(
                "golden set different de celui de la reference : les chiffres "
                "ne sont pas comparables"
            )

    settings = get_settings()
    await open_pool()
    try:
        print(
            f"{len(questions)} questions multi-sauts, strategie « {strategie_nom} », "
            f"reecriture {'active' if settings.reecriture_requetes else 'inactive'}"
        )
        lignes = await mesurer(STRATEGIES[strategie_nom], questions)
    finally:
        await close_pool()
        await close_client()

    agregats = agreger(lignes)
    afficher(agregats, reference["agregats"] if reference else None)
    if reference:
        afficher_changements(lignes, reference)

    echecs = [x["id"] for x in lignes if x["couverture_totale"] == 0]
    print(f"\n  questions incompletes : {', '.join(echecs) or 'aucune'}")

    RESULTATS.mkdir(exist_ok=True)
    chemin = RESULTATS / f"multisauts-{nom}.json"
    chemin.write_text(
        json.dumps(
            {
                "nom": nom,
                "strategie": strategie_nom,
                "date": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                "commit": commit_courant(),
                "golden_set": empreinte(),
                "parametres": {
                    "k": K,
                    "reecriture_requetes": settings.reecriture_requetes,
                    "modele_reecriture": settings.mistral_reecriture_model,
                    "modele_decomposition": settings.mistral_reecriture_model,
                },
                "agregats": agregats,
                "questions": lignes,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  resultats : {chemin}")


if __name__ == "__main__":
    parseur = argparse.ArgumentParser(description="Evaluation multi-sauts")
    parseur.add_argument("--nom", required=True)
    parseur.add_argument("--strategie", default="actuel", choices=sorted(STRATEGIES))
    parseur.add_argument("--comparer", help="nom d'un resultat de reference")
    parseur.add_argument(
        "--rejouer",
        metavar="SOURCE",
        help="renoter des resultats enregistres avec les etiquettes actuelles",
    )
    args = parseur.parse_args()
    # Comme les autres mesures : pas de trace, le temps d'envoi fausserait la
    # latence et remplirait le projet Langfuse.
    desactiver_traces()
    try:
        if args.rejouer:
            # Aucune recherche ni appel au modele : pas de boucle a configurer.
            rejouer(args.rejouer, args.nom)
            raise SystemExit(0)
        configurer_boucle()
        asyncio.run(principal(args.nom, args.strategie, args.comparer))
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERREUR : {exc}", file=sys.stderr)
        sys.exit(1)
