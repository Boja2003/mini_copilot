"""Evaluation du retrieval sur le golden set.

    python -m eval.retrieval --nom baseline
    python -m eval.retrieval --nom essai --comparer baseline

Chaque execution enregistre ses resultats dans eval/resultats/<nom>.json,
avec le commit git et les parametres du retrieval : un chiffre sans la
configuration qui l'a produit est inverifiable.
"""

import argparse
import asyncio
import datetime as dt
import hashlib
import json
import logging
import pathlib
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable

import yaml

from app import retrieval
from app.boucle import configurer as configurer_boucle
from app.config import get_settings
from app.db import close_pool, get_pool, open_pool
from app.mistral import close_client
from app.observabilite import desactiver as desactiver_traces
from app.retrieval import Passage

from .metriques import hit_at_k, moyenne, precision_at_k, reciprocal_rank

logging.basicConfig(level=logging.WARNING)

RACINE = pathlib.Path(__file__).parent
GOLDEN_SET = RACINE / "golden_set.yaml"
RESULTATS = RACINE / "resultats"
K = 5

CATEGORIES = {"fr_proche", "fr_eloigne", "en", "hors_sujet"}

# Une strategie renvoie les passages ET les requetes effectivement lancees.
Strategie = Callable[[str, int], Awaitable[tuple[list[Passage], list[str]]]]


async def _hybride(question: str, nb: int) -> tuple[list[Passage], list[str]]:
    return await retrieval.chercher(question, nb), [question]


async def _reecriture(question: str, nb: int) -> tuple[list[Passage], list[str]]:
    # Meme chemin que la production, en version detaillee. La reecriture
    # n'est pas reproductible : rappeler le LLM apres coup ne redonne pas
    # forcement les requetes que la mesure a vues. On les enregistre donc au
    # moment de la mesure.
    detail = await retrieval.chercher_avec_reecriture_detail(question, nb)
    return detail.passages, detail.requetes


# Les variantes de retrieval a comparer. En ajouter une = l'inscrire ici.
STRATEGIES: dict[str, Strategie] = {
    "hybride": _hybride,
    "reecriture": _reecriture,
}


def charger_golden_set(chemin: pathlib.Path = GOLDEN_SET) -> list[dict]:
    """Charge le golden set et refuse toute incoherence de structure.

    Une erreur ici (identifiant en double, page rattachee a un document non
    attendu, paire orpheline) fausserait les chiffres sans rien signaler.
    """
    questions = yaml.safe_load(chemin.read_text(encoding="utf-8"))
    ids = [q["id"] for q in questions]
    doublons = sorted({i for i in ids if ids.count(i) > 1})
    if doublons:
        raise ValueError(f"Identifiants en double : {doublons}")

    for q in questions:
        if q["categorie"] not in CATEGORIES:
            raise ValueError(f"{q['id']} : categorie inconnue « {q['categorie']} »")
        documents = q.get("documents") or []
        if q["categorie"] == "hors_sujet" and documents:
            raise ValueError(f"{q['id']} : une question hors-sujet n'attend rien")
        if q["categorie"] != "hors_sujet" and not documents:
            raise ValueError(f"{q['id']} : aucun document attendu")
        orphelines = set(q.get("pages") or {}) - set(documents)
        if orphelines:
            raise ValueError(f"{q['id']} : pages hors documents attendus {orphelines}")
        if q.get("paire") and q["paire"] not in ids:
            raise ValueError(f"{q['id']} : paire inconnue « {q['paire']} »")
    return questions


async def verifier_titres(questions: list[dict]) -> None:
    """Refuse un golden set qui cite un document absent de la base.

    Une faute de frappe dans un titre donnerait silencieusement 0 partout
    pour cette question : le pire des bugs d'evaluation, celui qui ressemble
    a un vrai resultat.
    """
    async with get_pool().connection() as conn:
        lignes = await (await conn.execute("SELECT titre FROM documents")).fetchall()
    en_base = {r[0] for r in lignes}
    inconnus = sorted(
        (q["id"], d)
        for q in questions
        for d in q.get("documents") or []
        if d not in en_base
    )
    if inconnus:
        raise ValueError(f"Documents inconnus en base : {inconnus}")


def empreinte_golden_set(chemin: pathlib.Path = GOLDEN_SET) -> str:
    """Empreinte des etiquettes utilisees pour noter.

    Deux resultats ne se comparent que s'ils ont ete notes avec le meme
    golden set : corriger une etiquette change le score sans que le
    retrieval ait bouge. Fins de ligne normalisees, pour que git (CRLF sous
    Windows, LF ailleurs) ne change pas l'empreinte d'un contenu identique.
    """
    contenu = chemin.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(contenu.encode("utf-8")).hexdigest()[:12]


def commit_courant() -> str:
    """Commit courant, suffixe si le code a des modifications non commitees.

    Sans ce suffixe, un resultat produit par du code non commite pretend
    venir du dernier commit. C'est arrive : les premieres mesures de la
    reecriture de requete ont ete enregistrees sous b82a66a, un commit qui
    ne contenait pas encore la reecriture. Les resultats eux-memes sont
    exclus du controle : sinon chaque execution rendrait la suivante
    « modifiee ».
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        modifications = subprocess.run(
            ["git", "status", "--porcelain", "--", ".", ":(exclude)eval/resultats"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "inconnu"
    return f"{commit}+modifications" if modifications else commit


async def evaluer(strategie: Strategie, questions: list[dict]) -> list[dict]:
    lignes = []
    for q in questions:
        debut = time.perf_counter()
        passages, requetes = await strategie(q["question"], K)
        duree_ms = (time.perf_counter() - debut) * 1000
        attendus = set(q.get("documents") or [])
        pages_attendues = {
            (doc, p) for doc, pages in (q.get("pages") or {}).items() for p in pages
        }
        pert_doc = [p.titre in attendus for p in passages]
        pert_page = [(p.titre, p.page) in pages_attendues for p in passages]

        ligne = {
            "id": q["id"],
            "categorie": q["categorie"],
            "question": q["question"],
            "duree_ms": round(duree_ms),
            "requetes": requetes,
            "renvoyes": [
                {"document": p.titre, "page": p.page, "distance": round(p.distance, 4)}
                for p in passages
            ],
        }
        if attendus:
            ligne["hit_doc"] = hit_at_k(pert_doc, K)
            ligne["rr_doc"] = reciprocal_rank(pert_doc)
            ligne["precision_doc"] = precision_at_k(pert_doc, K)
            if pages_attendues:
                ligne["hit_page"] = hit_at_k(pert_page, K)
        lignes.append(ligne)

        marque = "x" if ligne.get("hit_doc") == 0 else "."
        print(f"  {marque} {q['id']}", flush=True)
    return lignes


def agreger(lignes: list[dict]) -> dict:
    """Moyennes globales et par categorie. Le hors-sujet n'a pas de document
    attendu : il est exclu des moyennes de retrieval."""
    par_cat: dict[str, list[dict]] = defaultdict(list)
    for ligne in lignes:
        if "hit_doc" in ligne:
            par_cat[ligne["categorie"]].append(ligne)
            par_cat["TOUTES"].append(ligne)

    def resume(groupe: list[dict]) -> dict:
        avec_pages = [g["hit_page"] for g in groupe if "hit_page" in g]
        return {
            "n": len(groupe),
            "hit@5": round(moyenne([g["hit_doc"] for g in groupe]), 3),
            "mrr": round(moyenne([g["rr_doc"] for g in groupe]), 3),
            "precision@5": round(moyenne([g["precision_doc"] for g in groupe]), 3),
            "hit_page@5": round(moyenne(avec_pages), 3) if avec_pages else None,
        }

    return {cat: resume(groupe) for cat, groupe in sorted(par_cat.items())}


def _cellule(valeur: float | None, reference: float | None) -> str:
    if valeur is None:
        return f"{'-':>11}"
    texte = f"{valeur:.3f}"
    if reference is not None and abs(valeur - reference) >= 0.0005:
        texte += f" ({valeur - reference:+.2f})"
    return f"{texte:>15}"


def afficher(agregats: dict, reference: dict | None = None) -> None:
    colonnes = ["hit@5", "mrr", "precision@5", "hit_page@5"]
    entete = f"{'categorie':<12} {'n':>3} " + " ".join(f"{c:>15}" for c in colonnes)
    print("\n" + entete)
    print("-" * len(entete))
    for cat, m in agregats.items():
        ref = (reference or {}).get(cat, {})
        cellules = " ".join(_cellule(m[c], ref.get(c)) for c in colonnes)
        print(f"{cat:<12} {m['n']:>3} {cellules}")


def latences(lignes: list[dict]) -> dict:
    """Mediane et 90e centile du temps de retrieval par question.

    La mediane plutot que la moyenne : une pause de back-off sur quota
    (429) gonfle quelques mesures et fausserait une moyenne. Le temps
    inclut les appels reseau (LLM, embeddings, base) : c'est la latence
    que l'utilisateur paiera, pas un temps de calcul local."""
    durees = sorted(x["duree_ms"] for x in lignes)
    if not durees:
        return {"mediane": None, "p90": None}
    p90 = durees[min(len(durees) - 1, int(0.9 * len(durees)))]
    return {"mediane": durees[len(durees) // 2], "p90": p90}


def afficher_paires(questions: list[dict], lignes: list[dict]) -> None:
    """Meme besoin d'information, pose en francais puis en anglais : l'ecart
    entre les deux isole l'effet de la langue, toutes choses egales par
    ailleurs."""
    par_id = {ligne["id"]: ligne for ligne in lignes}
    paires = [(q["id"], q["paire"]) for q in questions if q.get("paire")]
    if not paires:
        return
    print("\nPaires FR / EN (meme besoin, deux langues)")
    for fr, en in paires:
        a, b = par_id[fr], par_id[en]
        print(
            f"  {fr:<22} hit={a['hit_doc']:.0f} rr={a['rr_doc']:.2f}"
            f"   |   {en:<14} hit={b['hit_doc']:.0f} rr={b['rr_doc']:.2f}"
        )


def afficher_changements(lignes: list[dict], reference: dict) -> None:
    avant = {ligne["id"]: ligne for ligne in reference["questions"]}

    def etat(ident: str) -> float | None:
        return avant.get(ident, {}).get("hit_doc")

    gagnees = [x["id"] for x in lignes if x.get("hit_doc") == 1 and etat(x["id"]) == 0]
    perdues = [x["id"] for x in lignes if x.get("hit_doc") == 0 and etat(x["id"]) == 1]
    print(f"\nGagnees ({len(gagnees)}) : {', '.join(gagnees) or '-'}")
    print(f"Perdues ({len(perdues)}) : {', '.join(perdues) or '-'}")


async def principal(nom: str, strategie_nom: str, comparer: str | None) -> None:
    questions = charger_golden_set()
    reference = None
    if comparer:
        chemin_ref = RESULTATS / f"{comparer}.json"
        reference = json.loads(chemin_ref.read_text(encoding="utf-8"))

    await open_pool()
    try:
        await verifier_titres(questions)
        print(f"{len(questions)} questions, strategie « {strategie_nom} »")
        lignes = await evaluer(STRATEGIES[strategie_nom], questions)
    finally:
        await close_pool()
        await close_client()

    agregats = agreger(lignes)
    afficher(agregats, reference["agregats"] if reference else None)
    afficher_paires(questions, lignes)
    latence = latences(lignes)
    ref_latence = (reference or {}).get("latence_ms")
    suffixe = (
        f"  (reference : {ref_latence['mediane']} / {ref_latence['p90']} ms)"
        if ref_latence
        else ""
    )
    print(
        f"\nLatence du retrieval : mediane {latence['mediane']} ms, "
        f"p90 {latence['p90']} ms{suffixe}"
    )
    if reference:
        afficher_changements(lignes, reference)

    echecs = [x["id"] for x in lignes if x.get("hit_doc") == 0]
    print(f"\nEchecs (hit@5 = 0) : {', '.join(echecs) or 'aucun'}")

    RESULTATS.mkdir(exist_ok=True)
    settings = get_settings()
    sortie = {
        "nom": nom,
        "strategie": strategie_nom,
        "date": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "commit": commit_courant(),
        "golden_set": empreinte_golden_set(),
        "parametres": {
            "k": K,
            "nb_candidats": retrieval.NB_CANDIDATS,
            "k_rrf": retrieval.K_RRF,
            "distance_maximale": retrieval.DISTANCE_MAXIMALE,
            "modele_reecriture": settings.mistral_reecriture_model,
            "modele_embeddings": settings.mistral_embed_model,
        },
        "latence_ms": latences(lignes),
        "agregats": agregats,
        "questions": lignes,
    }
    chemin = RESULTATS / f"{nom}.json"
    chemin.write_text(
        json.dumps(sortie, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Resultats : {chemin.relative_to(RACINE.parent)}")


if __name__ == "__main__":
    parseur = argparse.ArgumentParser(description="Evaluation du retrieval")
    parseur.add_argument("--nom", required=True, help="nom du fichier de resultats")
    parseur.add_argument("--strategie", default="hybride", choices=sorted(STRATEGIES))
    parseur.add_argument("--comparer", help="nom d'un resultat de reference")
    args = parseur.parse_args()
    # Les mesures ne tracent pas : 63 questions par execution rempliraient le
    # projet Langfuse, et le temps d'envoi fausserait la latence mesuree.
    desactiver_traces()
    configurer_boucle()
    try:
        asyncio.run(principal(args.nom, args.strategie, args.comparer))
    except ValueError as exc:
        print(f"ERREUR : {exc}", file=sys.stderr)
        sys.exit(1)
