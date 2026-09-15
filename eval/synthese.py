"""Synthese de mesures repetees.

    python -m eval.synthese \\
        --groupe hybride=hybride-r1,hybride-r2 \\
        --groupe reecriture-3b=reecriture-3b-r1,reecriture-3b-r2,reecriture-3b-r3

Le pipeline n'est pas parfaitement reproductible (voir eval/metriques.py) :
une mesure isolee ne suffit pas a departager deux variantes proches. On
repete, puis on regarde la moyenne ET l'etendue. Si les etendues de deux
variantes se chevauchent, l'ecart entre elles n'est pas etabli.
"""

import argparse
import json
import sys

from .metriques import moyenne
from .retrieval import RESULTATS

CATEGORIES_AFFICHEES = ["TOUTES", "fr_eloigne", "fr_proche", "en"]
METRIQUES = ["hit@5", "mrr", "precision@5"]


def charger(nom: str) -> dict:
    return json.loads((RESULTATS / f"{nom}.json").read_text(encoding="utf-8"))


def verifier_comparables(resultats: list[dict]) -> None:
    """Refuse de melanger des mesures notees avec des golden sets differents.

    Un resultat sans empreinte date d'avant son enregistrement : on ne peut
    pas prouver qu'il a ete note avec les memes etiquettes.
    """
    empreintes = {r.get("golden_set") for r in resultats}
    if None in empreintes:
        sans = [r["nom"] for r in resultats if r.get("golden_set") is None]
        raise ValueError(f"Resultats sans empreinte de golden set : {sans}")
    if len(empreintes) > 1:
        raise ValueError(f"Golden sets differents : {sorted(empreintes)}")


def resumer(valeurs: list[float]) -> dict | None:
    if not valeurs:
        return None
    return {
        "moyenne": moyenne(valeurs),
        "min": min(valeurs),
        "max": max(valeurs),
        "n": len(valeurs),
    }


def se_chevauchent(a: dict, b: dict) -> bool:
    """Deux etendues [min, max] qui se recouvrent : l'ecart n'est pas etabli."""
    return not (a["max"] < b["min"] or b["max"] < a["min"])


def synthetiser(resultats: list[dict]) -> dict:
    """Moyenne et etendue de chaque metrique, par categorie, sur les repetitions."""
    synthese: dict = {}
    for categorie in CATEGORIES_AFFICHEES:
        for metrique in METRIQUES:
            valeurs = [
                r["agregats"][categorie][metrique]
                for r in resultats
                if categorie in r["agregats"]
            ]
            synthese[(categorie, metrique)] = resumer(valeurs)
    latences = [r["latence_ms"]["mediane"] for r in resultats if r.get("latence_ms")]
    synthese[("latence", "mediane_ms")] = resumer(latences)
    return synthese


def _cellule(resume: dict | None) -> str:
    if resume is None:
        return f"{'-':>21}"
    if resume["n"] == 1:
        return f"{resume['moyenne']:>21.3f}"
    return f"{resume['moyenne']:.3f} [{resume['min']:.3f}-{resume['max']:.3f}]".rjust(
        21
    )


def afficher(groupes: dict[str, dict]) -> None:
    noms = list(groupes)
    cles = [(c, m) for c in CATEGORIES_AFFICHEES for m in METRIQUES]
    cles.append(("latence", "mediane_ms"))
    entete = f"{'':<24}" + "".join(f"{nom:>22}" for nom in noms)
    print(entete)
    print("-" * len(entete))
    for categorie, metrique in cles:
        ligne = f"{categorie + ' ' + metrique:<24}"
        ligne += "".join(
            f" {_cellule(groupes[n][(categorie, metrique)])}" for n in noms
        )
        print(ligne)

    if len(noms) >= 2:
        print("\nEcarts etablis (etendues disjointes) :")
        trouve = False
        for i, a in enumerate(noms):
            for b in noms[i + 1 :]:
                for cle in cles:
                    ra, rb = groupes[a][cle], groupes[b][cle]
                    if (
                        ra
                        and rb
                        and ra["n"] > 1
                        and rb["n"] > 1
                        and not se_chevauchent(ra, rb)
                    ):
                        trouve = True
                        print(
                            f"  {a} vs {b} sur {cle[0]} {cle[1]} : "
                            f"{ra['moyenne']:.3f} contre {rb['moyenne']:.3f}"
                        )
        if not trouve:
            print("  aucun : toutes les etendues se chevauchent")


def principal(groupes_bruts: list[str]) -> None:
    groupes: dict[str, dict] = {}
    tous: list[dict] = []
    for brut in groupes_bruts:
        nom, _, noms_resultats = brut.partition("=")
        resultats = [charger(n) for n in noms_resultats.split(",") if n]
        tous.extend(resultats)
        groupes[nom] = synthetiser(resultats)
    verifier_comparables(tous)
    afficher(groupes)


if __name__ == "__main__":
    parseur = argparse.ArgumentParser(description="Synthese de mesures repetees")
    parseur.add_argument(
        "--groupe",
        action="append",
        required=True,
        help="nom=resultat1,resultat2,... (repetable)",
    )
    args = parseur.parse_args()
    try:
        principal(args.groupe)
    except ValueError as exc:
        print(f"ERREUR : {exc}", file=sys.stderr)
        sys.exit(1)
