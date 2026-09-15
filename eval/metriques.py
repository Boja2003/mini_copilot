"""Metriques de retrieval, calculees a la main.

Aucun LLM ici, et c'est un choix : si le golden set dit « la reponse est
dans III. Breadth first search », verifier que ce document remonte est une
simple comparaison. Deterministe, gratuit, reproductible — deux executions
sur les memes donnees donnent exactement le meme chiffre, ce qu'aucun juge
LLM ne garantit.

Toutes les fonctions prennent une liste de booleens : « le passage au rang
i est-il pertinent ? », dans l'ordre ou le retrieval les a renvoyes.
"""


def hit_at_k(pertinences: list[bool], k: int) -> float:
    """1 si au moins un passage pertinent figure dans les k premiers, sinon 0.

    La question de base : le bon contenu arrive-t-il sous les yeux du modele ?
    """
    return 1.0 if any(pertinences[:k]) else 0.0


def reciprocal_rank(pertinences: list[bool]) -> float:
    """1/rang du premier passage pertinent, 0 si aucun.

    Distingue « trouve en 1re position » (1.0) de « trouve en 5e » (0.2) :
    le Hit@5 compte les deux pareil, alors qu'un passage en tete de contexte
    pese bien plus dans la reponse.
    """
    for rang, pertinent in enumerate(pertinences, start=1):
        if pertinent:
            return 1.0 / rang
    return 0.0


def precision_at_k(pertinences: list[bool], k: int) -> float:
    """Part des k premiers passages qui sont pertinents.

    Mesure l'encombrement : un Hit@5 a 1 avec une precision de 0.2 veut dire
    qu'un seul passage utile se bat contre quatre hors-sujet dans le prompt.
    """
    if k <= 0:
        raise ValueError("k doit etre strictement positif")
    return sum(pertinences[:k]) / k


def moyenne(valeurs: list[float]) -> float:
    return sum(valeurs) / len(valeurs) if valeurs else 0.0
