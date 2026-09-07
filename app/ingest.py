"""Ingestion d'un dossier de documents dans la base vectorielle.

Usage :
    python -m app.ingest "C:/chemin/vers/corpus"
    python -m app.ingest "C:/chemin/vers/corpus" --force

Le pipeline : lire -> normaliser -> decouper -> vectoriser -> stocker.

Une empreinte SHA-256 du fichier est enregistree : relancer l'ingestion ne
retraite que ce qui a change. C'est ce qui rend l'operation rejouable sans
gaspiller ton quota d'embeddings ni dupliquer les chunks.
"""

import argparse
import asyncio
import hashlib
import logging
import pathlib
import sys

from pgvector import Vector

from . import chunking, parsers
from .boucle import configurer as configurer_boucle
from .db import close_pool, get_pool, init_schema, open_pool
from .embeddings import embarquer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("ingest")


def empreinte(chemin: pathlib.Path) -> str:
    h = hashlib.sha256()
    with chemin.open("rb") as f:
        for bloc in iter(lambda: f.read(1 << 20), b""):
            h.update(bloc)
    return h.hexdigest()


def fichiers_a_traiter(dossier: pathlib.Path) -> list[pathlib.Path]:
    trouves = [
        f
        for f in sorted(dossier.rglob("*"))
        if f.is_file() and f.suffix.lower() in parsers.FORMATS_SUPPORTES
    ]
    ignores = [
        f.name
        for f in sorted(dossier.rglob("*"))
        if f.is_file() and f.suffix.lower() not in parsers.FORMATS_SUPPORTES
    ]
    if ignores:
        logger.info("Ignores (format non supporte) : %s", ", ".join(ignores))
    return trouves


async def ingerer_fichier(
    chemin: pathlib.Path, racine: pathlib.Path, force: bool
) -> int:
    source = str(chemin.relative_to(racine)).replace("\\", "/")
    signature = empreinte(chemin)

    async with get_pool().connection() as conn:
        ligne = await (
            await conn.execute(
                "SELECT id, empreinte FROM documents WHERE source = %s", (source,)
            )
        ).fetchone()

        if ligne and ligne[1] == signature and not force:
            logger.info("Inchange, ignore : %s", source)
            return 0

        unites = parsers.lire(chemin)
        chunks = chunking.decouper(unites)
        if not chunks:
            logger.warning("Aucun texte exploitable : %s", source)
            return 0

        logger.info("%s -> %s chunks, vectorisation...", source, len(chunks))
        vecteurs = await embarquer([c.contenu for c in chunks])

        # Tout ou rien : si la vectorisation a reussi mais l'insertion
        # echoue a mi-chemin, on ne veut pas d'un document a moitie indexe.
        async with conn.transaction():
            if ligne:
                # ON DELETE CASCADE efface les anciens chunks : une
                # re-ingestion remplace, elle ne duplique pas.
                await conn.execute("DELETE FROM documents WHERE id = %s", (ligne[0],))
            document_id = (
                await (
                    await conn.execute(
                        "INSERT INTO documents (source, titre, format, empreinte) "
                        "VALUES (%s, %s, %s, %s) RETURNING id",
                        (
                            source,
                            chemin.stem,
                            chemin.suffix.lower().lstrip("."),
                            signature,
                        ),
                    )
                ).fetchone()
            )[0]

            async with conn.cursor() as cur:
                await cur.executemany(
                    "INSERT INTO chunks (document_id, position, page, contenu, "
                    "embedding) VALUES (%s, %s, %s, %s, %s)",
                    [
                        (document_id, c.position, c.page, c.contenu, Vector(v))
                        for c, v in zip(chunks, vecteurs, strict=True)
                    ],
                )
    return len(chunks)


async def principal(dossier: pathlib.Path, force: bool) -> None:
    if not dossier.is_dir():
        logger.error("Dossier introuvable : %s", dossier)
        sys.exit(1)

    # Le schema d'abord (hors pool), le pool ensuite : sans le type
    # `vector` en base, aucune connexion du pool ne peut s'ouvrir.
    # L'index vectoriel, lui, est cree APRES l'insertion : batir un index
    # HNSW au fil des inserts est bien plus lent que le batir une fois.
    await init_schema(avec_index_vectoriel=False)
    await open_pool()
    try:
        fichiers = fichiers_a_traiter(dossier)
        logger.info("%s fichiers a examiner", len(fichiers))
        total = 0
        for chemin in fichiers:
            total += await ingerer_fichier(chemin, dossier, force)

        logger.info("Construction de l'index vectoriel...")
        await init_schema(avec_index_vectoriel=True)

        async with get_pool().connection() as conn:
            docs = (
                await (await conn.execute("SELECT count(*) FROM documents")).fetchone()
            )[0]
            morceaux = (
                await (await conn.execute("SELECT count(*) FROM chunks")).fetchone()
            )[0]
        logger.info(
            "Termine : %s chunks ajoutes. Base : %s documents, %s chunks.",
            total,
            docs,
            morceaux,
        )
    finally:
        await close_pool()


if __name__ == "__main__":
    parseur = argparse.ArgumentParser(description="Ingestion du corpus")
    parseur.add_argument("dossier", type=pathlib.Path)
    parseur.add_argument(
        "--force", action="store_true", help="reindexe meme les fichiers inchanges"
    )
    args = parseur.parse_args()
    configurer_boucle()
    asyncio.run(principal(args.dossier, args.force))
