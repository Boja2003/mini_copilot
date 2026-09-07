"""Acces a Postgres + pgvector.

SQL ecrit a la main plutot qu'un ORM : la recherche vectorielle est du SQL
tres particulier, et le cacher derriere un ORM t'empecherait de voir ce qui
se passe vraiment — or c'est exactement ce qu'on veut apprendre ici.
"""

import logging

import psycopg
from pgvector.psycopg import register_vector_async
from psycopg_pool import AsyncConnectionPool

from .config import get_settings

logger = logging.getLogger(__name__)

# mistral-embed produit des vecteurs de dimension 1024. La colonne doit
# declarer cette taille : Postgres refusera tout vecteur d'une autre
# dimension, ce qui t'evite d'indexer par erreur avec un autre modele.
DIMENSION_EMBEDDING = 1024

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id          BIGSERIAL PRIMARY KEY,
    source      TEXT NOT NULL UNIQUE,
    titre       TEXT NOT NULL,
    format      TEXT NOT NULL,
    empreinte   TEXT NOT NULL,
    ingere_le   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    position    INT NOT NULL,
    page        INT,
    contenu     TEXT NOT NULL,
    embedding   vector({DIMENSION_EMBEDDING}) NOT NULL
);

CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks (document_id);
"""

# Index vectoriel separe : sa creation est longue, on la fait apres
# l'insertion des donnees (construire un index HNSW au fur et a mesure des
# inserts est bien plus lent que le construire une fois a la fin).
INDEX_VECTORIEL = """
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
"""

_pool: AsyncConnectionPool | None = None


async def _configurer(conn) -> None:
    """Apprend a psycopg a convertir les listes Python <-> type vector.

    `register_vector_async` interroge la base pour trouver l'OID du type
    `vector` : elle echoue donc tant que l'extension n'existe pas. Comme
    cette fonction tourne a l'ouverture de CHAQUE connexion du pool, une
    exception ici rend le pool incapable de fournir la moindre connexion —
    l'erreur se manifeste alors en PoolTimeout, qui ne dit rien de la vraie
    cause. D'ou `init_schema()`, qui cree l'extension hors du pool, avant.
    """
    await register_vector_async(conn)


def get_pool() -> AsyncConnectionPool:
    global _pool
    if _pool is None:
        _pool = AsyncConnectionPool(
            get_settings().database_url,
            min_size=1,
            max_size=5,
            open=False,
            configure=_configurer,
        )
    return _pool


async def open_pool() -> None:
    await get_pool().open()


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


async def init_schema(avec_index_vectoriel: bool = True) -> None:
    """Cree extension, tables et index. Idempotent : on peut la rejouer.

    Volontairement sur une connexion directe, PAS sur le pool : le pool ne
    peut pas s ouvrir tant que le type `vector` n existe pas (voir
    `_configurer`). C est l ordre d amorcage de la base.
    """
    async with await psycopg.AsyncConnection.connect(
        get_settings().database_url, autocommit=True
    ) as conn:
        await conn.execute(SCHEMA)
        if avec_index_vectoriel:
            await conn.execute(INDEX_VECTORIEL)
    logger.info("Schema pret")
