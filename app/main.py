"""Porte d'entree HTTP : la couche FastAPI.

Elle ne fait que trois choses : valider l'entree, deleguer a `llm.py`,
traduire une panne en code HTTP correct. Aucune logique metier ici.
"""

import logging
import secrets
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from .config import get_settings
from .db import close_pool, init_schema, open_pool
from .llm import generate_answer
from .mistral import LLMError, close_client

logging.basicConfig(level=logging.INFO)


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(provided: str | None = Security(api_key_header)) -> None:
    """Protege les endpoints couteux : sans ca, une fois en ligne, /chat est
    un proxy LLM gratuit offert a tout Internet, paye avec ton quota.

    `compare_digest` et pas `==` : la comparaison est a temps constant, donc
    elle ne fuit pas la cle caractere par caractere via le temps de reponse.
    """
    expected = get_settings().api_key
    if provided is None or not secrets.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Cle API manquante ou invalide (header X-API-Key)",
        )


class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        examples=["Explique-moi ce qu'est un embedding."],
    )


class Source(BaseModel):
    """D'ou vient un passage utilise pour repondre.

    Exposer les sources n'est pas cosmetique : c'est ce qui rend une reponse
    verifiable. Sans elles, l'utilisateur doit croire le modele sur parole.
    """

    document: str
    page: int | None
    distance: float


class ChatResponse(BaseModel):
    answer: str
    model: str
    sources: list[Source]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Au demarrage : on lit la config tout de suite. Si une variable
    # obligatoire manque, l'app refuse de demarrer ici, pas a la premiere
    # requete.
    get_settings()
    # Le schema avant le pool : sans le type `vector` en base, aucune
    # connexion du pool ne peut s'ouvrir (voir db._configurer).
    await init_schema()
    await open_pool()
    yield
    await close_pool()
    await close_client()


app = FastAPI(
    title="RAG Assistant",
    version="0.1.0",
    summary="Etape 1 : un endpoint /chat qui tape directement l'API LLM.",
    lifespan=lifespan,
)


@app.get("/health", tags=["infra"])
async def health() -> dict[str, str]:
    """Sonde de liveness : pas d'appel externe, doit toujours repondre vite.
    C'est ce que Fly.io interroge pour savoir si l'app vit — elle reste donc
    volontairement NON protegee par la cle API."""
    return {"status": "ok"}


@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["chat"],
    dependencies=[Depends(require_api_key)],
)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        reponse = await generate_answer(request.question)
    except LLMError as exc:
        # 502 et pas 500 : la faute vient d'un service en amont, pas de nous.
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return ChatResponse(
        answer=reponse.texte,
        model=get_settings().mistral_model,
        sources=[
            Source(document=p.titre, page=p.page, distance=round(p.distance, 4))
            for p in reponse.passages
        ],
    )
