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
from .observabilite import (
    attributs_de_trace,
    identifiant_trace,
    observer,
)
from .observabilite import fermer as fermer_traces

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
    trace_id: str | None = Field(
        default=None,
        description=(
            "Identifiant de la trace Langfuse de cette requete : il relie une "
            "reponse signalee a sa trace complete. Null si le tracage est "
            "desactive."
        ),
    )


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
    # En dernier : les etapes des requetes deja servies partent avant l'arret.
    fermer_traces()


app = FastAPI(
    title="RAG Assistant",
    version="0.1.0",
    summary="Questions sur des supports de cours : recherche hybride, "
    "reponses sourcees, traces Langfuse.",
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
    settings = get_settings()
    strategie = "reecriture" if settings.reecriture_requetes else "hybride"

    # La trace ne commence qu'APRES l'authentification (dependance ci-dessus) :
    # une requete refusee ne produit pas de trace, donc pas de bruit ni de
    # consommation du quota Langfuse par des inconnus.
    with attributs_de_trace(
        trace_name="chat",
        tags=[strategie],
        metadata={"strategie": strategie, "modele": settings.mistral_model},
    ):
        with observer("chat", input={"question": request.question}) as etape:
            trace_id = identifiant_trace()
            try:
                reponse = await generate_answer(request.question)
            except LLMError as exc:
                etape.update(level="ERROR", status_message=str(exc))
                # 502 et pas 500 : la faute vient d'un service en amont.
                raise HTTPException(status_code=502, detail=str(exc)) from exc

            sources = [
                Source(document=p.titre, page=p.page, distance=round(p.distance, 4))
                for p in reponse.passages
            ]
            etape.update(
                output={
                    "answer": reponse.texte,
                    "sources": [s.model_dump() for s in sources],
                }
            )

    return ChatResponse(
        answer=reponse.texte,
        model=settings.mistral_model,
        sources=sources,
        trace_id=trace_id,
    )
