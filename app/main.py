"""Porte d'entree HTTP : la couche FastAPI.

Elle ne fait que trois choses : valider l'entree, deleguer a `llm.py`,
traduire une panne en code HTTP correct. Aucune logique metier ici.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import get_settings
from .llm import LLMError, close_client, generate_answer

logging.basicConfig(level=logging.INFO)


class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        examples=["Explique-moi ce qu'est un embedding."],
    )


class ChatResponse(BaseModel):
    answer: str
    model: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Au demarrage : on lit la config tout de suite. Si MISTRAL_API_KEY
    # manque, l'app refuse de demarrer ici, pas a la premiere requete.
    get_settings()
    yield
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
    C'est ce que Fly.io / Railway interrogeront pour savoir si l'app vit."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        answer = await generate_answer(request.question)
    except LLMError as exc:
        # 502 et pas 500 : la faute vient d'un service en amont, pas de nous.
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return ChatResponse(answer=answer, model=get_settings().mistral_model)
