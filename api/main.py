"""
API REST de DocuMind — Sprint 6.

Expose le pipeline RAG existant :
retrieval -> garde-fou de pertinence -> génération LLM.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import settings
from src.generation import GenerationError, generate_answer

logger = logging.getLogger(__name__)

app = FastAPI(
    title="DocuMind API",
    description=(
        "API RAG pour interroger la documentation FastAPI. "
        "Les réponses sont accompagnées de leurs sources."
    ),
    version="1.0.0",
)

GENERATION_TIMEOUT = 180


# --------------------------------------------------
# Modèles de requête et de réponse
# --------------------------------------------------


class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Question à poser sur la documentation.",
        examples=["How do I create a POST route in FastAPI?"],
    )


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    refused: bool
    refusal_reason: Optional[str] = None
    language: Optional[str] = None
    min_distance: Optional[float] = None
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    latency_seconds: float = 0.0


class ErrorResponse(BaseModel):
    detail: str


# --------------------------------------------------
# Vérification de l'index
# --------------------------------------------------


def index_is_available() -> bool:
    """
    Vérifie le chemin d'index configuré dans config.py.

    Cette vérification convient à un index persistant local.
    Pour un backend distant, il faudra vérifier sa disponibilité
    avec une méthode adaptée au backend.
    """
    index_path = Path(settings.index_path)

    if not index_path.is_absolute():
        index_path = Path(__file__).resolve().parent.parent / index_path

    return index_path.exists()


# --------------------------------------------------
# Routes de santé
# --------------------------------------------------


@app.get("/", tags=["Health"])
async def root():
    return {
        "message": "DocuMind API is running",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health():
    return {
        "status": "ok",
        "index_exists": index_is_available(),
        "llm_provider": settings.llm_provider,
    }


# --------------------------------------------------
# Endpoint RAG
# --------------------------------------------------


@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["Chat"],
    responses={
        422: {"model": ErrorResponse, "description": "Question invalide"},
        503: {
            "model": ErrorResponse,
            "description": "Index ou fournisseur indisponible",
        },
        504: {"model": ErrorResponse, "description": "Délai de génération dépassé"},
        500: {"model": ErrorResponse, "description": "Erreur interne"},
    },
)
async def chat(request: ChatRequest) -> ChatResponse:
    question = request.question.strip()

    # Pydantic accepte une chaîne contenant uniquement des espaces ;
    # on vérifie donc aussi son contenu après strip().
    if not question:
        raise HTTPException(
            status_code=422,
            detail="La question ne doit pas être vide.",
        )

    # Vérification avant de charger le modèle d'embedding.
    if not index_is_available():
        raise HTTPException(
            status_code=503,
            detail="L'index documentaire est absent ou inaccessible.",
        )

    try:
        # generate_answer() est synchrone : le thread évite de bloquer
        # la boucle événementielle de FastAPI.
        result = await asyncio.wait_for(
            asyncio.to_thread(generate_answer, question),
            timeout=GENERATION_TIMEOUT,
        )

    except asyncio.TimeoutError as exc:
        logger.warning("Timeout du pipeline RAG.")
        raise HTTPException(
            status_code=504,
            detail="Le traitement de la question a dépassé le délai autorisé.",
        ) from exc

    except GenerationError as exc:
        logger.error("Erreur du pipeline RAG : %s", exc)

        message = str(exc).lower()

        # generate_answer() transforme RetrievalError en GenerationError.
        if "retrieval" in message or "index" in message or "recherche" in message:
            raise HTTPException(
                status_code=503,
                detail="La recherche documentaire est temporairement indisponible.",
            ) from exc

        # LLMProviderError est également enveloppée par GenerationError.
        if any(
            term in message
            for term in (
                "génération",
                "generation",
                "ollama",
                "anthropic",
                "connexion",
                "connection",
                "timeout",
            )
        ):
            raise HTTPException(
                status_code=503,
                detail="Le fournisseur LLM est indisponible.",
            ) from exc

        raise HTTPException(
            status_code=500,
            detail="Le pipeline RAG a rencontré une erreur.",
        ) from exc

    except Exception as exc:
        logger.exception("Erreur inattendue dans POST /chat")
        raise HTTPException(
            status_code=500,
            detail="Une erreur interne est survenue.",
        ) from exc

    # generate_answer() renvoie déjà les sources et le statut de refus.
    # On conserve directement ces informations, sans inventer de sources.
    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        refused=result["refused"],
        refusal_reason=result.get("refusal_reason"),
        language=result.get("language"),
        min_distance=result.get("min_distance"),
        llm_provider=result.get("llm_provider"),
        llm_model=result.get("llm_model"),
        latency_seconds=result.get("latency_seconds", 0.0),
    )
