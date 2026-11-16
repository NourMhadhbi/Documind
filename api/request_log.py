"""
Journal JSONL des requêtes /chat (Sprint 9).

Une ligne JSON par requête réussie (réponse ou refus du garde-fou).
Le texte de la question n'est JAMAIS enregistré (vie privée) : seulement
sa longueur, la langue détectée, les distances, le refus et la latence.
"""

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)

DEFAULT_LOG_PATH = "logs/requests.jsonl"
_lock = threading.Lock()


def log_path() -> Path:
    return Path(os.getenv("REQUEST_LOG_PATH", DEFAULT_LOG_PATH))


def build_record(question: str, result: Mapping[str, Any]) -> dict:
    return {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "language": result.get("language") or "unknown",
        "min_distance": result.get("min_distance"),
        "refused": bool(result.get("refused")),
        "refusal_reason": result.get("refusal_reason"),
        "latency_seconds": result.get("latency_seconds"),
        "n_sources": len(result.get("sources") or []),
        "question_length": len(question),
    }


def log_request(
    question: str, result: Mapping[str, Any], path: Optional[os.PathLike] = None
) -> None:
    """Ajoute une ligne au journal. Ne lève jamais d'exception : la journalisation
    ne doit pas faire échouer une requête."""
    try:
        record = build_record(question, result)
        target = Path(path) if path else log_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False)
        with _lock, target.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except Exception:
        logger.warning("Écriture du journal de requêtes impossible.", exc_info=True)
