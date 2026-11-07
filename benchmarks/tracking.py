"""
Suivi d'expériences MLflow (DagsHub), indépendant de config.py : il ne lit que
les variables d'environnement, ce qui permet de l'utiliser aussi dans le petit
environnement dédié à RAGAS (sans pydantic, torch, chromadb...).

Variables : MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD
(lues dans .env si python-dotenv est installé).
"""

import math
import os
import re
import sys
from typing import Dict, Iterable, Optional

_INVALID_METRIC_CHARS = re.compile(r"[^A-Za-z0-9_\-. /]")


def sanitize_metric_name(name: str) -> str:
    """MLflow n'accepte que lettres, chiffres, _ - . espace et /."""
    return _INVALID_METRIC_CHARS.sub("_", name)


def configure_mlflow():
    """Retourne le module mlflow configuré, ou lève une RuntimeError explicite."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError("mlflow n'est pas installé : `pip install mlflow`") from exc

    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        raise RuntimeError("MLFLOW_TRACKING_URI absent de l'environnement (.env)")
    mlflow.set_tracking_uri(uri)
    return mlflow


def log_run(
    experiment: str,
    run_name: str,
    params: Dict,
    metrics: Dict,
    artifacts: Optional[Iterable[str]] = None,
    mlflow_module=None,
) -> None:
    """Enregistre une exécution : paramètres, métriques (les valeurs None ou non finies
    sont ignorées) et fichiers joints."""
    # MLflow affiche des emojis : sous Windows (cp1252) cela fait planter l'impression.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    mlflow = mlflow_module or configure_mlflow()
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name):
        for key, value in params.items():
            mlflow.log_param(sanitize_metric_name(key), str(value)[:250])
        for key, value in metrics.items():
            if value is None or not math.isfinite(float(value)):
                continue
            mlflow.log_metric(sanitize_metric_name(key), float(value))
        for path in artifacts or []:
            if os.path.exists(path):
                mlflow.log_artifact(path)
