"""Garde-fou : .env.example doit rester synchronisé avec config.py.
Évite l'erreur 'Extra inputs are not permitted' rencontrée au Sprint 2
quand le .env contenait des variables absentes de la classe Settings."""
from pathlib import Path
from config import Settings


def test_env_example_keys_are_all_declared_in_settings():
    keys = []
    for line in Path(".env.example").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            keys.append(line.split("=", 1)[0].strip().lower())

    unknown = [k for k in keys if k not in Settings.model_fields]
    assert not unknown, f"Variables de .env.example absentes de Settings : {unknown}"


def test_settings_loads_from_env_example_without_error():
    settings = Settings(_env_file=".env.example")
    assert settings.llm_provider in ("anthropic", "ollama")
    assert settings.ollama_timeout_seconds > 0