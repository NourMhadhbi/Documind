"""
Tests unitaires du Sprint 4 (abstraction LLM). On mocke les clients
réseau (Anthropic SDK, requests pour Ollama) : jamais d'appel réel,
jamais de coût, jamais de dépendance à un serveur Ollama lancé.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.llm_provider import (
    AnthropicProvider,
    OllamaProvider,
    get_llm_provider,
    LLMProviderError,
)


# ---------------------------------------------------------------
# AnthropicProvider
# ---------------------------------------------------------------

def test_anthropic_provider_requires_api_key():
    with pytest.raises(LLMProviderError, match="api_key"):
        AnthropicProvider(api_key="", model_name="claude-sonnet-5",
                           max_tokens=100, temperature=0.0)


def test_anthropic_provider_complete_returns_text():
    with patch("anthropic.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Réponse générée")]
        mock_client.messages.create.return_value = mock_response
        mock_cls.return_value = mock_client

        provider = AnthropicProvider(api_key="fake-key", model_name="claude-sonnet-5",
                                      max_tokens=100, temperature=0.0)
        result = provider.complete("system prompt", "user prompt")

    assert result == "Réponse générée"


def test_anthropic_provider_wraps_api_errors():
    with patch("anthropic.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = RuntimeError("panne API simulée")
        mock_cls.return_value = mock_client

        provider = AnthropicProvider(api_key="fake-key", model_name="claude-sonnet-5",
                                      max_tokens=100, temperature=0.0)
        with pytest.raises(LLMProviderError, match="Anthropic"):
            provider.complete("system", "user")


# ---------------------------------------------------------------
# OllamaProvider
# ---------------------------------------------------------------

def test_ollama_provider_complete_returns_text():
    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {"message": {"content": "Réponse locale"}}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        provider = OllamaProvider(base_url="http://localhost:11434", model_name="qwen2.5:3b",
                                   max_tokens=100, temperature=0.0)
        result = provider.complete("system prompt", "user prompt")

    assert result == "Réponse locale"
    call_kwargs = mock_post.call_args.kwargs
    assert call_kwargs["json"]["model"] == "qwen2.5:3b"


def test_ollama_provider_connection_error_is_explicit():
    """Si Ollama n'est pas lancé, le message d'erreur doit guider
    l'utilisateur vers la solution ('ollama serve'), pas juste
    afficher une erreur réseau brute incompréhensible."""
    import requests as real_requests

    with patch("requests.post", side_effect=real_requests.exceptions.ConnectionError("refused")):
        provider = OllamaProvider(base_url="http://localhost:11434", model_name="qwen2.5:3b",
                                   max_tokens=100, temperature=0.0)
        with pytest.raises(LLMProviderError, match="ollama serve"):
            provider.complete("system", "user")


def test_ollama_provider_malformed_response_raises_clear_error():
    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {"unexpected": "shape"}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        provider = OllamaProvider(base_url="http://localhost:11434", model_name="qwen2.5:3b",
                                   max_tokens=100, temperature=0.0)
        with pytest.raises(LLMProviderError, match="champ manquant"):
            provider.complete("system", "user")


# ---------------------------------------------------------------
# Factory get_llm_provider() : le point d'entrée utilisé par generation.py
# ---------------------------------------------------------------

def test_factory_returns_anthropic_provider():
    fake_settings = MagicMock()
    fake_settings.llm_provider = "anthropic"
    fake_settings.anthropic_api_key = "fake-key"
    fake_settings.llm_model_name = "claude-sonnet-5"
    fake_settings.llm_max_tokens = 100
    fake_settings.llm_temperature = 0.0

    provider = get_llm_provider(fake_settings)
    assert isinstance(provider, AnthropicProvider)


def test_factory_returns_ollama_provider():
    fake_settings = MagicMock()
    fake_settings.llm_provider = "ollama"
    fake_settings.ollama_base_url = "http://localhost:11434"
    fake_settings.ollama_model_name = "qwen2.5:3b"
    fake_settings.llm_max_tokens = 100
    fake_settings.llm_temperature = 0.0

    provider = get_llm_provider(fake_settings)
    assert isinstance(provider, OllamaProvider)


def test_factory_unknown_provider_raises():
    fake_settings = MagicMock()
    fake_settings.llm_provider = "gpt4"  # non supporté

    with pytest.raises(LLMProviderError, match="inconnu"):
        get_llm_provider(fake_settings)


# ---------------------------------------------------------------
# Ajouts : température optionnelle et timeout configurable
# ---------------------------------------------------------------

def test_anthropic_omits_temperature_when_none():
    """Certains modèles récents refusent le paramètre temperature :
    None doit signifier 'ne pas l'envoyer du tout'."""
    with patch("anthropic.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = MagicMock(content=[MagicMock(text="ok")])
        mock_cls.return_value = mock_client

        provider = AnthropicProvider(api_key="k", model_name="m", max_tokens=10, temperature=None)
        provider.complete("sys", "usr")

    assert "temperature" not in mock_client.messages.create.call_args.kwargs


def test_ollama_omits_temperature_when_none():
    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {"message": {"content": "ok"}}
        mock_post.return_value = mock_response

        provider = OllamaProvider("http://localhost:11434", "qwen2.5:3b", 10, None)
        provider.complete("sys", "usr")

    options = mock_post.call_args.kwargs["json"]["options"]
    assert "temperature" not in options
    assert options["num_predict"] == 10


def test_ollama_timeout_comes_from_settings():
    fake_settings = MagicMock()
    fake_settings.llm_provider = "ollama"
    fake_settings.ollama_base_url = "http://localhost:11434"
    fake_settings.ollama_model_name = "qwen2.5:3b"
    fake_settings.llm_max_tokens = 100
    fake_settings.llm_temperature = 0.0
    fake_settings.ollama_timeout_seconds = 300

    provider = get_llm_provider(fake_settings)
    assert provider._timeout_seconds == 300