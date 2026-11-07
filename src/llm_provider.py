"""
Abstraction du fournisseur LLM -- même pattern Strategy/Adapter que
VectorStore (Sprint 2). generate.py ne doit JAMAIS appeler directement
l'API Anthropic ou Ollama : il dépend de l'interface LLMProvider, pas
d'une implémentation précise. Conséquence concrète : comparer Claude
Sonnet 5, Qwen2.5 3B et Llama 3.2 3B (via Ollama) ne demande qu'un
changement de config, jamais une modification de generation.py.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


class LLMProviderError(Exception):
    """Erreur métier du fournisseur LLM (distincte des exceptions
    internes de chaque SDK/API, pour ne pas fuiter de détail
    d'implémentation vers generation.py)."""

    pass


class LLMProvider(ABC):
    """Interface commune que tout fournisseur LLM doit respecter."""

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Envoie le prompt système et le prompt utilisateur, retourne
        le texte de la réponse générée."""
        raise NotImplementedError


class AnthropicProvider(LLMProvider):
    """Fournisseur cloud : API Anthropic (Claude)."""

    def __init__(
        self,
        api_key: str,
        model_name: str,
        max_tokens: int,
        temperature: Optional[float],
    ):
        import anthropic

        if not api_key:
            raise LLMProviderError("anthropic_api_key manquant dans la config")

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model_name = model_name
        self._max_tokens = max_tokens
        self._temperature = temperature

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        params = {
            "model": self._model_name,
            "max_tokens": self._max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        if self._temperature is not None:  # n'envoyer que si défini
            params["temperature"] = self._temperature

        try:
            response = self._client.messages.create(**params)
        except Exception as exc:
            raise LLMProviderError(f"Échec de l'appel à l'API Anthropic : {exc}") from exc

        return response.content[0].text


class OllamaProvider(LLMProvider):
    """Fournisseur local : modèle servi par Ollama (ex: Qwen2.5 3B,
    Llama 3.2 3B), via son API HTTP locale -- pas de dépendance SDK
    supplémentaire, juste `requests`."""

    def __init__(
        self,
        base_url: str,
        model_name: str,
        max_tokens: int,
        temperature: Optional[float],
        timeout_seconds: int = 360,
    ):
        import requests  # import local

        self._requests = requests
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._timeout_seconds = timeout_seconds

    def _options(self) -> dict:
        options = {"num_predict": self._max_tokens}
        if self._temperature is not None:
            options["temperature"] = self._temperature
        return options

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self._requests.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": self._model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                    "options": self._options(),
                },
                timeout=self._timeout_seconds,
            )
            response.raise_for_status()
        except self._requests.exceptions.ConnectionError as exc:
            raise LLMProviderError(
                f"Impossible de joindre Ollama à {self._base_url} -- "
                f"le serveur Ollama est-il lancé ? ('ollama serve') : {exc}"
            ) from exc
        except self._requests.exceptions.Timeout as exc:
            raise LLMProviderError(
                f"Ollama n'a pas répondu dans les {self._timeout_seconds}s : {exc}"
            ) from exc
        except Exception as exc:
            raise LLMProviderError(f"Échec de l'appel à Ollama : {exc}") from exc

        data = response.json()
        try:
            return data["message"]["content"]
        except KeyError as exc:
            raise LLMProviderError(
                f"Réponse Ollama inattendue, champ manquant : {exc}. Réponse brute : {data}"
            ) from exc


def get_llm_provider(settings) -> LLMProvider:
    """Factory : point d'entrée UNIQUE utilisé par generation.py,
    identique dans l'esprit à get_vector_store() du Sprint 2."""
    backend = settings.llm_provider

    if backend == "anthropic":
        return AnthropicProvider(
            api_key=settings.anthropic_api_key,
            model_name=settings.llm_model_name,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
        )
    if backend == "ollama":
        return OllamaProvider(
            base_url=settings.ollama_base_url,
            model_name=settings.ollama_model_name,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            timeout_seconds=settings.ollama_timeout_seconds,
        )

    raise LLMProviderError(f"Fournisseur LLM inconnu ou non implémenté : {backend}")
