"""
Détection de langue légère et déterministe (anglais, français, arabe).

Pas de bibliothèque externe et pas d'aléatoire : une règle simple, testable et
documentée. Elle sert à (1) répondre « je n'ai pas trouvé » dans la langue de
la question et (2) vérifier, dans l'évaluation, que la réponse est dans la
bonne langue.

Limites assumées : c'est une heuristique. Les textes très courts ou très
mélangés peuvent être mal classés, et toute autre langue est rangée en « en ».
"""

import re
from typing import Optional

ARABIC_RE = re.compile("[\u0600-\u06ff\u0750-\u077f\ufb50-\ufdff\ufe70-\ufeff]")
LATIN_RE = re.compile("[A-Za-z\u00c0-\u00ff]")
FRENCH_ACCENT_RE = re.compile("[àâçéèêëîïôùûüÿœæ]", re.IGNORECASE)
WORD_RE = re.compile("[a-zà-ÿœæ]+")

# Mots-outils discriminants. Les mots ambigus entre les deux langues
# (on, as, an, or, a) sont volontairement exclus.
FRENCH_WORDS = frozenset(
    {
        "le",
        "la",
        "les",
        "des",
        "du",
        "de",
        "un",
        "une",
        "est",
        "et",
        "dans",
        "pour",
        "comment",
        "que",
        "qui",
        "quoi",
        "quel",
        "quelle",
        "quels",
        "quelles",
        "avec",
        "sur",
        "pas",
        "au",
        "aux",
        "ce",
        "cette",
        "ces",
        "vous",
        "je",
        "tu",
        "il",
        "elle",
        "nous",
        "sont",
        "peut",
        "faire",
        "ou",
        "par",
        "plus",
        "si",
        "mon",
        "ma",
        "mes",
        "votre",
        "ne",
        "se",
        "sa",
        "son",
        "ses",
        "qu",
        "l",
        "j",
        "où",
        "trouvé",
        "information",
    }
)
ENGLISH_WORDS = frozenset(
    {
        "the",
        "of",
        "and",
        "to",
        "in",
        "is",
        "how",
        "what",
        "with",
        "for",
        "are",
        "can",
        "you",
        "your",
        "this",
        "that",
        "do",
        "does",
        "use",
        "using",
        "which",
        "when",
        "why",
        "it",
        "be",
        "by",
        "from",
        "have",
        "has",
        "not",
        "could",
        "find",
        "there",
        "an",
    }
)
# « information » est commun aux deux langues : on le retire des deux listes.
FRENCH_WORDS = FRENCH_WORDS - {"information"}
ENGLISH_WORDS = ENGLISH_WORDS - {"an"}

REFUSAL_MESSAGES = {
    "en": "I could not find information on this topic in the documentation.",
    "fr": "Je n'ai pas trouvé d'information sur ce sujet dans la documentation.",
    "ar": "لم أجد معلومات حول هذا الموضوع في الوثائق.",
}

_FENCED_CODE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`\n]*`")
_URL = re.compile(r"https?://\S+")
_FILE_NAME = re.compile(r"[\w./\-]+\.(?:md|py|json|toml|txt)\b")


def strip_code(text: str) -> str:
    """Retire code, URL et noms de fichiers : on ne juge la langue que sur la prose."""
    text = _FENCED_CODE.sub(" ", text or "")
    text = _INLINE_CODE.sub(" ", text)
    text = _URL.sub(" ", text)
    return _FILE_NAME.sub(" ", text)


def detect_language(text: Optional[str]) -> Optional[str]:
    """Retourne "ar", "fr" ou "en", ou None s'il n'y a aucune lettre exploitable."""
    if not text:
        return None
    arabic = len(ARABIC_RE.findall(text))
    latin = len(LATIN_RE.findall(text))
    if arabic == 0 and latin == 0:
        return None
    # Une question arabe contient souvent des termes latins (FastAPI, Pydantic) :
    # l'arabe l'emporte dès qu'il pèse au moins la moitié du latin.
    if arabic > 0 and arabic >= 0.5 * latin:
        return "ar"

    words = WORD_RE.findall(text.lower())
    french = sum(1 for w in words if w in FRENCH_WORDS)
    english = sum(1 for w in words if w in ENGLISH_WORDS)
    if FRENCH_ACCENT_RE.search(text):
        french += 2
    return "fr" if french > english else "en"


def answer_language_matches(answer: str, expected_lang: str) -> Optional[bool]:
    """True/False si la langue de la prose de la réponse est celle attendue ;
    None si la réponse ne contient pas de prose exploitable (code seul)."""
    detected = detect_language(strip_code(answer))
    if detected is None:
        return None
    return detected == expected_lang
