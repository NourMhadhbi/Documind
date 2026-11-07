"""
Ingestion : lit la documentation Markdown, la nettoie, la découpe en chunks.

Nettoyage corrigé (Sprint 4) : trois défauts de la première version, visibles
dans les chunks réels, sont réparés :
  1. les underscores des identifiants étaient supprimés (user_id -> userid) ;
  2. les directives `{* fichier.py *}` et les marqueurs `///` restaient dans le texte ;
  3. les blocs de code étaient abîmés (``` -> `) et leur indentation perdue.

Les ancres de titre `{ #mon-ancre }` sont CONSERVÉES dans le texte indexé : l'ablation
du Sprint 4 (ADR 004) a montré que les supprimer dégrade le retrieval en anglais
(Recall@5 1.00 -> 0.60). Elles sont retirées plus tard, au moment de construire le
prompt envoyé au LLM (generation.build_prompt), pour qu'il ne les recopie pas.

Principe : le code (blocs ``` et `inline`) est MIS À L'ABRI avant de nettoyer
le texte courant, puis remis en place. Les règles Markdown ne peuvent donc plus
toucher l'intérieur du code.
"""

import html
import json
import os
import re
from pathlib import Path
from typing import Dict, List

# --- Éléments propres à MkDocs / FastAPI ---------------------------------
RE_INCLUDE_DIRECTIVE = re.compile(r"\{\*.*?\*\}")  # {* ../../docs_src/x.py hl[1] *}
RE_HEADING_ANCHOR = re.compile(r"[ \t]*\{[ \t]*#[\w\-.:]+[ \t]*\}")  # { #first-steps }
RE_ADMONITION_SLASHES = re.compile(r"^[ \t]*///.*$", re.MULTILINE)  # /// tip ... ///
RE_ADMONITION_BANGS = re.compile(r"^!!!.*$", re.MULTILINE)  # !!! note

# --- Code à protéger -------------------------------------------------------
RE_CODE_BLOCK = re.compile(r"```.*?```", re.DOTALL)
RE_INLINE_CODE = re.compile(r"`[^`\n]+`")
RE_PLACEHOLDER = re.compile("\ue000(\\d+)\ue001")

# --- Markdown courant ------------------------------------------------------
RE_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
RE_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
RE_HEADER = re.compile(r"^#{1,6}[ \t]*", re.MULTILINE)
# Gras / italique : les délimiteurs doivent être aux frontières d'un mot, ce
# qui protège les identifiants comme user_id ou snake_case_name.
RE_BOLD_STAR = re.compile(r"\*\*(?!\s)(.+?)(?<!\s)\*\*")
RE_ITALIC_STAR = re.compile(r"(?<![\*\w])\*(?![\s\*])(.+?)(?<![\s\*])\*(?![\*\w])")
RE_BOLD_UNDERSCORE = re.compile(r"(?<!\w)__(?![\s_])(.+?)(?<![\s_])__(?!\w)")
RE_ITALIC_UNDERSCORE = re.compile(r"(?<!\w)_(?![\s_])(.+?)(?<![\s_])_(?!\w)")
RE_HTML_TAG = re.compile(r"<[^>]+>")
RE_HR = re.compile(r"^-{3,}[ \t]*$", re.MULTILINE)
RE_MULTI_SPACE = re.compile(r"[ \t]+")
RE_MULTI_BLANK_LINES = re.compile(r"\n{3,}")


def clean_text(text: str, keep_code_blocks: bool = False, strip_anchors: bool = False) -> str:
    """
    Nettoie le texte brut d'un document Markdown avant chunking.

    Args:
        text: contenu brut du fichier Markdown.
        keep_code_blocks: True conserve les blocs ``` (indentation comprise) ;
            False les supprime. Configuration validée (ADR 002) : True.
        strip_anchors: True retire les ancres de titre `{ #...}`. Défaut False
            (configuration retenue, voir ADR 004) ; True sert aux expériences.

    Returns:
        Texte nettoyé, prêt pour le chunking.
    """
    if not text:
        return ""

    # 1. Directives d'inclusion de fichiers : elles pointent vers du code
    #    stocké ailleurs (docs_src/) et n'apportent aucun sens.
    text = RE_INCLUDE_DIRECTIVE.sub("", text)

    # 2. Mettre le code à l'abri : blocs d'abord, puis code en ligne.
    protected: List[str] = []

    def _protect(match: "re.Match") -> str:
        protected.append(match.group(0))
        return f"\ue000{len(protected) - 1}\ue001"

    text = RE_CODE_BLOCK.sub(_protect, text)
    text = RE_INLINE_CODE.sub(_protect, text)

    # 3. Nettoyage du texte courant (le code est invisible pour ces règles).
    text = RE_IMAGE.sub("", text)
    text = RE_LINK.sub(r"\1", text)
    if strip_anchors:
        text = RE_HEADING_ANCHOR.sub("", text)
    text = RE_HEADER.sub("", text)
    text = RE_ADMONITION_SLASHES.sub("", text)
    text = RE_ADMONITION_BANGS.sub("", text)
    text = RE_BOLD_STAR.sub(r"\1", text)
    text = RE_ITALIC_STAR.sub(r"\1", text)
    text = RE_BOLD_UNDERSCORE.sub(r"\1", text)
    text = RE_ITALIC_UNDERSCORE.sub(r"\1", text)
    text = RE_HTML_TAG.sub(" ", text)
    text = RE_HR.sub("", text)
    text = RE_MULTI_SPACE.sub(" ", text)  # sans danger : le code est à l'abri

    # 4. Remettre le code en place.
    def _restore(match: "re.Match") -> str:
        original = protected[int(match.group(1))]
        if original.startswith("```"):
            if not keep_code_blocks:
                return " "
            return RE_HTML_TAG.sub("", original)  # balises <span> des sorties console
        return original.strip("`")  # code en ligne : garder le contenu

    text = RE_PLACEHOLDER.sub(_restore, text)

    # 5. Finitions.
    text = html.unescape(text)  # &apos; -> '
    text = RE_MULTI_BLANK_LINES.sub("\n\n", text)
    return text.strip()


def load_markdown_files(source_dir: str) -> List[Dict]:
    """Lit récursivement les .md ; "filename" est le chemin relatif (évite les
    collisions entre plusieurs index.md)."""
    files = []
    for file in Path(source_dir).rglob("*.md"):
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
        relative_path = file.relative_to(source_dir)
        files.append({"filename": str(relative_path).replace("\\", "/"), "content": content})
    return files


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """Découpe en chunks de chunk_size caractères avec chevauchement."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap doit être strictement inférieur à chunk_size")
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i : i + chunk_size])
        i += chunk_size - chunk_overlap
    return chunks


def build_chunks(
    documents: List[Dict],
    chunk_size: int,
    chunk_overlap: int,
    keep_code_blocks: bool = False,
) -> List[Dict]:
    """Nettoie puis découpe chaque document ; garde la source de chaque chunk."""
    chunks = []
    for doc in documents:
        filename = doc["filename"]
        cleaned_content = clean_text(doc["content"], keep_code_blocks=keep_code_blocks)
        if not cleaned_content:
            continue
        for index, chunk in enumerate(chunk_text(cleaned_content, chunk_size, chunk_overlap)):
            chunks.append({"chunk_id": f"{filename}_{index}", "source": filename, "text": chunk})
    return chunks


if __name__ == "__main__":
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100
    KEEP_CODE_BLOCKS = True  # configuration validée (ADR 002, Exp3)

    docs = load_markdown_files("data/raw")
    print(f"Documents chargés: {len(docs)}")

    chunks = build_chunks(docs, CHUNK_SIZE, CHUNK_OVERLAP, KEEP_CODE_BLOCKS)
    print(f"Chunks générés : {len(chunks)}")

    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print("Sauvegardé : data/processed/chunks.json")
