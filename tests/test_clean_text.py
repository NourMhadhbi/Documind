"""
Tests du nettoyage Markdown. Les exemples viennent des vrais fichiers de la
documentation FastAPI (first-steps.md, path-params.md, body.md).
"""

from src.ingestion import chunk_text, clean_text

# ---- Défaut 1 : les identifiants gardent leurs underscores -----------------


def test_snake_case_identifiers_keep_underscores():
    raw = (
        "Otherwise, the path for `/users/{user_id}` would match also for `/users/me`, "
        'receiving a parameter `user_id` with a value of "me". The value of item_id is passed.'
    )
    cleaned = clean_text(raw)
    assert "{user_id}" in cleaned
    assert "parameter user_id" in cleaned
    assert "item_id" in cleaned
    assert "userid" not in cleaned and "itemid" not in cleaned


def test_dunder_names_in_code_spans_are_preserved():
    cleaned = clean_text("directories with `__init__.py` files")
    assert "__init__.py" in cleaned


# ---- Défaut 2 : ancres de titre et directives d'inclusion ------------------


def test_heading_anchor_is_kept_by_default():
    """Configuration retenue (ADR 004) : les ancres aident le retrieval."""
    assert clean_text("# First Steps { #first-steps }") == "First Steps { #first-steps }"
    assert clean_text("#### Path { #path }") == "Path { #path }"


def test_heading_anchor_can_be_stripped_on_request():
    assert clean_text("# First Steps { #first-steps }", strip_anchors=True) == "First Steps"
    assert clean_text("#### Path { #path }", strip_anchors=True) == "Path"


def test_include_directive_is_removed():
    cleaned = clean_text(
        "Copy it.\n\n{* ../../docs_src/first_steps/tutorial001_py310.py hl[1] *}\n\nRun it."
    )
    assert "docs_src" not in cleaned
    assert "Copy it." in cleaned and "Run it." in cleaned


def test_admonition_markers_are_removed_but_content_is_kept():
    raw = "/// tip\n\nThis will give you editor support inside of your function.\n\n///\n\nNext paragraph."
    cleaned = clean_text(raw)
    assert "///" not in cleaned
    assert "editor support" in cleaned and "Next paragraph." in cleaned


# ---- Défaut 3 : les blocs de code restent intacts --------------------------

CODE_DOC = 'Example:\n\n```python\ndef read_item(item_id: int):\n    return {"item_id": item_id}\n```\n\nDone.'


def test_code_block_is_kept_intact_with_indentation():
    cleaned = clean_text(CODE_DOC, keep_code_blocks=True)
    assert "```python" in cleaned
    assert cleaned.count("```") == 2  # fences intactes
    assert '    return {"item_id": item_id}' in cleaned  # indentation conservée
    assert "item_id: int" in cleaned  # underscores conservés


def test_code_block_is_removed_when_disabled():
    cleaned = clean_text(CODE_DOC, keep_code_blocks=False)
    assert "def read_item" not in cleaned
    assert "```" not in cleaned
    assert "Example:" in cleaned and "Done." in cleaned


def test_no_stray_backticks_left_in_prose():
    cleaned = clean_text("Use `Path` and `Query` here, and `/users/me` too.")
    assert "`" not in cleaned
    assert "Path" in cleaned and "/users/me" in cleaned


def test_html_spans_inside_console_blocks_are_removed():
    raw = '```console\n$ fastapi dev\n  <span style="color:red"> FastAPI </span> Starting &apos;server&apos;\n```'
    cleaned = clean_text(raw, keep_code_blocks=True)
    assert "<span" not in cleaned
    assert "Starting 'server'" in cleaned


# ---- Markdown courant (non-régression) -------------------------------------


def test_bold_and_italic_are_stripped():
    assert (
        clean_text("You can define a **path operation** like this.")
        == "You can define a path operation like this."
    )
    assert clean_text("This is *important* text.") == "This is important text."


def test_bullet_list_markers_are_not_treated_as_italic():
    cleaned = clean_text("* `POST`: to create data.\n* `GET`: to read data.")
    assert "POST: to create data." in cleaned
    assert "GET: to read data." in cleaned


def test_links_keep_text_and_drop_url():
    assert clean_text("See [Pydantic](https://pydantic.dev/docs/) docs.") == "See Pydantic docs."
    assert clean_text("Use [`Depends`](../dependencies.md) now.") == "Use Depends now."


def test_images_are_removed():
    cleaned = clean_text("Before ![Swagger UI](https://example.com/a.png) after")
    assert "Swagger" not in cleaned and "Before" in cleaned and "after" in cleaned


def test_empty_input_returns_empty_string():
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_clean_text_is_idempotent_on_realistic_sample():
    raw = (
        "# Path Parameters { #path-parameters }\n\nThe value of `item_id` is **required**.\n\n"
        "{* ../../docs_src/p.py hl[6] *}\n\n/// tip\n\nUse `user_id`.\n\n///\n\n" + CODE_DOC
    )
    once = clean_text(raw, keep_code_blocks=True)
    assert clean_text(once, keep_code_blocks=True) == once


# ---- Chunking (inchangé, vérifie la validation ajoutée) --------------------


def test_chunk_text_rejects_overlap_not_smaller_than_size():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=10, chunk_overlap=10)


def test_chunk_text_overlaps_consecutive_chunks():
    chunks = chunk_text("a" * 25, chunk_size=10, chunk_overlap=2)
    assert chunks[0] == "a" * 10 and len(chunks) >= 3
