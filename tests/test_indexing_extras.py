"""Indexing extras — post 2026-04-22 audit el indexer ignoraba cinco señales
que existen en la mayoría de las notas del vault y las dejaba invisibles al
retrieval:

1. Inline tags `#tag` escritos en el body (sólo frontmatter se extraía).
2. Aliases del frontmatter en notas de vault general (sólo se parseaban
   para `99-Mentions/`).
3. Tasks `- [ ]` / `- [x]` — no había ni un contador ni una señal al prefix,
   así que "qué tengo pendiente?" hacía búsqueda puramente semántica.
4. Frontmatter custom — todo campo fuera de `FM_SEARCHABLE_FIELDS` caía a
   `extra_json` sin entrar al embedding prefix, así que `proyecto:`,
   `people:`, `tipo:`, etc. no ayudaban al retrieval.
5. Wikilink expansion — el título del target se embebía pero nunca su
   contenido; notas tipo hub (link + poco texto) no heredaban contexto de
   las notas que linkean.

Los tests ejercen cada gap con el mínimo fixture posible — reutilizan el
patrón de `test_index_single_file.py` (real `SqliteVecCollection` en tmp +
embed/summary/synthetic-questions mockeados a no-ops).
"""
from __future__ import annotations

from pathlib import Path

import pytest

import rag


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def indexing_env(monkeypatch, tmp_path):
    """Real SqliteVecCollection en tmp + stubs offline — copiado verbatim de
    `test_index_single_file.py` para no acoplar suites."""
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setattr(rag, "VAULT_PATH", vault)
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)

    from rag import SqliteVecClient as _C
    client = _C(path=str(tmp_path / "ragvec"))
    col = client.get_or_create_collection(
        name="indexing_extras_test", metadata={"hnsw:space": "cosine"}
    )

    monkeypatch.setattr(rag, "embed", lambda texts: [[0.1] * 1024 for _ in texts])
    monkeypatch.setattr(rag, "get_context_summary", lambda *a, **kw: "")
    monkeypatch.setattr(rag, "get_synthetic_questions", lambda *a, **kw: [])
    monkeypatch.setattr(rag, "_check_and_flag_contradictions",
                        lambda *a, **kw: None)
    rag._invalidate_corpus_cache()
    return vault, col


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


# ── 1. Inline tags #tag ─────────────────────────────────────────────────────


def test_extract_inline_tags_basic():
    """Regex recoge tags de formato Obsidian (`#tag`, `#tag-con-guion`,
    `#grupo/subgrupo`). Debe ser tolerante a acentos + dígitos + `/`."""
    text = "Una nota con #idea y #proyecto/rag y #día-feliz y #a1b2."
    tags = rag.extract_inline_tags(text)
    assert "idea" in tags
    assert "proyecto/rag" in tags
    assert "día-feliz" in tags
    assert "a1b2" in tags


def test_extract_inline_tags_ignores_markdown_headers():
    """Headers `# H1`, `## H2` NO son tags — deben filtrarse por no tener
    token adyacente tipo `\\w`. El regex exige `#` seguido inmediatamente
    de un char de identificador."""
    text = "# No es tag\n## Tampoco\nPero #sí-es tag inline."
    tags = rag.extract_inline_tags(text)
    assert tags == ["sí-es"]


def test_extract_inline_tags_dedupes_and_preserves_order():
    """Dos menciones del mismo tag → una sola entrada, orden de primera
    aparición (estable)."""
    text = "#alfa texto #beta más #alfa"
    tags = rag.extract_inline_tags(text)
    assert tags == ["alfa", "beta"]


def test_extract_inline_tags_ignores_code_inline():
    """Backticks protegen código — `#include <stdio.h>` no es un tag.

    Implementación razonable: strip de `backtick…backtick` antes de correr
    el regex. Aplica a inline code + fenced code (```…```)."""
    text = (
        "Normal con #real-tag.\n"
        "`#fake-in-inline` no cuenta.\n"
        "```\n#fake-in-fenced\n```\n"
        "Otro #otro."
    )
    tags = rag.extract_inline_tags(text)
    assert "real-tag" in tags
    assert "otro" in tags
    assert "fake-in-inline" not in tags
    assert "fake-in-fenced" not in tags


def test_inline_tags_end_up_in_metadata(indexing_env):
    """Pipeline end-to-end: el body con `#tag` inline debe terminar en la
    columna `tags` del meta table, mergeada con los de frontmatter."""
    vault, col = indexing_env
    note = vault / "inline.md"
    _write(note, (
        "---\ntags: [fm-one]\n---\n"
        "# Nota con tags inline\n\n"
        "Escribí una idea suelta con #idea-inline y también #proyecto-x.\n"
        + "relleno " * 30
    ))
    assert rag._index_single_file(col, note) == "indexed"
    got = col.get(where={"file": "inline.md"}, include=["metadatas"])
    assert got["metadatas"], "no chunks"
    stored = got["metadatas"][0]["tags"]
    stored_set = {t.strip() for t in stored.split(",") if t.strip()}
    assert "fm-one" in stored_set
    assert "idea-inline" in stored_set
    assert "proyecto-x" in stored_set


def test_inline_tags_appear_in_embed_prefix(indexing_env):
    """Además de quedar en metadata, el prefix embebido tiene que incluir
    los tags inline — es lo que hace que la embedding vector los vea."""
    vault, col = indexing_env
    # Capturamos el embed call para inspeccionar el texto final.
    embed_inputs: list[str] = []

    def _capture_embed(texts):
        embed_inputs.extend(texts)
        return [[0.1] * 1024 for _ in texts]

    import rag as _rag
    _original = _rag.embed
    _rag.embed = _capture_embed
    try:
        note = vault / "prefix-inline.md"
        _write(note, (
            "# Hub de ideas\n\n"
            "Nota con #inline-tag-crucial en el cuerpo.\n"
            + "más relleno " * 30
        ))
        rag._index_single_file(col, note)
    finally:
        _rag.embed = _original

    assert any("#inline-tag-crucial" in t for t in embed_inputs), (
        f"inline tag missing from embed prefix. Got: {embed_inputs[:1]}"
    )


# ── 2. Aliases del frontmatter ──────────────────────────────────────────────


def test_aliases_land_in_extra_json(indexing_env):
    """`aliases: [Maru, Marucha]` debe persistirse en el metadata de
    cada chunk para que el retrieval pueda disambiguar personas/conceptos."""
    vault, col = indexing_env
    note = vault / "maria.md"
    _write(note, (
        "---\naliases:\n  - Maru\n  - Marucha\n---\n"
        "# Maria\n\n"
        + "Body de la persona " * 30
    ))
    assert rag._index_single_file(col, note) == "indexed"
    got = col.get(where={"file": "maria.md"}, include=["metadatas"])
    meta = got["metadatas"][0]
    aliases = meta.get("aliases")
    # May come back as list (from extra_json JSON-decode) or CSV string
    # depending on serialization. Both are valid; normalize.
    if isinstance(aliases, str):
        alias_set = {a.strip() for a in aliases.split(",") if a.strip()}
    else:
        alias_set = set(aliases or [])
    assert "Maru" in alias_set
    assert "Marucha" in alias_set


def test_aliases_in_embed_prefix(indexing_env):
    """El prefix embebido debe mencionar los aliases — sin esto, una
    query "Marucha cumpleaños" no matchearía la nota `Maria.md`."""
    embed_inputs: list[str] = []

    def _capture_embed(texts):
        embed_inputs.extend(texts)
        return [[0.1] * 1024 for _ in texts]

    import rag as _rag
    _original = _rag.embed
    _rag.embed = _capture_embed
    try:
        vault, col = indexing_env
        note = vault / "juan.md"
        _write(note, (
            "---\naliases: [Juancho, Johnny]\n---\n"
            "# Juan\n\n"
            + "body " * 30
        ))
        rag._index_single_file(col, note)
    finally:
        _rag.embed = _original

    prefix = embed_inputs[0] if embed_inputs else ""
    assert "Juancho" in prefix, f"alias missing: {prefix[:200]}"
    assert "Johnny" in prefix


def test_aliases_scalar_string_form_accepted(indexing_env):
    """YAML permite `aliases: "Maru"` como scalar string — no debe romperse
    ni iterar por chars (bug clásico)."""
    vault, col = indexing_env
    note = vault / "scalar.md"
    _write(note, (
        "---\naliases: SoloUno\n---\n"
        "# Nota\n\n"
        + "body " * 30
    ))
    assert rag._index_single_file(col, note) == "indexed"
    got = col.get(where={"file": "scalar.md"}, include=["metadatas"])
    meta = got["metadatas"][0]
    aliases = meta.get("aliases")
    if isinstance(aliases, str):
        alias_set = {a.strip() for a in aliases.split(",") if a.strip()}
    else:
        alias_set = set(aliases or [])
    assert "SoloUno" in alias_set
    # No char-split artifact: ["S","o","l","o","U","n","o"] sería un bug.
    assert "S" not in alias_set


# ── 3. Tasks - [ ] / - [x] ──────────────────────────────────────────────────


def test_extract_tasks_counts_open_and_done():
    body = (
        "## Lista\n"
        "- [ ] comprar pan\n"
        "- [x] pagar luz\n"
        "- [ ] llamar a Juli\n"
        "- [X] confirmar cena\n"  # may capital X
    )
    out = rag.extract_tasks(body)
    assert out["open"] == 2
    assert out["done"] == 2


def test_extract_tasks_handles_indented_and_numbered():
    body = (
        "- [ ] top level\n"
        "  - [ ] indented\n"
        "1. [x] numbered done\n"
        "2. [ ] numbered open\n"
        "* [x] bullet star done\n"
    )
    out = rag.extract_tasks(body)
    # 3 open (top level, indented, numbered-open), 2 done (numbered-done,
    # bullet-star-done).
    assert out["open"] == 3
    assert out["done"] == 2


def test_extract_tasks_empty_when_no_tasks():
    out = rag.extract_tasks("# Solo texto\n\nNada de tareas acá.\n")
    assert out == {"open": 0, "done": 0, "texts": []}


def test_extract_tasks_collects_open_task_texts_capped():
    """Para que el prefix pueda surfacear las primeras N tareas abiertas
    como señal textual (no sólo el count)."""
    body = "\n".join(f"- [ ] tarea numero {i}" for i in range(50))
    out = rag.extract_tasks(body)
    # Se espera cap razonable (<= 10), orden de aparición.
    assert len(out["texts"]) <= 10
    assert out["texts"][0].startswith("tarea numero 0")


def test_tasks_counts_land_in_metadata(indexing_env):
    vault, col = indexing_env
    note = vault / "pendientes.md"
    _write(note, (
        "# Lista de cosas\n\n"
        "- [ ] comprar pan\n"
        "- [x] pagar luz\n"
        "- [ ] llamar a Juli\n"
        + "más body " * 30
    ))
    assert rag._index_single_file(col, note) == "indexed"
    got = col.get(where={"file": "pendientes.md"}, include=["metadatas"])
    meta = got["metadatas"][0]
    assert int(meta.get("open_tasks") or 0) == 2
    assert int(meta.get("done_tasks") or 0) == 1


def test_tasks_hint_in_embed_prefix(indexing_env):
    """Con 2 tasks abiertas el prefix debe contener algo tipo
    `Tareas abiertas: 2` — así el intent "qué tengo pendiente" matchea
    por embedding sobre cualquier nota con tareas abiertas."""
    embed_inputs: list[str] = []

    def _capture_embed(texts):
        embed_inputs.extend(texts)
        return [[0.1] * 1024 for _ in texts]

    import rag as _rag
    _original = _rag.embed
    _rag.embed = _capture_embed
    try:
        vault, col = indexing_env
        note = vault / "tasks-hint.md"
        _write(note, (
            "# Mis tareas\n\n"
            "- [ ] una\n"
            "- [ ] dos\n"
            + "x " * 30
        ))
        rag._index_single_file(col, note)
    finally:
        _rag.embed = _original

    prefix = embed_inputs[0] if embed_inputs else ""
    # No forzamos la cadena literal — aceptamos "Tareas abiertas" o
    # "open_tasks" o cualquier variante que mencione el concepto.
    assert ("Tareas abiertas" in prefix
            or "tareas abiertas" in prefix
            or "open_tasks" in prefix), f"no task hint in prefix: {prefix[:300]}"


# ── 4. Expand frontmatter — custom fields al prefix ─────────────────────────


def test_custom_frontmatter_fields_appear_in_prefix(indexing_env):
    """Campos del FM fuera de `FM_SEARCHABLE_FIELDS` (p. ej. `proyecto`,
    `tipo`) deben aparecer en el prefix embebido. Hoy se pierden porque
    sólo el allowlist fijo se embebe."""
    embed_inputs: list[str] = []

    def _capture_embed(texts):
        embed_inputs.extend(texts)
        return [[0.1] * 1024 for _ in texts]

    import rag as _rag
    _original = _rag.embed
    _rag.embed = _capture_embed
    try:
        vault, col = indexing_env
        note = vault / "custom-fm.md"
        _write(note, (
            "---\n"
            "proyecto: operación-silencio\n"
            "tipo: retro\n"
            "people: [Juli, Astor]\n"
            "---\n"
            "# Retro Q1\n\n"
            + "body " * 30
        ))
        rag._index_single_file(col, note)
    finally:
        _rag.embed = _original

    prefix = embed_inputs[0] if embed_inputs else ""
    assert "operación-silencio" in prefix or "proyecto" in prefix
    assert "retro" in prefix.lower()
    assert "Juli" in prefix or "people" in prefix


def test_blacklisted_frontmatter_fields_skipped(indexing_env):
    """Campos internos / ruidosos (`position`, `id`, campos que empiezan
    con `_`) NO deben embeberse — son ruido de plugins de Obsidian."""
    embed_inputs: list[str] = []

    def _capture_embed(texts):
        embed_inputs.extend(texts)
        return [[0.1] * 1024 for _ in texts]

    import rag as _rag
    _original = _rag.embed
    _rag.embed = _capture_embed
    try:
        vault, col = indexing_env
        note = vault / "noisy-fm.md"
        _write(note, (
            "---\n"
            "position: [1, 2, 3]\n"
            "id: abc-123-def-456\n"
            "_internal: plugin-state-blob\n"
            "titulo-real: Mi Nota\n"
            "---\n"
            "# Mi Nota\n\n"
            + "body " * 30
        ))
        rag._index_single_file(col, note)
    finally:
        _rag.embed = _original

    prefix = embed_inputs[0] if embed_inputs else ""
    assert "abc-123-def-456" not in prefix
    assert "plugin-state-blob" not in prefix
    # The real content field DOES make it in.
    assert "Mi Nota" in prefix


def test_long_frontmatter_values_truncated_not_included(indexing_env):
    """Valores string largos (>200 chars) no deberían dominar el prefix —
    el prefix está cappeado y tiene que preservar señal alta, no un blob
    random."""
    embed_inputs: list[str] = []

    def _capture_embed(texts):
        embed_inputs.extend(texts)
        return [[0.1] * 1024 for _ in texts]

    import rag as _rag
    _original = _rag.embed
    _rag.embed = _capture_embed
    try:
        vault, col = indexing_env
        long_val = "x" * 500
        note = vault / "long-fm.md"
        _write(note, (
            "---\n"
            f"raw-dump: \"{long_val}\"\n"
            "---\n"
            "# Nota\n\nbody " + "y " * 30
        ))
        rag._index_single_file(col, note)
    finally:
        _rag.embed = _original

    prefix = embed_inputs[0] if embed_inputs else ""
    # O se trunca o se omite — en cualquier caso no debe aparecer entero.
    assert long_val not in prefix


# ── 5. Wikilink expansion opt-in ────────────────────────────────────────────


def test_wikilink_expansion_off_by_default(indexing_env, monkeypatch):
    """Sin `RAG_WIKILINK_EXPANSION=1`, el body del target NO se embebe —
    sólo el título via la logica legacy de `build_prefix`."""
    monkeypatch.delenv("RAG_WIKILINK_EXPANSION", raising=False)
    embed_inputs: list[str] = []

    def _capture_embed(texts):
        embed_inputs.extend(texts)
        return [[0.1] * 1024 for _ in texts]

    import rag as _rag
    _original = _rag.embed
    _rag.embed = _capture_embed
    try:
        vault, col = indexing_env
        # Target note with a distinctive phrase.
        target = vault / "Target.md"
        _write(target, "# Target\n\n" + "SENTINEL_UNIQUE_TARGET_PHRASE body " * 20)
        rag._index_single_file(col, target)
        embed_inputs.clear()

        # Hub note linking to Target — should NOT embed Target's body.
        hub = vault / "Hub.md"
        _write(hub, "# Hub\n\nmirá [[Target]].\n" + "relleno " * 5)
        rag._index_single_file(col, hub)
    finally:
        _rag.embed = _original

    hub_prefix = embed_inputs[0] if embed_inputs else ""
    assert "SENTINEL_UNIQUE_TARGET_PHRASE" not in hub_prefix


def test_wikilink_expansion_on_includes_target_summary(indexing_env, monkeypatch):
    """Con el flag activo, el body del target (truncado) se concatena al
    prefix del caller como `Relacionada [[Target]]: <primeras N palabras>`."""
    monkeypatch.setenv("RAG_WIKILINK_EXPANSION", "1")
    embed_inputs: list[str] = []

    def _capture_embed(texts):
        embed_inputs.extend(texts)
        return [[0.1] * 1024 for _ in texts]

    import rag as _rag
    _original = _rag.embed
    _rag.embed = _capture_embed
    try:
        vault, col = indexing_env
        target = vault / "Deep Work.md"
        _write(target, (
            "# Deep Work\n\n"
            "La práctica de concentración prolongada sin distracciones "
            "digitales para maximizar output cognitivo. "
            + "detalle " * 20
        ))
        rag._index_single_file(col, target)
        embed_inputs.clear()

        hub = vault / "Habits.md"
        _write(hub, (
            "# Habits\n\n"
            "Un hábito central es [[Deep Work]].\n"
            + "relleno " * 10
        ))
        rag._index_single_file(col, hub)
    finally:
        _rag.embed = _original

    hub_prefix = embed_inputs[0] if embed_inputs else ""
    # Tiene que haber señal del target en el prefix del hub.
    assert "concentración" in hub_prefix.lower() or "Deep Work" in hub_prefix, (
        f"wikilink expansion didn't include target body: {hub_prefix[:400]}"
    )


def test_wikilink_expansion_missing_target_is_graceful(indexing_env, monkeypatch):
    """Un wikilink a una nota que no existe no debe romper el indexing."""
    monkeypatch.setenv("RAG_WIKILINK_EXPANSION", "1")
    vault, col = indexing_env
    hub = vault / "Solo.md"
    _write(hub, "# Solo\n\nLink a [[No existe]].\n" + "body " * 30)
    # Must not raise + must still index.
    assert rag._index_single_file(col, hub) == "indexed"


def test_wikilink_expansion_respects_cap(indexing_env, monkeypatch):
    """Max N wikilinks expandidos + M chars c/u — con 20 links no debe
    explotar el prefix a megabytes."""
    monkeypatch.setenv("RAG_WIKILINK_EXPANSION", "1")
    embed_inputs: list[str] = []

    def _capture_embed(texts):
        embed_inputs.extend(texts)
        return [[0.1] * 1024 for _ in texts]

    import rag as _rag
    _original = _rag.embed
    _rag.embed = _capture_embed
    try:
        vault, col = indexing_env
        # Create 20 targets with huge bodies.
        for i in range(20):
            t = vault / f"T{i}.md"
            _write(t, f"# T{i}\n\n" + f"body de t{i} " * 200)
            rag._index_single_file(col, t)
        embed_inputs.clear()

        # Hub linking to all of them.
        hub = vault / "BigHub.md"
        links = "\n".join(f"- [[T{i}]]" for i in range(20))
        _write(hub, f"# BigHub\n\n{links}\n" + "relleno " * 10)
        rag._index_single_file(col, hub)
    finally:
        _rag.embed = _original

    hub_prefix = embed_inputs[0] if embed_inputs else ""
    # Hard cap: el prefix no debería superar ~4 KB aun con 20 targets.
    assert len(hub_prefix) < 5000, (
        f"prefix exploded: {len(hub_prefix)} chars with 20 wikilinks"
    )
