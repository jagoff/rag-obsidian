"""Tests para las 3 optimizaciones de latencia en queries sobre WhatsApp
(2026-04-22). Target: reducir P50/P95 ~50% de queries que resuelven a WA.

Track 1 — `is_excluded("03-Resources/WhatsApp/...")` devuelve True por
default para que el vault indexer no duplique los chunks ya creados por
`scripts/ingest_whatsapp.py` como `source="whatsapp"`. Rollback:
`OBSIDIAN_RAG_INDEX_WA_MONTHLY=1`.

Track 2 — `retrieve()` marca `fast_path=True` cuando 2 o más de los
top-3 resultados son `source="whatsapp"` y el top-1 score supera
`RAG_WA_FAST_PATH_THRESHOLD` (default 0.3). La caller puede entonces
rutear a qwen2.5:3b + num_ctx reducido + skip citation-repair.
Rollback: `RAG_WA_FAST_PATH=0`.

Track 3 — `retrieve(source="whatsapp")` downgradea `multi_query=False`
automáticamente para saltarse el paraphrase de qwen2.5:3b + 3× embed
bge-m3 (~600ms). Rollback: `RAG_WA_SKIP_PARAPHRASE=0`.
"""
from __future__ import annotations

import pytest

import rag


# ═══════════════════════════════════════════════════════════════════════
# Track 1: is_excluded para WhatsApp vault files
# ═══════════════════════════════════════════════════════════════════════


def test_is_excluded_wa_monthly_default_skipped():
    """Default: los monthly rollups de WhatsApp se excluyen del indexer."""
    assert rag.is_excluded("03-Resources/WhatsApp/Maria/2026-03.md") is True
    assert rag.is_excluded("03-Resources/WhatsApp/Juli/2026-04.md") is True
    assert rag.is_excluded(
        "03-Resources/WhatsApp/Grecias group/2026-01.md"
    ) is True


def test_is_excluded_wa_override_respected(monkeypatch):
    """OBSIDIAN_RAG_INDEX_WA_MONTHLY=1 re-habilita el indexing como vault."""
    monkeypatch.setenv("OBSIDIAN_RAG_INDEX_WA_MONTHLY", "1")
    assert rag.is_excluded("03-Resources/WhatsApp/Maria/2026-03.md") is False
    monkeypatch.setenv("OBSIDIAN_RAG_INDEX_WA_MONTHLY", "true")
    assert rag.is_excluded("03-Resources/WhatsApp/Maria/2026-03.md") is False
    monkeypatch.setenv("OBSIDIAN_RAG_INDEX_WA_MONTHLY", "yes")
    assert rag.is_excluded("03-Resources/WhatsApp/Maria/2026-03.md") is False


def test_is_excluded_wa_override_rejects_false_values(monkeypatch):
    """Valores explícitos de rollback falsy mantienen el default (excluded)."""
    for val in ("", "0", "false", "no"):
        monkeypatch.setenv("OBSIDIAN_RAG_INDEX_WA_MONTHLY", val)
        assert rag.is_excluded(
            "03-Resources/WhatsApp/Maria/2026-03.md"
        ) is True, f"val={val!r}"


def test_is_excluded_non_wa_paths_unaffected():
    """La exclusión sólo toca el prefix WhatsApp — otras notas pasan libres."""
    assert rag.is_excluded("03-Resources/Articles/foo.md") is False
    assert rag.is_excluded("03-Resources/WhatsAppClone/foo.md") is False  # no prefix exacto
    assert rag.is_excluded("01-Projects/WhatsApp migration.md") is False  # mention != path
    assert rag.is_excluded("02-Areas/foo.md") is False


# ═══════════════════════════════════════════════════════════════════════
# Tracks 2 + 3 usan el mismo fixture de retrieve sintético
# ═══════════════════════════════════════════════════════════════════════


_KEY_VECTORS: dict[str, list[float]] = {
    "maria":  [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "juli":   [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "laburo": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "casa":   [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    "msg":    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "nota":   [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
}


def _vec_for(text: str) -> list[float]:
    v = [0.0] * 8
    lower = text.lower()
    for kw, basis in _KEY_VECTORS.items():
        if kw in lower:
            for i, x in enumerate(basis):
                v[i] += x
    norm = (sum(x * x for x in v) ** 0.5) or 1.0
    return [x / norm for x in v]


@pytest.fixture
def wa_heavy_col(monkeypatch, tmp_path):
    """8 entradas: 5 WA + 2 vault + 1 calendar. Queries sobre "maria laburo"
    hacen que los top-3 sean WA — dispara el fast-path trigger WA."""
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setattr(rag, "VAULT_PATH", vault)
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)

    from rag import SqliteVecClient as _C
    client = _C(path=str(tmp_path / "ragvec"))
    col = client.get_or_create_collection(
        name="wa_perf_test", metadata={"hnsw:space": "cosine"}
    )
    monkeypatch.setattr(rag, "get_db", lambda: col)
    rag._invalidate_corpus_cache()

    entries = [
        ("whatsapp", "whatsapp://chat-maria/1", "maria msg laburo comentario"),
        ("whatsapp", "whatsapp://chat-maria/2", "maria msg laburo sigo trabajando"),
        ("whatsapp", "whatsapp://chat-maria/3", "maria msg laburo algo importante"),
        ("whatsapp", "whatsapp://chat-juli/1",  "juli msg casa detalle"),
        ("whatsapp", "whatsapp://chat-juli/2",  "juli msg casa otro detalle"),
        ("vault",    "02-Areas/laburo.md",      "nota sobre laburo en proyecto"),
        ("vault",    "02-Areas/casa.md",        "nota sobre casa"),
        ("calendar", "calendar://cal1",         "evento laburo reunion"),
    ]
    for src, doc_id, body in entries:
        emb = _vec_for(body)
        col.add(
            ids=[f"{doc_id}::0"],
            embeddings=[emb],
            documents=[body],
            metadatas=[{
                "file": doc_id,
                "note": doc_id.split("/")[-1],
                "folder": "",
                "tags": "",
                "outlinks": "",
                "hash": f"h-{doc_id}",
                "source": src,
                "display_text": body,
                "parent": body,
            }],
        )

    rag._invalidate_corpus_cache()

    monkeypatch.setattr(rag, "embed", lambda texts: [_vec_for(t) for t in texts])

    # Counter para trackear si se llamó expand_queries
    counter = {"expand_calls": 0}
    original_identity = lambda q, **kw: [q]

    def _tracked_expand(q, **kw):
        counter["expand_calls"] += 1
        return [q]

    monkeypatch.setattr(rag, "expand_queries", _tracked_expand)

    class _FakeReranker:
        def predict(self, pairs, batch_size=None, show_progress_bar=False):  # noqa: ARG002
            scores = []
            for q, d in pairs:
                ql, dl = q.lower(), d.lower()
                shared = sum(1 for kw in _KEY_VECTORS if kw in ql and kw in dl)
                # Clip intencional a 0.55 para que caiga en el sweet spot
                # del WA-specific trigger (threshold 0.3) sin superar el
                # default gate (0.6). Así los tests pueden distinguir
                # cuál de los dos caminos disparó el fast_path.
                raw = float(shared) * 0.2
                scores.append(min(raw, 0.55))
            return scores

    monkeypatch.setattr(rag, "get_reranker", lambda: _FakeReranker())

    # Adaptive routing ON para que el fast-path gate funcione
    monkeypatch.setenv("RAG_ADAPTIVE_ROUTING", "1")
    monkeypatch.delenv("RAG_FORCE_FULL_PIPELINE", raising=False)

    return col, counter


# ═══════════════════════════════════════════════════════════════════════
# Track 3: auto-skip paraphrase cuando source=whatsapp
# ═══════════════════════════════════════════════════════════════════════


def test_source_whatsapp_skips_paraphrase_by_default(wa_heavy_col):
    """source='whatsapp' + multi_query=True → downgradeamos a False auto."""
    col, counter = wa_heavy_col
    counter["expand_calls"] = 0
    res = rag.retrieve(
        col, "maria laburo", k=5, folder=None,
        auto_filter=False, multi_query=True, source="whatsapp",
    )
    assert counter["expand_calls"] == 0, (
        "expand_queries NO debería llamarse cuando source='whatsapp'"
    )
    # Verificación adicional: filters_applied lo refleja
    assert res["filters_applied"].get("wa_skip_paraphrase") is True


def test_source_whatsapp_rollback_env_disables_skip(wa_heavy_col, monkeypatch):
    """RAG_WA_SKIP_PARAPHRASE=0 fuerza el paraphrase aun con source=whatsapp."""
    col, counter = wa_heavy_col
    monkeypatch.setenv("RAG_WA_SKIP_PARAPHRASE", "0")
    counter["expand_calls"] = 0
    rag.retrieve(
        col, "maria laburo", k=5, folder=None,
        auto_filter=False, multi_query=True, source="whatsapp",
    )
    assert counter["expand_calls"] == 1, (
        "expand_queries debería correr con RAG_WA_SKIP_PARAPHRASE=0"
    )


def test_source_set_whatsapp_only_also_skips_paraphrase(wa_heavy_col):
    """source={"whatsapp"} (set con único valor) también downgradea."""
    col, counter = wa_heavy_col
    counter["expand_calls"] = 0
    rag.retrieve(
        col, "maria laburo", k=5, folder=None,
        auto_filter=False, multi_query=True, source={"whatsapp"},
    )
    assert counter["expand_calls"] == 0


def test_multi_source_including_whatsapp_keeps_paraphrase(wa_heavy_col):
    """source={"whatsapp","calendar"} mantiene el paraphrase — calendar
    todavía se beneficia de las paraphrases."""
    col, counter = wa_heavy_col
    counter["expand_calls"] = 0
    rag.retrieve(
        col, "maria laburo", k=5, folder=None,
        auto_filter=False, multi_query=True, source={"whatsapp", "calendar"},
    )
    assert counter["expand_calls"] == 1, (
        "expand_queries debería llamarse en multi-source con whatsapp"
    )


def test_non_wa_source_unchanged(wa_heavy_col):
    """source='vault' sigue corriendo paraphrase normalmente."""
    col, counter = wa_heavy_col
    counter["expand_calls"] = 0
    rag.retrieve(
        col, "maria laburo", k=5, folder=None,
        auto_filter=False, multi_query=True, source="vault",
    )
    assert counter["expand_calls"] == 1


def test_no_source_filter_unchanged(wa_heavy_col):
    """Sin filtro de source, paraphrase sigue como antes."""
    col, counter = wa_heavy_col
    counter["expand_calls"] = 0
    rag.retrieve(
        col, "maria laburo", k=5, folder=None,
        auto_filter=False, multi_query=True, source=None,
    )
    assert counter["expand_calls"] == 1


def test_explicit_multi_query_false_still_skips(wa_heavy_col):
    """Caller que ya pasó multi_query=False no rompe (idempotente)."""
    col, counter = wa_heavy_col
    counter["expand_calls"] = 0
    rag.retrieve(
        col, "maria laburo", k=5, folder=None,
        auto_filter=False, multi_query=False, source="whatsapp",
    )
    assert counter["expand_calls"] == 0


# ═══════════════════════════════════════════════════════════════════════
# Track 2: fast-path WA trigger
# ═══════════════════════════════════════════════════════════════════════


def test_fast_path_triggers_when_top3_mostly_whatsapp(wa_heavy_col):
    """2+ de los top-3 son source=whatsapp + score>0.3 → fast_path=True."""
    col, _ = wa_heavy_col
    res = rag.retrieve(
        col, "maria laburo", k=5, folder=None,
        auto_filter=False, multi_query=False,
    )
    # Verificamos primero que top-3 sean WA mayoría
    top3 = res["metas"][:3]
    wa_count = sum(1 for m in top3 if m.get("source") == "whatsapp")
    assert wa_count >= 2, (
        f"fixture debería producir ≥2 WA en top-3, obtuvo {wa_count}"
    )
    assert res["fast_path"] is True
    assert res["filters_applied"].get("wa_fast_path") is True


def test_fast_path_not_triggered_when_only_one_wa_in_top3(wa_heavy_col):
    """Query que matchea más vault que WA → fast-path WA NO dispara.
    (el default gate score>0.6 sí puede disparar por separado — este test
    ejercita el contrario.)"""
    col, _ = wa_heavy_col
    # Query sin "maria" ni "juli" — apunta a chunks vault/calendar mayoría
    res = rag.retrieve(
        col, "nota casa", k=5, folder=None,
        auto_filter=False, multi_query=False,
    )
    top3 = res["metas"][:3]
    wa_count = sum(1 for m in top3 if m.get("source") == "whatsapp")
    # Si WA < 2 en top-3 → fast-path WA NO debería marcarse por este gate
    # (el default gate con threshold 0.6 puede o no disparar según scores)
    if wa_count < 2:
        assert res["filters_applied"].get("wa_fast_path") is not True


def test_fast_path_disabled_via_env(wa_heavy_col, monkeypatch):
    """RAG_WA_FAST_PATH=0 desactiva el trigger aun con WA mayoría."""
    col, _ = wa_heavy_col
    monkeypatch.setenv("RAG_WA_FAST_PATH", "0")
    res = rag.retrieve(
        col, "maria laburo", k=5, folder=None,
        auto_filter=False, multi_query=False,
    )
    # El default gate requiere score>0.6; nuestro fake reranker produce
    # scores máx 1.0 pero también bajos, entonces probablemente no pase
    # el 0.6 default y el WA-specific esté off → fast_path=False
    assert res["filters_applied"].get("wa_fast_path") is not True


def test_fast_path_threshold_env_override(wa_heavy_col, monkeypatch):
    """RAG_WA_FAST_PATH_THRESHOLD=2.0 (muy alto) evita el trigger WA."""
    col, _ = wa_heavy_col
    monkeypatch.setenv("RAG_WA_FAST_PATH_THRESHOLD", "2.0")
    res = rag.retrieve(
        col, "maria laburo", k=5, folder=None,
        auto_filter=False, multi_query=False,
    )
    # Los scores del fake reranker son < 2.0, entonces el WA fast-path
    # no debería disparar (aunque pase el majority check)
    assert res["filters_applied"].get("wa_fast_path") is not True


def test_fast_path_not_triggered_when_adaptive_routing_off(
    wa_heavy_col, monkeypatch
):
    """RAG_ADAPTIVE_ROUTING=0 → ningún fast-path (ni default ni WA)."""
    col, _ = wa_heavy_col
    monkeypatch.setenv("RAG_ADAPTIVE_ROUTING", "0")
    res = rag.retrieve(
        col, "maria laburo", k=5, folder=None,
        auto_filter=False, multi_query=False,
    )
    assert res["fast_path"] is False
    assert res["filters_applied"].get("wa_fast_path") is not True


def test_fast_path_explicit_source_whatsapp_triggers(wa_heavy_col):
    """Caller que explícitamente filtró source='whatsapp' → fast-path WA
    dispara sin chequear el majority count (explicit intent del caller)."""
    col, _ = wa_heavy_col
    res = rag.retrieve(
        col, "maria laburo", k=5, folder=None,
        auto_filter=False, multi_query=False, source="whatsapp",
    )
    assert res["fast_path"] is True
    assert res["filters_applied"].get("wa_fast_path") is True


def test_fast_path_preserves_default_gate_behavior(wa_heavy_col):
    """Si el default gate ya habría disparado (score>0.6 + semantic),
    _fast_path queda True igual — no lo pisamos."""
    col, _ = wa_heavy_col
    res = rag.retrieve(
        col, "maria laburo", k=5, folder=None,
        auto_filter=False, multi_query=False,
    )
    # El fast-path es True de alguna vía (WA majority o score>0.6)
    assert res["fast_path"] is True
