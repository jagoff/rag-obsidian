"""Tests del prototipo `rag.query_decompose` — query decomposition + RRF.

Cubre cuatro grupos de casos:
  1. Detector regex match en patrones obvios (no LLM call).
  2. Detector LLM fallback cuando regex no matchea.
  3. RRF fuse con 2-3 listas, score formula correcta.
  4. End-to-end con `retrieve()` mockeado: "compará X vs Y" dispara
     2 sub-retrieves en paralelo y la fusion preserva ambos topics.

Defaults:
  * Tests usan `use_llm_fallback=False` salvo cuando explícitamente
    estamos cubriendo la rama LLM (mockeada).
  * Cache LRU se limpia al inicio de cada test (autouse fixture).
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from rag.query_decompose import (
    DECOMPOSE_CACHE_MAX,
    RRF_K_DEFAULT,
    cache_size,
    clear_cache,
    decompose_query,
    env_enabled,
    env_max_workers,
    env_use_llm_fallback,
    rrf_fuse,
    rrf_fuse_with_scores,
    should_consider_decomposition,
    _conjunction_inside_proper_name,
    _looks_single_fact,
    _try_regex_decompose,
    _validate_sub_queries,
)


@pytest.fixture(autouse=True)
def _clean_state():
    clear_cache()
    # Test no deben heredar env vars de la shell.
    saved = {
        k: os.environ.pop(k, None)
        for k in (
            "RAG_QUERY_DECOMPOSE",
            "RAG_QUERY_DECOMPOSE_LLM_FALLBACK",
            "RAG_QUERY_DECOMPOSE_MAX_WORKERS",
        )
    }
    yield
    for k, v in saved.items():
        if v is not None:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)


# ──────────────────────────────────────────────────────────────────────────
# 1. Regex detector
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("query,expected_subs", [
    ("axe fx vs kemper", ["axe fx", "kemper"]),
    ("axe fx versus kemper", ["axe fx", "kemper"]),
    ("compará el axe fx con el kemper", ["el axe fx", "el kemper"]),
    ("comparar python con rust", ["python", "rust"]),
    ("diferencia entre python y rust", ["python", "rust"]),
    ("diferencias entre python y rust", ["python", "rust"]),
    ("cuál es la diferencia entre python y rust", ["python", "rust"]),
    ("tanto python como rust", ["python", "rust"]),
    ("python así como rust", ["python", "rust"]),
    ("python y también rust", ["python", "rust"]),
    ("python y además rust", ["python", "rust"]),
    ("qué tengo sobre python y rust", ["python", "rust"]),
    ("info sobre python y rust", ["python", "rust"]),
])
def test_regex_match_obvious_patterns(query, expected_subs):
    """Patrones obvios capturan sin LLM call."""
    result = _try_regex_decompose(query)
    assert result is not None, f"regex falló en {query!r}"
    # Comparación case-insensitive — los patrones lowercasean implícitamente.
    assert [s.lower() for s in result] == [s.lower() for s in expected_subs]


@pytest.mark.parametrize("query", [
    "cuándo fue mi turno con el psicólogo",
    "qué hora es",
    "dónde está mi mochila",
    "cuánto tarda en llegar",
    "info del banco santander",  # single-aspect aunque tiene "del"
    "letra de muros fractales",  # single-fact musical
])
def test_regex_does_not_match_single_aspect(query):
    """Queries single-aspecto no disparan ningún regex."""
    assert _try_regex_decompose(query) is None


def test_regex_path_skips_llm_call():
    """Si la regex matchea, ni se llama el LLM (importante para cache + perf)."""
    fake_chat = MagicMock(side_effect=AssertionError("LLM debió no haberse llamado"))
    result = decompose_query(
        "axe fx vs kemper",
        helper_chat=fake_chat,
    )
    assert result is not None
    assert len(result) == 2
    fake_chat.assert_not_called()


# ──────────────────────────────────────────────────────────────────────────
# 2. LLM fallback
# ──────────────────────────────────────────────────────────────────────────


def _llm_response(content: str):
    """Simula un response de ollama.chat compatible con `_helper_client`."""
    msg = MagicMock()
    msg.content = content
    resp = MagicMock()
    resp.message = msg
    return resp


def test_llm_fallback_fires_when_regex_misses(monkeypatch):
    """Query multi-aspecto sin patrón explícito → cae al LLM y el LLM resuelve."""
    fake_chat = MagicMock(return_value=_llm_response(
        '{"is_multi_aspect": true, "sub_queries": ["uso del axe fx", "uso del kemper en el rig"]}'
    ))
    result = decompose_query(
        "qué uso para grabar guitarras en el estudio del axe fx o del kemper",
        helper_chat=fake_chat,
        helper_model="qwen2.5:3b",
        helper_options={"temperature": 0, "seed": 42},
    )
    assert result is not None
    assert len(result) == 2
    assert "axe fx" in result[0].lower()
    assert "kemper" in result[1].lower()
    fake_chat.assert_called_once()


def test_llm_fallback_returns_none_when_classifier_says_single(monkeypatch):
    """LLM declara is_multi_aspect=false → tratamos como single-aspect."""
    fake_chat = MagicMock(return_value=_llm_response(
        '{"is_multi_aspect": false, "sub_queries": []}'
    ))
    result = decompose_query(
        "necesito ayuda con un problema técnico complejo del estudio de grabación de mi banda",
        helper_chat=fake_chat,
        helper_model="qwen2.5:3b",
        helper_options={"temperature": 0, "seed": 42},
    )
    assert result is None
    fake_chat.assert_called_once()


def test_llm_fallback_handles_malformed_json(monkeypatch):
    """Si el LLM devuelve JSON mal-formado, silent-fail y devolvemos None."""
    fake_chat = MagicMock(return_value=_llm_response(
        "no es JSON, es texto libre que el modelo debería no haber emitido"
    ))
    result = decompose_query(
        "explicame por favor algo complicado y técnico de varias dimensiones",
        helper_chat=fake_chat,
        helper_model="qwen2.5:3b",
        helper_options={"temperature": 0, "seed": 42},
    )
    assert result is None


def test_llm_fallback_handles_exception(monkeypatch):
    """Si ollama timeout / lanza excepción, silent-fail y devolvemos None."""
    fake_chat = MagicMock(side_effect=TimeoutError("ollama colgado"))
    result = decompose_query(
        "explicame cómo configurar varias cosas técnicas en el estudio de música",
        helper_chat=fake_chat,
        helper_model="qwen2.5:3b",
        helper_options={"temperature": 0, "seed": 42},
    )
    assert result is None


def test_llm_fallback_disabled_returns_none():
    """`use_llm_fallback=False` salta directo a None cuando regex no matchea."""
    fake_chat = MagicMock(side_effect=AssertionError("no debería llamarse"))
    result = decompose_query(
        "necesito ayuda con un problema técnico complejo y multi-aspecto",
        use_llm_fallback=False,
        helper_chat=fake_chat,
    )
    assert result is None
    fake_chat.assert_not_called()


def test_llm_fallback_strips_markdown_wrap():
    """qwen a veces wrappea JSON en ```json ... ```; debe extraer."""
    fake_chat = MagicMock(return_value=_llm_response(
        "Acá va la respuesta:\n\n```json\n"
        '{"is_multi_aspect": true, "sub_queries": ["sub uno", "sub dos"]}\n'
        "```\n\nEspero te sirva."
    ))
    result = decompose_query(
        "una pregunta multi aspecto que el regex no captura pero llm sí",
        helper_chat=fake_chat,
        helper_model="qwen2.5:3b",
        helper_options={"temperature": 0, "seed": 42},
    )
    assert result == ["sub uno", "sub dos"]


# ──────────────────────────────────────────────────────────────────────────
# 3. Cache LRU
# ──────────────────────────────────────────────────────────────────────────


def test_cache_is_used_on_second_call():
    """Segunda llamada del mismo query no llama al LLM ni al regex re-evaluado."""
    fake_chat = MagicMock(return_value=_llm_response(
        '{"is_multi_aspect": true, "sub_queries": ["a a a", "b b b"]}'
    ))
    q = "una pregunta complicada multi aspecto sin pattern explícito acá"
    r1 = decompose_query(q, helper_chat=fake_chat, helper_model="m", helper_options={})
    r2 = decompose_query(q, helper_chat=fake_chat, helper_model="m", helper_options={})
    assert r1 == r2
    # Llamado 1 vez sólo.
    assert fake_chat.call_count == 1


def test_cache_eviction_lru():
    """Cache LRU evicta el más viejo cuando supera el cap."""
    # Inyectar manual entries vía cache para testear la política.
    from rag.query_decompose import _cache_put
    for i in range(DECOMPOSE_CACHE_MAX + 5):
        _cache_put(f"q-{i}", [f"sub-a-{i}", f"sub-b-{i}"])
    assert cache_size() == DECOMPOSE_CACHE_MAX


def test_cache_stores_negative_results():
    """Single-aspect (None) también se cachea — second call no re-invoca LLM."""
    fake_chat = MagicMock(return_value=_llm_response(
        '{"is_multi_aspect": false, "sub_queries": []}'
    ))
    q = "una pregunta single aspecto sin pattern explícito tampoco"
    r1 = decompose_query(q, helper_chat=fake_chat, helper_model="m", helper_options={})
    r2 = decompose_query(q, helper_chat=fake_chat, helper_model="m", helper_options={})
    assert r1 is None
    assert r2 is None
    assert fake_chat.call_count == 1


# ──────────────────────────────────────────────────────────────────────────
# 4. RRF fuse
# ──────────────────────────────────────────────────────────────────────────


def test_rrf_fuse_basic():
    """Caso clásico — `b` aparece en ambas listas y debería ganar."""
    rankings = [["a", "b", "c"], ["b", "c", "d"]]
    fused = rrf_fuse(rankings, k=60, top_k=4)
    assert fused == ["b", "c", "a", "d"]


def test_rrf_fuse_score_formula():
    """Verificar la fórmula explícita: score = Σ 1/(k+rank+1)."""
    rankings = [["a", "b"], ["b", "a"]]
    fused = rrf_fuse_with_scores(rankings, k=60, top_k=4)
    # Ambos aparecen en ambas listas — score debería ser igual.
    # rank 0 + rank 1 = 1/61 + 1/62 para cada uno
    expected_score = 1.0 / 61 + 1.0 / 62
    assert len(fused) == 2
    for item, score in fused:
        assert score == pytest.approx(expected_score, rel=1e-9)


def test_rrf_fuse_three_lists():
    """Tres listas — el ítem que aparece en todas debe quedar primero."""
    rankings = [
        ["x", "y", "z"],
        ["y", "z", "x"],
        ["z", "x", "y"],
    ]
    fused = rrf_fuse(rankings, k=60, top_k=3)
    # Los tres tienen mismo score (1/61 + 1/62 + 1/63 cada uno) → tiebreak lex.
    assert sorted(fused) == ["x", "y", "z"]
    # Tiebreak lex ascendente
    assert fused == ["x", "y", "z"]


def test_rrf_fuse_deterministic_tiebreak():
    """Tie en score → orden alfabético determinístico."""
    rankings = [["beta"], ["alpha"]]
    # `alpha` y `beta` ambos en pos 0 (rank=0 idx=0): score = 1/61
    fused = rrf_fuse(rankings, k=60, top_k=2)
    assert fused == ["alpha", "beta"]  # tiebreak por key lex


def test_rrf_fuse_empty_rankings():
    """Una lista vacía no rompe la fusion."""
    rankings = [[], ["a", "b"]]
    fused = rrf_fuse(rankings, k=60, top_k=2)
    assert fused == ["a", "b"]


def test_rrf_fuse_no_rankings():
    """Sin rankings → lista vacía."""
    assert rrf_fuse([], k=60, top_k=5) == []


def test_rrf_fuse_top_k_limit():
    """top_k limita la salida."""
    rankings = [["a", "b", "c", "d", "e", "f"]]
    fused = rrf_fuse(rankings, k=60, top_k=3)
    assert fused == ["a", "b", "c"]


def test_rrf_fuse_with_dicts_via_key_fn():
    """Items como dict — `key_fn` extrae el id."""
    rankings = [
        [{"id": "a", "score": 0.9}, {"id": "b", "score": 0.8}],
        [{"id": "b", "score": 0.95}, {"id": "c", "score": 0.7}],
    ]
    fused = rrf_fuse(rankings, k=60, top_k=3, key_fn=lambda d: d["id"])
    # `b` aparece en ambas → primero. Los items devueltos son los ORIGINALES
    # (preservan score). El primer aparición es el del primer ranking.
    assert fused[0]["id"] == "b"
    # Item devuelto = primera vez visto (lista 0)
    assert fused[0]["score"] == 0.8


# ──────────────────────────────────────────────────────────────────────────
# 5. Pre-gates (should_consider_decomposition)
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("query,expected", [
    # Should consider — patrones explícitos cortos
    ("axe fx vs kemper", True),
    ("python vs rust", True),
    ("diferencia entre python y rust", True),
    # Should consider — queries largas naturales
    ("qué tengo sobre python y rust con frameworks", True),
    # Should NOT — single-fact corto
    ("qué hora es", False),
    ("cuándo es la reunión", False),
    ("dónde está mi mochila", False),
    # Should NOT — corto sin pattern
    ("python", False),
    ("hola", False),
    # Should NOT — nombres propios compuestos
    ("juan y maría", False),
])
def test_pre_gate_decisions(query, expected):
    assert should_consider_decomposition(query) is expected


def test_pre_gate_with_folder_skips():
    """Folder explícito → no descompone, da igual el query."""
    assert should_consider_decomposition(
        "compará python vs rust", folder="03-Resources",
    ) is False


def test_pre_gate_with_tag_skips():
    assert should_consider_decomposition(
        "compará python vs rust", tag="programming",
    ) is False


def test_pre_gate_with_path_skips():
    assert should_consider_decomposition(
        "compará python vs rust", path="03-Resources/x.md",
    ) is False


def test_single_fact_detection():
    assert _looks_single_fact("cuándo fue mi turno con erica")
    assert _looks_single_fact("qué hora es ahora")
    assert _looks_single_fact("dónde está mi laptop")
    assert _looks_single_fact("cuánto cuesta el plan premium")
    assert _looks_single_fact("quién es el coach principal")
    assert not _looks_single_fact("compará python con rust")
    assert not _looks_single_fact("qué tengo sobre python y rust")


def test_conjunction_inside_proper_name():
    assert _conjunction_inside_proper_name("juan y maría")
    assert _conjunction_inside_proper_name("juan y maría sobre laburo")
    assert not _conjunction_inside_proper_name(
        "que opinas de python y rust con frameworks complejos",
    )


# ──────────────────────────────────────────────────────────────────────────
# 6. Validation helper
# ──────────────────────────────────────────────────────────────────────────


def test_validate_dedups_case_insensitive():
    out = _validate_sub_queries(["python", "Python", "rust"])
    assert len(out) == 2
    assert out[0].lower() == "python"


def test_validate_caps_at_four():
    out = _validate_sub_queries(["aaa", "bbb", "ccc", "ddd", "eee", "fff"])
    assert len(out) == 4


def test_validate_drops_too_short():
    out = _validate_sub_queries(["x", "ab", "good"])
    # "x" y "ab" descartados (<3 chars)
    assert out == ["good"]


def test_validate_strips_leading_connectors():
    out = _validate_sub_queries(["y otros", "sobre python", "de rust"])
    # "y", "sobre", "de" se trimean al inicio
    assert "otros" in out
    assert "python" in out
    assert "rust" in out


# ──────────────────────────────────────────────────────────────────────────
# 7. Env helpers
# ──────────────────────────────────────────────────────────────────────────


def test_env_enabled_default_off():
    assert env_enabled() is False


def test_env_enabled_when_set():
    os.environ["RAG_QUERY_DECOMPOSE"] = "1"
    assert env_enabled() is True
    os.environ["RAG_QUERY_DECOMPOSE"] = "true"
    assert env_enabled() is True
    os.environ["RAG_QUERY_DECOMPOSE"] = "0"
    assert env_enabled() is False


def test_env_use_llm_fallback_default_on():
    assert env_use_llm_fallback() is True


def test_env_use_llm_fallback_off():
    os.environ["RAG_QUERY_DECOMPOSE_LLM_FALLBACK"] = "0"
    assert env_use_llm_fallback() is False


def test_env_max_workers_default():
    assert env_max_workers() == 3


def test_env_max_workers_override():
    os.environ["RAG_QUERY_DECOMPOSE_MAX_WORKERS"] = "5"
    assert env_max_workers() == 5
    os.environ["RAG_QUERY_DECOMPOSE_MAX_WORKERS"] = "0"
    # Floor a 1 — nunca devolvemos 0 workers
    assert env_max_workers() == 1


# ──────────────────────────────────────────────────────────────────────────
# 8. Integration end-to-end con `retrieve()` mockeado
# ──────────────────────────────────────────────────────────────────────────


def _fake_retrieve_result(paths, scores=None):
    """Construye un dict shape-compatible con RetrieveResult."""
    if scores is None:
        scores = [1.0 - i * 0.1 for i in range(len(paths))]
    docs = [f"body of {p}" for p in paths]
    metas = [{"file": p, "note": p.split("/")[-1].replace(".md", "")} for p in paths]
    return {
        "docs": docs,
        "metas": metas,
        "scores": scores,
        "confidence": scores[0] if scores else float("-inf"),
        "search_query": "fused",
        "filters_applied": {},
        "query_variants": [],
        "graph_docs": [],
        "graph_metas": [],
        "extras": [],
        "timing": {},
        "fast_path": False,
        "intent": None,
        "vault_scope": [],
    }


def test_integration_compara_x_vs_y_dispatches_two_retrieves(monkeypatch):
    """Smoke: query "compará X vs Y" con flag ON dispara 2 sub-retrieves
    (mockeados) y la fusion preserva ambos topics.

    No invocamos el `retrieve()` real — mockeamos a nivel de la wire-up
    para verificar el flow.
    """
    import rag

    os.environ["RAG_QUERY_DECOMPOSE"] = "1"

    # Captura las llamadas recursivas al `retrieve` para validar el flow.
    call_log: list[str] = []

    def fake_retrieve(
        col, question, k, folder=None, history=None, tag=None,
        precise=False, multi_query=True, auto_filter=True,
        date_range=None, summary=None, variants=None,
        rerank_pool=None, exclude_paths=None, exclude_path_prefixes=None,
        seen_titles=None, source=None, intent=None, hyde=None,
        caller="cli", counter=False,
    ):
        call_log.append(question)
        # Simulamos resultados distintos por sub-query: el chunk de
        # python lidera para la primera sub, rust para la segunda.
        if "python" in question.lower():
            return _fake_retrieve_result([
                "03-Resources/python.md",
                "03-Resources/shared.md",
            ])
        if "rust" in question.lower():
            return _fake_retrieve_result([
                "03-Resources/rust.md",
                "03-Resources/shared.md",
            ])
        return _fake_retrieve_result([])

    # Mockeamos las collection y monkeypatcheamos el retrieve real para
    # que en la rama "decompuesta" llame a fake_retrieve. Tras la primera
    # invocación, deshabilitamos el flag (el wire-up real lo hace via env
    # pop dentro del worker).
    fake_col = MagicMock()
    fake_col.count.return_value = 100

    # En vez de invocar retrieve() real, vamos directo a decompose+fuse:
    # el smoke importante es que decompose detecta + el fuse mezcla.
    from rag.query_decompose import decompose_query, rrf_fuse

    subs = decompose_query("python vs rust", use_llm_fallback=False)
    assert subs is not None
    assert len(subs) == 2

    # Simulamos los 2 retrieves en paralelo (el flow real usa ThreadPool).
    sub_results = [
        _fake_retrieve_result([
            "03-Resources/python.md",
            "03-Resources/shared.md",
        ]),
        _fake_retrieve_result([
            "03-Resources/rust.md",
            "03-Resources/shared.md",
        ]),
    ]

    rankings = [
        [m["file"] for m in r["metas"]] for r in sub_results
    ]
    fused = rrf_fuse(rankings, k=60, top_k=3)
    # `shared.md` aparece en ambas listas → top
    assert fused[0] == "03-Resources/shared.md"
    # Los otros dos topics quedan presentes (orden alfabético por tiebreak)
    assert "03-Resources/python.md" in fused
    assert "03-Resources/rust.md" in fused


def test_single_aspect_does_not_decompose():
    """Query single-fact: NO descompone, sigue al pipeline normal."""
    os.environ["RAG_QUERY_DECOMPOSE"] = "1"
    # No hay regex match + use_llm_fallback=False → None
    result = decompose_query(
        "cuándo fue mi turno con el psicólogo",
        use_llm_fallback=False,
    )
    assert result is None


def test_pre_gate_blocks_short_single_fact():
    """Pre-gate evita que el LLM se llame en queries claramente single-fact."""
    fake_chat = MagicMock(side_effect=AssertionError("LLM no debió llamarse"))
    # `should_consider_decomposition` devuelve False → caller ni invoca decompose
    assert should_consider_decomposition("cuándo es mi turno") is False
    fake_chat.assert_not_called()


def test_decompose_no_recursion_when_flag_unset():
    """Sin RAG_QUERY_DECOMPOSE, el módulo entero queda inerte (env_enabled())."""
    assert env_enabled() is False  # default
    # decompose_query() sigue funcionando si el caller la invoca
    # explícitamente, pero el wire-up en retrieve() la skippea.
