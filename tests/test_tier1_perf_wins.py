"""
Contract tests for el saneamiento Tier 1 (2026-04-24): tres fixes de
performance baratos y medibles.

1. **ocrmac lazy-import** — `ocrmac` arrastraba pyobjc (Vision + Quartz
   + CoreML + AppKit = ~130ms en macOS) al nivel de `import rag`. Como
   el CLI se invoca constantemente para subcomandos que no hacen OCR
   (query, chat, session list, --help), pagábamos ese cold-start cada
   vez. Ahora la import se dispara en `_load_ocrmac_module()` solo
   cuando `_ocr_image()` corre.
   Impacto medido: cold `rag --help` 340ms → 200-210ms (~38% menos).

2. **reranker `batch_size=1`** — artefacto stale del commit 79f6b8e
   que aplicaba a pool=2-4 donde el padding a pow-2 dominaba; a
   pool=30 forzaba 30 inferencias secuenciales MPS (~500ms-1s) en vez
   de una pasada batched (~100-200ms). Comentario explicativo ya
   existía en rag.py:17869-17874 para la rerank principal; este call
   en la multi-retrieve graph expansion path (rag.py:18700 aprox)
   había quedado olvidado con el batch_size=1 explícito.

3. **Eager warmup en CLI para subcomandos que retrievan** — antes,
   solo `serve` disparaba `warmup_async()` en su lifespan. Las CLI
   one-shots (`rag query`, `rag chat`, `rag morning`, ...) pagaban el
   cold-load 5-12s en foreground de la primera llamada a retrieve().
   Ahora el `cli()` callback dispara warmup eager si
   `ctx.invoked_subcommand in _CLI_WARMUP_SUBCOMMANDS`.

Estos tests son source-level + side-effect level — no triggereamos un
cold-load real en CI (requiere ~500MB HF cache + MPS). Validan el
contrato que los fixes promueven.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

import rag

ROOT = Path(__file__).resolve().parent.parent
RAG_PY = (ROOT / "rag.py").read_text(encoding="utf-8")


# ─── Fix 1: ocrmac lazy-import ────────────────────────────────────────────

def test_import_rag_does_not_load_ocrmac_or_vision():
    """`import rag` NO debe cargar ocrmac/Vision/Quartz/CoreML/AppKit en
    sys.modules. Ese era el cold-start tax de ~130ms que el CLI pagaba
    cada vez que se invocaba — incluso para subcomandos sin OCR."""
    # rag ya está importado por el test runner; verificamos que los
    # módulos pesados no hayan aterrizado como side-effect.
    #
    # NOTA: si otro test previo dispara _load_ocrmac_module() o
    # _ocr_image(), ocrmac SÍ va a estar en sys.modules. Este test
    # asume orden de ejecución donde ningún test previo tocó OCR —
    # en la práctica, los ocrmac tests aislan via monkeypatch y no
    # dejan el módulo real cargado. Si este test rompe por contamination
    # de otro test, el fix es `pytest -x tests/test_tier1_perf_wins.py`
    # aislado.
    heavy_modules = [
        "ocrmac.ocrmac",  # el import lazy
        "Vision",  # pyobjc framework, ~116ms
        "CoreML",  # pyobjc framework, ~4ms
    ]
    for mod in heavy_modules:
        assert mod not in sys.modules or _ocrmac_test_previously_fired(), (
            f"{mod} cargado al import de rag — el lazy-import no está "
            f"funcionando. sys.modules contiene: "
            f"{[m for m in heavy_modules if m in sys.modules]}"
        )


def _ocrmac_test_previously_fired() -> bool:
    """Detecta si algún test ya disparó la carga de ocrmac — usado como
    escape hatch para evitar false positives en test ordering."""
    # Si _ocrmac_import_attempted es True, alguien ya pagó el import.
    # En ese caso no podemos assert que sys.modules esté limpio.
    return getattr(rag, "_ocrmac_import_attempted", False)


def test_load_ocrmac_module_is_idempotent():
    """Llamarlo múltiples veces debe retornar el mismo singleton sin
    re-importar. Usamos _ocrmac_import_attempted como guarda."""
    # Save state
    saved_module = rag._ocrmac_module
    saved_attempted = rag._ocrmac_import_attempted
    try:
        # Reset a estado fresco.
        rag._ocrmac_module = None
        rag._ocrmac_import_attempted = False

        first = rag._load_ocrmac_module()
        second = rag._load_ocrmac_module()
        third = rag._load_ocrmac_module()

        # Si el sistema es macOS con ocrmac instalado, first/second/third
        # son el mismo módulo. Si no (CI Linux, ej.), los tres son None.
        assert first is second is third, (
            "_load_ocrmac_module() no devuelve singleton estable"
        )
        assert rag._ocrmac_import_attempted is True, (
            "_ocrmac_import_attempted debería haberse seteado en True"
        )
    finally:
        rag._ocrmac_module = saved_module
        rag._ocrmac_import_attempted = saved_attempted


def test_load_ocrmac_module_respects_monkeypatched_attribute(monkeypatch):
    """Si un test o setup externo setea `rag._ocrmac_module` a un fake,
    `_load_ocrmac_module()` debe devolver ese fake, no re-importar. Esto
    preserva el contract que los existing OCR tests usan via
    `monkeypatch.setattr(rag, '_ocrmac_module', fake_module)`."""
    class _FakeOCRMod:
        marker = "fake-for-test"

    fake = _FakeOCRMod()
    monkeypatch.setattr(rag, "_ocrmac_module", fake)

    # Aún si el flag "attempted" está en True, si hay module seteado,
    # se devuelve.
    monkeypatch.setattr(rag, "_ocrmac_import_attempted", True)

    assert rag._load_ocrmac_module() is fake


def test_ocr_image_uses_lazy_loader_not_module_global():
    """`_ocr_image()` en rag.py debe llamar a `_load_ocrmac_module()`
    (no accesar `_ocrmac_module` directo al principio). Si rompemos
    esto, los tests que monkeypatchen `rag._ocrmac_module=None` post-
    import podrían seguir viendo un valor cached en una local
    closure."""
    # Source-level check.
    fn_start = RAG_PY.find("def _ocr_image(image_path: Path) -> str:")
    assert fn_start >= 0, "_ocr_image() no encontrado"
    fn_body = RAG_PY[fn_start : fn_start + 3000]
    assert "_load_ocrmac_module()" in fn_body, (
        "_ocr_image() debe invocar _load_ocrmac_module() — el lazy wrapper "
        "es el que garantiza sincronía con los test mocks"
    )


# ─── Fix 2: reranker batch_size=1 removal ─────────────────────────────────

def test_multi_retrieve_reranker_no_longer_forces_batch_size_1():
    """La llamada a `reranker.predict` en el path multi-retrieve / graph
    expansion NO debe tener `batch_size=1`. Ese era un artefacto stale
    del fix de pool=2-4 que a pool=30 forzaba 30 inferencias
    secuenciales MPS (~500ms-1s) en vez de una pasada batched."""
    # El comentario marker vive antes del call; saltamos hasta el
    # `reranker.predict(` para leer solo la call site, no el comentario
    # que contiene referencias textuales a batch_size=1.
    anchor = RAG_PY.find("# batch_size default (32) en lugar de batch_size=1")
    assert anchor >= 0, (
        "Comentario sobre batch_size default no encontrado — ¿se revirtió "
        "el fix del saneamiento Tier 1?"
    )
    # Busca la apertura de `reranker.predict(` a partir del anchor.
    call_start = RAG_PY.find("reranker.predict", anchor)
    assert call_start >= 0, "reranker.predict no encontrado post-comentario"
    call_end = RAG_PY.find(")", call_start)
    assert call_end >= 0
    call_site = RAG_PY[call_start : call_end + 1]
    assert "batch_size=1" not in call_site, (
        f"La call de reranker.predict en multi-retrieve volvió a tener "
        f"batch_size=1. Call actual: {call_site!r}"
    )


def test_no_stale_batch_size_1_anywhere_in_reranker_calls():
    """Confirma que NO hay ninguna call a `reranker.predict` en rag.py
    con `batch_size=1`. Hay 6 call sites totales; ninguno debe tener
    batch_size=1 después del saneamiento Tier 1."""
    # Grep each `reranker.predict(` call and verify the arguments.
    import re

    # Match reranker.predict(...) across possibly-multiline args.
    pattern = re.compile(
        r"reranker\.predict\s*\((?:[^()]|\([^)]*\))*\)",
        re.DOTALL,
    )
    matches = pattern.findall(RAG_PY)
    assert len(matches) >= 4, (
        f"Esperabamos ≥4 calls a reranker.predict, encontré {len(matches)}"
    )
    for m in matches:
        assert "batch_size=1" not in m, (
            f"Call a reranker.predict con batch_size=1 todavía presente: "
            f"{m[:200]}..."
        )


# ─── Fix 3: eager warmup en CLI ───────────────────────────────────────────

def test_cli_warmup_subcommands_frozenset_exists():
    """El allowlist `_CLI_WARMUP_SUBCOMMANDS` debe existir como
    frozenset y contener los subcomandos retrieve-heavy."""
    assert hasattr(rag, "_CLI_WARMUP_SUBCOMMANDS"), (
        "Falta rag._CLI_WARMUP_SUBCOMMANDS — el fix de warmup eager no "
        "está aplicado"
    )
    allowlist = rag._CLI_WARMUP_SUBCOMMANDS
    assert isinstance(allowlist, frozenset)
    # Anchor subcommands — si cualquiera de estos no está, el warmup
    # se pierde para un path retrieve-heavy.
    for expected in ["query", "chat", "morning", "today", "digest", "eval"]:
        assert expected in allowlist, (
            f"{expected!r} no está en _CLI_WARMUP_SUBCOMMANDS — ese "
            f"subcomando retrieva y debe disparar warmup eager"
        )


def test_cli_triggers_warmup_for_retrieve_subcommand():
    """Invocar un subcomando del allowlist vía CliRunner debe disparar
    `warmup_async()`. Usamos un spy para no pagar el warmup real."""
    fired = {"count": 0}

    def _spy_warmup():
        fired["count"] += 1

    with patch.object(rag, "warmup_async", _spy_warmup):
        runner = CliRunner()
        # `rag query --help` dispara el callback del grupo pero no ejecuta
        # el retrieve — suficiente para testear el trigger del warmup.
        result = runner.invoke(rag.cli, ["query", "--help"])
        assert result.exit_code == 0, result.output
        assert fired["count"] == 1, (
            f"warmup_async no se disparó para `rag query --help`; "
            f"firings={fired['count']}"
        )


def test_cli_skips_warmup_for_non_retrieve_subcommand():
    """Subcomandos de mantenimiento/config no deben disparar warmup. Usamos
    `vault --help` que pertenece al group 'vault' (no en el allowlist)."""
    fired = {"count": 0}

    def _spy_warmup():
        fired["count"] += 1

    with patch.object(rag, "warmup_async", _spy_warmup):
        runner = CliRunner()
        result = runner.invoke(rag.cli, ["vault", "--help"])
        assert result.exit_code == 0, result.output
        assert fired["count"] == 0, (
            "warmup_async se disparó para `rag vault --help` — vault es "
            "un group de administración, no debe pagar warmup"
        )


def test_cli_skips_warmup_for_help_alone():
    """`rag --help` sin subcomando no debe disparar warmup tampoco."""
    fired = {"count": 0}

    def _spy_warmup():
        fired["count"] += 1

    with patch.object(rag, "warmup_async", _spy_warmup):
        runner = CliRunner()
        result = runner.invoke(rag.cli, ["--help"])
        assert result.exit_code == 0
        assert fired["count"] == 0, (
            "warmup_async se disparó para `rag --help` (sin subcommand)"
        )


def test_cli_warmup_failure_does_not_crash_cli():
    """Si `warmup_async()` lanza, el CLI debe seguir funcionando — es
    best-effort en background, no un requisito."""
    def _broken_warmup():
        raise RuntimeError("simulated warmup failure")

    with patch.object(rag, "warmup_async", _broken_warmup):
        runner = CliRunner()
        result = runner.invoke(rag.cli, ["query", "--help"])
        assert result.exit_code == 0, (
            f"El CLI crasheó cuando warmup_async() falló: {result.output}"
        )


# ─── Regression guard: no se re-agregan imports eager ─────────────────────

def test_ocrmac_not_imported_at_module_level():
    """Guard de regresión: nadie debe re-agregar `from ocrmac import ...`
    al top-level de rag.py. El import debe vivir SOLO adentro de
    `_load_ocrmac_module()`."""
    # Buscamos imports explícitos de ocrmac fuera del helper lazy.
    lines = RAG_PY.splitlines()
    offenders = []
    inside_helper = False
    for i, line in enumerate(lines, start=1):
        if "def _load_ocrmac_module" in line:
            inside_helper = True
            continue
        if inside_helper and line.startswith("def "):
            inside_helper = False
        stripped = line.strip()
        if not inside_helper and (
            stripped.startswith("from ocrmac import")
            or stripped.startswith("import ocrmac")
        ):
            offenders.append(f"line {i}: {line}")
    assert not offenders, (
        "Encontré imports eager de ocrmac fuera de _load_ocrmac_module():\n"
        + "\n".join(offenders)
    )
