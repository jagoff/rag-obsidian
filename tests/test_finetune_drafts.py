"""Tests for the WhatsApp drafts fine-tune pipeline (DPO+LoRA, 2026-05-01).

Cubre:
  1. Mining de preference pairs desde rag_draft_decisions (solo gold —
     decision='approved_editar'). Las rows 'rejected' se descartan
     porque DPO requiere AMBOS chosen y rejected; si el user rechaza
     un draft sin escribir alternativa, no hay chosen.
  2. Filtro --exclude-review-only.
  3. Insuficiente data (<100 pares gold) → exit code 1.
  4. Endpoint /api/draft/preview con flag OFF → echo del baseline.
  5. Endpoint con flag ON pero adapter missing → silent fallback + log.
  6. Smoke test del CLI `rag draft stats --plain`.

NO requiere peft / transformers / trl — los tests stubean el import
del modelo. Si esos paquetes están instalados, los tests siguen
pasando porque siguen el mismo path mock.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rag  # noqa: E402

# Importamos el script via path (no es un package). Usamos importlib
# para que el test se compile aunque cambien las internals del script.
import importlib.util as _ilu  # noqa: E402

_SCRIPT_PATH = ROOT / "scripts" / "finetune_drafts.py"
_spec = _ilu.spec_from_file_location("finetune_drafts", _SCRIPT_PATH)
finetune_drafts = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(finetune_drafts)


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def state_db(tmp_path, monkeypatch):
    """Aísla telemetry.db en tmp_path. Tablas se crean on-demand."""
    db_path = tmp_path / "ragvec"
    db_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(rag, "DB_PATH", db_path)
    rag.SqliteVecClient(path=str(db_path))
    with rag._ragvec_state_conn() as _conn:
        pass
    return db_path


def _seed_decision(
    *, draft_id: str, decision: str, bot_draft: str = "draft",
    sent_text: str | None = None, contact_jid: str = "x@s.whatsapp.net",
    contact_name: str = "Test", original_msgs: list | None = None,
    extra: dict | None = None, ts: str = "2026-04-29T10:00:00",
) -> int:
    """Inserta una row en rag_draft_decisions usando el helper público."""
    return rag._record_draft_decision(
        draft_id=draft_id, contact_jid=contact_jid, contact_name=contact_name,
        original_msgs=original_msgs or [{"id": "m1", "text": "hola", "ts": ts}],
        bot_draft=bot_draft, decision=decision, sent_text=sent_text,
        extra=extra,
    ) or 0


# ── Test 1: mining de pairs ───────────────────────────────────────────────


def test_mining_pairs_returns_gold_only_skips_rejected(state_db):
    """DPO requiere preference pairs completos (chosen + rejected).

    `decision='approved_editar'` → row gold (chosen=sent_text,
    rejected=bot_draft).
    `decision='rejected'` → no hay chosen alternativo → SKIP. El
    contador `n_rejected_skipped` documenta cuánta señal se descarta.
    `decision='approved_si'` / 'expired' → fuera de la query SQL.
    """
    # 2 approved_editar (gold), 3 rejected (skipped), 1 approved_si
    # (ignorado por SQL), 1 expired (ignorado por SQL).
    _seed_decision(draft_id="g1", decision="approved_editar",
                   bot_draft="draft1", sent_text="edited1")
    _seed_decision(draft_id="g2", decision="approved_editar",
                   bot_draft="draft2", sent_text="edited2")
    _seed_decision(draft_id="r1", decision="rejected", bot_draft="draft3")
    _seed_decision(draft_id="r2", decision="rejected", bot_draft="draft4")
    _seed_decision(draft_id="r3", decision="rejected", bot_draft="draft5")
    _seed_decision(draft_id="a1", decision="approved_si",
                   bot_draft="hi", sent_text="hi")
    _seed_decision(draft_id="e1", decision="expired", bot_draft="bye")

    data = finetune_drafts.fetch_draft_pairs(exclude_review_only=False)

    assert len(data["gold"]) == 2
    # `anti` queda vacía siempre — DPO no usa pseudo-anti-patterns.
    assert len(data["anti"]) == 0
    assert data["stats"]["n_gold"] == 2
    assert data["stats"]["n_anti"] == 0
    assert data["stats"]["n_rejected_skipped"] == 3
    # Total rows leídas del SQL (approved_editar + rejected, sin
    # approved_si/expired) = 5.
    assert data["stats"]["n_total_rows"] == 5

    # Verificá que las gold rows tienen sent_text correcto.
    gold_chosen = sorted(item["sent_text"] for item in data["gold"])
    assert gold_chosen == ["edited1", "edited2"]


def test_mining_skips_degenerate_pair_chosen_equals_rejected(state_db):
    """Si sent_text == bot_draft (preference vacía) → skip silently.

    DPO log-ratio = log p(chosen) - log p(rejected) = 0 cuando son
    iguales → gradient = 0 → el sample no aporta nada y solo dilye
    el batch. Edge case raro (no debería ocurrir en captura real
    porque sería approved_si) pero defensivo.
    """
    _seed_decision(draft_id="ok", decision="approved_editar",
                   bot_draft="hola", sent_text="che hola!")
    _seed_decision(draft_id="degen", decision="approved_editar",
                   bot_draft="igual", sent_text="igual")

    data = finetune_drafts.fetch_draft_pairs()
    # Solo el non-degenerate sobrevive.
    assert len(data["gold"]) == 1
    assert data["gold"][0]["draft_id"] == "ok"


# ── Test 2: filtro --exclude-review-only ─────────────────────────────────


def test_exclude_review_only_filters_correctly(state_db):
    """Rows con extra_json.review_only=true se excluyen cuando flag ON,
    se incluyen cuando flag OFF (default). Default = signal real igual.

    Nota: `decision='rejected'` rows se descartan SIEMPRE (DPO no las
    usa). Acá sembramos una rejected review-only solo para confirmar
    que el flag review-only la cuenta antes que el rejected-skip.
    """
    # Mix: 2 gold normales + 1 gold review-only + 1 rejected review-only.
    _seed_decision(draft_id="g_normal", decision="approved_editar",
                   bot_draft="d1", sent_text="e1")
    _seed_decision(draft_id="g_normal2", decision="approved_editar",
                   bot_draft="d2", sent_text="e2")
    _seed_decision(draft_id="g_review", decision="approved_editar",
                   bot_draft="d3", sent_text="e3",
                   extra={"review_only": True})
    _seed_decision(draft_id="r_review", decision="rejected",
                   bot_draft="d4", extra={"review_only": True})

    # Default: incluir review-only. 3 gold (2 normales + 1 review-only).
    # La rejected review-only se cuenta primero como review-only
    # (=2 total), después como rejected-skipped.
    data_inc = finetune_drafts.fetch_draft_pairs(exclude_review_only=False)
    assert len(data_inc["gold"]) == 3
    assert len(data_inc["anti"]) == 0
    assert data_inc["stats"]["n_rejected_skipped"] == 1
    assert data_inc["stats"]["n_review_only_total"] == 2
    assert data_inc["stats"]["n_review_only_excluded"] == 0

    # Con flag: 2 gold sobreviven (review-only excluidos antes del
    # rejected-skip → la rejected review-only NO incrementa el
    # rejected_skipped count).
    data_exc = finetune_drafts.fetch_draft_pairs(exclude_review_only=True)
    assert len(data_exc["gold"]) == 2
    assert len(data_exc["anti"]) == 0
    assert data_exc["stats"]["n_rejected_skipped"] == 0
    assert data_exc["stats"]["n_review_only_total"] == 2
    assert data_exc["stats"]["n_review_only_excluded"] == 2


# ── Test 3: insuficiente data → exit 1 ───────────────────────────────────


def test_insufficient_data_exits_with_clear_message(state_db, monkeypatch):
    """Si total pares <100 → exit code 1 con mensaje accionable.

    Smoke runs el script en subprocess para validar el exit code real.
    `RAG_DRAFTS_TEST_DB_PATH` no existe — usamos monkeypatch on env
    para que el script vea el state_db isolado (mismo HOME a nivel
    process). Workaround: seteamos HOME = tmp_path/home y poblamos
    el DB del script desde Python.

    Más simple: invocamos `main()` directo y catcheamos SystemExit.
    """
    # Sembramos solo 5 gold (claramente <100).
    for i in range(5):
        _seed_decision(draft_id=f"g{i}", decision="approved_editar",
                       bot_draft=f"d{i}", sent_text=f"e{i}")

    # Invoca main() del script con args [--dry-run]. main() llama a
    # sys.exit(1) cuando len(examples) < MIN_PAIRS.
    monkeypatch.setattr(sys, "argv", ["finetune_drafts.py", "--dry-run"])
    with pytest.raises(SystemExit) as exc_info:
        finetune_drafts.main()
    assert exc_info.value.code == 1


def test_dry_run_with_enough_pairs_exits_zero(state_db, monkeypatch):
    """Con ≥100 GOLD + --dry-run → reporta stats sin entrenar y returns.

    DPO requiere preference pairs completos: 100 gold (approved_editar
    con sent_text != bot_draft). Las rows 'rejected' YA NO cuentan
    porque no aportan a DPO sin un chosen alternativo.
    """
    # 110 gold (>100 mínimo). Nota: el threshold MIN_PAIRS=100 es
    # sobre GOLD, no gold+rejected.
    for i in range(110):
        _seed_decision(draft_id=f"g{i}", decision="approved_editar",
                       bot_draft=f"draft{i}", sent_text=f"sent{i}")
    # Sembramos 5 rejected para verificar que se cuentan como skipped
    # pero NO bloquean el dry-run.
    for i in range(5):
        _seed_decision(draft_id=f"r{i}", decision="rejected",
                       bot_draft=f"baddraft{i}")

    monkeypatch.setattr(sys, "argv", ["finetune_drafts.py", "--dry-run"])
    # main() returns normally (sin SystemExit) en dry-run con enough pairs.
    finetune_drafts.main()
    # No assert: el test pasa si main() retorna sin raise.


# ── Test 4: endpoint flag OFF → echo ──────────────────────────────────────


def test_preview_endpoint_flag_off_echoes_baseline(state_db, monkeypatch):
    """RAG_DRAFTS_FT no seteado → endpoint devuelve el baseline tal cual."""
    monkeypatch.delenv("RAG_DRAFTS_FT", raising=False)
    pytest.importorskip("fastapi.testclient")
    from fastapi.testclient import TestClient
    from web import server as _web_server

    client = TestClient(_web_server.app)
    payload = {
        "original_conversation": "hola, todo bien?",
        "bot_draft_baseline": "todo bien, vos?",
    }
    resp = client.post("/api/draft/preview", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True
    assert body["preview"] == "todo bien, vos?"
    assert body["ft_active"] is False


# ── Test 5: endpoint flag ON pero adapter missing → silent fallback ──────


def test_preview_endpoint_flag_on_adapter_missing_falls_back(
    state_db, monkeypatch, tmp_path,
):
    """RAG_DRAFTS_FT=1 + adapter dir vacío/inexistente → echo + log."""
    # Apuntamos el adapter dir a un path que NO existe — fuerza el
    # silent-fail path.
    fake_adapter = tmp_path / "nonexistent_adapter"
    monkeypatch.setattr(rag, "DRAFTS_FT_ADAPTER_DIR", fake_adapter)
    monkeypatch.setenv("RAG_DRAFTS_FT", "1")
    # Reset cache del modelo (puede haber sido populado por otro test).
    monkeypatch.setattr(rag, "_drafts_ft_model", None)
    monkeypatch.setattr(rag, "_drafts_ft_tokenizer", None)

    pytest.importorskip("fastapi.testclient")
    from fastapi.testclient import TestClient
    from web import server as _web_server
    client = TestClient(_web_server.app)

    payload = {
        "original_conversation": "qué hacés?",
        "bot_draft_baseline": "todo bien capo",
    }
    resp = client.post("/api/draft/preview", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True
    # Aunque la flag esté ON, sin adapter el endpoint cae a echo.
    assert body["preview"] == "todo bien capo"
    # ft_active es False porque el adapter no está disponible.
    assert body["ft_active"] is False


def test_preview_endpoint_helper_never_raises(monkeypatch, tmp_path):
    """Smoke: aunque `_load_drafts_ft_model` raisee internamente, el
    helper público `generate_draft_preview` SIEMPRE devuelve un string
    (mínimo el baseline)."""
    monkeypatch.setenv("RAG_DRAFTS_FT", "1")
    fake_adapter = tmp_path / "broken_adapter"
    fake_adapter.mkdir()
    # Adapter dir existe pero adapter_config.json falta → cheap check rebota.
    monkeypatch.setattr(rag, "DRAFTS_FT_ADAPTER_DIR", fake_adapter)
    monkeypatch.setattr(rag, "_drafts_ft_model", None)
    monkeypatch.setattr(rag, "_drafts_ft_tokenizer", None)

    out = rag.generate_draft_preview(
        original_conversation="hola",
        bot_draft_baseline="echo me",
    )
    assert out == "echo me"


# ── Test 6: smoke `rag drafts stats --plain` ──────────────────────────────


def test_cli_drafts_stats_plain_runs_without_error(
    state_db, monkeypatch, tmp_path,
):
    """Smoke: el CLI `rag drafts stats --plain` corre sin error y
    produce output esperado (total + breakdown + 30d window).

    Usa CliRunner para no spawnar subproceso (más rápido + determinista
    bajo conftest's autouse fixtures). Equivalente funcional a
    `subprocess.run(['rag', 'drafts', 'stats', '--plain'])`.

    Nota: aislamos `DRAFTS_FT_ADAPTER_DIR` a tmp_path para que el test
    no lea el adapter REAL del user (que existe en
    ~/.local/share/obsidian-rag/drafts_ft/ y haría fallar el assert
    "not trained yet"). Bug pre-existente arreglado en el refactor
    DPO 2026-05-01.
    """
    # Aislá el adapter dir — sin esto, el test lee el adapter del
    # filesystem real del user.
    isolated_adapter = tmp_path / "drafts_ft_isolated"
    monkeypatch.setattr(rag, "DRAFTS_FT_ADAPTER_DIR", isolated_adapter)

    # Sembramos algo para que stats tenga qué reportar.
    _seed_decision(draft_id="d1", decision="approved_si",
                   bot_draft="hi", sent_text="hi")
    _seed_decision(draft_id="d2", decision="approved_editar",
                   bot_draft="d", sent_text="e")
    _seed_decision(draft_id="d3", decision="rejected", bot_draft="x")

    from click.testing import CliRunner
    runner = CliRunner()
    # `rag drafts` (plural alias) → `rag draft` (singular subgroup).
    result = runner.invoke(rag.cli, ["drafts", "stats", "--plain"])
    assert result.exit_code == 0, (
        f"exit={result.exit_code}\noutput:\n{result.output}\n"
        f"exc:\n{result.exception}"
    )
    # Verifica que el output contiene los conteos esperados.
    assert "total: 3" in result.output
    assert "approved_si: 1" in result.output
    assert "approved_editar: 1" in result.output
    assert "rejected: 1" in result.output
    # Sin adapter entrenado debe avisar.
    assert "not trained yet" in result.output


def test_cli_drafts_stats_plain_singular_alias_works(state_db):
    """`rag draft stats --plain` (singular) sigue funcionando — no
    rompimos el shell history del user."""
    _seed_decision(draft_id="d1", decision="approved_si",
                   bot_draft="hi", sent_text="hi")
    from click.testing import CliRunner
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["draft", "stats", "--plain"])
    assert result.exit_code == 0, result.output
    assert "total: 1" in result.output


# ── Test extra: bleu1 + similarity sanity ────────────────────────────────


def test_bleu1_and_similarity_basic_sanity():
    """Métricas básicas. bleu1=1.0 cuando pred==ref, sim=1.0 idem."""
    assert finetune_drafts.bleu1("hola mundo", "hola mundo") == 1.0
    assert finetune_drafts.similarity("hola mundo", "hola mundo") == 1.0
    assert finetune_drafts.bleu1("", "hola") == 0.0
    # Parcial overlap.
    assert 0 < finetune_drafts.bleu1("hola che", "hola che boludo") <= 1.0


def test_split_train_val_no_leak_by_draft_id():
    """Stratified split por draft_id evita que el mismo draft aparezca
    en train Y val. Schema DPO: {prompt, chosen, rejected, draft_id}."""
    examples = [
        {"draft_id": "d1", "prompt": "p", "chosen": "c", "rejected": "r"},
        {"draft_id": "d2", "prompt": "p", "chosen": "c", "rejected": "r"},
        {"draft_id": "d3", "prompt": "p", "chosen": "c", "rejected": "r"},
        {"draft_id": "d4", "prompt": "p", "chosen": "c", "rejected": "r"},
        {"draft_id": "d5", "prompt": "p", "chosen": "c", "rejected": "r"},
    ]
    train, val = finetune_drafts.split_train_val(examples, val_frac=0.4)
    train_ids = {ex["draft_id"] for ex in train}
    val_ids = {ex["draft_id"] for ex in val}
    # Sin overlap entre train y val.
    assert train_ids.isdisjoint(val_ids)
    # Total preserved.
    assert len(train) + len(val) == 5


# ── Test 7: build_dpo_example shape ───────────────────────────────────────


def test_build_dpo_example_shape_and_content():
    """`build_dpo_example` produce el formato TRL DPOTrainer espera:
    {prompt: str, chosen: str, rejected: str, draft_id: str}.
    El prompt MATCHEA producción (sin incluir bot_draft).
    """
    item = {
        "draft_id": "abc123",
        "contact_name": "Lu",
        "original_msgs": [
            {"text": "che fer cómo va?"},
            {"text": "todo joya?"},
        ],
        "bot_draft": "Estimada Lu, todo en orden, gracias por consultar.",
        "sent_text": "joya che, vos? todo bien?",
    }
    out = finetune_drafts.build_dpo_example(item)

    # Schema TRL.
    assert set(out.keys()) == {"prompt", "chosen", "rejected", "draft_id"}
    assert out["draft_id"] == "abc123"
    # chosen = sent_text (la respuesta real del user).
    assert out["chosen"] == "joya che, vos? todo bien?"
    # rejected = bot_draft (el corporate que el modelo propuso).
    assert out["rejected"].startswith("Estimada Lu")
    # Prompt incluye contacto + contexto, NO el bot_draft (matchea
    # producción, donde el modelo solo ve el contexto y debe generar).
    assert "Lu" in out["prompt"]
    assert "che fer cómo va?" in out["prompt"]
    assert "Estimada Lu" not in out["prompt"]
    # Termina con el "## Tu respuesta:" cue.
    assert out["prompt"].rstrip().endswith("## Tu respuesta:")


def test_preference_win_correctly_classifies():
    """`preference_win`: True si pred se parece más al chosen que al
    rejected. Es la métrica única de DPO held-out eval.
    """
    # Pred idéntico a chosen → win garantizado.
    assert finetune_drafts.preference_win(
        pred="dale, todo joya",
        chosen="dale, todo joya",
        rejected="estimado señor, todo en orden",
    ) is True
    # Pred idéntico a rejected → loss.
    assert finetune_drafts.preference_win(
        pred="estimado señor, todo en orden",
        chosen="dale, todo joya",
        rejected="estimado señor, todo en orden",
    ) is False
    # Pred a mitad de camino → tie → False (la spec dice "strictly
    # greater" para win, evita pseudo-wins por jitter numérico).
    assert finetune_drafts.preference_win(
        pred="hola",
        chosen="hola",
        rejected="hola",
    ) is False
