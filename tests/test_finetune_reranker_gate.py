"""Tests for GC#2.C (2026-04-23) — LoRA adapter loader + fine-tune gate.

Invariants verified here (the 6 cases from `docs/gamechangers-plan-2026-04-22.md`
section "Game-changer #2.C"):

  1. Without `RAG_RERANKER_FT`, the reranker init MUST NOT touch the LoRA
     adapter dir or import peft. Default behaviour is bit-identical to
     pre-GC#2.C.
  2. With `RAG_RERANKER_FT=1` and an EMPTY adapter dir, the loader falls
     back to the base model and writes a row to `silent_errors.jsonl`.
     User-visible queries continue to work.
  3. With `RAG_RERANKER_FT=1` and a VALID-looking adapter dir, the loader
     calls peft and splices the adapter on top of the base model. We
     assert via mocks because peft isn't required to be installed in
     this test environment.
  4. The eval gate enforced by `scripts/finetune_reranker.py --mode lora`
     produces metrics that include before/after nDCG@5; we sanity-check
     the helper math (`_ndcg_at_k`, `_pair_ranking_correlation`) here so
     a future refactor that breaks them is caught.
  5. The `__call__` (`predict`) of a LoRA-wrapped CrossEncoder still
     returns scores in the same [0, 1] range as the base model. We can't
     verify the actual model in CI (peft not installed), so we validate
     the contract via mocking.
  6. Smoke: importing rag with `RAG_RERANKER_FT=1` set + a missing
     adapter dir does not raise — degraded mode keeps the CLI usable.

Notes on env isolation:
  - The autouse fixture below patches `Path.home()` so every test sees
    the LoRA adapter dir under `tmp_path`. Without this the tests would
    look at the real `~/.local/share/obsidian-rag/reranker_ft/` and
    leak state across runs.
  - Tests that need to exercise `get_reranker()` mock the heavy
    `sentence_transformers.CrossEncoder` import to avoid downloading
    a 2 GB model during a unit-test run.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import rag


# ── Shared fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def ft_home(tmp_path, monkeypatch):
    """Redirect `~/.local/share/obsidian-rag/reranker_ft/` to a tmp dir.

    Also unsets `RAG_RERANKER_FT` so each test starts clean. Tests that
    want the flag ON set it explicitly via `monkeypatch.setenv`.
    """
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.delenv("RAG_RERANKER_FT", raising=False)
    monkeypatch.delenv("RAG_RERANKER_FT_PATH", raising=False)
    # The module-level constant was bound at import time (computed from
    # `Path.home()` ANTES de patcharlo), so we also patch the module
    # attribute so callers that read `rag.RERANKER_FT_ADAPTER_DIR` see
    # the redirected path.
    adapter_dir = tmp_path / ".local" / "share" / "obsidian-rag" / "reranker_ft"
    monkeypatch.setattr(rag, "RERANKER_FT_ADAPTER_DIR", adapter_dir)
    yield adapter_dir


@pytest.fixture
def fake_cross_encoder():
    """Patch `sentence_transformers.CrossEncoder` with a lightweight mock.

    The mock has a `.model` attribute (where peft would splice the
    adapter), a `.predict()` method that returns deterministic scores
    in [0, 1], and tracks how many times it was instantiated so tests
    can assert "loaded once" / "loaded zero times".
    """
    instances = []

    class _FakeCE:
        def __init__(self, model_path, max_length=512, device="cpu"):
            self.model_path = model_path
            self.max_length = max_length
            self.device = device
            self.model = MagicMock(name="underlying_hf_model")
            instances.append(self)

        def predict(self, pairs):
            # Return [0,1] scores deterministic from input length so tests
            # can assert range without needing a real model.
            return [min(1.0, max(0.0, len(b) / 1000.0)) for _, b in pairs]

        # CrossEncoder is callable in some sentence-transformers versions
        # via __call__; mirror that.
        def __call__(self, pairs):
            return self.predict(pairs)

    with patch("sentence_transformers.CrossEncoder", _FakeCE):
        yield instances


@pytest.fixture
def reset_global_reranker(monkeypatch):
    """Drop the cached `_reranker` singleton so each test triggers a
    fresh `get_reranker()` flow. The CrossEncoder load is mocked so
    this is fast (<1ms).
    """
    monkeypatch.setattr(rag, "_reranker", None)
    yield


def _write_valid_adapter(adapter_dir: Path) -> None:
    """Drop a minimal adapter_config.json into `adapter_dir`.

    The actual peft loader would also need an adapter weights file, but
    our test doesn't exercise the real load — we mock peft.PeftModel.
    """
    adapter_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "peft_type": "LORA",
        "task_type": "SEQ_CLS",
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "target_modules": ["query", "value"],
        "bias": "none",
        "base_model_name_or_path": "BAAI/bge-reranker-v2-m3",
    }
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(cfg), encoding="utf-8",
    )


# ── Case 1: env unset → no adapter touch ──────────────────────────────────


def test_env_unset_does_not_touch_adapter(
    ft_home, fake_cross_encoder, reset_global_reranker, monkeypatch,
):
    """Default-OFF contract: with RAG_RERANKER_FT unset, the reranker
    init must not call into peft, must not even check for the adapter
    dir. The LoRA path is a strict opt-in.
    """
    # Make sure the adapter dir exists with valid config — if the loader
    # accidentally branches on dir-existence instead of the env flag,
    # this test catches it.
    _write_valid_adapter(ft_home)

    apply_call_count = {"n": 0}

    def _spy(model, adapter_dir):
        apply_call_count["n"] += 1
        return True

    monkeypatch.setattr(rag, "_apply_reranker_lora_adapter", _spy)

    # Force device picker into a deterministic state.
    with patch("torch.backends.mps.is_available", return_value=False), \
         patch("torch.cuda.is_available", return_value=False):
        rag.get_reranker()

    assert apply_call_count["n"] == 0, (
        "_apply_reranker_lora_adapter should NOT be called when "
        "RAG_RERANKER_FT is unset"
    )


# ── Case 2: empty adapter dir → fallback + silent_errors log ──────────────


def test_empty_adapter_dir_falls_back_with_warning(
    ft_home, fake_cross_encoder, reset_global_reranker, monkeypatch, tmp_path,
):
    """With RAG_RERANKER_FT=1 and an EMPTY adapter dir, the loader must:
      1. Not raise.
      2. Leave the reranker on the base model.
      3. Log to silent_errors.jsonl (we capture via _silent_log spy).
    """
    monkeypatch.setenv("RAG_RERANKER_FT", "1")
    # Ensure ft_home dir exists but is empty (no adapter_config.json).
    ft_home.mkdir(parents=True, exist_ok=True)

    silent_log_calls = []

    def _spy_silent_log(where, exc):
        silent_log_calls.append((where, type(exc).__name__))

    monkeypatch.setattr(rag, "_silent_log", _spy_silent_log)

    with patch("torch.backends.mps.is_available", return_value=False), \
         patch("torch.cuda.is_available", return_value=False):
        reranker = rag.get_reranker()

    # Should not raise; reranker is the base CrossEncoder.
    assert reranker is not None
    # Should have logged an error about the missing adapter_config.
    assert any(
        "adapter_config_missing" in where or "adapter_dir_missing" in where
        for where, _ in silent_log_calls
    ), f"expected silent_log to record missing adapter, got {silent_log_calls}"


# ── Case 3: valid adapter → peft load called ──────────────────────────────


def test_valid_adapter_dir_invokes_peft(
    ft_home, fake_cross_encoder, reset_global_reranker, monkeypatch,
):
    """With RAG_RERANKER_FT=1 and a valid-looking adapter dir, the loader
    must import peft and call PeftModel.from_pretrained. We mock peft
    to avoid the dependency in CI.
    """
    monkeypatch.setenv("RAG_RERANKER_FT", "1")
    _write_valid_adapter(ft_home)

    fake_peft_model = MagicMock(name="peft_model")
    fake_peft_module = MagicMock()
    fake_peft_module.PeftModel.from_pretrained = MagicMock(
        return_value=fake_peft_model,
    )
    # Inject our fake `peft` so the loader's `from peft import PeftModel`
    # finds it. The module-level import inside `_apply_reranker_lora_adapter`
    # picks up whatever is in sys.modules at call time.
    monkeypatch.setitem(sys.modules, "peft", fake_peft_module)

    with patch("torch.backends.mps.is_available", return_value=False), \
         patch("torch.cuda.is_available", return_value=False):
        reranker = rag.get_reranker()

    fake_peft_module.PeftModel.from_pretrained.assert_called_once()
    # The CrossEncoder's `.model` should now be the peft-wrapped one.
    assert reranker.model is fake_peft_model, (
        "loader should splice the PeftModel back into CrossEncoder.model"
    )


# ── Case 4: nDCG@5 helper math ────────────────────────────────────────────


def test_ndcg_at_5_helper_math():
    """Sanity-check the nDCG@5 helper used in the LoRA mode metrics.

    Three scenarios:
      - Perfect ranking (positives at the top) → nDCG = 1.0
      - Reverse ranking (negatives at the top) → nDCG < 1.0
      - All-positive list → nDCG = 1.0 (all positions rel=1)
    """
    spec = importlib.util.spec_from_file_location(
        "_finetune_script",
        Path(__file__).resolve().parent.parent / "scripts" / "finetune_reranker.py",
    )
    # We can't import the module wholesale (it imports rag and brings the
    # entire stack). Instead, copy out the helpers we want to test by
    # exec'ing only the relevant function bodies. Conservative — keeps
    # the test cheap and avoids touching telemetry on import.
    helper_src = (Path(__file__).resolve().parent.parent
                  / "scripts" / "finetune_reranker.py").read_text(encoding="utf-8")

    # Locate the `_ndcg_at_k` and `_pair_ranking_correlation` defs.
    import re
    ns: dict = {}
    for fn_name in ("_ndcg_at_k", "_pair_ranking_correlation"):
        match = re.search(
            rf"^def {fn_name}\(.*?(?=^\ndef |\Z)", helper_src, re.S | re.M,
        )
        assert match, f"could not extract {fn_name} from script"
        exec(match.group(0), ns)

    ndcg = ns["_ndcg_at_k"]
    auc = ns["_pair_ranking_correlation"]

    # Perfect: positives have higher scores than negatives.
    perfect = [(0.9, 1.0), (0.8, 1.0), (0.2, 0.0), (0.1, 0.0)]
    assert ndcg(perfect, k=5) == pytest.approx(1.0, abs=1e-6)

    # Reversed: negatives have higher scores → nDCG < 1.0
    reversed_ = [(0.9, 0.0), (0.8, 0.0), (0.2, 1.0), (0.1, 1.0)]
    assert ndcg(reversed_, k=5) < 1.0

    # AUC: perfect ranking → 1.0
    assert auc(perfect) == pytest.approx(1.0)
    # AUC: reversed ranking → 0.0
    assert auc(reversed_) == pytest.approx(0.0)
    # AUC: degenerate (only one bucket) → None
    assert auc([(0.5, 1.0), (0.6, 1.0)]) is None


# ── Case 5: __call__ returns scores in [0,1] ──────────────────────────────


def test_predict_scores_in_unit_range(
    ft_home, fake_cross_encoder, reset_global_reranker, monkeypatch,
):
    """The reranker's __call__/predict must keep returning scores in
    [0, 1] regardless of whether the LoRA path is active. The base
    bge-reranker-v2-m3 produces sigmoid'd cross-encoder scores; the LoRA
    overlay should not change the output range (only the magnitudes).
    """
    monkeypatch.setenv("RAG_RERANKER_FT", "1")
    _write_valid_adapter(ft_home)

    # Mock peft so the load path succeeds.
    fake_peft_module = MagicMock()
    fake_peft_module.PeftModel.from_pretrained = MagicMock(
        return_value=MagicMock(name="peft_model"),
    )
    monkeypatch.setitem(sys.modules, "peft", fake_peft_module)

    with patch("torch.backends.mps.is_available", return_value=False), \
         patch("torch.cuda.is_available", return_value=False):
        reranker = rag.get_reranker()

    pairs = [
        ("query short", "doc"),
        ("query medium length", "longer doc " * 20),
        ("query", "x" * 5000),
    ]
    scores = reranker.predict(pairs)
    assert all(0.0 <= s <= 1.0 for s in scores), (
        f"reranker scores out of [0,1] range: {scores}"
    )


# ── Case 6: smoke — flag ON with missing adapter does not break import ────


def test_smoke_flag_on_missing_adapter_does_not_raise(
    ft_home, fake_cross_encoder, reset_global_reranker, monkeypatch,
):
    """Flag ON + adapter dir missing entirely must NOT raise. The
    operator should be able to set RAG_RERANKER_FT=1 in their shell
    config and have everything still work (degraded to base model)
    even before they've trained any adapter.
    """
    monkeypatch.setenv("RAG_RERANKER_FT", "1")
    # ft_home points at a path that does not exist on disk yet.
    assert not ft_home.exists()

    # Silence silent_log noise in this smoke test.
    monkeypatch.setattr(rag, "_silent_log", lambda *_a, **_kw: None)

    with patch("torch.backends.mps.is_available", return_value=False), \
         patch("torch.cuda.is_available", return_value=False):
        # Should not raise — degraded path.
        reranker = rag.get_reranker()
    assert reranker is not None


# ── Bonus: helper-level invariants for `_reranker_ft_enabled()` ───────────


@pytest.mark.parametrize(
    "value, expected",
    [
        ("1", True),
        ("true", True),
        ("True", True),
        ("yes", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("", False),
        ("anything-else", False),
    ],
)
def test_reranker_ft_enabled_truthy_table(monkeypatch, value, expected):
    """The flag parser accepts the same truthy variants as the rest of
    the codebase ("1"/"true"/"yes"/"on"). Anything else (including empty
    string) is False — operators who unset the var don't get the LoRA.
    """
    monkeypatch.setenv("RAG_RERANKER_FT", value)
    assert rag._reranker_ft_enabled() is expected


def test_reranker_ft_enabled_unset(monkeypatch):
    monkeypatch.delenv("RAG_RERANKER_FT", raising=False)
    assert rag._reranker_ft_enabled() is False


# ── Bonus: `_apply_reranker_lora_adapter` direct contract ─────────────────


def test_apply_lora_adapter_dir_not_found_returns_false(monkeypatch, tmp_path):
    """Calling the helper directly with a non-existent dir must return
    False and not raise. This is the contract that protects the hot path
    in `get_reranker()` from a stale/missing adapter dir.
    """
    silent = []
    monkeypatch.setattr(rag, "_silent_log", lambda where, exc: silent.append(where))
    fake_model = MagicMock()
    result = rag._apply_reranker_lora_adapter(
        fake_model, tmp_path / "does-not-exist",
    )
    assert result is False
    assert any("adapter_dir_missing" in s for s in silent)


def test_apply_lora_adapter_peft_missing_returns_false(monkeypatch, tmp_path):
    """If peft is not installed, the helper logs and returns False.
    Simulated by injecting an ImportError-raising stand-in into
    sys.modules['peft'].
    """
    _write_valid_adapter(tmp_path)
    silent = []
    monkeypatch.setattr(rag, "_silent_log", lambda where, exc: silent.append(where))
    # Force the import to fail: remove any existing peft entry and
    # inject a dummy that raises on attribute access.
    monkeypatch.setitem(sys.modules, "peft", None)
    fake_model = MagicMock()
    result = rag._apply_reranker_lora_adapter(fake_model, tmp_path)
    assert result is False
    assert any("peft_missing" in s for s in silent)


def test_apply_lora_adapter_no_underlying_model(monkeypatch, tmp_path):
    """If the CrossEncoder has no `.model` attribute (older or future
    sentence-transformers versions might use a different name), the
    helper logs and falls back. We use a SimpleNamespace without
    `.model` to trigger this path.
    """
    import types
    _write_valid_adapter(tmp_path)
    silent = []
    monkeypatch.setattr(rag, "_silent_log", lambda where, exc: silent.append(where))
    # Provide a fake peft so we get past the import-check.
    fake_peft = MagicMock()
    fake_peft.PeftModel.from_pretrained = MagicMock(return_value=MagicMock())
    monkeypatch.setitem(sys.modules, "peft", fake_peft)
    fake_model = types.SimpleNamespace()  # no .model attr
    result = rag._apply_reranker_lora_adapter(fake_model, tmp_path)
    assert result is False
    assert any("no_underlying_model" in s for s in silent)
