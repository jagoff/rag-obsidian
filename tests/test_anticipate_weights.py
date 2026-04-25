"""Tests for `rag_anticipate.weights` — user-configurable signal weights.

Cubre:
- load_weights: archivo inexistente, JSON válido, JSON malformado, filtros.
- save_weights: crea archivo, atomicidad (tmp + rename), filtra inválidos.
- set_weight / remove_weight: insert, update, reject out-of-range, idempotente.
- apply_weight: default 1.0, clamp [0, 1].
- list_weights: orden lexicográfico.
"""

from __future__ import annotations

import json

import pytest

from rag_anticipate import weights


@pytest.fixture
def tmp_weights(tmp_path, monkeypatch):
    """Aísla `WEIGHTS_PATH` a `tmp_path` para que ningún test toque
    `~/.local/share/obsidian-rag/anticipate_weights.json` real."""
    path = tmp_path / "weights.json"
    monkeypatch.setattr(weights, "WEIGHTS_PATH", path)
    yield path


# ── load_weights ──────────────────────────────────────────────────────────


def test_load_weights_missing_file_returns_empty(tmp_weights):
    assert not tmp_weights.exists()
    assert weights.load_weights() == {}


def test_load_weights_valid_json(tmp_weights):
    tmp_weights.write_text(
        json.dumps({"anticipate-calendar": 1.5, "anticipate-echo": 0.7}),
        encoding="utf-8",
    )
    out = weights.load_weights()
    assert out == {"anticipate-calendar": 1.5, "anticipate-echo": 0.7}
    # Siempre floats, aunque el JSON tenga ints.
    for v in out.values():
        assert isinstance(v, float)


def test_load_weights_malformed_json_silent_fail(tmp_weights):
    tmp_weights.write_text("{not valid json", encoding="utf-8")
    assert weights.load_weights() == {}


def test_load_weights_filters_out_of_range(tmp_weights):
    tmp_weights.write_text(
        json.dumps(
            {
                "good": 1.5,
                "too-high": 9.0,       # > 5.0 → filtrado
                "negative": -0.5,      # < 0.0 → filtrado
                "boundary-zero": 0.0,  # OK
                "boundary-five": 5.0,  # OK
                "string": "1.0",       # tipo inválido
                "bool": True,          # bool no cuenta como número
            }
        ),
        encoding="utf-8",
    )
    out = weights.load_weights()
    assert out == {"good": 1.5, "boundary-zero": 0.0, "boundary-five": 5.0}


def test_load_weights_root_not_dict_returns_empty(tmp_weights):
    tmp_weights.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    assert weights.load_weights() == {}


# ── save_weights ──────────────────────────────────────────────────────────


def test_save_weights_creates_file(tmp_weights):
    assert weights.save_weights({"anticipate-calendar": 1.5}) is True
    assert tmp_weights.is_file()
    data = json.loads(tmp_weights.read_text(encoding="utf-8"))
    assert data == {"anticipate-calendar": 1.5}


def test_save_weights_atomic_write_tmp_and_rename(tmp_weights, monkeypatch):
    """Verifica que `save_weights` use el patrón `tmp + rename` — capturamos
    el tmp file via interceptando `Path.replace` para confirmar que se llamó
    desde un sufijo `.json.tmp`."""
    captured: dict[str, object] = {}

    real_replace = type(tmp_weights).replace

    def spy_replace(self, target):
        captured["src"] = str(self)
        captured["dst"] = str(target)
        return real_replace(self, target)

    monkeypatch.setattr(type(tmp_weights), "replace", spy_replace)
    assert weights.save_weights({"x": 2.0}) is True
    assert captured["src"].endswith(".json.tmp")
    assert captured["dst"] == str(tmp_weights)
    # Y el archivo final existe con el contenido correcto.
    assert json.loads(tmp_weights.read_text(encoding="utf-8")) == {"x": 2.0}
    # El tmp ya no debería existir tras el rename.
    assert not tmp_weights.with_suffix(".json.tmp").exists()


def test_save_weights_filters_invalid_before_write(tmp_weights):
    ok = weights.save_weights(
        {
            "valid": 1.0,
            "too-high": 99.0,
            "negative": -1.0,
            "not-a-number": "1.0",  # type: ignore[dict-item]
        }
    )
    assert ok is True
    data = json.loads(tmp_weights.read_text(encoding="utf-8"))
    assert data == {"valid": 1.0}


# ── set_weight ────────────────────────────────────────────────────────────


def test_set_weight_insert(tmp_weights):
    assert weights.set_weight("anticipate-calendar", 1.5) is True
    assert weights.load_weights() == {"anticipate-calendar": 1.5}


def test_set_weight_update_existing(tmp_weights):
    weights.save_weights({"anticipate-calendar": 1.0, "anticipate-echo": 0.7})
    assert weights.set_weight("anticipate-calendar", 2.0) is True
    out = weights.load_weights()
    assert out["anticipate-calendar"] == 2.0
    assert out["anticipate-echo"] == 0.7  # no se tocó


def test_set_weight_rejects_out_of_range(tmp_weights):
    assert weights.set_weight("foo", 10.0) is False
    assert weights.set_weight("foo", -1.0) is False
    # Y nada se persistió.
    assert weights.load_weights() == {}


# ── remove_weight ─────────────────────────────────────────────────────────


def test_remove_weight_existing(tmp_weights):
    weights.save_weights({"anticipate-calendar": 1.5, "anticipate-echo": 0.7})
    assert weights.remove_weight("anticipate-calendar") is True
    assert weights.load_weights() == {"anticipate-echo": 0.7}


def test_remove_weight_absent_is_idempotent(tmp_weights):
    # Sobre archivo inexistente.
    assert weights.remove_weight("never-set") is True
    # Sobre archivo existente sin esa key.
    weights.save_weights({"anticipate-echo": 0.7})
    assert weights.remove_weight("never-set") is True
    assert weights.load_weights() == {"anticipate-echo": 0.7}


# ── apply_weight ──────────────────────────────────────────────────────────


def test_apply_weight_default_one_when_kind_unconfigured(tmp_weights):
    # Sin archivo → todo kind devuelve score sin alterar.
    assert weights.apply_weight("anticipate-anything", 0.5) == 0.5
    # Con archivo pero sin la key.
    weights.save_weights({"anticipate-other": 2.0})
    assert weights.apply_weight("anticipate-anything", 0.42) == pytest.approx(0.42)


def test_apply_weight_clamps_to_unit_interval(tmp_weights):
    weights.save_weights({"boost": 5.0, "mute": 0.0})
    # 0.5 * 5.0 = 2.5 → clamp a 1.0
    assert weights.apply_weight("boost", 0.5) == 1.0
    # 0.5 * 0.0 = 0.0 (boundary)
    assert weights.apply_weight("mute", 0.5) == 0.0
    # 0.2 * 1.5 = 0.30 (sin clamp)
    weights.set_weight("normal", 1.5)
    assert weights.apply_weight("normal", 0.2) == pytest.approx(0.30)
    # Score negativo (no debería pasar pero defendemos): clamp inferior 0.
    assert weights.apply_weight("normal", -0.5) == 0.0


# ── list_weights ──────────────────────────────────────────────────────────


def test_list_weights_sorted(tmp_weights):
    weights.save_weights(
        {
            "anticipate-zeta": 0.5,
            "anticipate-alpha": 1.5,
            "anticipate-mike": 1.0,
        }
    )
    out = weights.list_weights()
    assert out == [
        ("anticipate-alpha", 1.5),
        ("anticipate-mike", 1.0),
        ("anticipate-zeta", 0.5),
    ]


def test_list_weights_empty_when_no_file(tmp_weights):
    assert weights.list_weights() == []
