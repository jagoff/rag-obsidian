"""Tests para `_load_cross_source_filters` no-cache-on-fail (H-7 fix, 2026-05-08).

Pre-fix: una falla transitoria leyendo / parseando el YAML
(`cross-source.yaml`) memoizaba `{}` en `_CROSS_SOURCE_FILTERS_CACHE`,
deshabilitando *todas* las reglas de privacidad cross-source hasta el
próximo restart. Resultado real: un YAML malformado durante un edit del
usuario podía dejar la sesión sin filtros para mails sensibles, chats
privados, etc.

Post-fix: la rama de error loguea via `_silent_log` y devuelve `{}`
SIN tocar el cache. El próximo call reintenta el read del archivo —
si el YAML fue corregido, levanta los filtros normalmente.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import rag


@pytest.fixture(autouse=True)
def _reset_filter_cache(monkeypatch, tmp_path):
    """Apunta DB_PATH al tmp_path y resetea el cache entre tests."""
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    monkeypatch.setattr(rag, "_CROSS_SOURCE_FILTERS_CACHE", None)
    monkeypatch.setattr(rag, "_CROSS_SOURCE_FILTERS_MTIME", 0.0)
    yield


def _write(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "cross-source.yaml"
    p.write_text(content, encoding="utf-8")
    return p


def test_yaml_parse_fail_does_not_memoize_empty_dict(tmp_path):
    """YAML inválido → primer call retorna {}, segundo call tras fix lee bien."""
    # 1. Escribimos YAML malformado.
    _write(tmp_path, "gmail:\n  exclude_labels: [unbalanced")
    first = rag._load_cross_source_filters()
    assert first == {}, "YAML inválido debería retornar {} (silent-fail)"
    # El cache NO se debe haber poblado con `{}`.
    assert rag._CROSS_SOURCE_FILTERS_CACHE is None, (
        "post-fix: el cache NO se memoiza en falla. Pre-fix: quedaba '{}' "
        "y el siguiente call ignoraba YAML válidos hasta restart."
    )

    # 2. El user corrige el YAML.
    _write(tmp_path, "gmail:\n  exclude_labels: [banking, 2fa]\n")

    # 3. Segundo call debe leer el YAML válido (no servir el cache vacío).
    second = rag._load_cross_source_filters()
    assert second == {"gmail": {"exclude_labels": ["banking", "2fa"]}}, (
        "post-fix: el segundo call retry el read y levanta los filtros corregidos."
    )
    # Ahora sí, el cache está poblado con el dict válido.
    assert rag._CROSS_SOURCE_FILTERS_CACHE == second


def test_success_path_still_caches(tmp_path):
    """Sanity: el success path sigue cacheando para ahorrar reads."""
    _write(tmp_path, "calendar:\n  exclude_calendars: [Personal]\n")
    rag._load_cross_source_filters()
    assert rag._CROSS_SOURCE_FILTERS_CACHE == {
        "calendar": {"exclude_calendars": ["Personal"]}
    }
