"""Validates that every launchd plist factory generates well-formed XML.

Atrapa el bug histórico del 2026-04-30 donde `_calibration_plist()` tenía
un comentario shell-style (`#`) dentro del `<dict>` que rompía
`plutil -lint` y bloqueaba `launchctl bootstrap`. La factory en código
estaba limpia — el bug fue un edit manual al plist on-disk — pero el
test es belt-and-suspenders contra futuros bugs similares en cualquier
factory.

También atrapó (2026-05-01) que `_implicit_feedback_plist` tenía `&&`
no-escapado en el XML, generando `last_exit≠0` runs=0 silencioso.
"""

from __future__ import annotations

import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

import rag


_HAS_PLUTIL = shutil.which("plutil") is not None
_RAG_BIN = "/tmp/rag"  # rag_bin arg, no se invoca; sólo se interpola

# Labels conocidos del spec manual (T2)
_EXPECTED_MANUAL_LABELS = frozenset({
    "com.fer.obsidian-rag-cloudflare-tunnel",
    "com.fer.obsidian-rag-cloudflare-tunnel-watcher",
    "com.fer.obsidian-rag-lgbm-train",
    "com.fer.obsidian-rag-paraphrases-train",
    "com.fer.obsidian-rag-synth-refresh",
    "com.fer.obsidian-rag-spotify-poll",
    "com.fer.obsidian-rag-log-rotate",
})


def _iter_managed_factories():
    """Yield (label, fname, xml_str) for each entry in _services_spec()."""
    spec = rag._services_spec(_RAG_BIN)
    for item in spec:
        # _services_spec() devuelve tuplas (label, plist_fname, plist_xml_string)
        if isinstance(item, tuple) and len(item) >= 3:
            label, fname, xml_str = item[0], item[1], item[2]
            yield label, fname, xml_str
        elif isinstance(item, dict):
            yield item["label"], item["plist_fname"], item["plist_xml"]


@pytest.mark.skipif(not _HAS_PLUTIL, reason="plutil only available on macOS")
def test_all_plist_factories_valid_plutil(tmp_path: Path):
    """Toda factory de _services_spec() genera XML que pasa plutil -lint."""
    failures = []
    for label, fname, xml_str in _iter_managed_factories():
        path = tmp_path / fname
        path.write_text(xml_str, encoding="utf-8")
        result = subprocess.run(
            ["plutil", "-lint", str(path)],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            failures.append(
                f"{label} ({fname}): plutil -lint exit={result.returncode}\n"
                f"  stderr: {result.stderr.strip()}\n"
                f"  stdout: {result.stdout.strip()}"
            )
    assert not failures, "Factories con XML inválido:\n" + "\n".join(failures)


def test_all_plist_factories_well_formed_xml():
    """xml.etree.ElementTree parsea cada factory sin ParseError."""
    failures = []
    for label, fname, xml_str in _iter_managed_factories():
        try:
            ET.fromstring(xml_str)
        except ET.ParseError as e:
            failures.append(f"{label} ({fname}): {e}")
    assert not failures, "Factories con XML mal formado:\n" + "\n".join(failures)


@pytest.mark.skipif(not _HAS_PLUTIL, reason="plutil only available on macOS")
def test_calibration_plist_no_shell_comment_plutil(tmp_path: Path):
    """Regression test: `_calibration_plist` no debe tener comentarios shell `#`
    sin escapar. Bug original: 2026-04-30, edit manual al plist on-disk."""
    xml_str = rag._calibration_plist(_RAG_BIN)
    # Heuristic: no líneas que contengan `#` y NO empiecen con whitespace + `<!--`
    for i, line in enumerate(xml_str.splitlines(), 1):
        stripped = line.lstrip()
        if "#" in stripped and not stripped.startswith("<!--"):
            # Permitido si está dentro de un `<string>` (uri schemes, etc).
            # Heurística: si la línea contiene `<string>` con `#`, es contenido legítimo.
            if "<string>" in stripped and "</string>" in stripped:
                continue
            pytest.fail(
                f"_calibration_plist linea {i} tiene `#` shell-style sin `<!--`: "
                f"{line!r}"
            )
    # Doble check con plutil sobre el output real.
    path = tmp_path / "calibrate.plist"
    path.write_text(xml_str, encoding="utf-8")
    result = subprocess.run(
        ["plutil", "-lint", str(path)],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0, (
        f"_calibration_plist no pasa plutil -lint: {result.stderr}"
    )


def test_services_spec_manual_shape():
    """`_services_spec_manual()` retorna 7 dicts con shape correcta."""
    spec = rag._services_spec_manual()
    assert isinstance(spec, list), "Debe retornar list"
    assert len(spec) == 7, f"Esperado 7 manuales, got {len(spec)}"
    labels = set()
    for item in spec:
        assert isinstance(item, dict), f"Cada item debe ser dict, got {type(item)}"
        assert set(item.keys()) >= {"label", "category"}, (
            f"Keys mínimas: label, category. Got {item.keys()}"
        )
        assert item["category"] == "manual_keep", (
            f"category={item['category']}, expected 'manual_keep'"
        )
        labels.add(item["label"])
    assert labels == _EXPECTED_MANUAL_LABELS, (
        f"Labels mismatch.\n  expected: {_EXPECTED_MANUAL_LABELS}\n  got: {labels}"
    )


def test_services_spec_manual_no_overlap_with_spec():
    """Ningún label de manual_keep debe estar también en _services_spec()."""
    managed_labels = {label for label, _, _ in _iter_managed_factories()}
    manual_labels = {item["label"] for item in rag._services_spec_manual()}
    overlap = managed_labels & manual_labels
    assert not overlap, f"Labels duplicados managed/manual: {overlap}"
