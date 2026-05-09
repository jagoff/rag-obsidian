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
# 2026-05-04 consolidation: cloudflare-tunnel, cloudflare-tunnel-watcher,
# lgbm-train, paraphrases-train y spotify-poll se removieron del manual
# spec; quedan 2 (synth-refresh + log-rotate).
_EXPECTED_MANUAL_LABELS = frozenset({
    "com.fer.obsidian-rag-synth-refresh",
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
    """`_services_spec_manual()` retorna 2 dicts con shape correcta.

    Post limpieza 2026-05-04: synth-refresh + log-rotate son los únicos
    manuales sobrevivientes (los 4 fantasmas + spotify-poll se removieron).
    """
    spec = rag._services_spec_manual()
    assert isinstance(spec, list), "Debe retornar list"
    assert len(spec) == 2, f"Esperado 2 manuales, got {len(spec)}"
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


def test_minimal_managed_labels_subset_of_spec():
    """`_MINIMAL_MANAGED_LABELS` debe ser subset de los labels de `_services_spec()`.

    Si alguien remueve un daemon del spec sin actualizar `_MINIMAL_MANAGED_LABELS`,
    `rag start --minimal` (default) intentaría instalar un label que ya no existe.
    """
    managed_labels = {label for label, _, _ in _iter_managed_factories()}
    minimal = rag._MINIMAL_MANAGED_LABELS
    assert minimal <= managed_labels, (
        f"_MINIMAL_MANAGED_LABELS tiene labels fuera del spec: "
        f"{minimal - managed_labels}"
    )
    # Mínimo viable per acuerdo 2026-05-08 con el user (rag start clean):
    # watch + web + daemon-watchdog + wake-hook + maintenance.
    assert len(minimal) == 5, f"Esperado 5 minimal labels, got {len(minimal)}"


def test_deprecated_plist_functions_are_gone():
    """Las 9 funciones deprecadas (serve + 7 ingesters consolidados) NO existen.

    Si alguien las re-introduce sin re-introducir el daemon en
    `_services_spec`, este test las detecta — re-creando solo la factory sin
    el entry en el spec deja el plist huérfano (no se instala, pero los
    tests verdes te hacen creer que sí). Los labels siguen en
    `_DEPRECATED_LABELS` para que `rag setup` haga el bootout en disco.

    Borradas en Fase 2a (2026-05-09):
      - `_serve_plist`, `_serve_watchdog_plist` (rag serve replaced by FastAPI web)
      - `_ingest_{gmail,calendar,reminders,calls,safari,drive,pillow}_plist`
        (consolidadas en ingest-cross-source desde 2026-05-04)
    """
    from rag import plists
    for name in [
        "_serve_plist",
        "_serve_watchdog_plist",
        "_ingest_gmail_plist",
        "_ingest_calendar_plist",
        "_ingest_reminders_plist",
        "_ingest_calls_plist",
        "_ingest_safari_plist",
        "_ingest_drive_plist",
        "_ingest_pillow_plist",
    ]:
        assert not hasattr(plists, name), (
            f"{name} fue removida en Fase 2a (2026-05-09) — si la "
            f"necesitás de nuevo, agregala TAMBIÉN a _services_spec()."
        )


def test_setup_install_only_labels_filter(tmp_path, monkeypatch):
    """`_setup_install(only_labels=X)` instala solo labels en X y skipea el resto.

    Mockea `_LAUNCH_AGENTS_DIR` a tmp_path y `subprocess.run` para evitar
    `launchctl load` real. Verifica que post-install el directorio tiene
    SOLO los plists del filtro.
    """
    import subprocess as sp

    monkeypatch.setattr(rag, "_LAUNCH_AGENTS_DIR", tmp_path)
    monkeypatch.setattr(rag, "_RAG_LOG_DIR", tmp_path / "logs")
    # `launchctl load`/`unload` mock: siempre retorna éxito sin tocar el sistema.
    monkeypatch.setattr(
        sp, "run",
        lambda *a, **kw: sp.CompletedProcess(args=a, returncode=0, stdout=b"", stderr=b""),
    )

    only = frozenset({
        "com.fer.obsidian-rag-watch",
        "com.fer.obsidian-rag-web",
    })
    rag._setup_install(_RAG_BIN, remove=False, only_labels=only)

    installed = {p.stem for p in tmp_path.glob("*.plist")}
    assert installed == only, f"Esperado solo {only}, got {installed}"


# ── Resource-budget audit (2026-05-09) ─────────────────────────────────────
#
# Garantizan que la ronda de optimización mantiene los defaults:
#   - Batch nocturnos: ProcessType=Background + LowPriorityIO=true.
#   - daemon-watchdog: ExitTimeOut presente (evita hang en SQL locked).
#   - Frecuencias largas: anticipate ≥ 900s, spotify-poll ≥ 300s.
#   - Stagger: calibrate corre 05:00 (no 04:30, evita overlap online-tune).
#   - HF_HUB_OFFLINE=1 explícito en plists que cargan modelos MLX.
#
# Si se borra una de estas keys por accidente, este test grita.


_BATCH_NIGHTLY_LABELS = frozenset({
    "com.fer.obsidian-rag-auto-harvest",
    "com.fer.obsidian-rag-online-tune",
    "com.fer.obsidian-rag-calibrate",
    "com.fer.obsidian-rag-implicit-feedback",
    "com.fer.obsidian-rag-whisper-vocab",
    "com.fer.obsidian-rag-drift-watcher",
    "com.fer.obsidian-rag-maintenance",
    "com.fer.obsidian-rag-vault-cleanup",
    "com.fer.obsidian-rag-consolidate",
    "com.fer.obsidian-rag-archive",
    "com.fer.obsidian-rag-distill",
    "com.fer.obsidian-rag-emergent",
    "com.fer.obsidian-rag-patterns",
    "com.fer.obsidian-rag-active-learning-nudge",
    "com.fer.obsidian-rag-active-learning-suggest-goldens",
    "com.fer.obsidian-rag-brief-auto-tune",
    "com.fer.obsidian-rag-wake-up",
    "com.fer.obsidian-rag-routing-rules",
    "com.fer.obsidian-rag-ingest-cross-source",
})


def _xml_for(label: str) -> str:
    for lbl, _fname, xml_str in _iter_managed_factories():
        if lbl == label:
            return xml_str
    raise AssertionError(f"Label {label} no está en _services_spec()")


def test_batch_nightly_have_background_and_low_priority_io():
    """Daemons batch nocturnos deben tener ProcessType=Background +
    LowPriorityIO=true para no pisar chat/web del user en horarios
    no programados (post-wake, user despierto temprano, etc)."""
    failures = []
    for label in _BATCH_NIGHTLY_LABELS:
        xml_str = _xml_for(label)
        if "<key>ProcessType</key><string>Background</string>" not in xml_str:
            failures.append(f"{label}: falta ProcessType=Background")
        if "<key>LowPriorityIO</key><true/>" not in xml_str:
            failures.append(f"{label}: falta LowPriorityIO=true")
    assert not failures, "\n".join(failures)


def test_daemon_watchdog_has_exit_timeout():
    """daemon-watchdog corre cada 5min. ExitTimeOut=10s evita hang
    si SQL queda locked durante reconcile."""
    xml_str = _xml_for("com.fer.obsidian-rag-daemon-watchdog")
    assert "<key>ExitTimeOut</key><integer>10</integer>" in xml_str
    assert "<key>ProcessType</key><string>Background</string>" in xml_str


def test_anticipate_cadence_is_15min():
    """anticipate baja 10min → 15min (audit 2026-05-09): daily_cap=3
    hace que pollear más seguido no compre coverage."""
    xml_str = _xml_for("com.fer.obsidian-rag-anticipate")
    assert "<key>StartInterval</key><integer>900</integer>" in xml_str
    assert "<key>StartInterval</key><integer>600</integer>" not in xml_str


def test_spotify_poll_cadence_is_5min():
    """spotify-poll baja 60s → 300s (audit 2026-05-09): track granular
    no aporta a briefs ni mood scoring; -83% spawn overhead."""
    xml_str = _xml_for("com.fer.obsidian-rag-spotify-poll")
    assert "<key>StartInterval</key><integer>300</integer>" in xml_str
    assert "<key>StartInterval</key><integer>60</integer>" not in xml_str


def test_calibrate_staggered_to_5am():
    """calibrate 04:30 → 05:00 (audit 2026-05-09): online-tune dura 24min
    en mac M-chip, 04:30 se solapaba al final del tune."""
    xml_str = _xml_for("com.fer.obsidian-rag-calibrate")
    assert "<key>Hour</key><integer>5</integer>" in xml_str
    sched = xml_str.split("StartCalendarInterval")[1].split("</dict>")[0]
    assert "<integer>30</integer>" not in sched


def test_frequent_workers_have_throttle():
    """Plists con StartInterval ≤ 5min deben tener ThrottleInterval
    para evitar spawn loops bajo backoff."""
    targets = {
        "com.fer.obsidian-rag-routing-rules",
        "com.fer.obsidian-rag-wa-fast",
    }
    failures = []
    for label in targets:
        xml_str = _xml_for(label)
        if "ThrottleInterval" not in xml_str and "<key>Throttle</key>" not in xml_str:
            failures.append(f"{label}: falta Throttle/ThrottleInterval")
    assert not failures, "\n".join(failures)


def test_mlx_plists_have_offline_flags():
    """Plists con RAG_LLM_BACKEND=mlx deben tener HF_HUB_OFFLINE=1 +
    TRANSFORMERS_OFFLINE=1 para que cold-start post-sleep no se cuelgue
    si la red está caída + ahorrar HEAD requests a HuggingFace."""
    failures = []
    for label, _fname, xml_str in _iter_managed_factories():
        if "<key>RAG_LLM_BACKEND</key><string>mlx</string>" not in xml_str:
            continue
        if "<key>HF_HUB_OFFLINE</key><string>1</string>" not in xml_str:
            failures.append(f"{label}: tiene RAG_LLM_BACKEND=mlx pero falta HF_HUB_OFFLINE=1")
        if "<key>TRANSFORMERS_OFFLINE</key><string>1</string>" not in xml_str:
            failures.append(f"{label}: tiene RAG_LLM_BACKEND=mlx pero falta TRANSFORMERS_OFFLINE=1")
    assert not failures, "\n".join(failures)
