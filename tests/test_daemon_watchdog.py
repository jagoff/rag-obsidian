"""Tests para la factory _daemon_watchdog_plist (T4, 2026-05-01).

Self-healing loop que corre `rag daemons reconcile --apply --gentle` cada
5 minutos.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import rag


_RAG_BIN = "/tmp/rag"


def test_daemon_watchdog_plist_generates_xml():
    """Factory genera un string XML válido."""
    xml_str = rag._daemon_watchdog_plist(_RAG_BIN)
    assert isinstance(xml_str, str)
    assert xml_str.startswith('<?xml')
    assert '<plist' in xml_str
    assert '</plist>' in xml_str


def test_daemon_watchdog_plist_well_formed():
    """El XML es válido y parseble por ElementTree."""
    xml_str = rag._daemon_watchdog_plist(_RAG_BIN)
    root = ET.fromstring(xml_str)
    assert root.tag == "plist"
    assert root.get("version") == "1.0"


def test_daemon_watchdog_plist_has_correct_label():
    """El plist tiene el label correcto."""
    xml_str = rag._daemon_watchdog_plist(_RAG_BIN)
    root = ET.fromstring(xml_str)
    # La estructura es <plist><dict> con <key>Label</key><string>...</string>
    dict_elem = root.find("dict")
    assert dict_elem is not None

    # Buscar la key "Label" y su valor
    label_found = False
    for i, elem in enumerate(dict_elem):
        if elem.tag == "key" and elem.text == "Label":
            # El siguiente elemento debe ser un <string>
            next_elem = dict_elem[i + 1] if i + 1 < len(dict_elem) else None
            assert next_elem is not None and next_elem.tag == "string"
            assert next_elem.text == "com.fer.obsidian-rag-daemon-watchdog"
            label_found = True
            break
    assert label_found, "Label not found or malformed"


def test_daemon_watchdog_plist_has_start_interval_300():
    """El plist tiene StartInterval = 300 (5 minutos)."""
    xml_str = rag._daemon_watchdog_plist(_RAG_BIN)
    root = ET.fromstring(xml_str)
    dict_elem = root.find("dict")
    assert dict_elem is not None

    # Buscar StartInterval
    interval_found = False
    for i, elem in enumerate(dict_elem):
        if elem.tag == "key" and elem.text == "StartInterval":
            next_elem = dict_elem[i + 1] if i + 1 < len(dict_elem) else None
            assert next_elem is not None and next_elem.tag == "integer"
            assert next_elem.text == "300"
            interval_found = True
            break
    assert interval_found, "StartInterval not found"


def test_daemon_watchdog_plist_in_services_spec():
    """El daemon-watchdog aparece en _services_spec()."""
    specs = rag._services_spec(_RAG_BIN)
    labels = [item[0] for item in specs]
    assert "com.fer.obsidian-rag-daemon-watchdog" in labels, \
        f"daemon-watchdog not in {labels}"


def test_daemon_watchdog_plist_has_program_arguments():
    """El plist tiene ProgramArguments con el subcomando correcto."""
    xml_str = rag._daemon_watchdog_plist(_RAG_BIN)
    root = ET.fromstring(xml_str)
    dict_elem = root.find("dict")

    # Buscar ProgramArguments array
    prog_args_array = None
    for i, elem in enumerate(dict_elem):
        if elem.tag == "key" and elem.text == "ProgramArguments":
            next_elem = dict_elem[i + 1] if i + 1 < len(dict_elem) else None
            if next_elem is not None and next_elem.tag == "array":
                prog_args_array = next_elem
            break

    assert prog_args_array is not None, "ProgramArguments array not found"

    # El array debe tener strings: rag_bin, "daemons", "reconcile", "--apply", "--gentle"
    strings = [elem.text for elem in prog_args_array if elem.tag == "string"]
    assert len(strings) >= 5, f"Expected ≥5 args, got {strings}"
    assert strings[0] == _RAG_BIN
    assert strings[1] == "daemons"
    assert strings[2] == "reconcile"
    assert "--apply" in strings
    assert "--gentle" in strings
