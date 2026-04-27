#!/usr/bin/env python3
"""Generate an Excalidraw diagram of the /api/chat flow.

Output: .excalidraw JSON (open with excalidraw.com or the Obsidian plugin).

Usage:
    .venv/bin/python scripts/gen_chat_flow_excalidraw.py > out.excalidraw

Layout strategy: top-down flowchart with swim-lanes. Color-coded by stage type
(entry / decision / short-circuit / cache / retrieve / LLM / post-process /
risk). Used to live alongside flow.md and audit.md in the vault under
99-AI/system/query-flow-audit-2026-04-27/.
"""

from __future__ import annotations
import json
import secrets
import time
import sys


def now_ms() -> int:
    return int(time.time() * 1000)


def rid() -> str:
    return secrets.token_hex(12)


def seed() -> int:
    return secrets.randbelow(2**31)


# Color palette — semantic per stage
COLORS = {
    # entry: blue
    "entry":      ("#1e3a8a", "#dbeafe"),
    # process / generic step: gray
    "process":    ("#374151", "#f3f4f6"),
    # decision diamond: amber
    "decision":   ("#92400e", "#fef3c7"),
    # short-circuit early return: green
    "shortcut":   ("#166534", "#dcfce7"),
    # cache: cyan
    "cache":      ("#155e75", "#cffafe"),
    # retrieve / heavy: pink-magenta
    "retrieve":   ("#9d174d", "#fce7f3"),
    # LLM: violet
    "llm":        ("#5b21b6", "#ede9fe"),
    # post-process: orange
    "post":       ("#9a3412", "#ffedd5"),
    # risk / bug: red dashed
    "risk":       ("#991b1b", "#fee2e2"),
    # final: slate
    "final":      ("#1e293b", "#e2e8f0"),
}


def make_box(x: int, y: int, w: int, h: int, text: str, kind: str,
             *, dashed: bool = False, shape: str = "rectangle",
             font_size: int = 16) -> tuple[list[dict], str]:
    """Create a box with centered text. Returns (elements, container_id)."""
    stroke, bg = COLORS[kind]
    box_id = rid()
    txt_id = rid()
    box = {
        "id": box_id,
        "type": shape,
        "x": x, "y": y, "width": w, "height": h,
        "angle": 0,
        "strokeColor": stroke,
        "backgroundColor": bg,
        "fillStyle": "solid",
        "strokeWidth": 2,
        "strokeStyle": "dashed" if dashed else "solid",
        "roughness": 1,
        "opacity": 100,
        "groupIds": [],
        "frameId": None,
        "roundness": {"type": 3} if shape == "rectangle" else None,
        "seed": seed(),
        "versionNonce": seed(),
        "isDeleted": False,
        "boundElements": [{"id": txt_id, "type": "text"}],
        "updated": now_ms(),
        "link": None,
        "locked": False,
    }
    txt = {
        "id": txt_id,
        "type": "text",
        "x": x, "y": y + h / 2 - font_size / 2,
        "width": w, "height": font_size * 1.25,
        "angle": 0,
        "strokeColor": stroke,
        "backgroundColor": "transparent",
        "fillStyle": "solid",
        "strokeWidth": 1,
        "strokeStyle": "solid",
        "roughness": 1,
        "opacity": 100,
        "groupIds": [],
        "frameId": None,
        "roundness": None,
        "seed": seed(),
        "versionNonce": seed(),
        "isDeleted": False,
        "boundElements": None,
        "updated": now_ms(),
        "link": None,
        "locked": False,
        "fontSize": font_size,
        "fontFamily": 1,  # Virgil — the iconic sketchy hand
        "text": text,
        "textAlign": "center",
        "verticalAlign": "middle",
        "containerId": box_id,
        "originalText": text,
        "autoResize": True,
        "lineHeight": 1.2,
    }
    return [box, txt], box_id


def make_arrow(from_id: str, to_id: str, *,
               from_x: float, from_y: float, to_x: float, to_y: float,
               label: str | None = None,
               dashed: bool = False,
               color: str = "#1e293b") -> list[dict]:
    """Arrow from (from_x, from_y) to (to_x, to_y) with bindings to from_id/to_id."""
    arr_id = rid()
    dx = to_x - from_x
    dy = to_y - from_y
    arr = {
        "id": arr_id,
        "type": "arrow",
        "x": from_x, "y": from_y,
        "width": abs(dx), "height": abs(dy),
        "angle": 0,
        "strokeColor": color,
        "backgroundColor": "transparent",
        "fillStyle": "solid",
        "strokeWidth": 2,
        "strokeStyle": "dashed" if dashed else "solid",
        "roughness": 1,
        "opacity": 100,
        "groupIds": [],
        "frameId": None,
        "roundness": {"type": 2},
        "seed": seed(),
        "versionNonce": seed(),
        "isDeleted": False,
        "boundElements": None,
        "updated": now_ms(),
        "link": None,
        "locked": False,
        "points": [[0.0, 0.0], [dx, dy]],
        "lastCommittedPoint": None,
        "startBinding": {"elementId": from_id, "focus": 0, "gap": 4},
        "endBinding":   {"elementId": to_id,   "focus": 0, "gap": 4},
        "startArrowhead": None,
        "endArrowhead": "arrow",
        "elbowed": False,
    }
    elements = [arr]
    if label:
        # Standalone label text near the arrow midpoint
        mx, my = (from_x + to_x) / 2, (from_y + to_y) / 2
        lbl_id = rid()
        lbl = {
            "id": lbl_id,
            "type": "text",
            "x": mx + 6, "y": my - 10,
            "width": max(60, len(label) * 9),
            "height": 18,
            "angle": 0,
            "strokeColor": color,
            "backgroundColor": "#ffffff",
            "fillStyle": "solid",
            "strokeWidth": 1,
            "strokeStyle": "solid",
            "roughness": 1,
            "opacity": 100,
            "groupIds": [],
            "frameId": None,
            "roundness": None,
            "seed": seed(),
            "versionNonce": seed(),
            "isDeleted": False,
            "boundElements": None,
            "updated": now_ms(),
            "link": None,
            "locked": False,
            "fontSize": 14,
            "fontFamily": 1,
            "text": label,
            "textAlign": "left",
            "verticalAlign": "top",
            "containerId": None,
            "originalText": label,
            "autoResize": True,
            "lineHeight": 1.2,
        }
        elements.append(lbl)
    return elements


def make_section_header(x: int, y: int, w: int, text: str,
                        color: str = "#1e293b") -> list[dict]:
    """Big section title above a swim-lane."""
    txt_id = rid()
    return [{
        "id": txt_id,
        "type": "text",
        "x": x, "y": y, "width": w, "height": 32,
        "angle": 0,
        "strokeColor": color,
        "backgroundColor": "transparent",
        "fillStyle": "solid",
        "strokeWidth": 1,
        "strokeStyle": "solid",
        "roughness": 1,
        "opacity": 100,
        "groupIds": [],
        "frameId": None,
        "roundness": None,
        "seed": seed(),
        "versionNonce": seed(),
        "isDeleted": False,
        "boundElements": None,
        "updated": now_ms(),
        "link": None,
        "locked": False,
        "fontSize": 24,
        "fontFamily": 1,
        "text": text,
        "textAlign": "left",
        "verticalAlign": "top",
        "containerId": None,
        "originalText": text,
        "autoResize": True,
        "lineHeight": 1.2,
    }]


def build() -> dict:
    elements: list[dict] = []

    # Layout grid
    cx_center = 800   # main column x-center
    box_w = 320
    box_w_wide = 480
    box_h = 70
    box_h_tall = 100

    def hcenter(x_center: int, w: int) -> int:
        return x_center - w // 2

    # ── Row 0: Title ──
    elements += make_section_header(
        hcenter(cx_center, 700), 30, 700,
        "POST /api/chat — flujo end-to-end",
        color="#0f172a",
    )

    # Row 1: Entry (POST /api/chat)
    e, entry_id = make_box(
        hcenter(cx_center, box_w_wide), 90, box_w_wide, box_h_tall,
        "POST /api/chat\n{ question, session_id?, redo_turn_id?, vault_scope? }",
        "entry", font_size=18,
    )
    elements += e

    # Row 2: rate-limit + device
    e, rate_id = make_box(
        hcenter(cx_center, box_w_wide), 230, box_w_wide, box_h,
        "rate-limit 30/60s · classify_device",
        "process",
    )
    elements += e
    elements += make_arrow(entry_id, rate_id,
                           from_x=cx_center, from_y=90 + box_h_tall,
                           to_x=cx_center, to_y=230)

    # Row 3: redo decision (diamond)
    redo_w, redo_h = 240, 100
    e, redo_id = make_box(
        hcenter(cx_center, redo_w), 350, redo_w, redo_h,
        "redo_turn_id?", "decision", shape="diamond",
    )
    elements += e
    elements += make_arrow(rate_id, redo_id,
                           from_x=cx_center, from_y=230 + box_h,
                           to_x=cx_center, to_y=350)

    # Row 3b: redo branch (left = sí, right = no)
    e, redo_yes_id = make_box(
        100, 480, 380, box_h,
        "_resolve_redo_question\nquestion = orig + ' — enfocá en: hint'",
        "process", font_size=14,
    )
    elements += e
    e, redo_no_id = make_box(
        cx_center + 200, 480, 280, box_h,
        "question = req.question.strip()",
        "process", font_size=14,
    )
    elements += e
    elements += make_arrow(redo_id, redo_yes_id,
                           from_x=cx_center - redo_w / 2, from_y=400,
                           to_x=290, to_y=480, label="sí")
    elements += make_arrow(redo_id, redo_no_id,
                           from_x=cx_center + redo_w / 2, from_y=400,
                           to_x=cx_center + 340, to_y=480, label="no")

    # Row 4: detect intents (merge point)
    e, detect_id = make_box(
        hcenter(cx_center, 600), 600, 600, 80,
        "detectar:  is_propose_intent · is_metachat · is_degenerate · is_finance",
        "process", font_size=15,
    )
    elements += e
    elements += make_arrow(redo_yes_id, detect_id,
                           from_x=290, from_y=480 + box_h,
                           to_x=cx_center - 100, to_y=600)
    elements += make_arrow(redo_no_id, detect_id,
                           from_x=cx_center + 340, from_y=480 + box_h,
                           to_x=cx_center + 100, to_y=600)

    # ── SHORT-CIRCUITS lane (4 columns side-by-side) ──
    elements += make_section_header(
        80, 720, 700, "Short-circuits (early returns)",
        color="#166534",
    )
    sc_y = 770
    sc_w = 220
    sc_h = 100
    sc_gap = 30
    sc_total = 4 * sc_w + 3 * sc_gap
    sc_x0 = (1600 - sc_total) // 2

    sc_specs = [
        ("degenerate\n(✗ <2 chars alfa)\n→ canned reply", "shortcut"),
        ("finance/cards\n(MOZE/banco)\n→ render det. <1s", "shortcut"),
        ("metachat\n(saludos/thanks)\n→ canned reply", "shortcut"),
        ("propose-intent\n(crear remind/cal)\n→ Apple Tools", "shortcut"),
    ]
    sc_ids = []
    for i, (text, kind) in enumerate(sc_specs):
        sx = sc_x0 + i * (sc_w + sc_gap)
        e, sid = make_box(sx, sc_y, sc_w, sc_h, text, kind, font_size=13)
        elements += e
        sc_ids.append((sid, sx + sc_w // 2))
        elements += make_arrow(detect_id, sid,
                               from_x=cx_center, from_y=600 + 80,
                               to_x=sx + sc_w // 2, to_y=sc_y)

    # Each short-circuit ends at "return" terminal
    for sid, sx_center in sc_ids:
        e, ret_id = make_box(
            sx_center - 60, sc_y + sc_h + 50, 120, 50,
            "return", "final", font_size=14,
        )
        elements += e
        elements += make_arrow(sid, ret_id,
                               from_x=sx_center, from_y=sc_y + sc_h,
                               to_x=sx_center, to_y=sc_y + sc_h + 50)

    # ── Main flow continues (none of the SC fired) ──
    # Row: ollama probe
    elements += make_section_header(
        hcenter(cx_center, 600), 1020, 600,
        "Si NINGÚN short-circuit aplica:",
        color="#0f172a",
    )

    e, probe_id = make_box(
        hcenter(cx_center, 420), 1080, 420, box_h,
        "_ollama_chat_probe + _ollama_restart_if_stuck",
        "process", font_size=14,
    )
    elements += e
    elements += make_arrow(detect_id, probe_id,
                           from_x=cx_center, from_y=600 + 80,
                           to_x=cx_center, to_y=1080,
                           label="todos no")

    # Row: cache lookup decision
    e, cache_dec_id = make_box(
        hcenter(cx_center, 280), 1190, 280, 100,
        "cache elegible?\n(no si: history,\nmulti-vault, ...)",
        "decision", shape="diamond", font_size=14,
    )
    elements += e
    elements += make_arrow(probe_id, cache_dec_id,
                           from_x=cx_center, from_y=1080 + box_h,
                           to_x=cx_center, to_y=1190)

    # Cache hit branch
    e, cache_hit_id = make_box(
        180, 1330, 280, box_h,
        "semantic_cache_lookup\n→ HIT → done(cache)",
        "cache", font_size=13,
    )
    elements += e
    elements += make_arrow(cache_dec_id, cache_hit_id,
                           from_x=cx_center - 140, from_y=1240,
                           to_x=320, to_y=1330, label="sí + hit")

    e, cache_ret_id = make_box(
        260, 1450, 120, 50, "return", "final", font_size=14,
    )
    elements += e
    elements += make_arrow(cache_hit_id, cache_ret_id,
                           from_x=320, from_y=1330 + box_h,
                           to_x=320, to_y=1450)

    # Pre-router (forced tools)
    e, prerouter_id = make_box(
        cx_center + 60, 1330, 380, box_h,
        "pre-router: _detect_tool_intent\n→ forced tools (gmail/cal/wa/reminders)",
        "process", font_size=13,
    )
    elements += e
    elements += make_arrow(cache_dec_id, prerouter_id,
                           from_x=cx_center + 140, from_y=1240,
                           to_x=cx_center + 250, to_y=1330,
                           label="no / miss")

    # ── multi_retrieve (heavy stage) ──
    elements += make_section_header(
        hcenter(cx_center, 700), 1450, 700,
        "Retrieval pipeline (la pieza pesada)",
        color="#9d174d",
    )

    e, retrieve_id = make_box(
        hcenter(cx_center, 720), 1500, 720, 220,
        "multi_retrieve(vaults × sources)\n\n"
        "1. embed query (local MPS o ollama)\n"
        "2. BM25 + sqlite-vec semantic\n"
        "3. RRF merge\n"
        "4. cross-encoder rerank (bge-reranker-v2-m3)\n"
        "5. scoring: recency · tags · PageRank · feedback · source-weight\n"
        "6. dedup (privacy + conv-window + cross-source)\n"
        "7. graph-expansion (1-2 hop wikilinks)\n"
        "8. fast-path detection",
        "retrieve", font_size=14,
    )
    elements += e
    elements += make_arrow(prerouter_id, retrieve_id,
                           from_x=cx_center + 250, from_y=1330 + box_h,
                           to_x=cx_center, to_y=1500)

    # Confidence gate decision
    e, conf_id = make_box(
        hcenter(cx_center, 320), 1760, 320, 110,
        "confidence < gate(source)?",
        "decision", shape="diamond", font_size=15,
    )
    elements += e
    elements += make_arrow(retrieve_id, conf_id,
                           from_x=cx_center, from_y=1500 + 220,
                           to_x=cx_center, to_y=1760)

    # Low-conf branch (left)
    e, lowconf_id = make_box(
        180, 1920, 320, 90,
        "low-conf gate\n→ 'No tengo info, /capture'\n→ done(gated)",
        "shortcut", font_size=13,
    )
    elements += e
    elements += make_arrow(conf_id, lowconf_id,
                           from_x=cx_center - 160, from_y=1815,
                           to_x=340, to_y=1920, label="sí")

    e, lowconf_ret = make_box(
        280, 2050, 120, 50, "return", "final", font_size=14,
    )
    elements += e
    elements += make_arrow(lowconf_id, lowconf_ret,
                           from_x=340, from_y=1920 + 90,
                           to_x=340, to_y=2050)

    # ── LLM streaming ──
    elements += make_section_header(
        hcenter(cx_center, 600), 1920, 600,
        "LLM streaming + post-process",
        color="#5b21b6",
    )

    e, sources_id = make_box(
        hcenter(cx_center, 380), 1980, 380, box_h,
        "yield sources(items, confidence, intent)\nyield status(stage=generating)",
        "llm", font_size=13,
    )
    elements += e
    elements += make_arrow(conf_id, sources_id,
                           from_x=cx_center + 160, from_y=1815,
                           to_x=cx_center, to_y=1980, label="no")

    e, ollama_id = make_box(
        hcenter(cx_center, 400), 2090, 400, 100,
        "ollama.chat(stream=True)\nqwen2.5:7b o LOOKUP_MODEL si fast_path\n→ yield token(delta)",
        "llm", font_size=13,
    )
    elements += e
    elements += make_arrow(sources_id, ollama_id,
                           from_x=cx_center, from_y=1980 + box_h,
                           to_x=cx_center, to_y=2090)

    # ── Parallel post-process ──
    e, enrich_id = make_box(
        cx_center - 420, 2260, 380, 130,
        "_emit_enrich (4s budget)\nThreadPoolExecutor +\nshutdown(wait=False, cancel_futures=True)\n\n✓ tiene timeout",
        "post", font_size=13,
    )
    elements += e
    elements += make_arrow(ollama_id, enrich_id,
                           from_x=cx_center - 60, from_y=2090 + 100,
                           to_x=cx_center - 230, to_y=2260)

    # GROUNDING — RISK box
    e, ground_id = make_box(
        cx_center + 40, 2260, 380, 130,
        "⚠ _emit_grounding\nground_claims_nli SÍNCRONO\nSIN TIMEOUT\n\n→ riesgo: PWA congelada\nsi NLI helper se cuelga",
        "risk", dashed=True, font_size=13,
    )
    elements += e
    elements += make_arrow(ollama_id, ground_id,
                           from_x=cx_center + 60, from_y=2090 + 100,
                           to_x=cx_center + 230, to_y=2260,
                           color="#991b1b")

    # ── Done + writeback ──
    e, done_id = make_box(
        hcenter(cx_center, 480), 2440, 480, box_h,
        "yield done(turn_id, elapsed_ms)",
        "final", font_size=15,
    )
    elements += e
    elements += make_arrow(enrich_id, done_id,
                           from_x=cx_center - 230, from_y=2260 + 130,
                           to_x=cx_center - 100, to_y=2440)
    elements += make_arrow(ground_id, done_id,
                           from_x=cx_center + 230, from_y=2260 + 130,
                           to_x=cx_center + 100, to_y=2440)

    e, write_id = make_box(
        hcenter(cx_center, 480), 2550, 480, box_h,
        "append_turn · save_session · log_query_event(cmd=web)\n+ spawn conversation_writer (background)",
        "process", font_size=13,
    )
    elements += e
    elements += make_arrow(done_id, write_id,
                           from_x=cx_center, from_y=2440 + box_h,
                           to_x=cx_center, to_y=2550)

    # ── Legend ──
    elements += make_section_header(80, 2700, 200, "Leyenda", color="#0f172a")
    legend_y = 2750
    legend_items = [
        ("entrada / final", "entry"),
        ("proceso", "process"),
        ("decisión", "decision"),
        ("short-circuit", "shortcut"),
        ("cache", "cache"),
        ("retrieve heavy", "retrieve"),
        ("LLM", "llm"),
        ("post-process", "post"),
        ("⚠ riesgo / bug", "risk"),
    ]
    for i, (label, kind) in enumerate(legend_items):
        e, _ = make_box(
            80 + (i % 3) * 240, legend_y + (i // 3) * 60, 220, 45,
            label, kind, font_size=12,
            dashed=(kind == "risk"),
        )
        elements += e

    # Annotations / callouts
    callout_y = 2960
    elements += make_section_header(80, callout_y, 800,
                                     "Bugs encontrados (ver audit.md)",
                                     color="#991b1b")
    elements += make_section_header(80, callout_y + 50, 1400,
        "🔴 #1 _emit_grounding sin timeout — copiar patrón de _emit_enrich",
        color="#991b1b")
    elements += make_section_header(80, callout_y + 90, 1400,
        "🔴 #2 Pre-router sin try/except envolvente — cliente cuelga",
        color="#991b1b")
    elements += make_section_header(80, callout_y + 130, 1400,
        "🟡 #3 /redo + hint nunca cachea (UX)",
        color="#92400e")
    elements += make_section_header(80, callout_y + 170, 1400,
        "🟡 #4 Logging gaps en error paths (observabilidad)",
        color="#92400e")

    # Wrap
    return {
        "type": "excalidraw",
        "version": 2,
        "source": "https://excalidraw.com",
        "elements": elements,
        "appState": {
            "gridSize": None,
            "viewBackgroundColor": "#fafaf9",
        },
        "files": {},
    }


def main() -> None:
    doc = build()
    json.dump(doc, sys.stdout, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
