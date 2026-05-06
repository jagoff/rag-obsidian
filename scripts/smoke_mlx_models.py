"""Smoke test: cargar y ejecutar los 4 modelos MLX via MLXBackend.

Verifica:
- Load (model + tokenizer)
- chat() devuelve ollama-shape ChatResponse (atributo .message.content)
- Output no contiene drift (PT/CJK)
- HQ tier (Qwen3-30B) responde JSON cuando format='json'

Uso:
    RAG_LLM_BACKEND=mlx .venv/bin/python scripts/smoke_mlx_models.py
    .venv/bin/python scripts/smoke_mlx_models.py --skip-big   # saltea el 30B
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time

os.environ.setdefault("RAG_LLM_BACKEND", "mlx")

from rag.llm_backend import ChatOptions, MLXBackend, MLX_MODEL_ALIAS  # noqa: E402

DRIFT_RE = re.compile(
    r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af\u0400-\u04ff\u0600-\u06ff]"
    r"|\b(você|também|obrigad|isso|essa|qualcosa|però)\b",
    re.IGNORECASE,
)

MODELS = [
    ("qwen2.5:3b", "helper",       False),
    ("qwen3:4b",   "experimental", False),
    ("qwen2.5:7b", "chat default", False),
    ("command-r",  "HQ tier",      True),   # 30B, big-tier, evict-everything
]

PROMPT = "Decí en una sola oración, en español rioplatense, qué es un vector embedding."

def banner(s: str) -> None:
    print(f"\n{'=' * 70}\n{s}\n{'=' * 70}")


def smoke_one(backend: MLXBackend, alias: str, role: str) -> dict:
    print(f"\n--- {alias} ({role}) → {MLX_MODEL_ALIAS[alias]} ---")
    t0 = time.monotonic()
    try:
        resp = backend.chat(
            model=alias,
            messages=[
                {"role": "system", "content": "Respondé en español rioplatense (voseo)."},
                {"role": "user", "content": PROMPT},
            ],
            options=ChatOptions(temperature=0.0, num_predict=120),
        )
    except Exception as exc:
        print(f"  FAIL load/chat: {exc!r}")
        return {"alias": alias, "ok": False, "error": repr(exc)}
    dt = time.monotonic() - t0

    # Ollama-shape compat check
    try:
        text = resp.message.content
    except AttributeError:
        text = resp["message"]["content"]
    drift = DRIFT_RE.search(text)
    ok = bool(text.strip()) and drift is None
    print(f"  wall: {dt:.1f}s")
    print(f"  reply: {text.strip()[:200]!r}")
    if drift:
        print(f"  DRIFT detected: {drift.group()!r}")
    print(f"  → {'OK' if ok else 'FAIL'}")
    return {"alias": alias, "ok": ok, "wall_s": dt, "reply": text, "drift": bool(drift)}


def smoke_json(backend: MLXBackend, alias: str) -> dict:
    print(f"\n--- {alias} JSON-mode ---")
    import json as _json
    t0 = time.monotonic()
    try:
        resp = backend.chat(
            model=alias,
            messages=[
                {"role": "system", "content": "Sos un extractor de entidades."},
                {"role": "user", "content": "Devolvé un JSON con keys 'topic' y 'lang' para: 'el pibe juega al fútbol en River'."},
            ],
            options=ChatOptions(temperature=0.0, num_predict=120),
            format="json",
        )
        text = resp.message.content
        parsed = _json.loads(text)
        ok = isinstance(parsed, dict) and "topic" in parsed
    except Exception as exc:
        print(f"  FAIL json: {exc!r}")
        return {"alias": alias, "ok": False, "error": repr(exc)}
    dt = time.monotonic() - t0
    print(f"  wall: {dt:.1f}s  → parsed: {parsed}")
    print(f"  → {'OK' if ok else 'FAIL'}")
    return {"alias": alias, "ok": ok}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-big", action="store_true", help="saltear command-r (30B)")
    parser.add_argument("--only", help="probar un solo alias (ej. qwen2.5:3b)")
    args = parser.parse_args()

    banner("MLXBackend smoke test")
    backend = MLXBackend()

    targets = [(a, r, big) for (a, r, big) in MODELS
               if not (args.skip_big and big)
               and (args.only is None or a == args.only)]

    results = []
    for alias, role, _big in targets:
        results.append(smoke_one(backend, alias, role))

    # JSON-mode en el HQ tier (si se incluyó)
    if any(r["alias"] == "command-r" and r.get("ok") for r in results):
        results.append(smoke_json(backend, "command-r"))

    banner("RESUMEN")
    fails = [r for r in results if not r.get("ok")]
    for r in results:
        flag = "OK  " if r.get("ok") else "FAIL"
        extra = f" ({r['wall_s']:.1f}s)" if "wall_s" in r else ""
        print(f"  [{flag}] {r['alias']}{extra}")
    if fails:
        print(f"\n{len(fails)}/{len(results)} fallaron")
        return 1
    print(f"\nTodos los modelos OK ({len(results)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
