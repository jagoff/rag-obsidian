#!/usr/bin/env python3
"""Eval cuantitativo del filter PT→ES sobre el corpus de respuestas
ya cacheadas en `rag_response_cache`. Mide:

  - % de respuestas con AL MENOS una palabra pt detectada (pre-filter)
  - % de palabras pt totales en el corpus (pre-filter)
  - mismas métricas POST-filter (debe ser ~0)
  - reducción absoluta y relativa
  - top palabras pt observadas (qué leaks son más comunes)

Uso:
    .venv/bin/python scripts/eval_pt_leaks.py [--limit N]

Output: tabla en terminal + JSON a stdout si `--json`.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

# Setup paths para importar rag.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag import _ragvec_state_conn  # noqa: E402
from rag.iberian_leak_filter import (  # noqa: E402
    _IBERIAN_LEAK_REPLACEMENTS,
    replace_iberian_leaks,
)

# Palabras pt observables (no patrones). Son las que el filter conoce —
# si encontramos alguna en el output, es leak.
PT_WORDS_TO_DETECT = (
    "primeira", "primeiro", "primeiramente",
    "falam", "falou", "fala",
    "vistes", "uma", "também",
    "tua", "teu", "tuas", "teus",
    "nos braços", "no braço",
    "avô", "avó", "irmão", "irmã", "filha", "filho", "criança", "mãe", "pai",
    "experiência", "ciência", "consciência", "paciência", "frequência",
    "importância", "circunstância",
    "ela é", "ela era", "ela foi", "ele tem", "ele é",
    "foi", "é",
    "esse", "essa", "esses", "essas", "isso", "isto", "aquilo",
    "estão", "melhor", "pior",
    "aqui está", "aqui estão",
    "ajudar", "ajuda",
    "você", "vocês",
    "ação", "ações", "solução", "soluções", "questão", "questões",
    "março", "maio", "junho", "julho", "setembro", "outubro",
    "novembro", "dezembro", "fevereiro",
    "hoje", "ontem", "amanhã",
    "não", "sim", "muito", "muita", "muitos", "muitas",
    "obrigado", "obrigada",
    "esqueças", "esqueça", "dessas", "desses",
    "voilà",
)

PT_DETECT_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in PT_WORDS_TO_DETECT) + r")\b",
    re.IGNORECASE,
)


def count_pt_words(text: str) -> Counter:
    """Cuenta cada palabra pt detectada en el texto."""
    if not text:
        return Counter()
    return Counter(m.group(1).lower() for m in PT_DETECT_RE.finditer(text))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Procesar solo N entries (default: todas)")
    parser.add_argument("--json", action="store_true",
                        help="Output JSON al final")
    args = parser.parse_args()

    with _ragvec_state_conn() as conn:
        cur = conn.execute(
            "SELECT id, ts, question, response FROM rag_response_cache "
            "ORDER BY ts DESC"
            + (f" LIMIT {int(args.limit)}" if args.limit else "")
        )
        rows = cur.fetchall()

    print(f"Eval sobre {len(rows)} entries de rag_response_cache.\n")

    pre_total_words = 0
    post_total_words = 0
    pre_leaked_responses = 0
    post_leaked_responses = 0
    pre_leak_count = Counter()
    post_leak_count = Counter()

    for _id, _ts, _q, response in rows:
        pre = count_pt_words(response)
        cleaned = replace_iberian_leaks(response)
        post = count_pt_words(cleaned)

        pre_total_words += sum(pre.values())
        post_total_words += sum(post.values())
        if pre:
            pre_leaked_responses += 1
        if post:
            post_leaked_responses += 1

        pre_leak_count += pre
        post_leak_count += post

    n = len(rows) if rows else 1

    print("┌─────────────────────────────────┬──────────┬──────────┐")
    print("│ Métrica                         │  Pre-fix │ Post-fix │")
    print("├─────────────────────────────────┼──────────┼──────────┤")
    print(f"│ Respuestas con al menos 1 leak  │ {pre_leaked_responses:5d} ({pre_leaked_responses/n*100:4.1f}%) │ {post_leaked_responses:5d} ({post_leaked_responses/n*100:4.1f}%) │")
    print(f"│ Total palabras pt detectadas    │ {pre_total_words:8d} │ {post_total_words:8d} │")
    print("└─────────────────────────────────┴──────────┴──────────┘")
    print()

    if pre_total_words > 0:
        reduction_pct = (1 - post_total_words / pre_total_words) * 100
        print(f"Reducción de leaks: {pre_total_words - post_total_words}/"
              f"{pre_total_words} = {reduction_pct:.1f}%\n")
    else:
        print("Sin leaks detectados pre-fix — corpus limpio.\n")

    if pre_leak_count:
        print("Top palabras pt observadas (pre-fix):")
        for word, cnt in pre_leak_count.most_common(20):
            print(f"  {cnt:4d}  {word}")
        print()

    if post_leak_count:
        print("⚠  Palabras pt que SOBREVIVEN al filter (post-fix):")
        for word, cnt in post_leak_count.most_common(10):
            print(f"  {cnt:4d}  {word}")
        print(f"\n  ↳ Considerá agregar reglas para estas palabras al filter.")
        print()

    if args.json:
        print(json.dumps({
            "n_responses": len(rows),
            "pre": {
                "leaked_responses": pre_leaked_responses,
                "leaked_pct": round(pre_leaked_responses / n * 100, 2),
                "total_pt_words": pre_total_words,
                "top_leaks": dict(pre_leak_count.most_common(20)),
            },
            "post": {
                "leaked_responses": post_leaked_responses,
                "leaked_pct": round(post_leaked_responses / n * 100, 2),
                "total_pt_words": post_total_words,
                "remaining_leaks": dict(post_leak_count.most_common(10)),
            },
            "reduction_pct": (
                round((1 - post_total_words / pre_total_words) * 100, 2)
                if pre_total_words > 0 else None
            ),
        }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
