"""Bench de latencia del CLI chat path sin TTY — reproduce lo que hace
`rag chat` (retrieve + LLM streaming) y reporta timing por etapa.

Uso:
    .venv/bin/python scripts/bench_chat.py
    .venv/bin/python scripts/bench_chat.py --vault home --queries q1 q2 q3
    .venv/bin/python scripts/bench_chat.py --runs 3  # warm vs cold

Output por query: retrieve_ms, ttft_ms, llm_ms, total_ms, confidence, n_docs.
Tail: P50/P95 por métrica + ranking de queries más lentas.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ollama  # noqa: E402

import rag  # noqa: E402


DEFAULT_QUERIES = [
    "que sabes de Grecia",
    "que info tenes de finops",
    "dame info sobre obsidian-rag",
    "cuales son mis proyectos activos",
    "que notas tengo sobre machine learning",
]


def run_query(
    question: str,
    vaults: list[tuple[str, Path]],
    k: int = 5,
    history: list[dict] | None = None,
    model: str | None = None,
) -> dict:
    """Ejecuta una query completa (retrieve + LLM streaming) y devuelve timing.

    Mirror-ea el flujo de rag.py `chat()` loop: multi_retrieve con no_deep,
    pool=RERANK_POOL_MAX default, luego ollama.chat con _CLI_CHAT_OPTIONS.
    No lee stdin.
    """
    history = history or []
    t_turn = time.perf_counter()

    t_retrieve = time.perf_counter()
    result = rag.multi_retrieve(
        vaults, question, k,
        folder=None, history=history, tag=None, precise=False,
        multi_query=True, auto_filter=True,
        date_range=None, summary=None,
    )
    retrieve_ms = int((time.perf_counter() - t_retrieve) * 1000)

    if not result["docs"]:
        return {
            "question": question,
            "retrieve_ms": retrieve_ms,
            "ttft_ms": 0,
            "llm_ms": 0,
            "total_ms": int((time.perf_counter() - t_turn) * 1000),
            "confidence": float(result.get("confidence", 0)),
            "n_docs": 0,
            "response": "",
        }

    # Reproducir armado de messages del chat CLI.
    rules = rag.SYSTEM_RULES_STRICT
    context_parts = []
    for doc, meta in zip(result["docs"], result["metas"]):
        fp = meta.get("file", "")
        context_parts.append(f"[{fp}]\n{doc}")
    context = "\n\n".join(context_parts)

    messages = [{"role": "user", "content": (
        f"{rules}\nCONTEXTO:\n{context}\n\n"
        f"PREGUNTA: {question}\n\nRESPUESTA:"
    )}]

    options = {**rag.CHAT_OPTIONS, "num_ctx": 4096, "num_predict": 256}
    parts: list[str] = []
    t_first_token: float | None = None
    t_llm = time.perf_counter()
    try:
        for chunk in ollama.chat(
            model=model or rag.resolve_chat_model(),
            messages=messages,
            options=options,
            stream=True,
            keep_alive=rag.OLLAMA_KEEP_ALIVE,
        ):
            c = chunk.message.content or ""
            if c and t_first_token is None:
                t_first_token = time.perf_counter()
            parts.append(c)
    except Exception as e:
        return {
            "question": question,
            "retrieve_ms": retrieve_ms,
            "ttft_ms": -1,
            "llm_ms": -1,
            "total_ms": int((time.perf_counter() - t_turn) * 1000),
            "confidence": float(result.get("confidence", 0)),
            "n_docs": len(result["docs"]),
            "response": f"[LLM err: {e}]",
        }

    t_done = time.perf_counter()
    ttft_ms = int(((t_first_token or t_done) - t_llm) * 1000)
    llm_ms = int((t_done - t_llm) * 1000)
    total_ms = int((t_done - t_turn) * 1000)
    return {
        "question": question,
        "retrieve_ms": retrieve_ms,
        "ttft_ms": ttft_ms,
        "llm_ms": llm_ms,
        "total_ms": total_ms,
        "confidence": float(result.get("confidence", 0)),
        "n_docs": len(result["docs"]),
        "response": "".join(parts),
    }


def summarize(rows: list[dict]) -> None:
    if not rows:
        print("No rows.")
        return

    print()
    print(f"{'query':<40} {'retrieve':>10} {'ttft':>8} {'llm':>8} {'total':>8} {'conf':>6} {'docs':>5}")
    print("-" * 95)
    for r in rows:
        q = r["question"][:38]
        print(
            f"{q:<40} {r['retrieve_ms']:>8}ms {r['ttft_ms']:>6}ms "
            f"{r['llm_ms']:>6}ms {r['total_ms']:>6}ms "
            f"{r['confidence']:>6.3f} {r['n_docs']:>5}"
        )

    def p(xs: list[int], pct: float) -> int:
        if not xs:
            return 0
        xs_sorted = sorted(xs)
        idx = min(len(xs_sorted) - 1, int(round(pct * (len(xs_sorted) - 1))))
        return xs_sorted[idx]

    retrieve = [r["retrieve_ms"] for r in rows]
    ttft = [r["ttft_ms"] for r in rows if r["ttft_ms"] >= 0]
    llm = [r["llm_ms"] for r in rows if r["llm_ms"] >= 0]
    total = [r["total_ms"] for r in rows]
    print("-" * 95)
    print(
        f"{'stats':<40} {'retrieve':>10} {'ttft':>8} {'llm':>8} {'total':>8}"
    )
    print(
        f"{'  P50':<40} {p(retrieve,0.5):>8}ms {p(ttft,0.5):>6}ms "
        f"{p(llm,0.5):>6}ms {p(total,0.5):>6}ms"
    )
    print(
        f"{'  P95':<40} {p(retrieve,0.95):>8}ms {p(ttft,0.95):>6}ms "
        f"{p(llm,0.95):>6}ms {p(total,0.95):>6}ms"
    )
    if len(total) > 1:
        print(
            f"{'  mean':<40} {int(statistics.mean(retrieve)):>8}ms "
            f"{int(statistics.mean(ttft)):>6}ms "
            f"{int(statistics.mean(llm)):>6}ms "
            f"{int(statistics.mean(total)):>6}ms"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vault", default="work",
                    help="vault name (home/work/all). Default: work (indexado).")
    ap.add_argument("--queries", nargs="*", default=None,
                    help="queries custom (default: suite built-in)")
    ap.add_argument("--runs", type=int, default=1,
                    help="correr cada query N veces (para warm vs cold)")
    ap.add_argument("--quiet", action="store_true",
                    help="no imprime respuesta del LLM")
    ap.add_argument("--model", default=None,
                    help="override chat model (ej: qwen3:30b-a3b, qwen2.5:7b). "
                         "Default: resolve_chat_model().")
    args = ap.parse_args()

    queries = args.queries or DEFAULT_QUERIES

    if args.vault == "all":
        vaults = rag.resolve_vault_paths(["all"])
    else:
        vaults = rag.resolve_vault_paths([args.vault])
    if not vaults:
        print(f"vault '{args.vault}' no resolvió. Registrados: {list(rag._load_vaults_config()['vaults'])}")
        return 2
    print(f"vault: {args.vault} → {[(n, str(p)) for n,p in vaults]}")

    # Warmup: carga el reranker + corpus + bge-m3 antes de contar.
    print("warmup…")
    rag.warmup_async()
    # Forzar sync wait: una query dummy con timing-free garantiza todos los
    # caches cargados antes de la primera medida.
    try:
        rag.multi_retrieve(
            vaults, "warmup", 3, folder=None, history=[], tag=None,
            precise=False, multi_query=False, auto_filter=False,
            date_range=None, summary=None, rerank_pool=5,
        )
    except Exception:
        pass

    rows: list[dict] = []
    for i in range(args.runs):
        for q in queries:
            print(f"\n[run {i+1}/{args.runs}] → {q}")
            r = run_query(q, vaults, model=args.model)
            rows.append(r)
            print(
                f"  retrieve={r['retrieve_ms']}ms ttft={r['ttft_ms']}ms "
                f"llm={r['llm_ms']}ms total={r['total_ms']}ms "
                f"conf={r['confidence']:.3f} docs={r['n_docs']}"
            )
            if not args.quiet and r["response"]:
                print(f"  resp: {r['response'][:200]}")

    summarize(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
