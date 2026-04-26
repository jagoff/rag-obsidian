# obsidian-rag — make targets.
#
# Conventions:
#   - `.venv/bin/python` is the authoritative interpreter. Tests and
#     `rag eval` run against the local editable install; `uv tool install`
#     is only for the global `rag` / `obsidian-rag-mcp` entry points.
#   - `-m "not slow"` skips multi-process stress tests by default (~6-10s
#     each). Use `test-all` when you need them.
#   - `eval` needs ollama up + the vault mounted + bge-reranker cached —
#     ~2-3 min wall. If you're iterating on something unrelated to
#     retrieval, don't run it on every commit, only before pushing.

.PHONY: help install test test-all test-fast lint format eval eval-fast \
        silent-errors silent-summary clean tune tune-apply \
        stats ruff check coverage drift-watcher

# Default target: show the list.
help:
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) \
	  | sort \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install:  ## Re-install the editable uv tool (after Python code changes)
	# Incluye los extras `entities` (gliner para NER, RAG_ENTITY_LOOKUP=ON
	# default desde 2026-04-21) y `stt` (faster-whisper para transcripción
	# del whisper learning loop). Sin estos extras los 5 ingesters loggean
	# `dep gliner not available` cada corrida y la feature de entity-aware
	# retrieval queda silentemente desactivada — bug operativo encontrado
	# en producción 2026-04-25, ahora bake-eado al install default.
	# `spotify` queda como opt-in puro porque requiere config OAuth manual.
	uv tool install --reinstall --editable '.[entities,stt]'

test:  ## Fast suite: pytest -q skipping @pytest.mark.slow
	.venv/bin/python -m pytest tests/ -q -m "not slow" --tb=short

test-fast:  ## Parallel suite via pytest-xdist (-n auto, skips slow)
	.venv/bin/python -m pytest tests/ -q -m "not slow" -n auto --tb=short

test-all:  ## Full suite including slow tests (sequential — slow tests share state)
	.venv/bin/python -m pytest tests/ -q --tb=short

coverage:  ## Coverage report (terminal + HTML at htmlcov/index.html)
	.venv/bin/python -m pytest tests/ -q -m "not slow" \
	  --cov=rag --cov=web --cov=mcp_server \
	  --cov-report=term-missing --cov-report=html

lint:  ## ruff check (CI parity)
	uvx ruff check

check: lint  ## Alias for lint

format:  ## ruff --fix + ruff format (safe auto-fixes + style)
	uvx ruff check --fix
	uvx ruff format

eval:  ## rag eval with latency buckets + CI gate (P95 <2500ms)
	unset RAG_EXPLORE && .venv/bin/python -m rag eval --latency --max-p95-ms 2500

eval-fast:  ## rag eval without latency measurement (just hit@k + MRR)
	unset RAG_EXPLORE && .venv/bin/python -m rag eval

tune:  ## Dry-run rag tune — shows winner weights without persisting
	unset RAG_EXPLORE && .venv/bin/python -m rag tune --samples 500

tune-apply:  ## rag tune --apply (persists winner to ranker.json + backup)
	unset RAG_EXPLORE && .venv/bin/python -m rag tune --samples 500 --apply --yes

silent-errors:  ## Tail silent_errors.jsonl (last 20)
	.venv/bin/python -m rag log --silent-errors -n 20

silent-summary:  ## Aggregate silent_errors.jsonl by (where, exc_type)
	.venv/bin/python -m rag log --silent-errors --summary -n 20

drift-watcher:  ## Run drift watcher manually (alerts if singles_hit5 drops >5pp run-over-run)
	.venv/bin/python scripts/drift_watcher.py

stats:  ## rag stats — index + models overview
	.venv/bin/python -m rag stats

clean:  ## Remove Python/ruff/pytest caches
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache \) \
	  -not -path "./.venv/*" -prune -exec rm -rf {} +
	@echo "cleaned."
