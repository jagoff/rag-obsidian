# benchmarks/

Scripts de medición **standalone** — miden latencia, throughput, drift, calidad
entre dos backends/configs. No son tests de pytest: no tienen `def test_*`, no
se corren con `pytest tests/`, y no son regresiones automáticas.

## Correr el que está

```bash
python benchmarks/bench_mlx_vs_ollama.py           # full bench (reinicia el web service)
python benchmarks/bench_mlx_vs_ollama.py --dry-run # 2 queries, sin restart, para sanity
```

## Agregar benchmarks nuevos

- Filename pattern: `bench_*.py`.
- Entrypoint: `def main()` + `if __name__ == "__main__": main()`.
- No importarlos desde `tests/` ni desde código de producción (`rag.py`, `web/`,
  `mcp_server.py`). Son scripts que se invocan a mano.
- Si el bench muta estado del sistema (plist, launchd, índice), que tenga
  `try/finally` con restore — igual que el actual.
