# Documentación de obsidian-rag

Esto es un índice simple. Cada archivo responde a una pregunta concreta.

## Empezar acá

| Querés… | Abrí |
|---|---|
| Arrancar de cero y hacer tu primera pregunta | [empezar.md](./empezar.md) |
| Ver todos los comandos y sus flags | [comandos.md](./comandos.md) |
| Entender cómo funciona por dentro (con dibujos) | [como-funciona.md](./como-funciona.md) |
| Saber qué variables de entorno existen | [variables-entorno.md](./variables-entorno.md) |
| Entender los servicios que corren solos en segundo plano | [automatizaciones.md](./automatizaciones.md) |
| Arreglar algo que se rompió | [problemas-comunes.md](./problemas-comunes.md) |

## Docs técnicas (diseño / experimentos)

Estos son más densos. Solo abrilos si querés entender decisiones de diseño o experimentos medidos:

- [design-cross-source-corpus.md](./design-cross-source-corpus.md) — diseño del corpus multi-fuente (vault + WhatsApp + Gmail + Calendar + Reminders)
- [eval-baselines-2026-04-15.md](./eval-baselines-2026-04-15.md) — baseline medido del retriever
- [eval-tune-2026-04-15.md](./eval-tune-2026-04-15.md) — calibración del ranker
- [regression-investigation-2026-04-15.md](./regression-investigation-2026-04-15.md) — investigación de regresión
- [gamechangers-plan-2026-04-22.md](./gamechangers-plan-2026-04-22.md) — roadmap de mejoras grandes
- [tune-run-2026-04-22.md](./tune-run-2026-04-22.md) — última corrida de tuning
- [code-review-followup.md](./code-review-followup.md) — follow-up de code review

## Diagramas (versiones renderizadas)

Los `.svg` viven en [`./diagrams/`](./diagrams/). Fuentes `.mmd` editables al lado. Los más importantes también están en [como-funciona.md](./como-funciona.md) con versiones ASCII simples.

## Para la referencia super-técnica

- [`../README.md`](../README.md) — referencia operativa completa (paths, schemas, recetas)
- [`../CLAUDE.md`](../CLAUDE.md) — guía de arquitectura + invariantes + decisiones históricas. Es denso. Solo abrilo si vas a tocar `rag.py` en serio.
