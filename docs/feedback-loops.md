# Feedback loops

Anticipatory agent + Bot WA draft + Brief feedback + Voice brief + Whisper learning + Implicit feedback. Resumen + invariantes en [`CLAUDE.md`](../CLAUDE.md).

## Anticipatory Agent

Daemon `com.fer.obsidian-rag-anticipate` (10min) push proactivo a WA. 3 señales: calendar proximity ([15,90]min), temporal echo (cosine ≥0.70 vs nota >60d), stale commitment (≥7d, reusa `find_followup_loops`). Doc completo: [`docs/anticipatory-agent.md`](anticipatory-agent.md).

CLI: `rag anticipate [run|explain|log] [-n N --only-sent]`, `rag silence anticipate-{calendar,echo,commitment}`. Kill switches: `RAG_ANTICIPATE_DISABLED=1`. Tabla nueva: `rag_anticipate_candidates` (analytics, todos candidates loggean).

**Footer pattern**: `proactive_push(dedup_key=<key>)` agrega `_anticipate:<key>_` al body. El listener TS parsea ese footer cuando user reacciona 👍/👎/🔇 y postea a `POST /api/anticipate/feedback` → `rag_anticipate_feedback`.

## Bot WA draft loop

Auto-aprendizaje del modelo de respuestas. Listener TS ([`/Users/fer/whatsapp-listener`](file:///Users/fer/whatsapp-listener)) genera `bot_draft` por LLM y postea al RagNet group. User responde `/si` / `/no` / `/editar <texto>`. Listener postea a `POST /api/draft/decision` → tabla `rag_draft_decisions` (append-only, retention infinita, **gold humano para fine-tunes**).

CLI: `rag draft stats [--plain]`. Activación: `WA_DRAFT_ALL_CONTACTS=1` en plist del listener.

**Bug pattern lección (2026-04-29)**: helper público (`isXEnabled()`) respetaba flag pero call site real (SQL builder de `processDraftIncoming`) la ignoraba — siempre filtraba por whitelist. Lección: cuando agregás flag de comportamiento, auditar TODOS los call sites donde el feature decide qué procesar; no alcanza con que el helper la respete.

**Latencia retrieve_only**: `loadVaultContextForDraft` timeout subido 8s→12s (commit `c160079`) por p50 ~9s del rerank cross-encoder + BM25 + embed secuenciales.

### Fine-tune drafts — DPO + LoRA

Cierra el loop: pares `(bot_draft, sent_text)` cuando user hace `/editar` se entrenan via [DPO](https://arxiv.org/abs/2305.18290) sobre Qwen2.5-7B-Instruct con LoRA r=8/alpha=16 sobre q+v projections. NO sustituye el modelo del listener — solo accesible via `POST /api/draft/preview` con `RAG_DRAFTS_FT=1`.

CLI: `rag drafts finetune [--dry-run --epochs 1 --lr 5e-6]`. Requiere ≥100 GOLD pairs (rows `decision='approved_editar'` con `sent_text != bot_draft`). Adapter en `~/.local/share/obsidian-rag/drafts_ft/`. Métricas held-out 80/20 split por draft_id: BLEU-1, similarity char-level, **preference win rate** (% donde sim(pred, chosen) > sim(pred, rejected)). Setup deps: `uv tool install --reinstall --editable '.[finetune]'`.

## Brief feedback loop

Reactions del user a briefs morning/evening/digest. Body lleva footer `_brief:<vault_relpath>_`. Listener TS detecta reaction 👍/👎/🔇 dentro de 30min → postea a `POST /api/brief/feedback` → `rag_brief_feedback`. CLI: `rag brief stats [--plain]`.

### Brief schedule auto-tuning

Si user mutea consistentemente el morning en primera hora, sistema mueve el plist a horario más tarde automáticamente. Lógica en [`rag/brief_schedule.py`](../rag/brief_schedule.py): `analyze_brief_feedback(brief_kind, lookback_days=30)` lee `rag_brief_feedback`, decision rule `mutes_first_hour ≥ 3 AND mute/(mute+positive) > 0.5` → shift `+30min` iterativo dentro de bandas seguras (`morning ∈ [06:30, 09:00]`, `today ∈ [18:00, 21:00]`, `digest ∈ [21:00, 23:30]`). Tabla `rag_brief_schedule_prefs` (single-row-per-kind, upsert). `_services_spec()` lee la pref antes de generar cada plist.

CLI: `rag brief schedule [status|reset|auto-tune] [--apply --kind morning|today|digest|all]`. Daemon: `com.fer.obsidian-rag-brief-auto-tune` Domingo 03:00.

## Voice brief

Phase 2.C: morning brief sintetiza voice note OGG/Opus + manda al WA antes del texto. Pipeline ([`rag/voice_brief.py`](../rag/voice_brief.py)): strip markdown → `say -v Mónica --file-format=AIFF` → `ffmpeg libopus 24k 16kHz mono` → cache `~/.local/share/obsidian-rag/voice_briefs/YYYY-MM-DD-morning.ogg` (idempotente).

Caps: texto >4000 chars → trim, audio >5MB → fallback text-only, sin `say` o `ffmpeg` → degrade graceful. Footer `_brief:<path>_` queda intacto en el texto.

CLI: `rag voice-brief generate --date YYYY-MM-DD [--apply --voice "Diego" --text "..."]`, `rag morning --voice`. Activar daemon: `RAG_MORNING_VOICE=1` en `com.fer.obsidian-rag-morning.plist`. Cleanup auto: `rag maintenance` borra audios >30d.

## Whisper learning loop

Sistema transcripción audios WA aprende del corpus + correcciones. Plan completo: [vault](obsidian://open?vault=Notes&file=04-Archive%2F99-obsidian-system%2F99-AI%2Fsystem%2Fwhatsapp-whisper-learning%2Fplan).

3 surfaces: (1) **Pasivo** — daemon `com.fer.obsidian-rag-whisper-vocab` (03:15) → `rag_whisper_vocab` (caps por source: 100 corrections, 100 contacts, 200 notes, 100 chats). (2) **Explícito** — `/fix <texto>` por WA marca última transcripción gold. (3) **Confidence-gated LLM correct** — si `avg_logprob < -0.8`, listener pasa output por `qwen2.5:7b` con sysprompt + few-shot + vocab hints.

3 tablas SQL: `rag_audio_transcripts` (con `corrected_text`, `correction_source`, `avg_logprob`, `audio_hash`), `rag_audio_corrections` (append-only), `rag_whisper_vocab` (refresh full nightly).

CLI: `rag whisper [stats|vocab refresh|vocab show|patterns|export|import]`. WA cmds: `/fix`, `/whisper [stats|recent N]`. Dashboard: [/transcripts](https://ra.ai/transcripts) (server-rendered, dark mode default).

Env tuning en plist listener: `WHISPER_LLM_CORRECT_THRESHOLD=-0.8`, `WHISPER_LLM_CORRECT_MODEL=qwen2.5:7b`. Kill switches: `WHISPER_LLM_CORRECT_DISABLE=1`, `WHISPER_TELEMETRY_DISABLE=1`.
