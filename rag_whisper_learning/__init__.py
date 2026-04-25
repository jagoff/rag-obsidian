"""Whisper learning loop — Phase 2 del plan whatsapp-whisper-learning.

Componentes:
- `vocab` — refresh + lookup del vocabulario aprendido (`rag_whisper_vocab`).
  Job nightly extrae términos raros de notas + chats + contactos + correcciones
  y replace la tabla. El listener lo lee y arma `--prompt` dinámico.
- `corrections` — utilities para `/fix` command + vault diff watcher.
- `auto_correct` — confidence-gated LLM correction con qwen2.5:7b (Step 3).

Esquema en rag.py:
- `rag_audio_transcripts` (extendido en Phase 2 con audio_hash, chat_id,
  avg_logprob, corrected_text, correction_source, note_path, note_initial_hash).
- `rag_audio_corrections` (id, audio_hash, original, corrected, source, ts,
  chat_id, context).
- `rag_whisper_vocab` (term PK, weight, source, last_seen_ts, refreshed_at).

Doc: ~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Notes/04-Archive/99-obsidian-system/99-Claude/system/whatsapp-whisper-learning/plan.md
"""

from rag_whisper_learning.vocab import (
    refresh_vocab,
    get_top_vocab_terms,
)

__all__ = [
    "refresh_vocab",
    "get_top_vocab_terms",
]
