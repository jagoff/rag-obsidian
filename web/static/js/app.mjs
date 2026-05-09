// app.mjs — Entry point del chat de obsidian-rag.
//
// Phase W4-phase-2 (2026-05-09): split de app.js (6584 LOC vanilla) en módulos ES.
//
// Este archivo es el punto de entrada declarado en el <script type="module">
// de index.html. Importa todos los módulos extraídos. El bundle clásico
// web/static/app.js se sigue cargando PRIMERO (ver index.html) para
// inicializar el estado global y exponer las funciones en window.*.
//
// Módulos Phase 1 (ya extraídos):
//   - utils.mjs          Utilitarios DOM + formatters de fecha
//   - markdown.mjs       Wrapper de marked + sanitización
//   - settings.mjs       Vault picker, chat-mode toggle, TTS toggle
//   - voice.mjs          TTS playback via /api/tts
//   - session.mjs        SessionId + historial de queries
//
// Módulos Phase 2 (esta fase):
//   - chat-client.mjs    send() / handleSlashCommand / side fetches / retry
//   - messages-render.mjs appendSources / appendFeedback / proposals / chips
//   - sidebar.mjs        Sessions list / new-chat / collapse / mobile drawer
//   - mobile-ui.mjs      Bottom-sheet / contact match overlay / quick-chips
//
// Globals expuestos al window (para compat con onclick/attributes HTML):
//   window.activeScope, window.setActiveScope, window.clearActiveScope,
//   window.getActiveScopePayload — Feature H en app.js.
//
//   window.send, window.refreshSessions, window.triggerNewChat,
//   window.openSidebarMobile, window.closeSidebarMobile — expuestos por
//   el bloque "Phase W4-phase-2" al final de app.js.

// ── Phase 1 ────────────────────────────────────────────────────────────────
import "./utils.mjs";
import "./markdown.mjs";
import "./settings.mjs";
import "./voice.mjs";
import "./session.mjs";

// ── Phase 2 ────────────────────────────────────────────────────────────────
// Importar para verificar que cargan sin error en el boot. Las funciones
// exportadas son re-wrappers de window.* (transición progresiva) — el
// código real sigue en app.js durante esta fase.
import "./chat-client.mjs";
import "./messages-render.mjs";
import "./sidebar.mjs";
import "./mobile-ui.mjs";
