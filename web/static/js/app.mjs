// app.mjs — Entry point del chat de obsidian-rag.
//
// Phase W4 (2026-05-09): split de app.js (6584 LOC vanilla) en módulos ES.
// Este archivo es el punto de entrada declarado en el <script type="module">
// de index.html. Importa los módulos extraídos y, como transición progresiva,
// también ejecuta el bundle legado web/static/app-bundle.js que contiene
// el resto del código que todavía no fue migrado a módulos separados.
//
// Módulos ya extraídos:
//   - utils.mjs          Utilitarios DOM + formatters de fecha
//   - markdown.mjs       Wrapper de marked + sanitización
//   - settings.mjs       Vault picker, chat-mode toggle, TTS toggle
//   - voice.mjs          TTS playback via /api/tts
//   - session.mjs        SessionId + historial de queries
//
// El código de rendering (proposals WhatsApp/Mail/Calendar, appendSources,
// appendFeedback, chat SSE client, sidebar, mobile-ui) sigue en app-bundle.js
// y se migra incrementalmente en fases posteriores.
//
// Globals expuestos al window (para compat con onclick/attributes HTML):
//   window.activeScope, window.setActiveScope, window.clearActiveScope,
//   window.getActiveScopePayload — ver Feature H en app-bundle.js.

// Importamos los módulos utilitarios para que estén disponibles como
// globals en el contexto del módulo (no se usan directamente acá pero
// verificar que cargan sin error en el boot).
import "./utils.mjs";
import "./markdown.mjs";
import "./settings.mjs";
import "./voice.mjs";
import "./session.mjs";

// El bundle legado se carga dinámicamente después de que los módulos
// de utilidades estén listos. Usamos import() en lugar de <script> para
// mantener el orden de ejecución dentro del módulo graph y evitar FOUC.
// El bundle legado (app.js original) se carga vía <script> clásico
// desde index.html ANTES de este módulo. No hay nada más que hacer acá —
// este archivo existe para que el <script type="module"> en index.html
// sea válido y para que los módulos de utilidades se importen una única
// vez en el module graph del browser (deduplicación automática del browser).
//
// Fases posteriores moverán funciones del bundle a módulos dedicados.
