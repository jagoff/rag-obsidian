// refresh.mjs — SSE progress bar para el refresh manual del brief
// via /api/home/stream. Gestiona EventSource, trickle timer, y el botón ↻.

// Mapa de etiqueta legible por stage.
const STAGE_LABELS = {
  "today": "leyendo evidencia del día…",
  "signals": "fan-out de 9 señales…",
  "signals.gmail": "consultando Gmail…",
  "signals.whatsapp": "consultando WhatsApp…",
  "signals.calendar": "consultando Calendar…",
  "signals.mail_unread": "consultando Apple Mail…",
  "signals.youtube": "consultando YouTube…",
  "signals.contradictions": "buscando contradicciones…",
  "signals.loops_activo": "rastreando loops…",
  "signals.low_conf": "revisando queries low-conf…",
  "tomorrow": "agenda de mañana…",
  "forecast": "pronóstico del clima…",
  "pagerank": "computando autoridad…",
  "vaults": "actividad del vault…",
  "drive": "Google Drive…",
  "wa_unreplied": "chats WhatsApp pendientes…",
  "bookmarks": "bookmarks de Chrome…",
  "chrome": "top sitios web…",
  "eval": "trend de retrieval…",
  "followup": "loops aging…",
  "finance": "snapshot de finanzas…",
  "cards": "movimientos de tarjeta…",
  "spotify": "Spotify del día…",
  "sleep": "sueño de anoche…",
  "mood": "score de mood…",
  "youtube": "videos vistos…",
  "narrative": "qwen2.5:7b escribiendo…",
  "correlator": "armando patrones cross-source…",
};

const totalBlocks = 20;
function setBar(progressBar, progressLabel, pct, label) {
  if (!progressBar) return;
  const filled = Math.round((pct / 100) * totalBlocks);
  const bar = "█".repeat(filled) + "░".repeat(totalBlocks - filled);
  progressBar.textContent = `[${bar}] ${Math.round(pct)}%`;
  if (label && progressLabel) progressLabel.textContent = label;
}

// `onDone(payload)` se llama cuando el SSE emite "done" con el payload completo.
export function initRefreshButton(onDone, onFallbackLoad) {
  const refreshBtn = document.getElementById("brief-refresh");
  if (!refreshBtn) return;
  const progressEl = document.getElementById("hero-progress");
  const progressBar = document.getElementById("progress-bar");
  const progressLabel = document.getElementById("progress-label");
  let activeStream = null;
  let trickleTimer = null;
  let estimatedTotalStages = 26;

  function stopProgress() {
    if (activeStream) {
      try { activeStream.close(); } catch {}
      activeStream = null;
    }
    if (trickleTimer) {
      clearInterval(trickleTimer);
      trickleTimer = null;
    }
    if (progressEl) progressEl.hidden = true;
    refreshBtn.disabled = false;
    refreshBtn.textContent = "↻";
  }

  function startProgressSSE(regenerate) {
    if (!progressEl || !progressBar) return null;
    progressEl.hidden = false;
    let pct = 0;
    let donesSeen = 0;
    let lastStageTs = Date.now();
    let lastStageLabel = "leyendo señales…";
    setBar(progressBar, progressLabel, 0, "iniciando compute…");

    const url = `/api/home/stream${regenerate ? "?regenerate=true" : ""}`;
    const es = new EventSource(url);

    // El server emite `hello` con la lista exacta de stages.
    es.addEventListener("hello", (e) => {
      try {
        const data = JSON.parse(e.data);
        const main = (data.stages || []).length;
        const sub = Object.values(data.substages || {})
          .reduce((sum, arr) => sum + (Array.isArray(arr) ? arr.length : 0), 0);
        if (main + sub > 0) {
          estimatedTotalStages = main + sub;
        }
      } catch (err) {
        console.warn("[home.v2] hello event parse failed:", err);
      }
    });

    es.addEventListener("stage", (e) => {
      try {
        const data = JSON.parse(e.data);
        const stage = data.stage || "";
        const status = data.status || "";
        if (status === "done" || status === "timeout" || status === "error") {
          donesSeen++;
          const real = Math.min(92, Math.round((donesSeen / estimatedTotalStages) * 92));
          pct = Math.max(pct, real);
        } else if (status === "start") {
          const label = STAGE_LABELS[stage] || stage.replace(/_/g, " ") + "…";
          lastStageLabel = label;
          lastStageTs = Date.now();
        }
        setBar(progressBar, progressLabel, pct, lastStageLabel);
      } catch (err) {
        console.warn("[home.v2] stage event parse failed:", err);
      }
    });

    es.addEventListener("done", (e) => {
      try {
        const payload = JSON.parse(e.data);
        setBar(progressBar, progressLabel, 100, "listo!");
        onDone(payload);
      } catch (err) {
        console.error("[home.v2] done event parse failed:", err);
      } finally {
        stopProgress();
      }
    });

    // Named "error" SSE event (server-sent hard cap or compute failure).
    // Distinct from the EventSource built-in onerror which fires on
    // network errors. The server sends this before closing the stream.
    es.addEventListener("error", (e) => {
      if (e.data) {
        // Named SSE event: server signalled an error, stop immediately.
        console.warn("[home.v2] SSE error from server:", e.data);
        stopProgress();
        return;
      }
      // Built-in EventSource error (network drop / auto-reconnect attempt).
      // readyState can be CONNECTING (auto-reconnect) or CLOSED.
      // Close in both cases — we have a safety-timeout fallback.
      console.warn("[home.v2] SSE connection error, readyState=", es.readyState);
      stopProgress();
    });

    // Trickle: avance mínimo cuando el server tarda en mandar events.
    // Límite 97% (no 100) para que el salto final al 100% sea visible
    // cuando el SSE `done` llega. Permite avanzar past 92% (all-stages-done)
    // mientras el narrative LLM genera (~40-60s).
    trickleTimer = setInterval(() => {
      const sinceLast = (Date.now() - lastStageTs) / 1000;
      if (sinceLast > 1 && pct < 97) {
        pct = Math.min(97, pct + 0.3);
        setBar(progressBar, progressLabel, pct, lastStageLabel);
      }
    }, 1000);

    return es;
  }

  refreshBtn.addEventListener("click", () => {
    if (activeStream) return;
    refreshBtn.disabled = true;
    refreshBtn.textContent = "↻";
    activeStream = startProgressSSE(true);
    // Safety timeout 90s: si el SSE no termina, fallback al endpoint normal.
    setTimeout(() => {
      if (activeStream) {
        console.warn("[home.v2] SSE timeout — falling back to /api/home");
        stopProgress();
        onFallbackLoad({ regenerate: false });
      }
    }, 90_000);
  });
}
