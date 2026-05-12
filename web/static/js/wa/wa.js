// Entry point de `/wa`. Conecta sidebar + thread + SSE + health indicator.
// Estado global mínimo (chat activo); el resto lo manejan los sub-módulos.

import * as chatlist from "./wa-chatlist.js";
import * as thread from "./wa-thread.js";
import * as sse from "./wa-sse.js";
import * as cmdk from "./wa-cmdk.js";
import * as liquid from "./wa-liquid-glass.js";
import * as anticipate from "./wa-anticipate.js";
import * as memory from "./wa-memory.js";
import * as promises from "./wa-promises.js";
import * as moodCheck from "./wa-mood-check.js";
import * as tts from "./wa-tts.js";

const $ = (id) => document.getElementById(id);

function init() {
  const listEl = $("wa-chatlist");
  const loadingEl = $("wa-chatlist-loading");
  const searchEl = $("wa-search-input");

  const bodyEl = $("wa-thread-body");
  const emptyEl = $("wa-empty-state");
  const nameEl = $("wa-thread-name");
  const avatarEl = $("wa-thread-avatar");
  const presenceEl = $("wa-thread-presence");
  const composerEl = $("wa-composer");

  const backBtn = $("wa-back-btn");
  if (backBtn) {
    backBtn.addEventListener("click", () => {
      document.body.dataset.pane = "sidebar";
    });
  }

  thread.init({ bodyEl, emptyEl, nameEl, avatarEl, presenceEl, composerEl });

  chatlist.init({
    listEl, loadingEl, searchEl,
    onSelect: (jid) => {
      document.body.dataset.pane = "thread";
      thread.open(jid).then(() => {
        // Después de que el header renderee el nombre, montar el botón 🧠
        // Recordar para que `wa-memory` apunte al contacto activo.
        const name = document.getElementById("wa-thread-name")?.textContent || "";
        memory.mountTrigger(jid, name);
      });
    },
  });

  chatlist.load();
  startSSE();
  startClock();
  startThemeToggle();
  startChatlistAutoRefresh();
  startThreadReopenOnResume();

  // Command palette Cmd+K — keymap global, abre overlay con search +
  // acciones. Click sobre un chat result llama el mismo onChatSelect
  // que la sidebar, así thread.open ejecuta navegación + render.
  cmdk.init({
    onChatSelect: (jid /* , messageId */) => {
      document.body.dataset.pane = "thread";
      thread.open(jid).then(() => {
        const name = document.getElementById("wa-thread-name")?.textContent || "";
        memory.mountTrigger(jid, name);
        promises.mountTrigger(jid, name);
        moodCheck.setActiveJid(jid);
        tts.setActiveJid(jid);
      });
    },
  });

  // Liquid Glass — mouse-tracked specular spotlight sobre bubbles +
  // tinted glass del thread header desde el avatar dominante.
  liquid.init();

  // Anticipador — drawer "✨ hoy" en sidebar header. Surface los top-N
  // candidates del daemon `com.fer.obsidian-rag-anticipate` para que el
  // user actúe sin abrir RagNet ni CLI. Endpoint: /api/wa/anticipate/today.
  anticipate.init({
    onChatSelect: (jid) => {
      document.body.dataset.pane = "thread";
      thread.open(jid).then(() => {
        const name = document.getElementById("wa-thread-name")?.textContent || "";
        memory.mountTrigger(jid, name);
        promises.mountTrigger(jid, name);
        moodCheck.setActiveJid(jid);
        tts.setActiveJid(jid);
      });
    },
  });

  // Memoria Universal — drawer "🧠 Recordar" en el thread header. Surface
  // vault notes + WA history del contacto activo. Endpoint: /api/wa/memory/<jid>.
  memory.init();

  // Promise Tracker — chip "🪨 N" en thread header. Lista pending de
  // promesas con el contacto + acciones resolver/cancelar.
  promises.init();

  // Mood Mirror tonal check — banner inline encima del composer cuando el
  // draft actual + mood actual sugieren cuidar el tono.
  moodCheck.init();

  // Voz Espejo — botón 🎙 en composer. TTS texto → Mónica → OGG/Opus → PTT.
  tts.init();

  // Cmd+K también dispara `thread.open` — propagar al memory drawer.
}

// Cada 30s re-fetcheamos el chatlist. SSE entrega chat_update events
// en vivo, pero si la conexión SSE muere brevemente (sleep del mac,
// network blip, web restart) los events de ese gap se pierden. El
// refresh periódico re-sincroniza el sidebar con la verdad del bridge.
function startChatlistAutoRefresh() {
  setInterval(() => {
    if (document.hidden) return;  // skip si la tab está background
    try { chatlist.load(); } catch (e) { console.warn("[wa] chatlist refresh failed", e); }
  }, 30_000);
}

// Cuando el tab vuelve de background (otro programa, otra tab) o el
// SSE se reconecta tras un drop, re-fetcheamos el thread activo +
// chatlist completo para asegurar coherencia con el bridge.
function startThreadReopenOnResume() {
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) {
      chatlist.load();
      const activeJID = thread.getActiveJID();
      if (activeJID) thread.reload();
    }
  });
}

function startThemeToggle() {
  const btn = document.getElementById("wa-theme-toggle");
  if (!btn) return;
  const apply = (t) => {
    document.documentElement.setAttribute("data-theme", t);
    try {
      localStorage.setItem("wa-theme", t);
    } catch {}
  };
  btn.addEventListener("click", () => {
    const cur = document.documentElement.getAttribute("data-theme") || "dark";
    apply(cur === "dark" ? "light" : "dark");
  });
}

function startClock() {
  const el = document.getElementById("wa-stat-clock");
  if (!el) return;
  const tick = () => {
    const d = new Date();
    const hh = String(d.getHours()).padStart(2, "0");
    const mm = String(d.getMinutes()).padStart(2, "0");
    el.textContent = `${hh}:${mm}`;
  };
  tick();
  // Align al próximo minuto para que el cambio sea exacto.
  const now = new Date();
  const ms = (60 - now.getSeconds()) * 1000;
  setTimeout(() => {
    tick();
    setInterval(tick, 60_000);
  }, ms);
}

function startSSE() {
  const el = $("wa-conn");

  sse.onConnectionState((open) => {
    if (!el) return;
    const label = el.querySelector(".wa-conn-label");
    if (open) {
      el.classList.remove("bad");
      el.classList.add("ok");
      if (label) label.textContent = "ONLINE";
    } else {
      el.classList.remove("ok");
      el.classList.add("bad");
      if (label) label.textContent = "OFFLINE";
    }
  });

  sse.on("hello", () => {
    if (el) {
      const label = el.querySelector(".wa-conn-label");
      if (label) label.textContent = "ONLINE";
      el.classList.remove("bad");
      el.classList.add("ok");
    }
    // Cualquier reconnect del SSE (incluído el primero) trae un
    // `hello`. Forzar reload del chatlist garantiza que cualquier
    // chat_update que se haya perdido durante el gap se aplique
    // implícitamente via el listing fresh del backend.
    try { chatlist.load(); } catch (e) { console.warn("[wa] post-hello reload failed", e); }
  });

  sse.on("new_message", (payload) => {
    if (payload && payload.message) {
      thread.appendMessageIfActive(payload.jid, payload.message);
    }
  });

  sse.on("chat_update", (payload) => {
    chatlist.applyChatUpdate(payload);
  });

  sse.on("reaction_changed", (payload) => {
    thread.applyReactionChange(payload);
  });

  sse.on("message_revoked", (payload) => {
    thread.applyRevoke(payload);
  });

  sse.on("presence", (payload) => {
    thread.applyPresence(payload);
  });

  sse.on("wa_call", (payload) => {
    thread.applyCallEvent(payload);
  });

  sse.connect();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
