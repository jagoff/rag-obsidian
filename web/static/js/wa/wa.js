// Entry point de `/wa`. Conecta sidebar + thread + SSE + health indicator.
// Estado global mínimo (chat activo); el resto lo manejan los sub-módulos.

import * as chatlist from "./wa-chatlist.js";
import * as thread from "./wa-thread.js";
import * as sse from "./wa-sse.js";

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
      thread.open(jid);
    },
  });

  chatlist.load();
  startSSE();
}

function startSSE() {
  const el = $("wa-conn");

  sse.onConnectionState((open) => {
    if (!el) return;
    el.textContent = "●";
    el.className = open ? "wa-conn-indicator ok" : "wa-conn-indicator bad";
  });

  sse.on("hello", () => {
    if (el) {
      el.textContent = "●";
      el.className = "wa-conn-indicator ok";
    }
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

  sse.connect();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
