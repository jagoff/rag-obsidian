// Entry point de `/wa`. Conecta sidebar + thread + health indicator.
// Estado global mínimo (chat activo); el resto lo manejan los sub-módulos.

import * as chatlist from "./wa-chatlist.js";
import * as thread from "./wa-thread.js";

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
  startHealthIndicator();
}

async function startHealthIndicator() {
  const el = $("wa-conn");
  if (!el) return;
  const tick = async () => {
    try {
      const r = await fetch("/api/wa/chats?limit=1");
      if (r.ok) {
        el.textContent = "●";
        el.className = "wa-conn-indicator ok";
      } else {
        el.textContent = "●";
        el.className = "wa-conn-indicator bad";
      }
    } catch {
      el.textContent = "●";
      el.className = "wa-conn-indicator bad";
    }
  };
  tick();
  setInterval(tick, 30_000);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
