/* diagnose-modal.js — modal "🩺 fix con IA" con agent loop server-side.
 *
 * Flow nuevo (vs. la versión anterior que diagnosticaba textualmente y
 * mostraba botones para ejecutar comandos uno por uno):
 *
 *   1. Click 🩺 → modal abre, llama POST /api/auto-fix con el contexto.
 *   2. El server arranca un agent loop (LLM → tool_call → result → LLM).
 *   3. Cada paso del agent llega vía SSE como un evento separado:
 *        {type: "model", name: "qwen2.5:7b"}
 *        {type: "turn", n: 1}
 *        {type: "thought", text: "Voy a verificar si el daemon está vivo"}
 *        {type: "action", command: "launchctl list com.fer..."}
 *        {type: "action_result", exit_code: 0, stdout: "...", stderr: "..."}
 *        {type: "action_rejected", command: "...", reason: "..."}
 *        {type: "done", summary: "Reinicié el daemon, errores cesaron"}
 *        {type: "max_turns_reached", limit: 8}
 *        {type: "error", message: "..."}
 *   4. El frontend renderea cada evento como una entrada en una timeline.
 *      Sin botones, sin confirmaciones — el agent corre solo hasta done.
 *
 * Public API global:
 *
 *   window.DiagnoseModal.open({
 *     service: "watch",
 *     file: "obsidian-rag/watch.log (stdout)",
 *     line_n: 1234,
 *     error_text: "OperationalError: ...",
 *     context_lines: ["...", "..."],   // optional, ~20 líneas anteriores
 *     timestamp: "2026-04-26T19:47:50",
 *   });
 */

(function () {
  "use strict";

  // ── Singleton state ────────────────────────────────────────────────
  let _activeModal = null;
  let _abortController = null;

  function close() {
    if (_abortController) {
      try { _abortController.abort(); } catch (_) {}
      _abortController = null;
    }
    if (_activeModal) {
      _activeModal.remove();
      _activeModal = null;
    }
    document.removeEventListener("keydown", _onKeydown);
  }

  function _onKeydown(e) {
    if (e.key === "Escape") {
      e.preventDefault();
      close();
    }
  }

  // ── Helpers ────────────────────────────────────────────────────────
  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function el(tag, attrs, ...kids) {
    const e = document.createElement(tag);
    if (attrs) {
      for (const k in attrs) {
        if (k === "class") e.className = attrs[k];
        else if (k === "html") e.innerHTML = attrs[k];
        else e.setAttribute(k, attrs[k]);
      }
    }
    for (const k of kids) {
      if (k == null) continue;
      e.appendChild(typeof k === "string" ? document.createTextNode(k) : k);
    }
    return e;
  }

  // ── Modal construction ─────────────────────────────────────────────
  async function open(payload) {
    close();  // singleton

    const overlay = el("div", { class: "diag-overlay" });
    overlay.addEventListener("click", (e) => {
      if (e.target === overlay) close();
    });

    const modal = el("div", { class: "diag-modal" });
    overlay.appendChild(modal);

    // Header.
    const header = el("header", { class: "diag-header" });
    const title = el("h2", { class: "diag-title" }, "🩺 fix con IA");
    header.appendChild(title);
    const meta = el("div", { class: "diag-meta" });
    const metaParts = [];
    if (payload.service) metaParts.push(payload.service);
    if (payload.timestamp) metaParts.push(payload.timestamp);
    if (payload.line_n) metaParts.push(`L${payload.line_n}`);
    meta.textContent = metaParts.join(" · ");
    header.appendChild(meta);
    const closeBtn = el("button", {
      type: "button", class: "diag-close", "aria-label": "Cerrar",
    }, "✕");
    closeBtn.addEventListener("click", close);
    header.appendChild(closeBtn);
    modal.appendChild(header);

    // Body.
    const body = el("section", { class: "diag-body" });

    // Error block (siempre arriba).
    const errBlock = el("div", { class: "diag-error-block" });
    const errLine = el("pre", { class: "diag-error-line" });
    errLine.textContent = payload.error_text || "(sin texto)";
    errBlock.appendChild(errLine);
    body.appendChild(errBlock);

    // Status arriba (cambia con cada turn).
    const status = el("div", { class: "diag-status" }, "iniciando agente…");
    body.appendChild(status);

    // Timeline — cada turn agrega un .diag-turn aquí.
    const timeline = el("div", { class: "diag-timeline" });
    body.appendChild(timeline);

    // Summary — visible al final cuando hay done event.
    const summaryBox = el("div", { class: "diag-summary", hidden: "true" });
    body.appendChild(summaryBox);

    modal.appendChild(body);

    // Footer.
    const footer = el("footer", { class: "diag-footer" });
    const retryBtn = el("button", { type: "button", class: "diag-btn", disabled: "true" }, "↻ reintentar");
    retryBtn.addEventListener("click", () => open(payload));
    footer.appendChild(retryBtn);
    const closeFooterBtn = el("button", { type: "button", class: "diag-btn diag-btn-close-footer" }, "cerrar");
    closeFooterBtn.addEventListener("click", close);
    footer.appendChild(closeFooterBtn);
    modal.appendChild(footer);

    document.body.appendChild(overlay);
    _activeModal = overlay;
    document.addEventListener("keydown", _onKeydown);

    // ── SSE fetch ───────────────────────────────────────────────────
    _abortController = new AbortController();
    let currentTurn = null;  // referencia al div del turn activo
    try {
      const resp = await fetch("/api/auto-fix", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: _abortController.signal,
      });
      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status}`);
      }
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let idx;
        while ((idx = buffer.indexOf("\n\n")) !== -1) {
          const chunk = buffer.slice(0, idx);
          buffer = buffer.slice(idx + 2);
          const dataLine = chunk.split("\n").find((l) => l.startsWith("data: "));
          if (!dataLine) continue;
          let event;
          try { event = JSON.parse(dataLine.slice(6)); } catch (_) { continue; }
          currentTurn = handleEvent(event, status, timeline, summaryBox, retryBtn, currentTurn);
        }
      }
    } catch (err) {
      if (err.name !== "AbortError") {
        status.textContent = `error de red: ${err.message || err}`;
        status.classList.add("diag-status-error");
        retryBtn.disabled = false;
      }
    } finally {
      _abortController = null;
    }
  }

  /** Procesa un evento del agent loop y actualiza el DOM. */
  function handleEvent(event, status, timeline, summaryBox, retryBtn, currentTurn) {
    switch (event.type) {
      case "model":
        status.textContent = `agente activo · modelo: ${event.name}`;
        return currentTurn;

      case "turn": {
        // Crear un nuevo turn en la timeline.
        const turn = el("article", { class: "diag-turn" });
        const turnHead = el("header", { class: "diag-turn-head" });
        turnHead.appendChild(el("span", { class: "diag-turn-n" }, `turno ${event.n}`));
        turn.appendChild(turnHead);
        timeline.appendChild(turn);
        // Auto-scroll al turn nuevo.
        requestAnimationFrame(() => {
          turn.scrollIntoView({ behavior: "smooth", block: "end" });
        });
        return turn;
      }

      case "thought": {
        if (!currentTurn) return currentTurn;
        const t = el("div", { class: "diag-thought" });
        t.appendChild(el("span", { class: "diag-thought-icon" }, "💭"));
        t.appendChild(el("span", { class: "diag-thought-text" }, event.text || ""));
        currentTurn.appendChild(t);
        return currentTurn;
      }

      case "action": {
        if (!currentTurn) return currentTurn;
        const a = el("div", { class: "diag-action" });
        a.appendChild(el("span", { class: "diag-action-icon" }, "▶"));
        a.appendChild(el("code", { class: "diag-action-cmd" }, event.command || ""));
        a.appendChild(el("span", { class: "diag-action-status" }, "ejecutando…"));
        currentTurn.appendChild(a);
        return currentTurn;
      }

      case "action_result": {
        if (!currentTurn) return currentTurn;
        // Encontrar el último .diag-action y actualizarlo.
        const lastAction = currentTurn.querySelector(".diag-action:last-of-type");
        if (lastAction) {
          const statusEl = lastAction.querySelector(".diag-action-status");
          const ok = event.exit_code === 0;
          if (statusEl) {
            statusEl.textContent = ok
              ? `✓ ok (${event.duration_s}s)`
              : `✗ exit ${event.exit_code} (${event.duration_s}s)`;
            statusEl.classList.add(ok ? "diag-action-ok" : "diag-action-fail");
          }
        }
        // Mostrar el output debajo de la acción.
        const out = (event.stdout || event.stderr || "").trim();
        if (out) {
          const pre = el("pre", { class: "diag-action-output" });
          pre.textContent = out.length > 800 ? out.slice(0, 800) + "\n…(truncado)" : out;
          currentTurn.appendChild(pre);
        }
        return currentTurn;
      }

      case "action_rejected": {
        if (!currentTurn) return currentTurn;
        const lastAction = currentTurn.querySelector(".diag-action:last-of-type");
        if (lastAction) {
          const statusEl = lastAction.querySelector(".diag-action-status");
          if (statusEl) {
            statusEl.textContent = `✗ rechazado: ${event.reason}`;
            statusEl.classList.add("diag-action-rejected");
          }
        }
        return currentTurn;
      }

      case "done": {
        status.textContent = "✓ resuelto";
        status.classList.add("diag-status-done");
        retryBtn.disabled = false;
        if (event.summary) {
          summaryBox.hidden = false;
          summaryBox.innerHTML = `<div class="diag-summary-icon">✓</div>` +
                                 `<div class="diag-summary-text">${escapeHtml(event.summary)}</div>`;
          requestAnimationFrame(() => {
            summaryBox.scrollIntoView({ behavior: "smooth", block: "end" });
          });
        }
        return currentTurn;
      }

      case "max_turns_reached": {
        status.textContent = `✗ agotó ${event.limit} turnos sin resolver`;
        status.classList.add("diag-status-error");
        retryBtn.disabled = false;
        summaryBox.hidden = false;
        summaryBox.innerHTML = `<div class="diag-summary-icon diag-summary-icon-fail">✗</div>` +
                               `<div class="diag-summary-text">El agente no pudo resolver el error en ${event.limit} turnos. ` +
                               `Revisá la timeline arriba para ver qué intentó.</div>`;
        return currentTurn;
      }

      case "error": {
        status.textContent = `error: ${event.message}`;
        status.classList.add("diag-status-error");
        retryBtn.disabled = false;
        return currentTurn;
      }

      default:
        return currentTurn;
    }
  }

  // Expose.
  window.DiagnoseModal = { open, close };
})();
