/* diagnose-modal.js — modal "🩺 fix con IA" que delega a Devin.
 *
 * Flow:
 *
 *   1. Click 🩺 → modal abre, llama POST /api/auto-fix-devin con el contexto.
 *   2. El server spawnea `devin -p "<prompt>"` como subprocess y stream-ea
 *      su stdout al cliente via SSE:
 *        {type: "start", cmd: ["devin", "-p", ...]}
 *        {type: "output", chunk: "..."}      (stream de stdout)
 *        {type: "done", exit_code: 0, output: "<full output>"}
 *        {type: "error", message: "..."}
 *   3. El frontend acumula los chunks y los muestra en vivo. Al final,
 *      renderea el output completo con markdown-light.
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
    // Acumulador del output de Devin — en el endpoint Devin no hay
    // turnos discretos, solo un stream de stdout. Renderamos el output
    // crudo en un <pre> que va creciendo.
    const streamState = {
      outputAccum: "",
      // thoughtEl: el `<article>` con el thought bubble (se crea en 'start').
      // outputEl: el `<pre>` con el texto de la respuesta (lazy, se crea
      //           en el primer evento 'output' que llega — porque devin -p
      //           block-buffera y no vamos a recibir nada hasta el final).
      thoughtEl: null,
      outputEl: null,
    };
    try {
      const resp = await fetch("/api/auto-fix-devin", {
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
          handleDevinEvent(event, status, timeline, summaryBox, retryBtn, streamState);
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

  /** Convertir `<ref_file file="..." />` / `<ref_snippet ... />` que
   *  Devin emite en su output a HTML clickeable. */
  function formatDevinOutput(raw) {
    const safe = escapeHtml(raw);
    // `<ref_file file="/path/to/file" />` → `<code>/path/to/file</code>`
    return safe
      .replace(/&lt;ref_file file=&quot;([^&]+?)&quot;\s*\/&gt;/g,
               (_, p) => `<code>${p}</code>`)
      .replace(/&lt;ref_snippet file=&quot;([^&]+?)&quot;\s*lines=&quot;([^&]+?)&quot;\s*\/&gt;/g,
               (_, p, l) => `<code>${p}:${l}</code>`);
  }

  /** Procesa un evento del stream de Devin y actualiza el DOM.
   *
   * Eventos posibles:
   *   {type: "start", cmd: [...]}
   *   {type: "output", chunk: "..."}
   *   {type: "done", exit_code: 0, output: "..."}
   *   {type: "error", message: "..."}
   */
  function handleDevinEvent(event, status, timeline, summaryBox, retryBtn, streamState) {
    switch (event.type) {
      case "start": {
        status.textContent = "Devin investigando…";
        // Crear el bloque del thought bubble. NO creamos el `<pre>` vacío
        // todavía — `devin -p` block-buffera stdout (sin TTY pasa a
        // block-buffered, no line-buffered) así que durante 3-10 min no
        // vamos a recibir output ninguno. Mostrar una caja vacía con
        // placeholder "esperando..." era engañoso. El `<pre>` se crea
        // recién en el primer evento `output` que llegue (típicamente al
        // final, cuando recovery del transcript JSON inyecta la respuesta
        // completa). Hasta entonces solo mostramos el spinner + counter
        // del status arriba.
        if (!streamState.thoughtEl) {
          const turn = el("article", { class: "diag-turn diag-devin-turn" });
          turn.appendChild(el("div", { class: "diag-thought" },
            el("span", { class: "diag-thought-icon" }, "🤖"),
            el("span", { class: "diag-thought-text" },
              "Devin tiene acceso completo al repo — está leyendo el código, ejecutando tools, y resolviendo el problema. Esto puede tardar 3-10 minutos. Si cerrás esta modal, Devin sigue investigando en segundo plano y el resultado queda guardado en el audit log.")
          ));
          timeline.appendChild(turn);
          streamState.thoughtEl = turn;
        }
        break;
      }

      case "output": {
        // Lazy-create el `<pre>` la primera vez que llega output (no en
        // 'start' como antes, porque `devin -p` block-buffera y la caja
        // vacía con "esperando..." se quedaba 3-10 min en pantalla).
        if (!streamState.outputEl) {
          const outputPre = el("pre", { class: "diag-action-output diag-devin-stream" });
          // Si tenemos thoughtEl (turn que creamos en 'start'), pegamos
          // el pre adentro de ese mismo turn. Si no, creamos uno nuevo.
          if (streamState.thoughtEl) {
            streamState.thoughtEl.appendChild(outputPre);
          } else {
            const turn = el("article", { class: "diag-turn diag-devin-turn" });
            turn.appendChild(outputPre);
            timeline.appendChild(turn);
            streamState.thoughtEl = turn;
          }
          streamState.outputEl = outputPre;
        }
        streamState.outputAccum += event.chunk || "";
        streamState.outputEl.textContent = streamState.outputAccum;
        // Al recibir output real, el status vuelve a "investigando"
        // (sin contador) por si venía de un heartbeat con número.
        status.textContent = "Devin investigando…";
        // Auto-scroll al fondo mientras stream-ea.
        requestAnimationFrame(() => {
          streamState.outputEl.scrollTop = streamState.outputEl.scrollHeight;
        });
        break;
      }

      case "heartbeat": {
        // Devin sigue vivo pero no emitió output nuevo — contador de tiempo
        // para que el user sepa que no está colgado (el server emite este
        // evento cada ~3s cuando no hay output del subprocess).
        const s = event.elapsed_s || 0;
        const mm = Math.floor(s / 60);
        const ss = s % 60;
        status.textContent = mm > 0
          ? `Devin investigando… (${mm}m ${ss}s)`
          : `Devin investigando… (${ss}s)`;
        break;
      }

      case "done": {
        const ok = event.exit_code === 0;
        status.textContent = ok ? "✓ Devin terminó" : `✗ Devin salió con exit ${event.exit_code}`;
        status.classList.add(ok ? "diag-status-done" : "diag-status-error");
        retryBtn.disabled = false;

        // Final output — preferimos lo acumulado por chunks. Si está
        // vacío (caso típico del block-buffer), usar el campo `output`
        // del evento done que el server completó con el transcript JSON
        // recuperado.
        const finalOutput = streamState.outputAccum || event.output || "";
        if (finalOutput) {
          // Si todavía no creamos el `<pre>` (porque nunca llegó un
          // 'output' event antes), creamos ahora — para evitar perder
          // la respuesta cuando todo viene en el 'done'.
          if (!streamState.outputEl) {
            const outputPre = el("pre", { class: "diag-action-output diag-devin-stream" });
            if (streamState.thoughtEl) {
              streamState.thoughtEl.appendChild(outputPre);
            } else {
              const turn = el("article", { class: "diag-turn diag-devin-turn" });
              turn.appendChild(outputPre);
              timeline.appendChild(turn);
              streamState.thoughtEl = turn;
            }
            streamState.outputEl = outputPre;
          }
          streamState.outputEl.innerHTML = formatDevinOutput(finalOutput);
        }

        // Summary box.
        summaryBox.hidden = false;
        summaryBox.innerHTML = ok
          ? `<div class="diag-summary-icon">✓</div>` +
            `<div class="diag-summary-text">Devin completó el análisis. El output de arriba tiene el detalle de lo que investigó + qué hizo o qué recomienda.</div>`
          : `<div class="diag-summary-icon diag-summary-icon-fail">✗</div>` +
            `<div class="diag-summary-text">Devin salió con exit ${event.exit_code}. Revisá el output arriba.</div>`;
        requestAnimationFrame(() => {
          summaryBox.scrollIntoView({ behavior: "smooth", block: "end" });
        });
        break;
      }

      case "error": {
        status.textContent = `error: ${event.message}`;
        status.classList.add("diag-status-error");
        retryBtn.disabled = false;
        break;
      }
    }
  }

  // Expose.
  window.DiagnoseModal = { open, close };
})();
