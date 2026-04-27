/* diagnose-modal.js — modal compartido para "🩺 fix con IA".
 *
 * Public API (global, sin module system para que /logs y /status puedan
 * cargarlo con un <script> tag simple):
 *
 *   window.DiagnoseModal.open({
 *     service: "watch",
 *     file: "obsidian-rag/watch.log (stdout)",
 *     line_n: 1234,
 *     error_text: "OperationalError: ...",
 *     context_lines: ["...", "..."],   // optional, las 20 anteriores
 *     timestamp: "2026-04-26T19:47:50", // optional
 *   });
 *
 * Lifecycle:
 *   1. open() construye el modal en document.body (singleton — si ya
 *      hay uno abierto, lo cierra antes).
 *   2. Empieza el fetch SSE a /api/diagnose-error con el payload.
 *   3. Renderea progresivamente: header con modelo, body con la
 *      respuesta del LLM en Markdown light-rendered.
 *   4. Cada bloque ```bash``` se renderea con un botón "▶ Ejecutar".
 *   5. Click en "▶ Ejecutar" → POST /api/diagnose-error/execute con el
 *      comando exacto. El server valida contra una whitelist de comandos
 *      seguros (rag/ollama/launchctl/tail/head/cat/ls/wc + validators
 *      por argumento). Si el comando NO está en whitelist o tiene
 *      metachars peligrosos (`;`, `|`, `>`, `$()`, etc.), responde 403
 *      con el motivo. El modal sigue mostrando "📋 copiar" como
 *      alternativa para que el user pueda llevarlo a su terminal real.
 *
 *      Por compat con versiones que devolvían 503 (auto-execute
 *      deshabilitado), el handler abajo trata 503 igual que 403 →
 *      ofrecer copiar como alternativa.
 *
 * Markdown rendering: light, manual. NO agregamos `marked` por algo tan
 * acotado. Sólo necesitamos:
 *   - Headers `## Title` → <h3>
 *   - Bullets `- item` → <ul><li>
 *   - Código inline `code` → <code>
 *   - Bloques ```bash\n...\n``` → <pre><code>
 *   - **bold** → <strong>
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

  // ── Markdown light renderer ────────────────────────────────────────
  // Returns { node, commands } — el DOM y la lista de bloques bash que
  // detectamos para wirear el botón "▶ Ejecutar".
  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function renderMarkdown(text) {
    const node = document.createElement("div");
    const commands = [];
    if (!text) return { node, commands };

    // Splitear por bloques ``` para procesar bash separado del resto.
    const parts = text.split(/```(\w*)\n([\s\S]*?)(?:```|$)/);
    // Resultados: [text0, lang1, code1, text2, lang2, code2, ...]
    for (let i = 0; i < parts.length; i++) {
      if (i % 3 === 0) {
        // Texto markdown normal.
        const block = parts[i];
        if (!block.trim()) continue;
        const html = renderInlineMarkdown(block);
        // Insertar como wrapper para que herede el flow del .diag-stream.
        const wrapper = document.createElement("div");
        wrapper.innerHTML = html;
        while (wrapper.firstChild) node.appendChild(wrapper.firstChild);
      } else if (i % 3 === 1) {
        // Idioma del bloque (ej. "bash"). Skip; lo usamos en i+1.
        continue;
      } else {
        // Código del bloque.
        const lang = parts[i - 1];
        const code = parts[i].replace(/\n+$/, "");
        if (lang === "bash" || lang === "sh" || lang === "shell") {
          // Cada línea de comando es un bloque clickeable.
          for (const cmd of code.split("\n").map((s) => s.trim()).filter(Boolean)) {
            // Skip lines que son comments del LLM (ej. "# ejemplo de uso").
            if (cmd.startsWith("#")) continue;
            const cmdNode = renderCommandBlock(cmd);
            commands.push(cmd);
            node.appendChild(cmdNode);
          }
        } else {
          const pre = document.createElement("pre");
          const codeEl = document.createElement("code");
          codeEl.textContent = code;
          pre.appendChild(codeEl);
          node.appendChild(pre);
        }
      }
    }
    return { node, commands };
  }

  function renderInlineMarkdown(text) {
    // Headers: ## o ### → h3.
    let html = text;
    html = html.replace(/^###\s+(.+)$/gm, "<h3>$1</h3>");
    html = html.replace(/^##\s+(.+)$/gm, "<h3>$1</h3>");
    html = html.replace(/^#\s+(.+)$/gm, "<h3>$1</h3>");
    // Bullets — convertir secuencias de líneas que empiezan con `- `.
    html = html.replace(/((?:^- .+(?:\n|$))+)/gm, (match) => {
      const items = match
        .trimEnd()
        .split("\n")
        .map((l) => l.replace(/^- /, "").trim())
        .filter(Boolean)
        .map((l) => `<li>${renderInlineFormatting(escapeHtml(l))}</li>`)
        .join("");
      return `<ul>${items}</ul>`;
    });
    // Numbered lists.
    html = html.replace(/((?:^\d+\.\s.+(?:\n|$))+)/gm, (match) => {
      const items = match
        .trimEnd()
        .split("\n")
        .map((l) => l.replace(/^\d+\.\s/, "").trim())
        .filter(Boolean)
        .map((l) => `<li>${renderInlineFormatting(escapeHtml(l))}</li>`)
        .join("");
      return `<ol>${items}</ol>`;
    });
    // Párrafos: dividir por dobles \n y envolver en <p> los que no son
    // ya un block element (ul/ol/h3/pre).
    const paragraphs = html.split(/\n\n+/);
    return paragraphs
      .map((p) => {
        p = p.trim();
        if (!p) return "";
        if (/^<(?:h3|ul|ol|pre)/.test(p)) return p;
        // Inline formatting + escape para texto plano.
        return `<p>${renderInlineFormatting(escapeHtml(p))}</p>`;
      })
      .join("");
  }

  function renderInlineFormatting(escapedText) {
    // **bold**, `code`. Ambos sobre HTML ya escapado.
    let s = escapedText;
    s = s.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    s = s.replace(/`([^`]+?)`/g, "<code>$1</code>");
    return s;
  }

  function renderCommandBlock(cmd) {
    const block = document.createElement("div");
    block.className = "diag-cmd-block";
    const codeEl = document.createElement("code");
    codeEl.textContent = cmd;
    block.appendChild(codeEl);

    const runBtn = document.createElement("button");
    runBtn.type = "button";
    runBtn.className = "diag-btn diag-btn-execute";
    runBtn.textContent = "▶ ejecutar";
    block.appendChild(runBtn);

    const resultEl = document.createElement("div");
    resultEl.className = "diag-cmd-result";
    block.appendChild(resultEl);

    runBtn.addEventListener("click", async () => {
      // Confirmación obligatoria — el comando vino de un LLM, no
      // querés ejecutar a ciegas.
      if (!runBtn.classList.contains("diag-btn-confirm")) {
        runBtn.classList.add("diag-btn-confirm");
        runBtn.textContent = "▶▶ confirmar";
        return;
      }
      runBtn.disabled = true;
      runBtn.textContent = "⏳ ejecutando…";
      resultEl.textContent = "";
      try {
        const resp = await fetch("/api/diagnose-error/execute", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ command: cmd }),
        });
        // 403 = comando rechazado por whitelist (caso esperado, no error).
        // 503 = backwards compat con versiones donde /execute estaba deshabilitado
        // entero. Ambos: degradamos a "copiá manualmente" con el motivo.
        if (resp.status === 503 || resp.status === 403) {
          const data = await resp.json();
          resultEl.textContent = data.detail || "comando no ejecutable";
          runBtn.textContent = "deshabilitado";
          // Botón "copiar" como alternativa.
          const copyBtn = document.createElement("button");
          copyBtn.type = "button";
          copyBtn.className = "diag-btn diag-btn-execute";
          copyBtn.textContent = "📋 copiar";
          copyBtn.addEventListener("click", async () => {
            try {
              await navigator.clipboard.writeText(cmd);
              copyBtn.textContent = "✓ copiado";
              setTimeout(() => { copyBtn.textContent = "📋 copiar"; }, 1200);
            } catch {}
          });
          block.appendChild(copyBtn);
          return;
        }
        const data = await resp.json();
        if (data.exit_code === 0) {
          resultEl.textContent = `✓ ok\n${data.stdout || ""}`;
          runBtn.textContent = "✓ hecho";
        } else {
          resultEl.textContent = `✗ exit ${data.exit_code}\n${data.stderr || data.stdout || ""}`;
          runBtn.textContent = "✗ falló";
        }
      } catch (e) {
        resultEl.textContent = `error: ${e.message || e}`;
        runBtn.textContent = "✗ error";
      }
    });

    return block;
  }

  // ── Modal construction ─────────────────────────────────────────────
  async function open(payload) {
    close();  // singleton

    const overlay = document.createElement("div");
    overlay.className = "diag-overlay";
    overlay.addEventListener("click", (e) => {
      if (e.target === overlay) close();
    });

    const modal = document.createElement("div");
    modal.className = "diag-modal";
    overlay.appendChild(modal);

    // Header.
    const header = document.createElement("header");
    header.className = "diag-header";
    const title = document.createElement("h2");
    title.className = "diag-title";
    title.textContent = "🩺 fix con IA";
    header.appendChild(title);
    const meta = document.createElement("div");
    meta.className = "diag-meta";
    const metaParts = [];
    if (payload.service) metaParts.push(payload.service);
    if (payload.timestamp) metaParts.push(payload.timestamp);
    if (payload.line_n) metaParts.push(`L${payload.line_n}`);
    meta.textContent = metaParts.join(" · ");
    header.appendChild(meta);
    const closeBtn = document.createElement("button");
    closeBtn.type = "button";
    closeBtn.className = "diag-close";
    closeBtn.setAttribute("aria-label", "Cerrar");
    closeBtn.textContent = "✕";
    closeBtn.addEventListener("click", close);
    header.appendChild(closeBtn);
    modal.appendChild(header);

    // Body.
    const body = document.createElement("section");
    body.className = "diag-body";

    // Error block.
    const errBlock = document.createElement("div");
    errBlock.className = "diag-error-block";
    const errLine = document.createElement("pre");
    errLine.className = "diag-error-line";
    errLine.textContent = payload.error_text || "(sin texto)";
    errBlock.appendChild(errLine);
    body.appendChild(errBlock);

    const status = document.createElement("div");
    status.className = "diag-status";
    status.textContent = "consultando al LLM…";
    body.appendChild(status);

    const stream = document.createElement("article");
    stream.className = "diag-stream";
    stream.setAttribute("aria-live", "polite");
    body.appendChild(stream);

    modal.appendChild(body);

    // Footer.
    const footer = document.createElement("footer");
    footer.className = "diag-footer";
    const retryBtn = document.createElement("button");
    retryBtn.type = "button";
    retryBtn.className = "diag-btn";
    retryBtn.textContent = "↻ reintentar";
    retryBtn.disabled = true;
    retryBtn.addEventListener("click", () => open(payload));
    footer.appendChild(retryBtn);
    const closeFooterBtn = document.createElement("button");
    closeFooterBtn.type = "button";
    closeFooterBtn.className = "diag-btn diag-btn-close-footer";
    closeFooterBtn.textContent = "cerrar";
    closeFooterBtn.addEventListener("click", close);
    footer.appendChild(closeFooterBtn);
    modal.appendChild(footer);

    document.body.appendChild(overlay);
    _activeModal = overlay;
    document.addEventListener("keydown", _onKeydown);

    // SSE fetch — usamos POST con body JSON, así que no podemos usar
    // EventSource (que sólo soporta GET). Usamos fetch + ReadableStream.
    _abortController = new AbortController();
    let accum = "";
    try {
      const resp = await fetch("/api/diagnose-error", {
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
          if (event.type === "model") {
            status.textContent = `Diagnosticando con ${event.name}…`;
          } else if (event.type === "token") {
            accum += event.content;
            stream.innerHTML = "";
            const { node } = renderMarkdown(accum);
            stream.appendChild(node);
          } else if (event.type === "done") {
            status.textContent = "✓ listo";
            status.classList.add("diag-status-done");
            retryBtn.disabled = false;
            stream.innerHTML = "";
            const { node } = renderMarkdown(accum);
            stream.appendChild(node);
          } else if (event.type === "error") {
            status.textContent = `error del LLM: ${event.message}`;
            status.classList.add("diag-status-error");
            retryBtn.disabled = false;
          }
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

  // Expose.
  window.DiagnoseModal = { open, close };
})();
