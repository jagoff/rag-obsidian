// markdown.mjs — Wrappers de renderizado Markdown vía marked (vendored).
// `marked` es un global cargado por /static/vendor/marked.min.js antes
// de este módulo.

import { escapeHtml, obsidianUrl } from "./utils.mjs";

/**
 * Sanitiza un nodo del DOM eliminando atributos peligrosos y elementos
 * de script/iframe inline. Sólo se llama sobre el output de marked.
 */
export function _sanitizeNode(node) {
  const BLOCKED_ATTRS = ["onerror", "onload", "onclick", "onmouseover",
    "onfocus", "onblur", "onkeydown", "onkeyup", "onkeypress",
    "onmousedown", "onmouseup", "onsubmit", "onreset",
    "formaction", "srcdoc"];
  if (node.nodeType !== Node.ELEMENT_NODE) return;
  const tag = node.tagName.toUpperCase();
  if (tag === "SCRIPT" || tag === "IFRAME" || tag === "OBJECT" || tag === "EMBED") {
    node.remove();
    return;
  }
  for (const attr of BLOCKED_ATTRS) {
    if (node.hasAttribute(attr)) node.removeAttribute(attr);
  }
  // data-* atributos en <a>/<img>: seguro; javascript: hrefs no.
  if (tag === "A") {
    const href = node.getAttribute("href") || "";
    if (/^javascript:/i.test(href.trim())) {
      node.removeAttribute("href");
    }
  }
  if (tag === "IMG") {
    const src = node.getAttribute("src") || "";
    if (/^javascript:/i.test(src.trim())) {
      node.removeAttribute("src");
    }
  }
  for (const child of [...node.childNodes]) {
    _sanitizeNode(child);
  }
}

export function _sanitizeHtml(html) {
  const div = document.createElement("div");
  div.innerHTML = html;
  _sanitizeNode(div);
  return div.innerHTML;
}

/**
 * Pre-procesa el texto antes de pasarlo a marked:
 *   - <<ext>>…<</ext>> → <span class="ext">⚠ …</span>
 *   - [[Wikilinks]]     → [Wikilinks](obsidian://…)
 */
export function preprocess(text) {
  // Notas de extensión (<<ext>>…<</ext>>).
  text = text.replace(/<<ext>>([\s\S]*?)<\/ext>>/g, (_, inner) => {
    return `<span class="ext">⚠ ${escapeHtml(inner.trim())}</span>`;
  });
  // Wikilinks [[Nota]] → [Nota](obsidian://open?file=Nota).
  text = text.replace(/\[\[([^\]]+)\]\]/g, (_, title) => {
    const href = obsidianUrl(title);
    return `[${title}](${href})`;
  });
  return text;
}

/**
 * Post-procesa el HTML de salida de marked:
 * Añade `target="_blank" rel="noopener noreferrer"` a todos los <a>.
 */
export function postprocess(html) {
  return html.replace(/<a\s/g, '<a target="_blank" rel="noopener noreferrer" ');
}

// Configuración de marked. Se ejecuta una vez al cargar el módulo.
// `marked` es el global del vendor bundle.
/* global marked */
marked.use({
  breaks: true,
  gfm: true,
  renderer: {
    // GFM strikethrough (~text~) destruye paths como `iCloud~md~obsidian`.
    // Reconstruimos los tildes literales en lugar de emitir <del>.
    del({ tokens }) {
      return "~" + this.parser.parseInline(tokens) + "~";
    },
    link({ href, title, tokens }) {
      const text = this.parser.parseInline(tokens);
      if (!href) return text;
      const isObsidian = href.startsWith("obsidian://");
      const target = isObsidian ? "" : ' target="_blank" rel="noopener noreferrer"';
      const titleAttr = title ? ` title="${escapeHtml(title)}"` : "";
      if (isObsidian) {
        return `<a href="${href}"${titleAttr} class="obsidian-link">${text}</a>`;
      }
      return `<a href="${href}"${titleAttr}${target}>${text}</a>`;
    },
  },
});

/** Renderiza texto Markdown a HTML sanitizado. */
export function renderMarkdown(text) {
  if (!text) return "";
  try {
    const preprocessed = preprocess(text);
    const raw = marked.parse(preprocessed);
    const postprocessed = postprocess(raw);
    return _sanitizeHtml(postprocessed);
  } catch (err) {
    console.error("[markdown] error al renderizar:", err);
    return escapeHtml(text);
  }
}
