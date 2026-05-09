// utils.mjs — Utilitarios DOM y de formato compartidos por todos los módulos.
// Sin dependencias externas (no importa otros módulos del proyecto).

/**
 * Crea un elemento DOM con clase y texto opcionales.
 * Equivalente al `el()` del app.js original.
 */
export function el(tag, className, text) {
  const e = document.createElement(tag);
  if (className) e.className = className;
  if (text !== undefined && text !== null) e.textContent = text;
  return e;
}

/** Escapa caracteres HTML para insertar texto user-supplied en el DOM. */
export function escapeHtml(s) {
  if (!s) return "";
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

/** Formatea milisegundos: <1000ms → "Xms", >=1000ms → "X.Xs". */
export function formatMs(ms) {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

/** Detecta iOS < 16 (sin soporte para la API Web Speech de calidad). */
export function isIOSVersionBelow16() {
  const ua = navigator.userAgent;
  const m = ua.match(/OS (\d+)_/);
  if (!m) return false;
  return parseInt(m[1], 10) < 16;
}

/** Devuelve "smooth" o "auto" según prefers-reduced-motion. */
export function smoothBehavior() {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches
    ? "auto" : "smooth";
}

export function scrollBottom() {
  window.scrollTo({ top: document.body.scrollHeight, behavior: smoothBehavior() });
}

export function obsidianUrl(filePath) {
  return "obsidian://open?file=" + encodeURIComponent(filePath);
}

/**
 * Convierte un JID de WhatsApp o `whatsapp://<jid>/<msg_id>` a un link
 * `https://wa.me/<phone>`. Devuelve "" para grupos o inputs inválidos.
 */
export function waHref(uri) {
  if (!uri || typeof uri !== "string") return "";
  let jid = uri;
  if (jid.indexOf("whatsapp://") === 0) {
    jid = jid.slice("whatsapp://".length);
    const slash = jid.indexOf("/");
    if (slash >= 0) jid = jid.slice(0, slash);
  }
  if (jid.indexOf("@g.us") >= 0) return "";
  const phone = jid.split("@")[0];
  if (/^\d{6,}$/.test(phone)) return "https://wa.me/" + phone;
  return "";
}

// ── Formatters de fecha ────────────────────────────────────────────────

/** Formatea ISO datetime a "jue, 23/04 16:00" (es-AR). */
export function formatIsoDatetime(iso) {
  if (!iso) return "";
  try {
    const d = new Date(iso);
    if (isNaN(d.getTime())) return iso;
    return d.toLocaleString("es-AR", {
      weekday: "short", day: "2-digit", month: "2-digit",
      hour: "2-digit", minute: "2-digit",
    });
  } catch {
    return iso;
  }
}

/** Formatea ISO a "jue, 23/04" (sin hora). */
export function formatDateOnly(iso) {
  if (!iso) return "";
  try {
    const d = new Date(iso);
    if (isNaN(d.getTime())) return iso;
    return d.toLocaleDateString("es-AR", {
      weekday: "short", day: "2-digit", month: "2-digit",
    });
  } catch {
    return iso;
  }
}

/**
 * Formatea fecha de forma amigable: "hoy 14:30", "mañana 9:00", etc.
 * Usa delta de días calendario (no ventana de 24h).
 */
export function formatFriendlyDate(isoStr) {
  if (!isoStr) return "";
  try {
    const d = new Date(isoStr);
    if (isNaN(d.getTime())) return isoStr;
    const now = new Date();
    const dayMs = 24 * 60 * 60 * 1000;
    const startOfDay = (x) => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime();
    const dayDiff = Math.round((startOfDay(d) - startOfDay(now)) / dayMs);
    const time = d.toLocaleTimeString("es-AR", {
      hour: "2-digit", minute: "2-digit",
    });
    if (dayDiff === 0) return `hoy ${time}`;
    if (dayDiff === 1) return `mañana ${time}`;
    if (dayDiff === -1) return `ayer ${time}`;
    return d.toLocaleString("es-AR", {
      weekday: "short", day: "numeric", month: "short",
      hour: "2-digit", minute: "2-digit",
    });
  } catch {
    return isoStr;
  }
}

/**
 * Convierte un valor `<input type="datetime-local">` a ISO8601 con offset
 * explícito de Argentina (-03:00).
 */
export function toIsoArgentina(localDateTime) {
  if (!localDateTime) return "";
  let v = String(localDateTime).trim();
  if (!v) return "";
  if (/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$/.test(v)) {
    v = v + ":00";
  }
  return v + "-03:00";
}

/**
 * Formatea tiempo relativo para "último contacto" en cards de WhatsApp.
 * Buckets: <60min, hoy, ayer, anteayer, días, semanas, meses, años.
 */
export function formatRelativeContact(isoStr) {
  if (!isoStr) return "";
  try {
    const d = new Date(isoStr);
    if (isNaN(d.getTime())) return "";
    const now = new Date();
    const diffMs = now.getTime() - d.getTime();
    const diffMin = Math.round(diffMs / 60000);
    const hh = String(d.getHours()).padStart(2, "0");
    const mm = String(d.getMinutes()).padStart(2, "0");
    if (diffMin < 0) {
      return formatFriendlyDate(isoStr);
    }
    if (diffMin < 60) {
      return diffMin <= 1 ? "hace un instante" : `hace ${diffMin} min`;
    }
    const dDay = new Date(d.getFullYear(), d.getMonth(), d.getDate());
    const nowDay = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const dayDiff = Math.round((nowDay - dDay) / (1000 * 60 * 60 * 24));
    if (dayDiff === 0) return `hoy ${hh}:${mm}`;
    if (dayDiff === 1) return `ayer ${hh}:${mm}`;
    if (dayDiff === 2) return `antes de ayer ${hh}:${mm}`;
    if (dayDiff < 7) return `hace ${dayDiff} días`;
    if (dayDiff < 30) {
      const weeks = Math.round(dayDiff / 7);
      return weeks === 1 ? "hace 1 semana" : `hace ${weeks} semanas`;
    }
    if (dayDiff < 365) {
      const months = Math.round(dayDiff / 30);
      return months === 1 ? "hace 1 mes" : `hace ${months} meses`;
    }
    const years = Math.round(dayDiff / 365);
    return years === 1 ? "hace 1 año" : `hace ${years} años`;
  } catch (_) {
    return "";
  }
}

/**
 * Detector heurístico de fechas en lenguaje natural rioplatense.
 * Devuelve `{ phrase, iso }` o `null` si no detecta nada.
 */
export function detectDateInMessage(text) {
  if (!text) return null;
  const now = new Date();
  const pad = (n) => String(n).padStart(2, "0");

  const patterns = [
    {
      re: /\bhoy\b.*?(\d{1,2})(?::(\d{2}))?\s*(?:hs?|horas?)?\b/i,
      extract(m) {
        const h = parseInt(m[1], 10);
        const min = parseInt(m[2] || "0", 10);
        const d = new Date(now.getFullYear(), now.getMonth(), now.getDate(), h, min);
        return { phrase: m[0], iso: `${d.getFullYear()}-${pad(d.getMonth()+1)}-${pad(d.getDate())}T${pad(h)}:${pad(min)}:00` };
      },
    },
    {
      re: /\bmañana\b.*?(\d{1,2})(?::(\d{2}))?\s*(?:hs?|horas?)?\b/i,
      extract(m) {
        const h = parseInt(m[1], 10);
        const min = parseInt(m[2] || "0", 10);
        const d = new Date(now.getFullYear(), now.getMonth(), now.getDate() + 1, h, min);
        return { phrase: m[0], iso: `${d.getFullYear()}-${pad(d.getMonth()+1)}-${pad(d.getDate())}T${pad(h)}:${pad(min)}:00` };
      },
    },
    {
      re: /\b(\d{1,2})\/(\d{1,2})(?:\/(\d{4}|\d{2}))?\b.*?(\d{1,2})(?::(\d{2}))?\s*(?:hs?|horas?)?\b/i,
      extract(m) {
        const day = parseInt(m[1], 10);
        const month = parseInt(m[2], 10) - 1;
        const year = m[3] ? parseInt(m[3].length === 2 ? "20" + m[3] : m[3], 10) : now.getFullYear();
        const h = parseInt(m[4], 10);
        const min = parseInt(m[5] || "0", 10);
        const d = new Date(year, month, day, h, min);
        return { phrase: m[0], iso: `${d.getFullYear()}-${pad(d.getMonth()+1)}-${pad(d.getDate())}T${pad(h)}:${pad(min)}:00` };
      },
    },
  ];

  for (const { re, extract } of patterns) {
    const m = text.match(re);
    if (m) {
      try {
        return extract(m);
      } catch (_) {}
    }
  }
  return null;
}

/**
 * Elimina una frase de fecha detectada del cuerpo del mensaje
 * para no duplicar la info en el preview.
 */
export function stripDatePhraseFromMessage(text, phrase) {
  if (!text || !phrase) return text;
  const escaped = phrase.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const rx = new RegExp(`\\s*[,.;]?\\s*${escaped}\\s*[,.;]?\\s*`, "i");
  return text.replace(rx, " ").replace(/\s+/g, " ").trim();
}

/** Formatea bytes humanos: B / KB / MB con 1 decimal. */
export function formatFileSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
