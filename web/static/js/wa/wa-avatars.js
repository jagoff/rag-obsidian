// Avatares — chain del backend:
//   1) Apple Contacts (foto que vos pusiste — tu libreta).
//   2) WhatsApp bridge (foto de perfil pública del peer / grupo).
//   3) 404 → frontend cae a iniciales + color hash deterministic.
// Aplica a contactos individuales Y grupos (los grupos tienen foto
// seteada por el admin, el bridge la sirve en `?jid=...@g.us`).

const PALETTE = [
  "#5b67ce", "#cd5e7c", "#7c8d3b", "#3a8788",
  "#a8743f", "#7e57c2", "#4b8aaf", "#c8703e",
  "#5d7c5b", "#9a5a8d", "#c44e4e", "#4a90a4",
];

function hashJID(jid) {
  let h = 5381;
  for (let i = 0; i < jid.length; i++) {
    h = ((h << 5) + h + jid.charCodeAt(i)) | 0;
  }
  return Math.abs(h);
}

export function colorFor(jid) {
  return PALETTE[hashJID(jid || "?") % PALETTE.length];
}

// Cache in-memory de avatares conocidos. Status:
//   - "miss": el endpoint devolvió 404, no insistir.
//   - URL string: el endpoint devolvió 200, esta es la URL que cargó OK.
// Bug previo: cacheabamos "loading" y skipeabamos requests posteriores;
// pero el LI puede recrearse (FLIP / SSE / re-render del chatlist) y
// el `el` capturado en el closure quedaba detached → la imagen cargaba
// pero nunca se appendaba a nada visible. Fix: en onload, queryselectear
// TODOS los `.wa-chat-avatar[data-jid="..."]` actuales para appendear
// al el vivo. Múltiples requests al mismo URL son free (HTTP cache).
const _avatarStatus = new Map(); // jid → "miss" | <url> when known good

function cssEscapeAttr(s) {
  return String(s).replace(/"/g, '\\"');
}

function applyImage(jid, url) {
  // Busca TODOS los avatar divs vivos con este jid (header del thread +
  // sidebar) y les pone la imagen. El el original del closure puede
  // estar detached por re-renders, pero estos queries siempre traen
  // los actuales del DOM.
  const els = document.querySelectorAll(
    `.wa-chat-avatar[data-jid="${cssEscapeAttr(jid)}"], `
    + `.wa-thread-avatar[data-jid="${cssEscapeAttr(jid)}"]`,
  );
  for (const el of els) {
    // Si ya tiene img con el src correcto, skip.
    const existing = el.querySelector("img");
    if (existing && existing.src.endsWith(url.split("/").pop())) continue;
    const img = new Image();
    img.alt = "";
    img.decoding = "async";
    img.referrerPolicy = "no-referrer";
    img.style.width = "100%";
    img.style.height = "100%";
    img.style.objectFit = "cover";
    img.src = url;
    img.onload = () => {
      el.innerHTML = "";
      el.style.background = "transparent";
      el.appendChild(img);
    };
  }
}

export function renderInto(el, jid, initials, chatName) {
  if (!el) return;
  el.innerHTML = "";
  // Marcar el data-jid también en el thread-avatar para que applyImage
  // lo encuentre vía querySelectorAll.
  if (jid && !el.dataset.jid) el.dataset.jid = jid;
  const fallback = document.createElement("span");
  fallback.textContent = (initials || "?").slice(0, 2);
  el.style.background = colorFor(jid);
  el.appendChild(fallback);

  if (!jid) return;

  const cached = _avatarStatus.get(jid);
  if (cached === "miss") return;
  if (typeof cached === "string") {
    // Ya sabemos que carga OK — re-usar la URL sin nuevo fetch.
    applyImage(jid, cached);
    return;
  }

  // Primera vez para este jid — dispará el fetch real.
  // NO setear `loading="lazy"` acá: cuando un <img> está OFF-DOM con
  // lazy, el browser nunca dispara el request (verificado en Chrome
  // 2026-05-11 — el load event no firea hasta que se appende, pero
  // el código depende del onload PARA appenderlo → deadlock).
  const params = new URLSearchParams();
  if (chatName) params.set("name", chatName);
  const qs = params.toString();
  const url = `/api/wa/avatar/${encodeURIComponent(jid)}${qs ? "?" + qs : ""}`;
  const probe = new Image();
  probe.decoding = "async";
  probe.referrerPolicy = "no-referrer";
  probe.onload = () => {
    _avatarStatus.set(jid, url);
    applyImage(jid, url);
  };
  probe.onerror = () => {
    _avatarStatus.set(jid, "miss");
    // El fallback ya está visible.
  };
  probe.src = url;
}
