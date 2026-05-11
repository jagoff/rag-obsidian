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

// Cache in-memory para evitar pegarle al endpoint mil veces por scroll.
const _avatarStatus = new Map(); // jid → "ok" | "miss" | "loading"

export function renderInto(el, jid, initials, chatName) {
  if (!el) return;
  el.innerHTML = "";
  const fallback = document.createElement("span");
  fallback.textContent = (initials || "?").slice(0, 2);
  el.style.background = colorFor(jid);
  el.appendChild(fallback);

  if (!jid) return;

  // Antes skipeaba grupos por design. Bug: los grupos también tienen
  // foto de perfil (set por el admin, bridge la sirve). Ahora pedimos
  // para todos — si no existe, el `onerror` se queda con iniciales.

  const status = _avatarStatus.get(jid);
  if (status === "miss") return;
  if (status === "loading") return;

  const img = new Image();
  img.alt = "";
  // NO setear `loading="lazy"` acá: cuando un <img> está OFF-DOM con
  // lazy, el browser nunca dispara el request (verificado en Chrome
  // 2026-05-11 — el load event no firea hasta que se appende, pero
  // el código depende del onload PARA appenderlo → deadlock).
  img.decoding = "async";
  img.referrerPolicy = "no-referrer";
  img.style.width = "100%";
  img.style.height = "100%";
  img.style.objectFit = "cover";
  img.onload = () => {
    _avatarStatus.set(jid, "ok");
    if (el.contains(img)) return;
    el.innerHTML = "";
    el.style.background = "transparent";
    el.appendChild(img);
  };
  img.onerror = () => {
    _avatarStatus.set(jid, "miss");
    // El fallback ya está visible — no hacer nada.
  };
  const params = new URLSearchParams();
  if (chatName) params.set("name", chatName);
  const qs = params.toString();
  img.src = `/api/wa/avatar/${encodeURIComponent(jid)}${qs ? "?" + qs : ""}`;
  _avatarStatus.set(jid, "loading");
}
