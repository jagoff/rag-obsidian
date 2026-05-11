// Fallback de avatares con iniciales + color hash deterministic por JID.
// En Fase 10 vamos a preferir `/api/wa/avatar/{jid}` (proxy al bridge) y
// caer acá solo si 404. Por ahora todos los chats usan este fallback.

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

export function renderInto(el, jid, initials) {
  if (!el) return;
  el.innerHTML = "";
  el.style.background = colorFor(jid);
  const span = document.createElement("span");
  span.textContent = (initials || "?").slice(0, 2);
  el.appendChild(span);
}
