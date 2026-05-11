// Avatares: primero intenta `/api/wa/avatar/{jid}` (Apple Contacts via
// AppleScript), si 404 cae al fallback de iniciales + color hash
// deterministic por JID.

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

  // Skip grupos — no aplican.
  if (jid.endsWith("@g.us")) return;

  const status = _avatarStatus.get(jid);
  if (status === "miss") return;
  if (status === "loading") return;

  const img = new Image();
  img.alt = "";
  img.loading = "lazy";
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
