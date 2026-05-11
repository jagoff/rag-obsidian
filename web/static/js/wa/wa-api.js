// Wrappers `fetch()` para `/api/wa/*`. Centralizan baseURL + errores.
// Sin auth en read/send (el repo es local-first); el revoke (delete-for-
// everyone) requiere admin token, que `admin-auth.js` inyecta vía
// monkey-patch de `window.fetch` cuando el browser está en loopback.

const BASE = "";

async function jsonGET(path) {
  const r = await fetch(BASE + path, { credentials: "same-origin" });
  if (!r.ok) throw new Error(`GET ${path} → ${r.status}`);
  return r.json();
}

async function jsonPOST(path, body) {
  const r = await fetch(BASE + path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    credentials: "same-origin",
    body: JSON.stringify(body || {}),
  });
  if (!r.ok) throw new Error(`POST ${path} → ${r.status}`);
  return r.json();
}

export async function fetchChats({ limit = 50, beforeTs = null, q = null } = {}) {
  const params = new URLSearchParams({ limit: String(limit) });
  if (beforeTs) params.set("before_ts", beforeTs);
  if (q) params.set("q", q);
  return jsonGET(`/api/wa/chats?${params}`);
}

export async function fetchThread(jid, { limit = 50, beforeTs = null } = {}) {
  const params = new URLSearchParams({ limit: String(limit) });
  if (beforeTs) params.set("before_ts", beforeTs);
  return jsonGET(`/api/wa/thread/${encodeURIComponent(jid)}?${params}`);
}

export async function markRead(jid, lastSeenTs = null) {
  return jsonPOST("/api/wa/mark_read", { jid, last_seen_ts: lastSeenTs });
}

export async function sendText(jid, text, replyTo = null) {
  const body = { jid, text };
  if (replyTo) body.reply_to = replyTo;
  return jsonPOST("/api/wa/send", body);
}

export async function react(jid, messageId, senderJid, fromMe, emoji) {
  return jsonPOST("/api/wa/react", {
    jid,
    message_id: messageId,
    sender_jid: senderJid || "",
    from_me: !!fromMe,
    emoji: emoji || "",
  });
}

export async function revoke(jid, messageId) {
  // admin-auth.js monkey-patcha fetch() y agrega el Bearer del
  // admin_token automáticamente para /api/wa/revoke (loopback-only).
  const r = await fetch("/api/wa/revoke", {
    method: "POST",
    headers: { "content-type": "application/json" },
    credentials: "same-origin",
    body: JSON.stringify({ jid, message_id: messageId }),
  });
  if (!r.ok) {
    const txt = await r.text().catch(() => "");
    throw new Error(`POST /api/wa/revoke → ${r.status}${txt ? ` — ${txt.slice(0, 200)}` : ""}`);
  }
  return r.json();
}

export async function pinChat(jid) {
  const r = await fetch(`/api/wa/chats/${encodeURIComponent(jid)}/pin`, {
    method: "POST",
    credentials: "same-origin",
  });
  if (!r.ok) throw new Error(`pin ${r.status}`);
  return r.json();
}

export async function unpinChat(jid) {
  const r = await fetch(`/api/wa/chats/${encodeURIComponent(jid)}/unpin`, {
    method: "POST",
    credentials: "same-origin",
  });
  if (!r.ok) throw new Error(`unpin ${r.status}`);
  return r.json();
}

export async function hide(jid, messageId) {
  // "Delete for me" — solo escribe en la tabla `revokes` del bridge
  // local, no envía protocol message. Usado para mensajes inbound
  // (WhatsApp no permite revoke-for-everyone de mensajes ajenos).
  const r = await fetch("/api/wa/hide", {
    method: "POST",
    headers: { "content-type": "application/json" },
    credentials: "same-origin",
    body: JSON.stringify({ jid, message_id: messageId }),
  });
  if (!r.ok) {
    const txt = await r.text().catch(() => "");
    throw new Error(`POST /api/wa/hide → ${r.status}${txt ? ` — ${txt.slice(0, 200)}` : ""}`);
  }
  return r.json();
}

export async function typing(jid, state) {
  // Fire-and-forget; los errores los swallow para no romper UX.
  try {
    return await jsonPOST("/api/wa/typing", { jid, state });
  } catch (e) {
    console.debug("[wa-api] typing failed (ignored)", e);
    return { ok: false };
  }
}

// Health del bridge — usado por el indicador visual del header.
export async function bridgeHealth() {
  try {
    const r = await fetch("/api/whatsapp/contacts/match", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query: "" }),
    });
    return r.ok;
  } catch {
    return false;
  }
}
