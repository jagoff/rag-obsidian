// Wrappers `fetch()` para `/api/wa/*`. Centralizan baseURL + errores.
// Sin auth en read/send (el repo es local-first); admin-token solo cuando
// agreguemos revoke en Fase 6.

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
