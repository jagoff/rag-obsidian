// Shared helpers for layout persistence.
// localStorage remains the hot path; pages can opt into server sync so
// layouts survive service restarts and ra.ai/local origin changes.

let _serverLayoutPage = null;
let _serverHydrating = false;
let _serverLayoutKeys = [];
let _snapshotTimer = null;
const LOCAL_UPDATED_AT_KEY = "rag.layout.updatedAt.v1";

function _readUpdatedAtMap() {
  try {
    const raw = localStorage.getItem(LOCAL_UPDATED_AT_KEY);
    const parsed = raw ? JSON.parse(raw) : {};
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

function _updatedAtSlot(page, key) {
  const cleanPage = String(page || _serverLayoutPage || "").trim();
  return cleanPage && key ? `${cleanPage}:${key}` : "";
}

function _markLocalUpdated(key) {
  if (_serverHydrating || !key || typeof localStorage === "undefined") return;
  const slot = _updatedAtSlot(_serverLayoutPage, key);
  if (!slot) return;
  const map = _readUpdatedAtMap();
  map[slot] = Date.now();
  try {
    localStorage.setItem(LOCAL_UPDATED_AT_KEY, JSON.stringify(map));
  } catch {}
}

function _localUpdatedAt(page, key) {
  const slot = _updatedAtSlot(page, key);
  if (!slot) return 0;
  const raw = _readUpdatedAtMap()[slot];
  const n = Number(raw || 0);
  return Number.isFinite(n) ? n : 0;
}

function _serverUpdatedAtMs(updatedAt, key) {
  const raw = updatedAt && typeof updatedAt === "object" ? updatedAt[key] : null;
  const n = raw ? Date.parse(raw) : 0;
  return Number.isFinite(n) ? n : 0;
}

function _storageSetRaw(key, value, options = {}) {
  if (typeof localStorage === "undefined") return false;
  if (typeof value === "string") {
    localStorage.setItem(key, value);
  } else {
    localStorage.setItem(key, JSON.stringify(value));
  }
  if (options.markUpdated !== false) _markLocalUpdated(key);
  return true;
}

function _syncServerLayoutKey(key, value) {
  if (!_serverLayoutPage || _serverHydrating || !key) return;
  try {
    fetch(`/api/ui-layout/${encodeURIComponent(_serverLayoutPage)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      keepalive: true,
      body: JSON.stringify({ key, value }),
    }).catch(() => {});
  } catch {}
  _scheduleServerLayoutSnapshot();
}

function _readLocalValueForServer(key) {
  try {
    if (!key || typeof localStorage === "undefined") return undefined;
    const raw = localStorage.getItem(key);
    if (raw == null) return undefined;
    try {
      return JSON.parse(raw);
    } catch {
      return raw;
    }
  } catch {
    return undefined;
  }
}

function _snapshotLocalLayout(keys) {
  const out = {};
  for (const key of keys || []) {
    const value = _readLocalValueForServer(key);
    if (value !== undefined) out[key] = value;
  }
  return out;
}

function _syncServerLayoutSnapshot(page, state) {
  if (!page || !state || Object.keys(state).length === 0) return;
  try {
    fetch(`/api/ui-layout/${encodeURIComponent(page)}/snapshot`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      keepalive: true,
      body: JSON.stringify({ state }),
    }).catch(() => {});
  } catch {}
}

function _scheduleServerLayoutSnapshot() {
  if (!_serverLayoutPage || !_serverLayoutKeys.length || _serverHydrating) return;
  if (_snapshotTimer) clearTimeout(_snapshotTimer);
  _snapshotTimer = setTimeout(() => {
    _snapshotTimer = null;
    _syncServerLayoutSnapshot(
      _serverLayoutPage,
      _snapshotLocalLayout(_serverLayoutKeys),
    );
  }, 500);
}

function _flushServerLayoutSnapshot() {
  if (!_serverLayoutPage || !_serverLayoutKeys.length || typeof localStorage === "undefined") return;
  const state = _snapshotLocalLayout(_serverLayoutKeys);
  if (!Object.keys(state).length) return;
  try {
    fetch(`/api/ui-layout/${encodeURIComponent(_serverLayoutPage)}/snapshot`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      keepalive: true,
      body: JSON.stringify({ state }),
    }).catch(() => {});
  } catch {}
}

if (typeof window !== "undefined") {
  window.addEventListener("pagehide", _flushServerLayoutSnapshot);
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "hidden") _flushServerLayoutSnapshot();
  });
}

export async function hydrateServerLayout(page, keys, options = {}) {
  const cleanPage = String(page || "").trim();
  if (!cleanPage) return false;
  _serverLayoutPage = cleanPage;
  _serverLayoutKeys = Array.isArray(keys) ? [...keys] : [];
  const localWins = {};
  const localDeletes = [];
  const timeoutMs = Number(options.timeoutMs || 1200);
  const controller = typeof AbortController !== "undefined" ? new AbortController() : null;
  const tid = controller ? setTimeout(() => controller.abort(), timeoutMs) : null;
  try {
    const res = await fetch(`/api/ui-layout/${encodeURIComponent(cleanPage)}`, {
      headers: { Accept: "application/json" },
      signal: controller?.signal,
    });
    if (!res.ok) return false;
    const payload = await res.json();
    const state = payload?.state && typeof payload.state === "object" ? payload.state : {};
    const updatedAt = payload?.updated_at && typeof payload.updated_at === "object"
      ? payload.updated_at
      : {};
    const allowed = new Set(Array.isArray(keys) ? keys : []);
    if (Object.keys(state).length === 0 && allowed.size) {
      _syncServerLayoutSnapshot(cleanPage, _snapshotLocalLayout(keys));
    }
    _serverHydrating = true;
    try {
      for (const [key, value] of Object.entries(state)) {
        if (allowed.size && !allowed.has(key)) continue;
        const localValue = _readLocalValueForServer(key);
        const localTs = _localUpdatedAt(cleanPage, key);
        const serverTs = _serverUpdatedAtMs(updatedAt, key);
        if (localValue !== undefined && (!localTs || !serverTs || localTs >= serverTs)) {
          localWins[key] = localValue;
          continue;
        }
        if (localValue === undefined && localTs && (!serverTs || localTs >= serverTs)) {
          localDeletes.push(key);
          continue;
        }
        if (value == null) localStorage.removeItem(key);
        else _storageSetRaw(key, value, { markUpdated: false });
      }
      for (const key of allowed) {
        if (Object.prototype.hasOwnProperty.call(state, key)) continue;
        const localValue = _readLocalValueForServer(key);
        if (localValue !== undefined) localWins[key] = localValue;
      }
    } finally {
      _serverHydrating = false;
    }
    for (const [key, value] of Object.entries(localWins)) {
      _syncServerLayoutKey(key, value);
    }
    for (const key of localDeletes) {
      _syncServerLayoutKey(key, null);
    }
    return true;
  } catch {
    return false;
  } finally {
    if (tid) clearTimeout(tid);
    _serverHydrating = false;
  }
}

export function persistServerLayoutKey(key, value) {
  _syncServerLayoutKey(key, value);
}

export function clearServerLayout(page = _serverLayoutPage) {
  const cleanPage = String(page || "").trim();
  if (!cleanPage) return;
  try {
    fetch(`/api/ui-layout/${encodeURIComponent(cleanPage)}`, {
      method: "DELETE",
      keepalive: true,
    }).catch(() => {});
  } catch {}
}

export function readString(key, fallback = null) {
  try {
    if (!key || typeof localStorage === "undefined") return fallback;
    const raw = localStorage.getItem(key);
    return raw == null ? fallback : raw;
  } catch {
    return fallback;
  }
}

export function readJSON(key, fallback = null) {
  const raw = readString(key, null);
  if (raw == null) return fallback;
  try {
    return JSON.parse(raw);
  } catch {
    return fallback;
  }
}

export function readObject(key, fallback = {}) {
  const parsed = readJSON(key, fallback);
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return fallback;
  return parsed;
}

export function writeJSON(key, value) {
  try {
    if (!key || typeof localStorage === "undefined") return false;
    _storageSetRaw(key, value);
    _syncServerLayoutKey(key, value);
    return true;
  } catch {
    return false;
  }
}

export function writeString(key, value) {
  try {
    if (!key || typeof localStorage === "undefined") return false;
    const s = String(value ?? "");
    _storageSetRaw(key, s);
    _syncServerLayoutKey(key, s);
    return true;
  } catch {
    return false;
  }
}

export function removeKey(key) {
  try {
    if (!key || typeof localStorage === "undefined") return false;
    localStorage.removeItem(key);
    _markLocalUpdated(key);
    _syncServerLayoutKey(key, null);
    return true;
  } catch {
    return false;
  }
}

export function removeKeys(keys) {
  let ok = true;
  for (const key of keys || []) ok = removeKey(key) && ok;
  return ok;
}

export function writeObjectOrRemove(key, objectValue) {
  const value = (objectValue && typeof objectValue === "object" && !Array.isArray(objectValue))
    ? objectValue
    : {};
  if (Object.keys(value).length === 0) return removeKey(key);
  return writeJSON(key, value);
}

export function compactBooleanMap(map) {
  const trimmed = {};
  if (!map || typeof map !== "object") return trimmed;
  for (const [key, value] of Object.entries(map)) {
    if (key && value) trimmed[key] = true;
  }
  return trimmed;
}

export function isValidSizeOverride(override, options = {}) {
  const widths = Array.isArray(options.widths) ? options.widths : ["half", "full"];
  const heights = Array.isArray(options.heights) ? options.heights : ["half", "full", "xl"];
  return (
    override &&
    typeof override === "object" &&
    widths.includes(override.w) &&
    heights.includes(override.h)
  );
}

export function readSizeOverrides(key, options = {}) {
  const raw = readObject(key, {});
  const cleaned = {};
  for (const [id, override] of Object.entries(raw)) {
    if (id && isValidSizeOverride(override, options)) {
      cleaned[id] = { w: override.w, h: override.h };
    }
  }
  return cleaned;
}

export function writeSizeOverrides(key, overrides, options = {}) {
  const cleaned = {};
  if (overrides && typeof overrides === "object") {
    for (const [id, override] of Object.entries(overrides)) {
      if (id && isValidSizeOverride(override, options)) {
        cleaned[id] = { w: override.w, h: override.h };
      }
    }
  }
  writeObjectOrRemove(key, cleaned);
  return cleaned;
}

export function hasSizeOverrides(key, options = {}) {
  return Object.keys(readSizeOverrides(key, options)).length > 0;
}

export function applySizeDataset(element, override) {
  if (!element || !override) return;
  if (override.w) element.dataset.w = override.w;
  if (override.h) element.dataset.h = override.h;
}
