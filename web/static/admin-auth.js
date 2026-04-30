/**
 * admin-auth.js — gestión del Bearer token para los 8 endpoints admin.
 *
 * Boot:
 *   1. fetch GET /api/admin/token (solo accesible desde localhost)
 *   2. cache en memoria (NO localStorage — el token no debería persistir
 *      en almacenamiento del browser que es leíble por extensiones)
 *   3. monkey-patch global de fetch() — toda request a un path admin
 *      automáticamente recibe Authorization: Bearer <token>.
 *
 * Si la red devuelve 403 (no-localhost), el token queda null y los
 * endpoints admin van a fallar con 401 desde el server. Es el
 * comportamiento esperado: un user en LAN/tunnel NO puede invocarlos.
 */

(function () {
  let adminToken = null;
  let tokenFetchPromise = null;

  // Lista exacta de los endpoints admin protegidos por _require_admin_token.
  // Match por prefijo: una request a /api/auto-fix-devin/anything también
  // recibe el token (defense in depth — el server decide qué path requiere).
  const ADMIN_PATHS = [
    "/api/auto-fix-devin",
    "/api/auto-fix",
    "/api/reindex",
    "/api/ollama/restart",
    "/api/ollama/unload",
    "/api/status/action",
    "/api/diagnose-error/execute",
    "/api/chat/model",
  ];

  function isAdminPath(url) {
    // url puede ser absoluta (https://...) o relativa (/api/...).
    // Extraer el pathname para comparación robusta.
    let path;
    try {
      path = new URL(url, window.location.origin).pathname;
    } catch (_e) {
      path = String(url || "");
    }
    return ADMIN_PATHS.some((p) => path === p || path.startsWith(p + "/"));
  }

  async function loadToken() {
    if (adminToken !== null) return adminToken;
    if (tokenFetchPromise) return tokenFetchPromise;
    tokenFetchPromise = (async () => {
      try {
        const resp = await origFetch("/api/admin/token", {
          method: "GET",
          credentials: "same-origin",
        });
        if (resp.ok) {
          const data = await resp.json();
          adminToken = data.token || null;
        } else {
          // 403: el frontend está en LAN/tunnel. Token queda null.
          adminToken = null;
        }
      } catch (_e) {
        adminToken = null;
      }
      return adminToken;
    })();
    return tokenFetchPromise;
  }

  // Monkey-patch fetch() global — inyecta el header solo en admin paths.
  const origFetch = window.fetch.bind(window);
  window.fetch = async function (input, init) {
    const url = typeof input === "string" ? input : (input && input.url) || "";
    if (!isAdminPath(url)) {
      return origFetch(input, init);
    }
    // Asegurar token cargado.
    const token = await loadToken();
    if (!token) {
      // Sin token: dejar pasar el request — el server responde 401, que el
      // caller original puede manejar mostrando "feature solo disponible
      // desde localhost".
      return origFetch(input, init);
    }
    const opts = { ...(init || {}) };
    opts.headers = new Headers(opts.headers || {});
    if (!opts.headers.has("Authorization")) {
      opts.headers.set("Authorization", `Bearer ${token}`);
    }
    return origFetch(input, opts);
  };

  // Pre-fetch del token al boot — evita el primer click en panic button
  // pague la latencia del round-trip extra.
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", loadToken, { once: true });
  } else {
    loadToken();
  }

  // Expose para debug en consola.
  window.__ragAdminAuth = { loadToken, isAdminPath };
})();
