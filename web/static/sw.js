/* obsidian-rag service worker.
 *
 * Servido desde /sw.js (no /static/sw.js) para que el scope pueda ser "/"
 * sin tener que mandar el header `Service-Worker-Allowed`. Una SW sólo
 * puede controlar URLs bajo el path donde vive — si la servieramos desde
 * /static/sw.js el scope sería /static/ y no podríamos interceptar
 * /chat /dashboard ni /.
 *
 * Estrategia:
 *   - App shell (/, /chat, /dashboard): stale-while-revalidate — sirve
 *     la última copia cacheada instant y refresca en background. Primer
 *     paint del "add to home screen" = instantáneo aunque haya 4G lento.
 *   - /static/**: cache-first con versionado por URL (?v=...) — si el
 *     HTML manda una URL nueva, el cache miss fuerza el fetch. Si viene
 *     con la misma URL, sirvo del cache siempre.
 *   - /api/**: network-only. No cacheamos respuestas del RAG porque
 *     son dinámicas y privadas; además SSE/streaming de /api/chat no
 *     se puede cachear sin romperlo.
 *   - Navigation preload: ON. El browser arranca el fetch al `GET /chat`
 *     apenas ve la navegación, en paralelo con el boot del SW, así el
 *     primer request no paga la penalidad del SW cold-start (~50-100ms).
 *
 * Historia / rollback:
 *   Si el SW rompe algo (por ejemplo, cache stale que no se invalida),
 *   desregistrar con: navigator.serviceWorker.getRegistrations().then(
 *   rs => rs.forEach(r => r.unregister()))  en DevTools. O bumpear
 *   CACHE_VERSION abajo y hacer hard-reload — el SW nuevo limpia los
 *   caches viejos en `activate`.
 */

/* eslint-env serviceworker */

// Bump cuando cambie el shell / la estrategia. El activate handler borra
// todo cache que no matchee esta versión, así no se acumulan huérfanos.
const CACHE_VERSION = "rag-pwa-v12-2026-04-26-auto-fix";
const SHELL_CACHE = `${CACHE_VERSION}-shell`;
const STATIC_CACHE = `${CACHE_VERSION}-static`;

// Lista de URLs del shell que queremos pre-cachear en install. Si alguno
// falla, el SW no se instala (por diseño — si el shell no está, la PWA
// no sirve offline). Por eso la lista es corta: HTML de las 3 páginas.
const SHELL_URLS = [
  "/",
  "/chat",
  "/dashboard",
  "/manifest.webmanifest",
  "/static/pwa/icon-192.png",
  "/static/pwa/icon-512.png",
  "/static/pwa/apple-touch-icon.png",
];

self.addEventListener("install", (event) => {
  // skipWaiting: el SW nuevo reemplaza al viejo apenas termina de
  // instalarse, sin esperar a que todas las tabs se cierren. Trade-off:
  // si el shell cambió de forma incompatible con el JS cacheado, el
  // user podría ver una página rota por unos segundos hasta que refresque.
  // Mitigación: versionar los static assets con ?v= (ya lo hace el HTML).
  self.skipWaiting();

  event.waitUntil(
    (async () => {
      const cache = await caches.open(SHELL_CACHE);
      // addAll es all-or-nothing; si un URL falla el install falla.
      // En dev con el server apagado esto se salta por el catch del
      // navegador — no propaga error al usuario.
      await cache.addAll(SHELL_URLS);
    })()
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      // Navigation preload: el browser arranca el network fetch para
      // navigation requests mientras el SW está booting, y nos entrega
      // la response como `event.preloadResponse`. Saves ~50-100ms en
      // cold-start cuando el user abre la PWA desde el home screen.
      if (self.registration.navigationPreload) {
        try {
          await self.registration.navigationPreload.enable();
        } catch (_) {
          // Safari iOS <16.4 no soporta — fine, sin preload.
        }
      }

      // Reclama control de todas las tabs abiertas ahora mismo, sin
      // esperar a que naveguen. Empareja con skipWaiting() arriba.
      await self.clients.claim();

      // GC de caches viejos: cualquier cache que no matchee la version
      // actual se borra. Esto evita que un deploy con muchos SW updates
      // deje acumulados 5+ caches en el device del user.
      const names = await caches.keys();
      await Promise.all(
        names
          .filter((n) => !n.startsWith(CACHE_VERSION))
          .map((n) => caches.delete(n))
      );
    })()
  );
});

/**
 * Helper: ¿esta request es para el app shell (una de las 3 páginas HTML)?
 * Detectamos por el `Accept: text/html` + método GET, más confiable que
 * matchear por path (que tendría que coincidir con /chat, /chat?foo=bar,
 * /chat#session, etc.).
 */
function isShellRequest(request) {
  return (
    request.method === "GET" &&
    request.mode === "navigate"
  );
}

/**
 * Helper: ¿esto es un static asset (bajo /static/)?
 */
function isStaticAsset(url) {
  return url.pathname.startsWith("/static/");
}

/**
 * Helper: ¿esto es una API call que NO queremos cachear nunca?
 */
function isApi(url) {
  return url.pathname.startsWith("/api/");
}

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);

  // Solo mismo origen — no interceptamos fetches a CDNs externos
  // (vendor/marked está vendorado, no hay externals). Si en algún
  // futuro se cargan recursos cross-origin, dejarlos pasar intactos.
  if (url.origin !== self.location.origin) {
    return; // bypass
  }

  // API: network-only siempre, con respuesta de error offline amable
  // si el fetch falla (el client decide si retrya).
  if (isApi(url)) {
    // No llamamos `event.respondWith` → el browser hace su fetch normal.
    // Esto deja intactos los SSE (streaming) y los POST al /api/chat,
    // que se romperían si los agarrara el SW.
    return;
  }

  // Shell (navigation): stale-while-revalidate con navigation preload.
  if (isShellRequest(event.request)) {
    event.respondWith(shellStrategy(event));
    return;
  }

  // Static: cache-first con fallback a red, actualiza el cache oportunistamente.
  if (isStaticAsset(url)) {
    event.respondWith(staticStrategy(event.request));
    return;
  }

  // Todo lo demás (el manifest, el sw.js mismo, otros): passthrough.
});

async function shellStrategy(event) {
  const cache = await caches.open(SHELL_CACHE);

  // 1) Intento usar la preload response si el browser ya empezó el fetch.
  const preload = event.preloadResponse ? await event.preloadResponse : null;

  // 2) Paralelo: cached copy (puede no existir) + network fetch fresh.
  const cached = await cache.match(event.request, { ignoreSearch: true });
  const networkFetch = (async () => {
    try {
      const resp = preload || (await fetch(event.request));
      // Solo cacheamos 2xx. Errores 4xx/5xx no los queremos pisar el shell.
      if (resp && resp.ok && resp.type !== "opaque") {
        // clone() porque el body del Response es single-use.
        cache.put(event.request, resp.clone()).catch(() => {});
      }
      return resp;
    } catch (err) {
      return null; // offline o DNS fail
    }
  })();

  // 3) Si hay copia en cache: devolverla ya y dejar que el fetch se
  //    complete en background (stale-while-revalidate). Si no hay,
  //    esperar al network.
  if (cached) {
    // Fire-and-forget del networkFetch para revalidar.
    event.waitUntil(networkFetch);
    return cached;
  }
  const fromNet = await networkFetch;
  if (fromNet) return fromNet;

  // 4) Offline sin cache: página de error mínima inline (no tenemos
  //    una /offline.html dedicada; mejor evitar el 404 feo del browser).
  return new Response(
    `<!doctype html><meta charset=utf-8><title>rag — offline</title>` +
      `<style>body{background:#1a1a1f;color:#ececed;font:14px/1.4 -apple-system,SF Mono,Menlo,monospace;display:grid;place-items:center;min-height:100vh;margin:0}main{text-align:center;max-width:320px;padding:24px}h1{font-size:18px;margin:0 0 8px}p{color:#a0a0a6;margin:8px 0}</style>` +
      `<main><h1>rag · offline</h1><p>No hay conexión al servidor.</p><p>Volvé a intentar cuando tengas red.</p></main>`,
    {
      status: 200,
      headers: { "Content-Type": "text/html; charset=utf-8" },
    }
  );
}

async function staticStrategy(request) {
  const cache = await caches.open(STATIC_CACHE);
  const cached = await cache.match(request);
  if (cached) {
    // Stale refresh en background para que, si cambiaron el ?v=, la
    // próxima vez ya esté fresh. No bloqueamos el request actual.
    fetch(request)
      .then((resp) => {
        if (resp && resp.ok && resp.type !== "opaque") {
          cache.put(request, resp.clone()).catch(() => {});
        }
      })
      .catch(() => {});
    return cached;
  }

  try {
    const resp = await fetch(request);
    if (resp && resp.ok && resp.type !== "opaque") {
      cache.put(request, resp.clone()).catch(() => {});
    }
    return resp;
  } catch (_) {
    // Sin cache ni red — devolver un 503 mudo. El browser va a mostrar
    // el broken-image default si era una imagen, o vacío si era JS/CSS.
    return new Response("", { status: 503, statusText: "Offline" });
  }
}

// Permite forzar un refresh del SW desde la página (al clickear el
// banner "nueva versión disponible" si alguna vez lo agregamos). El
// client manda `{type: 'SKIP_WAITING'}` y el SW se activa ya.
self.addEventListener("message", (event) => {
  if (event.data && event.data.type === "SKIP_WAITING") {
    self.skipWaiting();
  }
});
