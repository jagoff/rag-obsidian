/*
 * Registra el service worker + maneja el prompt de "Add to Home Screen".
 *
 * Se carga como <script defer> desde home.html / index.html / dashboard.html
 * para que los 3 HTML no dupliquen código (principio DRY > micro-optim).
 *
 * Por qué defer: queremos que el SW se registre rápido (así el browser
 * puede cachear el shell en el background de la primera visita), pero
 * no queremos bloquear el primer paint. `defer` ejecuta después del DOM
 * parse pero antes de DOMContentLoaded — momento justo.
 *
 * iOS-specific: Safari iOS sí soporta service workers desde iOS 11.3,
 * pero con varias limitaciones vs Chrome Android:
 *  - No hay `beforeinstallprompt` (el user tiene que hacer share →
 *    "Add to Home Screen" a mano; no podemos triggerar el flow).
 *  - Push notifications sí desde iOS 16.4 pero SOLO cuando la PWA
 *    está instalada en home screen. Las webapp en tab no reciben push.
 *  - El SW se mata de forma mucho más agresiva que en Chrome.
 *
 * Por eso para iOS no intentamos el `beforeinstallprompt` (sería un
 * no-op); en su lugar, mostramos una banner con el gesto manual si
 * detectamos iOS + no-standalone.
 */

(function () {
  "use strict";

  if (!("serviceWorker" in navigator)) {
    // Browser sin SW (Safari <11.3, algunos browsers alternativos).
    // No pasa nada — la web sigue andando vía network normal.
    return;
  }

  window.addEventListener("load", function () {
    // updateViaCache: "none" fuerza al browser a revalidar el sw.js
    // con el server en cada carga (en vez de usar HTTP cache). El server
    // también manda Cache-Control: no-cache para reforzar — cinturón y
    // tiradores, así un bump de CACHE_VERSION en sw.js se propaga en la
    // próxima visita, no en 24h.
    navigator.serviceWorker
      .register("/sw.js", { scope: "/", updateViaCache: "none" })
      .then(function (reg) {
        // Listener para cuando aparece una versión nueva en background.
        // Si el user tiene la tab abierta durante un deploy, el nuevo
        // SW instala en paralelo y queda `waiting`. Acá lo activamos
        // al toque con SKIP_WAITING → la próxima navegación sirve el
        // shell actualizado sin que el user tenga que cerrar la tab.
        reg.addEventListener("updatefound", function () {
          var nw = reg.installing;
          if (!nw) return;
          nw.addEventListener("statechange", function () {
            if (
              nw.state === "installed" &&
              navigator.serviceWorker.controller
            ) {
              // Nueva versión lista + había un SW viejo controlando →
              // upgrade seamless.
              nw.postMessage({ type: "SKIP_WAITING" });
            }
          });
        });
      })
      .catch(function (err) {
        // No mostramos error al user — registrar SW es best-effort,
        // la app anda igual sin él. Log para debug en DevTools.
        console.warn("[pwa] SW register failed:", err);
      });

    // Auto-reload cuando el SW activo cambia (post skipWaiting) — el
    // browser ya tiene el shell nuevo cacheado, recargamos para que el
    // user vea los assets actualizados sin tener que refrescar a mano.
    // El comentario previo decía "es agresivo, dejamos que el user
    // refresque cuando quiera" — pero esto causaba bugs invisibles:
    // fixes deployados al frontend no llegaban al user que tenía la
    // tab abierta porque el JS viejo seguía corriendo aunque el shell
    // ya estuviera actualizado en cache. Mejor un reload silencioso
    // (~200ms) que dejar fixes sin propagar.
    let _swRefreshing = false;
    navigator.serviceWorker.addEventListener("controllerchange", () => {
      if (_swRefreshing) return;
      _swRefreshing = true;
      window.location.reload();
    });
  });

  // ── iOS add-to-home-screen hint ────────────────────────────────────
  // En iOS no existe `beforeinstallprompt`; el user tiene que apretar
  // compartir (↑ con flecha) → "Agregar a pantalla de inicio". Mostramos
  // un banner leve la primera vez que un iPhone abre la web sin estar
  // en standalone, con opción a dismiss persistente.
  //
  // Detección:
  //   navigator.standalone === true  →  ya está en home screen (iOS)
  //   matchMedia("(display-mode: standalone)") → Android / desktop PWA
  //
  // El banner es puro vanilla DOM + inline styles para no depender
  // del style.css del caller — así el mismo snippet sirve para home
  // (colorful) y chat (dark) sin que clashee.
  try {
    var isIOS =
      /iphone|ipod|ipad/i.test(navigator.userAgent) &&
      !window.MSStream;
    var isStandalone =
      window.matchMedia("(display-mode: standalone)").matches ||
      window.navigator.standalone === true;
    var dismissed =
      (function () {
        try {
          return localStorage.getItem("rag-pwa-ios-dismissed") === "1";
        } catch (e) {
          return false;
        }
      })();

    if (isIOS && !isStandalone && !dismissed) {
      window.addEventListener("load", function () {
        // Esperamos 3s post-load para no meternos con el primer paint.
        setTimeout(showIosBanner, 3000);
      });
    }
  } catch (_) {
    // Si matchMedia/localStorage fallan — fine, skippeamos el banner.
  }

  function showIosBanner() {
    if (document.getElementById("rag-pwa-ios-banner")) return;
    var div = document.createElement("div");
    div.id = "rag-pwa-ios-banner";
    div.setAttribute("role", "dialog");
    div.setAttribute("aria-label", "Instalar rag en pantalla de inicio");
    div.style.cssText = [
      "position:fixed",
      "left:12px",
      "right:12px",
      "bottom:calc(12px + env(safe-area-inset-bottom, 0px))",
      "background:#26262c",
      "color:#ececed",
      "border:1px solid #3e3e46",
      "border-radius:14px",
      "padding:14px 16px 12px",
      "font:13px/1.4 -apple-system,BlinkMacSystemFont,'SF Pro Text',system-ui,sans-serif",
      "box-shadow:0 8px 24px rgba(0,0,0,.45)",
      "z-index:9999",
      "max-width:420px",
      "margin:0 auto",
      "animation:ragPwaSlideUp .3s ease-out",
    ].join(";");
    div.innerHTML =
      '<div style="display:flex;align-items:flex-start;gap:10px">' +
      '  <div style="flex:1;min-width:0">' +
      '    <div style="font-weight:600;margin-bottom:4px">Instalá rag como app</div>' +
      '    <div style="color:#a0a0a6;font-size:12px">' +
      '      Tocá <span aria-hidden="true" style="display:inline-block;transform:translateY(2px)">' +
      '        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#79c0ff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
      '          <path d="M12 3v12"/><path d="M7 8l5-5 5 5"/><path d="M5 15v4a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-4"/>' +
      '        </svg>' +
      '      </span>' +
      '      (Compartir) → <b>Agregar a pantalla de inicio</b>.' +
      '    </div>' +
      '  </div>' +
      '  <button type="button" data-dismiss aria-label="Cerrar" ' +
      '    style="background:transparent;border:0;color:#7a7a82;font-size:22px;line-height:1;cursor:pointer;padding:0 4px;margin:-4px -6px 0 0">×</button>' +
      '</div>';
    var style = document.createElement("style");
    style.textContent =
      "@keyframes ragPwaSlideUp{from{transform:translateY(120%);opacity:0}to{transform:none;opacity:1}}";
    document.head.appendChild(style);
    document.body.appendChild(div);
    div.querySelector("[data-dismiss]").addEventListener("click", function () {
      div.remove();
      try {
        localStorage.setItem("rag-pwa-ios-dismissed", "1");
      } catch (e) {}
    });
  }
})();
