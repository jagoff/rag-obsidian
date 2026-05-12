// wzp · Mood Mirror — tonal hint pre-send.
//
// Listens al composer (`#wa-composer-input`). Debounced 800ms post-keystroke
// llama /api/wa/thread/check-tone con el draft actual. Si el server devuelve
// warning con severity >= medium → render banner inline encima del composer.
//
// NO bloquea el send. Hint visual solo — el user decide. Banner se borra
// solo cuando el draft vuelve a estar empty o cuando el send completa.

const DEBOUNCE_MS = 800;
const MIN_LEN = 10;

let _input = null;
let _banner = null;
let _activeJid = null;
let _timer = null;
let _lastCheckedDraft = "";
let _activeWarning = null;

export function init() {
  _input = document.getElementById("wa-composer-input");
  if (!_input) return;
  mountBanner();
  _input.addEventListener("input", onInput);
  // Hide cuando el send completa (input se limpia).
  _input.addEventListener("blur", () => {
    if (!_input.value.trim()) hide();
  });
}

export function setActiveJid(jid) {
  _activeJid = jid;
  hide();
}

function mountBanner() {
  const composer = document.getElementById("wa-composer");
  if (!composer) return;
  _banner = document.createElement("div");
  _banner.id = "wa-mood-check-banner";
  _banner.className = "wa-mood-check-banner";
  _banner.hidden = true;
  _banner.innerHTML = `
    <span class="wa-mood-check-icon" aria-hidden="true">💭</span>
    <span class="wa-mood-check-msg" id="wa-mood-check-msg"></span>
    <button class="wa-mood-check-close" type="button" aria-label="ocultar">✕</button>
  `;
  composer.parentNode.insertBefore(_banner, composer);
  _banner.querySelector(".wa-mood-check-close").addEventListener("click", () => {
    _banner.hidden = true;
    _activeWarning = null;
  });
}

function onInput() {
  if (_timer) clearTimeout(_timer);
  const text = (_input.value || "").trim();
  if (text.length < MIN_LEN) {
    hide();
    return;
  }
  _timer = setTimeout(() => checkNow(text), DEBOUNCE_MS);
}

async function checkNow(text) {
  if (!_activeJid || text === _lastCheckedDraft) return;
  _lastCheckedDraft = text;
  try {
    const r = await fetch("/api/wa/thread/check-tone", {
      method: "POST",
      headers: { "content-type": "application/json" },
      credentials: "same-origin",
      body: JSON.stringify({ jid: _activeJid, draft: text }),
    });
    if (!r.ok) return;
    const data = await r.json();
    const warning = data.warning;
    if (!warning) {
      hide();
      return;
    }
    show(warning);
  } catch (e) {
    // silent — tonal hint es nice-to-have
  }
}

function show(warning) {
  if (!_banner) return;
  _activeWarning = warning;
  const msg = document.getElementById("wa-mood-check-msg");
  const icon = _banner.querySelector(".wa-mood-check-icon");
  if (msg) msg.textContent = warning.message || "";
  if (icon) icon.textContent = warning.icon || "💭";
  _banner.className = `wa-mood-check-banner sev-${warning.severity || "low"}`;
  _banner.hidden = false;
}

function hide() {
  if (!_banner) return;
  _banner.hidden = true;
  _activeWarning = null;
  _lastCheckedDraft = "";
}
