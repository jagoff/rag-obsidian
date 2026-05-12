// wzp · Voz Espejo — "enviar como voz" desde el composer.
//
// El user escribe texto, hace click en 🎙 → server genera OGG/Opus con
// `say -v Mónica` + ffmpeg y manda como PTT al bridge. La burbuja del
// receptor sale como audio PTT (no como texto + audio adjunto).
//
// Trade-off: latencia 2-5s (say synthesis + ffmpeg encode + bridge send).
// Por eso el botón muestra estado pending mientras procesa.

let _btn = null;
let _input = null;
let _activeJid = null;
let _healthOk = false;
let _healthMissing = [];

export function init() {
  _btn = document.getElementById("wa-tts-btn");
  _input = document.getElementById("wa-composer-input");
  if (!_btn || !_input) return;
  _btn.addEventListener("click", onClick);
  // Toggle disabled según contenido del input.
  _input.addEventListener("input", updateState);
  // Pre-flight check: si `say` o `ffmpeg` no están instalados, disable
  // el botón con tooltip diagnóstico — evita confusión del primer click
  // que devuelve "tts_failed" sin contexto.
  checkHealth();
  updateState();
}

async function checkHealth() {
  try {
    const r = await fetch("/api/wa/voice/healthcheck", { credentials: "same-origin" });
    if (!r.ok) return;
    const data = await r.json();
    _healthOk = !!data.ok;
    _healthMissing = data.missing || [];
    if (!_healthOk && _btn) {
      _btn.classList.add("unavailable");
      _btn.title = `Voz Espejo no disponible · falta: ${_healthMissing.join(", ")}`;
    }
  } catch (e) {
    // silent — si el healthcheck falla, dejamos pasar el click normal
  }
}

export function setActiveJid(jid) {
  _activeJid = jid;
  updateState();
}

function updateState() {
  if (!_btn) return;
  const text = (_input?.value || "").trim();
  _btn.disabled = !_healthOk || !_activeJid || text.length < 2;
}

async function onClick(ev) {
  ev.preventDefault();
  ev.stopPropagation();
  if (!_activeJid) return;
  const text = (_input?.value || "").trim();
  if (text.length < 2) return;
  if (text.length > 2000) {
    showError("texto demasiado largo (>2000 chars)");
    return;
  }

  setPending(true);
  try {
    const r = await fetch("/api/wa/send_voice", {
      method: "POST",
      headers: { "content-type": "application/json" },
      credentials: "same-origin",
      body: JSON.stringify({ jid: _activeJid, text }),
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok || !data.ok) {
      showError(data.error_kind || `error ${r.status}`);
      return;
    }
    // Limpiar input post-send exitoso.
    _input.value = "";
    _input.dispatchEvent(new Event("input", { bubbles: true }));
    flashSuccess();
  } catch (e) {
    showError("network");
  } finally {
    setPending(false);
  }
}

function setPending(pending) {
  if (!_btn) return;
  _btn.classList.toggle("pending", pending);
  _btn.disabled = pending;
  if (pending) _btn.title = "generando voz…";
  else _btn.title = "enviar como voz · TTS Mónica + opus PTT";
}

function flashSuccess() {
  if (!_btn) return;
  _btn.classList.add("sent");
  setTimeout(() => _btn.classList.remove("sent"), 800);
}

function showError(kind) {
  if (!_btn) return;
  _btn.classList.add("failed");
  _btn.title = `falló: ${kind}`;
  setTimeout(() => {
    _btn.classList.remove("failed");
    _btn.title = "enviar como voz · TTS Mónica + opus PTT";
  }, 2500);
}
