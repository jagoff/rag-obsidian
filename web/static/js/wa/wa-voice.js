// Voice notes — record con MediaRecorder + send PTT.
//
// UX:
// - Botón mic 🎙 al lado del send button.
// - Click: arranca recording. Mientras graba muestra timer `0:12`
//   con barras animadas (canvas waveform fake = bars random pulsing).
// - Click otra vez para parar.
// - Modal preview con transcript del Whisper + botones [enviar / cancelar].
// - Cancelar → descarta. Enviar → POST `/api/wa/voice` con `transcribe_only=false`.
//
// Las API de browser usadas: MediaRecorder, getUserMedia. Sin polyfill —
// Safari iOS 14+ y Chrome ya lo soportan.

const RECORD_OPTS = { mimeType: "audio/webm;codecs=opus" };

let mediaRecorder = null;
let mediaStream = null;
let chunks = [];
let startTs = 0;
let timerInterval = null;
let activeJID = null;
let onSendCallback = null;

const els = {
  btn: null,
  recordBar: null,
  recordTimer: null,
  stopBtn: null,
  cancelBtn: null,
};

export function init({ btnEl, recordBarEl, recordTimerEl, onSend }) {
  els.btn = btnEl;
  els.recordBar = recordBarEl;
  els.recordTimer = recordTimerEl;
  els.stopBtn = document.getElementById("wa-record-stop");
  els.cancelBtn = document.getElementById("wa-record-cancel");
  onSendCallback = onSend;
  if (els.btn) {
    els.btn.addEventListener("click", toggle);
  }
  if (els.stopBtn) {
    els.stopBtn.addEventListener("click", (e) => {
      e.preventDefault();
      stop();
    });
  }
  if (els.cancelBtn) {
    els.cancelBtn.addEventListener("click", (e) => {
      e.preventDefault();
      cancel();
    });
  }
  // Esc cancela, Enter envía (paridad con el modal preview).
  document.addEventListener("keydown", (e) => {
    if (!mediaRecorder || mediaRecorder.state !== "recording") return;
    if (e.key === "Escape") { e.preventDefault(); cancel(); }
    else if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); stop(); }
  });
}

export function setActiveJID(jid) {
  activeJID = jid;
  if (mediaRecorder && mediaRecorder.state === "recording") {
    cancel();
  }
  updateBtnDisabled();
}

function updateBtnDisabled() {
  if (els.btn) els.btn.disabled = !activeJID;
}

function isSupported() {
  return (
    typeof navigator !== "undefined" &&
    navigator.mediaDevices &&
    navigator.mediaDevices.getUserMedia &&
    typeof MediaRecorder !== "undefined"
  );
}

async function toggle() {
  if (!activeJID) {
    window.alert("elegí un chat antes de grabar.");
    return;
  }
  if (!isSupported()) {
    window.alert("este browser no soporta grabar audio.");
    return;
  }
  if (mediaRecorder && mediaRecorder.state === "recording") {
    stop();
  } else {
    await start();
  }
}

async function start() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) {
    console.error("[wa-voice] mic permission denied", e);
    window.alert("no se pudo acceder al micrófono. Revisá los permisos en macOS.");
    return;
  }
  try {
    mediaRecorder = new MediaRecorder(mediaStream, RECORD_OPTS);
  } catch (e) {
    try {
      mediaRecorder = new MediaRecorder(mediaStream);
    } catch (e2) {
      console.error("[wa-voice] MediaRecorder init failed", e2);
      teardownStream();
      window.alert("no se pudo iniciar la grabación.");
      return;
    }
  }
  chunks = [];
  mediaRecorder.ondataavailable = (e) => {
    if (e.data && e.data.size > 0) chunks.push(e.data);
  };
  mediaRecorder.onstop = onStop;
  mediaRecorder.start(250);
  startTs = Date.now();
  showRecording(true);
  timerInterval = setInterval(updateTimer, 200);
}

function stop() {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
  }
}

function cancel() {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.onstop = () => { /* discard */ };
    mediaRecorder.stop();
  }
  chunks = [];
  showRecording(false);
  teardownStream();
}

function teardownStream() {
  if (mediaStream) {
    for (const t of mediaStream.getTracks()) t.stop();
    mediaStream = null;
  }
  if (timerInterval) {
    clearInterval(timerInterval);
    timerInterval = null;
  }
}

function showRecording(on) {
  if (els.btn) els.btn.classList.toggle("recording", on);
  if (els.recordBar) els.recordBar.hidden = !on;
  if (!on && els.recordTimer) els.recordTimer.textContent = "0:00";
}

function updateTimer() {
  if (!els.recordTimer) return;
  const elapsed = Math.floor((Date.now() - startTs) / 1000);
  const m = Math.floor(elapsed / 60);
  const s = String(elapsed % 60).padStart(2, "0");
  els.recordTimer.textContent = `${m}:${s}`;
}

async function onStop() {
  teardownStream();
  showRecording(false);
  if (!chunks.length) return;
  const blob = new Blob(chunks, { type: RECORD_OPTS.mimeType });
  chunks = [];
  await openPreview(blob);
}

async function openPreview(blob) {
  // Modal con waveform stub + transcript + botones.
  const modal = document.createElement("div");
  modal.className = "wa-voice-preview";
  modal.innerHTML = `
    <div class="wa-voice-preview-box">
      <header>voice note · ${(blob.size / 1024).toFixed(1)} KB</header>
      <audio controls src="${URL.createObjectURL(blob)}" preload="metadata"></audio>
      <div class="wa-voice-transcript" id="wa-voice-transcript">
        <span class="wa-voice-loader">◜ transcribiendo con whisper…</span>
      </div>
      <div class="wa-voice-actions">
        <button class="wa-voice-cancel">cancelar</button>
        <button class="wa-voice-send" disabled>enviar voice</button>
      </div>
    </div>
  `;
  document.body.appendChild(modal);

  const close = () => {
    try { modal.remove(); } catch {}
  };

  modal.querySelector(".wa-voice-cancel").addEventListener("click", close);

  // Transcribe-only primero para preview.
  let transcript = "";
  try {
    const fd = new FormData();
    fd.append("jid", activeJID);
    fd.append("transcribe_only", "true");
    fd.append("audio", blob, "voice.webm");
    const r = await fetch("/api/wa/voice", {
      method: "POST",
      credentials: "same-origin",
      body: fd,
    });
    const data = await r.json();
    if (data.ok && data.text) {
      transcript = data.text;
      modal.querySelector("#wa-voice-transcript").textContent = transcript;
    } else {
      modal.querySelector("#wa-voice-transcript").innerHTML =
        `<span class="wa-voice-error">⚠ transcript falló: ${escapeHtml(data.error || data.error_kind || "?")}</span>`;
    }
  } catch (e) {
    console.error("[wa-voice] transcribe failed", e);
    modal.querySelector("#wa-voice-transcript").innerHTML =
      `<span class="wa-voice-error">⚠ ${escapeHtml(e.message)}</span>`;
  } finally {
    modal.querySelector(".wa-voice-send").disabled = false;
  }

  modal.querySelector(".wa-voice-send").addEventListener("click", async () => {
    const btn = modal.querySelector(".wa-voice-send");
    btn.disabled = true;
    btn.textContent = "enviando…";
    if (onSendCallback) {
      onSendCallback({
        jid: activeJID,
        blob,
        transcript,
      });
    }
    try {
      const fd = new FormData();
      fd.append("jid", activeJID);
      fd.append("transcribe_only", "false");
      fd.append("audio", blob, "voice.webm");
      const r = await fetch("/api/wa/voice", {
        method: "POST",
        credentials: "same-origin",
        body: fd,
      });
      const data = await r.json();
      if (!data.ok) {
        btn.textContent = `falló: ${data.error_kind || "?"}`;
        btn.classList.add("failed");
        setTimeout(close, 2500);
        return;
      }
    } catch (e) {
      btn.textContent = `error: ${e.message}`;
      btn.classList.add("failed");
      setTimeout(close, 2500);
      return;
    }
    close();
  });
}

function escapeHtml(s) {
  if (s == null) return "";
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
