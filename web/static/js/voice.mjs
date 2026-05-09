// voice.mjs — TTS playback via /api/tts (Mónica voice).
// El STT (whisper) no está conectado al browser todavía — hay un stub
// en mobile-ui.mjs para el botón mic.

import { ttsEnabled } from "./settings.mjs";

// Referencia al <audio> en vuelo para abortar si el user manda otro turno.
let currentAudio = null;

/**
 * Reproduce el texto vía /api/tts si ttsEnabled está activo.
 * Aborta cualquier audio en vuelo antes de arrancar el nuevo.
 */
export async function speak(text) {
  // Re-leer ttsEnabled en el momento de la llamada (puede haber cambiado).
  const { ttsEnabled: enabled } = await import("./settings.mjs");
  if (!enabled || !text || !text.trim()) return;
  if (currentAudio) { currentAudio.pause(); currentAudio = null; }
  try {
    const res = await fetch("/api/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text.slice(0, 1500) }),
    });
    if (!res.ok) return;
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);
    audio.addEventListener("ended", () => URL.revokeObjectURL(url));
    currentAudio = audio;
    audio.play().catch(() => {});
  } catch {}
}

/** Detiene la reproducción en curso (si la hay). */
export function stopAudio() {
  if (currentAudio) {
    currentAudio.pause();
    currentAudio = null;
  }
}
