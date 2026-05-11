// Render de media inbound + upload outbound.
//
// Inbound:
// - image/* → <img loading="lazy"> con onclick → modal fullscreen.
// - audio/* → <audio controls preload="metadata">.
// - video/* → <video controls preload="metadata"> con poster lazy.
// - todo otro → icono + filename + download link.
//
// Outbound: el composer hostea drag+drop + paste-from-clipboard;
// envía via POST /api/wa/send_media. La lógica de upload propiamente
// dicha está acá; el composer solo llama `uploadMedia()`.

import { fetchChats } from "./wa-api.js";  // noqa — ya estaba en el module graph

const SUPPORTED_INBOUND = ["image", "audio", "video", "document", "sticker"];

export function renderInto(container, msg) {
  if (!container || !msg) return null;
  const media = (msg.media_type || "").toLowerCase();
  const filename = msg.filename || "archivo";
  if (!media) return null;

  const url = `/api/wa/media/${encodeURIComponent(msg.id)}?jid=${encodeURIComponent(msg.chat_jid || msg.jid || "")}`;

  if (media === "image" || media === "sticker") {
    const wrap = document.createElement("div");
    wrap.className = "wa-media wa-media-image";
    const img = document.createElement("img");
    img.loading = "lazy";
    img.decoding = "async";
    img.src = url;
    img.alt = filename;
    img.addEventListener("click", () => openLightbox(url, filename));
    img.addEventListener("error", () => {
      wrap.classList.add("missing");
      wrap.innerHTML = `<div class="wa-media-missing">▢ imagen no disponible<br><small>${escapeHtml(filename)}</small></div>`;
    });
    wrap.appendChild(img);
    container.appendChild(wrap);
    return wrap;
  }

  if (media === "audio") {
    const wrap = document.createElement("div");
    wrap.className = "wa-media wa-media-audio";
    const audio = document.createElement("audio");
    audio.controls = true;
    audio.preload = "metadata";
    audio.src = url;
    wrap.appendChild(audio);
    // Toggle de transcript: fetch on-demand al endpoint
    // /api/wa/voice/transcript/{msg_id}. Si hay cache lo trae instant;
    // si no, corre whisper y devuelve (~3-8s típico).
    const trBtn = document.createElement("button");
    trBtn.type = "button";
    trBtn.className = "wa-voice-transcript-btn";
    trBtn.textContent = "📝 ver transcript";
    trBtn.dataset.state = "idle";
    const trBox = document.createElement("div");
    trBox.className = "wa-voice-transcript-box";
    trBox.hidden = true;
    trBtn.addEventListener("click", async () => {
      if (trBtn.dataset.state === "loading") return;
      if (trBtn.dataset.state === "done") {
        // toggle hide/show
        trBox.hidden = !trBox.hidden;
        trBtn.textContent = trBox.hidden ? "📝 ver transcript" : "📝 ocultar transcript";
        return;
      }
      trBtn.dataset.state = "loading";
      trBtn.textContent = "📝 transcribiendo…";
      try {
        const chatJid = msg.chat_jid || msg.jid || "";
        const resp = await fetch(
          `/api/wa/voice/transcript/${encodeURIComponent(msg.id)}?jid=${encodeURIComponent(chatJid)}`
        );
        const data = await resp.json();
        if (data.ok && data.text) {
          trBox.textContent = data.text;
          trBox.hidden = false;
          trBtn.dataset.state = "done";
          trBtn.textContent = "📝 ocultar transcript";
        } else {
          trBox.textContent = data.error ? `error: ${data.error}` : "sin transcript";
          trBox.hidden = false;
          trBtn.dataset.state = "error";
          trBtn.textContent = "📝 reintentar";
        }
      } catch (e) {
        trBox.textContent = `error de red: ${e.message}`;
        trBox.hidden = false;
        trBtn.dataset.state = "error";
        trBtn.textContent = "📝 reintentar";
      }
    });
    wrap.appendChild(trBtn);
    wrap.appendChild(trBox);
    container.appendChild(wrap);
    return wrap;
  }

  if (media === "video") {
    const wrap = document.createElement("div");
    wrap.className = "wa-media wa-media-video";
    const video = document.createElement("video");
    video.controls = true;
    video.preload = "metadata";
    video.src = url;
    wrap.appendChild(video);
    container.appendChild(wrap);
    return wrap;
  }

  // PDF inline preview — los browsers modernos renderean PDFs
  // nativamente via `<embed type="application/pdf">`. Mostramos la
  // primera página en max-height 320px con scroll interno, header
  // con filename + botones "abrir" / "descargar". Click sobre el
  // embed da los controls nativos del PDF viewer del browser
  // (Chrome/Safari). Si no es PDF, cae al fallback document link.
  const isPdf = /\.pdf$/i.test(filename || "");
  if (isPdf) {
    const wrap = document.createElement("div");
    wrap.className = "wa-media wa-media-pdf";
    wrap.innerHTML = `
      <div class="wa-media-pdf-head">
        <span class="wa-media-pdf-icon" aria-hidden="true">📄</span>
        <span class="wa-media-pdf-name" title="${escapeHtml(filename)}">${escapeHtml(filename)}</span>
        <a class="wa-media-pdf-act" href="${escapeHtml(url)}" target="_blank" rel="noopener" title="Abrir en nueva pestaña">↗</a>
        <a class="wa-media-pdf-act" href="${escapeHtml(url)}" download="${escapeHtml(filename)}" title="Descargar">↓</a>
      </div>
      <embed class="wa-media-pdf-embed" src="${escapeHtml(url)}#view=FitH&toolbar=0" type="application/pdf">
    `;
    container.appendChild(wrap);
    return wrap;
  }

  // document / fallback (non-PDF: docx, xlsx, zip, etc.)
  const wrap = document.createElement("a");
  wrap.className = "wa-media wa-media-doc";
  wrap.href = url;
  wrap.download = filename;
  wrap.target = "_blank";
  wrap.rel = "noopener";
  wrap.innerHTML = `
    <span class="wa-media-doc-icon" aria-hidden="true">◰</span>
    <span class="wa-media-doc-name">${escapeHtml(filename)}</span>
    <span class="wa-media-doc-action">↓</span>
  `;
  container.appendChild(wrap);
  return wrap;
}

// Lightbox modal — overlay con la imagen full + click outside cierra.
let lightboxEl = null;

function openLightbox(url, alt) {
  closeLightbox();
  lightboxEl = document.createElement("div");
  lightboxEl.className = "wa-lightbox";
  lightboxEl.setAttribute("role", "dialog");
  lightboxEl.setAttribute("aria-modal", "true");
  lightboxEl.innerHTML = `
    <button class="wa-lightbox-close" aria-label="cerrar">✕</button>
    <img class="wa-lightbox-img" src="${url}" alt="${escapeHtml(alt)}">
  `;
  const close = () => closeLightbox();
  lightboxEl.addEventListener("click", (e) => {
    if (e.target === lightboxEl || e.target.classList.contains("wa-lightbox-close")) {
      close();
    }
  });
  const esc = (e) => {
    if (e.key === "Escape") {
      close();
      document.removeEventListener("keydown", esc);
    }
  };
  document.addEventListener("keydown", esc);
  document.body.appendChild(lightboxEl);
}

function closeLightbox() {
  if (lightboxEl) {
    lightboxEl.remove();
    lightboxEl = null;
  }
}

// Upload: multipart POST. El caller (composer) usa el File del input
// o del DataTransfer del paste/drop.
export async function uploadMedia(jid, file, { caption = "", replyToId = "" } = {}) {
  if (!jid || !file) throw new Error("uploadMedia: jid + file requeridos");
  const fd = new FormData();
  fd.append("jid", jid);
  fd.append("caption", caption);
  if (replyToId) fd.append("reply_to_id", replyToId);
  fd.append("file", file, file.name || "upload.bin");
  const r = await fetch("/api/wa/send_media", {
    method: "POST",
    credentials: "same-origin",
    body: fd,
  });
  if (!r.ok) {
    let detail = "";
    try { detail = (await r.json()).detail || ""; } catch {}
    throw new Error(`upload ${r.status}${detail ? ": " + detail : ""}`);
  }
  return r.json();
}

function escapeHtml(s) {
  if (s == null) return "";
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
