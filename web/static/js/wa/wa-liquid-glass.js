// Liquid Glass — efectos dinámicos inspirados en Apple WWDC25:
// 1. Specular spotlight: el mouse pintea una luz radial sutil que
//    sigue el cursor sobre cualquier `.wa-msg` (CSS lee `--lq-x/--lq-y`).
// 2. Tinted glass del thread header: extraemos el color dominante del
//    avatar del chat activo y lo aplicamos como overlay del header
//    (CSS variables `--tint-r/--tint-g/--tint-b`).
//
// Modulo pasivo — wire en init(). No bloquea nada si falla.

const _bodyEl = () => document.getElementById("wa-thread-body");
const _headerEl = () => document.querySelector(".wa-thread-header");

export function init() {
  const body = _bodyEl();
  if (body) {
    body.addEventListener("mousemove", onMouseMove, { passive: true });
  }
}

function onMouseMove(ev) {
  const target = ev.target.closest?.(".wa-msg");
  if (!target) return;
  const r = target.getBoundingClientRect();
  const x = ((ev.clientX - r.left) / r.width) * 100;
  const y = ((ev.clientY - r.top) / r.height) * 100;
  target.style.setProperty("--lq-x", `${x.toFixed(1)}%`);
  target.style.setProperty("--lq-y", `${y.toFixed(1)}%`);
}

// Tinted header — el caller pasa la URL del avatar (img src del
// thread header) y extraemos el color promedio del centro.
export async function tintHeaderFromAvatar(imgUrl) {
  if (!imgUrl) return;
  const header = _headerEl();
  if (!header) return;
  try {
    const rgb = await sampleAvatarColor(imgUrl);
    if (!rgb) return;
    header.style.setProperty("--tint-r", rgb[0]);
    header.style.setProperty("--tint-g", rgb[1]);
    header.style.setProperty("--tint-b", rgb[2]);
  } catch (e) {
    /* silent — header queda con tint default verde accent */
  }
}

async function sampleAvatarColor(url) {
  return new Promise((resolve) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      try {
        const canvas = document.createElement("canvas");
        canvas.width = 16;
        canvas.height = 16;
        const ctx = canvas.getContext("2d", { willReadFrequently: true });
        ctx.drawImage(img, 0, 0, 16, 16);
        const data = ctx.getImageData(0, 0, 16, 16).data;
        // Promedio de los 256 pixels (skipear los muy oscuros/claros).
        let r = 0, g = 0, b = 0, n = 0;
        for (let i = 0; i < data.length; i += 4) {
          const lum = (data[i] + data[i+1] + data[i+2]) / 3;
          if (lum < 24 || lum > 232) continue;
          r += data[i]; g += data[i+1]; b += data[i+2]; n++;
        }
        if (n === 0) return resolve(null);
        resolve([Math.round(r/n), Math.round(g/n), Math.round(b/n)]);
      } catch (e) {
        resolve(null);
      }
    };
    img.onerror = () => resolve(null);
    img.src = url;
  });
}
