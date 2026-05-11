"""Helpers para `rag start` y `rag stop` commands.

Funcs extraídas de `rag/cli/setup.py` para mantener ese file < 1000 LOC:
- Detectión de LAN IP (cuando OBSIDIAN_RAG_BIND_HOST=0.0.0.0).
- Lectura de Cloudflare tunnel URL.
- Health probe del web server.
- Construcción de URLs accesibles.
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
from pathlib import Path

__all__ = [
    "get_lan_ip",
    "get_cloudflared_url",
    "health_probe_web",
    "read_plist_env_var",
]


def get_lan_ip() -> str | None:
    """Detectar la primera IPv4 no-loopback del host.

    Heurística simple: socket.gethostbyname(socket.gethostname()).
    Si falla o devuelve loopback, retorna None (fallback: solo localhost).
    """
    try:
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        if ip and ip != "127.0.0.1":
            return ip
    except Exception:
        pass
    return None


def get_cloudflared_url() -> str | None:
    """Leer la URL del tunnel de Cloudflare desde el archivo de estado.

    Path: ~/.local/share/obsidian-rag/cloudflared-url.txt
    Retorna None si no existe o está vacío.
    """
    url_file = Path.home() / ".local" / "share" / "obsidian-rag" / "cloudflared-url.txt"
    if url_file.exists():
        url = url_file.read_text().strip()
        if url:
            return url
    return None


def health_probe_web(
    host: str = "127.0.0.1",
    port: int = 8765,
    timeout: float = 2.0,
    retries: int = 15,
    retry_delay: float = 2.0,
) -> tuple[bool, int]:
    """Probe a /health del web server. Retorna (ok, latency_ms).

    Retry budget total: `retries * retry_delay + retries * timeout` ≈ 60s
    (default 15 × 2s sleep + 15 × 2s timeout). El warmup MLX del web
    incluye chat (~2s) + embedder (~3-6s) + reranker (~2s) + corpus
    prewarm (variable, hasta 30s en vaults grandes) — total puede llegar
    a ~30-45s en el peor caso. Budget de 60s da margen para no falsos
    negativos. Si el web realmente está down el probe gasta hasta 60s
    antes de declararlo muerto.

    ok=True si responde 200 dentro del budget; False si todos los intentos
    fallan. latency_ms=0 si nunca conectó.
    """
    import urllib.request
    import urllib.error

    url = f"http://{host}:{port}/health"
    last_exc: Exception | None = None
    for attempt in range(retries):
        t0 = time.time()
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                if resp.status == 200:
                    latency_ms = int((time.time() - t0) * 1000)
                    return True, latency_ms
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            last_exc = exc
        if attempt < retries - 1:
            time.sleep(retry_delay)
    _ = last_exc  # silencioso — el caller solo necesita el bool
    return False, 0


def read_plist_env_var(label: str, var_name: str) -> str | None:
    """Leer una env var del plist cargado via `launchctl print`.

    Retorna el valor si existe, None en caso contrario.
    """
    uid = os.getuid()
    try:
        proc = subprocess.run(
            ["launchctl", "print", f"gui/{uid}/{label}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode == 0:
            # Buscar línea "VAR_NAME => value"
            for line in proc.stdout.splitlines():
                stripped = line.strip()
                if stripped.startswith(f"{var_name} =>"):
                    # Extraer el valor después del "=>"
                    parts = stripped.split("=>", 1)
                    if len(parts) == 2:
                        return parts[1].strip()
    except Exception:
        pass
    return None
