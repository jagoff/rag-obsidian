"""Weather integration — leaf ETL extracted from `rag/__init__.py` (Phase 1b).

Sources:
- Primary: [wttr.in](https://wttr.in/) — JSON `format=j1` endpoint, hourly + daily.
- Fallback: [Open-Meteo](https://open-meteo.com/) — geocoding + forecast, no API key.

Two consumers:

- `_fetch_weather_rain(location)` → returns a summary dict ONLY if rain ≥
  `WEATHER_RAIN_THRESHOLD` is forecast for today (or it's already raining).
  Used by morning briefs to mention "vas a necesitar paraguas hoy".
- `_fetch_weather_forecast(location)` → returns a multi-day forecast (3 days)
  used by the `weather` CLI subcommand and the `_agent_tool_weather` tool.
  Falls back to Open-Meteo if wttr.in fails. Open-Meteo also handles `null`
  values defensively (sensor outages emit `null` instead of missing keys —
  pre-fix `round(None)` killed the morning brief).
- `_weather_comment(question, forecast)` → uses the helper LLM (qwen2.5:3b) to
  generate a 1-2 sentence opinion on a forecast. Tied to the helper-LLM stack
  (`_helper_client`, `HELPER_MODEL`, `HELPER_OPTIONS`, `OLLAMA_KEEP_ALIVE`),
  imported lazily because that stack lives in `rag.__init__`.

## Invariants
- Silent-fail: any network/parse/timeout error → return `None` (never raise).
- The `_req.urlopen` import inside each function is preserved deliberately —
  tests patch `urllib.request.urlopen` and rely on the lookup happening
  through the `urllib.request` module attribute (so `_req.urlopen` resolves
  to the patched one at call time, not import time).
- `WEATHER_LOCATION` default ("Santa+Fe,Argentina") matches the user's
  configured city; tests pass other strings (e.g. "TestCity", "Buenos+Aires").

## Why deferred imports for the helper LLM
`_weather_comment` calls Ollama via `_helper_client()` — that helper, plus
`HELPER_MODEL`, `HELPER_OPTIONS`, and `OLLAMA_KEEP_ALIVE`, all live in
`rag.__init__`. Module-level `from rag import …` here would trigger a circular
import during package load. Deferred (function-body) imports run after
`rag.__init__` finishes loading, so they always succeed.
"""

from __future__ import annotations

import json
import re
from datetime import datetime


# ── Weather (only if rain) ──────────────────────────────────────────────────
WEATHER_LOCATION = "Santa+Fe,Argentina"
WEATHER_RAIN_THRESHOLD = 70  # contract in repo CLAUDE.md: hint only if rain ≥70%


def _fetch_weather_rain(location: str = WEATHER_LOCATION) -> dict | None:
    """Query wttr.in. Returns a summary dict ONLY if rain is in the forecast
    for today (chance ≥ WEATHER_RAIN_THRESHOLD in any upcoming 3h block, or
    current condition is raining). Returns ``None`` otherwise. Silent on any
    network error.
    """
    import urllib.request as _req
    url = f"https://wttr.in/{location}?format=j1"
    try:
        with _req.urlopen(url, timeout=8.0) as resp:
            raw = resp.read()
    except Exception:
        return None
    try:
        data = json.loads(raw)
    except Exception:
        return None

    # Current conditions: "Rain", "Light rain", "Thunderstorm", etc.
    current = ""
    try:
        cc = data.get("current_condition") or []
        if cc:
            desc = (cc[0].get("weatherDesc") or [{}])[0].get("value", "")
            current = desc.strip()
    except Exception:
        current = ""
    currently_raining = bool(re.search(r"rain|shower|thunder|storm|drizzle", current, re.I))

    # Today hourly: each entry has time="0|300|600|…|2100" (3h blocks) and
    # chanceofrain / chanceofthunder as string ints.
    weather_days = data.get("weather") or []
    if not weather_days:
        return None if not currently_raining else {
            "summary": f"ahora: {current}",
            "max_chance": 100, "blocks": [],
        }
    today = weather_days[0]
    hourly = today.get("hourly") or []

    now = datetime.now()
    now_minutes = now.hour * 60 + now.minute
    rain_blocks: list[dict] = []
    max_chance = 0
    for h in hourly:
        try:
            t = int(h.get("time", "0"))
        except Exception:
            continue
        block_minutes = (t // 100) * 60
        if block_minutes + 180 < now_minutes:
            continue  # block already past
        try:
            chance_rain = int(h.get("chanceofrain", "0") or 0)
            chance_thunder = int(h.get("chanceofthunder", "0") or 0)
        except Exception:
            continue
        chance = max(chance_rain, chance_thunder)
        if chance >= WEATHER_RAIN_THRESHOLD:
            hh = block_minutes // 60
            rain_blocks.append({"hour": hh, "chance": chance})
            if chance > max_chance:
                max_chance = chance

    if not rain_blocks and not currently_raining:
        return None

    pieces = []
    if currently_raining:
        pieces.append(f"ahora: {current.lower()}")
    if rain_blocks:
        hour_str = ", ".join(f"{b['hour']:02d}h ({b['chance']}%)" for b in rain_blocks[:5])
        pieces.append(f"bloques: {hour_str}")
    return {
        "summary": " · ".join(pieces) or current,
        "max_chance": max_chance if max_chance else (100 if currently_raining else 0),
        "blocks": rain_blocks,
        "current": current,
    }


_WMO_WEATHER_CODES: dict[int, str] = {
    0: "Despejado", 1: "Mayormente despejado", 2: "Parcialmente nublado",
    3: "Cubierto", 45: "Neblina", 48: "Neblina helada",
    51: "Llovizna leve", 53: "Llovizna moderada", 55: "Llovizna densa",
    56: "Llovizna helada leve", 57: "Llovizna helada densa",
    61: "Lluvia leve", 63: "Lluvia moderada", 65: "Lluvia fuerte",
    66: "Lluvia helada leve", 67: "Lluvia helada fuerte",
    71: "Nieve leve", 73: "Nieve moderada", 75: "Nieve fuerte",
    77: "Granizo fino", 80: "Chubasco leve", 81: "Chubasco moderado",
    82: "Chubasco fuerte", 85: "Nevada leve", 86: "Nevada fuerte",
    95: "Tormenta", 96: "Tormenta con granizo leve", 99: "Tormenta con granizo fuerte",
}


def _fetch_weather_openmeteo(location: str = WEATHER_LOCATION) -> dict | None:
    """Fallback weather via Open-Meteo (free, no API key)."""
    import urllib.request as _req
    import urllib.parse as _parse

    city = location.replace("+", " ").split(",")[0]
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={_parse.quote(city)}&count=1&language=es"
    try:
        with _req.urlopen(geo_url, timeout=5.0) as resp:
            geo = json.loads(resp.read())
        r = geo["results"][0]
        lat, lon, name = r["latitude"], r["longitude"], r.get("name", city)
    except Exception:
        return None

    wx_url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        "&current=temperature_2m,weather_code"
        "&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code"
        "&timezone=auto&forecast_days=3"
    )
    try:
        with _req.urlopen(wx_url, timeout=5.0) as resp:
            data = json.loads(resp.read())
    except Exception:
        return None

    # `dict.get(key, default)` returns the default ONLY when key is missing —
    # if the key exists with value `null` (Open-Meteo emits this during
    # geo/sensor outages), `.get()` returns None and `round(None)` raises
    # TypeError. The `or` short-circuit catches both missing-key and
    # null-value cases. Same pattern below for daily series.
    cur = data.get("current", {}) or {}
    cur_code = cur.get("weather_code") or 0
    cur_desc = _WMO_WEATHER_CODES.get(cur_code, f"Código {cur_code}")
    cur_temp = str(round(cur.get("temperature_2m") or 0))

    daily = data.get("daily", {}) or {}
    dates = daily.get("time", []) or []
    maxs = daily.get("temperature_2m_max", []) or []
    mins = daily.get("temperature_2m_min", []) or []
    rain_probs = daily.get("precipitation_probability_max", []) or []
    codes = daily.get("weather_code", []) or []

    days: list[dict] = []
    for i, d in enumerate(dates):
        code = codes[i] if i < len(codes) and codes[i] is not None else 0
        # `mins[i]` / `maxs[i]` may be None even with len(...) > i.
        min_v = mins[i] if i < len(mins) else None
        max_v = maxs[i] if i < len(maxs) else None
        rain_v = rain_probs[i] if i < len(rain_probs) else None
        days.append({
            "date": d,
            "minC": str(round(min_v)) if min_v is not None else "",
            "maxC": str(round(max_v)) if max_v is not None else "",
            "avgC": "",
            "description": _WMO_WEATHER_CODES.get(code, f"Código {code}"),
            "chanceofrain": rain_v if rain_v is not None else 0,
            "chanceofthunder": 0,
        })

    country = location.split(",")[-1].strip().replace("+", " ") if "," in location else ""
    loc_str = f"{name},{country}" if country else name
    return {
        "location": loc_str,
        "current": {"description": cur_desc, "temp_C": cur_temp},
        "days": days,
    }


def _fetch_weather_forecast(location: str = WEATHER_LOCATION) -> dict | None:
    """Query wttr.in, fallback to Open-Meteo. Returns multi-day forecast summary.

    Returns dict with keys: location, current, days (list of up to 3 day
    forecasts with date, minC, maxC, avgC, description, chanceofrain,
    chanceofthunder). Returns None on network/parse errors from both sources.
    """
    import urllib.request as _req

    # Primary: wttr.in
    url = f"https://wttr.in/{location}?format=j1"
    try:
        with _req.urlopen(url, timeout=8.0) as resp:
            raw = resp.read()
        data = json.loads(raw)
    except Exception:
        return _fetch_weather_openmeteo(location)

    # Current conditions
    current = ""
    temp_c = ""
    try:
        cc = data.get("current_condition") or []
        if cc:
            desc = (cc[0].get("weatherDesc") or [{}])[0].get("value", "")
            current = desc.strip()
            temp_c = cc[0].get("temp_C", "")
    except Exception:
        pass

    # Daily forecasts (wttr.in gives today + 2 more days)
    days: list[dict] = []
    for wd in data.get("weather") or []:
        # Max rain chance across hourly blocks
        max_rain = 0
        max_thunder = 0
        descs: list[str] = []
        for h in wd.get("hourly") or []:
            try:
                max_rain = max(max_rain, int(h.get("chanceofrain", "0") or 0))
                max_thunder = max(max_thunder, int(h.get("chanceofthunder", "0") or 0))
            except Exception:
                pass
            d = (h.get("weatherDesc") or [{}])[0].get("value", "")
            if d and d not in descs:
                descs.append(d)
        days.append({
            "date": wd.get("date", ""),
            "minC": wd.get("mintempC", ""),
            "maxC": wd.get("maxtempC", ""),
            "avgC": wd.get("avgtempC", ""),
            "description": ", ".join(descs[:3]),
            "chanceofrain": max_rain,
            "chanceofthunder": max_thunder,
        })

    if not days:
        return _fetch_weather_openmeteo(location)

    return {
        "location": location.replace("+", " "),
        "current": {"description": current, "temp_C": temp_c},
        "days": days,
    }


def _weather_comment(question: str, forecast: str) -> str:
    """Generate a brief conversational comment relating the forecast to the question."""
    from rag import HELPER_MODEL, HELPER_OPTIONS, OLLAMA_KEEP_ALIVE, _helper_client
    prompt = (
        "Pronóstico:\n"
        f"{forecast}\n\n"
        f"Pregunta: \"{question}\"\n\n"
        "Respondé EN ESPAÑOL RIOPLATENSE en 1-2 oraciones cortas. "
        "Tono informal, directo al punto. "
        "No repitas números ni datos del pronóstico. "
        "Solo dá tu opinión o consejo basado en los datos. "
        "Sin emojis. Sin saludos. Sin preámbulos."
    )
    try:
        resp = _helper_client().chat(
            model=HELPER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={**HELPER_OPTIONS, "num_predict": 80},
            keep_alive=OLLAMA_KEEP_ALIVE,
        )
        return resp.message.content.strip()
    except Exception:
        return ""
