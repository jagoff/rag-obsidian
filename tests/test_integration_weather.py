"""Tests para `rag.integrations.weather` — leaf ETL de wttr.in + Open-Meteo.

Surfaces cubiertas:
- `_translate_weather_desc(en)` — traducción ES de descripciones wttr.in.
- `_fetch_weather_rain(location)` — summary cuando el día tiene
  bloques con prob de lluvia ≥ WEATHER_RAIN_THRESHOLD (70%).
- `_fetch_weather_forecast(location)` — multi-day forecast con fallback
  a Open-Meteo cuando wttr.in falla.
- `_fetch_weather_openmeteo(location)` — fallback con geocoding +
  forecast, defensa contra `null` values en sensor outages.

Mocking: `urllib.request.urlopen` se patchea con un context-manager
que yields JSON bytes. El `_req.urlopen` en el módulo se resuelve a
través del attribute lookup en `urllib.request` (NO importado a level
módulo en el target → patch.object falla; la salida es `patch(
"urllib.request.urlopen")`). Idéntico patrón al existing
`test_weather_tool.py`.
"""
from __future__ import annotations

import io
import json
from unittest.mock import MagicMock, patch

import pytest

from rag.integrations import weather as weather_mod


# ── Helper para mockear urlopen ─────────────────────────────────────────────


def _mock_urlopen_ctx(payload):
    """Devuelve un context manager que yields `payload` (dict → JSON bytes)
    o raw bytes/string si se pasa así."""
    cm = MagicMock()
    if isinstance(payload, (bytes, bytearray)):
        body = bytes(payload)
    elif isinstance(payload, str):
        body = payload.encode("utf-8")
    else:
        body = json.dumps(payload).encode("utf-8")
    cm.__enter__ = lambda s: io.BytesIO(body)
    cm.__exit__ = MagicMock(return_value=False)
    return cm


# ── _translate_weather_desc ──────────────────────────────────────────────────


@pytest.mark.parametrize("en,es", [
    ("Clear", "despejado"),
    ("Sunny", "soleado"),
    ("Partly cloudy", "parcialmente nublado"),
    ("Light rain", "lluvia ligera"),
    ("Heavy rain", "lluvia intensa"),
    ("Thundery outbreaks possible", "posibles tormentas"),
])
def test_translate_weather_desc_known_descriptions(en, es):
    assert weather_mod._translate_weather_desc(en) == es


def test_translate_weather_desc_case_insensitive():
    """El lookup es case-insensitive — wttr.in mezcla casing
    inconsistentemente."""
    assert weather_mod._translate_weather_desc("PARTLY CLOUDY") == "parcialmente nublado"
    assert weather_mod._translate_weather_desc("partly cloudy") == "parcialmente nublado"


def test_translate_weather_desc_unknown_passes_through():
    """Verbatim cuando el texto no está en la tabla — preferimos pasar
    el original a hacer un guess malo."""
    assert weather_mod._translate_weather_desc("Volcanic ash") == "Volcanic ash"


def test_translate_weather_desc_empty_returns_empty():
    assert weather_mod._translate_weather_desc("") == ""


# ── _fetch_weather_rain ──────────────────────────────────────────────────────


_SAMPLE_RAINY = {
    "current_condition": [{
        "weatherDesc": [{"value": "Light rain"}],
    }],
    "weather": [{
        "date": "2026-04-29",
        "hourly": [
            # Bloque a las 12 con 80% lluvia → enganchá WEATHER_RAIN_THRESHOLD (70).
            {"time": "1200", "chanceofrain": "80", "chanceofthunder": "0",
             "weatherDesc": [{"value": "Light rain"}]},
            # Bloque a las 15 con 30% — abajo del threshold.
            {"time": "1500", "chanceofrain": "30", "chanceofthunder": "0",
             "weatherDesc": [{"value": "Partly cloudy"}]},
        ],
    }],
}

_SAMPLE_DRY = {
    "current_condition": [{
        "weatherDesc": [{"value": "Sunny"}],
    }],
    "weather": [{
        "date": "2026-04-29",
        "hourly": [
            {"time": "1200", "chanceofrain": "10", "chanceofthunder": "0",
             "weatherDesc": [{"value": "Clear"}]},
        ],
    }],
}


def test_fetch_weather_rain_returns_summary_when_currently_raining():
    """Si el current_condition matchea el regex `rain|shower|...` → emit
    summary con `ahora: ...` aunque no haya bloques futuros."""
    with patch("urllib.request.urlopen", return_value=_mock_urlopen_ctx(_SAMPLE_RAINY)):
        out = weather_mod._fetch_weather_rain("TestCity")
    assert out is not None
    assert "ahora" in out["summary"].lower() or "lluvia" in out["summary"].lower()
    assert out["max_chance"] >= 70


def test_fetch_weather_rain_returns_none_when_dry():
    """Sin lluvia ahora ni bloques ≥70% → `None` (el brief omite la
    sección)."""
    with patch("urllib.request.urlopen", return_value=_mock_urlopen_ctx(_SAMPLE_DRY)):
        out = weather_mod._fetch_weather_rain("TestCity")
    assert out is None


def test_fetch_weather_rain_silent_on_network_error():
    """Network error → `None` (el morning brief no se rompe por wttr.in
    caído)."""
    with patch("urllib.request.urlopen", side_effect=OSError("network down")):
        out = weather_mod._fetch_weather_rain("TestCity")
    assert out is None


def test_fetch_weather_rain_silent_on_bad_json():
    """JSON malformado → `None` (defensive parse)."""
    with patch("urllib.request.urlopen",
               return_value=_mock_urlopen_ctx(b"not-json-at-all")):
        out = weather_mod._fetch_weather_rain("TestCity")
    assert out is None


# ── _fetch_weather_forecast ──────────────────────────────────────────────────


_SAMPLE_WTTR = {
    "current_condition": [{
        "temp_C": "22",
        "weatherDesc": [{"value": "Partly cloudy"}],
    }],
    "weather": [
        {
            "date": "2026-04-29",
            "mintempC": "16", "maxtempC": "25", "avgtempC": "20",
            "hourly": [
                {"time": "0", "chanceofrain": "0", "chanceofthunder": "0",
                 "weatherDesc": [{"value": "Clear"}]},
                {"time": "1200", "chanceofrain": "80", "chanceofthunder": "30",
                 "weatherDesc": [{"value": "Light rain shower"}]},
            ],
        },
        {
            "date": "2026-04-30",
            "mintempC": "14", "maxtempC": "22", "avgtempC": "18",
            "hourly": [
                {"time": "1200", "chanceofrain": "5", "chanceofthunder": "0",
                 "weatherDesc": [{"value": "Sunny"}]},
            ],
        },
    ],
}


def test_fetch_weather_forecast_parses_days_with_translations():
    """Happy path: response wttr.in → dict con `days` (con descriptions
    traducidas) + `current` + `text_summary`."""
    with patch("urllib.request.urlopen", return_value=_mock_urlopen_ctx(_SAMPLE_WTTR)):
        out = weather_mod._fetch_weather_forecast("TestCity")
    assert out is not None
    assert len(out["days"]) == 2

    day0 = out["days"][0]
    assert day0["date"] == "2026-04-29"
    assert day0["maxC"] == "25"
    assert day0["chanceofrain"] == 80
    assert day0["chanceofthunder"] == 30
    # La traducción debe haber pasado: "Light rain shower" → "chubasco ligero".
    assert "chubasco" in day0["description"].lower() or "lluvia" in day0["description"].lower()

    # text_summary tiene location + current + temp.
    assert "TestCity" in out["text_summary"] or "22" in out["text_summary"]


def test_fetch_weather_forecast_falls_back_to_openmeteo_on_wttr_error(monkeypatch):
    """Si wttr.in raisea, debe llamar a `_fetch_weather_openmeteo` como
    fallback. Capturamos la llamada con un sentinel."""
    sentinel = {"location": "Sentinel", "current": {"description": "OK", "temp_C": "10"},
                "days": [], "text_summary": "Sentinel: OK (10°C)"}

    monkeypatch.setattr(weather_mod, "_fetch_weather_openmeteo",
                        lambda loc=None: sentinel)
    with patch("urllib.request.urlopen", side_effect=OSError("wttr down")):
        out = weather_mod._fetch_weather_forecast("Madrid")
    assert out is sentinel


def test_fetch_weather_forecast_falls_back_when_wttr_returns_no_days(monkeypatch):
    """Si wttr.in responde con shape OK pero `weather=[]` (raro pero
    ocurre con cities ambiguas), debe caer a Open-Meteo."""
    sentinel = {"location": "FB", "current": {}, "days": [], "text_summary": ""}
    monkeypatch.setattr(weather_mod, "_fetch_weather_openmeteo",
                        lambda loc=None: sentinel)
    empty_data = {"current_condition": [], "weather": []}
    with patch("urllib.request.urlopen",
               return_value=_mock_urlopen_ctx(empty_data)):
        out = weather_mod._fetch_weather_forecast("X")
    assert out is sentinel


# ── _fetch_weather_openmeteo ────────────────────────────────────────────────


def test_fetch_weather_openmeteo_handles_null_values_in_daily_series():
    """Open-Meteo emite `null` durante outages de sensor; el `or 0` /
    `is not None` guard previene `TypeError: round(None)`."""
    geo_payload = {
        "results": [{"latitude": 40.4, "longitude": -3.7, "name": "Madrid"}],
    }
    forecast_payload = {
        "current": {"temperature_2m": None, "weather_code": None},  # null current
        "daily": {
            "time": ["2026-04-29", "2026-04-30", "2026-05-01"],
            "temperature_2m_max": [20.0, None, 22.0],  # null en posición 1
            "temperature_2m_min": [None, 12.0, 14.0],
            "precipitation_probability_max": [None, 30, 0],
            "weather_code": [1, None, 3],
        },
    }

    # Open-Meteo hace 2 calls: geocoding + forecast. Side_effect con la lista.
    geo_cm = _mock_urlopen_ctx(geo_payload)
    fx_cm = _mock_urlopen_ctx(forecast_payload)
    with patch("urllib.request.urlopen", side_effect=[geo_cm, fx_cm]):
        out = weather_mod._fetch_weather_openmeteo("Madrid,Spain")

    assert out is not None
    assert out["location"].startswith("Madrid")
    assert len(out["days"]) == 3
    # Día 0: max=20, min=null → "" (string vacío)
    assert out["days"][0]["maxC"] == "20"
    assert out["days"][0]["minC"] == ""
    # Día 1: max=null → ""
    assert out["days"][1]["maxC"] == ""
    assert out["days"][1]["minC"] == "12"
    # current con nulls → cur_temp "0" (defensivo)
    assert out["current"]["temp_C"] == "0"


def test_fetch_weather_openmeteo_returns_none_on_geo_failure():
    """Si geocoding API levanta, devolvemos `None` (no llamamos forecast
    ni inventamos coords)."""
    with patch("urllib.request.urlopen", side_effect=OSError("dns fail")):
        out = weather_mod._fetch_weather_openmeteo("CiudadInexistente")
    assert out is None


def test_fetch_weather_openmeteo_returns_none_on_no_geo_results():
    """Si la geocoding API responde OK pero `results=[]` (city no
    encontrada), `KeyError` en `geo["results"][0]` → caught → None."""
    with patch("urllib.request.urlopen",
               return_value=_mock_urlopen_ctx({"results": []})):
        out = weather_mod._fetch_weather_openmeteo("XYZNoExiste")
    assert out is None


def test_fetch_weather_openmeteo_text_summary_format():
    """`text_summary` tiene el formato `<loc>: <desc> (<temp>°C)` para
    que el LLM lo eche en la síntesis (BUG-2a 2026-04-28)."""
    geo_payload = {
        "results": [{"latitude": 0, "longitude": 0, "name": "Quito"}],
    }
    forecast_payload = {
        "current": {"temperature_2m": 18, "weather_code": 1},  # "Mayormente despejado"
        "daily": {"time": [], "temperature_2m_max": [],
                  "temperature_2m_min": [], "precipitation_probability_max": [],
                  "weather_code": []},
    }
    with patch("urllib.request.urlopen",
               side_effect=[_mock_urlopen_ctx(geo_payload),
                            _mock_urlopen_ctx(forecast_payload)]):
        out = weather_mod._fetch_weather_openmeteo("Quito,Ecuador")
    assert "Quito" in out["text_summary"]
    assert "18" in out["text_summary"]
    assert "°C" in out["text_summary"]
