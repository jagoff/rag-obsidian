"""Tests for _fetch_weather_forecast and _agent_tool_weather."""
import json
from unittest.mock import patch, MagicMock
import io

import rag

# ── Sample wttr.in response (minimal) ─────────────────────────────────────

SAMPLE_WTTR = {
    "current_condition": [
        {
            "temp_C": "22",
            "weatherDesc": [{"value": "Partly cloudy"}],
        }
    ],
    "weather": [
        {
            "date": "2026-04-15",
            "mintempC": "16",
            "maxtempC": "25",
            "avgtempC": "20",
            "hourly": [
                {"time": "0", "chanceofrain": "0", "chanceofthunder": "0",
                 "weatherDesc": [{"value": "Clear"}]},
                {"time": "600", "chanceofrain": "10", "chanceofthunder": "0",
                 "weatherDesc": [{"value": "Partly cloudy"}]},
                {"time": "1200", "chanceofrain": "80", "chanceofthunder": "30",
                 "weatherDesc": [{"value": "Light rain shower"}]},
            ],
        },
        {
            "date": "2026-04-16",
            "mintempC": "14",
            "maxtempC": "22",
            "avgtempC": "18",
            "hourly": [
                {"time": "0", "chanceofrain": "0", "chanceofthunder": "0",
                 "weatherDesc": [{"value": "Clear"}]},
                {"time": "1200", "chanceofrain": "5", "chanceofthunder": "0",
                 "weatherDesc": [{"value": "Sunny"}]},
            ],
        },
    ],
}


def _mock_urlopen(data):
    """Return a context-manager mock that yields JSON bytes."""
    cm = MagicMock()
    cm.__enter__ = lambda s: io.BytesIO(json.dumps(data).encode())
    cm.__exit__ = MagicMock(return_value=False)
    return cm


# ── _fetch_weather_forecast ───────────────────────────────────────────────


def test_forecast_parses_days():
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(SAMPLE_WTTR)):
        result = rag._fetch_weather_forecast("TestCity")
    assert result is not None
    assert len(result["days"]) == 2
    assert result["current"]["temp_C"] == "22"
    day0 = result["days"][0]
    assert day0["date"] == "2026-04-15"
    assert day0["maxC"] == "25"
    assert day0["chanceofrain"] == 80  # max across hourly blocks


def test_forecast_tomorrow_no_rain():
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(SAMPLE_WTTR)):
        result = rag._fetch_weather_forecast("TestCity")
    day1 = result["days"][1]
    assert day1["date"] == "2026-04-16"
    assert day1["chanceofrain"] == 5


def test_forecast_network_error():
    with patch("urllib.request.urlopen", side_effect=OSError("timeout")):
        result = rag._fetch_weather_forecast("TestCity")
    assert result is None


def test_forecast_bad_json():
    cm = MagicMock()
    cm.__enter__ = lambda s: io.BytesIO(b"not json")
    cm.__exit__ = MagicMock(return_value=False)
    with patch("urllib.request.urlopen", return_value=cm):
        result = rag._fetch_weather_forecast("TestCity")
    assert result is None


def test_forecast_location_in_result():
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(SAMPLE_WTTR)):
        result = rag._fetch_weather_forecast("Santa+Fe")
    assert result["location"] == "Santa Fe"


# ── _agent_tool_weather ──────────────────────────────────────────────────


def test_agent_tool_weather_default_location():
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(SAMPLE_WTTR)):
        raw = rag._agent_tool_weather()
    data = json.loads(raw)
    assert "days" in data
    assert len(data["days"]) == 2


def test_agent_tool_weather_custom_location():
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(SAMPLE_WTTR)):
        raw = rag._agent_tool_weather(location="Buenos Aires")
    data = json.loads(raw)
    assert data["location"] == "Buenos Aires"


def test_agent_tool_weather_error():
    with patch("urllib.request.urlopen", side_effect=OSError):
        raw = rag._agent_tool_weather()
    assert "Error" in raw
