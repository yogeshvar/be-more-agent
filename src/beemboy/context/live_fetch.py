"""Fetch short weather + news snippets for the system prompt (stdlib only)."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from beemboy.config.settings import Settings

USER_AGENT = "BeemboyLiveContext/1.0 (+local-assistant)"


def _get(url: str, timeout: float = 12.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _wmo_label(code: int) -> str:
    m = {
        0: "clear",
        1: "mainly clear",
        2: "partly cloudy",
        3: "overcast",
        45: "fog",
        48: "depositing rime fog",
        51: "light drizzle",
        53: "drizzle",
        55: "dense drizzle",
        61: "slight rain",
        63: "rain",
        65: "heavy rain",
        71: "slight snow",
        73: "snow",
        75: "heavy snow",
        77: "snow grains",
        80: "rain showers",
        81: "moderate rain showers",
        82: "violent rain showers",
        85: "snow showers",
        86: "heavy snow showers",
        95: "thunderstorm",
        96: "thunderstorm with hail",
        99: "thunderstorm with heavy hail",
    }
    return m.get(code, f"code {code}")


def _weather_open_meteo(settings: Settings) -> str | None:
    city = (settings.weather_city or "").strip()
    lat_s, lon_s = (settings.weather_lat or "").strip(), (settings.weather_lon or "").strip()

    lat: float | None = None
    lon: float | None = None
    label = ""

    if lat_s and lon_s:
        try:
            lat, lon = float(lat_s), float(lon_s)
            label = f"{lat_s}, {lon_s}"
        except ValueError:
            return None
    elif city:
        q = urllib.parse.quote(city)
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={q}&count=1&language=en&format=json"
        try:
            raw = _get(geo_url)
            data = json.loads(raw.decode())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError):
            return None
        results = data.get("results") or []
        if not results:
            return None
        r0 = results[0]
        lat, lon = float(r0["latitude"]), float(r0["longitude"])
        label = f"{r0.get('name', city)}, {r0.get('country_code', '')}".strip(", ")
    else:
        return None

    fc_url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m"
        f"&wind_speed_unit=kmh&timezone=auto"
    )
    try:
        raw = _get(fc_url)
        data = json.loads(raw.decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError, KeyError):
        return None

    cur = data.get("current") or {}
    temp = cur.get("temperature_2m")
    hum = cur.get("relative_humidity_2m")
    wcode = cur.get("weather_code")
    wind = cur.get("wind_speed_10m")
    if temp is None or wcode is None:
        return None

    temp_f = temp * 9 / 5 + 32
    desc = _wmo_label(int(wcode))
    parts = [
        f"{label}: {desc}, {temp:.0f}°C ({temp_f:.0f}°F)",
    ]
    if hum is not None:
        parts.append(f"humidity {hum:.0f}%")
    if wind is not None:
        parts.append(f"wind {wind:.0f} km/h")
    return ", ".join(parts)


def _weather_openweathermap(settings: Settings) -> str | None:
    key = (settings.openweathermap_api_key or "").strip()
    if not key:
        return None

    city = (settings.weather_city or "").strip()
    lat_s, lon_s = (settings.weather_lat or "").strip(), (settings.weather_lon or "").strip()
    units = (settings.weather_units or "metric").strip()
    if units not in ("metric", "imperial", "standard"):
        units = "metric"

    if lat_s and lon_s:
        q = f"lat={urllib.parse.quote(lat_s)}&lon={urllib.parse.quote(lon_s)}"
    elif city:
        q = f"q={urllib.parse.quote(city)}"
    else:
        return None

    url = f"https://api.openweathermap.org/data/2.5/weather?{q}&appid={urllib.parse.quote(key)}&units={units}"
    try:
        raw = _get(url)
        data = json.loads(raw.decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError):
        return None

    name = data.get("name", city or "")
    country = (data.get("sys") or {}).get("country", "")
    loc = ", ".join(x for x in (name, country) if x)
    w = (data.get("weather") or [{}])[0].get("description", "")
    main = data.get("main") or {}
    temp = main.get("temp")
    hum = main.get("humidity")
    wind = (data.get("wind") or {}).get("speed")
    if temp is None:
        return None

    unit_sym = "°C" if units == "metric" else "°F"
    parts = [f"{loc}: {w}, {temp:.0f}{unit_sym}"]
    if hum is not None:
        parts.append(f"humidity {hum:.0f}%")
    if wind is not None:
        wu = "m/s" if units == "metric" else "mph"
        parts.append(f"wind {wind:.1f} {wu}")
    return ", ".join(parts)


def _news_newsapi(settings: Settings) -> str | None:
    key = (settings.newsapi_key or "").strip()
    if not key:
        return None

    country = (settings.news_country or "us").strip().lower()
    if len(country) != 2:
        country = "us"

    url = (
        "https://newsapi.org/v2/top-headlines?"
        f"country={urllib.parse.quote(country)}&pageSize=5&apiKey={urllib.parse.quote(key)}"
    )
    try:
        raw = _get(url)
        data = json.loads(raw.decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError):
        return None

    if data.get("status") != "ok":
        return None

    titles: list[str] = []
    for art in data.get("articles") or []:
        t = (art.get("title") or "").strip()
        if t:
            titles.append(t)
        if len(titles) >= 5:
            break
    if not titles:
        return None
    return "; ".join(titles)


def _news_rss_feed(url: str, max_items: int = 5) -> list[str]:
    try:
        raw = _get(url)
    except (urllib.error.URLError, TimeoutError):
        return []

    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return []

    titles: list[str] = []
    for el in root.iter():
        tag = el.tag.split("}")[-1]
        if tag in ("item", "entry"):
            for child in el:
                ctag = child.tag.split("}")[-1]
                if ctag == "title" and (child.text or "").strip():
                    titles.append((child.text or "").strip())
                    break
            if len(titles) >= max_items:
                break
    return titles[:max_items]


def _news_fallback_rss(settings: Settings) -> str | None:
    custom = (settings.news_rss_urls or "").strip()
    if custom:
        urls = [u.strip() for u in custom.split(",") if u.strip()]
    else:
        urls = ["https://feeds.bbci.co.uk/news/world/rss.xml"]

    all_titles: list[str] = []
    for u in urls:
        all_titles.extend(_news_rss_feed(u, 5 - len(all_titles)))
        if len(all_titles) >= 5:
            break
    if not all_titles:
        return None
    return "; ".join(all_titles)


def build_live_context_block(settings: Settings) -> str | None:
    lines: list[str] = []

    w = _weather_openweathermap(settings)
    if w is None:
        w = _weather_open_meteo(settings)
    if w:
        lines.append(f"Weather: {w}")

    n = _news_newsapi(settings)
    if n is None:
        n = _news_fallback_rss(settings)
    if n:
        lines.append(f"Recent headlines: {n}")

    if not lines:
        return None

    return (
        "Live data for this session (trust for weather and headline questions; do not invent beyond it):\n"
        + "\n".join(lines)
    )
