# server_fun.py
# MCP Tools Server — exposes free, no-key APIs as MCP tools
# Tools: Weather (Open-Meteo), Books (Open Library), Jokes (JokeAPI),
#         Dog Photo (Dog CEO), Trivia (Open Trivia DB)

from mcp.server.fastmcp import FastMCP
from typing import Optional, Dict, Any, List
import requests
import html

mcp = FastMCP("FunTools")


# ──────────────────────────────────────────────
# Weather (Open-Meteo)
# ──────────────────────────────────────────────
@mcp.tool()
def get_weather(latitude: float, longitude: float) -> Dict[str, Any]:
    """Current weather at coordinates via Open-Meteo.

    Args:
        latitude:  Latitude of the location  (e.g. 40.7128 for NYC)
        longitude: Longitude of the location (e.g. -74.0060 for NYC)
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,weather_code,wind_speed_10m",
        "timezone": "auto",
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json().get("current", {})


# ──────────────────────────────────────────────
# Book recommendations (Open Library)
# ──────────────────────────────────────────────
@mcp.tool()
def book_recs(topic: str, limit: int = 5) -> Dict[str, Any]:
    """Simple book suggestions for a topic via Open Library search.

    Args:
        topic: A genre or topic to search for (e.g. "mystery", "sci-fi")
        limit: Maximum number of results to return (default 5)
    """
    r = requests.get(
        "https://openlibrary.org/search.json",
        params={"q": topic, "limit": limit},
        timeout=20,
    )
    r.raise_for_status()
    docs = r.json().get("docs", [])
    picks: List[Dict[str, Any]] = []
    for d in docs:
        picks.append(
            {
                "title": d.get("title"),
                "author": (d.get("author_name") or ["Unknown"])[0],
                "year": d.get("first_publish_year"),
                "work": d.get("key"),
            }
        )
    return {"topic": topic, "results": picks}


# ──────────────────────────────────────────────
# Jokes (JokeAPI)
# ──────────────────────────────────────────────
@mcp.tool()
def random_joke() -> Dict[str, Any]:
    """Return a safe, single-line joke from JokeAPI."""
    r = requests.get(
        "https://v2.jokeapi.dev/joke/Any?type=single&safe-mode", timeout=20
    )
    r.raise_for_status()
    data = r.json()
    return {"joke": data.get("joke", "No joke found")}


# ──────────────────────────────────────────────
# Dog photo (Dog CEO)
# ──────────────────────────────────────────────
@mcp.tool()
def random_dog() -> Dict[str, Any]:
    """Return a random dog image URL from the Dog CEO API."""
    r = requests.get("https://dog.ceo/api/breeds/image/random", timeout=20)
    r.raise_for_status()
    return r.json()


# ──────────────────────────────────────────────
# (Optional) Trivia (Open Trivia DB)
# ──────────────────────────────────────────────
@mcp.tool()
def trivia() -> Dict[str, Any]:
    """Return one multiple-choice trivia question from Open Trivia DB."""
    r = requests.get(
        "https://opentdb.com/api.php?amount=1&type=multiple", timeout=20
    )
    r.raise_for_status()
    data = r.json().get("results", [])
    if not data:
        return {"error": "no trivia"}
    q = data[0]
    q["question"] = html.unescape(q["question"])
    q["correct_answer"] = html.unescape(q["correct_answer"])
    q["incorrect_answers"] = [html.unescape(x) for x in q["incorrect_answers"]]
    return q


# ──────────────────────────────────────────────
if __name__ == "__main__":
    mcp.run()  # stdio server
