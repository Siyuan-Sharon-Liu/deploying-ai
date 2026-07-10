from langchain.tools import tool
import requests
from dotenv import load_dotenv

from utils.logger import get_logger

_logs = get_logger(__name__)
load_dotenv(".env")
load_dotenv(".secrets")

_RESTRICTED = {"taylor swift", "taylor", "swift", "tay tay"}


@tool
def get_artist_releases(artist: str, limit: int = 5) -> str:
    """Look up real album releases for an artist from the public iTunes catalog.
    Returns compact structured facts (album, year, genre, track count) that the
    assistant must rephrase in its own words. Do not present these facts verbatim."""
    if artist.strip().lower() in _RESTRICTED or "taylor swift" in artist.lower():
        return "REFUSE: restricted artist."
    try:
        resp = requests.get(
            "https://itunes.apple.com/search",
            params={"term": artist, "entity": "album", "limit": limit},
            timeout=10,
        )
        results = resp.json().get("results", [])
    except Exception as e:
        _logs.warning(f"iTunes lookup failed for {artist!r}: {e}")
        return f"Could not reach the catalog for '{artist}' right now."

    if not results:
        return f"No catalog results found for '{artist}'."

    lines = []
    for r in results:
        album = r.get("collectionName")
        if not album:
            continue
        year = (r.get("releaseDate") or "")[:4]
        genre = r.get("primaryGenreName", "N/A")
        tracks = r.get("trackCount", "?")
        lines.append(f"- {album} ({year}), {genre}, {tracks} tracks")
    return "\n".join(lines) if lines else f"No albums found for '{artist}'."
