from langchain.tools import tool
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

from utils.logger import get_logger

_logs = get_logger(__name__)
load_dotenv(".env")
load_dotenv(".secrets")

_search = TavilySearch(max_results=3)


@tool
def web_search(query: str) -> str:
    """Search the web for current music news, tour dates, or brand-new releases that a
    static review database cannot cover. Returns a few trimmed results. This is a simple
    single-shot search, not an agentic or deep-research pipeline."""
    try:
        results = _search.invoke({"query": query})
    except Exception as e:
        _logs.warning(f"Tavily search failed for {query!r}: {e}")
        return "Could not reach the wire for news right now."

    if isinstance(results, dict) and results.get("error"):
        _logs.warning(f"Tavily returned an error: {results['error']}")
        return "Could not reach the wire for news right now."
    items = results.get("results", []) if isinstance(results, dict) else results
    trimmed = []
    for r in items[:3]:
        if not isinstance(r, dict):
            continue
        title = r.get("title", "")
        content = (r.get("content", "") or "")[:300]
        url = r.get("url", "")
        trimmed.append(f"{title}: {content} ({url})")
    return "\n\n".join(trimmed) if trimmed else "No web results."
