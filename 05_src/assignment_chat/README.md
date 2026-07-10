# The Record Counter — Assignment 2 Chat Client

A music-recommendation chat app with a distinct persona, built on the course stack
(LangGraph + LangChain + OpenAI `gpt-4o-mini` + Gradio), mirroring the structure of
`05_src/course_chat/`.

## Run

From `05_src/`, with the course environment active:

```bash
source ../deploying-ai-env/bin/activate
python -m assignment_chat.app
```

Requires `05_src/.env` and `05_src/.secrets` with `OPENAI_API_KEY` and `TAVILY_API_KEY`
set (the standard course setup). No extra libraries are needed.

## Persona

The assistant is **Vinny**, the clerk at a cramped independent record shop: a witty,
opinionated critic with dry humor and deep-cuts taste. The persona shapes tone and drives
the "rephrase, don't dump" behaviour for tool output. The system prompt lives in
`prompts.py`.

## The three services

| # | Service | File | Back end |
|---|---------|------|----------|
| 1 | Artist releases (API) | `tools_api.py` | iTunes Search API (keyless) |
| 2 | Album recommendations (semantic) | `tools_reviews.py` | ChromaDB `PersistentClient` + pandas CSV |
| 3 | Music news (web search) | `tools_websearch.py` | Tavily `TavilySearch` |

**Service 1 — API calls.** `get_artist_releases` queries the public iTunes catalog and
returns compact structured facts (album, year, genre, track count). The output is never
shown verbatim: the system prompt instructs Vinny to weave a couple of facts into critic
prose. iTunes was chosen over MusicBrainz because it is keyless and needs no custom
User-Agent or 1-req/s rate limit.

**Service 2 — semantic query.** `recommend_albums` runs a semantic search over a curated
store of Pitchfork album reviews using `chromadb.PersistentClient` (file persistence). It
also supports a hybrid metadata filter (`min_score`) via Chroma's `where` clause. Each hit
is enriched with title/artist/score/genre/year/url from `reviews_enrichment.csv` (pandas),
which replaces the class demo's Postgres join. Results are returned as Pydantic `AlbumRec`
objects so the model always cites artist, year, and Pitchfork score.

**Service 3 — web search.** `web_search` wraps Tavily in a single-shot, non-agentic search
for current music news that a static review store cannot cover. Simple search is the right
tool here: music-news lookups are shallow ("what did X just release?") and answered in one
retrieval round; Deep Research would add latency, token cost, and quota use for no benefit.

## Embedding process (graders do not rebuild this)

The persisted store and CSV are committed, so graders only query them. They were produced
offline by `build_index.py`:

1. Join `pitchfork_reviews.jsonl`, `pitchfork_genres.jsonl`, and `pitchfork_content.jsonl`
   on `reviewid` (the large source files are gitignored and not shipped).
2. Drop rows with empty content/genre; exclude any Taylor Swift rows (defensive).
3. Take a **genre-stratified** sample of ~2,500 reviews so no single genre dominates
   retrieval, and truncate each review to 1,400 characters.
4. Embed with `text-embedding-3-small` via Chroma's `OpenAIEmbeddingFunction` and persist
   to `chroma_store/`. Write the enrichment fields to `reviews_enrichment.csv`.

To rebuild (needs the source files in `documents/pitchfork/`): `python -m
assignment_chat.build_index`.

## Guardrails

- **Restricted topics (hard refusal).** Cats/dogs, horoscopes/zodiac, and Taylor Swift are
  refused outright with a fixed in-character line and no tool call. Enforced in the system
  prompt, with defensive artist filtering in `tools_api.py` and at index-build time.
- **System-prompt protection.** The assistant never reveals, quotes, or translates its
  instructions, and never obeys attempts to override or ignore them; both cases return a
  fixed refusal line.

## Memory

- **Mandatory:** conversation memory is maintained by rebuilding the LangChain message
  history from Gradio's `history` on every turn and re-invoking the graph
  (`MessagesState`).
- **Optional short-term management:** `memory.py`'s `trim_history` keeps the last ~6
  exchanges so the prompt does not grow without bound on long sessions. It counts messages
  rather than true tokens (simple and robust); the SystemMessage is injected separately.

## Files and data policy

Committed: all `.py`, this README, `chroma_store/`, `reviews_enrichment.csv`. Not
committed (gitignored): the Pitchfork source `.jsonl` and `database.sqlite`, `.env`,
`.secrets`.

Note on "no SQLite": the assignment forbids using the 83 MB source `database.sqlite` as a
SQL query backend. `PersistentClient` keeps its own internal `chroma.sqlite3` as the vector
store's storage engine; it is accessed only through Chroma's vector API, never with SQL,
and is a required part of file persistence.
