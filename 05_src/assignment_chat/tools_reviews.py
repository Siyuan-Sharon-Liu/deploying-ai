from langchain.tools import tool
import chromadb
from pydantic import BaseModel, Field
import pandas as pd
from dotenv import load_dotenv

from assignment_chat.gateway import get_embedding_function
from utils.logger import get_logger

_logs = get_logger(__name__)
load_dotenv(".env")
load_dotenv(".secrets")

_CHROMA_PATH = "assignment_chat/chroma_store"
_CSV_PATH = "assignment_chat/reviews_enrichment.csv"
_COLLECTION = "pitchfork_reviews"

_client = chromadb.PersistentClient(path=_CHROMA_PATH)
_collection = _client.get_collection(
    name=_COLLECTION,
    embedding_function=get_embedding_function(),
)
_enrich = pd.read_csv(_CSV_PATH, dtype={"reviewid": str}).set_index("reviewid")


class AlbumRec(BaseModel):
    """A single album recommendation grounded in a Pitchfork review."""
    title: str = Field(..., description="Album title.")
    artist: str = Field(..., description="Artist name.")
    review: str = Field(..., description="Relevant excerpt from the Pitchfork review.")
    score: float = Field(0.0, description="Pitchfork score, 0-10.")
    genre: str = Field("N/A", description="Genre.")
    year: int = Field(0, description="Publication year.")
    url: str = Field("", description="Review URL.")


@tool
def recommend_albums(query: str, n_results: int = 3, min_score: float = None) -> list[AlbumRec]:
    """Semantic search over a curated store of Pitchfork album reviews. Pass min_score to
    only return albums rated at least that high (hybrid metadata filter). Use this to
    ground every album recommendation, opinion, or score."""
    kwargs = {"query_texts": [query], "n_results": n_results}
    if min_score is not None:
        kwargs["where"] = {"score": {"$gte": min_score}}
    res = _collection.query(**kwargs)

    recs = []
    ids = res["ids"][0]
    docs = res["documents"][0]
    for i, cid in enumerate(ids):
        review_id = cid.split("_")[0]
        excerpt = docs[i][:600]
        if review_id not in _enrich.index:
            continue
        row = _enrich.loc[review_id]
        recs.append(
            AlbumRec(
                title=str(row["title"]),
                artist=str(row["artist"]),
                review=excerpt,
                score=float(row["score"]) if pd.notna(row["score"]) else 0.0,
                genre=str(row["genre"]) if pd.notna(row["genre"]) else "N/A",
                year=int(row["year"]) if pd.notna(row["year"]) else 0,
                url=str(row["url"]) if pd.notna(row["url"]) else "",
            )
        )
    return recs
