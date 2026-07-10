"""OFFLINE index builder. Run once from 05_src/:

    python -m assignment_chat.build_index

Graders do NOT run this. It reads the large Pitchfork source files (NOT committed),
selects a small genre-stratified subset, embeds it with text-embedding-3-small, and
writes two COMMITTED artifacts that the chat app queries at runtime:
    - assignment_chat/chroma_store/       (persisted ChromaDB, file persistence)
    - assignment_chat/reviews_enrichment.csv
"""
import pandas as pd
import chromadb
from dotenv import load_dotenv

from assignment_chat.gateway import get_embedding_function
from utils.logger import get_logger

_logs = get_logger(__name__)
load_dotenv(".env")
load_dotenv(".secrets")

SRC = "documents/pitchfork"
N_TARGET = 2000
MAX_CHARS = 1400
CHROMA_PATH = "assignment_chat/chroma_store"
CSV_PATH = "assignment_chat/reviews_enrichment.csv"
COLLECTION = "pitchfork_reviews"
BATCH = 200


def build_dataframe() -> pd.DataFrame:
    reviews = pd.read_json(f"{SRC}/pitchfork_reviews.jsonl", lines=True)
    genres = pd.read_json(f"{SRC}/pitchfork_genres.jsonl", lines=True).drop_duplicates("reviewid")
    content = pd.read_json(f"{SRC}/pitchfork_content.jsonl", lines=True)

    df = reviews.merge(genres, on="reviewid", how="left").merge(content, on="reviewid", how="left")
    df = df.dropna(subset=["content", "genre"])
    df = df[df["content"].str.len() > 200]
    # Defensive Taylor Swift exclusion (0 rows today; keeps the guardrail airtight).
    df = df[~df["artist"].str.contains("taylor swift", case=False, na=False)]

    frac = min(1.0, N_TARGET / len(df))
    df = df.groupby("genre", group_keys=False).sample(frac=frac, random_state=42)
    df = df.head(N_TARGET).reset_index(drop=True)
    df["content"] = df["content"].str.slice(0, MAX_CHARS)
    _logs.info(f"Selected {len(df)} reviews across {df['genre'].nunique()} genres")
    return df


def main():
    df = build_dataframe()

    df[["reviewid", "title", "artist", "score", "genre", "pub_year", "url"]].rename(
        columns={"pub_year": "year"}
    ).to_csv(CSV_PATH, index=False)
    _logs.info(f"Wrote enrichment CSV: {CSV_PATH}")

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    coll = client.create_collection(
        name=COLLECTION,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )

    ids = [f"{r}_0" for r in df["reviewid"].astype(str)]
    docs = df["content"].tolist()
    metas = [
        {"score": float(s) if pd.notna(s) else 0.0, "genre": str(g), "year": int(y) if pd.notna(y) else 0}
        for s, g, y in zip(df["score"], df["genre"], df["pub_year"])
    ]

    for i in range(0, len(ids), BATCH):
        coll.add(ids=ids[i:i + BATCH], documents=docs[i:i + BATCH], metadatas=metas[i:i + BATCH])
        _logs.info(f"Embedded {min(i + BATCH, len(ids))}/{len(ids)}")

    _logs.info(f"Done. Collection count: {coll.count()}. Commit chroma_store/ and the CSV.")


if __name__ == "__main__":
    main()
