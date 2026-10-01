# api/index.py
import os
import traceback
from pathlib import Path

CACHE_DIR = Path("/tmp")
CACHE_DIR.mkdir(exist_ok=True)

try:
    from fastapi import FastAPI, Query
    import duckdb
    from huggingface_hub import HfFileSystem
except Exception:
    traceback.print_exc()
    raise

app = FastAPI()

HF_PARQUET = "hf://datasets/bitaimkingfree/num-info-only/users_data.parquet"
_FS = HfFileSystem()

_con = None

# tune these based on your Vercel/container CPU
THREADS = os.getenv("DUCKDB_THREADS", "2")


def _get_con():
    global _con
    if _con is not None:
        return _con

    con = duckdb.connect(database=":memory:", config={
        "threads": THREADS,
        "preserve_insertion_order": "false",
    })

    con.execute("SET enable_progress_bar = false")
    con.execute("SET enable_object_cache = true")
    con.execute("SET parquet_metadata_cache = true")
    con.execute("PRAGMA enable_optimizer")

    con.register_filesystem(_FS)

    # Warm up: fetch schema + row group stats once so subsequent
    # queries can prune row groups instead of re-reading metadata.
    try:
        con.execute(f"SELECT * FROM read_parquet('{HF_PARQUET}') LIMIT 0").fetchall()
    except Exception:
        traceback.print_exc()

    _con = con
    return con


# Push the filter down to parquet scan; only LIMIT 1.
# No aggregation — that's what was killing you.
QUERY = """
    SELECT
        "mobile", "name", "fname", "address",
        "alt", "circle", "id", "email"
    FROM read_parquet(?)
    WHERE "mobile" = ?
    LIMIT 1
"""

COLS = ["mobile", "name", "fname", "address", "alt", "circle", "id", "email"]


@app.get("/")
def root():
    return {"status": "online"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/search")
def search(phone: str = Query(..., description="Mobile number")):
    phone = str(phone).strip()
    if not phone:
        return {"success": False, "query": phone, "count": 0,
                "results": [], "error": "Phone number is required"}

    try:
        row = _get_con().execute(QUERY, [HF_PARQUET, phone]).fetchone()

        if not row or all(v is None for v in row):
            return {"success": False, "query": phone, "count": 0, "results": []}

        item = {k: (v if v is not None else "") for k, v in zip(COLS, row)}
        return {"success": True, "query": phone, "count": 1, "results": [item]}

    except Exception as e:
        traceback.print_exc()
        return {"success": False, "query": phone, "count": 0,
                "results": [], "error": str(e)}
