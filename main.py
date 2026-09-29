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


def _get_con():
    global _con
    if _con is not None:
        return _con

    # ✅ only global-safe options in config
    con = duckdb.connect(database=":memory:", config={
        "threads": "1",
        "preserve_insertion_order": "false",
    })

    # ✅ session-local options go through SET, not config
    con.execute("SET enable_progress_bar = false")

    con.register_filesystem(_FS)
    _con = con
    return con


QUERY = """
    SELECT
        MAX("mobile")  AS "mobile",
        MAX("name")    AS "name",
        MAX("fname")   AS "fname",
        MAX("address") AS "address",
        MAX("alt")     AS "alt",
        MAX("circle")  AS "circle",
        MAX("id")      AS "id",
        MAX("email")   AS "email"
    FROM read_parquet(?)
    WHERE CAST("mobile" AS VARCHAR) = ?
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
