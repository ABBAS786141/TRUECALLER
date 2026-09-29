# api/index.py
import os, sys, traceback
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
_FS = HfFileSystem()  # no custom httpx client — keep it minimal

_con = None

def _get_con():
    global _con
    if _con is None:
        _con = duckdb.connect(":memory:", config={
            "threads": "1",
            "enable_progress_bar": "false",
            "preserve_insertion_order": "false",
        })
        _con.register_filesystem(_FS)
    return _con

QUERY = """
    SELECT MAX("mobile") AS "mobile", MAX("name") AS "name",
           MAX("fname") AS "fname", MAX("address") AS "address",
           MAX("alt") AS "alt", MAX("circle") AS "circle",
           MAX("id") AS "id", MAX("email") AS "email"
    FROM read_parquet(?) WHERE CAST("mobile" AS VARCHAR) = ?
"""
COLS = ["mobile","name","fname","address","alt","circle","id","email"]

@app.get("/")
def root():
    return {"status": "online"}

@app.get("/search")
def search(phone: str = Query(...)):
    try:
        row = _get_con().execute(QUERY, [HF_PARQUET, str(phone).strip()]).fetchone()
        if not row or all(v is None for v in row):
            return {"success": False, "query": phone, "count": 0, "results": []}
        item = {k: (v if v is not None else "") for k, v in zip(COLS, row)}
        return {"success": True, "query": phone, "count": 1, "results": [item]}
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "query": phone, "count": 0,
                "results": [], "error": str(e)}
