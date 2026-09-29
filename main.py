# api/index.py
import os
import traceback
from fastapi import FastAPI, Query
import duckdb
from huggingface_hub import HfFileSystem

app = FastAPI()

HF_DB  = "hf://datasets/bitaimkingfree/num-info-only/numinfo.duckdb"
_FS    = HfFileSystem()

_con = None


def _get_con():
    global _con
    if _con is not None:
        return _con

    con = duckdb.connect(":memory:", config={"threads": "1"})
    con.execute("SET enable_progress_bar = false")
    con.register_filesystem(_FS)

    # attach remote duckdb READ_ONLY — DuckDB fetches only needed pages
    con.execute(f"ATTACH '{HF_DB}' AS num (READ_ONLY)")

    _con = con
    return con


QUERY = """
    SELECT mobile, name, fname, address, alt, circle, id, email
    FROM num.users
    WHERE mobile = ?
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
        row = _get_con().execute(QUERY, [phone]).fetchone()

        if not row:
            return {"success": False, "query": phone, "count": 0, "results": []}

        item = {k: (v if v is not None else "") for k, v in zip(COLS, row)}
        return {"success": True, "query": phone, "count": 1, "results": [item]}

    except Exception as e:
        traceback.print_exc()
        return {"success": False, "query": phone, "count": 0,
                "results": [], "error": str(e)}
