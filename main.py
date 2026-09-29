# api/index.py
import os
import traceback
from pathlib import Path

try:
    from fastapi import FastAPI, Query
    import duckdb
    from huggingface_hub import hf_hub_download
except Exception:
    traceback.print_exc()
    raise

app = FastAPI()

HF_REPO = "bitaimkingfree/num-info-only"
HF_FILE = "numinfo.duckdb"
LOCAL_DB = Path("/tmp/numinfo.duckdb")

_con = None


def _ensure_db() -> Path:
    if LOCAL_DB.exists() and LOCAL_DB.stat().st_size > 0:
        return LOCAL_DB
    path = hf_hub_download(
        repo_id=HF_REPO,
        filename=HF_FILE,
        repo_type="dataset",
        local_dir="/tmp",
        token=os.environ.get("HF_TOKEN"),
    )
    p = Path(path)
    if p.resolve() != LOCAL_DB.resolve():
        p.replace(LOCAL_DB)
    return LOCAL_DB


def _get_con():
    global _con
    if _con is not None:
        return _con
    db_path = _ensure_db()
    con = duckdb.connect(str(db_path), read_only=True, config={
        "threads": "1",
        "enable_progress_bar": "false",
    })
    _con = con
    return con


QUERY = """
    SELECT mobile, name, fname, address, alt, circle, id, email
    FROM users
    WHERE mobile = ?
    LIMIT 1
"""

COLS = ["mobile", "name", "fname", "address", "alt", "circle", "id", "email"]


@app.get("/")
def root():
    return {"status": "online"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "db_present": LOCAL_DB.exists(),
        "db_size": LOCAL_DB.stat().st_size if LOCAL_DB.exists() else 0,
    }


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
