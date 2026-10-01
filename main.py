# api/index.py
import os, gzip, json, threading, traceback
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
_CACHE_FILE = CACHE_DIR / "users.json.gz"

# ---- Global in-memory index ----
DB = {}            # mobile(str) -> record dict
_ready = False
_lock = threading.Lock()


def _load_from_hf():
    """Ek hi baar: HF parquet -> Python dict (gzip cached)."""
    global DB, _ready

    # 1) Try local gzip cache first
    if _CACHE_FILE.exists():
        try:
            with gzip.open(_CACHE_FILE, "rt") as f:
                DB = json.load(f)
            _ready = True
            print(f"[init] loaded from cache: {len(DB)} rows")
            return
        except Exception:
            traceback.print_exc()

    # 2) Build from HF (one-time)
    fs = HfFileSystem()
    con = duckdb.connect(":memory:", config={
        "threads": "4",
        "preserve_insertion_order": "false",
    })
    con.execute("SET enable_progress_bar = false")
    con.register_filesystem(fs)

    rows = con.execute(f"""
        SELECT "mobile","name","fname","address","alt","circle","id","email"
        FROM read_parquet('{HF_PARQUET}')
    """).fetchall()

    data = {}
    for r in rows:
        mob = str(r[0]).strip() if r[0] is not None else ""
        if not mob:
            continue
        data[mob] = {
            "mobile":  r[0] or "",
            "name":    r[1] or "",
            "fname":   r[2] or "",
            "address": r[3] or "",
            "alt":     r[4] or "",
            "circle":  r[5] or "",
            "id":      r[6] or "",
            "email":   r[7] or "",
        }
    con.close()

    DB = data
    _ready = True
    print(f"[init] built in-memory DB: {len(DB)} rows")

    # 3) Save gzip cache for next container
    try:
        with gzip.open(_CACHE_FILE, "wt", compresslevel=1) as f:
            json.dump(DB, f)
    except Exception:
        traceback.print_exc()


# Kick off load in background so first request doesn't block (if possible)
threading.Thread(target=lambda: _load_from_hf(), daemon=True).start()


def _ensure_ready():
    if _ready:
        return
    with _lock:
        if not _ready:
            _load_from_hf()


@app.get("/")
def root():
    return {"status": "online", "ready": _ready, "rows": len(DB)}


@app.get("/health")
def health():
    return {"status": "ok", "ready": _ready}


# ---- THE FAST PATH: pure dict lookup, ~1-5ms ----
@app.get("/search")
def search(phone: str = Query(..., description="Mobile number")):
    _ensure_ready()
    phone = str(phone).strip()
    if not phone:
        return {"success": False, "query": phone, "count": 0,
                "results": [], "error": "Phone number is required"}

    item = DB.get(phone)
    if item is None:
        return {"success": False, "query": phone, "count": 0, "results": []}

    return {"success": True, "query": phone, "count": 1, "results": [item]}
