from fastapi import FastAPI, Query
from huggingface_hub import HfFileSystem
import duckdb
import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

app = FastAPI(title="Num Info API")

# ============================================================
# CONFIG
# ============================================================

DATASET = "bitaimkingfree/num-info-only"

# ⚠️ Chunks banane ke baad ye path use karo:
# PARQUET_PATH = "hf://datasets/bitaimkingfree/num-info-only/chunks/*.parquet"
# Abhi single file (jab tak chunks nahi banate):
PARQUET_PATH = "hf://datasets/bitaimkingfree/num-info-only/users_data.parquet"

WORKERS = 35
POOL = ThreadPoolExecutor(max_workers=WORKERS)

_FS = HfFileSystem()

# ============================================================
# QUERY CACHE (in-memory, TTL-based)
# ============================================================

class SimpleCache:
    def __init__(self, max_size=500, ttl=300):
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            if key in self.cache:
                value, ts = self.cache[key]
                if time.time() - ts < self.ttl:
                    return value
                del self.cache[key]
        return None

    def set(self, key, value):
        with self.lock:
            if len(self.cache) >= self.max_size:
                oldest = min(self.cache, key=lambda k: self.cache[k][1])
                del self.cache[oldest]
            self.cache[key] = (value, time.time())

    def size(self):
        with self.lock:
            return len(self.cache)


_cache = SimpleCache(max_size=500, ttl=300)

# ============================================================
# DUCKDB CONNECTION POOL (per-thread persistent connection)
# ============================================================

_conns: list = []
_conns_lock = threading.Lock()
_thread_local = threading.local()


def _new_conn():
    """Har thread ka apna persistent DuckDB connection."""
    con = duckdb.connect()

    con.register_filesystem(_FS)

    # Speed tuning
    con.execute("SET threads=4")
    con.execute("SET enable_object_cache=true")          # ⭐ metadata cache
    con.execute("SET enable_progress_bar=false")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET memory_limit='1GB'")

    # Pre-built view — query time pe path parse nahi hoga
    con.execute(
        f"CREATE OR REPLACE VIEW people AS "
        f"SELECT * FROM read_parquet('{PARQUET_PATH}', union_by_name=true)"
    )

    return con


def _thread_id():
    tid = getattr(_thread_local, "id", None)
    if tid is None:
        with _conns_lock:
            tid = len(_conns)
            _thread_local.id = tid
    return tid


def _get_conn():
    ident = _thread_id()
    with _conns_lock:
        while len(_conns) <= ident:
            _conns.append(_new_conn())
    return _conns[ident]


# ============================================================
# SEARCH
# ============================================================

COLUMNS = ["mobile", "name", "fname", "address", "alt", "circle", "id", "email"]


def search_phone(phone: str):
    # 1. Cache check (0 ms on hit)
    cached = _cache.get(phone)
    if cached is not None:
        return cached

    # 2. Query via view (metadata already cached)
    con = _get_conn()

    query = """
        SELECT
            MAX("mobile")  AS "mobile",
            MAX("name")    AS "name",
            MAX("fname")   AS "fname",
            MAX("address") AS "address",
            MAX("alt")     AS "alt",
            MAX("circle")  AS "circle",
            MAX("id")      AS "id",
            MAX("email")   AS "email"
        FROM people
        WHERE CAST("mobile" AS VARCHAR) = ?
    """

    row = con.execute(query, [str(phone)]).fetchone()

    if row is None:
        return []

    item = dict(zip(COLUMNS, row))

    # Saare NULL = no match
    if all(v is None for v in item.values()):
        return []

    # NULL -> ""
    for k in COLUMNS:
        if item[k] is None:
            item[k] = ""

    result = [item]

    # 3. Cache store
    _cache.set(phone, result)

    return result


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def root():
    return {
        "status": "online",
        "api": "Num Info API",
        "dataset": DATASET,
        "file": PARQUET_PATH,
        "workers": WORKERS,
        "cache_size": _cache.size(),
        "connections": len(_conns),
        "search": "/search?phone=9999999891"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "dataset": DATASET,
        "connections": len(_conns),
        "cache_size": _cache.size()
    }


@app.get("/search")
async def search(
    phone: str = Query(..., description="Mobile number"),
):
    phone = str(phone).strip()

    if not phone:
        return {
            "success": False,
            "query": phone,
            "count": 0,
            "results": [],
            "error": "Phone number is required"
        }

    try:
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(POOL, search_phone, phone)

        return {
            "success": bool(results),
            "query": phone,
            "count": len(results),
            "results": results
        }

    except Exception as e:
        return {
            "success": False,
            "query": phone,
            "count": 0,
            "results": [],
            "error": str(e)
        }


# ============================================================
# STARTUP: warm up connections + metadata
# ============================================================

@app.on_event("startup")
def startup():
    print("⏳ Warming up DuckDB connections & metadata...")
    try:
        # Ek connection bana ke metadata cache warm karo
        con = _get_conn()
        con.execute(f"SELECT COUNT(*) FROM people").fetchone()
        print("✅ Warmup complete")
    except Exception as e:
        print(f"⚠️ Warmup failed: {e}")
