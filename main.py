# app.py — Num Info API (10x)
# pip install fastapi uvicorn duckdb huggingface_hub httpx[http2] aiofiles

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import duckdb
import httpx
from fastapi import FastAPI, Query
from huggingface_hub import HfFileSystem

# ============================================================
# CONFIG
# ============================================================

DATASET       = "bitaimkingfree/num-info-only"
HF_PARQUET    = "hf://datasets/bitaimkingfree/num-info-only/users_data.parquet"
LOCAL_CACHE   = Path(os.getenv("NUMINFO_CACHE", "./cache"))
LOCAL_PARQUET = LOCAL_CACHE / "users_data.parquet"

WORKERS       = int(os.getenv("NUMINFO_WORKERS", "200"))
THREADS_PER   = int(os.getenv("NUMINFO_THREADS", "2"))
REFRESH_SEC   = int(os.getenv("NUMINFO_REFRESH", "3600"))  # 1h

COLUMNS = ["mobile", "name", "fname", "address", "alt", "circle", "id", "email"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("numinfo")

# ============================================================
# HF FILESYSTEM (HTTP/2 keep-alive)
# ============================================================

_FS = HfFileSystem(
    client=httpx.Client(http2=True, timeout=30.0, limits=httpx.Limits(
        max_keepalive_connections=64, max_connections=128,
    )),
)

# ============================================================
# LOCAL PARQUET MIRROR
# ============================================================

_cache_etag: str | None = None
_cache_lock = threading.Lock()


def _remote_etag() -> str | None:
    """Cheap HEAD-style probe for the remote parquet's etag/size."""
    try:
        info = _FS.info(HF_PARQUET)
        return f"{info.get('size')}:{info.get('mtime') or info.get('last_modified')}"
    except Exception as e:
        log.warning("etag probe failed: %s", e)
        return None


def ensure_local_parquet(force: bool = False) -> Path:
    """Download remote parquet to local disk once. Refresh on etag change."""
    global _cache_etag
    LOCAL_CACHE.mkdir(parents=True, exist_ok=True)

    with _cache_lock:
        etag = _remote_etag()

        fresh = (
            not force
            and LOCAL_PARQUET.exists()
            and etag is not None
            and etag == _cache_etag
        )
        if fresh:
            return LOCAL_PARQUET

        # If remote probe failed but we have a local copy, keep using it.
        if etag is None and LOCAL_PARQUET.exists():
            return LOCAL_PARQUET

        log.info("syncing parquet → %s", LOCAL_PARQUET)
        t0 = time.time()
        tmp = LOCAL_PARQUET.with_suffix(".parquet.tmp")
        with _FS.open(HF_PARQUET, "rb") as src, open(tmp, "wb") as dst:
            while chunk := src.read(1 << 20):  # 1 MiB
                dst.write(chunk)
        tmp.replace(LOCAL_PARQUET)
        _cache_etag = etag
        log.info("parquet synced in %.2fs (%.1f MB)",
                 time.time() - t0, LOCAL_PARQUET.stat().st_size / 1e6)
        return LOCAL_PARQUET


# ============================================================
# DUCKDB CONNECTION POOL
# ============================================================
# One long-lived DuckDB connection per worker thread.
# DuckDB is thread-safe at the connection level only for reads,
# but per-thread connections avoid lock contention entirely.

_tls = threading.local()


def _get_con() -> duckdb.DuckDBPyConnection:
    con = getattr(_tls, "con", None)
    if con is not None:
        return con

    con = duckdb.connect(database=":memory:", config={
        "threads": str(THREADS_PER),
        "enable_progress_bar": "false",
        "preserve_insertion_order": "false",
        "enable_object_cache": "true",
    })
    con.register_filesystem(_FS)
    # Force parquet metadata cache warm on the local file
    _tls.con = con
    return con


# ============================================================
# SEARCH — hit local parquet, row-group pruning handles speed
# ============================================================

_QUERY = f"""
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


def search_phone(phone: str) -> list[dict[str, Any]]:
    con = _get_con()
    try:
        pq = str(LOCAL_PARQUET)
        row = con.execute(_QUERY, [pq, str(phone)]).fetchone()
    except duckdb.IOException:
        # Local file missing / corrupt → re-sync once and retry
        ensure_local_parquet(force=True)
        row = con.execute(_QUERY, [str(LOCAL_PARQUET), str(phone)]).fetchone()

    if row is None:
        return []

    item = {k: (v if v is not None else "") for k, v in zip(COLUMNS, row)}
    if not any(item.values()):
        return []
    return [item]


# ============================================================
# APP + LIFECYCLE
# ============================================================

app = FastAPI(title="Num Info API")
POOL = ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="search")


@app.on_event("startup")
async def _startup() -> None:
    # Block startup until parquet is on disk. First boot = cold sync.
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, ensure_local_parquet, False)
    log.info("startup complete — workers=%d threads/conn=%d",
             WORKERS, THREADS_PER)


@app.on_event("shutdown")
async def _shutdown() -> None:
    POOL.shutdown(wait=False, cancel_futures=True)


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def root():
    return {
        "status": "online",
        "api": "Num Info API",
        "dataset": DATASET,
        "local_parquet": str(LOCAL_PARQUET),
        "workers": WORKERS,
        "threads_per_conn": THREADS_PER,
        "search": "/search?phone=9999999891",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "dataset": DATASET,
        "local_parquet": str(LOCAL_PARQUET),
        "parquet_present": LOCAL_PARQUET.exists(),
        "workers": WORKERS,
    }


@app.post("/admin/refresh")
async def refresh():
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, ensure_local_parquet, True)
    return {"ok": True, "etag": _cache_etag}


@app.get("/search")
async def search(phone: str = Query(..., description="Mobile number")):
    phone = str(phone).strip()
    if not phone:
        return {"success": False, "query": phone, "count": 0,
                "results": [], "error": "Phone number is required"}

    try:
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(POOL, search_phone, phone)
        return {
            "success": bool(results),
            "query": phone,
            "count": len(results),
            "results": results,
        }
    except Exception as e:
        log.exception("search failed for %s", phone)
        return {"success": False, "query": phone, "count": 0,
                "results": [], "error": str(e)}
