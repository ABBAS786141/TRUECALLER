from fastapi import FastAPI, Query
from huggingface_hub import HfFileSystem
import duckdb
import asyncio
from concurrent.futures import ThreadPoolExecutor

app = FastAPI(title="Num Info API")

# ============================================================
# CONFIG
# ============================================================

DATASET = "bitaimkingfree/num-info-only"

# Single parquet file
PARQUET_PATH = "hf://datasets/bitaimkingfree/num-info-only/users_data.parquet"

WORKERS = 100                                   # ⬅️ 35 se 100
POOL = ThreadPoolExecutor(max_workers=WORKERS)

# One shared HfFileSystem instance
_FS = HfFileSystem()


# ============================================================
# SEARCH
# ============================================================

def search_phone(phone: str):
    con = duckdb.connect()
    try:
        con.register_filesystem(_FS)
        con.execute("SET threads=4")
        con.execute("SET enable_progress_bar=false")
        con.execute("SET preserve_insertion_order=false")

        # MAX() NULL ignore karta hai -> best filled value per column
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
            FROM read_parquet(?)
            WHERE CAST("mobile" AS VARCHAR) = ?
        """

        row = con.execute(query, [PARQUET_PATH, str(phone)]).fetchone()

        if row is None:
            return []

        columns = ["mobile", "name", "fname", "address", "alt", "circle", "id", "email"]
        item = dict(zip(columns, row))

        if all(v is None for v in item.values()):
            return []

        for k in columns:
            if item[k] is None:
                item[k] = ""

        return [item]

    finally:
        con.close()


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
        "search": "/search?phone=9999999891"
    }


@app.get("/health")
def health():
    return {"status": "ok", "dataset": DATASET, "workers": WORKERS}


@app.get("/search")
async def search(
    phone: str = Query(..., description="Mobile number"),
):
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
            "results": results
        }
    except Exception as e:
        return {"success": False, "query": phone, "count": 0,
                "results": [], "error": str(e)}
