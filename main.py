from fastapi import FastAPI, Query
from huggingface_hub import HfFileSystem
import duckdb
import asyncio
from concurrent.futures import ThreadPoolExecutor

app = FastAPI(title="Truecaller API")

# ============================================================
# CONFIG
# ============================================================

DATASET = "wannabeyour/truecallerdata"

PARQUET_FILES = [
    "hf://datasets/wannabeyour/truecallerdata/final_combined_data.parquet",
    "hf://datasets/wannabeyour/truecallerdata/combined_truecaller_data.parquet",
    "hf://datasets/wannabeyour/truecallerdata/combined_selected_columns.parquet",
]

WORKERS = 35
BATCH_SIZE = 64

POOL = ThreadPoolExecutor(max_workers=WORKERS)


# ============================================================
# SEARCH
# ============================================================

def search_phone(phone: str, limit: int):

    con = duckdb.connect()

    try:
        fs = HfFileSystem()
        con.register_filesystem(fs)

        con.execute("SET threads=1")

        query = """
            SELECT
                "Number",
                "Name",
                "Gender",
                "Address",
                "Email"
            FROM read_parquet(?, union_by_name=true)
            WHERE CAST("Number" AS VARCHAR) = ?
            LIMIT ?
        """

        rows = con.execute(
            query,
            [PARQUET_FILES, str(phone), int(limit)]
        ).fetchall()

        columns = ["Number", "Name", "Gender", "Address", "Email"]
        results = []

        for row in rows:
            item = dict(zip(columns, row))

            # NULL -> ""
            for k in columns:
                if item[k] is None:
                    item[k] = ""

            results.append(item)

        return results

    finally:
        con.close()


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def root():
    return {
        "status": "online",
        "api": "Truecaller API",
        "dataset": DATASET,
        "files": PARQUET_FILES,
        "workers": WORKERS,
        "batch_size": BATCH_SIZE,
        "search": "/search?phone=917207000711"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "dataset": DATASET,
        "files": len(PARQUET_FILES),
        "workers": WORKERS
    }


@app.get("/search")
async def search(
    phone: str = Query(..., description="Phone number"),
    limit: int = Query(20, ge=1, le=100)
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

        results = await loop.run_in_executor(
            POOL,
            search_phone,
            phone,
            limit
        )

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