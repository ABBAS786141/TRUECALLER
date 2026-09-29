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

POOL = ThreadPoolExecutor(max_workers=WORKERS)

# One shared HfFileSystem instance (auth + session reuse)
_FS = HfFileSystem()

# ============================================================
# FAST SEARCH (merged best record)
# ============================================================

def search_phone(phone: str):

    con = duckdb.connect()

    try:
        con.register_filesystem(_FS)

        # Speed tweaks
        con.execute("SET threads=4")
        con.execute("SET enable_progress_bar=false")
        con.execute("SET preserve_insertion_order=false")

        # MAX() ignores NULLs -> gives best non-empty value per column
        query = """
            SELECT
                MAX("Number")  AS "Number",
                MAX("Name")    AS "Name",
                MAX("Gender")  AS "Gender",
                MAX("Address") AS "Address",
                MAX("Email")   AS "Email"
            FROM read_parquet(?, union_by_name=true)
            WHERE CAST("Number" AS VARCHAR) = ?
        """

        row = con.execute(
            query,
            [PARQUET_FILES, str(phone)]
        ).fetchone()

        if row is None:
            return []

        columns = ["Number", "Name", "Gender", "Address", "Email"]
        item = dict(zip(columns, row))

        # If everything is NULL, no match
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
        "api": "Truecaller API",
        "dataset": DATASET,
        "files": len(PARQUET_FILES),
        "workers": WORKERS,
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
            phone
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
