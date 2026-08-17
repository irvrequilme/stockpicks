import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from picks import get_categorized_picks, search_ticker
from crypto_universe import CRYPTO_UNIVERSE

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

app = FastAPI(title="Stock Predictor Web")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/picks")
def picks(top: int = 10, interval: str = "1d"):
    try:
        return get_categorized_picks(top_n=top, interval=interval)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Couldn't reach Yahoo Finance right now: {e}")


@app.get("/api/picks/crypto")
def crypto_picks(top: int = 10, interval: str = "1d"):
    try:
        return get_categorized_picks(top_n=top, tickers=CRYPTO_UNIVERSE, interval=interval)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Couldn't reach Yahoo Finance right now: {e}")


@app.get("/api/search")
def search(ticker: str, interval: str = "1d"):
    if not ticker or not ticker.strip():
        raise HTTPException(status_code=400, detail="ticker is required")
    try:
        result = search_ticker(ticker, interval=interval)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Couldn't reach Yahoo Finance right now: {e}")
    if result is None:
        return {"error": f"Couldn't find data for '{ticker}' on Yahoo Finance."}
    return result


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
