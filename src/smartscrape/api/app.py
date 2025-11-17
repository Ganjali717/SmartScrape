from __future__ import annotations
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any

from ..core.parsers import parse_product

app = FastAPI(title="SmartScrape URL-only")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/extract")
async def extract(req: Request) -> Dict[str, Any]:
    try:
        data = await req.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    domain = (data.get("domain") or "product").lower()
    url = data.get("url") or data.get("pageUrl")

    if not url:
        raise HTTPException(status_code=400, detail="Provide url")

    try:
        if domain == "product":
            rec, proof = parse_product(url)
        else:
            raise HTTPException(
                status_code=400, detail="Only 'product' implemented in URL-only demo"
            )

        filled = sum(1 for v in rec.values() if v not in (None, ""))
        return {"record": rec, "filled_fields": filled, "proof": proof}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
