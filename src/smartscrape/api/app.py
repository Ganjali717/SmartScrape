from __future__ import annotations

import inspect
import json
import os
import pathlib
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..core.parsers import (
    parse_product_new,
    parse_job_new,
    parse_event_new,
    load_labels,
    save_labels,
)
from ..core.validation import validate_manual, validate_jsonschema
from ..core.drift import drift_transform

try:
    from dotenv import load_dotenv, find_dotenv

    load_dotenv(find_dotenv())
except Exception:
    pass


def _load_schema(path: str) -> Dict[str, Any]:
    p = pathlib.Path(path)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"type": "object"}


SCHEMAS = {
    "product": _load_schema("configs/schemas/product.schema.json"),
    "job": _load_schema("configs/schemas/job.schema.json"),
    "event": _load_schema("configs/schemas/event.schema.json"),
}

PARSERS = {
    "product": parse_product_new,
    "job": parse_job_new,
    "event": parse_event_new,
}

app = FastAPI(title="SmartScrape API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ExtractRequest(BaseModel):
    html: Optional[str] = None
    url: Optional[str] = None
    domain: str


class DriftRequest(BaseModel):
    html: str
    domain: str


class LabelRequest(BaseModel):
    domain: str = "product"
    hints: Dict[str, str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/debug/env")
def debug_env():
    t = os.getenv("FITLAYOUT_TOKEN") or ""
    return {
        "FITLAYOUT_URL": os.getenv("FITLAYOUT_URL"),
        "FITLAYOUT_REPO": os.getenv("FITLAYOUT_REPO"),
        "FITLAYOUT_TOKEN_len": len(t),
    }


@app.post("/extract")
async def extract(request: ExtractRequest) -> Dict[str, Any]:
    if request.domain not in PARSERS:
        raise HTTPException(status_code=400, detail="Unsupported domain")

    parser = PARSERS[request.domain]

    try:
        if request.domain == "product":
            if request.url:
                rec, proof = parser(url=request.url)
            elif request.html:
                rec, proof = parser(html=request.html)
            else:
                raise HTTPException(status_code=400, detail="Provide html or url")
        else:
            if not request.html:
                raise HTTPException(status_code=400, detail="Provide html")
            # job/event expect only html
            rec, proof = parser(request.html)
        return {"record": rec, "proof": proof}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/drift_extract")
def drift_extract(req: DriftRequest):
    if req.domain not in PARSERS:
        raise HTTPException(status_code=400, detail="Unsupported domain")

    html2 = drift_transform(req.html)
    parser = PARSERS[req.domain]
    rec, proof = parser(html2)

    manual = validate_manual(rec, SCHEMAS[req.domain])
    js_err = validate_jsonschema(rec, SCHEMAS[req.domain])

    filled = sum(v not in (None, "") for v in rec.values())
    total = len(rec)
    return {
        "drifted_html": html2,
        "record": rec,
        "validation_manual": manual,
        "validation_jsonschema": js_err,
        "filled_fields": filled,
        "total_fields": total,
        "proof": proof,
    }


@app.post("/label")
def add_label(req: LabelRequest):
    labels = load_labels()
    labels[req.domain] = req.hints
    save_labels(labels)
    return {"status": "ok", "stored": labels[req.domain]}
