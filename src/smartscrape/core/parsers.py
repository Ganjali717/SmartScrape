from __future__ import annotations

import json
import pathlib
import re
from typing import Dict, Any, Tuple, Optional

import numpy as np
from bs4 import BeautifulSoup
from joblib import load

from ..ml.features import (
    fl_nodes_and_feats_from_html,
    fl_nodes_and_feats_from_url,
)
from ..adapters.fitlayout import (
    fetch_html,
    jsonld_product_from_html,
    jsonld_product_from_url,
)

LABELS_PATH = pathlib.Path("data/labels/labels.json")
MODELS_DIR = pathlib.Path("artifacts/models")

PRICE_RE = re.compile(
    r"(€|\$|£|czk|eur|usd)\s*\d[\d\s.,]*|\d[\d\s.,]*\s*(€|\$|£|czk|eur|usd)",
    re.I,
)


# ---------- models & labels ----------
def load_model(domain: str, field: str):
    p = MODELS_DIR / f"{domain}_{field}.pkl"
    return load(p) if p.exists() else None


def load_labels() -> Dict[str, Dict[str, str]]:
    if LABELS_PATH.exists():
        return json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    return {}


def save_labels(labels: Dict[str, Dict[str, str]]) -> None:
    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    LABELS_PATH.write_text(json.dumps(labels, indent=2), encoding="utf-8")


# ---------- parsing ----------
def parse_product_new(
    url: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Unified product parser with JSON-LD → ML (FitLayout) → heuristics fallbacks."""
    jd: Dict[str, Any] = {}

    # 1) Try JSON-LD (prefer URL if provided)
    try:
        jd = jsonld_product_from_url(url, timeout=30) or {}
    except Exception:
        jd = {}

    if jd:

        def _num(s):
            m = re.search(r"[\d]+(?:[.,]\d+)?", str(s) or "")
            return float(m.group(0).replace(",", ".")) if m else None

        avail_raw = str(jd.get("availability") or "").lower()
        avail = (
            "in_stock"
            if "instock" in avail_raw
            else ("out_of_stock" if "outofstock" in avail_raw else None)
        )
        rec = {
            "title": jd.get("title") or jd.get("name"),
            "price": _num(jd.get("price")),
            "sku": jd.get("sku"),
            "availability": avail,
        }
        if any(v not in (None, "") for v in rec.values()):
            return rec, {"ops": ["jsonld"], "jsonld": jd}

    # 2) ML via FitLayout (title & price)
    title, it = pick_by_model_fl(
        "product",
        "title",
        url=url,
        fallback_selector="h1,.heading",
    )
    price, ip = pick_by_model_fl(
        "product",
        "price",
        url=url,
        fallback_selector=".price,[itemprop='price']",
        postproc=lambda s: (
            float(re.search(r"[\d.,]+", str(s)).group().replace(",", "."))
            if s
            else None
        ),
    )

    # 3) Heuristic DOM for sku/availability from page text
    page_text = ""
    sku = None
    avail = None
    m = None
    if page_text:
        m = re.search(
            r"\b(SKU|Артикул|Model|Part\s*No\.?)\b[:\s#]*([A-Za-z0-9][A-Za-z0-9\-._/]+)",
            page_text,
            re.I,
        )
    if m:
        sku = m.group(2).strip()

    low = page_text.lower()
    if ("in stock" in low) or ("в наличии" in low) or ("available" in low):
        avail = "in_stock"
    elif ("out of stock" in low) or ("нет в наличии" in low) or ("unavailable" in low):
        avail = "out_of_stock"

    rec = {"title": title, "price": price, "sku": sku, "availability": avail}
    proof = {
        "nodes": {
            "title": "h1/.heading",
            "price": "[itemprop='price']/.price",
            "sku": "#code",
            "availability": ".status.ok",
        },
        "ml_nodes": {"title": it, "price": ip},
        "ops": ["css-select", it.get("op"), ip.get("op")],
        "constraints": [],
    }
    return rec, proof


def parse_job_new(html: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.select_one(".heading") or soup.select_one(".title")
    company = soup.select_one("[data-k='co']") or soup.select_one(".company")
    location = soup.select_one("[data-k='loc']") or soup.select_one(".location")
    posted = soup.select_one("[data-k='posted']") or soup.select_one(".posted")
    salary_node = soup.select_one("[data-k='pay']") or soup.select_one(".salary")
    salary = salary_node.get_text(strip=True) if salary_node else None

    rec = {
        "title": title.get_text(strip=True) if title else None,
        "company": company.get_text(strip=True) if company else None,
        "location": location.get_text(strip=True) if location else None,
        "salary": salary,
        "posted_date": posted.get_text(strip=True) if posted else None,
    }
    proof = {
        "nodes": {
            "title": ".heading|.title",
            "company": "[data-k='co']|.company",
            "location": "[data-k='loc']|.location",
            "salary": "[data-k='pay']|.salary",
            "posted_date": "[data-k='posted']|.posted",
        },
        "ops": ["css-select"],
        "constraints": [],
    }
    return rec, proof


def parse_event_new(html: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    name = soup.select_one(".name") or soup.find(["h1", "h2"])
    venue = soup.select_one(".venue") or soup.find(string=re.compile("Where:", re.I))
    when = soup.select_one(".time") or soup.find(string=re.compile("When:", re.I))
    price_txt = soup.select_one(".price") or soup.find(
        string=re.compile("Tickets:", re.I)
    )

    name_v = (
        name.get_text(strip=True)
        if hasattr(name, "get_text")
        else (str(name).strip() if name else None)
    )
    venue_v = (
        venue.parent.get_text(strip=True).replace("Where:", "").strip()
        if hasattr(venue, "parent")
        else (venue.get_text(strip=True) if venue else None)
    )
    when_v = (
        when.parent.get_text(strip=True).replace("When:", "").strip()
        if hasattr(when, "parent")
        else (when.get_text(strip=True) if when else None)
    )

    price = None
    if price_txt:
        if isinstance(price_txt, str) and "Free" in price_txt:
            price = 0.0
        else:
            m = re.search(r"[\d.]+", str(price_txt).replace(" ", ""))
            price = float(m.group()) if m else None

    rec = {"name": name_v, "venue": venue_v, "datetime": when_v, "price": price}
    proof = {
        "nodes": {
            "name": ".name|h1|h2",
            "venue": ".venue|text:Where",
            "datetime": ".time|text:When",
            "price": ".price|text:Tickets",
        },
        "ops": ["css-select", "keyword", "regex"],
        "constraints": [],
    }
    return rec, proof


# ---------- ML integration (FitLayout) ----------
def pick_by_model_fl(
    domain: str,
    field: str,
    *,
    url: Optional[str] = None,
    fallback_selector: str = "",
    postproc=None,
):
    try:
        # 1) FitLayout nodes & features
        nodes, X = fl_nodes_and_feats_from_url(url)

        # 2) ML model inference
        model = load_model(domain, field)
        if model is not None and X.shape[0]:
            probs = model.predict_proba(X)[:, 1]
            idx = int(np.argmax(probs))
            n = nodes[idx]
            val = (n.text or "").strip()
            if postproc:
                try:
                    val = postproc(val)
                except Exception:
                    pass
            return val, {
                "op": f"ml:fl:{field}",
                "dom_path": n.dom_path,
                "bbox": list(n.bbox),
                "proba": float(probs[idx]),
            }

        # 3) Heuristic fallback from nodes
        val = _heuristic_from_nodes(nodes, field)
        if val is not None:
            if postproc:
                try:
                    val = postproc(val)
                except Exception:
                    pass
            return val, {"op": f"heuristic:{field}"}
    except Exception:
        pass

    # 4) DOM fallback (requires html)
    if html:
        soup = BeautifulSoup(html, "html.parser")
        node = soup.select_one(fallback_selector) if fallback_selector else None
        if node:
            val = node.get_text(" ", strip=True)
            if postproc:
                try:
                    val = postproc(val)
                except Exception:
                    pass
            return val, {"op": f"css:{fallback_selector}"}

    return None, {"op": "fallback:none"}


def _heuristic_from_nodes(nodes, field):
    if not nodes:
        return None
    if field == "title":
        # large, near top, reasonably short, bold preferred
        best = None
        score_best = -1.0
        for n in nodes:
            txt = (n.text or "").strip()
            if not (5 <= len(txt) <= 120):
                continue
            if txt.lower() in ("home", "menu", "signin", "cart"):
                continue
            x, y, w, h = n.bbox
            size = float(n.font_size or 0.0)
            score = (
                size * 2.0
                + (1.0 - min(y / 2000.0, 1.0))
                + (1.0 if (n.font_weight or 400) >= 600 else 0.0)
            )
            if score > score_best:
                best, score_best = txt, score
        return best
    if field == "price":
        cands = [
            (n.text or "").strip() for n in nodes if PRICE_RE.search((n.text or ""))
        ]
        if not cands:
            return None
        cand = sorted(cands, key=len)[0]
        m = re.search(r"[\d\s.,]+", cand)
        return float(m.group().replace(" ", "").replace(",", ".")) if m else None
    return None
