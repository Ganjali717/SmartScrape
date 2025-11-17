from __future__ import annotations

import json, pathlib, re
from typing import Dict, Any, Optional, Tuple
import numpy as np
from joblib import load

from ..adapters.fitlayout import jsonld_product_from_url
from ..ml.features import fl_nodes_and_feats_from_url

MODELS_DIR = pathlib.Path("artifacts/models")


def _best_by_model(
    domain: str, field: str, url: str
) -> Tuple[Optional[str], Dict[str, Any]]:
    model_path = MODELS_DIR / f"{domain}_{field}.pkl"
    if not model_path.exists():
        return None, {"op": "fallback:none"}

    nodes, X = fl_nodes_and_feats_from_url(url)
    if X.shape[0] == 0:
        return None, {"op": "fallback:none"}

    clf = load(model_path)
    if hasattr(clf, "predict_proba"):
        prob = clf.predict_proba(X)[:, 1]
        idx = int(np.argmax(prob))
        score = float(prob[idx])
    else:
        score = float(clf.decision_function(X).max())
        idx = int(np.argmax(clf.decision_function(X)))

    txt = nodes[idx].text.strip()
    return txt, {"op": "ml", "idx": idx, "score": score}


def _price_to_float(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    m = re.search(r"[\d]+(?:[.,]\d+)?", s.replace(" ", ""))
    return float(m.group(0).replace(",", ".")) if m else None


def _availability_guess(texts: list[str]) -> Optional[str]:
    t = " ".join(texts).lower()
    if "out of stock" in t:
        return "out_of_stock"
    if "in stock" in t:
        return "in_stock"
    return None


def parse_product(url: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
    # 1) JSON-LD
    jd = jsonld_product_from_url(url) or {}
    if jd.get("title") and (jd.get("price") or jd.get("offers")):
        price = jd.get("price") or (jd.get("offers") or {}).get("price")
        rec = {
            "title": jd.get("title") or jd.get("name"),
            "price": _price_to_float(str(price) if price is not None else None),
            "sku": jd.get("sku"),
            "availability": (
                "in_stock"
                if "instock" in str(jd.get("availability", "")).lower()
                else (
                    "out_of_stock"
                    if "outofstock" in str(jd.get("availability", "")).lower()
                    else None
                )
            ),
        }
        return rec, {"ops": ["jsonld"]}

    # 2) ML по FitLayout
    title, it = _best_by_model("product", "title", url)
    price_str, ip = _best_by_model("product", "price", url)

    # 3) лёгкие эвристики по всем узлам
    nodes, _ = fl_nodes_and_feats_from_url(url)
    texts = [n.text for n in nodes]
    avail = _availability_guess(texts)
    # sku как «похожее на артикул»
    sku = None
    for t in texts:
        if re.search(r"\b[a-z0-9]{3,}[-_/][a-z0-9]{3,}\b", t.lower()):
            sku = t.strip()
            break

    rec = {
        "title": (title or None),
        "price": _price_to_float(price_str),
        "sku": sku,
        "availability": avail,
    }
    proof = {
        "ml_nodes": {"title": it, "price": ip},
        "ops": [it.get("op"), ip.get("op")],
    }
    return rec, proof
