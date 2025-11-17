from __future__ import annotations

import json
import pathlib
import re
from typing import Tuple, List

import numpy as np
from bs4 import BeautifulSoup
from dotenv import load_dotenv, find_dotenv
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample

from ..adapters.fitlayout import (
    fetch_html,
    jsonld_product_from_url,
    render_url_to_nodes,
)
from ..ml.features import (
    fl_node_features,
    soup_nodes_and_feats,
    fl_nodes_and_feats_from_file,
)

load_dotenv(find_dotenv(usecwd=True))

LABELS_PATH = pathlib.Path("data/labels/ml_labels.json")
LABELS = (
    json.loads(LABELS_PATH.read_text(encoding="utf-8")) if LABELS_PATH.exists() else {}
)
MODELS_DIR = pathlib.Path("artifacts/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _target_from_html(html: str, selectors: List[str]):
    soup = BeautifulSoup(html or "", "html.parser")
    for sel in selectors or []:
        if sel.startswith("text:"):
            needle = sel.split(":", 1)[1].strip().lower()
            if needle and needle in soup.get_text(" ", strip=True).lower():
                return needle
        else:
            try:
                el = soup.select_one(sel)
                if el:
                    return el.get_text(" ", strip=True)
            except Exception:
                continue
    return None


def build_dataset_fitlayout_urls(
    domain: str, field: str
) -> Tuple[np.ndarray, np.ndarray]:
    urls = LABELS.get(domain, {}).get("urls", [])
    Xs, ys = [], []
    for url in urls:
        try:
            gold = jsonld_product_from_url(url) or {}
        except Exception:
            gold = {}

        if field == "title":
            target_txt = gold.get("title")
        elif field == "sku":
            target_txt = gold.get("sku")
        elif field == "price":
            target_txt = (
                str(gold.get("price")) if gold.get("price") is not None else None
            )
        elif field == "availability":
            av = str(gold.get("availability") or "")
            target_txt = (
                "InStock"
                if "InStock" in av
                else ("OutOfStock" if "OutOfStock" in av else None)
            )
        else:
            target_txt = None

        if not target_txt:
            html = fetch_html(url, timeout=30)
            target_txt = _target_from_html(html, LABELS.get(domain, {}).get(field, []))
        if not target_txt:
            continue

        nodes = render_url_to_nodes(url)
        if not nodes:
            continue
        X = np.vstack([fl_node_features(n) for n in nodes])

        norm_t = str(target_txt).strip().lower()
        y = []
        for n in nodes:
            t = (n.text or "").strip().lower()
            if field == "price":
                y.append(
                    1
                    if (norm_t and norm_t in t)
                    or any(k in t for k in ["$", "€", "czk", "usd", "eur", "£"])
                    else 0
                )
            elif field == "availability":
                y.append(1 if ("in stock" in t) or ("out of stock" in t) else 0)
            else:
                y.append(1 if norm_t and norm_t in t else 0)
        if any(y):
            Xs.append(X)
            ys.extend(y)

    if not Xs:
        return np.zeros((0, 10)), np.zeros((0,))

    X = np.vstack(Xs)
    y = np.array(ys)
    if y.sum() > 0 and (len(y) - y.sum()) > 0:
        pos = (y == 1).nonzero()[0]
        neg = (y == 0).nonzero()[0]
        neg = resample(
            neg,
            replace=False,
            n_samples=min(len(neg), max(50, len(pos) * 3)),
            random_state=42,
        )
        keep = np.concatenate([pos, neg])
        X = X[keep]
        y = y[keep]
    return X, y


def css_match(n, soup, sel):
    if sel.startswith("text:"):
        needle = sel.split(":", 1)[1].lower()
        return needle in n.get_text(" ", strip=True).lower()
    try:
        return n in soup.select(sel)
    except Exception:
        return False


def build_dataset(domain: str, field: str) -> Tuple[np.ndarray, np.ndarray]:
    files = LABELS.get(domain, {}).get("files", [])
    Xs, ys = [], []
    for fp in files:
        html = pathlib.Path(fp).read_text(encoding="utf-8")
        soup = BeautifulSoup(html, "html.parser")
        nodes, X = soup_nodes_and_feats(soup)
        y = []
        sels = LABELS.get(domain, {}).get(field, [])
        for n in nodes:
            y.append(1 if any(css_match(n, soup, s) for s in sels) else 0)
        Xs.append(X)
        ys.extend(y)

    if not Xs:
        return np.zeros((0, 10)), np.zeros((0,))
    X = np.vstack(Xs)
    y = np.array(ys)

    if y.sum() > 0 and (len(y) - y.sum()) > 0:
        pos_idx = (y == 1).nonzero()[0]
        neg_idx = (y == 0).nonzero()[0]
        neg_idx = resample(
            neg_idx,
            replace=False,
            n_samples=min(len(neg_idx), max(50, len(pos_idx) * 3)),
            random_state=42,
        )
        keep = list(pos_idx) + list(neg_idx)
        X = X[keep]
        y = y[keep]
    return X, y


def train_domain(domain: str):
    fields = [k for k in LABELS.get(domain, {}).keys() if k not in ("files", "urls")]
    for field in fields:
        if "urls" in LABELS.get(domain, {}):
            X, y = build_dataset_fitlayout_urls(domain, field)
        else:
            X, y = build_dataset(domain, field)
        if X.shape[0] == 0 or y.sum() == 0:
            print(f"[{domain}:{field}] skipped (no positives)")
            continue
        clf = LogisticRegression(max_iter=200)
        clf.fit(X, y)
        out = MODELS_DIR / f"{domain}_{field}.pkl"
        dump(clf, out)
        print(f"saved {out} (n={X.shape[0]}, pos={int(y.sum())})")


if __name__ == "__main__":
    for dom in LABELS.keys():
        train_domain(dom)
    print("done.")
