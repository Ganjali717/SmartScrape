from __future__ import annotations

import json, pathlib, re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from joblib import dump

from ..adapters.fitlayout import jsonld_product_from_url
from ..ml.features import fl_nodes_and_feats_from_url

BASE = pathlib.Path(".")
LABELS = json.loads(
    pathlib.Path("data/labels/ml_labels.json").read_text(encoding="utf-8")
)

MODELS_DIR = pathlib.Path("artifacts/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _price_norm(s: str) -> str:
    return re.sub(r"[^\d.,]", "", s or "").replace(",", ".")


def _label_vector_for_field(nodes, field: str, gold: dict) -> np.ndarray:
    target = None
    if field == "title":
        target = _norm(gold.get("title"))
    elif field == "price":
        target = _price_norm(str(gold.get("price")))
    elif field == "sku":
        target = _norm(gold.get("sku"))
    elif field == "availability":
        av = str(gold.get("availability") or "").lower()
        target = (
            "instock"
            if "instock" in av
            else ("outofstock" if "outofstock" in av else None)
        )

    y = []
    for n in nodes:
        t = _norm(n.text)
        if field == "title":
            y.append(1 if target and target and target in t else 0)
        elif field == "price":
            tnum = _price_norm(n.text)
            y.append(1 if target and target != "" and target in tnum else 0)
        elif field == "sku":
            y.append(1 if target and target in t else 0)
        elif field == "availability":
            y.append(1 if ("in stock" in t or "out of stock" in t) else 0)
        else:
            y.append(0)
    return np.array(y, dtype=np.int32)


def build_dataset(domain: str, field: str):
    urls = LABELS.get(domain, {}).get("urls", [])
    Xs, ys = [], []
    for url in urls:
        gold = jsonld_product_from_url(url) or {}
        nodes, X = fl_nodes_and_feats_from_url(url)
        if not len(nodes):
            continue
        y = _label_vector_for_field(nodes, field, gold)
        if y.sum() == 0:
            continue
        Xs.append(X)
        ys.extend(y)
    if not Xs:
        return np.zeros((0, 10)), np.zeros((0,))
    X = np.vstack(Xs)
    y = np.array(ys)

    # балансировка
    if y.sum() > 0 and (len(y) - y.sum()) > 0:
        pos_idx = (y == 1).nonzero()[0]
        neg_idx = (y == 0).nonzero()[0]
        neg_idx = resample(
            neg_idx,
            replace=False,
            n_samples=min(len(neg_idx), max(50, len(pos_idx) * 3)),
            random_state=42,
        )
        keep = np.concatenate([pos_idx, neg_idx])
        X = X[keep]
        y = y[keep]
    return X, y


def train_domain(domain: str):
    fields = [
        k
        for k in LABELS.get(domain, {}).get(
            "fields", ["title", "price", "sku", "availability"]
        )
    ]
    for field in fields:
        X, y = build_dataset(domain, field)
        if X.shape[0] == 0 or y.sum() == 0:
            print(f"[{domain}:{field}] skipped (no positives)")
            continue
        clf = LogisticRegression(max_iter=400)
        clf.fit(X, y)
        out = MODELS_DIR / f"{domain}_{field}.pkl"
        dump(clf, out)
        print(f"saved {out}  n={X.shape[0]}  pos={int(y.sum())}")


if __name__ == "__main__":
    for dom in LABELS.keys():
        train_domain(dom)
    print("done.")
