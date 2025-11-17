from __future__ import annotations

import re
from typing import List, Dict, Any, Tuple

import numpy as np
from bs4 import BeautifulSoup, NavigableString

from ..adapters.fitlayout import (
    FLNode,
    render_with_fitlayout,
    render_from_html,
    render_url_to_nodes,
)

CURRENCY_PAT = re.compile(r"(\$|USD|EUR|CZK|€|£)", re.I)
NUMBER_PAT = re.compile(r"\d")

TAG_VOCAB = ["h1", "h2", "h3", "div", "span", "p", "small", "em", "strong", "a"]
TAG2ID = {t: i for i, t in enumerate(TAG_VOCAB)}


def fl_node_features(n: FLNode, page_w=1200, page_h=800) -> np.ndarray:
    x, y, w, h = n.bbox
    txt = n.text or ""
    area = (w * h) / max(1.0, page_w * page_h)
    digits = sum(c.isdigit() for c in txt)
    has_curr = 1.0 if re.search(r"(€|\$|czk|eur|usd|£)", txt, re.I) else 0.0
    is_bold = 1.0 if (n.font_weight or 400) >= 600 else 0.0
    return np.array(
        [
            x / page_w,
            y / page_h,
            w / page_w,
            h / page_h,
            area,
            len(txt),
            digits,
            has_curr,
            is_bold,
            float(n.font_size or 0.0),
        ],
        dtype=float,
    )


def fl_nodes_and_feats_from_url(
    url: str, viewport=(1200, 800)
) -> Tuple[List[FLNode], np.ndarray]:
    nodes = render_url_to_nodes(url, viewport=viewport)
    X = (
        np.vstack([fl_node_features(n, *viewport) for n in nodes])
        if nodes
        else np.zeros((0, 10))
    )
    return nodes, X


def fl_nodes_and_feats_from_html(
    html: str, viewport=(1366, 768)
) -> Tuple[List[FLNode], np.ndarray]:
    nodes = render_from_html(html, viewport)
    X = (
        np.vstack([fl_node_features(n, *viewport) for n in nodes])
        if nodes
        else np.zeros((0, 10))
    )
    return nodes, X


def fl_nodes_and_feats_from_file(
    path: str, viewport=(1366, 768)
) -> Tuple[List[FLNode], np.ndarray]:
    nodes = render_with_fitlayout(path, viewport)
    X = (
        np.vstack([fl_node_features(n, *viewport) for n in nodes])
        if nodes
        else np.zeros((0, 10))
    )
    return nodes, X


def _walk_with_depth(soup: BeautifulSoup):
    stack = [(soup, 0)]
    res = []
    while stack:
        node, d = stack.pop()
        if hasattr(node, "name") and node.name is not None:
            res.append((node, d))
            for child in reversed(list(node.children)):
                if isinstance(child, NavigableString):
                    continue
                stack.append((child, d + 1))
    return res


def node_features(node, depth: int, order: int) -> np.ndarray:
    text = node.get_text(" ", strip=True)
    tag = node.name.lower() if node.name else ""
    cls = " ".join(node.get("class", []))
    _ = 1.0 if node.get("id") else 0.0  # reserved

    feats = []
    tag_vec = np.zeros(len(TAG_VOCAB))
    if tag in TAG2ID:
        tag_vec[TAG2ID[tag]] = 1.0
    feats.extend(tag_vec.tolist())
    feats += [
        len(text),
        float(bool(CURRENCY_PAT.search(text))),
        float(bool(NUMBER_PAT.search(text))),
        float(depth),
        float(order),
        float("price" in cls.lower() or "ttl" in cls.lower()),
        float("title" in cls.lower() or "heading" in cls.lower()),
    ]
    return np.array(feats, dtype=float)


def soup_nodes_and_feats(soup: BeautifulSoup) -> Tuple[List[Any], np.ndarray]:
    items = _walk_with_depth(soup)
    X, nodes = [], []
    for i, (n, d) in enumerate(items):
        X.append(node_features(n, d, i))
        nodes.append(n)
    return nodes, np.vstack(X) if X else np.zeros((0, len(TAG_VOCAB) + 6))
