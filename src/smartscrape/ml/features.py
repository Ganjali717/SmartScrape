from __future__ import annotations

from typing import Tuple, List
import numpy as np
from ..adapters.fitlayout import FLNode, render_url_to_nodes

TAG_VOCAB = ["price", "eur", "usd", "czk", "in stock", "out of stock"]


def _norm_text(t: str) -> str:
    return " ".join((t or "").lower().split())


def _tag_hits(t: str) -> np.ndarray:
    tt = _norm_text(t)
    return np.array([1.0 if k in tt else 0.0 for k in TAG_VOCAB], dtype=np.float32)


def fl_node_features(n: FLNode, viewport: Tuple[int, int] = (1200, 800)) -> np.ndarray:
    x, y, w, h = n.bbox
    vw, vh = viewport
    fx = x / max(vw, 1)
    fy = y / max(vh, 1)
    fw = w / max(vw, 1)
    fh = h / max(vh, 1)
    fs = (n.font_size or 0.0) / 64.0
    fwght = (n.font_weight or 400.0) / 900.0
    tags = _tag_hits(n.text)
    return np.concatenate(
        [np.array([fx, fy, fw, fh, fs, fwght], dtype=np.float32), tags]
    )


def fl_nodes_and_feats_from_url(
    url: str, viewport: Tuple[int, int] = (1200, 800)
) -> tuple[List[FLNode], np.ndarray]:
    nodes = render_url_to_nodes(url, viewport=viewport)
    X = (
        np.vstack([fl_node_features(n, viewport) for n in nodes])
        if nodes
        else np.zeros((0, len(TAG_VOCAB) + 6), dtype=np.float32)
    )
    return nodes, X
