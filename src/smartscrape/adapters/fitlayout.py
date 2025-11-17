# src/smartscrape/adapters/fitlayout.py
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

# --------- Basic HTTP client for FitLayout ---------


def _auth_headers(token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        # Будь максимально совместим: некоторые эндпоинты чувствительны к Accept
        "Accept": "*/*",
        "User-Agent": "SmartScrape/1.0",
    }


class FitLayoutClient:
    """
    Minimal client for FitLayout REST API.
    base_url: e.g. https://layout.fit.vutbr.cz/api   (важно: именно /api)
    repo:     repository id (UUID)
    token:    bearer token
    """

    def __init__(self, base_url: str, repo: str, token: str, timeout: int = 90):
        self.base_url = base_url.rstrip("/")
        self.repo = repo
        self.token = token
        self.timeout = timeout

    # --- low-level ---
    def _get(self, path: str, **kw) -> requests.Response:
        url = f"{self.base_url}{path}"
        r = requests.get(
            url, headers=_auth_headers(self.token), timeout=self.timeout, **kw
        )
        r.raise_for_status()
        return r

    def _post(self, path: str, **kw) -> requests.Response:
        url = f"{self.base_url}{path}"
        headers = _auth_headers(self.token)
        data_headers = kw.pop("headers", {})
        headers.update(data_headers)
        r = requests.post(url, headers=headers, timeout=self.timeout, **kw)
        r.raise_for_status()
        return r

    # --- services ---
    def list_services(self) -> List[Dict[str, Any]]:
        r = self._get(f"/r/{self.repo}/service")
        return r.json()

    def get_service(self, predicate) -> Dict[str, Any]:
        for s in self.list_services():
            if predicate(s):
                return s
        raise ValueError("Required service not found")

    def invoke_service(self, service_id: str, params: Dict[str, Any]) -> None:
        body = {"serviceId": service_id, "params": params}
        # Тут ответ нам не важен (артефакт попадает в репозиторий)
        self._post(
            f"/r/{self.repo}/service",
            json=body,
            headers={"Content-Type": "application/json"},
        )

    # --- SPARQL ---
    def select(self, sparql: str) -> Dict[str, Any]:
        # Возвращаем стандартный JSON: head/vars + results/bindings
        r = self._post(
            f"/r/{self.repo}/sparql/select",
            data={"query": sparql},
            headers={"Accept": "application/sparql-results+json"},
        )
        return r.json()


# --------- HTML + JSON-LD helpers (используются в parsers.py) ---------

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123 Safari/537.36"


def fetch_html(url: str, timeout: int = 30) -> str:
    r = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"},
    )
    r.raise_for_status()
    return r.text


def jsonld_product_from_url(url: str) -> dict:
    fullHtml = fetch_html(url)

    soup = BeautifulSoup(fullHtml or "", "html.parser")
    for tag in soup.find_all("script", type="application/ld+json"):
        txt = (tag.string or tag.get_text() or "").strip()
        if not txt:
            continue
        try:
            data = json.loads(txt)
        except Exception:
            continue

        def walk(o):
            if isinstance(o, dict):
                yield o
                for v in o.values():
                    yield from walk(v)
            elif isinstance(o, list):
                for it in o:
                    yield from walk(it)

        for it in walk(data):
            t = it.get("@type")
            is_product = (isinstance(t, str) and t.lower() == "product") or (
                isinstance(t, list) and any(str(x).lower() == "product" for x in t)
            )
            if not is_product:
                continue
            offers = it.get("offers") or {}
            if isinstance(offers, list):
                offers = offers[0] if offers else {}
            return {
                "title": it.get("name") or it.get("title"),
                "price": offers.get("price"),
                "priceCurrency": offers.get("priceCurrency"),
                "sku": it.get("sku") or offers.get("sku"),
                "availability": offers.get("availability") or it.get("availability"),
            }
    return {}


# --------- Rendering → nodes via Puppeteer ---------


@dataclass
class FLNode:
    id: str
    dom_path: Optional[str]
    text: str
    bbox: Tuple[float, float, float, float]
    font_size: Optional[float]
    font_weight: Optional[float]
    parent: Optional[str]


def _sparql_rows(tbl_json: Dict[str, Any]) -> List[Dict[str, str]]:
    """Normalize SPARQL JSON (head/vars + results/bindings) to list of dicts."""
    if "head" in tbl_json and "vars" in tbl_json["head"]:
        vars_ = tbl_json["head"]["vars"]
        rows = []
        for b in tbl_json.get("results", {}).get("bindings", []):
            row = {}
            for v in vars_:
                cell = b.get(v)
                row[v] = cell.get("value") if isinstance(cell, dict) else None
            rows.append(row)
        return rows
    # fallback (если когда-то вернётся табличный формат)
    if "columns" in tbl_json and "data" in tbl_json:
        vars_ = [c["name"] for c in tbl_json["columns"]]
        return [dict(zip(vars_, r)) for r in tbl_json["data"]]
    raise ValueError("Unexpected SPARQL result format")


def render_url_to_nodes(
    url: str,
    base_url: Optional[str] = None,
    repo: Optional[str] = None,
    token: Optional[str] = None,
    viewport: Tuple[int, int] = (1200, 800),
    persist: int = 1,
) -> List[FLNode]:
    base_url = (base_url or os.getenv("FITLAYOUT_URL") or "").rstrip("/")
    repo = repo or os.getenv("FITLAYOUT_REPO")
    token = token or os.getenv("FITLAYOUT_TOKEN")
    if not (base_url and repo and token):
        raise ValueError(
            "FITLAYOUT_URL/FITLAYOUT_REPO/FITLAYOUT_TOKEN must be set (check your .env)"
        )

    # ВАЖНО: base_url должен указывать на /api, пример: https://layout.fit.vutbr.cz/api
    if not base_url.endswith("/api"):
        # позволим ставить просто https://layout.fit.vutbr.cz — тогда добавим /api
        if base_url.endswith("/fitlayout-web"):
            base_url = base_url.rsplit("/fitlayout-web", 1)[0] + "/api"
        elif not base_url.endswith("/api"):
            base_url = base_url + "/api"

    fl = FitLayoutClient(base_url, repo, token, timeout=90)

    # 1) найдём Puppeteer renderer
    renderer = fl.get_service(
        lambda s: str(s.get("id", "")).lower() == "fitlayout.puppeteer"
    )

    # 2) вызов рендера
    params = {
        "url": url,
        "width": int(viewport[0]),
        "height": int(viewport[1]),
        "persist": int(persist),
        "acquireImages": False,
        "includeScreenshot": False,
    }
    try:
        fl.invoke_service(renderer["id"], params)
    except requests.HTTPError as e:
        # если сайт отрубил браузер — попробуем offline режим через HTML (это опционально, но полезно)
        raise ValueError(
            f"FitLayout render failed: {getattr(e.response,'status_code', '')} {getattr(e.response,'text','')[:300]}"
        )

    # 3) вытащим боксы через SPARQL
    sparql = """
    PREFIX b: <http://fitlayout.github.io/ontology/render.owl#>
    SELECT ?id ?x ?y ?w ?h ?text ?fsize ?fweight ?parent ?xpath WHERE {
      ?b a b:Box ; b:visualBounds ?vb .
      ?vb b:positionX ?x ; b:positionY ?y ; b:width ?w ; b:height ?h .
      OPTIONAL { ?b b:text ?text } 
      OPTIONAL { ?b b:fontSize ?fsize }
      OPTIONAL { ?b b:fontWeight ?fweight } 
      OPTIONAL { ?b b:isChildOf ?parent }
      OPTIONAL { ?b b:sourceXPath ?xpath }
      BIND(STR(?b) AS ?id)
    }
    """
    tbl = fl.select(sparql)
    rows = _sparql_rows(tbl)

    nodes: List[FLNode] = []
    for r in rows:

        def f(x):
            try:
                return float(x) if x is not None else None
            except Exception:
                return None

        nodes.append(
            FLNode(
                id=str(r.get("id") or ""),
                dom_path=r.get("xpath"),
                text=(r.get("text") or "").strip(),
                bbox=(
                    f(r.get("x")) or 0.0,
                    f(r.get("y")) or 0.0,
                    f(r.get("w")) or 0.0,
                    f(r.get("h")) or 0.0,
                ),
                font_size=f(r.get("fsize")),
                font_weight=f(r.get("fweight")),
                parent=r.get("parent"),
            )
        )
    return nodes
