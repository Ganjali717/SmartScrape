from __future__ import annotations

import os, json, re, time
from dotenv import find_dotenv, load_dotenv
import requests
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122 Safari/537.36"

load_dotenv(find_dotenv(usecwd=True))


def _fix_base(url: Optional[str]) -> str:
    if not url:
        return "https://layout.fit.vutbr.cz/api"
    u = url.rstrip("/")
    # если случайно дали URL UI:
    if u.endswith("/fitlayout-web"):
        u = u[: -len("/fitlayout-web")] + "/api"
    if not u.endswith("/api"):
        if not u.endswith("/api/"):
            u = u + "/api"
    return u


@dataclass
class FLNode:
    id: str
    text: str
    bbox: Tuple[float, float, float, float]  # x,y,w,h
    font_size: Optional[float]
    font_weight: Optional[float]
    dom_path: Optional[str]
    parent: Optional[str]


class FitLayoutClient:
    def __init__(
        self, base_url: Optional[str], repo: str, token: str, timeout: int = 90
    ):
        self.base = _fix_base(base_url)
        self.repo = repo
        self.token = token
        self.timeout = timeout

    @property
    def _h(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "User-Agent": UA,
            "Accept": "*/*",
            "Content-Type": "application/json",
        }

    @property
    def _h_sparql(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "User-Agent": UA,
            "Accept": "application/sparql-results+json",
            "Content-Type": "application/sparql-query",
        }

    def list_services(self) -> list[dict]:
        r = requests.get(
            f"{self.base}/r/{self.repo}/service", headers=self._h, timeout=self.timeout
        )
        r.raise_for_status()
        return r.json()

    def get_service(self, pred) -> dict:
        for s in self.list_services():
            if pred(s):
                return s
        raise RuntimeError("FitLayout service not found by predicate")

    def service_config(self, service_id: str) -> dict:
        r = requests.get(
            f"{self.base}/r/{self.repo}/service/config?id={service_id}",
            headers=self._h,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def postService(self, service_id: str, params: dict) -> dict:
        body = {"serviceId": service_id, "params": params}
        print(body)
        r = requests.post(
            f"{self.base}/r/{self.repo}/service",
            headers=self._h,
            json=body,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def repositoryQuery(self, sparql: str) -> dict:
        # API: POST /api/r/{repo}/sparql   { query: "..."}
        r = requests.post(
            f"{self.base}/r/{self.repo}/repository/query",
            headers=self._h_sparql,
            data=sparql,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()


def _first_present(names: list[str], present: set[str], value: Any) -> dict:
    for n in names:
        if n in present:
            return {n: value}
    return {}


def render_url_to_nodes(
    url: str, viewport: Tuple[int, int] = (1200, 800), persist: int = 1
) -> List[FLNode]:
    base_url = os.getenv("FITLAYOUT_URL")
    repo = os.getenv("FITLAYOUT_REPO")
    token = os.getenv("FITLAYOUT_TOKEN")

    if not repo or not token:
        raise ValueError("Set FITLAYOUT_REPO and FITLAYOUT_TOKEN in environment/.env")

    fl = FitLayoutClient(base_url, repo, token, timeout=90)

    # ищем Puppeteer
    pupp = fl.get_service(
        lambda s: "puppeteer" in s.get("id", "").lower()
        or "puppeteer" in s.get("name", "").lower()
    )

    cfg = fl.service_config(pupp["id"])
    raw = cfg.get("params", {})
    if isinstance(raw, dict):
        pnames = set(raw.keys())
    elif isinstance(raw, list):
        pnames = set((p.get("name") if isinstance(p, dict) else str(p)) for p in raw)
    else:
        pnames = set()

    params = {}
    # params |= _first_present(["url", "pageUrl", "sourceUrl"], pnames, url)
    params["url"] = url
    if "width" in pnames:
        params["width"] = viewport[0]
    if "height" in pnames:
        params["height"] = viewport[1]
    if "persist" in pnames:
        params["persist"] = persist
    if "acquireImages" in pnames:
        params["acquireImages"] = False
    if "includeScreenshot" in pnames:
        params["includeScreenshot"] = False

    # рендер
    try:
        fl.postService(pupp["id"], params)
    except requests.HTTPError as e:
        raise ValueError(
            f"FitLayout render failed: {e.response.status_code if e.response else ''} {getattr(e.response,'text', '')[:300]}"
        )

    # достаём визуальные боксы
    sparql = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
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
    tbl = fl.repositoryQuery(sparql)

    def _normalize_table(t: dict) -> tuple[list[str], list[list[Any]]]:
        # Вариант А: {"columns":[{"name":"x"},...], "data":[[...], ...]}
        if "columns" in t and "data" in t:
            cols = [c["name"] for c in t["columns"]]
            return cols, t["data"]
        # Вариант B: SPARQL JSON {"head":{"vars":[...]}, "results":{"bindings":[...]}}
        if "head" in t and "results" in t:
            cols = t["head"].get("vars", [])
            rows = []
            for b in t["results"].get("bindings", []):
                row = []
                for c in cols:
                    v = b.get(c, {}).get("value")
                    row.append(v)
                rows.append(row)
            return cols, rows
        raise KeyError("Unexpected SPARQL response shape")

    cols, data = _normalize_table(tbl)
    idx = {n: i for i, n in enumerate(cols)}
    nodes: List[FLNode] = []

    def g(row, k):
        return row[idx[k]] if k in idx else None

    for row in data:
        nodes.append(
            FLNode(
                id=str(g(row, "id")),
                dom_path=g(row, "xpath") or None,
                text=(g(row, "text") or "").strip(),
                bbox=(
                    float(g(row, "x") or 0),
                    float(g(row, "y") or 0),
                    float(g(row, "w") or 0),
                    float(g(row, "h") or 0),
                ),
                font_size=(
                    float(g(row, "fsize")) if g(row, "fsize") is not None else None
                ),
                font_weight=(
                    float(g(row, "fweight")) if g(row, "fweight") is not None else None
                ),
                parent=str(g(row, "parent")) if g(row, "parent") else None,
            )
        )
    return nodes


# ---------- Helpers: HTML + JSON-LD ----------


def fetch_html(url: str, timeout: int = 30) -> str:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": UA})
    r.raise_for_status()
    return r.text


def _walk(o):
    if isinstance(o, dict):
        yield o
        for v in o.values():
            yield from _walk(v)
    elif isinstance(o, list):
        for it in o:
            yield from _walk(it)


def jsonld_product_from_html(html: str) -> dict:
    # простейший парсер JSON-LD без BeautifulSoup (чтобы не тащить зависимость)
    # возьмём все <script type="application/ld+json"> примитивно регуляркой:
    scripts = re.findall(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        html,
        flags=re.I | re.S,
    )
    for block in scripts:
        txt = block.strip()
        if not txt:
            continue
        try:
            data = json.loads(txt)
        except Exception:
            continue
        for it in _walk(data):
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
                "price": offers.get("price") or it.get("price"),
                "priceCurrency": offers.get("priceCurrency") or it.get("priceCurrency"),
                "sku": it.get("sku") or offers.get("sku"),
                "availability": offers.get("availability") or it.get("availability"),
            }
    return {}


def jsonld_product_from_url(url: str, timeout: int = 30) -> dict:
    try:
        html = fetch_html(url, timeout=timeout)
    except Exception:
        return {}
    return jsonld_product_from_html(html)
