from __future__ import annotations

import json
import os
import pathlib
import tempfile
from dataclasses import dataclass
from typing import Optional, List, Tuple
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup

try:
    from dotenv import load_dotenv

    load_dotenv()  # FITLAYOUT_URL / FITLAYOUT_REPO / FITLAYOUT_TOKEN
except Exception:
    pass

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)


@dataclass
class FLNode:
    id: str
    dom_path: Optional[str]
    text: str
    bbox: Tuple[float, float, float, float]
    font_size: Optional[float]
    font_weight: Optional[float]
    parent: Optional[str]


class FitLayoutClient:
    def __init__(
        self,
        base_url: str,
        repo_id: str,
        token: Optional[str] = None,
        timeout: int = 60,
    ):
        base_url = base_url.rstrip("/")
        self.base_api = base_url if base_url.endswith("/api") else base_url + "/api"
        self.repo = quote(repo_id, safe="")
        self.s = requests.Session()
        self.s.headers.update({"Accept": "application/json"})
        if token:
            self.s.headers["Authorization"] = f"Bearer {token}"
        self.timeout = timeout

    def list_services(self) -> list[dict]:
        r = self.s.get(f"{self.base_api}/r/{self.repo}/service", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def get_service(self, predicate) -> dict:
        for s in self.list_services():
            if predicate(s):
                return s
        raise RuntimeError("FitLayout service not found")

    def service_config(self, service_id: str) -> dict:
        r = self.s.get(
            f"{self.base_api}/r/{self.repo}/service/config",
            params={"id": service_id},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def invoke_service(
        self, service_id: str, params: dict, parent_iri: Optional[str] = None
    ):
        body = {"serviceId": service_id, "params": params}
        if parent_iri:
            body["parentIri"] = parent_iri
        r = self.s.post(
            f"{self.base_api}/r/{self.repo}/service",
            json=body,
            headers={
                "Accept": "application/ld+json",
                "Content-Type": "application/json",
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r

    def select(self, sparql: str) -> dict:
        r = self.s.post(
            f"{self.base_api}/r/{self.repo}/repository/selectQuery",
            data=sparql.encode("utf-8"),
            headers={"Content-Type": "application/sparql-query"},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()


def fetch_html(url: str, timeout: int = 30) -> str:
    r = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"},
    )
    r.raise_for_status()
    return r.text


def jsonld_product_from_html(html: str) -> dict:
    soup = BeautifulSoup(html or "", "html.parser")
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


def jsonld_product_from_url(url: str, timeout: int = 30) -> dict:
    html = fetch_html(url, timeout=timeout)
    return jsonld_product_from_html(html)


def render_url_to_nodes(
    url: str,
    *,
    base_url: Optional[str] = None,
    repo: Optional[str] = None,
    token: Optional[str] = None,
    viewport: Tuple[int, int] = (1200, 800),
    persist: int = 1,
) -> List[FLNode]:
    base_url = base_url or os.getenv("FITLAYOUT_URL")
    repo = repo or os.getenv("FITLAYOUT_REPO")
    token = token or os.getenv("FITLAYOUT_TOKEN")
    if not token:
        raise ValueError("FITLAYOUT_TOKEN is missing. Set it in .env or environment.")

    fl = FitLayoutClient(base_url, repo, token, timeout=90)
    renderer = fl.get_service(lambda s: s.get("id") == "FitLayout.Puppeteer")

    base_params = {
        "url": url,
        "width": viewport[0],
        "height": viewport[1],
        "persist": persist,
        "acquireImages": False,
        "includeScreenshot": False,
    }

    try:
        fl.invoke_service(renderer["id"], base_params)
    except requests.HTTPError as e:
        # fallback: download HTML and render as a local file
        if url.startswith("http"):
            try:
                html = requests.get(
                    url,
                    headers={"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"},
                    timeout=30,
                ).text
                return render_from_html(html, viewport)
            except Exception:
                pass
        raise ValueError(
            f"FitLayout render failed: {e.response.status_code if e.response else ''} "
            f"{getattr(e.response, 'text', '')[:300]}"
        )

    sparql = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX b: <http://fitlayout.github.io/ontology/render.owl#>
SELECT ?id ?x ?y ?w ?h ?text ?fsize ?fweight ?parent ?xpath WHERE {
  ?b a b:Box ; b:visualBounds ?vb .
  ?vb b:positionX ?x ; b:positionY ?y ; b:width ?w ; b:height ?h .
  OPTIONAL { ?b b:text ?text } OPTIONAL { ?b b:fontSize ?fsize }
  OPTIONAL { ?b b:fontWeight ?fweight } OPTIONAL { ?b b:isChildOf ?parent }
  OPTIONAL { ?b b:sourceXPath ?xpath }
  BIND(STR(?b) AS ?id)
}
"""
    res = fl.select(sparql)
    binds = (res.get("results") or {}).get("bindings", [])

    def v(b, key):
        x = b.get(key)
        return None if x is None else x.get("value")

    nodes: List[FLNode] = []
    for b in binds:
        try:
            x = float(v(b, "x")) if v(b, "x") is not None else 0.0
            y = float(v(b, "y")) if v(b, "y") is not None else 0.0
            w = float(v(b, "w")) if v(b, "w") is not None else 0.0
            h = float(v(b, "h")) if v(b, "h") is not None else 0.0
            fsize = v(b, "fsize")
            fweight = v(b, "fweight")
            nodes.append(
                FLNode(
                    id=str(v(b, "id") or ""),
                    dom_path=v(b, "xpath") or None,
                    text=(v(b, "text") or "").strip(),
                    bbox=(x, y, w, h),
                    font_size=float(fsize) if fsize is not None else None,
                    font_weight=float(fweight) if fweight is not None else None,
                    parent=str(v(b, "parent")) if v(b, "parent") else None,
                )
            )
        except Exception:
            continue

    return nodes


def render_with_fitlayout(html_path: str, viewport=(1200, 800)) -> List[FLNode]:
    p = pathlib.Path(html_path).resolve()
    return render_url_to_nodes(f"file://{p.as_posix()}", viewport=viewport)


def render_from_html(html_str: str, viewport=(1200, 800)) -> List[FLNode]:
    with tempfile.NamedTemporaryFile(
        suffix=".html", delete=False, mode="w", encoding="utf-8"
    ) as f:
        f.write(html_str)
        tmp = f.name
    try:
        return render_with_fitlayout(tmp, viewport)
    finally:
        pathlib.Path(tmp).unlink(missing_ok=True)
