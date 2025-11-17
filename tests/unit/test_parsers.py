import json
from smartscrape.core.parsers import parse_product_newV6, parse_job_new, parse_event_new
from smartscrape.core.validation import validate_manual, validate_jsonschema


# ---- product ----
def test_product_parser_minimal_jsonld(tmp_path):
    html = """
    <html><head>
      <script type="application/ld+json">
      {
        "@context":"https://schema.org",
        "@type":"Product",
        "name":"Test Phone",
        "sku":"TP-001",
        "offers": {"@type":"Offer","price":"199.90","priceCurrency":"EUR","availability":"https://schema.org/InStock"}
      }
      </script>
    </head><body></body></html>
    """
    rec, proof = parse_product_newV6(html=html)
    assert rec["title"] == "Test Phone"
    assert rec["sku"] == "TP-001"
    assert rec["price"] == 199.90
    assert rec["availability"] == "in_stock"


def test_product_schema_validation():
    from pathlib import Path

    schema = json.loads(
        Path("configs/schemas/product.schema.json").read_text(encoding="utf-8")
    )
    rec = {"title": "X", "price": 10.0, "sku": None, "availability": "in_stock"}
    assert validate_manual(rec, schema) == []
    assert validate_jsonschema(rec, schema) == []


# ---- job ----
def test_job_parser_dom_only():
    html = """
    <div class="title">Python Dev</div>
    <div class="company">ACME</div>
    <div class="location">Brno</div>
    <div class="salary">50-70k</div>
    <div class="posted">Yesterday</div>
    """
    rec, _ = parse_job_new(html)
    assert rec["title"] == "Python Dev"
    assert rec["company"] == "ACME"


# ---- event ----
def test_event_parser_dom_only():
    html = """
    <h1>PyCon CZ</h1>
    <div class="venue">Brno Expo</div>
    <div class="time">2025-10-01 10:00</div>
    <div class="price">€0</div>
    """
    rec, _ = parse_event_new(html)
    assert rec["name"] == "PyCon CZ"
    assert rec["venue"] == "Brno Expo"
