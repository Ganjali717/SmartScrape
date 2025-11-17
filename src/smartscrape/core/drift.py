from __future__ import annotations

import re


def drift_transform(html: str) -> str:
    html2 = re.sub(
        r'class="([^"]+)"',
        lambda m: 'class="'
        + m.group(1)
        .replace("title", "ttl")
        .replace("price", "p-x")
        .replace("company", "co-x")
        + '"',
        html,
    )
    return f'<div class="wrap1"><div class="wrap2">{html2}</div></div>'
