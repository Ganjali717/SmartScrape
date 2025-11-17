from __future__ import annotations

from typing import Dict, Any, List

from jsonschema import validate as js_validate, ValidationError


def validate_manual(record: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
    probs = []
    for req in schema.get("required", []):
        if record.get(req) in (None, ""):
            probs.append(f"missing required field: {req}")
    return probs


def validate_jsonschema(record: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
    try:
        js_validate(record, schema)
        return []
    except ValidationError as e:
        return [e.message]
