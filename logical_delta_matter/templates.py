from __future__ import annotations

from typing import Any


DEFAULT_TEMPLATES: dict[str, Any] = {
    "element_analogs": [
        {"id": "E1", "min_size": 2, "max_size": 3, "label": "element-analog-1"},
        {"id": "E2", "min_size": 4, "max_size": 6, "label": "element-analog-2"},
        {"id": "E3", "min_size": 7, "max_size": 12, "label": "element-analog-3"},
    ],
    "molecule_analogs": [
        {"id": "M1", "min_elements": 2, "label": "molecule-analog-1"},
        {"id": "M2", "min_elements": 3, "label": "molecule-analog-2"},
    ],
}


def promote_cluster(signature: dict[str, Any], templates: dict[str, Any] | None = None) -> dict[str, Any]:
    """Promote a cluster signature to a model-level object.

    This is deliberately an engineering rule, not a claim about real chemistry.
    """

    templates = templates or DEFAULT_TEMPLATES
    size = int(signature["size"])

    for template in templates.get("element_analogs", []):
        if int(template["min_size"]) <= size <= int(template["max_size"]):
            return {
                "level": "element_analog",
                "template_id": template["id"],
                "label": template["label"],
                "signature": signature,
            }

    return {
        "level": "cluster",
        "template_id": "unpromoted",
        "label": "unpromoted-cluster",
        "signature": signature,
    }


def promote_all(signatures: list[dict[str, Any]], templates: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    return [promote_cluster(signature, templates) for signature in signatures]
