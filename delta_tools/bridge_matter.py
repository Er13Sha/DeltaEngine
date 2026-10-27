"""Мост: data/matter-deltas.json (прототип) → дерево delta-tree@0.1 ядра.

Каждая дельта прототипа читает один булев признак `x.traits.<имя>`. В новом дереве
этот признак становится атрибутом схемы. Признак обязателен только там, где до него
доходит путь, поэтому он объявляется с `required_when`: условием, при котором
объект попадает в область дельты. Русские названия узлов берутся из mermaid-схемы.
"""
from __future__ import annotations

import re
from typing import Any

from delta_kernel.tree import TREE_FORMAT

_EDGE = re.compile(r"-->\|(not )?(\w+)\(x\)\|\s*\w+\[([^\]]+)\]")
_ROOT = re.compile(r"^\s*\w+\[([^\]]+)\]\s*$", re.M)


def _labels(mermaid: str) -> dict[tuple[str, bool], str]:
    out = {}
    for neg, pred, label in _EDGE.findall(mermaid):
        out[(pred, not neg)] = label
    return out


def matter_tree_from_deltas(data: dict[str, Any]) -> dict[str, Any]:
    deltas = data["deltas"]
    labels = _labels(data.get("mermaid", ""))
    by_parent_branch: dict[str, dict[str, Any]] = {}
    for d in deltas:
        domain = d["domain"]
        if "=" in domain:
            by_parent_branch[domain.split("=")[0].strip()] = d

    root_id = data["root_domain"]["root_delta"]
    root = next(d for d in deltas if d["delta_id"] == root_id)
    attributes: list[dict[str, Any]] = []

    def attr_of(d: dict[str, Any]) -> str:
        return d["decider"]["input"].removeprefix("x.traits.")

    def build(d: dict[str, Any], condition: str) -> dict[str, Any]:
        attr = attr_of(d)
        pred_name = d["predicate"].removesuffix("(x)")
        spec: dict[str, Any] = {"name": attr, "type": {"kind": "bool"}, "doc": d["obligations"]["total"]}
        if condition:
            spec["required"] = False
            spec["required_when"] = condition
        attributes.append(spec)
        node: dict[str, Any] = {
            "id": d["delta_id"],
            "predicate": attr,
            "knowledge": "constatation",
            "evidence": d.get("evidence_source", []),
        }
        for key, value in (("true", True), ("false", False)):
            branch = d["branches"][key]
            child = by_parent_branch.get(branch)
            if child is not None:
                sub = build(child, attr if value else f"not {attr}")
                sub["label"] = labels.get((pred_name, value), branch)
                sub["concept"] = branch
                node[key] = sub
            else:
                node[key] = {"id": branch, "label": labels.get((pred_name, value), branch)}
        return node

    root_node = build(root, "")
    root_label = _ROOT.search(data.get("mermaid", ""))
    if root_label:
        root_node["label"] = root_label.group(1)
    return {
        "format": TREE_FORMAT,
        "id": "matter",
        "version": data["schema_version"],
        "schema": {
            "name": data["root_domain"]["domain"],
            "version": data["schema_version"],
            "attributes": attributes,
        },
        "root": root_node,
    }
