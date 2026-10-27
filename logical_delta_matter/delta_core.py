from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

Predicate = Callable[[dict[str, Any]], bool]


class DeltaCompilerError(ValueError):
    """Raised when a delta tree cannot be compiled or validated."""


@dataclass(frozen=True, slots=True)
class DeltaNode:
    name: str
    predicate_name: str | None = None
    yes: "DeltaNode | None" = None
    no: "DeltaNode | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.predicate_name is None


def compile_delta_tree(data: dict[str, Any]) -> DeltaNode:
    """Compile a JSON-like delta tree into immutable nodes."""

    if not isinstance(data, dict):
        raise DeltaCompilerError("Delta node must be an object.")
    if not isinstance(data.get("name"), str) or not data["name"]:
        raise DeltaCompilerError("Delta node must contain non-empty 'name'.")

    predicate = data.get("predicate")
    if predicate is None:
        return DeltaNode(name=data["name"])
    if not isinstance(predicate, str) or not predicate:
        raise DeltaCompilerError(f"Node '{data['name']}' has invalid predicate.")
    if "yes" not in data or "no" not in data:
        raise DeltaCompilerError(f"Node '{data['name']}' must contain both 'yes' and 'no' branches.")

    return DeltaNode(
        name=data["name"],
        predicate_name=predicate,
        yes=compile_delta_tree(data["yes"]),
        no=compile_delta_tree(data["no"]),
    )


def validate_delta_tree(root: DeltaNode, predicates: dict[str, Predicate]) -> None:
    """Check structural totality of the tree and predicate availability.

    Runtime totality for each predicate is checked by executing classification:
    a predicate must return a plain bool, otherwise classification fails.
    """

    seen_names: set[str] = set()

    def walk(node: DeltaNode) -> None:
        if node.name in seen_names:
            raise DeltaCompilerError(f"Duplicate delta node name: {node.name}")
        seen_names.add(node.name)

        if node.is_leaf:
            if node.yes is not None or node.no is not None:
                raise DeltaCompilerError(f"Leaf '{node.name}' cannot have branches.")
            return

        if node.predicate_name not in predicates:
            raise DeltaCompilerError(f"Unknown predicate '{node.predicate_name}' at node '{node.name}'.")
        if node.yes is None or node.no is None:
            raise DeltaCompilerError(f"Node '{node.name}' must have both branches.")
        walk(node.yes)
        walk(node.no)

    walk(root)


def classify_delta(root: DeltaNode, obj: dict[str, Any], predicates: dict[str, Predicate]) -> list[str]:
    """Return a unique root-to-leaf path.

    Path items are `node:+` or `node:-` for internal nodes, followed by the leaf
    name. If a predicate is missing or does not return bool, the function raises
    instead of guessing.
    """

    path: list[str] = []
    node = root

    while not node.is_leaf:
        if node.predicate_name is None:
            raise RuntimeError(f"Broken delta tree at node '{node.name}'.")
        predicate = predicates.get(node.predicate_name)
        if predicate is None:
            raise KeyError(f"Unknown predicate: {node.predicate_name}")

        result = predicate(obj)
        if type(result) is not bool:
            raise TypeError(f"Predicate '{node.predicate_name}' must return bool.")

        path.append(f"{node.name}:{'+' if result else '-'}")
        node = node.yes if result else node.no
        if node is None:
            raise RuntimeError("Broken delta tree: missing branch.")

    path.append(node.name)
    return path


def leaves(root: DeltaNode) -> list[str]:
    if root.is_leaf:
        return [root.name]
    assert root.yes is not None and root.no is not None
    return leaves(root.yes) + leaves(root.no)
